# sdpose_yaw.py
# 偏航角计算核心 + SDPoseEstimateYawSimple / SDPoseEstimateYawAdvanced 节点
# 幅值估算使用肩/髋X坐标差（|dx|），正/背面判定使用X差符号

import math
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch


# ==================== 偏航角辅助函数 ====================
def _get_flat_keypoints(person: Dict):
    def _flat(key):
        return person.get(key, [])
    body = _flat("pose_keypoints_2d")
    foot = _flat("foot_keypoints_2d")
    face = _flat("face_keypoints_2d")
    rhand = _flat("hand_right_keypoints_2d")
    lhand = _flat("hand_left_keypoints_2d")
    flat = body + foot + face[:204] + rhand + lhand
    return flat

def _get_point(flat, idx):
    base = idx * 3
    if base + 2 >= len(flat):
        return (0.0, 0.0, 0.0)
    return (flat[base], flat[base+1], flat[base+2])

def _is_valid(point, conf_threshold):
    x, y, c = point
    return c >= conf_threshold and x > 0 and y > 0

def _unwrap_sequence(angles):
    """解缠角度序列，使角度连续累计（而非折叠到 [-180,180]）"""
    unwrapped = []
    accum = 0.0
    prev = None
    for a in angles:
        if a is None:
            unwrapped.append(None)
            continue
        if prev is None:
            accum = a
        else:
            diff = a - prev
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            accum += diff
        unwrapped.append(accum)
        prev = a
    return unwrapped

def _wrap_to_180(angles):
    """将角度折叠到 [-180, 180] 区间"""
    def wrap_one(a):
        if a is None:
            return None
        return (a + 180) % 360 - 180
    return [wrap_one(a) for a in angles]


# ==================== 偏航角核心计算 ====================
def compute_yaw_core(pose_keypoints_segment, conf_threshold, unwrap_angle,
                     nose_neck_thresh, shoulder_weight,
                     max_angle_limit,
                     pose_keypoints_full=None, merge_threshold=30.0,
                     enable_filter=False,
                     enable_continuity=True):
    """
    核心偏航角计算。
    
    幅值估算：X坐标差（|dx|）→ acos → 角度
    符号判定：眼睛偏航角 → 鼻颈偏航角 → 连续性保护
    """
    IDX_NOSE = 0
    IDX_NECK = 1
    IDX_R_SHOULDER = 2
    IDX_L_SHOULDER = 5
    IDX_R_HIP = 8
    IDX_L_HIP = 11
    # COCO / BODY_25 通用索引
    IDX_R_EYE = 14
    IDX_L_EYE = 15
    IDX_R_EAR = 16
    IDX_L_EAR = 17
    IDX_R_KNEE = 9
    IDX_L_KNEE = 12
    IDX_R_ANKLE = 10
    IDX_L_ANKLE = 13

    anchor_seq = pose_keypoints_full if pose_keypoints_full else pose_keypoints_segment
    if not isinstance(anchor_seq, list) or not anchor_seq:
        return ("Error: No valid keypoints for anchor", torch.zeros(len(pose_keypoints_segment)))

    # ====== 第1遍：收集所有帧的关键点数据 ======
    neck_hip_dists = []
    all_scaled_sh_abs = []       # 所有帧 scaled |shoulder_dx|
    all_front_sh_abs = []        # 正面帧 scaled |shoulder_dx| (dx < 0)
    all_back_sh_abs = []         # 背面帧 scaled |shoulder_dx| (dx > 0)
    all_front_hip_abs = []       # 正面帧 scaled |hip_dx| (dx < 0)
    all_back_hip_abs = []        # 背面帧 scaled |hip_dx| (dx > 0)
    all_raw_data = []            # (shldr_dx, hip_dx, nh_dist, sh_y_diff, hip_y_diff)
    shoulder_widths = []         # 降级模式备选：各帧肩宽（半身/无髋时替代颈髋距离做参考尺度）

    for frame in anchor_seq:
        for person in frame.get("people", []):
            flat = _get_flat_keypoints(person)
            neck = _get_point(flat, IDX_NECK)
            r_hip = _get_point(flat, IDX_R_HIP)
            l_hip = _get_point(flat, IDX_L_HIP)
            r_shoulder = _get_point(flat, IDX_R_SHOULDER)
            l_shoulder = _get_point(flat, IDX_L_SHOULDER)

            neck_valid = _is_valid(neck, conf_threshold)
            hip_valid = _is_valid(r_hip, conf_threshold) and _is_valid(l_hip, conf_threshold)
            shldr_valid = _is_valid(r_shoulder, conf_threshold) and _is_valid(l_shoulder, conf_threshold)

            nh_dist = None
            if neck_valid and hip_valid:
                hip_cx = (r_hip[0] + l_hip[0]) / 2.0
                hip_cy = (r_hip[1] + l_hip[1]) / 2.0
                nh_dist = math.hypot(neck[0] - hip_cx, neck[1] - hip_cy)
                if nh_dist > 1e-6:
                    neck_hip_dists.append(nh_dist)
                else:
                    nh_dist = None

            shldr_dx = r_shoulder[0] - l_shoulder[0] if shldr_valid else None
            hip_dx = r_hip[0] - l_hip[0] if hip_valid else None
            shoulder_y_diff = r_shoulder[1] - l_shoulder[1] if shldr_valid else None
            hip_y_diff = r_hip[1] - l_hip[1] if hip_valid else None
            all_raw_data.append((shldr_dx, hip_dx, nh_dist, shoulder_y_diff, hip_y_diff))
            # 收集肩宽（降级模式备选：半身无髋时用肩宽比例映射到6方向）
            if shldr_valid:
                shoulder_widths.append(abs(r_shoulder[0] - l_shoulder[0]))

    if not neck_hip_dists:
        # 降级模式：无有效颈髋距离（常见于半身舞蹈），自动降级为6方向
        if shoulder_widths:
            body_ref_dist = np.percentile(shoulder_widths, 95)
            logging.warning(f"[Yaw] 降级模式：无有效颈髋距离（可能为半身），使用肩宽P95={body_ref_dist:.1f}作为参考尺度")
        else:
            body_ref_dist = 0.0
            logging.warning("[Yaw] 降级模式：肩髋均不可用，所有帧输出0°（正面）")
        return _compute_yaw_quantized_6dir(
            pose_keypoints_segment, body_ref_dist, conf_threshold
        )

    anchor_neck_hip = np.percentile(neck_hip_dists, 99)

    for shldr_dx, hip_dx, nh_dist, shoulder_y_diff, hip_y_diff in all_raw_data:
        if nh_dist is not None and nh_dist > 1e-6:
            scale = anchor_neck_hip / nh_dist
        else:
            scale = 1.0
        if shldr_dx is not None:
            s_abs = abs(shldr_dx * scale)
            all_scaled_sh_abs.append(s_abs)
            if shldr_dx < 0:
                all_front_sh_abs.append(s_abs)
            else:
                all_back_sh_abs.append(s_abs)
        if hip_dx is not None:
            h_abs = abs(hip_dx * scale)
            if hip_dx < 0:
                all_front_hip_abs.append(h_abs)
            else:
                all_back_hip_abs.append(h_abs)

    # 参考宽度：使用正/背面各自P95的最大值作为统一参考
    ref_sh_front = np.percentile(all_front_sh_abs, 95) if len(all_front_sh_abs) >= 3 else max(all_front_sh_abs) if all_front_sh_abs else None
    ref_sh_back = np.percentile(all_back_sh_abs, 95) if len(all_back_sh_abs) >= 3 else max(all_back_sh_abs) if all_back_sh_abs else None
    if ref_sh_front is not None and ref_sh_back is not None:
        ref_shoulder_width = max(ref_sh_front, ref_sh_back)
    elif ref_sh_front is not None:
        ref_shoulder_width = ref_sh_front
    elif ref_sh_back is not None:
        ref_shoulder_width = ref_sh_back
    else:
        ref_shoulder_width = max(all_scaled_sh_abs) if all_scaled_sh_abs else 1.0

    # 髋部参考宽度：与肩部完全一致，分正面/背面各自P95再取max
    ref_hip_front = np.percentile(all_front_hip_abs, 95) if len(all_front_hip_abs) >= 3 else max(all_front_hip_abs) if all_front_hip_abs else None
    ref_hip_back = np.percentile(all_back_hip_abs, 95) if len(all_back_hip_abs) >= 3 else max(all_back_hip_abs) if all_back_hip_abs else None
    if ref_hip_front is not None and ref_hip_back is not None:
        ref_hip_width = max(ref_hip_front, ref_hip_back)
    elif ref_hip_front is not None:
        ref_hip_width = ref_hip_front
    elif ref_hip_back is not None:
        ref_hip_width = ref_hip_back
    else:
        ref_hip_width = max(*(all_front_hip_abs + all_back_hip_abs)) if (all_front_hip_abs or all_back_hip_abs) else 1.0

    # 鼻颈偏航角参考宽度：面部宽度约为肩宽的0.2倍
    face_ref_width = ref_shoulder_width * 0.2

    # ====== 第2遍：使用肩髋加权计算映射角度（用于连续性阈值和滤波步长） ======
    # 与第3遍使用完全一致的计算方式，确保max_abs_diff准确反映真实帧间变化
    w_sh = shoulder_weight
    w_hip = 1.0 - shoulder_weight
    all_mapped_abs = []
    for frame in anchor_seq:
        for person in frame.get("people", []):
            flat = _get_flat_keypoints(person)
            neck = _get_point(flat, IDX_NECK)
            r_shoulder = _get_point(flat, IDX_R_SHOULDER)
            l_shoulder = _get_point(flat, IDX_L_SHOULDER)
            r_hip = _get_point(flat, IDX_R_HIP)
            l_hip = _get_point(flat, IDX_L_HIP)

            def valid(p): return _is_valid(p, conf_threshold)
            shldr_valid_p2 = valid(r_shoulder) and valid(l_shoulder)
            hip_valid_p2 = valid(r_hip) and valid(l_hip)
            if not shldr_valid_p2 and not hip_valid_p2:
                continue

            shldr_dx = r_shoulder[0] - l_shoulder[0] if shldr_valid_p2 else None
            hip_dx = r_hip[0] - l_hip[0] if hip_valid_p2 else None

            scale = 1.0
            if valid(neck) and valid(r_hip) and valid(l_hip):
                hip_cx = (r_hip[0] + l_hip[0]) / 2.0
                hip_cy = (r_hip[1] + l_hip[1]) / 2.0
                nh_dist = math.hypot(neck[0] - hip_cx, neck[1] - hip_cy)
                if nh_dist > 1e-6:
                    scale = anchor_neck_hip / nh_dist

            # 肩部角度
            ang_sh = None
            if shldr_dx is not None:
                shldr_corr = abs(shldr_dx) * scale
                ratio_sh = shldr_corr / ref_shoulder_width if ref_shoulder_width > 0 else 1.0
                ratio_sh = max(0.0, min(1.0, ratio_sh))
                ang_sh = math.degrees(math.acos(ratio_sh))

            # 髋部角度
            ang_hip = None
            if hip_dx is not None:
                hip_corr = abs(hip_dx) * scale
                ratio_hip = hip_corr / ref_hip_width if ref_hip_width > 0 else 1.0
                ratio_hip = max(0.0, min(1.0, ratio_hip))
                ang_hip = math.degrees(math.acos(ratio_hip))

            # 肩髋加权融合（同第3遍）
            valid_angs = []
            weights = []
            if ang_sh is not None:
                valid_angs.append(ang_sh)
                weights.append(w_sh)
            if ang_hip is not None:
                valid_angs.append(ang_hip)
                weights.append(w_hip)
            if valid_angs:
                w_sum = sum(weights) or 1.0
                weights = [w / w_sum for w in weights]
                body_abs = sum(a * w for a, w in zip(valid_angs, weights))
            else:
                continue

            # 正面/背面映射
            is_front = shldr_dx is not None and shldr_dx < 0
            mapped_abs = body_abs if is_front else 180.0 - body_abs
            all_mapped_abs.append(mapped_abs)

    # 基于帧间变化量确定连续性阈值和滤波步长（用P95抗噪声）
    if all_mapped_abs:
        abs_diffs = []
        for i in range(1, len(all_mapped_abs)):
            abs_diffs.append(abs(all_mapped_abs[i] - all_mapped_abs[i-1]))
        max_abs_diff = np.percentile(abs_diffs, 95) if len(abs_diffs) >= 3 else (max(abs_diffs) if abs_diffs else 30.0)
    else:
        max_abs_diff = 30.0

    continuity_threshold = max(30.0, max_abs_diff * 0.5)
    filter_step = max(max_angle_limit, max_abs_diff * 0.8)

    # ====== 第3遍：逐帧计算 ======
    frames_data = []
    raw_yaws = []
    last_valid_sign = 1        # 持久化的符号
    prev_raw_yaw = None        # 上一帧 raw_yaw（用于连续性检测）

    for frame_idx, frame in enumerate(pose_keypoints_segment):
        frame_person_yaws = []
        frame_rows = []

        for p_idx, person in enumerate(frame.get("people", [])):
            flat = _get_flat_keypoints(person)
            nose = _get_point(flat, IDX_NOSE)
            neck = _get_point(flat, IDX_NECK)
            r_shoulder = _get_point(flat, IDX_R_SHOULDER)
            l_shoulder = _get_point(flat, IDX_L_SHOULDER)
            r_hip = _get_point(flat, IDX_R_HIP)
            l_hip = _get_point(flat, IDX_L_HIP)
            r_ear = _get_point(flat, IDX_R_EAR)
            l_ear = _get_point(flat, IDX_L_EAR)
            r_knee = _get_point(flat, IDX_R_KNEE)
            l_knee = _get_point(flat, IDX_L_KNEE)
            r_ankle = _get_point(flat, IDX_R_ANKLE)
            l_ankle = _get_point(flat, IDX_L_ANKLE)

            def valid(p): return _is_valid(p, conf_threshold)

            shldr_valid_p3 = valid(r_shoulder) and valid(l_shoulder)
            hip_valid_p3 = valid(r_hip) and valid(l_hip)

            shldr_dx = r_shoulder[0] - l_shoulder[0] if shldr_valid_p3 else None
            hip_dx = r_hip[0] - l_hip[0] if hip_valid_p3 else None

            nose_neck_x = nose[0] - neck[0] if (valid(nose) and valid(neck)) else None

            scale = 1.0
            if valid(neck) and valid(r_hip) and valid(l_hip):
                hip_cx = (r_hip[0] + l_hip[0]) / 2.0
                hip_cy = (r_hip[1] + l_hip[1]) / 2.0
                nh_dist = math.hypot(neck[0] - hip_cx, neck[1] - hip_cy)
                if nh_dist > 1e-6:
                    scale = anchor_neck_hip / nh_dist

            shldr_corr = abs(shldr_dx) * scale if shldr_dx is not None else None
            hip_corr = abs(hip_dx) * scale if hip_dx is not None else None

            # ====== 幅值估算（肩髋独立计算signed yaw，再加权融合） ======
            is_front = shldr_dx is not None and shldr_dx < 0
            hip_is_front = hip_dx is not None and hip_dx < 0

            ang_sh = None
            ang_hip = None
            if shldr_corr is not None and ref_shoulder_width > 0:
                ratio_sh = shldr_corr / ref_shoulder_width
                ratio_sh = max(0.0, min(1.0, ratio_sh))
                ang_sh = math.degrees(math.acos(ratio_sh))
            if hip_corr is not None and ref_hip_width > 0:
                ratio_hip = hip_corr / ref_hip_width
                ratio_hip = max(0.0, min(1.0, ratio_hip))
                ang_hip = math.degrees(math.acos(ratio_hip))

            w_shoulder = shoulder_weight
            w_hip = 1.0 - shoulder_weight

            # 肩部独立signed yaw（使用肩部dx符号）
            yaw_sh = None
            if ang_sh is not None:
                yaw_sh = ang_sh if is_front else -(180.0 - ang_sh)

            # 髋部独立signed yaw（使用髋部自身的dx符号）
            yaw_hip = None
            if ang_hip is not None:
                yaw_hip = ang_hip if hip_is_front else -(180.0 - ang_hip)

            # 加权融合
            valid_yaws = []
            weights = []
            if yaw_sh is not None:
                valid_yaws.append(yaw_sh)
                weights.append(w_shoulder)
            if yaw_hip is not None:
                valid_yaws.append(yaw_hip)
                weights.append(w_hip)
            if valid_yaws:
                w_sum = sum(weights) or 1.0
                weights = [w / w_sum for w in weights]
                fused_yaw = sum(y * w for y, w in zip(valid_yaws, weights))
            else:
                fused_yaw = 0.0

            # 幅值供后续符号判定使用
            mapped_abs = abs(fused_yaw)
            mapped_abs_for_sign = mapped_abs

            # ====== 耳朵偏航角计算（基于鼻子在左右耳之间的x位置） ======
            # 耳朵间距约为眼间距的2倍，角度分辨率更高；受被遮挡时自然退化为单耳逻辑
            # 不受表情（眨眼、眯眼）影响，比眼睛更可靠
            ear_yaw_deg = None
            ear_confidence = 0.0
            r_ear_valid = valid(r_ear)
            l_ear_valid = valid(l_ear)
            nose_valid = valid(nose)

            if r_ear_valid and l_ear_valid and nose_valid:
                # 鼻子到两耳的x距离比值，右转为正，透视收缩时自动稳定
                # 依赖索引语义（r_ear=16是右耳），背身时耳朵不可见由valid()保护
                # 右转时鼻子靠近右耳→right_dist小, left_dist大→right_dist-left_dist为负→取反得正
                right_dist = abs(nose[0] - r_ear[0])
                left_dist = abs(nose[0] - l_ear[0])
                sum_dist = right_dist + left_dist
                if sum_dist > 1.0:
                    ratio = (right_dist - left_dist) / sum_dist
                    ear_yaw_deg = ratio * 90.0  # [-90°, +90°], 面向右为正
                    # 置信度 = |角度| / 90°，越接近0°越不可信
                    ear_confidence = abs(ear_yaw_deg) / 90.0
            elif (r_ear_valid or l_ear_valid) and nose_valid:
                # 单耳可见
                if r_ear_valid:
                    dx = nose[0] - r_ear[0]
                    ear_yaw_deg = max(-90.0, min(90.0, -dx * 3.0))
                else:
                    dx = nose[0] - l_ear[0]
                    ear_yaw_deg = max(-90.0, min(90.0, -dx * 3.0))
                # 单耳置信度比双耳稍低，同样基于角度大小
                ear_confidence = abs(ear_yaw_deg) / 90.0 * 0.8

            # ====== 腿部弯曲符号计算（基于膝在髋-踝连线两侧的叉积方向） ======
            # 两条腿各自独立计算弯曲方向，同号时取平均作为leg_sign
            leg_sign = 0
            leg_bend_avg = 0.0
            leg_valid_count = 0

            r_hip_valid_indiv = valid(r_hip)
            l_hip_valid_indiv = valid(l_hip)
            r_knee_valid = valid(r_knee)
            l_knee_valid = valid(l_knee)
            r_ankle_valid = valid(r_ankle)
            l_ankle_valid = valid(l_ankle)

            if r_hip_valid_indiv and r_knee_valid and r_ankle_valid:
                # 右腿：髋→踝连线 × 髋→膝向量（2D叉积Z分量）
                ra_x = r_ankle[0] - r_hip[0]
                ra_y = r_ankle[1] - r_hip[1]
                rk_x = r_knee[0] - r_hip[0]
                rk_y = r_knee[1] - r_hip[1]
                r_cross = ra_x * rk_y - ra_y * rk_x
                # 叉积符号：膝在连线哪一侧
                leg_bend_avg += (1.0 if r_cross > 0 else -1.0)
                leg_valid_count += 1

            if l_hip_valid_indiv and l_knee_valid and l_ankle_valid:
                # 左腿：髋→踝连线 × 髋→膝向量（2D叉积Z分量）
                la_x = l_ankle[0] - l_hip[0]
                la_y = l_ankle[1] - l_hip[1]
                lk_x = l_knee[0] - l_hip[0]
                lk_y = l_knee[1] - l_hip[1]
                l_cross = la_x * lk_y - la_y * lk_x
                leg_bend_avg += (1.0 if l_cross > 0 else -1.0)
                leg_valid_count += 1

            if leg_valid_count == 2:
                # 两条腿都有效时检查是否同号
                avg = leg_bend_avg / 2.0
                if abs(avg) >= 0.5:  # 同号（都>0或都<0）
                    leg_sign = -1 if avg > 0 else 1  # 取反：叉积>0的面朝左，>0赋-1
                # 异号时leg_sign保持0，不参与投票
            elif leg_valid_count == 1:
                avg = leg_bend_avg
                if abs(avg) >= 0.5:
                    leg_sign = -1 if avg > 0 else 1  # 同上

            # ====== 鼻颈偏航角计算（asin映射到±90°，不区正面/背面） ======
            nose_yaw_deg = None
            nose_confidence = 0.0
            if nose_neck_x is not None and face_ref_width > 1.0:
                scaled_nose_dx = nose_neck_x * scale
                nose_ratio = abs(scaled_nose_dx) / face_ref_width
                nose_ratio = min(1.0, nose_ratio)
                nose_yaw_deg = math.degrees(math.asin(nose_ratio))
                if scaled_nose_dx < 0:
                    nose_yaw_deg = -nose_yaw_deg
                # 置信度 = |角度| / 90°，越接近±90°越可信
                nose_confidence = abs(nose_yaw_deg) / 90.0

            # ====== 符号判定：综合耳朵、腿部弯曲与鼻颈 ======
            # 背身时左右耳在画面中的位置翻转，翻转耳朵符号以保持语义一致
            if not is_front and ear_yaw_deg is not None:
                ear_yaw_deg = -ear_yaw_deg

            # 头部复合运动降权：耳鼻偏差>45°时线性降权
            head_penalty = 1.0
            if ear_yaw_deg is not None and nose_yaw_deg is not None:
                angle_diff = abs(ear_yaw_deg - nose_yaw_deg)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff > 20.0:
                    ratio = (angle_diff - 20.0) / (110.0 - 20.0)
                    head_penalty = max(0.3, 1.0 - ratio * 0.7)

            # 分别赋予各信号一个角度值，加权平均后取符号
            ear_angle = 0.0
            leg_angle = 0.0
            nose_angle = 0.0
            ear_w = 0.0
            leg_w = 0.0
            nose_w = 0.0

            if ear_yaw_deg is not None:
                ear_angle = ear_yaw_deg
                ear_w = 1.0 * head_penalty

            if leg_sign != 0:
                leg_angle = 90.0 if leg_sign > 0 else -90.0
                leg_w = 1.0

            if nose_yaw_deg is not None:
                nose_angle = nose_yaw_deg
                nose_w = 1.0 * head_penalty

            total_w = ear_w + leg_w + nose_w
            if total_w > 0.001:
                fused_angle = (ear_angle * ear_w + leg_angle * leg_w + nose_angle * nose_w) / total_w
                effective_sign = 1 if fused_angle > 0 else -1
            else:
                effective_sign = None

            # 符号 × 幅值 = 最终偏航角
            if effective_sign is not None:
                raw_yaw = effective_sign * mapped_abs_for_sign
            else:
                raw_yaw = 0.0

            # ====== 连续性约束 ======
            if enable_continuity and effective_sign is not None:
                flip_jump = 2.0 * mapped_abs_for_sign
                safe_to_flip = flip_jump <= continuity_threshold
                cand_signs = [effective_sign, -effective_sign]
                if prev_raw_yaw is not None:
                    diff0 = abs(cand_signs[0] * mapped_abs_for_sign - prev_raw_yaw)
                    diff1 = abs(cand_signs[1] * mapped_abs_for_sign - prev_raw_yaw)
                    if diff1 < diff0 - (1e-6 if safe_to_flip else continuity_threshold):
                        best_sign = cand_signs[1]
                    else:
                        best_sign = cand_signs[0]
                    raw_yaw = best_sign * mapped_abs_for_sign
                else:
                    raw_yaw = raw_yaw

            prev_raw_yaw = raw_yaw

            frame_person_yaws.append(raw_yaw)
            frame_rows.append((frame_idx, p_idx, shldr_dx, hip_dx,
                               nose_neck_x, shldr_corr, hip_corr,
                               mapped_abs, is_front, 0, raw_yaw,
                               0.0,
                               ear_yaw_deg, ear_confidence,
                               nose_yaw_deg, nose_confidence,
                               leg_sign))

        raw_yaws.extend(frame_person_yaws)
        frames_data.append(frame_rows)

    # ====== 后处理 ======
    num_frames = len(pose_keypoints_segment)
    raw_seq = raw_yaws

    cum_raw = _unwrap_sequence(raw_seq)

    cum_interp = cum_raw[:]
    for f_idx in range(num_frames):
        if cum_interp[f_idx] is not None:
            continue
        left = f_idx - 1
        while left >= 0 and cum_interp[left] is None:
            left -= 1
        right = f_idx + 1
        while right < num_frames and cum_interp[right] is None:
            right += 1
        if left >= 0 and right < num_frames:
            left_val = cum_interp[left]
            right_val = cum_interp[right]
            diff_val = right_val - left_val
            interpolated = left_val + diff_val * (f_idx - left) / (right - left)
            cum_interp[f_idx] = interpolated
        elif left >= 0:
            cum_interp[f_idx] = cum_interp[left]
        elif right < num_frames:
            cum_interp[f_idx] = cum_interp[right]

    if enable_filter:
        cum_filtered = [None] * num_frames
        for f_idx in range(num_frames):
            cur = cum_interp[f_idx]
            if cur is None:
                continue
            if f_idx == 0:
                cum_filtered[f_idx] = cur
                continue
            prev = cum_filtered[f_idx-1]
            if prev is None:
                cum_filtered[f_idx] = cur
                continue
            if abs(cur - prev) > filter_step:
                # 限步长而非卡死：每帧最多变化 filter_step 度
                direction = 1 if cur > prev else -1
                cum_filtered[f_idx] = prev + direction * filter_step
            else:
                cum_filtered[f_idx] = cur
    else:
        cum_filtered = cum_interp[:]

    if unwrap_angle:
        final_seq = cum_filtered
    else:
        final_seq = _wrap_to_180(cum_filtered)

    yaw_list = [a if a is not None else 0.0 for a in final_seq]
    yaw_tensor = torch.tensor(yaw_list, dtype=torch.float32)

    # ====== 诊断表格 ======
    raw_yaws_for_coverage = [a for a in raw_seq if a is not None]
    start_angle = None
    range_str = ""
    if raw_yaws_for_coverage:
        start_angle = raw_yaws_for_coverage[0]
        pos_angles = sorted(set(a for a in raw_yaws_for_coverage if a >= 0))
        neg_angles = sorted(set(a for a in raw_yaws_for_coverage if a < 0), reverse=True)

        def merge_intervals(angles_sorted, min_val, max_val):
            if not angles_sorted:
                return []
            intervals = []
            cur_start = angles_sorted[0]
            cur_end = angles_sorted[0]
            for ang in angles_sorted[1:]:
                if ang - cur_end <= merge_threshold:
                    cur_end = ang
                else:
                    intervals.append((cur_start, cur_end))
                    cur_start = ang
                    cur_end = ang
            intervals.append((cur_start, cur_end))
            return intervals

        pos_intervals = merge_intervals(pos_angles, 0, 180)
        neg_angles_asc = sorted(set(a for a in raw_yaws_for_coverage if a < 0))
        neg_intervals = merge_intervals(neg_angles_asc, -180, 0)
        range_parts = []
        for start, end in pos_intervals:
            range_parts.append(f"[{start:.0f}, {end:.0f}]")
        for start, end in neg_intervals:
            range_parts.append(f"[{start:.0f}, {end:.0f}]")
        range_str = " U ".join(range_parts) if range_parts else "None"
    else:
        start_angle = "N/A"
        range_str = "N/A"

    lines = []
    if isinstance(start_angle, float):
        lines.append(f"Start Angle (folded): {start_angle:.1f}")
    else:
        lines.append(f"Start Angle (folded): {start_angle}")
    lines.append(f"Coverage Ranges: {range_str}")
    lines.append(f"Filter: {'ON' if enable_filter else 'OFF'}  |  Unwrap: {'ON' if unwrap_angle else 'OFF'}"
                 f"  |  Continuity: {'ON' if enable_continuity else 'OFF'}"
                 f"  |  NoseThresh: {nose_neck_thresh:.0f}°")
    lines.append(f"Ref_sh_abs: {ref_shoulder_width:.1f} (P95_front={ref_sh_front if ref_sh_front else 0:.1f}, P95_back={ref_sh_back if ref_sh_back else 0:.1f})"
                 f"  |  Ref_hip_abs: {ref_hip_width:.1f}  |  Anchor_NH: {anchor_neck_hip:.1f}"
                 f"  |  FaceRef: {face_ref_width:.1f}")
    lines.append(f"MaxDiff: {max_abs_diff:.1f} (P95)  |  ContThrsh: {continuity_threshold:.1f}  |  Step: {filter_step:.1f}")
    lines.append("")

    header = (f"{'Frame':>5} {'Pers':>4} | "
              f"{'ShldrX':>8} {'HipX':>8} {'NoseNekX':>8} "
              f"{'CorSh':>8} {'CorHip':>7} "
              f"{'F/B':>4} {'MappedAbs':>9} "
              f"{'EarYaw':>7} {'EConf':>5} "
              f"{'LegVote':>8} {'NoseYaw':>8} {'NConf':>5} "
              f"{'Sign':>5} {'RawYaw':>8} {'FinalYaw':>9}")
    lines.append(header)
    lines.append("-" * len(header))

    for frame_idx, frame_rows in enumerate(frames_data):
        for (f_idx, p_idx, shldr_dx, hip_dx,
             nose_neck_x, shldr_corr, hip_corr,
             mapped_abs, is_front, sign, raw_yaw,
             mapped_abs_for_sign,
             ear_yaw_deg, ear_confidence,
             nose_yaw_deg, nose_confidence,
             leg_sign) in frame_rows:
            final_yaw = final_seq[frame_idx] if frame_idx < len(final_seq) else None

            def fmt(val, w=8):
                if val is None:
                    return " " * w
                if isinstance(val, float):
                    return f"{val:{w}.2f}"
                return f"{str(val):>{w}}"

            fb_str = "F" if is_front else "B"
            ear_yaw_str = fmt(ear_yaw_deg, 7)
            ear_conf_str = fmt(ear_confidence, 5)
            nose_yaw_str = fmt(nose_yaw_deg, 8)
            nose_conf_str = fmt(nose_confidence, 5)
            leg_vote_str = fmt(leg_sign, 8)
            line = (f"{f_idx:5d} {p_idx:4d} | "
                    f"{fmt(shldr_dx,8)} {fmt(hip_dx,8)} {fmt(nose_neck_x,8)} "
                    f"{fmt(shldr_corr,8)} {fmt(hip_corr,7)} "
                    f"{fb_str:>4} {fmt(mapped_abs,9)} "
                    f"{ear_yaw_str} {ear_conf_str} "
                    f"{leg_vote_str} {nose_yaw_str} {nose_conf_str} "
                    f"{fmt(sign,5)} {fmt(raw_yaw,8)} {fmt(final_yaw,9)}")
            lines.append(line)

    return (yaw_tensor, "\n".join(lines))



# ==================== 降级模式：6方向量化（半身/无髋场景） ====================
def _compute_yaw_quantized_6dir(pose_keypoints_segment, body_ref_dist, conf_threshold):
    """
    降级模式：无有效颈髋距离时使用（常见于半身舞蹈）。
    依赖肩宽比例映射到6个离散方向，正面判定放宽。
    
    6方向: 0°(正面), ±60°(前侧), ±120°(后侧), 180°(背面)
    肩髋均不可用时(body_ref_dist<=0)全部输出0°。
    
    Returns: (yaw_tensor, table_string)
    """
    FRONT_ZOOM = 1.2                # 正面放大系数，使比例→角度映射时正面区间更宽
    NOSE_FRONT_THRESH = 15.0        # 鼻颈X差阈值，超过才尝试区分正背面（放宽到15°）
    
    yaw_list = []
    
    for frame in pose_keypoints_segment:
        for person in frame.get("people", []):
            flat = _get_flat_keypoints(person)
            r_sh = _get_point(flat, 2)
            l_sh = _get_point(flat, 5)
            nose = _get_point(flat, 0)
            neck = _get_point(flat, 1)
            
            def v(p): return _is_valid(p, conf_threshold)
            shldr_ok = v(r_sh) and v(l_sh)
            
            # 肩髋均不可用 → 默认正面
            if body_ref_dist <= 0 or not shldr_ok:
                yaw_list.append(0.0)
                continue
            
            dx = r_sh[0] - l_sh[0]
            dx_abs = abs(dx)
            ratio = min(1.0, dx_abs / body_ref_dist)
            
            # 粗略角度：肩宽比例映射到0~90°（带正面放大系数）
            raw_abs = min(90.0, ratio * 90.0 * FRONT_ZOOM)
            
            # 符号：肩dx>0 表示右肩在画面右侧 → 面朝左；dx<0 → 面朝右
            sign = 1.0 if dx > 0 else -1.0
            
            # 判断正/背面：用鼻颈X差（放宽判定）
            nose_ok = v(nose) and v(neck)
            is_front = True
            if nose_ok:
                nose_neck_x = nose[0] - neck[0]
                if abs(nose_neck_x) > NOSE_FRONT_THRESH:
                    # 正面时鼻颈X差与肩dx符号相反（鼻子朝哪边转，哪边显宽）
                    if (dx < 0 and nose_neck_x > 0) or (dx > 0 and nose_neck_x < 0):
                        is_front = False
            
            if is_front:
                yaw = sign * raw_abs
            else:
                yaw = sign * (180.0 - raw_abs)
            
            # 量化到6方向：四舍五入到最近的60°倍数
            quantized = round(yaw / 60.0) * 60.0
            # 边界处理：clamp到[-180, 180]，-180统一为180
            if quantized > 180.0:
                quantized = 180.0
            elif quantized < -180.0:
                quantized = 180.0
            if quantized == -180.0:
                quantized = 180.0
            
            yaw_list.append(quantized)
    
    yaw_tensor = torch.tensor(yaw_list, dtype=torch.float32)
    
    # 统计各方向分布
    from collections import Counter
    direction_counts = Counter(yaw_list)
    dir_str = ", ".join(f"{d:.0f}°: {c}帧" for d, c in sorted(direction_counts.items()))
    
    # 构建诊断表格
    lines = []
    lines.append(f"【降级模式】无有效颈髋距离，使用肩宽比例映射到6方向")
    lines.append(f"参考尺度: {body_ref_dist:.1f} (0=肩髋均不可用→全部输出0°)")
    lines.append(f"正面放大系数: {FRONT_ZOOM} | 鼻颈阈值: {NOSE_FRONT_THRESH}")
    lines.append(f"6方向: 0°=正面  ±60°=前侧  ±120°=后侧  180°=背面")
    lines.append(f"方向分布: {dir_str}")
    
    return (yaw_tensor, "\n".join(lines))


# ==================== 简化版节点 ====================
class SDPoseEstimateYawSimple:
    """
    计算人物偏航角（简化版）。默认输出折叠 [-180,180] 角度。
    幅值估算使用肩/髋X坐标差（|dx|）。
    输出: yaw_array (FLOAT), yaw_table (STRING), yaw_json (STRING)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "shoulder_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "肩部角度融合权重，髋部权重自动为 1 - 该值"}),
                "enable_filter": ("BOOLEAN", {"default": False,
                                              "tooltip": "是否启用帧间限幅平滑滤波。"}),
                "enable_continuity": ("BOOLEAN", {"default": True,
                                                  "tooltip": "是否启用帧间连续性约束，防止短期无可靠信号时的符号翻转。关闭则完全依赖鼻子判定。"}),
            },
            "optional": {
                "pose_keypoints_full": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("yaw_array", "yaw_table", "yaw_json")
    FUNCTION = "calculate_yaw"
    CATEGORY = "SDPose"

    def calculate_yaw(self, pose_keypoints_segment, shoulder_weight,
                      enable_filter=False, enable_continuity=True,
                      pose_keypoints_full=None):
        # 硬编码参数
        conf_threshold = 0.3
        unwrap_angle = False
        nose_neck_thresh = 30.0
        max_angle_limit = 60.0

        yaw_tensor, yaw_table = compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            nose_neck_thresh, shoulder_weight,
            max_angle_limit,
            pose_keypoints_full,
            merge_threshold=30.0,
            enable_filter=enable_filter,
            enable_continuity=enable_continuity
        )
        yaw_list = [round(v, 2) for v in yaw_tensor.tolist()]
        yaw_json = json.dumps(yaw_list)
        return (yaw_tensor, yaw_table, yaw_json)


# ==================== 进阶版节点 ====================
class SDPoseEstimateYawAdvanced:
    """
    计算人物偏航角（完整参数版）。
    幅值估算使用肩/髋X坐标差（|dx|）。
    输出: yaw_array (FLOAT), yaw_table (STRING), yaw_json (STRING)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unwrap_angle": ("BOOLEAN", {"default": False}),
                "nose_neck_thresh": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 80.0, "step": 5.0,
                                               "tooltip": "鼻颈偏航角阈值(°)。头部偏转超过此角度才信任鼻子判定左右方向，低于此角度需依赖眼睛或连续性保护"}),
                "shoulder_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_angle_limit": ("FLOAT", {"default": 60.0, "min": 20.0, "max": 120.0, "step": 5.0}),
                "enable_filter": ("BOOLEAN", {"default": False,
                                              "tooltip": "是否启用帧间限幅平滑滤波"}),
                "enable_continuity": ("BOOLEAN", {"default": True,
                                                  "tooltip": "是否启用帧间连续性约束，防止短期无可靠信号时的符号翻转。"}),
                "coverage_merge_threshold": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 180.0, "step": 1.0,
                    "tooltip": "在计算偏航角覆盖范围时，角度间隔小于此值的相邻区间将被合并"
                }),
            },
            "optional": {
                "pose_keypoints_full": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("yaw_array", "yaw_table", "yaw_json")
    FUNCTION = "calculate_yaw"
    CATEGORY = "SDPose"

    def calculate_yaw(self, pose_keypoints_segment, conf_threshold, unwrap_angle,
                      nose_neck_thresh, shoulder_weight,
                      max_angle_limit,
                      enable_filter=False, enable_continuity=True,
                      coverage_merge_threshold=30.0,
                      pose_keypoints_full=None):
        yaw_tensor, yaw_table = compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            nose_neck_thresh, shoulder_weight,
            max_angle_limit,
            pose_keypoints_full,
            merge_threshold=coverage_merge_threshold,
            enable_filter=enable_filter,
            enable_continuity=enable_continuity
        )
        yaw_list = [round(v, 2) for v in yaw_tensor.tolist()]
        yaw_json = json.dumps(yaw_list)
        return (yaw_tensor, yaw_table, yaw_json)


# ==================== 节点映射 ====================
NODE_CLASS_MAPPINGS = {
    "SDPoseEstimateYawSimple": SDPoseEstimateYawSimple,
    "SDPoseEstimateYawAdvanced": SDPoseEstimateYawAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPoseEstimateYawSimple": "Estimate Yaw (Simple)",
    "SDPoseEstimateYawAdvanced": "Estimate Yaw (Advanced)",
}