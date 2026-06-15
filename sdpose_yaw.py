# sdpose_yaw.py
# 偏航角计算核心 + SDPoseEstimateYawSimple / SDPoseEstimateYawAdvanced 节点
# 
# 核心改进（相对旧算法）：
# 1. 身体尺度参考：躯干垂直高度替代颈髋欧氏距离（不受偏航旋转影响）
# 2. 角度映射：线性映射替代 acos（避免边界斜率无穷大）
# 3. 质量评分：筛选正对帧计算参考比值
# 4. 前/背身分离基准：各自独立映射，背身角度更接近180°
# 5. 比率时序平滑：5帧滑动窗口抑制抖动
# 6. 符号保护：微弱信号不能推翻肩部方向 + 正面权重衰减

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

def _smooth_ratios(ratios, window_size=5):
    """对比值序列做滑动窗口平均（边界处缩窄窗口）"""
    if not ratios or window_size <= 1:
        return ratios
    smoothed = []
    n = len(ratios)
    for i in range(n):
        half = window_size // 2
        left = max(0, i - half)
        right = min(n, i + half + 1)
        smoothed.append(sum(ratios[left:right]) / (right - left))
    return smoothed


# ==================== 偏航角核心计算 ====================
def compute_yaw_core(pose_keypoints_segment, conf_threshold, unwrap_angle,
                     nose_neck_thresh, shoulder_weight,
                     max_angle_limit,
                     pose_keypoints_full=None, merge_threshold=30.0,
                     enable_filter=False,
                     enable_continuity=True,
                     # ====== 新增参数（阶段5改进） ======
                     enable_smoothing=True,
                     smoothing_window=5,
                     front_back_separate=True,
                     enable_front_weight_attenuation=True,
                     front_atten_threshold=75.0,
                     front_atten_factor=0.3,
                     ear_min_angle=5.0,
                     nose_min_angle=5.0,
                     continuity_history=3):
    """
    核心偏航角计算（阶段5改进版）。
    
    使用躯干垂直高度作为身体尺度参考，线性映射替代acos，
    前/背身分离基准，时序平滑，符号保护机制。
    """
    IDX_NOSE = 0
    IDX_NECK = 1
    IDX_R_SHOULDER = 2
    IDX_L_SHOULDER = 5
    IDX_R_HIP = 8
    IDX_L_HIP = 11
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

    # ====== 第1遍：收集所有帧的数据 ======
    frame_data_list = []  # (sh_dx, hip_dx, torso_h, nose_neck_x, sh_y_diff, hip_y_diff, is_front)

    for frame in anchor_seq:
        for person in frame.get("people", []):
            flat = _get_flat_keypoints(person)
            neck = _get_point(flat, IDX_NECK)
            r_hip = _get_point(flat, IDX_R_HIP)
            l_hip = _get_point(flat, IDX_L_HIP)
            r_shoulder = _get_point(flat, IDX_R_SHOULDER)
            l_shoulder = _get_point(flat, IDX_L_SHOULDER)
            nose = _get_point(flat, IDX_NOSE)

            nv = _is_valid(neck, conf_threshold)
            hv = _is_valid(r_hip, conf_threshold) and _is_valid(l_hip, conf_threshold)
            sv = _is_valid(r_shoulder, conf_threshold) and _is_valid(l_shoulder, conf_threshold)
            nosev = _is_valid(nose, conf_threshold)

            sh_dx = r_shoulder[0] - l_shoulder[0] if sv else None
            hip_dx = r_hip[0] - l_hip[0] if hv else None

            torso_h = None
            nose_neck_x = None
            sh_y_diff = None
            hip_y_diff = None
            is_front = None

            if nv:
                if hv:
                    hip_cy = (r_hip[1] + l_hip[1]) / 2.0
                    torso_h = abs(neck[1] - hip_cy)
                    if torso_h < 1e-6:
                        torso_h = None
                if sv:
                    sh_y_diff = abs(r_shoulder[1] - l_shoulder[1])
                if nosev:
                    nose_neck_x = nose[0] - neck[0]
            if hv:
                hip_y_diff = abs(r_hip[1] - l_hip[1])

            if sh_dx is not None:
                is_front = sh_dx < 0

            frame_data_list.append((sh_dx, hip_dx, torso_h, nose_neck_x, sh_y_diff, hip_y_diff, is_front))
            break  # 只取第一个人
        else:
            frame_data_list.append((None, None, None, None, None, None, None))

    # 筛选有有效数据的帧
    valid_frames = [(i, d) for i, d in enumerate(frame_data_list) if d[0] is not None and d[2] is not None]
    if len(valid_frames) < 5:
        return _compute_yaw_quantized_6dir(pose_keypoints_segment, 0, conf_threshold)

    # ====== 第2遍：收集比值 + 质量评分 ======
    raw_ratios = []
    front_ratios = []
    back_ratios = []

    for i, (sh_dx, hip_dx, torso_h, nose_neck_x, sh_y_diff, hip_y_diff, is_front) in valid_frames:
        ratio = abs(sh_dx) / torso_h
        raw_ratios.append(ratio)
        if is_front:
            front_ratios.append(ratio)
        else:
            back_ratios.append(ratio)

    if not raw_ratios:
        return _compute_yaw_quantized_6dir(pose_keypoints_segment, 0, conf_threshold)

    # 初始参考值（P95）
    max_ratio_initial = np.percentile(raw_ratios, 95)
    if max_ratio_initial <= 0:
        max_ratio_initial = max(raw_ratios)
    if max_ratio_initial <= 0:
        max_ratio_initial = 1.0

    # 第1轮粗略角度（仅用于质量筛选）
    first_pass_angles = []
    for idx, (sh_dx, hip_dx, torso_h, nose_neck_x, sh_y_diff, hip_y_diff, is_front) in valid_frames:
        norm = (abs(sh_dx) / torso_h) / max_ratio_initial
        norm = max(0.0, min(1.0, norm))
        ang = math.degrees(math.acos(norm))
        first_pass_angles.append(ang)

    # 质量评分
    quality_scores = []
    for idx, (sh_dx, hip_dx, torso_h, nose_neck_x, sh_y_diff, hip_y_diff, is_front) in valid_frames:
        q = 0.0
        # ① 鼻颈X对齐度：正面时 nose_x - neck_x ≈ 0
        if nose_neck_x is not None and torso_h is not None and torso_h > 0:
            q_nose = 1.0 - min(1.0, abs(nose_neck_x) / torso_h)
            q += q_nose * 0.4
        # ② 肩部水平度
        if sh_y_diff is not None and torso_h is not None and torso_h > 0:
            q_sh = 1.0 - min(1.0, sh_y_diff / torso_h)
            q += q_sh * 0.3
        # ③ 髋部水平度
        if hip_y_diff is not None and torso_h is not None and torso_h > 0:
            q_hip = 1.0 - min(1.0, hip_y_diff / torso_h)
            q += q_hip * 0.2
        # ④ 角度接近0°
        if idx < len(first_pass_angles):
            q_ang = 1.0 - min(1.0, first_pass_angles[idx] / 60.0)
            q += q_ang * 0.1
        quality_scores.append((idx, q, is_front))

    quality_scores.sort(key=lambda x: -x[1])
    top_n = max(5, len(quality_scores) // 6)

    # ====== 按躯干高分组计算参考值（应对不同景别） ======
    # 收集所有帧数据
    all_torso_h = [d[2] for idx, d in valid_frames]
    all_is_front = [d[6] for idx, d in valid_frames]
    all_ratios_all = [abs(d[0]) / d[2] for idx, d in valid_frames]

    # 按躯干高分为3组（远/中/近景）
    h_arr = np.array(all_torso_h)
    h_33 = np.percentile(h_arr, 33) if len(h_arr) >= 3 else np.mean(h_arr)
    h_66 = np.percentile(h_arr, 66) if len(h_arr) >= 3 else np.mean(h_arr) * 1.5

    # 同时检测镜头突变边界
    shot_boundaries = [0]
    for i in range(1, len(all_torso_h)):
        curr_h = all_torso_h[i]; prev_h = all_torso_h[i - 1]
        if curr_h > 0 and prev_h > 0:
            rc = curr_h / prev_h if prev_h < curr_h else prev_h / curr_h
            if rc > 1.5: shot_boundaries.append(i)
    shot_boundaries.append(len(all_torso_h))
    # 合并短段
    merged = [shot_boundaries[0]]
    for b in shot_boundaries[1:]:
        if b - merged[-1] < 5: merged[-1] = b
        else: merged.append(b)
    if merged[-1] != shot_boundaries[-1]: merged.append(shot_boundaries[-1])
    merged[-1] = len(all_torso_h)
    shot_boundaries = merged

    # 把帧分配到 zoom segment + torso_h group
    def _get_torso_h_group(th):
        if th <= h_33: return 0  # 远景
        elif th <= h_66: return 1  # 中景
        else: return 2  # 近景

    # 每组的参考值：按(shot_idx, h_group)分组计算
    group_data = {}
    for si in range(len(shot_boundaries) - 1):
        ss, se = shot_boundaries[si], shot_boundaries[si + 1]
        for i in range(ss, se):
            if i >= len(all_torso_h): break
            hg = _get_torso_h_group(all_torso_h[i])
            key = (si, hg)
            if key not in group_data:
                group_data[key] = {"front": [], "back": [], "all": []}
            group_data[key]["all"].append(all_ratios_all[i])
            if all_is_front[i]:
                group_data[key]["front"].append(all_ratios_all[i])
            else:
                group_data[key]["back"].append(all_ratios_all[i])

    # 为每组计算参考值
    shot_refs = []
    for si in range(len(shot_boundaries) - 1):
        ss, se = shot_boundaries[si], shot_boundaries[si + 1]
        for hg in range(3):
            key = (si, hg)
            if key not in group_data or len(group_data[key]["all"]) < 5:
                continue
            d = group_data[key]
            sf = np.percentile(d["front"], 95) if d["front"] else np.percentile(d["all"], 95)
            sb = np.percentile(d["back"], 95) if d["back"] else sf
            sm = np.percentile(d["all"], 1)
            sf = max(sf, 1e-6); sb = max(sb, 1e-6); sm = max(sm, 1e-6)
            sfr = max(sf - sm, sf * 0.5); sbr = max(sb - sm, sb * 0.5)
            shot_refs.append((key, sf, sb, sm, sfr, sbr, ss, se))

    # 构造描述
    h_labels = {0: "远景", 1: "中景", 2: "近景"}
    desc_lines = ["阶段5: 景别感知+分离+平滑"]
    for key, sf, sb, sm, _, _, ss, se in shot_refs:
        si, hg = key
        end_frame = min(se, len(anchor_seq))
        desc_lines.append(f"  段{si+1}_{h_labels[hg]}: 帧~{end_frame} front={sf:.4f} back={sb:.4f} min={sm:.4f}")

    # 为每帧确定该用哪组参考值
    # 寻找帧号对应数组索引的函数
    def find_ref_for_frame(frame_idx):
        """根据帧号对应的原始帧索引，找到合适的分组参考值"""
        # 简化实现：使用第一个匹配的shot和h group
        for key, sf, sb, sm, sfr, sbr, ss, se in shot_refs:
            si, hg = key
            end_valid = min(se, len(all_torso_h))
            if ss <= frame_idx < end_valid:
                actual_hg = _get_torso_h_group(all_torso_h[frame_idx]) if frame_idx < len(all_torso_h) else hg
                if actual_hg == hg:
                    return sf, sb, sm, sfr, sbr
        # 回退
        return shot_refs[0][1], shot_refs[0][2], shot_refs[0][3], shot_refs[0][4], shot_refs[0][5] if shot_refs else (max_ratio_initial, max_ratio_initial, 0, 0, 0)

    # Use first group's reference as default
    if shot_refs:
        _, f0, b0, m0, fr0, br0, _, _ = shot_refs[0]
        front_max_ratio = f0; back_max_ratio = b0; min_ratio = m0; front_range = fr0; back_range = br0
    else:
        front_max_ratio = max_ratio_initial; back_max_ratio = max_ratio_initial
        min_ratio = np.percentile(raw_ratios, 1) if len(raw_ratios) >= 10 else max_ratio_initial * 0.15
        min_ratio = max(min_ratio, 1e-6); front_range = max(front_max_ratio - min_ratio, front_max_ratio * 0.5); back_range = front_range

    # ====== 第3遍：计算每帧 ratio（带平滑） ======
    seg_ratios = []
    seg_front_back = []
    seg_has_data = []

    for frame_idx, frame in enumerate(pose_keypoints_segment):
        person = None
        for p in frame.get("people", []):
            person = p
            break
        if person is None:
            seg_ratios.append(None)
            seg_front_back.append(None)
            seg_has_data.append(False)
            continue

        flat = _get_flat_keypoints(person)
        neck = _get_point(flat, IDX_NECK)
        r_hip = _get_point(flat, IDX_R_HIP)
        l_hip = _get_point(flat, IDX_L_HIP)
        r_shoulder = _get_point(flat, IDX_R_SHOULDER)
        l_shoulder = _get_point(flat, IDX_L_SHOULDER)

        def valid(p): return _is_valid(p, conf_threshold)
        nv = valid(neck)
        hv = valid(r_hip) and valid(l_hip)
        sv = valid(r_shoulder) and valid(l_shoulder)

        if not sv or not nv or not hv:
            seg_ratios.append(None)
            seg_front_back.append(None)
            seg_has_data.append(False)
            continue

        sh_dx = r_shoulder[0] - l_shoulder[0]
        hip_cy = (r_hip[1] + l_hip[1]) / 2.0
        torso_h = abs(neck[1] - hip_cy)
        if torso_h < 1e-6:
            seg_ratios.append(None)
            seg_front_back.append(None)
            seg_has_data.append(False)
            continue

        ratio = abs(sh_dx) / torso_h
        seg_ratios.append(ratio)
        seg_front_back.append(sh_dx < 0)
        seg_has_data.append(True)

    # 时序平滑
    if enable_smoothing and smoothing_window > 1:
        valid_indices = [i for i, has in enumerate(seg_has_data) if has]
        valid_ratios = [seg_ratios[i] for i in valid_indices]
        smoothed = _smooth_ratios(valid_ratios, smoothing_window)
        for idx, val in zip(valid_indices, smoothed):
            seg_ratios[idx] = val

    # ====== 第4遍：逐帧计算偏航角 ======
    raw_yaws = []
    prev_raw_yaw = None
    history_signs = []
    rotation_direction = 0

    for frame_idx in range(len(pose_keypoints_segment)):
        if not seg_has_data[frame_idx]:
            raw_yaws.append(0.0)
            continue

        ratio = seg_ratios[frame_idx]
        is_front = seg_front_back[frame_idx]

        # 用该帧所属景别组的参考值
        _ref = find_ref_for_frame(frame_idx)
        _fm, _bm, _mn, _fr, _br = _ref
        if is_front:
            cur_max_ratio = _fm
            cur_range = _fr
            min_ratio = _mn
        else:
            cur_max_ratio = _bm
            cur_range = _br
            min_ratio = _mn

        # 线性映射：[min_ratio, max_ratio] → [90°, 0°]
        norm = (ratio - min_ratio) / cur_range if cur_range > 0 else 0.5
        norm = max(0.0, min(1.0, norm))
        body_abs = 90.0 * (1.0 - norm)
        body_abs = max(0.0, min(90.0, body_abs))

        if is_front:
            # 前身: body_abs=0°(正对) → 0°, body_abs=90°(侧身) → 90°
            mapped_abs = body_abs
        else:
            # 背身: body_abs=0°(完全背对,肩宽最大) → 180°, body_abs=90°(侧身,肩宽最小) → 90°
            mapped_abs = 180.0 - body_abs

        # ====== 获取帧数据用于符号判定 ======
        frame = pose_keypoints_segment[frame_idx]
        person = None
        for p in frame.get("people", []):
            person = p
            break
        if person is None:
            raw_yaws.append(0.0)
            continue

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
        nose_valid = valid(nose)
        neck_valid = valid(neck)

        # ====== 耳朵偏航角 ======
        ear_yaw_deg = None
        ear_confidence = 0.0
        r_ear_valid = valid(r_ear)
        l_ear_valid = valid(l_ear)

        if r_ear_valid and l_ear_valid and nose_valid:
            right_dist = abs(nose[0] - r_ear[0])
            left_dist = abs(nose[0] - l_ear[0])
            sum_dist = right_dist + left_dist
            if sum_dist > 1.0:
                ratio_e = (right_dist - left_dist) / sum_dist
                ear_yaw_deg = ratio_e * 90.0
                ear_confidence = abs(ear_yaw_deg) / 90.0
        elif (r_ear_valid or l_ear_valid) and nose_valid:
            if r_ear_valid:
                dx = nose[0] - r_ear[0]
                ear_yaw_deg = max(-90.0, min(90.0, -dx * 3.0))
            else:
                dx = nose[0] - l_ear[0]
                ear_yaw_deg = max(-90.0, min(90.0, -dx * 3.0))
            ear_confidence = abs(ear_yaw_deg) / 90.0 * 0.8

        # ====== 腿部弯曲 ======
        leg_sign = 0
        leg_bend_avg = 0.0
        leg_valid_count = 0

        r_hip_v = valid(r_hip); l_hip_v = valid(l_hip)
        r_knee_v = valid(r_knee); l_knee_v = valid(l_knee)
        r_ankle_v = valid(r_ankle); l_ankle_v = valid(l_ankle)

        if r_hip_v and r_knee_v and r_ankle_v:
            ra_x = r_ankle[0] - r_hip[0]; ra_y = r_ankle[1] - r_hip[1]
            rk_x = r_knee[0] - r_hip[0]; rk_y = r_knee[1] - r_hip[1]
            r_cross = ra_x * rk_y - ra_y * rk_x
            leg_bend_avg += (1.0 if r_cross > 0 else -1.0)
            leg_valid_count += 1

        if l_hip_v and l_knee_v and l_ankle_v:
            la_x = l_ankle[0] - l_hip[0]; la_y = l_ankle[1] - l_hip[1]
            lk_x = l_knee[0] - l_hip[0]; lk_y = l_knee[1] - l_hip[1]
            l_cross = la_x * lk_y - la_y * lk_x
            leg_bend_avg += (1.0 if l_cross > 0 else -1.0)
            leg_valid_count += 1

        if leg_valid_count == 2:
            avg = leg_bend_avg / 2.0
            if abs(avg) >= 0.5: leg_sign = -1 if avg > 0 else 1
        elif leg_valid_count == 1:
            avg = leg_bend_avg
            if abs(avg) >= 0.5: leg_sign = -1 if avg > 0 else 1

        # ====== 鼻颈偏航角 ======
        nose_neck_x = nose[0] - neck[0] if (nose_valid and neck_valid) else None
        # 获取当前帧的躯干高度用于面部参考宽度
        frame_torso_h = None
        for person_p in [person]:
            flat_p = _get_flat_keypoints(person_p)
            neck_p = _get_point(flat_p, IDX_NECK)
            r_hip_p = _get_point(flat_p, IDX_R_HIP)
            l_hip_p = _get_point(flat_p, IDX_L_HIP)
            if _is_valid(neck_p, conf_threshold) and _is_valid(r_hip_p, conf_threshold) and _is_valid(l_hip_p, conf_threshold):
                hip_cy_p = (r_hip_p[1] + l_hip_p[1]) / 2.0
                frame_torso_h = abs(neck_p[1] - hip_cy_p)
            break
        if frame_torso_h and frame_torso_h > 1.0:
            face_ref_width = cur_max_ratio * 0.2 * frame_torso_h
        else:
            face_ref_width = 50.0
        nose_yaw_deg = None
        nose_confidence = 0.0
        if nose_neck_x is not None:
            nose_norm = min(1.0, abs(nose_neck_x) / (face_ref_width + 1e-6))
            nose_yaw_deg = math.degrees(math.asin(nose_norm))
            if nose_neck_x < 0:
                nose_yaw_deg = -nose_yaw_deg
            nose_confidence = abs(nose_yaw_deg) / 90.0

        # ====== 符号判定 ======
        if not is_front and ear_yaw_deg is not None:
            ear_yaw_deg = -ear_yaw_deg

        head_penalty = 1.0
        if ear_yaw_deg is not None and nose_yaw_deg is not None:
            angle_diff = abs(ear_yaw_deg - nose_yaw_deg)
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if angle_diff > 20.0:
                ratio_h = min(1.0, (angle_diff - 20.0) / 90.0)
                head_penalty = max(0.3, 1.0 - ratio_h * 0.7)

        ear_angle = 0.0; leg_angle = 0.0; nose_angle = 0.0
        ear_w = 0.0; leg_w = 0.0; nose_w = 0.0
        shoulder_sign = 1 if is_front else -1

        if ear_yaw_deg is not None and abs(ear_yaw_deg) >= ear_min_angle:
            ear_angle = ear_yaw_deg
            ear_w = ear_confidence * head_penalty

        if leg_sign != 0:
            leg_dir = 1 if leg_sign > 0 else -1
            if leg_dir == shoulder_sign:
                leg_angle = 90.0 if leg_sign > 0 else -90.0
                leg_w = 1.0

        if nose_yaw_deg is not None and abs(nose_yaw_deg) >= nose_min_angle:
            nose_angle = nose_yaw_deg
            nose_w = nose_confidence * head_penalty

        # ====== 正面信号权重衰减：body_abs越小，耳/鼻权重越低 ======
        if enable_front_weight_attenuation:
            if body_abs < front_atten_threshold:
                progress = body_abs / front_atten_threshold
                atten = front_atten_factor + (1.0 - front_atten_factor) * progress
                ear_w *= atten
                nose_w *= atten

        total_w = ear_w + leg_w + nose_w
        if total_w > 0.001:
            fused_angle = (ear_angle * ear_w + leg_angle * leg_w + nose_angle * nose_w) / total_w
            effective_sign = 1 if fused_angle > 0 else -1
        else:
            if history_signs:
                hist_sign = 1 if sum(history_signs) > 0 else -1
                effective_sign = hist_sign
            else:
                effective_sign = shoulder_sign

        # ====== 微弱信号不翻符号 ======
        if effective_sign != shoulder_sign and total_w < 0.15:
            effective_sign = shoulder_sign

        history_signs.append(effective_sign)
        if len(history_signs) > continuity_history:
            history_signs.pop(0)

        folded_yaw = effective_sign * mapped_abs

        # ====== 旋转方向检测 + 180°边界穿越 ======
        if len(raw_yaws) >= 3:
            recent = raw_yaws[-3:] + [folded_yaw]
            deltas = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            clean_deltas = []
            for d in deltas:
                if d > 180: d -= 360
                elif d < -180: d += 360
                clean_deltas.append(d)
            avg_delta = sum(clean_deltas) / len(clean_deltas)
            if abs(avg_delta) > 10:
                rotation_direction = 1 if avg_delta > 0 else -1

        if rotation_direction != 0 and prev_raw_yaw is not None:
            cand = [folded_yaw]
            if folded_yaw > 0:
                cand.append(folded_yaw - 360)
            else:
                cand.append(folded_yaw + 360)
            best_fit = folded_yaw
            best_diff = float('inf')
            for c in cand:
                diff = abs(c - prev_raw_yaw)
                if diff < best_diff and diff < 360:
                    best_diff = diff
                    best_fit = c
            if best_fit != folded_yaw:
                delta = best_fit - prev_raw_yaw
                if (rotation_direction > 0 and delta > 0) or (rotation_direction < 0 and delta < 0):
                    folded_yaw = best_fit

        prev_raw_yaw = folded_yaw
        raw_yaws.append(folded_yaw)

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

    if unwrap_angle:
        final_seq = cum_interp
    else:
        final_seq = _wrap_to_180(cum_interp)

    yaw_list = [a if a is not None else 0.0 for a in final_seq]
    yaw_tensor = torch.tensor(yaw_list, dtype=torch.float32)

    # ====== 统计摘要 ======
    num_fr = len(pose_keypoints_segment)
    yarr = np.array(yaw_list)
    min_val = float(np.min(yarr))
    max_val = float(np.max(yarr))
    std_val = float(np.std(yarr))

    lines = []
    lines.append("=" * 60)
    lines.append("  偏航角估算结果")
    lines.append("=" * 60)
    lines.append(f"  总帧数: {num_fr}")
    lines.append(f"  角度范围: [{min_val:>+.1f}°, {max_val:>+.1f}°]  |  标准差: {std_val:>6.1f}°")
    lines.append("-" * 60)
    # 参数信息
    params = []
    params.append(f"Smooth={'ON' if enable_smoothing else 'OFF'}")
    params.append(f"Separate={'ON' if front_back_separate else 'OFF'}")
    params.append(f"Atten={'ON' if enable_front_weight_attenuation else 'OFF'}")
    lines.append(f"  参数: {' | '.join(params)}")
    lines.extend(desc_lines)
    lines.append("-" * 60)

    # 全量逐帧表
    # 偏航角: 正值=朝右, 负值=朝左; 0°=正对, ±180°=背对
    # 肩X差=右肩.x-左肩.x, 负值=正面, 正值=背面
    # 躯干高=|颈.y-髋中心.y|, 比值=|肩X差|/躯干高
    header = (f"{'帧号':>5}  {'偏航角':>8} {'方向':>4} "
              f"{'肩X差':>8} {'髋X差':>8} {'躯干高':>8} {'比值':>7} "
              f"{'耳信号':>8} {'鼻信号':>8} {'可信度':>6}")
    lines.append(header)
    lines.append("-" * 76)

    for frame_idx in range(num_fr):
        raw_final = final_seq[frame_idx] if frame_idx < len(final_seq) else 0.0
        final_yaw = raw_final if raw_final is not None else 0.0
        # 方向：正值=朝右(R), 负值=朝左(L)
        if final_yaw > 0:
            dir_str = "R"
        elif final_yaw < 0:
            dir_str = "L"
        else:
            dir_str = "C"

        # 提取每帧的原始数据+耳/鼻信号
        frame = pose_keypoints_segment[frame_idx]
        person = None
        for p in frame.get("people", []):
            person = p
            break

        sh_dx_str = "---"
        hip_dx_str = "---"
        torso_h_str = "---"
        ratio_str = "---"
        ear_sig = "---"
        nose_sig = "---"
        conf = "---"

        if person is not None:
            flat_diag = _get_flat_keypoints(person)

            # 原始数据
            neck_p = _get_point(flat_diag, IDX_NECK)
            r_sh_p = _get_point(flat_diag, IDX_R_SHOULDER)
            l_sh_p = _get_point(flat_diag, IDX_L_SHOULDER)
            r_hip_p = _get_point(flat_diag, IDX_R_HIP)
            l_hip_p = _get_point(flat_diag, IDX_L_HIP)
            nose_p = _get_point(flat_diag, IDX_NOSE)

            nv_d = _is_valid(neck_p, conf_threshold)
            sv_d = _is_valid(r_sh_p, conf_threshold) and _is_valid(l_sh_p, conf_threshold)
            hv_d = _is_valid(r_hip_p, conf_threshold) and _is_valid(l_hip_p, conf_threshold)

            if sv_d:
                sdx = r_sh_p[0] - l_sh_p[0]
                sh_dx_str = f"{sdx:>+6.0f}"
            if hv_d:
                hdx = r_hip_p[0] - l_hip_p[0]
                hip_dx_str = f"{hdx:>+6.0f}"
            if nv_d and hv_d:
                hip_cy_d = (r_hip_p[1] + l_hip_p[1]) / 2.0
                th = abs(neck_p[1] - hip_cy_d)
                if th > 1e-6:
                    torso_h_str = f"{th:>6.0f}"
                    if sv_d:
                        ratio_val = abs(sdx) / th
                        ratio_str = f"{ratio_val:>6.3f}"

            # 耳信号
            r_ear_p = _get_point(flat_diag, IDX_R_EAR)
            l_ear_p = _get_point(flat_diag, IDX_L_EAR)
            re_v = _is_valid(r_ear_p, conf_threshold)
            le_v = _is_valid(l_ear_p, conf_threshold)
            no_v = _is_valid(nose_p, conf_threshold)
            if no_v:
                if re_v and le_v:
                    rd = abs(nose_p[0] - r_ear_p[0])
                    ld = abs(nose_p[0] - l_ear_p[0])
                    sd = rd + ld
                    if sd > 1.0:
                        ear_val = ((rd - ld) / sd) * 90.0
                        if not (final_yaw >= 0):
                            ear_val = -ear_val
                        ear_sig = f"{ear_val:>+7.1f}"
                elif (re_v or le_v):
                    if re_v:
                        dx = nose_p[0] - r_ear_p[0]
                        ear_val = max(-90.0, min(90.0, -dx * 3.0))
                    else:
                        dx = nose_p[0] - l_ear_p[0]
                        ear_val = max(-90.0, min(90.0, -dx * 3.0))
                    if not (final_yaw >= 0):
                        ear_val = -ear_val
                    ear_sig = f"{ear_val:>+7.1f}"

            # 鼻信号
            nckv_d = _is_valid(neck_p, conf_threshold)
            if no_v and nckv_d and torso_h_str != "---":
                th_val = float(torso_h_str)
                if th_val > 1.0:
                    nnx = nose_p[0] - neck_p[0]
                    fw = cur_max_ratio * 0.2 * th_val
                    if fw > 1.0:
                        nn = min(1.0, abs(nnx) / fw)
                        nose_val = math.degrees(math.asin(nn))
                        if nnx < 0:
                            nose_val = -nose_val
                        nose_sig = f"{nose_val:>+7.1f}"

            # 可信度
            try:
                ear_abs = abs(ear_val)
            except:
                ear_abs = 0
            try:
                nose_abs = abs(nose_val)
            except:
                nose_abs = 0
            conf_score = max(ear_abs, nose_abs) / 90.0
            if conf_score > 0.7:
                conf = "高"
            elif conf_score > 0.3:
                conf = "中"
            elif conf_score > 0.0:
                conf = "低"
            else:
                conf = "无"

        line = (f"{frame_idx+1:>5d} {final_yaw:>+10.1f}° {dir_str:>4} "
                f"{sh_dx_str:>8} {hip_dx_str:>8} {torso_h_str:>8} {ratio_str:>7} "
                f"{ear_sig:>8} {nose_sig:>8} {conf:>6}")
        lines.append(line)

    return (yaw_tensor, "\n".join(lines))


# ==================== 降级模式：6方向量化（半身/无髋场景） ====================
def _compute_yaw_quantized_6dir(pose_keypoints_segment, body_ref_dist, conf_threshold):
    """
    降级模式：无有效躯干高度时使用。
    依赖肩宽比例映射到6个离散方向。
    """
    FRONT_ZOOM = 1.2
    NOSE_FRONT_THRESH = 15.0

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

            if body_ref_dist <= 0 or not shldr_ok:
                yaw_list.append(0.0)
                continue

            dx = r_sh[0] - l_sh[0]
            dx_abs = abs(dx)
            ratio = min(1.0, dx_abs / body_ref_dist)

            raw_abs = min(90.0, ratio * 90.0 * FRONT_ZOOM)
            sign = 1.0 if dx > 0 else -1.0

            nose_ok = v(nose) and v(neck)
            is_front = True
            if nose_ok:
                nose_neck_x = nose[0] - neck[0]
                if abs(nose_neck_x) > NOSE_FRONT_THRESH:
                    if (dx < 0 and nose_neck_x > 0) or (dx > 0 and nose_neck_x < 0):
                        is_front = False

            if is_front:
                yaw = sign * raw_abs
            else:
                yaw = sign * (180.0 - raw_abs)

            quantized = round(yaw / 60.0) * 60.0
            if quantized > 180.0:
                quantized = 180.0
            elif quantized < -180.0:
                quantized = 180.0
            if quantized == -180.0:
                quantized = 180.0

            yaw_list.append(quantized)

    yaw_tensor = torch.tensor(yaw_list, dtype=torch.float32)

    from collections import Counter
    direction_counts = Counter(yaw_list)
    dir_str = ", ".join(f"{d:.0f}°: {c}帧" for d, c in sorted(direction_counts.items()))

    lines = []
    lines.append("【降级模式】无有效躯干高度，使用肩宽比例映射到6方向")
    lines.append(f"参考尺度: {body_ref_dist:.1f} (0=全部输出0°)")
    lines.append(f"正面放大系数: {FRONT_ZOOM} | 鼻颈阈值: {NOSE_FRONT_THRESH}")
    lines.append(f"6方向: 0°=正面  ±60°=前侧  ±120°=后侧  180°=背面")
    lines.append(f"方向分布: {dir_str}")

    return (yaw_tensor, "\n".join(lines))


# ==================== 简化版节点 ====================
class SDPoseEstimateYawSimple:
    """
    计算人物偏航角（简化版）。
    使用躯干垂直高度作为身体尺度参考，线性映射，前/背身分离基准。
    输出: yaw_array (FLOAT), yaw_table (STRING), yaw_json (STRING)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                               "tooltip": "关键点置信度阈值，提高可减少噪声但可能丢失有效数据。Keypoint confidence threshold; higher values reduce noise but may lose valid data."}),
                "enable_smoothing": ("BOOLEAN", {"default": True,
                                                  "tooltip": "平滑开关，开启时对角度做时序平滑（5帧滑动窗口），关闭时跟随原始关键点更灵敏。Smoothing switch; when enabled, applies temporal smoothing (5-frame sliding window); when disabled, follows raw keypoints more responsively."}),
            },
            "optional": {
                "pose_keypoints_full": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("yaw_array", "yaw_table", "yaw_json")
    FUNCTION = "calculate_yaw"
    CATEGORY = "SDPose"

    def calculate_yaw(self, pose_keypoints_segment, conf_threshold=0.3, enable_smoothing=True,
                      pose_keypoints_full=None):
        unwrap_angle = False
        shoulder_weight = 0.7
        max_angle_limit = 60.0

        yaw_tensor, yaw_table = compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            30.0, shoulder_weight,
            max_angle_limit,
            pose_keypoints_full,
            merge_threshold=30.0,
            enable_filter=False,
            enable_continuity=True,
            # 阶段5改进默认开启
            enable_smoothing=enable_smoothing,
            smoothing_window=5,
            front_back_separate=True,
            enable_front_weight_attenuation=True,
            front_atten_threshold=75.0,
            front_atten_factor=0.3,
            ear_min_angle=5.0,
            nose_min_angle=5.0,
            continuity_history=3,
        )
        yaw_list = [round(v, 2) for v in yaw_tensor.tolist()]
        yaw_json = json.dumps(yaw_list)
        return (yaw_tensor, yaw_table, yaw_json)


# ==================== 进阶版节点 ====================
class SDPoseEstimateYawAdvanced:
    """
    计算人物偏航角（完整参数版）。
    使用躯干垂直高度作为身体尺度参考，线性映射，前/背身分离基准。
    输出: yaw_array (FLOAT), yaw_table (STRING), yaw_json (STRING)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unwrap_angle": ("BOOLEAN", {"default": False}),
                "shoulder_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_smoothing": ("BOOLEAN", {"default": True,
                                                  "tooltip": "是否启用比率时序平滑（5帧滑动窗口）。Whether to enable ratio temporal smoothing (5-frame sliding window)."}),
                "front_back_separate": ("BOOLEAN", {"default": True,
                                                     "tooltip": "前/背身分离基准，背身角度更接近180°。Front/back separate baseline, back angles closer to 180°."}),
                "enable_front_weight_attenuation": ("BOOLEAN", {"default": True,
                                                                 "tooltip": "正面范围削弱耳/鼻信号权重，防止符号误翻转。Attenuate ear/nose signal weights in front-facing range to prevent sign flipping."}),
                "front_atten_threshold": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 120.0, "step": 5.0,
                                                     "tooltip": "正面权重衰减阈值(°)。Front weight attenuation threshold (°)."}),
                "ear_min_angle": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 1.0,
                                             "tooltip": "耳信号启用最小角度(°)，小于此忽略。Minimum angle for ear signal activation (°); ignored below this."}),
                "nose_min_angle": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 1.0,
                                              "tooltip": "鼻信号启用最小角度(°)。Minimum angle for nose signal activation (°)."}),
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
                      shoulder_weight,
                      enable_smoothing=True,
                      front_back_separate=True,
                      enable_front_weight_attenuation=True,
                      front_atten_threshold=75.0,
                      ear_min_angle=5.0,
                      nose_min_angle=5.0,
                      pose_keypoints_full=None):
        max_angle_limit = 60.0

        yaw_tensor, yaw_table = compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            30.0, shoulder_weight,
            max_angle_limit,
            pose_keypoints_full,
            merge_threshold=30.0,
            enable_filter=False,
            enable_continuity=True,
            enable_smoothing=enable_smoothing,
            smoothing_window=5,
            front_back_separate=front_back_separate,
            enable_front_weight_attenuation=enable_front_weight_attenuation,
            front_atten_threshold=front_atten_threshold,
            front_atten_factor=0.3,
            ear_min_angle=ear_min_angle,
            nose_min_angle=nose_min_angle,
            continuity_history=3,
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