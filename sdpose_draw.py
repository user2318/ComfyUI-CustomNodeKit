# sdpose_draw.py
# 骨骼绘制引擎：常量定义 + KeypointDraw 类

import math
import numpy as np
import colorsys
import cv2

# ==================== 常量定义 ====================
PROPAGATION_LIMBS = [
    [2, 3], [3, 4], [4, 5],   # 右臂
    [2, 6], [6, 7], [7, 8],   # 左臂
    [2, 9], [9, 10], [10, 11], # 右腿
    [2, 12], [12, 13], [13, 14], # 左腿
    [2, 1],                    # 颈 -> 鼻
]

DRAW_LIMBS = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
    [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]
]

HAND_EDGES = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]

# 脚部骨骼连线（踝关节→大趾），keypoints 数组中的 0-based 索引
# 右脚踝(body[10]) → 右脚大趾(foot[21])
# 左脚踝(body[13]) → 左脚大趾(foot[18])
FOOT_LIMBS = [
    [13, 18],   # 左脚踝 → 左脚大趾 (左 -> 索引小)
    [10, 21],   # 右脚踝 → 右脚大趾 (右 -> 索引大)
]

# 标准 DWPose/OpenPose 配色（与 comfyui_controlnet_aux 和 BodyRatioMapper 一致）
STANDARD_BODY_COLORS = [
    [255,   0,   0],  # limb  0 右肩 (颈→右肩)
    [255,  85,   0],  # limb  1 左肩 (颈→左肩)
    [255, 170,   0],  # limb  2 右肘 (右肩→右肘)
    [255, 255,   0],  # limb  3 右腕 (右肘→右腕)
    [170, 255,   0],  # limb  4 左肘 (左肩→左肘)
    [ 85, 255,   0],  # limb  5 左腕 (左肘→左腕)
    [  0, 255,   0],  # limb  6 右髋 (颈→右髋)
    [  0, 255,  85],  # limb  7 右膝 (右髋→右膝)
    [  0, 255, 170],  # limb  8 右踝 (右膝→右踝)
    [  0, 255, 255],  # limb  9 左髋 (颈→左髋)
    [  0, 170, 255],  # limb 10 左膝 (左髋→左膝)
    [  0,  85, 255],  # limb 11 左踝 (左膝→左踝)
    [  0,   0, 255],  # limb 12 颈→鼻
    [ 85,   0, 255],  # limb 13 鼻→右眼
    [170,   0, 255],  # limb 14 右眼→右耳
    [255,   0, 255],  # limb 15 鼻→左眼
    [255,   0, 170],  # limb 16 左眼→左耳
]

# 单色骨骼模式下的统一颜色（灰色）
MONOCHROME_LIMB_COLOR = [128, 128, 128]

# 预计算手部 20 条骨骼边缘的 HSV 颜色
HAND_EDGE_COLORS = [
    tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0))
    for i in range(len(HAND_EDGES))
]

# 新配色方案 V4：灰度兼容版 + 同侧同肢体内部区分
BODY_COLORS = [
    [145, 60, 35],     #  0 右臂上段·颈→右肩       L≈78  [右暗·臂]
    [145, 215, 250],   #  1 左臂上段·颈→左肩       L≈198 [左亮·臂]
    [115, 45, 25],     #  2 右臂中段·右肩→右肘     L≈65  [右暗·臂]
    [95, 70, 40],      #  3 右臂下段·右肘→右腕     L≈68  [右暗·臂]
    [130, 225, 255],   #  4 左臂中段·左肩→左肘     L≈210 [左亮·臂]
    [155, 205, 240],   #  5 左臂下段·左肘→左腕     L≈195 [左亮·臂]
    [45, 140, 38],     #  6 右腿上段·颈→右髋       L≈102 [右暗·腿]
    [55, 128, 32],     #  7 右腿中段·右髋→右膝     L≈95  [右暗·腿]
    [65, 118, 42],     #  8 右腿下段·右膝→右踝     L≈88  [右暗·腿]
    [172, 128, 218],   #  9 左腿上段·颈→左髋       L≈152 [左亮·腿]
    [182, 133, 228],   # 10 左腿中段·左髋→左膝     L≈161 [左亮·腿]
    [162, 122, 208],   # 11 左腿下段·左膝→左踝     L≈144 [左亮·腿]
    [135, 125, 115],   # 12 颈→鼻                  L≈130 [中线]
    [148, 58, 48],     # 13 脸右·鼻→右眼           L≈80  [右暗·脸]
    [128, 42, 35],     # 14 脸右·右眼→右耳         L≈68  [右暗·脸]
    [162, 208, 242],   # 15 脸左·鼻→左眼           L≈199 [左亮·脸]
    [172, 218, 248],   # 16 脸左·左眼→左耳         L≈207 [左亮·脸]
]

# 脚部骨骼颜色 - Standard (OpenPose 标准)，来自 OpenPose 官方配色
FOOT_LIMB_COLORS_STANDARD = [
    [0, 235, 150],    # 左脚踝→大趾 (青绿色)
    [100, 0, 215],    # 右脚踝→大趾 (蓝紫色)
]

# 脚部骨骼颜色 - V4 Custom（与对应小腿有明显区分）
FOOT_LIMB_COLORS_V4 = [
    [200, 180, 80],    # 左脚踝→大趾 (黄绿色，与左小腿紫灰[162,122,208]区分)
    [50, 30, 160],     # 右脚踝→大趾 (深蓝紫色，与右小腿暗绿[65,118,42]区分)
]

# 身体关节点圆点颜色 (索引 0-17, 统一蓝色系)
BODY_DOT_COLORS = [
    [75, 125, 210],    #  0 鼻 (头部)
    [75, 125, 210],    #  1 颈 (头部)
    [50, 85, 178],     #  2 右肩 (右臂深)
    [65, 105, 195],    #  3 右肘 (右臂中)
    [80, 125, 212],    #  4 右腕 (右臂浅)
    [55, 90, 172],     #  5 左肩 (左臂深)
    [70, 110, 189],    #  6 左肘 (左臂中)
    [85, 130, 206],    #  7 左腕 (左臂浅)
    [38, 68, 156],     #  8 右髋 (右腿深)
    [52, 82, 170],     #  9 右膝 (右腿中)
    [66, 96, 184],     # 10 右踝 (右腿浅)
    [42, 72, 150],     # 11 左髋 (左腿深)
    [56, 86, 164],     # 12 左膝 (左腿中)
    [70, 100, 178],    # 13 左踝 (左腿浅)
    [75, 125, 210],    # 14 右眼 (头部)
    [75, 125, 210],    # 15 左眼 (头部)
    [75, 125, 210],    # 16 右耳 (头部)
    [75, 125, 210],    # 17 左耳 (头部)
]

# 脚部关节点圆点颜色 (索引 18-23, 左亮右暗, 每点不同)
FOOT_DOT_COLORS = [
    [175, 205, 105],   # 18 左脚大趾  L≈190 [左亮]
    [160, 215, 130],   # 19 左脚小趾  L≈198 [左亮]
    [185, 195, 90],    # 20 左脚跟    L≈182 [左亮]
    [125, 65, 55],     # 21 右脚大趾  L≈75  [右暗]
    [110, 55, 45],     # 22 右脚小趾  L≈65  [右暗]
    [135, 75, 50],     # 23 右脚跟    L≈80  [右暗]
]

# 面部70点连线索引（与DWPose输出对应）
FACE_70_INDICES = {
    "contour": list(range(0, 17)),        # 脸部外轮廓 0-16
    "left_eyebrow": list(range(17, 22)),  # 左眉 17-21
    "right_eyebrow": list(range(22, 27)), # 右眉 22-26
    "nose": list(range(27, 36)),          # 鼻子 27-35（仅点）
    "left_eye": list(range(36, 42)),      # 左眼眶 36-41
    "right_eye": list(range(42, 48)),     # 右眼眶 42-47
    "inner_lip": list(range(60, 68)),     # 内嘴唇 60-67
    "left_pupil": 68,                     # 左瞳孔
    "right_pupil": 69,                    # 右瞳孔
}

# 分组辅助常量（DRAW_LIMBS索引）
# 右侧身体骨骼（画面左侧的右臂/右腿）
RIGHT_BODY_LIMBS = [0, 2, 3, 6, 7, 8]
# 左侧身体骨骼
LEFT_BODY_LIMBS = [1, 4, 5, 9, 10, 11]

# 拆分手臂和腿（用于分层绘制，先腿后臂）
RIGHT_ARM_LIMBS = [0, 2, 3]     # 右臂上/中/下段
RIGHT_LEG_LIMBS = [6, 7, 8]     # 右腿上/中/下段
LEFT_ARM_LIMBS = [1, 4, 5]      # 左臂上/中/下段
LEFT_LEG_LIMBS = [9, 10, 11]    # 左腿上/中/下段
ALL_ARM_LIMBS = [0, 1, 2, 3, 4, 5]     # 所有手臂
ALL_LEG_LIMBS = [6, 7, 8, 9, 10, 11]   # 所有腿
# 右侧脸部骨骼
RIGHT_FACE_LIMBS = [13, 14]
# 左侧脸部骨骼
LEFT_FACE_LIMBS = [15, 16]
# 中线
MID_LIMB = [12]
# 全部身体骨骼
ALL_BODY_LIMBS = list(range(12))

# 脚部骨骼分组（FOOT_LIMBS索引）
LEFT_FOOT_LIMBS = [0]   # 左脚
RIGHT_FOOT_LIMBS = [1]  # 右脚
ALL_FOOT_LIMBS = [0, 1] # 全部


# ==================== 绘制类 ====================
class KeypointDraw:
    # OpenCV 加速版本
    @staticmethod
    def circle(canvas_np, center, radius, color, thickness=-1):
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = int(round(radius))
        if r < 1:
            r = 1
        cv2.circle(canvas_np, (cx, cy), r, color, thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def line(canvas_np, pt1, pt2, color, thickness=1):
        x0, y0 = int(round(pt1[0])), int(round(pt1[1]))
        x1, y1 = int(round(pt2[0])), int(round(pt2[1]))
        if thickness > 1:
            cv2.line(canvas_np, (x0, y0), (x1, y1), color, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(canvas_np, (x0, y0), (x1, y1), color, thickness=1, lineType=cv2.LINE_AA)

    @staticmethod
    def fillConvexPoly(canvas_np, pts, color):
        if len(pts) < 3:
            return
        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(canvas_np, pts_array, color, lineType=cv2.LINE_AA)

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1):
        axes_int = (int(round(axes[0])), int(round(axes[1])))
        # OpenCV 使用顺时针角度（图像坐标系 y 轴向下）
        # 原始代码使用数学逆时针角度（y 轴向上），需取反
        angle_int = int(round((-angle) % 360))
        arc_start_int = int(round(arc_start % 360))
        arc_end_int = int(round(arc_end % 360))
        if arc_end_int < arc_start_int:
            arc_end_int += 360
        pts_raw = cv2.ellipse2Poly(
            (int(round(center[0])), int(round(center[1]))),
            axes_int,
            angle_int,
            arc_start_int,
            arc_end_int,
            delta
        )
        # 确保为 numpy.ndarray 再转换
        pts_np = np.asarray(pts_raw, dtype=np.int32).reshape(-1, 2)
        pts_list = pts_np.tolist()
        if len(pts_list) < 2:
            pts_list = [[int(round(center[0])), int(round(center[1]))],
                        [int(round(center[0])), int(round(center[1]))]]
        return pts_list

    @staticmethod
    def _get_limb_layer_order(yaw, keypoints, scores, threshold, nose_neck_threshold=3.0):
        """
        确定四层渲染的分组信息。
        
        返回 (bottom_side, top_side) 字符串:
            bottom_side — "left" 或 "right" 或 "both" 或 None
            top_side    — "left" 或 "right" 或 "both" 或 None
        """
        # ---------- 确定符号 ----------
        sign = 0
        yaw_val = yaw if yaw is not None else None
        if yaw_val is not None:
            if yaw_val >= 150.0 or yaw_val <= -150.0:
                # 背对镜头: 所有四肢在底层, 头部在顶层
                return "both", None
            # 接近正对时（yaw 绝对值很小），强制为"both"，避免微小偏航角导致误分层
            if abs(yaw_val) < 5.0:
                return "both", None
            sign = 1 if yaw_val > 0 else (-1 if yaw_val < 0 else 0)
        else:
            if keypoints is not None and len(keypoints) >= 2:
                p_nose = keypoints[0]
                p_neck = keypoints[1]
                nose_valid = (scores is None or (len(scores) > 0 and scores[0] >= threshold))
                neck_valid = (scores is None or (len(scores) > 1 and scores[1] >= threshold))
                if nose_valid and neck_valid and p_nose[0] > 0 and p_nose[1] > 0 and p_neck[0] > 0 and p_neck[1] > 0:
                    nose_neck_x = p_nose[0] - p_neck[0]
                    if nose_neck_x > nose_neck_threshold:
                        sign = 1
                    elif nose_neck_x < -nose_neck_threshold:
                        sign = -1
                    else:
                        sign = 0
                else:
                    sign = 0
            else:
                sign = 0

        # ---------- 按修正后的语义映射 ----------
        if sign > 0:
            return "left", "right"    # sign>0=面朝右→左侧=后侧(bottom), 右侧=前侧(top)
        elif sign < 0:
            return "right", "left"    # sign<0=面朝左→右侧=后侧(bottom), 左侧=前侧(top)
        else:
            return "both", None       # 正对: 所有在底层, 按Y排序

    def _sort_limbs_by_y(self, indices, keypoints, reverse=False):
        """按骨骼中点Y坐标排序。"""
        if not indices:
            return []
        y_mids = []
        for idx in indices:
            limb = DRAW_LIMBS[idx]
            p1 = keypoints[limb[0] - 1]
            p2 = keypoints[limb[1] - 1]
            y_mid = (p1[1] + p2[1]) / 2.0
            y_mids.append(y_mid)
        sorted_pairs = sorted(zip(indices, y_mids), key=lambda x: x[1], reverse=reverse)
        return [pair[0] for pair in sorted_pairs]

    @staticmethod
    def _get_limb_color(limb_idx, color_scheme="v4_custom"):
        """根据配色方案返回骨骼颜色。"""
        if color_scheme == "monochrome":
            return MONOCHROME_LIMB_COLOR
        elif color_scheme == "standard":
            return STANDARD_BODY_COLORS[limb_idx % len(STANDARD_BODY_COLORS)]
        else:  # "v4_custom"
            return BODY_COLORS[limb_idx % len(BODY_COLORS)]

    @staticmethod
    def _get_foot_limb_color(foot_idx, color_scheme="v4_custom"):
        """根据配色方案返回脚部骨骼颜色。foot_idx: FOOT_LIMBS 的索引（0=左脚, 1=右脚）。"""
        if color_scheme == "monochrome":
            return MONOCHROME_LIMB_COLOR
        elif color_scheme == "standard":
            return FOOT_LIMB_COLORS_STANDARD[foot_idx]
        else:  # "v4_custom"
            return FOOT_LIMB_COLORS_V4[foot_idx]

    # ---------- 辅助方法: 绘制单根骨骼 ----------
    def _draw_single_limb(self, canvas, limb_idx, keypoints, scores, threshold, stick_width, color_scheme="v4_custom"):
        limb = DRAW_LIMBS[limb_idx]
        idx1, idx2 = limb[0]-1, limb[1]-1
        if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
            return
        p1, p2 = keypoints[idx1], keypoints[idx2]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return

        # 计算中心、长度、角度
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-3:
            return
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # 椭圆半轴：长轴 = length/2，短轴 = stick_width//2 （确保至少为1）
        radius_long = max(1, int(length / 2))
        radius_short = max(1, stick_width)
        axes = (radius_long, radius_short)

        color = self._get_limb_color(limb_idx, color_scheme)

        # 直接使用 OpenCV 绘制填充椭圆
        cv2.ellipse(canvas, (int(cx), int(cy)), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
        

    # ---------- 辅助方法: 绘制单根脚部骨骼（踝→大趾） ----------
    def _draw_single_foot_limb(self, canvas, foot_idx, keypoints, scores, threshold, stick_width, color_scheme="v4_custom"):
        """绘制一根脚部骨骼（踝关节到脚尖线）。foot_idx: FOOT_LIMBS 的索引。"""
        limb = FOOT_LIMBS[foot_idx]
        idx1, idx2 = limb[0], limb[1]  # 0-based 索引
        if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
            return
        p1, p2 = keypoints[idx1], keypoints[idx2]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return

        # 计算中心、长度、角度
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-3:
            return
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # 椭圆半轴：长轴 = length/2，短轴 = stick_width//2 （确保至少为1）
        radius_long = max(1, int(length / 2))
        radius_short = max(1, stick_width)
        axes = (radius_long, radius_short)

        color = self._get_foot_limb_color(foot_idx, color_scheme)

        # 直接使用 OpenCV 绘制填充椭圆
        cv2.ellipse(canvas, (int(cx), int(cy)), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)

    # ---------- 辅助方法: 绘制手部(指定索引范围) ----------
    def _draw_hands_range(self, canvas, keypoints, scores, threshold, hand_start, hand_end, hand_point_size, eps, W, H, hand_scale=1.0, color_scheme="v4_custom"):
        """绘制指定索引范围的手部骨骼和圆点。hand_scale 控制手部缩放（1.0=原始大小），手腕点保持不动。"""
        if len(keypoints) < hand_end:
            return
        # 手腕锚点（手部关键点索引0，即 hand_start）
        wrist_x, wrist_y = keypoints[hand_start][0], keypoints[hand_start][1]
        
        # 如果 hand_scale != 1.0，先计算缩放后的位置
        if abs(hand_scale - 1.0) > 1e-6:
            scaled_kps = keypoints.copy()
            for i in range(hand_start + 1, hand_end):
                if scores is not None and i < len(scores) and scores[i] < threshold:
                    continue
                ox, oy = keypoints[i][0], keypoints[i][1]
                if ox > eps and oy > eps:
                    scaled_kps[i][0] = wrist_x + (ox - wrist_x) * hand_scale
                    scaled_kps[i][1] = wrist_y + (oy - wrist_y) * hand_scale
        else:
            scaled_kps = keypoints
        
        for ie, edge in enumerate(HAND_EDGES):
            idx1, idx2 = hand_start + edge[0], hand_start + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1,y1 = int(scaled_kps[idx1][0]), int(scaled_kps[idx1][1])
            x2,y2 = int(scaled_kps[idx2][0]), int(scaled_kps[idx2][1])
            if x1>eps and y1>eps and x2>eps and y2>eps and 0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H:
                if color_scheme == "monochrome":
                    color = MONOCHROME_LIMB_COLOR
                else:
                    color = HAND_EDGE_COLORS[ie]
                self.line(canvas, (x1,y1), (x2,y2), color, thickness=2)
        for i in range(hand_start, hand_end):
            if scores is not None and i < len(scores) and scores[i] < threshold:
                continue
            x,y = int(scaled_kps[i][0]), int(scaled_kps[i][1])
            if x>eps and y>eps and 0<=x<W and 0<=y<H:
                self.circle(canvas, (x,y), hand_point_size, (0,0,255))

    # ---------- 辅助方法: 绘制一组骨骼 ----------
    def _draw_limb_group(self, canvas, limb_indices, keypoints, scores, threshold, stick_width, reverse=False, color_scheme="v4_custom"):
        """绘制一组骨骼, 先按Y排序后画。reverse=True时Y降序（下方先画，上方后画）。"""
        sorted_idx = self._sort_limbs_by_y(limb_indices, keypoints, reverse=reverse)
        for i in sorted_idx:
            self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)

    # ---------- 辅助方法: 绘制一组脚部骨骼 ----------
    def _draw_foot_limb_group(self, canvas, foot_indices, keypoints, scores, threshold, stick_width, color_scheme="v4_custom"):
        """绘制一组脚部骨骼（踝→大趾线）。"""
        for i in foot_indices:
            self._draw_single_foot_limb(canvas, i, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)

    # ---------- 辅助方法: 绘制脚部圆点 ----------
    def _draw_foot_dots(self, canvas, keypoints, scores, threshold, foot_range, W, H, dot_size=4):
        """绘制指定范围内脚部关键点圆点。"""
        for i in foot_range:
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                self.circle(canvas, (x, y), dot_size, FOOT_DOT_COLORS[i - 18])

    # ---------- 自适应臂腿覆盖规则 ----------
    def _arms_cover_legs(self, yaw, keypoints):
        """
        判断臂是否应该覆盖腿（|偏航角| < 90°).
        """
        if yaw is not None:
            return abs(yaw) < 90.0
        
        # 回退：基于肩宽/髋宽比估算偏航角
        if keypoints is None or len(keypoints) < 18:
            return True
        shoulder_span = abs(keypoints[2][0] - keypoints[5][0])
        hip_span = abs(keypoints[8][0] - keypoints[11][0])
        neck = keypoints[1]
        hip_cy = (keypoints[8][1] + keypoints[11][1]) / 2.0
        neck_hip_y = max(abs(neck[1] - hip_cy), 1.0)
        
        max_span = max(shoulder_span, hip_span)
        if max_span < 5.0:
            return True
        ratio = max_span / (neck_hip_y * 0.4)
        return ratio > 0.25

    def draw_wholebody_keypoints(self, canvas, keypoints, scores=None, threshold=0.3,
                                 draw_body=True, draw_feet=True, draw_face=True, draw_hands=True,
                                 stick_width=4, face_point_size=3, hand_point_size=4,
                                 yaw=None, hand_scale=1.0, color_scheme="v4_custom",
                                 foot_mode="dots"):
        H, W = canvas.shape[:2]
        eps = 0.01

        # ---- 获取分组信息 ----
        bottom_side, top_side = self._get_limb_layer_order(
            yaw, keypoints, scores, threshold
        )

        # ---- 确定各分组包含的索引 ----
        def get_limb_group(side):
            """根据side返回(leg_indices, arm_indices, face_indices)"""
            if side == "left":
                return LEFT_LEG_LIMBS, LEFT_ARM_LIMBS, LEFT_FACE_LIMBS
            elif side == "right":
                return RIGHT_LEG_LIMBS, RIGHT_ARM_LIMBS, RIGHT_FACE_LIMBS
            elif side == "both":
                return ALL_LEG_LIMBS, ALL_ARM_LIMBS, []
            else:
                return [], [], []

        def get_foot_group(side):
            """根据side返回脚部骨骼索引和脚部圆点索引范围。"""
            if side == "left":
                return LEFT_FOOT_LIMBS, range(18, 21)
            elif side == "right":
                return RIGHT_FOOT_LIMBS, range(21, 24)
            elif side == "both":
                return ALL_FOOT_LIMBS, range(18, 24)
            else:
                return [], []

        bottom_legs, bottom_arms, bottom_faces = get_limb_group(bottom_side)
        top_legs, top_arms, top_faces = get_limb_group(top_side)

        # ---- 脚部前后分组 ----
        bottom_feet_limbs, bottom_feet_dots = get_foot_group(bottom_side)
        top_feet_limbs, top_feet_dots = get_foot_group(top_side)

        # ---- 自适应臂腿覆盖规则 ----
        arms_cover = self._arms_cover_legs(yaw, keypoints)

        # ======================================================
        # Layer 0 (底层, 先画): 后侧四肢 + 后侧脸部骨骼 + 后侧手 + 后侧脚 + 后侧眼
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            if bottom_side == "both":
                # 正对/背身: 保持现有 Y 降序模式
                all_limbs = ALL_LEG_LIMBS + ALL_ARM_LIMBS
                self._draw_limb_group(canvas, all_limbs, keypoints, scores, threshold, stick_width, reverse=True, color_scheme=color_scheme)
            else:
                # 侧面: 后侧四肢按自适应规则绘制
                if arms_cover:
                    # 臂覆盖腿：先画腿，再画臂
                    self._draw_limb_group(canvas, bottom_legs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                    self._draw_limb_group(canvas, bottom_arms, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                else:
                    # 腿覆盖臂：先画臂，再画腿
                    self._draw_limb_group(canvas, bottom_arms, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                    self._draw_limb_group(canvas, bottom_legs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
            # 脸部骨骼(后侧)
            for i in bottom_faces:
                self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)

        # 后侧手
        if draw_hands:
            if bottom_side == "left" and len(keypoints) >= 134:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)
            elif bottom_side == "right" and len(keypoints) >= 113:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)
            elif bottom_side == "both":
                if len(keypoints) >= 113:
                    self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)
                if len(keypoints) >= 134:
                    self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)

        # 后侧脚
        if draw_feet and len(keypoints) >= 24:
            if foot_mode == "line":
                # 线条模式：画踝→大趾线，不画圆点
                self._draw_foot_limb_group(canvas, bottom_feet_limbs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
            else:
                # dots 模式：画圆点（原有行为）
                self._draw_foot_dots(canvas, keypoints, scores, threshold, bottom_feet_dots, W, H)

        # 后侧眼 (面部点 DWPose: 左眼 36-41, 右眼 42-47, 在 keypoints 中索引为 24+36=60 到 24+48=72)
        if draw_face and len(keypoints) >= 92:
            if bottom_side == "left":
                eye_indices = list(range(60, 66))  # 左眼 face 36-41 → kp 60-65
            elif bottom_side == "right":
                eye_indices = list(range(66, 72))  # 右眼 face 42-47 → kp 66-71
            elif bottom_side == "both":
                eye_indices = list(range(60, 72))
            else:
                eye_indices = []
            for i in eye_indices:
                if scores is not None and i < len(scores) and scores[i] < threshold: continue
                x,y = int(keypoints[i][0]), int(keypoints[i][1])
                if x>eps and y>eps and 0<=x<W and 0<=y<H:
                    self.circle(canvas, (x,y), face_point_size, (255,255,255))

        # ======================================================
        # Layer 1 (中层): 颈→鼻骨骼 + 身体中线关键点圆点 + 面部其余点
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            self._draw_single_limb(canvas, 12, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)

        # 面部其余点 (轮廓 0-16、眉毛 17-26、鼻子 27-35、嘴唇 48-67、瞳孔 68-69)
        # 映射到 keypoints: 24 + face_idx
        # 排除眼睛 36-47 (已在 Layer 0/2)
        if draw_face and len(keypoints) >= 92:
            face_rest = list(range(24, 60)) + list(range(72, 92))  # face 0-35(轮廓+眉+鼻) + face 48-67(嘴唇)+68-69(瞳孔)
            for i in face_rest:
                if scores is not None and i < len(scores) and scores[i] < threshold: continue
                x,y = int(keypoints[i][0]), int(keypoints[i][1])
                if x>eps and y>eps and 0<=x<W and 0<=y<H:
                    self.circle(canvas, (x,y), face_point_size, (255,255,255))

        # ======================================================
        # Layer 2 (顶层, 后画): 前侧四肢 + 前侧脸部骨骼 + 前侧手 + 前侧脚 + 前侧眼
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            if top_side is not None and top_side != "both":
                # 侧面: 前侧四肢按自适应规则绘制
                if arms_cover:
                    # 臂覆盖腿：先画腿，再画臂
                    self._draw_limb_group(canvas, top_legs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                    self._draw_limb_group(canvas, top_arms, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                else:
                    # 腿覆盖臂：先画臂，再画腿
                    self._draw_limb_group(canvas, top_arms, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                    self._draw_limb_group(canvas, top_legs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
                for i in top_faces:
                    self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
            elif top_side == "both":
                # 背对时全部在 Layer 0 已处理
                pass
            else:
                # top_side is None（正对时）：无前侧四肢
                pass

        # 前侧手
        if draw_hands:
            if top_side == "left" and len(keypoints) >= 134:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)
            elif top_side == "right" and len(keypoints) >= 113:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H, hand_scale, color_scheme=color_scheme)

        # 前侧脚
        if draw_feet and len(keypoints) >= 24:
            if foot_mode == "line":
                # 线条模式：画踝→大趾线，不画圆点
                self._draw_foot_limb_group(canvas, top_feet_limbs, keypoints, scores, threshold, stick_width, color_scheme=color_scheme)
            else:
                # dots 模式：画圆点（原有行为）
                self._draw_foot_dots(canvas, keypoints, scores, threshold, top_feet_dots, W, H)

        # 前侧眼
        if draw_face and len(keypoints) >= 92:
            if top_side == "left":
                eye_indices = list(range(60, 66))  # 左眼
            elif top_side == "right":
                eye_indices = list(range(66, 72))  # 右眼
            else:
                eye_indices = []
            for i in eye_indices:
                if scores is not None and i < len(scores) and scores[i] < threshold: continue
                x,y = int(keypoints[i][0]), int(keypoints[i][1])
                if x>eps and y>eps and 0<=x<W and 0<=y<H:
                    self.circle(canvas, (x,y), face_point_size, (255,255,255))

        # ======================================================
        # Layer 3 (覆盖层): 全体身体关键点圆点 (0-17, 蓝色系)
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            for i in range(18):
                if scores is not None and scores[i] < threshold: continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.circle(canvas, (x, y), 4, BODY_DOT_COLORS[i])

        return canvas