# sdpose_draw.py
# 骨骼绘制引擎：常量定义 + KeypointDraw 类

import math
import numpy as np
import colorsys

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


# ==================== 绘制类 ====================
class KeypointDraw:
    @staticmethod
    def circle(canvas_np, center, radius, color, thickness=-1):
        cx, cy = center
        h, w = canvas_np.shape[:2]
        r_int = int(np.ceil(radius))
        y_min, y_max = max(0, cy - r_int), min(h, cy + r_int + 1)
        x_min, x_max = max(0, cx - r_int), min(w, cx + r_int + 1)
        if y_max <= y_min or x_max <= x_min:
            return
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def line(canvas_np, pt1, pt2, color, thickness=1):
        x0, y0, x1, y1 = int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1])
        h, w = canvas_np.shape[:2]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        points = []
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        if thickness > 1:
            r = thickness / 2.0
            for px, py in points:
                y_min = max(0, py - int(np.ceil(r)))
                y_max = min(h, py + int(np.ceil(r)) + 1)
                x_min = max(0, px - int(np.ceil(r)))
                x_max = min(w, px + int(np.ceil(r)) + 1)
                if y_max > y_min and x_max > x_min:
                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                    mask = (xx - px)**2 + (yy - py)**2 <= r**2
                    canvas_np[y_min:y_max, x_min:x_max][mask] = color
        else:
            for px, py in points:
                if 0 <= px < w and 0 <= py < h:
                    canvas_np[py, px] = color

    @staticmethod
    def fillConvexPoly(canvas_np, pts, color):
        if len(pts) < 3:
            return
        pts = np.array(pts, dtype=np.int32)
        h, w = canvas_np.shape[:2]
        y_min, y_max = max(0, pts[:,1].min()), min(h, pts[:,1].max()+1)
        x_min, x_max = max(0, pts[:,0].min()), min(w, pts[:,0].max()+1)
        if y_max <= y_min or x_max <= x_min:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = np.zeros((y_max-y_min, x_max-x_min), dtype=bool)
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i+1)%len(pts)]
            y1, y2 = p1[1], p2[1]
            if y1 == y2:
                continue
            if y1 > y2:
                p1, p2 = p2, p1
                y1, y2 = y2, y1
            edge_mask = (yy >= y1) & (yy < y2)
            if not edge_mask.any():
                continue
            x_interp = p1[0] + (yy - y1) * (p2[0] - p1[0]) / (y2 - y1)
            mask ^= edge_mask & (xx >= x_interp)
        canvas_np[y_min:y_max, x_min:x_max][mask] = color

    @staticmethod
    def ellipse2Poly(center, axes, angle, arc_start, arc_end, delta=1):
        axes = (axes[0]+0.5, axes[1]+0.5)
        angle = angle % 360
        if arc_start > arc_end:
            arc_start, arc_end = arc_end, arc_start
        while arc_start < 0:
            arc_start += 360
            arc_end += 360
        while arc_end > 360:
            arc_end -= 360
            arc_start -= 360
        if arc_end - arc_start > 360:
            arc_start, arc_end = 0, 360
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        pts = []
        for i in range(arc_start, arc_end+delta, delta):
            theta = math.radians(min(i, arc_end))
            x = axes[0] * math.cos(theta)
            y = axes[1] * math.sin(theta)
            px = int(round(center[0] + x*cos_a - y*sin_a))
            py = int(round(center[1] + x*sin_a + y*cos_a))
            pts.append([px, py])
        unique = []
        for p in pts:
            if not unique or tuple(p) != tuple(unique[-1]):
                unique.append(p)
        if len(unique) < 2:
            unique = [[center[0], center[1]], [center[0], center[1]]]
        return unique

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

    # ---------- 辅助方法: 绘制单根骨骼 ----------
    def _draw_single_limb(self, canvas, limb_idx, keypoints, scores, threshold, stick_width, color_scheme="v4_custom"):
        limb = DRAW_LIMBS[limb_idx]
        idx1, idx2 = limb[0]-1, limb[1]-1
        if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
            return
        p1, p2 = keypoints[idx1], keypoints[idx2]
        if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
            return
        Y = np.array([p1[0], p2[0]])
        X = np.array([p1[1], p2[1]])
        mX, mY = (X[0]+X[1])/2, (Y[0]+Y[1])/2
        length = math.hypot(X[0]-X[1], Y[0]-Y[1])
        if length < 1:
            return
        angle = math.degrees(math.atan2(X[0]-X[1], Y[0]-Y[1]))
        polygon = self.ellipse2Poly((int(mY), int(mX)), (int(length/2), stick_width), int(angle), 0, 360, 1)
        color = self._get_limb_color(limb_idx, color_scheme)
        self.fillConvexPoly(canvas, polygon, color)

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
                    r,g,b = colorsys.hsv_to_rgb(ie/len(HAND_EDGES), 1.0, 1.0)
                    color = (int(r*255), int(g*255), int(b*255))
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
                                 yaw=None, hand_scale=1.0, color_scheme="v4_custom"):
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

        bottom_legs, bottom_arms, bottom_faces = get_limb_group(bottom_side)
        top_legs, top_arms, top_faces = get_limb_group(top_side)

        # ---- 自适应臂腿覆盖规则 ----
        arms_cover = self._arms_cover_legs(yaw, keypoints)

        # ======================================================
        # Layer 0 (底层, 先画): 后侧四肢 + 后侧脸部骨骼 + 后侧手 + 后侧脚圆点 + 后侧眼
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
            if bottom_side == "left":
                foot_range = range(18, 21)
            elif bottom_side == "right":
                foot_range = range(21, 24)
            elif bottom_side == "both":
                foot_range = range(18, 24)
            else:
                foot_range = []
            for i in foot_range:
                if scores is not None and scores[i] < threshold: continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.circle(canvas, (x, y), 4, FOOT_DOT_COLORS[i - 18])

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
            if top_side == "left":
                foot_range = range(18, 21)
            elif top_side == "right":
                foot_range = range(21, 24)
            else:
                foot_range = []
            for i in foot_range:
                if scores is not None and scores[i] < threshold: continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.circle(canvas, (x, y), 4, FOOT_DOT_COLORS[i - 18])

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