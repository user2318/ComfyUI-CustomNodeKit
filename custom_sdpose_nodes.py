# custom_sdpose_nodes.py
# 整合版：包含 SDPose 加载/切片/拼接、全身骨骼绘制、面部连线绘制、五官比例对齐

import torch
import numpy as np
import math
import colorsys
import json
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

import comfy.utils
import folder_paths

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

# 新配色方案 V3：灰度兼容版
# 核心策略——左侧高亮度(浅色) + 右侧低亮度(深色)，灰度化后左侧明显更亮、右侧明显更暗
# 彩色模式下：右侧暖/深色系(红橙深绿)，左侧冷/浅色系(青蓝紫亮绿)
# 灰度模式下：右侧(L≈83-112) vs 左侧(L≈167-206)，亮度差55-123，清晰可辨
# 头部关键点同样遵守：头右(暗) vs 头左(亮)，确保头部在灰度下也能区分左右
BODY_COLORS = [
    [135, 50, 20],     #  0 右臂上段 (2→3)      L≈72  [右暗·臂]
    [145, 215, 250],   #  1 左臂上段 (2→6)      L≈198 [左亮·臂]
    [140, 45, 15],     #  2 右臂中段 (3→4)      L≈70  [右暗·臂]
    [130, 55, 25],     #  3 右臂下段 (4→5)      L≈74  [右暗·臂]
    [140, 210, 245],   #  4 左臂中段 (6→7)      L≈193 [左亮·臂]
    [150, 218, 255],   #  5 左臂下段 (7→8)      L≈202 [左亮·臂]
    [50, 142, 40],     #  6 右腿上段 (2→9)      L≈103 [右暗·腿]
    [55, 145, 42],     #  7 右腿中段 (9→10)     L≈106 [右暗·腿]
    [45, 138, 38],     #  8 右腿下段 (10→11)    L≈99  [右暗·腿]
    [175, 128, 218],   #  9 左腿上段 (2→12)     L≈152 [左亮·腿]
    [180, 125, 215],   # 10 左腿中段 (12→13)    L≈152 [左亮·腿]
    [172, 132, 222],   # 11 左腿下段 (13→14)    L≈154 [左亮·腿]
    [140, 130, 120],   # 12 颈→鼻 (2→1)        L≈132 [中线]
    [140, 50, 40],     # 13 脸右 (1→15)         L≈76  [右暗·脸]
    [135, 45, 35],     # 14 右耳 (15→17)        L≈71  [右暗·脸]
    [165, 210, 245],   # 15 脸左 (1→16)         L≈201 [左亮·脸]
    [175, 205, 240],   # 16 左耳 (16→18)        L≈200 [左亮·脸]
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

# keypoints 索引
# 右手：92-112, 左手：113-133
# 脚：18-23, 右脚 21-23, 左脚 18-20
# 面部点：24-91, 右眼 42-47, 左眼 36-41, 右耳 face_idx 16? 左耳 face_idx 17?
# 面部关键点：face 0-69 映射到 kp 24-93

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
        含义：
            bottom_side = "left"  → 左侧放入 Layer 0 (底层后侧)
            bottom_side = "right" → 右侧放入 Layer 0
            bottom_side = "both"  → 所有四肢放入 Layer 0 (背身)
            top_side    = "left"  → 左侧放入 Layer 2 (顶层前侧)
            top_side    = "right" → 右侧放入 Layer 2
            top_side    = None    → Layer 2 为空 (正对/背身)
            所有身体骨骼(0-11)按此分组到 Layer 0/2
            脸部骨骼(13-16)按同侧分组到 Layer 0/2
            中线(12)始终在 Layer 1
        """
        # ---------- 确定符号 ----------
        sign = 0
        yaw_val = yaw if yaw is not None else None
        if yaw_val is not None:
            if yaw_val >= 150.0 or yaw_val <= -150.0:
                # 背对镜头: 所有四肢在底层, 头部在顶层
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
        # 修正: sign>0 = 右侧面朝镜头更多 → 右侧是前侧(top), 左侧是后侧(bottom)
        #       sign<0 = 左侧面朝镜头更多 → 左侧是前侧(top), 右侧是后侧(bottom)
        #       sign=0 = 正对 → 所有四肢在bottom按Y排序, top为空
        if sign > 0:
            return "left", "right"    # 左后右前
        elif sign < 0:
            return "right", "left"    # 右后左前
        else:
            return "both", None       # 正对: 所有在底层, 按Y排序

    def _sort_limbs_by_y(self, indices, keypoints):
        """按骨骼中点Y坐标升序排列（Y小=上方=先画，Y大=下方=后画=覆盖上方）"""
        if not indices:
            return []
        y_mids = []
        for idx in indices:
            limb = DRAW_LIMBS[idx]
            p1 = keypoints[limb[0] - 1]
            p2 = keypoints[limb[1] - 1]
            y_mid = (p1[1] + p2[1]) / 2.0
            y_mids.append(y_mid)
        sorted_pairs = sorted(zip(indices, y_mids), key=lambda x: x[1])
        return [pair[0] for pair in sorted_pairs]

    # ---------- 辅助方法: 绘制单根骨骼 ----------
    def _draw_single_limb(self, canvas, limb_idx, keypoints, scores, threshold, stick_width):
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
        self.fillConvexPoly(canvas, polygon, BODY_COLORS[limb_idx % len(BODY_COLORS)])

    # ---------- 辅助方法: 绘制手部(指定索引范围) ----------
    def _draw_hands_range(self, canvas, keypoints, scores, threshold, hand_start, hand_end, hand_point_size, eps, W, H):
        """绘制指定索引范围的手部骨骼和圆点"""
        if len(keypoints) < hand_end:
            return
        for ie, edge in enumerate(HAND_EDGES):
            idx1, idx2 = hand_start + edge[0], hand_start + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1,y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2,y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1>eps and y1>eps and x2>eps and y2>eps and 0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H:
                r,g,b = colorsys.hsv_to_rgb(ie/len(HAND_EDGES), 1.0, 1.0)
                color = (int(r*255), int(g*255), int(b*255))
                self.line(canvas, (x1,y1), (x2,y2), color, thickness=2)
        for i in range(hand_start, hand_end):
            if scores is not None and i < len(scores) and scores[i] < threshold:
                continue
            x,y = int(keypoints[i][0]), int(keypoints[i][1])
            if x>eps and y>eps and 0<=x<W and 0<=y<H:
                self.circle(canvas, (x,y), hand_point_size, (0,0,255))

    # ---------- 辅助方法: 绘制一组骨骼 ----------
    def _draw_limb_group(self, canvas, limb_indices, keypoints, scores, threshold, stick_width):
        """绘制一组骨骼, 先按Y排序后画"""
        sorted_idx = self._sort_limbs_by_y(limb_indices, keypoints)
        for i in sorted_idx:
            self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width)

    def draw_wholebody_keypoints(self, canvas, keypoints, scores=None, threshold=0.3,
                                 draw_body=True, draw_feet=True, draw_face=True, draw_hands=True,
                                 stick_width=4, face_point_size=3, hand_point_size=4,
                                 yaw=None):
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

        # ======================================================
        # Layer 0 (底层, 先画): 后侧腿 + 后侧手臂 + 后侧脸部骨骼 + 后侧手 + 后侧脚圆点 + 后侧眼
        # 绘制顺序: 腿(先画) → 手臂(后画覆盖腿) → 脸部骨骼 → 手 → 脚 → 眼
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            # 先画腿（底层），再画手臂（覆盖腿）
            self._draw_limb_group(canvas, bottom_legs, keypoints, scores, threshold, stick_width)
            self._draw_limb_group(canvas, bottom_arms, keypoints, scores, threshold, stick_width)
            # 脸部骨骼(后侧)
            for i in bottom_faces:
                self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width)

        # 后侧手
        if draw_hands:
            if bottom_side == "left" and len(keypoints) >= 134:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H)
            elif bottom_side == "right" and len(keypoints) >= 113:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H)
            elif bottom_side == "both":
                if len(keypoints) >= 113:
                    self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H)
                if len(keypoints) >= 134:
                    self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H)

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
                    self.circle(canvas, (x, y), 4, BODY_COLORS[i % len(BODY_COLORS)])

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
            self._draw_single_limb(canvas, 12, keypoints, scores, threshold, stick_width)

        # 面部其余点 (轮廓 0-16、眉毛 17-26、鼻子 27-35、嘴唇 48-67、瞳孔 68-69)
        # 映射到 keypoints: 24 + face_idx
        # 排除眼睛 36-47 (已在 Layer 0/2)
        if draw_face and len(keypoints) >= 92:
            face_rest = list(range(24, 60)) + list(range(72, 92))  # face 0-35(轮廓+眉+鼻) + face 48-67(嘴唇)+68-69(瞳孔)
            # face 0-35 → kp 24-59, face 48-69 → kp 72-93
            for i in face_rest:
                if scores is not None and i < len(scores) and scores[i] < threshold: continue
                x,y = int(keypoints[i][0]), int(keypoints[i][1])
                if x>eps and y>eps and 0<=x<W and 0<=y<H:
                    self.circle(canvas, (x,y), face_point_size, (255,255,255))

        # ======================================================
        # Layer 2 (顶层, 后画): 前侧腿 + 前侧手臂 + 前侧脸部骨骼 + 前侧手 + 前侧脚 + 前侧眼
        # 绘制顺序: 腿(先画) → 手臂(后画覆盖腿) → 脸部骨骼 → 手 → 脚 → 眼
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            # 先画腿（底层），再画手臂（覆盖腿）
            self._draw_limb_group(canvas, top_legs, keypoints, scores, threshold, stick_width)
            self._draw_limb_group(canvas, top_arms, keypoints, scores, threshold, stick_width)
            for i in top_faces:
                self._draw_single_limb(canvas, i, keypoints, scores, threshold, stick_width)

        # 前侧手
        if draw_hands:
            if top_side == "left" and len(keypoints) >= 134:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 113, 134, hand_point_size, eps, W, H)
            elif top_side == "right" and len(keypoints) >= 113:
                self._draw_hands_range(canvas, keypoints, scores, threshold, 92, 113, hand_point_size, eps, W, H)

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
                    self.circle(canvas, (x, y), 4, BODY_COLORS[i % len(BODY_COLORS)])

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
        # Layer 3 (覆盖层): 全体身体关键点圆点 (0-17)
        # ======================================================
        if draw_body and len(keypoints) >= 18:
            for i in range(18):
                if scores is not None and scores[i] < threshold: continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0 <= x < W and 0 <= y < H:
                    self.circle(canvas, (x, y), 4, BODY_COLORS[i % len(BODY_COLORS)])

        return canvas


# ==================== 辅助转换函数 ====================
def get_keypoint_arrays(pose_frame: Dict, person_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    people = pose_frame.get("people", [])
    if not people or person_idx >= len(people):
        return np.zeros((134, 2), dtype=np.float32), np.zeros(134, dtype=np.float32)
    person = people[person_idx]
    def parse_flat(flat, n):
        if not flat:
            return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=np.float32)
        arr = np.array(flat, dtype=np.float32).reshape(n, 3)
        return arr[:, :2], arr[:, 2]
    body_kp, body_sc = parse_flat(person.get("pose_keypoints_2d", []), 18)
    foot_kp, foot_sc = parse_flat(person.get("foot_keypoints_2d", []), 6)
    face_kp, face_sc = parse_flat(person.get("face_keypoints_2d", []), 70)
    rhand_kp, rhand_sc = parse_flat(person.get("hand_right_keypoints_2d", []), 21)
    lhand_kp, lhand_sc = parse_flat(person.get("hand_left_keypoints_2d", []), 21)
    kp = np.concatenate([body_kp, foot_kp, face_kp[:68], rhand_kp, lhand_kp], axis=0)
    sc = np.concatenate([body_sc, foot_sc, face_sc[:68], rhand_sc, lhand_sc], axis=0)
    return kp, sc


# ==================== 面部连线绘制辅助函数 ====================
def extract_face_points_70(frame: Dict, person_idx: int = 0,
                           conf_thresh: float = 0.0, min_coord: float = 0.0,
                           debug: bool = False) -> Tuple[List[Optional[Tuple[float, float]]], int, int]:
    """从帧中提取70个面部关键点，返回点列表(None表示无效)、宽、高"""
    w = frame.get("canvas_width", 512)
    h = frame.get("canvas_height", 512)
    people = frame.get("people", [])
    if not people or person_idx >= len(people):
        if debug: logging.debug("[FaceExtract] No people in frame")
        return [], w, h
    person = people[person_idx]
    face_flat = person.get("face_keypoints_2d", [])
    if len(face_flat) != 210:
        if debug: logging.debug(f"[FaceExtract] Invalid face_keypoints_2d length: {len(face_flat)} (expected 210)")
        return [], w, h
    pts = []
    valid_cnt = 0
    for i in range(0, 210, 3):
        x, y, c = face_flat[i], face_flat[i+1], face_flat[i+2]
        if c >= conf_thresh and 0 <= x <= w and 0 <= y <= h and x > min_coord and y > min_coord:
            pts.append((x, y))
            valid_cnt += 1
        else:
            pts.append(None)
    if debug:
        logging.debug(f"[FaceExtract] Valid points: {valid_cnt}/70 (conf_thresh={conf_thresh}, min_coord={min_coord})")
    return pts, w, h

def draw_face_lines(canvas: np.ndarray, pts: List[Optional[Tuple[float, float]]],
                    line_thickness: int = 2, point_radius: int = 3, pupil_radius: int = 4):
    """在canvas上绘制白色面部连线及点（仅限有效点）"""
    white = (255, 255, 255)

    # 使用OpenCV（若可用）或回退到KeypointDraw
    try:
        import cv2  # type: ignore[import-untyped]
        def cv2_line(img, pt1, pt2, color, thickness):
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness=thickness, lineType=cv2.LINE_AA)
        def cv2_circle(img, center, radius, color, thickness):
            cv2.circle(img, (int(center[0]), int(center[1])), radius, color, thickness=thickness, lineType=cv2.LINE_AA)
    except ImportError:
        _kd = KeypointDraw()
        def cv2_line(img, pt1, pt2, color, thickness):
            _kd.line(img, pt1, pt2, color, thickness)
        def cv2_circle(img, center, radius, color, thickness):
            _kd.circle(img, center, radius, color, thickness)

    def draw_segments(indices, closed=False):
        if len(indices) < 2:
            return
        for i in range(len(indices)-1):
            p1 = pts[indices[i]]
            p2 = pts[indices[i+1]]
            if p1 is not None and p2 is not None:
                cv2_line(canvas, p1, p2, white, line_thickness)
        if closed:
            p1 = pts[indices[0]]
            p2 = pts[indices[-1]]
            if p1 is not None and p2 is not None:
                cv2_line(canvas, p1, p2, white, line_thickness)

    def draw_points(indices, radius):
        for idx in indices:
            p = pts[idx]
            if p is not None:
                cv2_circle(canvas, p, radius, white, -1)

    # 外轮廓
    draw_segments(FACE_70_INDICES["contour"], closed=False)
    # 眉毛
    draw_segments(FACE_70_INDICES["left_eyebrow"], closed=False)
    draw_segments(FACE_70_INDICES["right_eyebrow"], closed=False)
    # 鼻子点
    draw_points(FACE_70_INDICES["nose"], point_radius)
    # 眼眶（闭合）
    draw_segments(FACE_70_INDICES["left_eye"], closed=True)
    draw_segments(FACE_70_INDICES["right_eye"], closed=True)
    # 内唇（闭合）
    draw_segments(FACE_70_INDICES["inner_lip"], closed=True)
    # 瞳孔
    left_p = pts[FACE_70_INDICES["left_pupil"]]
    right_p = pts[FACE_70_INDICES["right_pupil"]]
    if left_p is not None:
        cv2_circle(canvas, left_p, pupil_radius, white, -1)
    if right_p is not None:
        cv2_circle(canvas, right_p, pupil_radius, white, -1)


# ==================== 保存 JSON 节点（含 overwrite 开关） ====================
class SDPoseSaveJson:
    """
    保存 POSE_KEYPOINT 为 JSON 文件。
    - overwrite=True: 每次覆盖同一文件（向后兼容）
    - overwrite=False: 文件名自动递增编号
    可选 fps: 若 >0，保存为 {"fps": fps, "frames": [...]} 格式。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "filename_prefix": ("STRING", {"default": "PoseKeypoint"})
            },
            "optional": {
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1,
                                  "tooltip": "若 >0，将 fps 写入 JSON 头部；=0 则存为裸数组（兼容旧版）"}),
                "overwrite": ("BOOLEAN", {"default": True,
                                          "tooltip": "True=覆盖上次文件（向后兼容）；False=自动递增编号不覆盖"}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_pose_kps"
    OUTPUT_NODE = True
    CATEGORY = "SDPose"
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    def save_pose_kps(self, pose_kps, filename_prefix, fps=0, overwrite=True):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, pose_kps[0]["canvas_width"], pose_kps[0]["canvas_height"])
        if overwrite:
            file = f"{filename}_{counter:05}.json"
        else:
            file = f"{filename}_{counter:05}_.json"
        if fps > 0:
            out_data = {"fps": fps, "frames": pose_kps}
        else:
            out_data = pose_kps
        with open(os.path.join(full_output_folder, file), 'w', encoding='utf-8') as f:
            json.dump(out_data, f)
        return {}


# ==================== 加载 JSON 文件（FPS 增强版） ====================
class SDPoseLoadJson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "target_fps": ("FLOAT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "目标帧率：0=全部读取；>0 时根据 JSON 中的 fps 自动抽帧/补帧"
                }),
                "interp_method": ([
                    "interpolate",
                    "duplicate"
                ], {
                    "default": "interpolate",
                    "tooltip": "仅升帧（补帧）时有效：interpolate=线性插值，duplicate=复制最近帧"
                }),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("pose_keypoints", "frame_count", "canvas_width", "canvas_height", "effective_fps")
    FUNCTION = "load"
    CATEGORY = "SDPose"

    def load(self, json_path, target_fps=0, interp_method="interpolate"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_fps = None
        if isinstance(data, dict):
            json_fps = data.get("fps", None)
            frames = data.get("frames", [])
            if not isinstance(frames, list):
                raise ValueError("JSON wrapper format: 'frames' must be a list")
        elif isinstance(data, list):
            frames = data
            json_fps = None
        else:
            raise ValueError("JSON root must be a list of frames or a dict with 'frames' key")

        for frame in frames:
            if "canvas_width" not in frame or "canvas_height" not in frame:
                raise ValueError("Each frame must contain canvas_width and canvas_height")
            if "people" not in frame:
                frame["people"] = []

        original_count = len(frames)
        canvas_w = frames[0]["canvas_width"] if frames else 0
        canvas_h = frames[0]["canvas_height"] if frames else 0

        if json_fps is not None and json_fps > 0 and target_fps > 0 and original_count > 1:
            ratio = json_fps / target_fps
            if abs(ratio - 1.0) < 1e-6:
                result = frames
            elif ratio > 1.0:
                result = self._downsample(frames, ratio, json_fps)
            else:
                if interp_method == "interpolate":
                    result = self._upsample_interpolate(frames, ratio, json_fps)
                else:
                    result = self._upsample_duplicate(frames, ratio, json_fps)
        else:
            result = frames

        frame_count = len(result)
        # 计算 effective_fps
        if target_fps == 0:
            effective_fps = float(json_fps) if json_fps else 0.0
        else:
            if json_fps is not None and json_fps > 0:
                effective_fps = float(target_fps)
            else:
                effective_fps = 0.0

        return (result, frame_count, canvas_w, canvas_h, effective_fps)

    @staticmethod
    def _downsample(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n >= n:
            return frames[:]
        if target_n <= 0:
            return [frames[0]]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        indices = np.array([np.argmin(np.abs(orig_times - t)) for t in target_times])
        return [frames[i] for i in indices]

    @staticmethod
    def _upsample_duplicate(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n <= n:
            return frames[:]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        out = []
        for t in target_times:
            nearest = int(np.argmin(np.abs(orig_times - t)))
            nearest = max(0, min(n - 1, nearest))
            out.append(frames[nearest])
        return out

    @staticmethod
    def _is_abrupt_jump(frame_a, frame_b, threshold_ratio=0.1):
        w = frame_a.get("canvas_width", 1)
        h = frame_a.get("canvas_height", 1)
        diag = math.hypot(w, h)
        threshold = diag * threshold_ratio
        people_a = frame_a.get("people", [])
        people_b = frame_b.get("people", [])
        if not people_a or not people_b:
            return False
        total_disp = 0.0
        count = 0
        for pa, pb in zip(people_a, people_b):
            for key in pa:
                if key.endswith("_keypoints_2d"):
                    va = pa.get(key, [])
                    vb = pb.get(key, [])
                    max_len = min(len(va), len(vb))
                    for i in range(0, max_len, 3):
                        dx = va[i] - vb[i]
                        dy = va[i+1] - vb[i+1]
                        total_disp += math.hypot(dx, dy)
                        count += 1
        if count == 0:
            return False
        avg_disp = total_disp / count
        return avg_disp > threshold

    @staticmethod
    def _upsample_interpolate(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n <= n:
            return frames[:]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        out = []
        for t in target_times:
            if t <= orig_times[0]:
                out.append(frames[0])
                continue
            if t >= orig_times[-1]:
                out.append(frames[-1])
                continue
            right_idx = np.searchsorted(orig_times, t)
            left_idx = right_idx - 1
            alpha = (t - orig_times[left_idx]) / (orig_times[right_idx] - orig_times[left_idx])
            left_frame = frames[left_idx]
            right_frame = frames[right_idx]
            if SDPoseLoadJson._is_abrupt_jump(left_frame, right_frame):
                out.append(right_frame if alpha > 0.5 else left_frame)
                continue
            interp_frame = {
                "canvas_width": left_frame["canvas_width"],
                "canvas_height": left_frame["canvas_height"],
                "people": []
            }
            max_people = max(
                len(left_frame.get("people", [])),
                len(right_frame.get("people", []))
            )
            for p_idx in range(max_people):
                left_person = left_frame["people"][p_idx] if p_idx < len(left_frame["people"]) else {}
                right_person = right_frame["people"][p_idx] if p_idx < len(right_frame["people"]) else {}
                new_person = {}
                all_keys = set(list(left_person.keys()) + list(right_person.keys()))
                for key in all_keys:
                    left_val = left_person.get(key, [])
                    right_val = right_person.get(key, [])
                    if key.endswith("_keypoints_2d") and isinstance(left_val, list) and isinstance(right_val, list):
                        max_len = max(len(left_val), len(right_val))
                        l_arr = np.array(left_val + [0.0] * (max_len - len(left_val)), dtype=np.float32)
                        r_arr = np.array(right_val + [0.0] * (max_len - len(right_val)), dtype=np.float32)
                        interpolated = l_arr * (1.0 - alpha) + r_arr * alpha
                        new_person[key] = interpolated.tolist()
                    else:
                        new_person[key] = left_val if left_val else right_val
                interp_frame["people"].append(new_person)
            out.append(interp_frame)
        return out

# ==================== 重采样（抽帧/补帧） ====================
class SDPoseResampleKeypoints:
    """
    对 POSE_KEYPOINT 数据进行重采样（抽帧/补帧）。
    接收输入 fps 和输出 fps，独立于 JSON 载入逻辑。
    当 input_fps 或 output_fps 为 0 时透传，fps 输出 0（表示未知）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "input_fps": ("FLOAT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "输入关键点的原始帧率：0=未知（不处理）；>0 时根据 ratio = input_fps / output_fps 进行抽帧/补帧"
                }),
                "output_fps": ("FLOAT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "目标输出帧率：0=不处理；>0 时对输入进行重采样"
                }),
                "interp_method": ([
                    "interpolate",
                    "duplicate"
                ], {
                    "default": "interpolate",
                    "tooltip": "仅升帧（补帧）时有效：interpolate=线性插值，duplicate=复制最近帧"
                }),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "FLOAT")
    RETURN_NAMES = ("pose_keypoints", "fps")
    FUNCTION = "resample"
    CATEGORY = "SDPose"

    def resample(self, pose_keypoints, input_fps=0, output_fps=0, interp_method="interpolate"):
        if not isinstance(pose_keypoints, list) or len(pose_keypoints) == 0:
            return (pose_keypoints, 0.0)

        if input_fps <= 0 or output_fps <= 0 or len(pose_keypoints) <= 1:
            return (pose_keypoints, 0.0)

        ratio = input_fps / output_fps
        if abs(ratio - 1.0) < 1e-6:
            result = pose_keypoints[:]
        elif ratio > 1.0:
            result = self._downsample(pose_keypoints, ratio, input_fps)
        else:
            if interp_method == "interpolate":
                result = self._upsample_interpolate(pose_keypoints, ratio, input_fps)
            else:
                result = self._upsample_duplicate(pose_keypoints, ratio, input_fps)

        return (result, float(output_fps))

    @staticmethod
    def _downsample(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n >= n:
            return frames[:]
        if target_n <= 0:
            return [frames[0]]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        indices = np.array([np.argmin(np.abs(orig_times - t)) for t in target_times])
        return [frames[i] for i in indices]

    @staticmethod
    def _upsample_duplicate(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n <= n:
            return frames[:]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        out = []
        for t in target_times:
            nearest = int(np.argmin(np.abs(orig_times - t)))
            nearest = max(0, min(n - 1, nearest))
            out.append(frames[nearest])
        return out

    @staticmethod
    def _is_abrupt_jump(frame_a, frame_b, threshold_ratio=0.1):
        w = frame_a.get("canvas_width", 1)
        h = frame_a.get("canvas_height", 1)
        diag = math.hypot(w, h)
        threshold = diag * threshold_ratio
        people_a = frame_a.get("people", [])
        people_b = frame_b.get("people", [])
        if not people_a or not people_b:
            return False
        total_disp = 0.0
        count = 0
        for pa, pb in zip(people_a, people_b):
            for key in pa:
                if key.endswith("_keypoints_2d"):
                    va = pa.get(key, [])
                    vb = pb.get(key, [])
                    max_len = min(len(va), len(vb))
                    for i in range(0, max_len, 3):
                        dx = va[i] - vb[i]
                        dy = va[i+1] - vb[i+1]
                        total_disp += math.hypot(dx, dy)
                        count += 1
        if count == 0:
            return False
        avg_disp = total_disp / count
        return avg_disp > threshold

    @staticmethod
    def _upsample_interpolate(frames, ratio, json_fps):
        n = len(frames)
        target_n = int(round(n / ratio))
        if target_n <= n:
            return frames[:]
        duration = (n - 1) / json_fps
        orig_times = np.arange(n) / json_fps
        target_times = np.linspace(0, duration, target_n)
        out = []
        for t in target_times:
            if t <= orig_times[0]:
                out.append(frames[0])
                continue
            if t >= orig_times[-1]:
                out.append(frames[-1])
                continue
            right_idx = np.searchsorted(orig_times, t)
            left_idx = right_idx - 1
            alpha = (t - orig_times[left_idx]) / (orig_times[right_idx] - orig_times[left_idx])
            left_frame = frames[left_idx]
            right_frame = frames[right_idx]
            if SDPoseResampleKeypoints._is_abrupt_jump(left_frame, right_frame):
                out.append(right_frame if alpha > 0.5 else left_frame)
                continue
            interp_frame = {
                "canvas_width": left_frame["canvas_width"],
                "canvas_height": left_frame["canvas_height"],
                "people": []
            }
            max_people = max(
                len(left_frame.get("people", [])),
                len(right_frame.get("people", []))
            )
            for p_idx in range(max_people):
                left_person = left_frame["people"][p_idx] if p_idx < len(left_frame["people"]) else {}
                right_person = right_frame["people"][p_idx] if p_idx < len(right_frame["people"]) else {}
                new_person = {}
                all_keys = set(list(left_person.keys()) + list(right_person.keys()))
                for key in all_keys:
                    left_val = left_person.get(key, [])
                    right_val = right_person.get(key, [])
                    if key.endswith("_keypoints_2d") and isinstance(left_val, list) and isinstance(right_val, list):
                        max_len = max(len(left_val), len(right_val))
                        l_arr = np.array(left_val + [0.0] * (max_len - len(left_val)), dtype=np.float32)
                        r_arr = np.array(right_val + [0.0] * (max_len - len(right_val)), dtype=np.float32)
                        interpolated = l_arr * (1.0 - alpha) + r_arr * alpha
                        new_person[key] = interpolated.tolist()
                    else:
                        new_person[key] = left_val if left_val else right_val
                interp_frame["people"].append(new_person)
            out.append(interp_frame)
        return out

# ==================== 切片 ====================
class SDPoseSliceKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 20000, "step": 1}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 20000, "step": 1}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "INT")
    RETURN_NAMES = ("sliced_pose", "input_frame_count")
    FUNCTION = "slice"
    CATEGORY = "SDPose"

    def slice(self, pose_keypoints, start_frame, frame_count):
        if not isinstance(pose_keypoints, list):
            raise ValueError("pose_keypoints must be a list of frames")
        total = len(pose_keypoints)
        if start_frame >= total:
            return ([], total)
        end_frame = min(start_frame + frame_count, total)
        sliced = pose_keypoints[start_frame:end_frame]
        return (sliced, total)

# ==================== 拼接 ====================
class SDPoseConcatKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pose_a": ("POSE_KEYPOINT",),
                "pose_b": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("concatenated_pose",)
    FUNCTION = "concat"
    CATEGORY = "SDPose"

    def concat(self, pose_a=None, pose_b=None):
        if pose_a is None:
            pose_a = []
        if pose_b is None:
            pose_b = []
        if not isinstance(pose_a, list) or not isinstance(pose_b, list):
            raise ValueError("Both inputs must be lists of frames (or None)")
        if not pose_a and not pose_b:
            return ([],)
        if not pose_a:
            return (pose_b,)
        if not pose_b:
            return (pose_a,)
        return (pose_a + pose_b,)

# ==================== 绘制节点（整合面部连线模式） ====================
class SDPoseDrawKeypointsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "draw_body": ("BOOLEAN", {"default": True}),
                "draw_hands": ("BOOLEAN", {"default": True}),
                "draw_face": ("BOOLEAN", {"default": True}),
                "draw_feet": ("BOOLEAN", {"default": False}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "face_point_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "hand_point_size": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_frames": ("INT", {"default": 2500, "min": 1, "max": 5000, "step": 1}),
                "mouth_mode": (["draw_all", "no_draw", "inner_lip_only"], {"default": "draw_all"}),
                "enable_yaw_thickness": ("BOOLEAN", {"default": False, "tooltip": "开启后根据偏航角动态调整骨骼粗细：正面(0°/±180°)最粗，侧面(±90°)最细"}),
                "yaw_thickness_min": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1, "tooltip": "侧面(±90°)时的最细粗细"}),
                "yaw_thickness_max": ("INT", {"default": 8, "min": 1, "max": 20, "step": 1, "tooltip": "正面(0°/±180°)时的最粗粗细"}),
            },
            "optional": {
                "yaw_angles": ("FLOAT", { "tooltip": "来自 SDPoseEstimateYawSimple/Advanced 的 yaw_array，用于动态调整骨骼粗细和遮挡顺序" }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "SDPose"

    def draw(self, keypoints, draw_body, draw_hands, draw_face, draw_feet,
             stick_width, face_point_size, hand_point_size, score_threshold, max_frames,
             mouth_mode="draw_all",
             enable_yaw_thickness=False,
             yaw_thickness_min=1,
             yaw_thickness_max=8,
             yaw_angles=None):
        if not keypoints:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        original_len = len(keypoints)
        if original_len > max_frames:
            logging.warning(f"Input has {original_len} frames, exceeds max_frames={max_frames}. "
                            f"Only the first {max_frames} frames will be drawn.")
            keypoints = keypoints[:max_frames]

        height = keypoints[0]["canvas_height"]
        width = keypoints[0]["canvas_width"]
        drawer = KeypointDraw()
        pose_outputs = []

        for idx, frame in enumerate(tqdm(keypoints, desc="Drawing keypoints")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            cur_yaw = None
            if yaw_angles is not None:
                if isinstance(yaw_angles, (torch.Tensor, np.ndarray)):
                    cur_yaw = float(yaw_angles[idx]) if idx < len(yaw_angles) else float(yaw_angles[-1])
                elif isinstance(yaw_angles, (list, tuple)):
                    cur_yaw = float(yaw_angles[idx]) if idx < len(yaw_angles) else float(yaw_angles[-1])
                else:
                    cur_yaw = float(yaw_angles)

            # 根据偏航角动态调整骨骼粗细
            cur_stick_width = stick_width
            if enable_yaw_thickness and isinstance(cur_yaw, (int, float)):
                yaw_val = float(cur_yaw)
                ratio = abs(math.cos(math.radians(yaw_val)))
                cur_stick_width = int(yaw_thickness_min + (yaw_thickness_max - yaw_thickness_min) * ratio)
                cur_stick_width = max(yaw_thickness_min, min(yaw_thickness_max, cur_stick_width))
            
            for person_idx in range(len(frame.get("people", []))):
                kp, sc = get_keypoint_arrays(frame, person_idx)
                
                if draw_face and mouth_mode != "draw_all":
                    kp = kp.copy()
                    sc = sc.copy()
                    if mouth_mode == "no_draw":
                        hide_indices = list(range(72, 92))
                    elif mouth_mode == "inner_lip_only":
                        hide_indices = list(range(72, 84))
                    else:
                        hide_indices = []
                    for i in hide_indices:
                        if i < len(kp):
                            kp[i] = [-1.0, -1.0]
                            sc[i] = 0.0

                canvas = drawer.draw_wholebody_keypoints(
                    canvas, kp, sc,
                    threshold=score_threshold,
                    draw_body=draw_body,
                    draw_feet=draw_feet,
                    draw_face=draw_face,
                    draw_hands=draw_hands,
                    stick_width=cur_stick_width,
                    face_point_size=face_point_size,
                    hand_point_size=hand_point_size,
                    yaw=cur_yaw
                )

            pose_outputs.append(canvas)

        pose_outputs_np = np.stack(pose_outputs) if len(pose_outputs) > 1 else np.expand_dims(pose_outputs[0], 0)
        result = torch.from_numpy(pose_outputs_np).float() / 255.0
        return (result,)


# ==================== 人脸五官比例对齐（稳健版） ====================
class SDPoseAlignFaceScale:
    """
    对齐人脸五官比例：先整体缩放面部（可选），再独立调整各五官区域尺度。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "ref_keypoints": ("POSE_KEYPOINT",),
                "global_scale": ("BOOLEAN", {"default": True, "tooltip": "先整体对齐面部尺寸（以鼻尖为中心）"}),
                "align_eyes": ("BOOLEAN", {"default": True}),
                "align_nose": ("BOOLEAN", {"default": True}),
                "align_mouth": ("BOOLEAN", {"default": True}),
                "align_contour": ("BOOLEAN", {"default": False, "tooltip": "通常不建议缩放轮廓，以免破坏脸型"}),
                "blend_smooth": ("BOOLEAN", {"default": True, "tooltip": "在区域边界平滑权重，避免接缝"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("aligned_pose",)
    FUNCTION = "align_face_scale"
    CATEGORY = "SDPose"

    PART_INDICES = {
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
        "contour": list(range(0, 17)),
    }

    GLOBAL_SCALE_PAIR = (36, 45)

    SCALE_REF_PAIRS = {
        "left_eye": (36, 39),
        "right_eye": (42, 45),
        "nose": (31, 35),
        "mouth": (48, 54),
        "contour": (0, 16),
    }

    SCALE_CENTER_IDX = {
        "left_eye": None,
        "right_eye": None,
        "nose": 30,
        "mouth": 66,
        "contour": 8,
    }

    @staticmethod
    def get_face_points(pose_frame: Dict, person_idx: int = 0):
        w = pose_frame.get("canvas_width", 512)
        h = pose_frame.get("canvas_height", 512)
        people = pose_frame.get("people", [])
        if not people or person_idx >= len(people):
            return [], w, h
        person = people[person_idx]
        face_flat = person.get("face_keypoints_2d", [])
        if len(face_flat) != 210:
            return [], w, h
        pts = []
        for i in range(0, 210, 3):
            x, y, c = face_flat[i], face_flat[i+1], face_flat[i+2]
            if c > 0.0 and 0 <= x <= w and 0 <= y <= h:
                pts.append((x, y))
            else:
                pts.append(None)
        return pts, w, h

    @classmethod
    def compute_scale(cls, pts, pair):
        idx1, idx2 = pair
        p1 = pts[idx1] if len(pts) > idx1 else None
        p2 = pts[idx2] if len(pts) > idx2 else None
        if p1 is None or p2 is None:
            return 0.0
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

    @classmethod
    def compute_part_center(cls, pts, part_name):
        center_idx = cls.SCALE_CENTER_IDX.get(part_name)
        if center_idx is not None and pts[center_idx] is not None:
            return pts[center_idx]
        indices = cls.PART_INDICES.get(part_name, [])
        valid = [pts[i] for i in indices if pts[i] is not None]
        if not valid:
            return None
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        return (cx, cy)

    @classmethod
    def get_blend_weights(cls, all_pts, part_name, blend_smooth):
        weights = [0.0] * 70
        indices = cls.PART_INDICES.get(part_name, [])
        if not blend_smooth:
            for i in indices:
                weights[i] = 1.0
            return weights
        center = cls.compute_part_center(all_pts, part_name)
        if center is None:
            for i in indices:
                weights[i] = 1.0
            return weights
        max_dist = 0.0
        for i in indices:
            p = all_pts[i]
            if p is not None:
                d = math.hypot(p[0]-center[0], p[1]-center[1])
                if d > max_dist:
                    max_dist = d
        if max_dist == 0.0:
            max_dist = 10.0
        sigma = max_dist * 0.8
        for i in range(70):
            p = all_pts[i]
            if p is None:
                continue
            d = math.hypot(p[0]-center[0], p[1]-center[1])
            weights[i] = math.exp(- (d**2) / (2 * sigma**2))
        return weights

    def align_face_scale(self, pose_keypoints, ref_keypoints,
                         global_scale, align_eyes, align_nose, align_mouth, align_contour, blend_smooth):
        if not isinstance(pose_keypoints, list) or not pose_keypoints:
            return (pose_keypoints,)
        if not isinstance(ref_keypoints, list) or not ref_keypoints:
            return (pose_keypoints,)
        ref_pts, _, _ = self.get_face_points(ref_keypoints[0])
        if len(ref_pts) != 70:
            logging.warning("[AlignFace] Reference face invalid.")
            return (pose_keypoints,)
        ref_global_scale = self.compute_scale(ref_pts, self.GLOBAL_SCALE_PAIR) if global_scale else 1.0
        if global_scale and ref_global_scale <= 0.0:
            logging.warning("[AlignFace] Reference global scale zero, disable global scaling.")
            global_scale = False
        parts_to_align = []
        if align_eyes:
            parts_to_align.extend(["left_eye", "right_eye"])
        if align_nose:
            parts_to_align.append("nose")
        if align_mouth:
            parts_to_align.append("mouth")
        if align_contour:
            parts_to_align.append("contour")
        ref_part_scales = {}
        for part in parts_to_align:
            sc = self.compute_scale(ref_pts, self.SCALE_REF_PAIRS[part])
            if sc > 0.0:
                ref_part_scales[part] = sc
        result_frames = []
        for frame in pose_keypoints:
            new_frame = {
                "canvas_width": frame["canvas_width"],
                "canvas_height": frame["canvas_height"],
                "people": []
            }
            for person in frame.get("people", []):
                new_person = person.copy()
                face_flat = person.get("face_keypoints_2d", [])
                if len(face_flat) != 210:
                    new_frame["people"].append(new_person)
                    continue
                cur_pts, w, h = self.get_face_points(frame, person_idx=0)
                if len(cur_pts) != 70:
                    new_frame["people"].append(new_person)
                    continue
                if global_scale:
                    cur_global_scale = self.compute_scale(cur_pts, self.GLOBAL_SCALE_PAIR)
                    if cur_global_scale > 0.0:
                        global_ratio = ref_global_scale / cur_global_scale
                        nose_center = self.compute_part_center(cur_pts, "nose")
                        if nose_center is None:
                            valid = [p for p in cur_pts if p is not None]
                            if valid:
                                nose_center = (sum(p[0] for p in valid)/len(valid),
                                               sum(p[1] for p in valid)/len(valid))
                        if nose_center is not None:
                            new_cur_pts = []
                            for p in cur_pts:
                                if p is None:
                                    new_cur_pts.append(None)
                                else:
                                    nx = nose_center[0] + (p[0] - nose_center[0]) * global_ratio
                                    ny = nose_center[1] + (p[1] - nose_center[1]) * global_ratio
                                    new_cur_pts.append((nx, ny))
                            cur_pts = new_cur_pts
                accum_dx = [0.0] * 70
                accum_dy = [0.0] * 70
                weight_sum = [0.0] * 70
                for part in parts_to_align:
                    ref_scale = ref_part_scales.get(part, 0.0)
                    if ref_scale <= 0.0:
                        continue
                    cur_scale = self.compute_scale(cur_pts, self.SCALE_REF_PAIRS[part])
                    if cur_scale <= 0.0:
                        continue
                    scale_ratio = ref_scale / cur_scale
                    center = self.compute_part_center(cur_pts, part)
                    if center is None:
                        continue
                    weights = self.get_blend_weights(cur_pts, part, blend_smooth)
                    for i in range(70):
                        if cur_pts[i] is None:
                            continue
                        wgt = weights[i]
                        if wgt <= 0.0:
                            continue
                        nx = center[0] + (cur_pts[i][0] - center[0]) * scale_ratio
                        ny = center[1] + (cur_pts[i][1] - center[1]) * scale_ratio
                        accum_dx[i] += (nx - cur_pts[i][0]) * wgt
                        accum_dy[i] += (ny - cur_pts[i][1]) * wgt
                        weight_sum[i] += wgt
                new_face_flat = []
                for i in range(0, 210, 3):
                    x, y, c = face_flat[i], face_flat[i+1], face_flat[i+2]
                    idx = i // 3
                    if weight_sum[idx] > 0.0:
                        dx = accum_dx[idx] / weight_sum[idx]
                        dy = accum_dy[idx] / weight_sum[idx]
                        nx = x + dx
                        ny = y + dy
                        nx = max(0.0, min(w, nx))
                        ny = max(0.0, min(h, ny))
                    else:
                        nx, ny = x, y
                    new_face_flat.extend([float(nx), float(ny), float(c)])
                new_person["face_keypoints_2d"] = new_face_flat
                new_frame["people"].append(new_person)
            result_frames.append(new_frame)
        return (result_frames,)


# ==================== 偏航角计算核心函数（共用） ====================
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
    return [(a + 180) % 360 - 180 if a is not None else None for a in angles]

def _compute_yaw_core(pose_keypoints_segment, conf_threshold, unwrap_angle,
                      nose_neck_thresh, smooth_window, enable_side_calib,
                      shoulder_weight, head_body_diff_threshold, confirmation_frames,
                      ema_alpha, min_angle_limit, max_angle_limit,
                      pose_keypoints_full=None,merge_threshold=30.0):
    IDX_NOSE = 0
    IDX_NECK = 1
    IDX_R_SHOULDER = 2
    IDX_L_SHOULDER = 5
    IDX_R_HIP = 8
    IDX_L_HIP = 11

    anchor_seq = pose_keypoints_full if pose_keypoints_full else pose_keypoints_segment
    if not isinstance(anchor_seq, list) or not anchor_seq:
        return ("Error: No valid keypoints for anchor", torch.zeros(len(pose_keypoints_segment)))

    neck_hip_dists = []
    front_shoulder_ratios = []
    front_hip_ratios = []
    side_candidates = []

    for frame in anchor_seq:
        for person in frame.get("people", []):
            flat = _get_flat_keypoints(person)
            neck = _get_point(flat, IDX_NECK)
            r_hip = _get_point(flat, IDX_R_HIP)
            l_hip = _get_point(flat, IDX_L_HIP)
            r_shoulder = _get_point(flat, IDX_R_SHOULDER)
            l_shoulder = _get_point(flat, IDX_L_SHOULDER)
            nose = _get_point(flat, IDX_NOSE)

            if not (_is_valid(neck, conf_threshold) and
                    _is_valid(r_hip, conf_threshold) and
                    _is_valid(l_hip, conf_threshold)):
                continue
            hip_cx = (r_hip[0] + l_hip[0]) / 2.0
            hip_cy = (r_hip[1] + l_hip[1]) / 2.0
            cur_dist = math.hypot(neck[0] - hip_cx, neck[1] - hip_cy)
            if cur_dist > 1e-6:
                neck_hip_dists.append(cur_dist)

            shoulder_valid = _is_valid(r_shoulder, conf_threshold) and _is_valid(l_shoulder, conf_threshold)
            hip_valid = _is_valid(r_hip, conf_threshold) and _is_valid(l_hip, conf_threshold)
            nose_neck_x = nose[0] - neck[0] if (_is_valid(nose, conf_threshold) and _is_valid(neck, conf_threshold)) else None

            shldr_raw = r_shoulder[0] - l_shoulder[0] if shoulder_valid else None
            hip_raw = r_hip[0] - l_hip[0] if hip_valid else None
            side_candidates.append((shldr_raw, hip_raw, cur_dist, nose_neck_x))

            if nose_neck_x is not None and abs(nose_neck_x) < 3.0 and cur_dist > 1e-6:
                if shoulder_valid and shldr_raw < 0:
                    front_shoulder_ratios.append(abs(shldr_raw) / cur_dist)
                if hip_valid:
                    hip_raw_abs = abs(hip_raw) if hip_raw is not None else 0.0
                    if hip_raw_abs > 0:
                        front_hip_ratios.append(hip_raw_abs / cur_dist)

    if not neck_hip_dists:
        return ("Error: No valid neck-hip distance found", torch.zeros(len(pose_keypoints_segment)))

    anchor_neck_hip = np.percentile(neck_hip_dists, 99)

    def compute_ref_width(front_ratios, candidates_key_idx, default_width=1.0):
        if len(front_ratios) >= 3:
            body_ratio = np.median(front_ratios)
            return body_ratio * anchor_neck_hip
        else:
            all_corrected = []
            for cand in side_candidates:
                raw_val = cand[candidates_key_idx]
                cur_dist = cand[2]
                if raw_val is not None and cur_dist > 1e-6:
                    scale = anchor_neck_hip / cur_dist
                    corr_abs = abs(raw_val * scale)
                    if corr_abs < 400:
                        all_corrected.append(corr_abs)
            return np.percentile(all_corrected, 90) if all_corrected else default_width

    ref_shoulder_width = compute_ref_width(front_shoulder_ratios, 0, 1.0)
    ref_hip_width = compute_ref_width(front_hip_ratios, 1, 1.0)

    if enable_side_calib and len(side_candidates) > 5:
        side_angles = []
        for shldr_raw, _, cur_dist, nose_neck_x in side_candidates:
            if shldr_raw is None or cur_dist < 1e-6:
                continue
            scale = anchor_neck_hip / cur_dist
            corr_shldr = shldr_raw * scale
            corr_abs = abs(corr_shldr)
            if corr_abs < 15.0 and nose_neck_x is not None:
                ratio = corr_abs / ref_shoulder_width
                ratio = max(0.0, min(1.0, ratio))
                ang = math.degrees(math.acos(ratio))
                side_angles.append(ang)
        if side_angles:
            median_side_deg = np.median(side_angles)
            if abs(median_side_deg - 90.0) > 10.0:
                adjust_factor = 1.0 + 0.05 * np.sign(90.0 - median_side_deg)
                ref_shoulder_width *= adjust_factor

    frames_data = []
    raw_yaws = []

    prev_shldr_corr = None
    prev_hip_corr = None
    prev_nose_neck_x = None
    ema_delta_shldr = 0.0
    ema_delta_hip = 0.0
    ema_delta_nose = 0.0

    last_valid_body_sign = 1
    head_only_counter = 0
    sign_locked = False

    hip_weight = 1.0 - shoulder_weight
    total_weight = shoulder_weight + hip_weight
    if total_weight <= 0:
        w_shoulder_norm = 0.7
        w_hip_norm = 0.3
    else:
        w_shoulder_norm = shoulder_weight / total_weight
        w_hip_norm = hip_weight / total_weight

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

            def valid(p): return _is_valid(p, conf_threshold)

            shldr_x = r_shoulder[0] - l_shoulder[0] if (valid(r_shoulder) and valid(l_shoulder)) else None
            hip_x = r_hip[0] - l_hip[0] if (valid(r_hip) and valid(l_hip)) else None
            nose_neck_x = nose[0] - neck[0] if (valid(nose) and valid(neck)) else None

            cur_neck_hip: Optional[float] = None
            scale = 1.0
            if valid(neck) and valid(r_hip) and valid(l_hip):
                hip_cx = (r_hip[0] + l_hip[0]) / 2.0
                hip_cy = (r_hip[1] + l_hip[1]) / 2.0
                neck_hip_val = math.hypot(neck[0] - hip_cx, neck[1] - hip_cy)
                cur_neck_hip = neck_hip_val
                if neck_hip_val > 1e-6:
                    scale = anchor_neck_hip / neck_hip_val

            shldr_corr = shldr_x * scale if shldr_x is not None else None
            hip_corr = hip_x * scale if hip_x is not None else None

            delta_shldr = 0.0
            delta_hip = 0.0
            delta_nose = 0.0
            if prev_shldr_corr is not None and shldr_corr is not None and ref_shoulder_width > 0:
                delta_shldr = abs(shldr_corr - prev_shldr_corr) / ref_shoulder_width
            if prev_hip_corr is not None and hip_corr is not None and ref_hip_width > 0:
                delta_hip = abs(hip_corr - prev_hip_corr) / ref_hip_width
            if prev_nose_neck_x is not None and nose_neck_x is not None:
                norm_factor = anchor_neck_hip * 0.1
                if norm_factor > 0:
                    delta_nose = abs(nose_neck_x - prev_nose_neck_x) / norm_factor

            if ema_alpha > 0:
                ema_delta_shldr = ema_alpha * delta_shldr + (1 - ema_alpha) * ema_delta_shldr
                ema_delta_hip   = ema_alpha * delta_hip   + (1 - ema_alpha) * ema_delta_hip
                ema_delta_nose  = ema_alpha * delta_nose  + (1 - ema_alpha) * ema_delta_nose
            else:
                ema_delta_shldr, ema_delta_hip, ema_delta_nose = delta_shldr, delta_hip, delta_nose

            diff = abs(ema_delta_nose - ema_delta_shldr)
            is_head_independent = diff > head_body_diff_threshold

            if is_head_independent:
                head_only_counter += 1
            else:
                head_only_counter = 0

            if head_only_counter >= confirmation_frames:
                sign_locked = True
            else:
                sign_locked = False

            candidate_sign = None
            if nose_neck_x is not None and abs(nose_neck_x) >= nose_neck_thresh:
                candidate_sign = 1 if nose_neck_x > 0 else -1

            if not sign_locked and candidate_sign is not None:
                last_valid_body_sign = candidate_sign

            def angle_from_diff(diff_corr, ref_width):
                if diff_corr is None or ref_width <= 0:
                    return None
                ratio = abs(diff_corr) / ref_width
                ratio = max(0.0, min(1.0, ratio))
                return math.degrees(math.acos(ratio))

            ang_shldr = angle_from_diff(shldr_corr, ref_shoulder_width)
            ang_hip = angle_from_diff(hip_corr, ref_hip_width)

            valid_angs = []
            weights = []
            if ang_shldr is not None:
                valid_angs.append(ang_shldr)
                weights.append(w_shoulder_norm)
            if ang_hip is not None:
                valid_angs.append(ang_hip)
                weights.append(w_hip_norm)

            if valid_angs:
                w_sum = sum(weights)
                weights = [w / w_sum for w in weights]
                body_base_deg = sum(a * w for a, w in zip(valid_angs, weights))
            else:
                body_base_deg = None

            raw_yaw = None
            body_abs = body_base_deg if body_base_deg is not None else 0.0
            if body_base_deg is not None:
                if body_abs < 10.0 and nose_neck_x is not None and abs(nose_neck_x) < nose_neck_thresh:
                    raw_yaw = 0.0
                    last_valid_body_sign = 1 if nose_neck_x > 0 else -1
                else:
                    sign = last_valid_body_sign
                    if shldr_corr is not None:
                        if shldr_corr < 0:
                            raw_yaw = sign * body_abs
                        else:
                            raw_yaw = sign * (180.0 - body_abs)
                    else:
                        raw_yaw = sign * body_abs
                    raw_yaw = (raw_yaw + 180) % 360 - 180

            body_change_rate = w_shoulder_norm * ema_delta_shldr + w_hip_norm * ema_delta_hip
            dynamic_limit = min_angle_limit + (max_angle_limit - min_angle_limit) * min(1.0, body_change_rate / 0.1)

            frame_person_yaws.append(raw_yaw)
            frame_rows.append((frame_idx, p_idx, shldr_x, hip_x, nose_neck_x,
                               cur_neck_hip, shldr_corr, hip_corr,
                               ema_delta_shldr, ema_delta_nose, diff,
                               is_head_independent, sign_locked,
                               body_abs, last_valid_body_sign, raw_yaw,
                               body_change_rate, dynamic_limit))

            prev_shldr_corr = shldr_corr
            prev_hip_corr = hip_corr
            prev_nose_neck_x = nose_neck_x

        raw_yaws.extend(frame_person_yaws)
        frames_data.append(frame_rows)

    num_frames = len(pose_keypoints_segment)
    raw_seq = raw_yaws
    cum_raw = _unwrap_sequence(raw_seq)

    dyn_limits = [max_angle_limit] * num_frames
    for f_idx in range(num_frames):
        if f_idx < len(frames_data) and frames_data[f_idx]:
            dyn_limits[f_idx] = frames_data[f_idx][0][-1]

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
        diff_abs = abs(cur - prev)
        limit = dyn_limits[f_idx]
        if diff_abs > limit:
            left_idx = f_idx - 1
            right_idx = f_idx + 1
            while right_idx < num_frames and cum_interp[right_idx] is None:
                right_idx += 1
            if left_idx >= 0 and right_idx < num_frames:
                left_val = cum_filtered[left_idx]
                right_val = cum_interp[right_idx]
                if left_val is not None and right_val is not None:
                    diff_lr = right_val - left_val
                    interpolated = left_val + diff_lr * (f_idx - left_idx) / (right_idx - left_idx)
                    cum_filtered[f_idx] = interpolated
                else:
                    cum_filtered[f_idx] = prev
            else:
                cum_filtered[f_idx] = prev
        else:
            cum_filtered[f_idx] = cur

    if unwrap_angle:
        final_seq = cum_filtered
    else:
        final_seq = _wrap_to_180(cum_filtered)

    yaw_list = [a if a is not None else 0.0 for a in final_seq]
    yaw_tensor = torch.tensor(yaw_list, dtype=torch.float32)

    filt_yaws_wrapped = _wrap_to_180(cum_filtered)

    start_angle = None
    range_str = ""

    folded_valid = [a for a in raw_seq if a is not None]
    if folded_valid:
        start_angle = folded_valid[0]
        pos_angles = sorted(set(a for a in folded_valid if a >= 0))
        neg_angles = sorted(set(a for a in folded_valid if a < 0), reverse=True)
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
        neg_angles_asc = sorted(set(a for a in folded_valid if a < 0))
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
    lines.append("")

    header = (f"{'Frame':>5} {'Pers':>4} | "
              f"{'ShldrX':>7} {'HipX':>7} {'NoseNekX':>9} "
              f"{'NekHip':>7} {'CorShldr':>8} {'CorHip':>7} "
              f"{'dShldr':>7} {'dNose':>7} {'Diff':>7} "
              f"{'Indep':>5} {'Lock':>4} {'BodyCh':>7} {'DynLim':>7} "
              f"{'BodyAbs':>7} {'Sign':>4} {'RawYaw':>7} {'FiltYaw':>8} {'CumYaw':>8} {'FinalYaw':>9}")
    lines.append(header)
    lines.append("-" * len(header))

    for frame_idx, frame_rows in enumerate(frames_data):
        for (f_idx, p_idx, shldr_x, hip_x, nose_neck_x,
             cur_neck_hip, shldr_corr, hip_corr,
             d_shldr, d_nose, diff_ch,
             is_indep, locked, body_abs, used_sign, raw_yaw,
             body_ch, dyn_lim) in frame_rows:

            filt_yaw = filt_yaws_wrapped[frame_idx] if frame_idx < len(filt_yaws_wrapped) else None
            cum_yaw = cum_filtered[frame_idx] if frame_idx < len(cum_filtered) else None
            final_yaw = final_seq[frame_idx] if frame_idx < len(final_seq) else None

            def fmt(val, w=7):
                if val is None: return " " * w
                if isinstance(val, float):
                    return f"{val:{w}.2f}" if w <= 7 else f"{val:{w}.3f}"
                return f"{str(val):>{w}}"
            def fmt_int(val, w=4): return f"{val:{w}d}" if val is not None else " " * w
            def fmt_bool(val, w=5): return f"{'True':>{w}}" if val else f"{'False':>{w}}"

            line = (f"{f_idx:5d} {p_idx:4d} | "
                    f"{fmt(shldr_x,7)} {fmt(hip_x,7)} {fmt(nose_neck_x,9)} "
                    f"{fmt(cur_neck_hip,7)} {fmt(shldr_corr,8)} {fmt(hip_corr,7)} "
                    f"{fmt(d_shldr,7)} {fmt(d_nose,7)} {fmt(diff_ch,7)} "
                    f"{fmt_bool(is_indep,5)} {fmt_bool(locked,4)} {fmt(body_ch,7)} {fmt(dyn_lim,7)} "
                    f"{fmt(body_abs,7)} {fmt_int(used_sign,4)} "
                    f"{fmt(raw_yaw,7)} {fmt(filt_yaw,8)} {fmt(cum_yaw,8)} {fmt(final_yaw,9)}")
            lines.append(line)

    return (yaw_tensor, "\n".join(lines))


# ==================== 简化版节点 ====================
class SDPoseEstimateYawSimple:
    """
    计算人物偏航角（简化版）。只需设置四个核心参数。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unwrap_angle": ("BOOLEAN", {"default": True, "tooltip": "是否输出解缠累计转向角，否则折叠至[-180,180]"}),
                "shoulder_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "肩部角度融合权重，髋部权重自动为 1 - 该值"}),
            },
            "optional": {
                "pose_keypoints_full": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("yaw_array", "yaw_table")
    FUNCTION = "calculate_yaw"
    CATEGORY = "SDPose"

    def calculate_yaw(self, pose_keypoints_segment, conf_threshold, unwrap_angle,
                      shoulder_weight, pose_keypoints_full=None):
        nose_neck_thresh = 3.0
        smooth_window = 3
        enable_side_calib = True
        head_body_diff_threshold = 0.05
        confirmation_frames = 2
        ema_alpha = 0.4
        min_angle_limit = 1.0
        max_angle_limit = 60.0

        return _compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            nose_neck_thresh, smooth_window, enable_side_calib,
            shoulder_weight, head_body_diff_threshold, confirmation_frames,
            ema_alpha, min_angle_limit, max_angle_limit,
            pose_keypoints_full,
            merge_threshold=30.0
        )
    

# ==================== 进阶版节点 ====================
class SDPoseEstimateYawAdvanced:
    """
    计算人物偏航角（完整参数版）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints_segment": ("POSE_KEYPOINT",),
                "conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unwrap_angle": ("BOOLEAN", {"default": True}),
                "nose_neck_thresh": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "smooth_window": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "enable_side_calib": ("BOOLEAN", {"default": True}),
                "shoulder_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "head_body_diff_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.3, "step": 0.01}),
                "confirmation_frames": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "ema_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_angle_limit": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 1.0}),
                "max_angle_limit": ("FLOAT", {"default": 60.0, "min": 20.0, "max": 120.0, "step": 5.0}),
                "coverage_merge_threshold": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 180.0, "step": 1.0,
                    "tooltip": "在计算偏航角覆盖范围时，角度间隔小于此值的相邻区间将被合并"
                }),
            },
            "optional": {
                "pose_keypoints_full": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("yaw_array", "yaw_table")
    FUNCTION = "calculate_yaw"
    CATEGORY = "SDPose"

    def calculate_yaw(self, pose_keypoints_segment, conf_threshold, unwrap_angle,
                      nose_neck_thresh, smooth_window, enable_side_calib,
                      shoulder_weight, head_body_diff_threshold, confirmation_frames,
                      ema_alpha, min_angle_limit, max_angle_limit,
                      coverage_merge_threshold=30.0,
                      pose_keypoints_full=None):
        return _compute_yaw_core(
            pose_keypoints_segment, conf_threshold, unwrap_angle,
            nose_neck_thresh, smooth_window, enable_side_calib,
            shoulder_weight, head_body_diff_threshold, confirmation_frames,
            ema_alpha, min_angle_limit, max_angle_limit,
            pose_keypoints_full,
            merge_threshold=coverage_merge_threshold
        )
    

# ==================== 骨骼缩放 ====================
class SDPoseResizeKeypoints:
    """
    缩放骨骼关键点坐标，并更新画布尺寸。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "new_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "new_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "allow_crop": ("BOOLEAN", {"default": False,
                                           "tooltip": "当保持宽高比且比例不匹配时，基于关键点包围盒智能裁剪，保证不丢失关键点"}),
                "padding_top": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1,
                                        "tooltip": "包围盒向上扩展的像素数"}),
                "padding_bottom": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1,
                                           "tooltip": "包围盒向下扩展的像素数"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("resized_pose",)
    FUNCTION = "resize"
    CATEGORY = "SDPose"

    def resize(self, pose_keypoints, new_width, new_height, keep_aspect_ratio,
               allow_crop, padding_top, padding_bottom):
        if not isinstance(pose_keypoints, list) or not pose_keypoints:
            return (pose_keypoints,)

        frames = pose_keypoints

        if keep_aspect_ratio and allow_crop:
            all_x = []
            all_y = []
            for frame in frames:
                for person in frame.get("people", []):
                    for key, value in person.items():
                        if key.endswith("_keypoints_2d") and isinstance(value, list):
                            for i in range(0, len(value), 3):
                                c = value[i+2] if i+2 < len(value) else 0.0
                                if c > 0.0:
                                    all_x.append(value[i])
                                    all_y.append(value[i+1])
            if not all_x:
                pass
            else:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                padding_left = padding_right = (padding_top + padding_bottom) / 2.0
                safe_min_x = min_x - padding_left
                safe_max_x = max_x + padding_right
                safe_min_y = min_y - padding_top
                safe_max_y = max_y + padding_bottom
                safe_w = safe_max_x - safe_min_x
                safe_h = safe_max_y - safe_min_y
                target_ratio = new_width / new_height
                if safe_w / safe_h > target_ratio:
                    new_h = safe_w / target_ratio
                    expand = (new_h - safe_h) / 2.0
                    safe_min_y -= expand
                    safe_max_y += expand
                else:
                    new_w = safe_h * target_ratio
                    expand = (new_w - safe_w) / 2.0
                    safe_min_x -= expand
                    safe_max_x += expand
                box_w = safe_max_x - safe_min_x
                box_h = safe_max_y - safe_min_y
                result_frames = []
                for frame in frames:
                    new_frame = {
                        "canvas_width": new_width,
                        "canvas_height": new_height,
                        "people": []
                    }
                    for person in frame.get("people", []):
                        new_person = {}
                        for key, value in person.items():
                            if key.endswith("_keypoints_2d") and isinstance(value, list):
                                new_keypoints = []
                                for i in range(0, len(value), 3):
                                    x = value[i]
                                    y = value[i+1]
                                    c = value[i+2] if i+2 < len(value) else 0.0
                                    nx = (x - safe_min_x) * new_width / box_w
                                    ny = (y - safe_min_y) * new_height / box_h
                                    new_keypoints.extend([nx, ny, c])
                                new_person[key] = new_keypoints
                            else:
                                new_person[key] = value
                        new_frame["people"].append(new_person)
                    result_frames.append(new_frame)
                return (result_frames,)

        result_frames = []
        for frame in frames:
            old_w = frame.get("canvas_width", new_width)
            old_h = frame.get("canvas_height", new_height)

            if keep_aspect_ratio:
                scale_w = new_width / old_w if old_w > 0 else 1.0
                scale_h = new_height / old_h if old_h > 0 else 1.0
                scale = min(scale_w, scale_h)
                scaled_w = old_w * scale
                scaled_h = old_h * scale
                offset_x = (new_width - scaled_w) / 2.0
                offset_y = (new_height - scaled_h) / 2.0
            else:
                scale_w = new_width / old_w if old_w > 0 else 1.0
                scale_h = new_height / old_h if old_h > 0 else 1.0
                scale = 1.0
                offset_x = 0.0
                offset_y = 0.0

            new_frame = {
                "canvas_width": new_width,
                "canvas_height": new_height,
                "people": []
            }
            for person in frame.get("people", []):
                new_person = {}
                for key, value in person.items():
                    if key.endswith("_keypoints_2d") and isinstance(value, list):
                        new_keypoints = []
                        for i in range(0, len(value), 3):
                            x = value[i]
                            y = value[i+1]
                            c = value[i+2] if i+2 < len(value) else 0.0
                            if keep_aspect_ratio:
                                nx = x * scale + offset_x
                                ny = y * scale + offset_y
                            else:
                                nx = x * scale_w
                                ny = y * scale_h
                            new_keypoints.extend([nx, ny, c])
                        new_person[key] = new_keypoints
                    else:
                        new_person[key] = value
                new_frame["people"].append(new_person)
            result_frames.append(new_frame)

        return (result_frames,)
    


# ==================== 更新节点映射 ====================
NODE_CLASS_MAPPINGS = {
    "SDPoseDrawKeypointsV2": SDPoseDrawKeypointsV2,
    "SDPoseSaveJson": SDPoseSaveJson,
    "SDPoseLoadJson": SDPoseLoadJson,
    "SDPoseSliceKeypoints": SDPoseSliceKeypoints,
    "SDPoseConcatKeypoints": SDPoseConcatKeypoints,
    "SDPoseAlignFaceScale": SDPoseAlignFaceScale,
    "SDPoseEstimateYawSimple": SDPoseEstimateYawSimple,
    "SDPoseEstimateYawAdvanced": SDPoseEstimateYawAdvanced,
    "SDPoseResizeKeypoints": SDPoseResizeKeypoints,
    "SDPoseResampleKeypoints": SDPoseResampleKeypoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPoseDrawKeypointsV2": "Draw SDPose Keypoints (V2)",
    "SDPoseSaveJson": "Save SDPose Keypoints as JSON",
    "SDPoseLoadJson": "Load SDPose JSON",
    "SDPoseSliceKeypoints": "Slice SDPose Keypoints",
    "SDPoseConcatKeypoints": "Concat SDPose Keypoints",
    "SDPoseAlignFaceScale": "Align Face Scale (SDPose)",
    "SDPoseEstimateYawSimple": "Estimate Yaw (Simple)",
    "SDPoseEstimateYawAdvanced": "Estimate Yaw (Advanced)",
    "SDPoseResizeKeypoints": "Resize SDPose Keypoints",
    "SDPoseResampleKeypoints": "Resample SDPose Keypoints",
}
