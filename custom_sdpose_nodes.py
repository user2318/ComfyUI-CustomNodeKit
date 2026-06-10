# custom_sdpose_nodes.py
# 整合版：包含 SDPose 加载/切片/拼接/绘制/缩放节点
# 注意：绘制依赖 sdpose_draw.py，偏航角依赖 sdpose_yaw.py

import torch
import numpy as np
import math
import json
import os
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import comfy.utils
import folder_paths

from .sdpose_draw import KeypointDraw


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


# ==================== 公用重采样辅助函数 ====================
# 将 SDPoseLoadJson 和 SDPoseResampleKeypoints 中的重复逻辑提取为模块级函数

def _resample_downsample(frames, ratio, json_fps):
    n = len(frames)
    target_n = int(round(n / ratio))
    if target_n >= n:
        return frames[:]
    if target_n <= 0:
        return [frames[0]]
    duration = (n - 1) / json_fps
    orig_times = np.arange(n) / json_fps
    target_times = np.linspace(0, duration, target_n)
    # O(log n) per target: searchsorted + 向量化最近邻
    right = np.searchsorted(orig_times, target_times, side='right')
    right = np.clip(right, 1, n - 1)
    left = right - 1
    # 向量化最近邻：左近取左，右近取右
    closer_left = (target_times - orig_times[left]) <= (orig_times[right] - target_times)
    indices = np.where(closer_left, left, right)
    return [frames[int(i)] for i in indices]

def _resample_upsample_duplicate(frames, ratio, json_fps):
    n = len(frames)
    target_n = int(round(n / ratio))
    if target_n <= n:
        return frames[:]
    duration = (n - 1) / json_fps
    orig_times = np.arange(n) / json_fps
    target_times = np.linspace(0, duration, target_n)
    right = np.searchsorted(orig_times, target_times, side='right')
    right = np.clip(right, 1, n - 1)
    left = right - 1
    closer_left = (target_times - orig_times[left]) <= (orig_times[right] - target_times)
    indices = np.where(closer_left, left, right)
    return [frames[int(i)] for i in indices]

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

def _resample_upsample_interpolate(frames, ratio, json_fps):
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
        if _is_abrupt_jump(left_frame, right_frame):
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

def _fix_empty_frames(frames):
    """
    扫描帧列表中 people 为空的帧，用前后有效帧的插值数据填充。
    返回修复后的新列表（不影响原列表）。
    """
    n = len(frames)
    if n == 0:
        return frames

    def _is_valid(frame):
        people = frame.get("people", [])
        if not people:
            return False
        for person in people:
            for key, val in person.items():
                if key.endswith("_keypoints_2d") and isinstance(val, list) and len(val) > 0:
                    for i in range(0, len(val), 3):
                        x = val[i]
                        y = val[i+1] if i+1 < len(val) else 0.0
                        if x > 0 or y > 0:
                            return True
        return False

    valid_flags = [_is_valid(f) for f in frames]

    if all(valid_flags) or not any(valid_flags):
        return frames[:]

    result = []
    for i in range(n):
        if valid_flags[i]:
            result.append(frames[i])
            continue

        left_idx = i - 1
        while left_idx >= 0 and not valid_flags[left_idx]:
            left_idx -= 1

        right_idx = i + 1
        while right_idx < n and not valid_flags[right_idx]:
            right_idx += 1

        if left_idx >= 0 and right_idx < n:
            total_gap = right_idx - left_idx
            alpha = (i - left_idx) / total_gap
            left_frame = frames[left_idx]
            right_frame = frames[right_idx]

            if _is_abrupt_jump(left_frame, right_frame):
                fixed = right_frame if alpha > 0.5 else left_frame
            else:
                fixed = {
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
                    fixed["people"].append(new_person)
            result.append(fixed)
        elif left_idx >= 0:
            result.append(frames[left_idx])
        elif right_idx < n:
            result.append(frames[right_idx])
        else:
            result.append(frames[i])

    return result


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
                                  "tooltip": "若 >0，将 fps 写入 JSON 头部；=0 则存为裸数组（兼容旧版）" }),
                "overwrite": ("BOOLEAN", {"default": True,
                                          "tooltip": "True=覆盖上次文件（向后兼容）；False=自动递增编号不覆盖" }),
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
                "fix_empty_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后，自动检测并修复 people 为空的帧，用前后有效帧插值填充，消除黑帧"
                }),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("pose_keypoints", "frame_count", "canvas_width", "canvas_height", "effective_fps")
    FUNCTION = "load"
    CATEGORY = "SDPose"

    def load(self, json_path, target_fps=0, interp_method="interpolate", fix_empty_frames=False):
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
                result = _resample_downsample(frames, ratio, json_fps)
            else:
                if interp_method == "interpolate":
                    result = _resample_upsample_interpolate(frames, ratio, json_fps)
                else:
                    result = _resample_upsample_duplicate(frames, ratio, json_fps)
        else:
            result = frames

        # 自动修复空帧
        if fix_empty_frames:
            result = _fix_empty_frames(result)

        frame_count = len(result)
        if target_fps == 0:
            effective_fps = float(json_fps) if json_fps else 0.0
        else:
            if json_fps is not None and json_fps > 0:
                effective_fps = float(target_fps)
            else:
                effective_fps = 0.0

        return (result, frame_count, canvas_w, canvas_h, effective_fps)


# ==================== 重采样（抽帧/补帧） ====================
class SDPoseResampleKeypoints:
    """
    对 POSE_KEYPOINT 数据进行重采样（抽帧/补帧）。
    接收输入 fps 和输出 fps，独立于 JSON 载入逻辑。
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
            },
            "optional": {
                "fix_empty_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后，自动检测并修复 people 为空的帧，用前后有效帧插值填充，消除黑帧"
                }),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "FLOAT")
    RETURN_NAMES = ("pose_keypoints", "fps")
    FUNCTION = "resample"
    CATEGORY = "SDPose"

    def resample(self, pose_keypoints, input_fps=0, output_fps=0, interp_method="interpolate", fix_empty_frames=False):
        if not isinstance(pose_keypoints, list) or len(pose_keypoints) == 0:
            return (pose_keypoints, 0.0)

        if input_fps <= 0 or output_fps <= 0 or len(pose_keypoints) <= 1:
            result = pose_keypoints
        else:
            ratio = input_fps / output_fps
            if abs(ratio - 1.0) < 1e-6:
                result = pose_keypoints[:]
            elif ratio > 1.0:
                result = _resample_downsample(pose_keypoints, ratio, input_fps)
            else:
                if interp_method == "interpolate":
                    result = _resample_upsample_interpolate(pose_keypoints, ratio, input_fps)
                else:
                    result = _resample_upsample_duplicate(pose_keypoints, ratio, input_fps)

        # 自动修复空帧
        if fix_empty_frames:
            result = _fix_empty_frames(result)

        if output_fps > 0 and input_fps > 0 and len(pose_keypoints) > 1:
            return (result, float(output_fps))
        return (result, 0.0)


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
                "yaw_thickness_min": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1, "tooltip": "侧面(±90°)时的最细粗细，自动受 stick_width 约束（不会高于 stick_width）"}),
                "foot_mode": (["dots", "line"], {"default": "dots", "tooltip": "脚部绘制模式：dots=三个彩色圆点（原有行为），line=从踝关节到脚尖画一条骨骼线"}),
                "color_scheme": (["v4_custom", "standard", "monochrome"], {"default": "v4_custom", "tooltip": "骨骼配色方案：v4_custom=自定义V4方案(默认), standard=标准OpenPose彩虹配色, monochrome=统一灰色"}),
            },
            "optional": {
                "yaw_angles": ("FLOAT", { "tooltip": "来自 SDPoseEstimateYawSimple/Advanced 的 yaw_array，用于动态调整骨骼粗细和遮挡顺序" }),
                "hand_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01,
                                         "tooltip": "手部骨骼缩放比例，1.0=原始大小。对手部骨骼（从手腕向外）进行缩放，手腕位置保持不动避免错位"}),
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
             foot_mode="dots",
             yaw_angles=None,
             hand_scale=1.0,
             color_scheme="v4_custom"):
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
                effective_min = min(yaw_thickness_min, stick_width)
                ratio = abs(math.cos(math.radians(yaw_val)))
                cur_stick_width = int(effective_min + (stick_width - effective_min) * ratio)
                cur_stick_width = max(effective_min, min(stick_width, cur_stick_width))
            
            for person_idx in range(len(frame.get("people", []))):
                kp, sc = get_keypoint_arrays(frame, person_idx)
                
                if draw_face and mouth_mode != "draw_all":
                    kp = kp.copy()
                    sc = sc.copy()
                    if mouth_mode == "no_draw":
                        hide_idx = range(72, 92)
                    elif mouth_mode == "inner_lip_only":
                        hide_idx = range(72, 84)
                    else:
                        hide_idx = []
                    idx_arr = np.array(list(hide_idx), dtype=int)
                    mask = idx_arr < len(kp)
                    valid_idx = idx_arr[mask]
                    if len(valid_idx):
                        kp[valid_idx] = [-1.0, -1.0]
                        sc[valid_idx] = 0.0

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
                    yaw=cur_yaw,
                    hand_scale=hand_scale,
                    color_scheme=color_scheme,
                    foot_mode=foot_mode
                )

            pose_outputs.append(canvas)

        pose_outputs_np = np.stack(pose_outputs) if len(pose_outputs) > 1 else np.expand_dims(pose_outputs[0], 0)
        result = torch.from_numpy(pose_outputs_np).float() / 255.0
        return (result,)


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
                                        "tooltip": "【仅在允许裁剪时有效】包围盒向上扩展的像素数"}),
                "padding_bottom": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1,
                                           "tooltip": "【仅在允许裁剪时有效】包围盒向下扩展的像素数"}),
                "score_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "【仅在允许裁剪时有效】参与包围盒计算的最低关键点置信度"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("resized_pose",)
    FUNCTION = "resize"
    CATEGORY = "SDPose"

    def resize(self, pose_keypoints, new_width, new_height, keep_aspect_ratio,
               allow_crop, padding_top, padding_bottom, score_threshold=0.1):
        if not isinstance(pose_keypoints, list) or not pose_keypoints:
            return (pose_keypoints,)

        frames = pose_keypoints

        if keep_aspect_ratio and allow_crop:
            all_x = []
            all_y = []
            for frame in frames:
                canvas_w = frame.get("canvas_width", 0)
                canvas_h = frame.get("canvas_height", 0)
                for person in frame.get("people", []):
                    for key, value in person.items():
                        if key.endswith("_keypoints_2d") and isinstance(value, list):
                            for i in range(0, len(value), 3):
                                c = value[i+2] if i+2 < len(value) else 0.0
                                x = value[i]
                                y = value[i+1]
                                if c > score_threshold and 0 <= x < canvas_w and 0 <= y < canvas_h:
                                    all_x.append(x)
                                    all_y.append(y)
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
                box_w = safe_max_x - safe_min_x
                box_h = safe_max_y - safe_min_y
                # 包围盒只外扩不内缩，等比缩放+居中填入目标画布
                # 保证所有关键点始终完整保留在包围盒内
                scale = min(new_width / box_w, new_height / box_h) if box_w > 0 and box_h > 0 else 1.0
                scaled_w = box_w * scale
                scaled_h = box_h * scale
                offset_x = (new_width - scaled_w) / 2.0
                offset_y = (new_height - scaled_h) / 2.0
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
                                    nx = (x - safe_min_x) * scale + offset_x
                                    ny = (y - safe_min_y) * scale + offset_y
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
# 从 sdpose_yaw 导入偏航角节点的映射
from .sdpose_yaw import NODE_CLASS_MAPPINGS as YAW_NODE_CLASS_MAPPINGS
from .sdpose_yaw import NODE_DISPLAY_NAME_MAPPINGS as YAW_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "SDPoseDrawKeypointsV2": SDPoseDrawKeypointsV2,
    "SDPoseSaveJson": SDPoseSaveJson,
    "SDPoseLoadJson": SDPoseLoadJson,
    "SDPoseSliceKeypoints": SDPoseSliceKeypoints,
    "SDPoseConcatKeypoints": SDPoseConcatKeypoints,
    "SDPoseResizeKeypoints": SDPoseResizeKeypoints,
    "SDPoseResampleKeypoints": SDPoseResampleKeypoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPoseDrawKeypointsV2": "Draw SDPose Keypoints (V2)",
    "SDPoseSaveJson": "Save SDPose Keypoints as JSON",
    "SDPoseLoadJson": "Load SDPose JSON",
    "SDPoseSliceKeypoints": "Slice SDPose Keypoints",
    "SDPoseConcatKeypoints": "Concat SDPose Keypoints",
    "SDPoseResizeKeypoints": "Resize SDPose Keypoints",
    "SDPoseResampleKeypoints": "Resample SDPose Keypoints",
}

# 合并偏航角节点映射
NODE_CLASS_MAPPINGS.update(YAW_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(YAW_NODE_DISPLAY_NAME_MAPPINGS)