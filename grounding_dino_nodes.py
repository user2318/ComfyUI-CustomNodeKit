# grounding_dino_nodes.py
# GroundingDINO 检测节点组：与官方 SDPoseKeypointExtractor 无缝配合
# 包含两个节点：
#   GD_ModelLoader   - 加载 GD 模型（全局缓存，常驻 GPU）
#   GD_BBoxDetect    - 检测 + 筛选 + 预览，一站式输出 bbox 给官方 SDPose
#
# 模型文件位置：
#   GD 模型:    {models_dir}/grounding-dino/   (自动下载)
#   bert 编码器: {models_dir}/clip/bert-base-uncased/ (自动下载)
#
# 依赖安装: pip install groundingdino-py transformers

import os
import sys
import json
import math
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2

import folder_paths
import comfy.model_management
import comfy.utils

# ──────────────────── 模块级全局缓存 ────────────────────
_GD_MODEL_CACHE: Dict[str, dict] = {}

# ──────────────────── 模型文件管理 ────────────────────

# HuggingFace 上的 GD 模型列表
GROUNDINGDINO_MODEL_LIST = {
    "GroundingDINO_SwinT_OGC": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "display": "SwinT (694MB, 快速)",
    },
    "GroundingDINO_SwinB": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
        "display": "SwinB (938MB, 更精准)",
    },
}

GD_MODEL_DIR = os.path.join(folder_paths.models_dir, "grounding-dino")
BERT_MODEL_DIR = os.path.join(folder_paths.models_dir, "clip", "bert-base-uncased")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _get_local_filepath(url: str, subdir: str) -> str:
    """从 url 下载文件到 models/{subdir}/，返回本地路径（已缓存则跳过）。"""
    dest_dir = os.path.join(folder_paths.models_dir, subdir)
    _ensure_dir(dest_dir)
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        return dest_path
    print(f"[GD Nodes] Downloading {filename} from {url} ...")
    from torch.hub import download_url_to_file
    download_url_to_file(url, dest_path)
    return dest_path


# ──────────────────── GD 模型加载（全局缓存） ────────────────────

def _load_gd_model(model_name: str) -> dict:
    """加载 GroundingDINO 模型，全局缓存，常驻 GPU。"""
    global _GD_MODEL_CACHE
    if model_name in _GD_MODEL_CACHE:
        return _GD_MODEL_CACHE[model_name]

    try:
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
    except ImportError:
        raise ImportError(
            "[GD Nodes] 'groundingdino' library not found. "
            "Please install: pip install groundingdino-py"
        )

    model_info = GROUNDINGDINO_MODEL_LIST[model_name]
    
    config_path = _get_local_filepath(model_info["config_url"], "grounding-dino")
    model_path  = _get_local_filepath(model_info["model_url"], "grounding-dino")

    print(f"[GD Nodes] Building GroundingDINO model: {model_name} ...")
    args = SLConfig.fromfile(config_path)
    
    if args.text_encoder_type == "bert-base-uncased":
        if os.path.exists(BERT_MODEL_DIR):
            args.text_encoder_type = BERT_MODEL_DIR

    model = build_model(args)
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()

    device = comfy.model_management.get_torch_device()
    model = model.to(device)
    print(f"[GD Nodes] Model loaded on {device}, will stay resident.")

    wrapper = {"model": model, "model_name": model_name}
    _GD_MODEL_CACHE[model_name] = wrapper
    return wrapper


# ──────────────────── GroundingDINO 推理（GPU 批处理版） ────────────────────

import torch.nn.functional as _F

# ImageNet 归一化常量（CPU 张量，使用时移至对应设备）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _preprocess_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """将 BHWC 图像 batch (0~1, float32) 在 GPU 上直接做缩放+归一化。
    返回 (B, 3, H', W') 符合 GD 模型输入的 tensor。
    """
    x = images.permute(0, 3, 1, 2).to(device)

    _, _, H, W = x.shape
    h_f, w_f = float(H), float(W)

    # 等比例缩放：短边到 800，长边不超过 1333
    ratio = min(800.0 / min(h_f, w_f), 1333.0 / max(h_f, w_f))
    new_h = int(round(h_f * ratio))
    new_w = int(round(w_f * ratio))

    x = _F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # ImageNet 归一化（直接用模块级常量，移到对应设备）
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    return x


def _gd_predict_batch(model_wrapper: dict, images_batch: torch.Tensor,
                      prompt: str, threshold: float,
                      batch_size: int = 4) -> Tuple[List[List[List[float]]], List[List[float]]]:
    """用 GD 模型在 BHWC 图像 batch 上推理（内部自动拆分防 OOM）。
    batch_size 控制每批处理多少帧，减少显存占用。
    返回 (all_boxes, all_scores):
      - all_boxes:  [[[x1,y1,x2,y2], ...], ...] — 外层帧索引，内层每帧的 bbox 列表
      - all_scores: [[score, ...], ...] — 每个 bbox 对应的最高置信度
    """
    model = model_wrapper["model"]
    device = next(model.parameters()).device
    B, H, W, C = images_batch.shape

    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    all_boxes_list = []
    all_scores_list = []

    # 内部自动拆分为 batch_size 大小的子批次
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        chunk = images_batch[start:end]

        preprocessed = _preprocess_batch(chunk, device)

        with torch.no_grad():
            outputs = model(preprocessed, captions=[caption] * (end - start))

        logits_chunk = outputs["pred_logits"].sigmoid()
        boxes_chunk  = outputs["pred_boxes"]

        for i in range(end - start):
            scores_per_box, _ = logits_chunk[i].max(dim=1)
            mask = scores_per_box > threshold
            boxes = boxes_chunk[i][mask]
            scores = scores_per_box[mask]

            if boxes.shape[0] == 0:
                all_boxes_list.append([])
                all_scores_list.append([])
                continue

            cx, cy, w, h = boxes.unbind(1)
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H

            frame_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            all_boxes_list.append(frame_boxes.cpu().tolist())
            all_scores_list.append(scores.cpu().tolist())

    return all_boxes_list, all_scores_list


# ──────────────────── 节点 1: GD 模型加载器 ────────────────────

class GD_ModelLoader:
    """加载 GroundingDINO 模型。全局缓存 + 常驻 GPU。首次加载 ~5s，后续瞬间。"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(GROUNDINGDINO_MODEL_LIST.keys()), {
                    "default": "GroundingDINO_SwinT_OGC"
                }),
            }
        }

    RETURN_TYPES = ("GD_MODEL",)
    RETURN_NAMES = ("gd_model",)
    FUNCTION = "load"
    CATEGORY = "SDPose/GD"

    def load(self, model_name: str):
        wrapper = _load_gd_model(model_name)
        return (wrapper,)


# ──────────────────── 节点 2: GD BBox 检测（内置筛选 + 预览） ────────────────────

class GD_BBoxDetect:
    """一站式节点：用 GroundingDINO 检测 → 筛选 → 预览 → 输出 bbox 给官方 SDPose。
    
    筛选模式说明:
      - pass_all:    保留 GD 检测出的所有目标（不做筛选）
      - by_area:     每帧选面积最大/最小的目标（适合主角始终是最大的人）
      - by_position: 每帧选特定区域的目标（如只保留画面左半侧的人）
      - track:       跨帧追踪同一个人（第1帧选最大的人，后续帧自动追踪）
      - by_index:    按检测顺序的索引选（适合单帧调试，视频帧间不稳定）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gd_model": ("GD_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person .",
                    "multiline": False,
                    "placeholder": '例: "1girl", "man in red shirt", "person wearing glasses"'
                }),
                "threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "置信度阈值。越高越精确但可能漏检；越低检测越多但可能有误检"
                }),
                "mode": ([
                    "pass_all",
                    "by_area",
                    "by_position",
                    "track",
                    "by_index",
                ], {
                    "default": "pass_all",
                    "tooltip": "pass_all=保留全部 | by_area=按面积选最大/最小 | by_position=按区域选 | track=跨帧追踪 | by_index=按索引选"
                }),
                "strategy": ([
                    "largest",
                    "smallest",
                    "highest_score",
                ], {
                    "default": "largest",
                    "tooltip": "largest=面积最大的 | smallest=面积最小的 | highest_score=置信度最高的（仅by_area/by_position模式有效）"
                }),
                "index": ("INT", {
                    "default": 0, "min": -1, "max": 100, "step": 1,
                    "tooltip": "by_index模式下选择第几个bbox（0=第一个，-1=最后一个）。注意：视频帧间索引可能不稳定"
                }),
                "region": ([
                    "all",
                    "left_half",
                    "right_half",
                    "center_third",
                    "top_half",
                    "bottom_half",
                ], {
                    "default": "all",
                    "tooltip": "by_position模式下筛选区域：left_half=左半侧 | right_half=右半侧 | center_third=中间1/3 | top_half=上半 | bottom_half=下半"
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "每批处理的帧数。减小以节省显存：4GB显存→1, 6GB→2, 8GB→4(默认), 12GB+→8~16"
                }),
                "preview_count": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "预览帧数。自动选首帧+bbox最多帧+末帧+等间隔补足，不影响实际筛选结果"
                }),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bboxes", "preview_images", "preview_info")
    FUNCTION = "detect"
    CATEGORY = "SDPose/GD"

    # ── 筛选辅助 ──

    def _filter_frame(self, bboxes: List[dict], mode: str, strategy: str,
                      index: int, region: str, W: float, H: float) -> List[dict]:
        """对单帧 bbox 列表执行筛选。"""
        if not bboxes or mode == "pass_all":
            return bboxes

        # 区域预筛（仅 by_position 模式下生效）
        if region != "all":
            bboxes = [b for b in bboxes if self._in_region(b, region, W, H)]
            if not bboxes:
                return []

        if mode == "by_index":
            idx = index if index >= 0 else len(bboxes) + index
            return [bboxes[idx]] if 0 <= idx < len(bboxes) else []

        # by_area / by_position
        if strategy == "largest":
            return [max(bboxes, key=lambda b: b["width"] * b["height"])]
        elif strategy == "smallest":
            return [min(bboxes, key=lambda b: b["width"] * b["height"])]
        elif strategy == "highest_score":
            return [max(bboxes, key=lambda b: b.get("score", 0))]

        return bboxes

    def _in_region(self, bbox: dict, region: str, W: float, H: float) -> bool:
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"] / 2
        if region == "left_half":    return cx < W / 2
        if region == "right_half":   return cx >= W / 2
        if region == "center_third": return W / 3 <= cx <= W * 2 / 3
        if region == "top_half":     return cy < H / 2
        if region == "bottom_half":  return cy >= H / 2
        return True

    # ── 预览绘制 ──

    def _draw_preview(self, img_tensor, all_bboxes, selected_set, frame_idx, total_frames):
        canvas = (img_tensor.cpu().numpy() * 255).astype(np.uint8).copy()
        colors = [
            (255, 0, 0), (0, 165, 255), (0, 255, 0), (0, 165, 255),
            (128, 0, 128), (255, 192, 203), (0, 255, 255), (255, 255, 0),
            (255, 0, 255), (0, 128, 128),
        ]
        for i, b in enumerate(all_bboxes):
            x, y = int(b["x"]), int(b["y"])
            x2, y2 = int(b["x"] + b["width"]), int(b["y"] + b["height"])
            sel = i in selected_set
            color = (0, 255, 255) if sel else colors[i % len(colors)]
            cv2.rectangle(canvas, (x, y), (x2, y2), color, 3 if sel else 1)
            label = f"{'✓ ' if sel else ''}[{i}] {b.get('label','')} ({b.get('score',0):.2f})"
            cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame {frame_idx}/{total_frames}", (8, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return canvas

    # ── 主入口 ──

    def detect(self, gd_model, image, prompt, threshold, mode, strategy, index, region, batch_size, preview_count):
        B, H, W, C = image.shape

        # 第 1 步：GPU 批处理检测所有帧
        raw_boxes, raw_scores = _gd_predict_batch(gd_model, image, prompt, threshold, batch_size)
        all_detections = []
        for frame_boxes, frame_scores in zip(raw_boxes, raw_scores):
            frame_bboxes = []
            for box, score in zip(frame_boxes, frame_scores):
                x1, y1, x2, y2 = box
                frame_bboxes.append({
                    "x": max(0.0, x1), "y": max(0.0, y1),
                    "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1),
                    "label": prompt.strip().rstrip("."),
                    "score": score,
                })
            all_detections.append(frame_bboxes)

        # 第 2 步：筛选
        filtered = []
        track_target = None  # (cx, cy) for track mode

        for frame_idx in range(B):
            frame_bboxes = all_detections[frame_idx]

            if mode == "track" and frame_bboxes:
                if track_target is not None:
                    tx, ty = track_target
                    best, best_dist = 0, float('inf')
                    for i, b in enumerate(frame_bboxes):
                        bx, by = b["x"] + b["width"]/2, b["y"] + b["height"]/2
                        d = math.hypot(bx - tx, by - ty)
                        if d < best_dist:
                            best_dist, best = d, i
                    sel = [frame_bboxes[best]]
                    track_target = (sel[0]["x"] + sel[0]["width"]/2,
                                    sel[0]["y"] + sel[0]["height"]/2)
                else:
                    sel = self._filter_frame(frame_bboxes, "by_area", strategy, 0, "all", W, H)
                    if sel:
                        track_target = (sel[0]["x"] + sel[0]["width"]/2,
                                        sel[0]["y"] + sel[0]["height"]/2)
            else:
                sel = self._filter_frame(frame_bboxes, mode, strategy, index, region, W, H)

            filtered.append(sel)

        # 第 3 步：智能采样预览
        max_frame = max(range(B), key=lambda i: len(all_detections[i])) if B > 0 else 0
        preview_set = {0, B - 1}
        if B > 1 and len(all_detections[max_frame]) > 0:
            preview_set.add(max_frame)
        remaining = preview_count - len(preview_set)
        if remaining > 0 and B > len(preview_set):
            step = (B - 1) / (remaining + 1)
            for i in range(1, remaining + 1):
                preview_set.add(int(round(step * i)))

        preview_imgs, preview_samples = [], []
        for idx in sorted(preview_set):
            fb = all_detections[idx] if idx < len(all_detections) else []
            sel_set = set()
            if filtered[idx]:
                for i, f in enumerate(fb):
                    for s in filtered[idx]:
                        if abs(f["x"] - s["x"]) < 1 and abs(f["y"] - s["y"]) < 1:
                            sel_set.add(i)
                            break
            preview_imgs.append(self._draw_preview(image[idx], fb, sel_set, idx, B))
            preview_samples.append({"frame": idx, "count": len(fb), "selected": list(sel_set)})

        if preview_imgs:
            preview_tensor = torch.from_numpy(np.stack(preview_imgs).astype(np.float32) / 255.0)
        else:
            preview_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        preview_json = json.dumps({"samples": preview_samples}, indent=2)

        return (filtered, preview_tensor, preview_json)


# ──────────────────── 节点映射 ────────────────────

NODE_CLASS_MAPPINGS = {
    "GD_ModelLoader": GD_ModelLoader,
    "GD_BBoxDetect": GD_BBoxDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GD_ModelLoader": "Load GroundingDINO Model",
    "GD_BBoxDetect": "GD BBox Detect",
}