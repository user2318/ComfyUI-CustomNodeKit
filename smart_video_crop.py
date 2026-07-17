"""
SmartVideoCrop — 基于 SAM3 追踪数据的智能视频裁剪节点。

根据 SAM3_VideoTrack 输出的追踪数据（mask），计算每帧人物质心位置，
以质心为基准做水平平移裁剪，固定保留原始画面全高，
按输出宽高比计算裁剪宽度，最终缩放到指定输出尺寸。

核心特性：
- 纯数学变换，完全确定性 → 分段间零跳变
- 支持多 ID 追踪（逗号分隔的 object_ids，空=默认 ID=1）
- 帧间 EMA 平滑防抖动
- 裁剪框始终 clamp 在画面内，不会出画
- 支持第二路图像同步裁剪（如遮罩视频），共用同一裁剪窗口
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional

import comfy.model_management
import comfy.utils

try:
    from comfy.ldm.sam3.tracker import unpack_masks
except ImportError:
    unpack_masks = None

logger = logging.getLogger("SmartVideoCrop")


class SmartVideoCrop:
    """基于 SAM3 追踪数据的智能视频裁剪节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "当前段的原始视频帧 (B, H, W, C)。Original video frames of the current chunk."
                }),
                "track_data": ("SAM3_TRACK_DATA", {
                    "tooltip": "SAM3_VideoTrack 输出的追踪数据（接 driving_track_data）。Track data from SAM3_VideoTrack (connect to driving_track_data)."
                }),
                "output_width": ("INT", {
                    "default": 480, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "输出视频宽度。Output video width."
                }),
                "output_height": ("INT", {
                    "default": 864, "min": 64, "max": 8192, "step": 16,
                    "tooltip": "输出视频高度。Output video height."
                }),
                "object_ids": ("STRING", {
                    "default": "1", "multiline": False,
                    "tooltip": "追踪的人物 ID，逗号分隔（如 '1' 或 '1,2'）。空=默认 ID=1。Person IDs to track, comma-separated (e.g. '1' or '1,2'). Empty=default ID=1."
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "帧间平滑度（0=无平滑，1=完全不跟随）。Inter-frame smoothing (0=no smoothing, 1=completely static)."
                }),
                "mask_fallback": (["hold_last", "center"], {
                    "default": "hold_last",
                    "tooltip": "mask 为空时的处理方式。hold_last=保持最后有效位置；center=回退到画面中心。Behavior when mask is empty. hold_last=keep last valid position; center=fall back to center."
                }),
                "scene_cut_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "切镜检测阈值（归一化，0=关闭检测）。仅当质心在连续有 mask 的帧间跳变超过此值时判定为切镜，关闭平滑。Scene cut threshold (normalized, 0=disabled). Only triggers when centroid jumps more than this between consecutive masked frames; disables smoothing on cut."
                }),
            },
            "optional": {
                "fallback_center": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "仅在 mask_fallback=center 时生效。回退水平中心比例（0=最左, 0.5=居中, 1=最右）。Only effective when mask_fallback=center. Fallback horizontal center ratio (0=left, 0.5=center, 1=right)."
                }),
                "mask_frames": ("IMAGE", {
                    "tooltip": "可选第二路图像（如遮罩），与 frames 同步裁剪。两路共用同一质心和裁剪窗口。Optional second image (e.g. mask) to crop with same parameters as frames."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("cropped_frames", "cropped_mask", "debug_info")
    FUNCTION = "crop"
    OUTPUT_NODE = False
    CATEGORY = "Custom Nodes/Video"

    def _parse_object_ids(self, ids_str: str) -> list[int]:
        ids_str = ids_str.strip()
        if not ids_str:
            return [1]
        ids = []
        for part in ids_str.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        return ids if ids else [1]

    def _unpack_and_resize(self, track_data: dict,
                           target_H: int, target_W: int,
                           device: torch.device) -> Optional[torch.Tensor]:
        if unpack_masks is None:
            logger.error("无法导入 unpack_masks")
            return None

        packed = track_data.get("packed_masks")
        if packed is None or packed.shape[1] == 0:
            logger.warning("track_data 中没有有效 packed_masks")
            return None

        T = packed.shape[0]
        N_obj = packed.shape[1] if len(packed.shape) > 1 else 1

        try:
            masks_raw = unpack_masks(packed.to(device))
            Hm, Wm = masks_raw.shape[-2], masks_raw.shape[-1]

            if len(masks_raw.shape) == 4 and masks_raw.shape[0] == T and masks_raw.shape[1] == N_obj:
                masks_flat = masks_raw.float()
            else:
                masks_flat = masks_raw.view(T * N_obj, 1, Hm, Wm).float()

            masks_resized = F.interpolate(masks_flat, size=(target_H, target_W), mode="nearest")

            if len(masks_resized.shape) == 4 and masks_resized.shape[0] == T and masks_resized.shape[1] == N_obj:
                masks_bool = masks_resized > 0.5
            else:
                masks_bool = masks_resized.view(T, N_obj, target_H, target_W) > 0.5

            return masks_bool
        except Exception as e:
            logger.warning(f"解包 mask 失败: {e}")
            return None

    def _get_combined_mask(self, masks_bool: torch.Tensor,
                           obj_ids: list[int],
                           frame_idx: int) -> Optional[torch.Tensor]:
        T = masks_bool.shape[0]
        if T == 0:
            return None
        idx = min(frame_idx, T - 1)
        frame_masks = masks_bool[idx]
        N_obj = frame_masks.shape[0]

        indices = [i - 1 for i in obj_ids if 1 <= i <= N_obj]
        if not indices:
            return None

        combined = frame_masks[indices[0]]
        for i in indices[1:]:
            combined = combined | frame_masks[i]
        return combined

    def _compute_centroid(self, mask: torch.Tensor) -> float:
        ys, xs = torch.where(mask > 0)
        if xs.numel() == 0:
            return 0.5
        cx = xs.float().mean().item()
        W = mask.shape[-1]
        return cx / W

    def _crop_single(self, frames: torch.Tensor,
                     crop_x_px: float, crop_w: float,
                     output_width: int, output_height: int,
                     device: torch.device) -> torch.Tensor:
        """对单路帧进行裁剪（质心已计算好）。"""
        B, H, W, C = frames.shape
        output_frames = []
        for frame_idx in range(B):
            frame = frames[frame_idx:frame_idx + 1].to(device)
            if crop_w >= W - 1e-6:
                cropped = comfy.utils.common_upscale(
                    frame.movedim(-1, 1), output_width, output_height, "bicubic", "center"
                ).movedim(1, -1)
            else:
                x_start = 2.0 * crop_x_px / max(W - 1, 1) - 1.0
                x_end = 2.0 * (crop_x_px + crop_w) / max(W - 1, 1) - 1.0
                out_h_int = int(round(H))
                out_w_int = int(round(crop_w))

                grid_y = torch.linspace(-1.0, 1.0, out_h_int, device=device)
                grid_x = torch.linspace(x_start, x_end, out_w_int, device=device)
                gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

                sampled = F.grid_sample(
                    frame.permute(0, 3, 1, 2).float(),
                    grid.expand(1, -1, -1, -1),
                    mode='bicubic', padding_mode='border', align_corners=True
                ).permute(0, 2, 3, 1)

                if out_w_int != output_width or out_h_int != output_height:
                    cropped = comfy.utils.common_upscale(
                        sampled.movedim(-1, 1), output_width, output_height, "bicubic", "center"
                    ).movedim(1, -1)
                else:
                    cropped = sampled
            output_frames.append(cropped.cpu())
        return torch.cat(output_frames, dim=0)

    def crop(self, frames: torch.Tensor, track_data: dict,
             output_width: int = 480, output_height: int = 864,
             object_ids: str = "1", smoothing: float = 0.3,
             mask_fallback: str = "hold_last",
             scene_cut_threshold: float = 0.3,
             fallback_center: float = 0.5,
             mask_frames: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        主处理函数。
        frames: (B, H, W, C) float32, 0~1
        mask_frames: (B, H, W, C) 可选，与 frames 同步裁剪
        """
        B, H, W, C = frames.shape
        device = comfy.model_management.get_torch_device()

        ids = self._parse_object_ids(object_ids)
        logger.info(f"SmartVideoCrop: {B} frames, {W}x{H}, output={output_width}x{output_height}, "
                     f"ids={ids}, smoothing={smoothing}")

        # ── 解包 mask ──
        masks_bool = self._unpack_and_resize(track_data, H, W, device)
        if masks_bool is None:
            logger.warning("无法获取 mask，回退到画面居中裁剪")
            crop_w = H * output_width / output_height
            crop_x_px = max(0, (W - crop_w) / 2)
            fallback_frames = self._fallback_crop(frames, crop_x_px, crop_w, output_width, output_height, device)
            fallback_mask = self._fallback_crop(mask_frames, crop_x_px, crop_w, output_width, output_height, device) if mask_frames is not None else torch.zeros(1, output_height, output_width, 3)
            return (fallback_frames, fallback_mask, "{}")

        T_total = masks_bool.shape[0]
        logger.info(f"mask 数据: {T_total} 帧, {masks_bool.shape[1]} 对象")

        # ── 计算裁剪窗口尺寸 ──
        # 默认：水平裁剪（横屏场景），保留全高，按输出比例算宽度
        crop_w = H * output_width / output_height
        vertical_mode = False  # 是否垂直裁剪（输入比目标窄时）
        if crop_w > W:
            # 输入视频比目标输出更窄（例如 400×864 → 480×864）
            # 改为垂直居中裁剪：使用全宽，按比例算高度
            crop_h = W * output_height / output_width
            crop_w = float(W)
            vertical_mode = True
            logger.info(f"输入比目标窄，切换为垂直居中裁剪: crop_h={crop_h:.0f}, crop_w={crop_w:.0f}")

        # ── 遍历帧，计算每帧的裁剪窗口 ──
        # 垂直裁剪：居中裁剪，不需要质心追踪
        # 水平裁剪：需要计算每帧人物质心，跟随平移
        smooth_cx_ratio: Optional[float] = None
        prev_cx_ratio: Optional[float] = None
        prev_has_mask: bool = False
        ever_had_mask: bool = False

        # 每帧的 (crop_x, crop_y, crop_w_actual, crop_h_actual)
        crop_params_per_frame = []

        if vertical_mode:
            # 垂直居中裁剪，所有帧位置一致
            crop_y_px = (H - crop_h) / 2.0
            for frame_idx in range(B):
                crop_params_per_frame.append({
                    "x": 0.0,
                    "y": crop_y_px,
                    "w": crop_w,
                    "h": crop_h,
                })
        else:
            for frame_idx in range(B):
                mask_idx = frame_idx if frame_idx < T_total else (T_total - 1)
                mask = self._get_combined_mask(masks_bool, ids, mask_idx)
                has_mask = mask is not None

                if has_mask:
                    cx_ratio = self._compute_centroid(mask)
                else:
                    cx_ratio = None

                is_scene_cut = False
                if cx_ratio is None:
                    if ever_had_mask and mask_fallback == "hold_last" and smooth_cx_ratio is not None:
                        used_cx = smooth_cx_ratio
                    else:
                        used_cx = fallback_center
                else:
                    is_scene_cut = (
                        scene_cut_threshold > 0
                        and prev_has_mask
                        and prev_cx_ratio is not None
                        and abs(cx_ratio - prev_cx_ratio) > scene_cut_threshold
                    )
                    used_cx = cx_ratio
                    prev_cx_ratio = cx_ratio
                    ever_had_mask = True

                if smooth_cx_ratio is None or is_scene_cut:
                    smooth_cx_ratio = used_cx
                else:
                    smooth_cx_ratio = smoothing * smooth_cx_ratio + (1 - smoothing) * used_cx

                crop_x_px = smooth_cx_ratio * W - crop_w / 2.0
                crop_x_px = max(0.0, min(crop_x_px, W - crop_w))
                crop_params_per_frame.append({
                    "x": crop_x_px,
                    "y": 0.0,
                    "w": crop_w,
                    "h": float(H),
                })

                prev_has_mask = has_mask

        # ── 执行裁剪 ──
        def _do_crop(src_frames: torch.Tensor, params_list: list) -> torch.Tensor:
            out_list = []
            for idx in range(min(len(params_list), src_frames.shape[0])):
                p = params_list[idx]
                f = src_frames[idx:idx + 1].to(device)
                # 如果裁剪框几乎覆盖全帧 → 直接 resize
                if p["w"] >= W - 1e-6 and p["h"] >= H - 1e-6:
                    cropped = comfy.utils.common_upscale(
                        f.movedim(-1, 1), output_width, output_height, "bicubic", "center"
                    ).movedim(1, -1)
                else:
                    x_start = 2.0 * p["x"] / max(W - 1, 1) - 1.0
                    x_end = 2.0 * (p["x"] + p["w"]) / max(W - 1, 1) - 1.0
                    y_start = 2.0 * p["y"] / max(H - 1, 1) - 1.0
                    y_end = 2.0 * (p["y"] + p["h"]) / max(H - 1, 1) - 1.0
                    out_h_int = int(round(p["h"]))
                    out_w_int = int(round(p["w"]))
                    grid_y = torch.linspace(y_start, y_end, out_h_int, device=device)
                    grid_x = torch.linspace(x_start, x_end, out_w_int, device=device)
                    gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
                    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                    sampled = F.grid_sample(
                        f.permute(0, 3, 1, 2).float(),
                        grid.expand(1, -1, -1, -1),
                        mode='bicubic', padding_mode='border', align_corners=True
                    ).permute(0, 2, 3, 1)
                    if out_w_int != output_width or out_h_int != output_height:
                        cropped = comfy.utils.common_upscale(
                            sampled.movedim(-1, 1), output_width, output_height, "bicubic", "center"
                        ).movedim(1, -1)
                    else:
                        cropped = sampled
                out_list.append(cropped.cpu())
            return torch.cat(out_list, dim=0)

        result_frames = _do_crop(frames, crop_params_per_frame)

        # ── 裁剪第二路 ──
        if mask_frames is not None and mask_frames.shape[0] > 0:
            result_mask = _do_crop(mask_frames, crop_params_per_frame)
        else:
            result_mask = torch.zeros(1, output_height, output_width, 3)

        # ── debug info ──
        import json
        debug_records = []
        for i in range(B):
            p = crop_params_per_frame[i] if i < len(crop_params_per_frame) else None
            debug_records.append({
                "f": i,
                "crop_x": round(p["x"], 1) if p else 0,
                "crop_y": round(p["y"], 1) if p else 0,
            })
        debug_info = json.dumps({
            "meta": {
                "frame_count": B,
                "mask_frames": T_total,
                "orig_size": [H, W],
                "output_size": [output_height, output_width],
                "crop_width": round(crop_w, 1),
                "vertical_mode": vertical_mode,
                "has_mask_input": mask_frames is not None,
            },
            "frames": debug_records,
        }, indent=2)

        logger.info(f"SmartVideoCrop 完成: {B} 帧 → {result_frames.shape}")
        return (result_frames, result_mask, debug_info)

    def _fallback_crop(self, frames: torch.Tensor, crop_x: float, crop_w: float,
                       output_width: int, output_height: int,
                       device: torch.device) -> torch.Tensor:
        """回退：居中裁剪 + 缩放。"""
        if frames is None or frames.shape[0] == 0:
            return torch.zeros(1, output_height, output_width, 3)
        B, H, W, C = frames.shape
        crop_x = max(0.0, min(crop_x, W - crop_w))
        output_frames = []
        for frame_idx in range(B):
            frame = frames[frame_idx:frame_idx + 1].to(device)
            x_start = 2.0 * crop_x / max(W - 1, 1) - 1.0
            x_end = 2.0 * (crop_x + crop_w) / max(W - 1, 1) - 1.0
            out_h_int = int(round(H))
            out_w_int = int(round(crop_w))
            grid_y = torch.linspace(-1.0, 1.0, out_h_int, device=device)
            grid_x = torch.linspace(x_start, x_end, out_w_int, device=device)
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(
                frame.permute(0, 3, 1, 2).float(),
                grid.expand(1, -1, -1, -1),
                mode='bicubic', padding_mode='border', align_corners=True
            ).permute(0, 2, 3, 1)
            if out_w_int != output_width or out_h_int != output_height:
                cropped = comfy.utils.common_upscale(
                    sampled.movedim(-1, 1), output_width, output_height, "bicubic", "center"
                ).movedim(1, -1)
            else:
                cropped = sampled
            output_frames.append(cropped.cpu())
        return torch.cat(output_frames, dim=0)


NODE_CLASS_MAPPINGS = {
    "SmartVideoCrop": SmartVideoCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartVideoCrop": "智能视频裁剪 (Smart Video Crop)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]