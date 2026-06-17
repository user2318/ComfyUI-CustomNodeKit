"""
AutoColorDriftCorrection V2 - 自对齐/prev对齐模式

核心：放弃reference_folder锚点（输出文件夹会被校正后的帧覆盖，锚点不断漂移）。

对齐方案：
1. prev_frames → 用上一段尾重叠帧做精确对齐（最准，但需连线）
2. 兜底(self_overlap) → 重叠帧均值 vs 新生成帧整体均值修正段间偏移
3. 保留段内斜率校正

两种方案互相独立、逻辑一致。
"""

import torch
import json


def _linear_regression(frame_means):
    N = frame_means.shape[0]
    if N <= 1:
        return torch.zeros(3, device=frame_means.device), frame_means[0] if N == 1 else torch.zeros(3, device=frame_means.device)
    x = torch.arange(N, device=frame_means.device, dtype=frame_means.dtype)
    x_mean = x.mean()
    y_mean = frame_means.mean(dim=0)
    x_centered = x - x_mean
    y_centered = frame_means - y_mean
    numerator = (x_centered.view(-1, 1) * y_centered).sum(dim=0)
    denominator = (x_centered ** 2).sum()
    slopes = numerator / denominator if denominator > 0 else torch.zeros(3, device=frame_means.device)
    intercepts = y_mean - slopes * x_mean
    return slopes, intercepts


class AutoColorDriftCorrection:
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "当前段生成的全部帧序列 (B, H, W, C)。"
                }),
                "overlap_count": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "重叠帧数量。前 overlap_count 帧保持不动。"
                }),
            },
            "optional": {
                "prev_frames": ("IMAGE", {
                    "tooltip": "（可选）上一段尾重叠帧。接入后可精确消除段间漂移。"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "校正强度。1.0=完整校正。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "V2: self_overlap + prev_frames 对齐。不依赖外部文件，利用重叠帧消除段间漂移。"

    def correct(self, images, overlap_count=5,
                prev_frames=None, strength=1.0):
        info_lines = []
        B, H, W, C = images.shape
        info_lines.append(f"V2 输入帧数: {B}, 重叠帧数: {overlap_count}")

        if C != 3:
            info_lines.append(f"警告: 通道数={C}，期望3")
            return (images, "\n".join(info_lines))
        if B <= overlap_count:
            info_lines.append(f"警告: 帧数({B}) <= 重叠帧数({overlap_count})，无新生成帧")
            return (images, "\n".join(info_lines))

        new_frames = images[overlap_count:]  # 新生成帧（第6-81帧）
        overlap_frames = images[:overlap_count]  # 重叠帧（第1-5帧）
        N = new_frames.shape[0]

        # ========== 段内回归 ==========
        frame_means = new_frames.mean(dim=(1, 2))
        slopes, intercepts = _linear_regression(frame_means)
        info_lines.append(f"新生成帧数: {N}")
        info_lines.append(f"段内斜率: R={slopes[0]:+.4f} G={slopes[1]:+.4f} B={slopes[2]:+.4f}")

        # ========== 基准对齐 ==========
        has_prev = prev_frames is not None

        if has_prev:
            # --- 方案A: prev_frames 精确对齐 ---
            if prev_frames.shape[0] >= overlap_count:
                prev_mean = prev_frames[-overlap_count:].mean(dim=(0, 1, 2))
                new_mean = new_frames.mean(dim=(0, 1, 2))
                base_offset = new_mean - prev_mean
                ref_type = "prev_frame"
                info_lines.append(f"基准: prev_frames (上一段尾{overlap_count}帧)")
                info_lines.append(f"prev均值: R={prev_mean[0]:.4f} G={prev_mean[1]:.4f} B={prev_mean[2]:.4f}")
                info_lines.append(f"新生成整体: R={new_mean[0]:.4f} G={new_mean[1]:.4f} B={new_mean[2]:.4f}")
                info_lines.append(f"偏移: R={base_offset[0]:+.4f} G={base_offset[1]:+.4f} B={base_offset[2]:+.4f}")
            else:
                # prev_frames不足，降级self_overlap
                has_prev = False

        if not has_prev:
            # --- 方案B: self_overlap 兜底 ---
            ov_mean = overlap_frames.mean(dim=(0, 1, 2))
            new_mean = new_frames.mean(dim=(0, 1, 2))
            base_offset = new_mean - ov_mean
            ref_type = "self_overlap"
            info_lines.append(f"基准: self_overlap (本段前{overlap_count}帧重叠帧)")
            info_lines.append(f"重叠帧均值: R={ov_mean[0]:.4f} G={ov_mean[1]:.4f} B={ov_mean[2]:.4f}")
            info_lines.append(f"新生成整体: R={new_mean[0]:.4f} G={new_mean[1]:.4f} B={new_mean[2]:.4f}")
            info_lines.append(f"偏移: R={base_offset[0]:+.4f} G={base_offset[1]:+.4f} B={base_offset[2]:+.4f}")

        info_lines.append(f"基准偏移: R={base_offset[0]:+.4f} G={base_offset[1]:+.4f} B={base_offset[2]:+.4f}")

        # ========== 钳制 + 应用 ==========
        # 钳制到单通道最大0.15 * 128 ≈ 19.2 像素值
        max_abs_delta = 0.15 * 128.0
        base_clamped = torch.clamp(base_offset, -max_abs_delta, max_abs_delta)
        slopes_clamped = torch.clamp(slopes, -max_abs_delta / max(N, 1) * 2, max_abs_delta / max(N, 1) * 2)

        base_applied = base_clamped * strength
        slopes_applied = slopes_clamped * strength

        info_lines.append(f"基准偏移(钳制后): R={base_applied[0]:+.4f} G={base_applied[1]:+.4f} B={base_applied[2]:+.4f}")
        info_lines.append(f"斜率(钳制后): R={slopes_applied[0]:+.4f} G={slopes_applied[1]:+.4f} B={slopes_applied[2]:+.4f}")

        result = images.clone()
        frames_to_correct = result[overlap_count:]
        t = torch.arange(N, device=images.device, dtype=images.dtype)
        drift_per_frame = base_applied.view(1, 3) + slopes_applied.view(1, 3) * t.view(-1, 1)
        drift_per_frame = drift_per_frame.view(N, 1, 1, 3)

        corrected = frames_to_correct - drift_per_frame
        corrected = torch.clamp(corrected, 0.0, 1.0)
        result[overlap_count:] = corrected
        info_lines.append(f"校正帧: [{overlap_count+1}-{B}] ({N}帧)")

        drift_data = {
            "v2_mode": ref_type,
            "slope_rgb": [float(slopes[i]) for i in range(3)],
            "offset_rgb": [float(base_offset[i]) for i in range(3)],
            "offset_clamped_rgb": [float(base_applied[i]) for i in range(3)],
            "frames_corrected": N,
        }
        info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
        return (result, info)


NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction V2 (自对齐)",
