"""
AutoColorDriftCorrection - 自动色彩漂移校正节点

原理：对接续生成中每段的新生成帧做逐帧线性回归，检测
RGB三通道的漂移趋势，然后对本段所有新生成帧做精确的
逐帧校正。前 overlap_count 帧（重叠帧）保持不动。

优势：
- 线性回归 + 逐帧校正 → 更精准，非头尾近似
- 全局统一参数 → 无帧间闪烁
- PyTorch 直算 → GPU毫秒级处理
- 重叠帧不动 → 无接缝感

使用方式：
1) 接 prev_frames → 以上一段末尾帧为基准
2) 不接 prev_frames + reference_mode=segment_self → 段内自参考
3) 不接 prev_frames + reference_mode=first_segment_cache →
   缓存第一段为固定基准（推荐，不需要额外接线）
"""

import torch
import json

# 类变量缓存：存储第一段尾帧均值作为固定基准
_first_segment_cache = None


def _linear_regression(frame_means):
    """对帧均值序列做最小二乘法线性回归
    
    Args:
        frame_means: (N, 3) tensor, N帧的RGB均值
    
    Returns:
        slopes: (3,) 每帧漂移速率
        intercepts: (3,) 截距（第0帧的回归值）
    """
    N = frame_means.shape[0]
    if N <= 1:
        return torch.zeros(3, device=frame_means.device), frame_means[0] if N == 1 else torch.zeros(3, device=frame_means.device)
    
    # x = [0, 1, 2, ..., N-1]
    x = torch.arange(N, device=frame_means.device, dtype=frame_means.dtype)
    x_mean = x.mean()
    y_mean = frame_means.mean(dim=0)  # (3,)
    
    # 最小二乘: slope = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)^2)
    x_centered = x - x_mean
    y_centered = frame_means - y_mean  # (N, 3)
    
    numerator = (x_centered.view(-1, 1) * y_centered).sum(dim=0)  # (3,)
    denominator = (x_centered ** 2).sum()
    
    slopes = numerator / denominator if denominator > 0 else torch.zeros(3, device=frame_means.device)
    intercepts = y_mean - slopes * x_mean
    
    return slopes, intercepts


class AutoColorDriftCorrection:
    """自动色彩漂移校正节点"""

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
                    "tooltip": "重叠帧数量，默认5。前 overlap_count 帧保持不动，从第 overlap_count+1 帧开始校正。"
                }),
            },
            "optional": {
                "prev_frames": ("IMAGE", {
                    "tooltip": "（可选）上一段末尾的重叠帧。接入后以上一段为基准。"
                }),
                "reference_mode": (["segment_self", "first_segment_cache"], {
                    "default": "segment_self",
                    "tooltip": "不接prev_frames时的参考基准：segment_self=段内自参考；first_segment_cache=缓存第一段为固定基准(推荐)。"
                }),
                "correction_mode": (["subtract", "scale"], {
                    "default": "subtract",
                    "tooltip": "subtract=减法校正；scale=缩放校正(暗部影响小)。"
                }),
                "max_correction": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "单通道最大校正幅度。0.15=最多校正19.2像素值。"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "校正强度。1.0=完整校正，0.5=一半。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "利用线性回归精确检测色彩漂移趋势并做逐帧校正，重叠帧保持不动。"

    def correct(self, images, overlap_count=5,
                prev_frames=None, reference_mode="segment_self",
                correction_mode="subtract",
                max_correction=0.15, strength=1.0):
        global _first_segment_cache

        info_lines = []
        B, H, W, C = images.shape
        info_lines.append(f"输入帧数: {B}, 重叠帧数: {overlap_count}")

        # ==================== 安全检查 ====================
        if C != 3:
            info_lines.append(f"警告: 通道数={C}，期望3")
            return (images, "\n".join(info_lines))
        if B <= overlap_count:
            info_lines.append(f"警告: 帧数({B}) <= 重叠帧数({overlap_count})，无新生成帧")
            return (images, "\n".join(info_lines))

        # ==================== 对新生成帧做线性回归 ====================
        # 新生成帧 = images[overlap_count:]
        new_frames = images[overlap_count:]  # (N, H, W, C), N = B - overlap_count
        N = new_frames.shape[0]
        
        # 计算每帧的RGB均值
        frame_means = new_frames.mean(dim=(1, 2))  # (N, C)
        
        # 线性回归
        slopes, intercepts = _linear_regression(frame_means)
        
        info_lines.append(f"新生成帧数: {N}")
        info_lines.append(f"回归斜率(每帧): R={slopes[0]:+.4f} G={slopes[1]:+.4f} B={slopes[2]:+.4f}")
        info_lines.append(f"回归截距(首帧): R={intercepts[0]:.2f} G={intercepts[1]:.2f} B={intercepts[2]:.2f}")

        # ==================== 计算基准偏移 ====================
        has_prev_frames = prev_frames is not None
        # 本段首帧新生成帧的回归值（即新生成帧第0帧的预期颜色）
        # 校正的目标是让本段首帧新生成帧的颜色 ≈ 基准色
        new_start_mean = intercepts  # 回归截距 = 新生成帧第0帧的预期值
        
        if has_prev_frames:
            # --- 模式1: 以上一段尾帧为基准 ---
            if prev_frames.shape[0] < overlap_count:
                info_lines.append(f"警告: prev_frames({prev_frames.shape[0]}) < overlap({overlap_count})")
                return (images, "\n".join(info_lines))
            prev_tail = prev_frames[-overlap_count:]
            ref_mean = prev_tail.mean(dim=(0, 1, 2))
            ref_type = "prev_frames"
            info_lines.append("基准: 上一段尾帧")
        elif reference_mode == "first_segment_cache":
            # --- 模式2: 缓存第一段尾帧为固定基准 ---
            if _first_segment_cache is None:
                # 第一段：缓存尾帧，跳过校正
                first_tail = images[-overlap_count:] if B >= overlap_count * 2 else new_frames[:overlap_count]
                _first_segment_cache = first_tail.mean(dim=(0, 1, 2)).clone()
                info_lines.append("第一段初始化: 缓存尾帧为基准")
                info_lines.append(f"缓存: R={_first_segment_cache[0]:.2f} G={_first_segment_cache[1]:.2f} B={_first_segment_cache[2]:.2f}")
                info_lines.append("第一段跳过校正")
                return (images, "\n".join(info_lines))
            ref_mean = _first_segment_cache
            ref_type = "first_segment_cache"
            info_lines.append("基准: 第一段缓存")
        else:
            # --- 模式3: 段内自参考 ---
            # 基准 = 回归截距本身，delta=0只靠斜率校正
            ref_mean = new_start_mean
            ref_type = "segment_self"
            info_lines.append("基准: 段内自参考(仅斜率)")

        # ==================== 计算偏移量 ====================
        if ref_type == "segment_self":
            # 段内自参考：只校正段内趋势，不拉基准
            base_offset = torch.zeros(3, device=images.device)
        else:
            # 其他模式：校正新生成帧首帧与基准的差距
            base_offset = new_start_mean - ref_mean  # 正=偏蓝/亮，需要减去

        # 总漂移: 基准偏移 + 逐帧趋势
        # 对第 t 帧(从0开始)：drift[t] = base_offset + slopes * t
        # 第0帧的漂移 = base_offset，最后一帧的漂移 = base_offset + slopes * (N-1)

        info_lines.append(f"基准偏移: R={base_offset[0]:+.2f} G={base_offset[1]:+.2f} B={base_offset[2]:+.2f}")
        total_drift_end = base_offset + slopes * (N - 1)
        info_lines.append(f"尾帧总偏移: R={total_drift_end[0]:+.2f} G={total_drift_end[1]:+.2f} B={total_drift_end[2]:+.2f}")

        # ==================== 钳制 ====================
        max_abs_delta = max_correction * 128.0
        # 钳制基准偏移
        base_clamped = torch.clamp(base_offset, -max_abs_delta, max_abs_delta)
        # 钳制斜率（基于最大帧数的总漂移量）
        max_slope_delta = max_abs_delta / max(N, 1) * 2  # 允许斜率总漂移2倍于基准
        slopes_clamped = torch.clamp(slopes, -max_slope_delta, max_slope_delta)
        
        base_applied = base_clamped * strength
        slopes_applied = slopes_clamped * strength

        info_lines.append(f"基准偏移(钳制后): R={base_applied[0]:+.2f} G={base_applied[1]:+.2f} B={base_applied[2]:+.2f}")
        info_lines.append(f"斜率(钳制后): R={slopes_applied[0]:+.4f} G={slopes_applied[1]:+.4f} B={slopes_applied[2]:+.4f}")

        # ==================== 应用校正 ====================
        result = images.clone()
        # 前 overlap_count 帧不动
        frames_to_correct = result[overlap_count:]  # (N, H, W, C)
        
        # 对每帧生成精确的漂移量
        # t = [0, 1, 2, ..., N-1]
        t = torch.arange(N, device=images.device, dtype=images.dtype)
        # drift_per_frame[t] = base_applied + slopes_applied * t
        drift_per_frame = base_applied.view(1, 3) + slopes_applied.view(1, 3) * t.view(-1, 1)  # (N, 3)
        drift_per_frame = drift_per_frame.view(N, 1, 1, 3)  # (N, 1, 1, 3)
        
        if correction_mode == "scale":
            # 缩放模式: pixel *= (1 - drift/128)
            scale_factors = 1.0 - drift_per_frame / 128.0
            corrected = frames_to_correct * scale_factors
        else:
            # 减法模式: pixel -= drift
            corrected = frames_to_correct - drift_per_frame
        
        corrected = torch.clamp(corrected, 0.0, 1.0)
        result[overlap_count:] = corrected
        
        br_ratio_drift = 0.0
        if ref_type != "segment_self" and ref_mean[0] > 0:
            br_ratio_drift = (new_start_mean[2]/new_start_mean[0] - ref_mean[2]/ref_mean[0]) * 100

        info_lines.append(f"校正帧: [{overlap_count+1}-{B}] ({N}帧)")
        info_lines.append(f"B/R比率偏移: {br_ratio_drift:+.2f}%")

        # ==================== 输出 ====================
        drift_data = {
            "ref_mode": ref_type,
            "correction_type": correction_mode,
            "slope_r": float(slopes[0]),
            "slope_g": float(slopes[1]),
            "slope_b": float(slopes[2]),
            "base_offset_r": float(base_offset[0]),
            "base_offset_g": float(base_offset[1]),
            "base_offset_b": float(base_offset[2]),
            "total_drift_r": float(total_drift_end[0]),
            "total_drift_g": float(total_drift_end[1]),
            "total_drift_b": float(total_drift_end[2]),
            "overlap_count": overlap_count,
            "strength": strength,
            "frames_corrected": N,
            "max_correction_pct": max_correction * 100,
        }
        drift_json = json.dumps(drift_data, indent=2)
        info = "\n".join(info_lines) + f"\n\nDrift JSON:\n{drift_json}"

        return (result, info)


def reset_first_segment_cache():
    """重置第一段缓存（调试用）"""
    global _first_segment_cache
    _first_segment_cache = None


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction (自动色彩漂移校正)",
}