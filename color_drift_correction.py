"""
AutoColorDriftCorrection V4.1 - 接缝对齐 + 段内趋势线性补偿

核心策略：
1. 接缝对齐（seam_strength）：修正段间跳变 —— 基础校正
2. 段内趋势校正（drift_trend_strength）：基于当前帧序列RGB斜率做线性反向补偿
3. 完全自适应，不依赖任何预设模板或跨段学习

V4.0 改进（基于真实数据实测验证）：
- 移除固定隆起模板（_BUMP_PARAMS），该模板基于特定参考图标定，不通用
- 移除自适应学习（bump_learning_rate），换参考图后学习结果不再适用
- auto 模式新增分通道独立处理：R通道漂移方向固定（↓），G/B随参考图变化
- 接缝淡入帧数自适应（fade_frames = min(overlap_count, 5)），大重叠更平滑
- standard 模式下也可使用斜率线性补偿（不再依赖预设模板）
- 简化参数，移除 bump_learning_rate，合并为 drift_trend_strength

V4.1 修复：
- 修复 auto 模式跳变检测：有 prev_frames 时改在接缝处对比（overlap_frames[-1] vs new_frames[0]），
  不再用多帧平均稀释跳变值（prev_frames[-5] vs overlap_frames 是相同内容，diff≈0）
- 修复接缝对齐计算：同样改为接缝处对比（overlap_frames[-fade_frames:] vs new_frames[:fade_frames]）
- 修复有 prev_frames 时 seam_strength 参数被忽略的问题
- 调高 max_offset 默认值 0.01→0.02，上限 0.02→0.05
"""

import torch
import json


def _detect_drift_slope(frame_means):
    """
    检测帧序列的漂移斜率（线性回归）。
    frame_means: (N, 3) tensor, 每帧的RGB均值
    返回: (slope_r, slope_g, slope_b), 每帧的平均变化量
    """
    N = frame_means.shape[0]
    if N < 3:
        return (0.0, 0.0, 0.0)
    x = torch.arange(N, dtype=torch.float32, device=frame_means.device)
    x_mean = x.mean()
    y_mean = frame_means.mean(dim=0)
    numerator = ((x - x_mean).unsqueeze(1) * (frame_means - y_mean.unsqueeze(0))).sum(dim=0)
    denominator = ((x - x_mean) ** 2).sum()
    if denominator < 1e-8:
        return (0.0, 0.0, 0.0)
    slopes = numerator / denominator
    return (float(slopes[0]), float(slopes[1]), float(slopes[2]))


class AutoColorDriftCorrection:
    OUTPUT_NODE = False

    def __init__(self):
        pass  # V4.0: 无实例级状态，完全无状态节点

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "当前段生成的全部帧序列 (B, H, W, C)。"
                }),
                "mode": (["standard", "auto", "off"], {
                    "default": "standard",
                    "tooltip": "standard=手动参数控制; auto=自动检测漂移后决策; off=完全旁路"
                }),
                "overlap_count": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "重叠帧数量。前 overlap_count 帧保持不动。"
                }),
            },
            "optional": {
                "prev_frames": ("IMAGE", {
                    "tooltip": "（可选）上一段尾部重叠帧。接入后做精确段间对齐。"
                }),
                "seam_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "接缝对齐强度（基础校正）。0=禁用。auto模式下基于跳变幅度自动决策。"
                }),
                "drift_trend_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "段内趋势校正强度。基于检测到的RGB漂移斜率做线性反向补偿。0=禁用。"
                }),
                "max_offset": ("FLOAT", {
                    "default": 0.02, "min": 0.001, "max": 0.05, "step": 0.001,
                    "tooltip": "最大单通道偏移钳制值。0.02≈5.1像素值。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "V4.1: 接缝对齐 + 段内趋势自适应线性补偿。修复 auto 模式跳变检测帧索引问题。"

    def correct(self, images, mode="standard", overlap_count=5,
                prev_frames=None, seam_strength=0.0, drift_trend_strength=0.0,
                max_offset=0.02):
        info_lines = []
        B, H, W, C = images.shape
        info_lines.append(f"V4.1 输入帧数: {B}, 重叠帧数: {overlap_count}, 模式: {mode}")

        # ========== off 模式：完全旁路 ==========
        if mode == "off":
            info_lines.append("模式: off，跳过所有处理")
            drift_data = {
                "mode": "off",
                "v4_mode": "bypass",
                "frames_total": B,
            }
            info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
            return (images, info)

        if C != 3:
            info_lines.append(f"警告: 通道数={C}，期望3")
            return (images, "\n".join(info_lines))
        if B <= overlap_count:
            info_lines.append(f"警告: 帧数({B}) <= 重叠帧数({overlap_count})，无新生成帧")
            return (images, "\n".join(info_lines))

        new_frames = images[overlap_count:]  # (N, H, W, 3)
        overlap_frames = images[:overlap_count]  # (overlap, H, W, 3)
        N = new_frames.shape[0]
        # 自适应淡入帧数：重叠数越大，淡入越平滑
        fade_frames = min(overlap_count, 5)

        # ========== 检测段内漂移斜率（线性回归） ==========
        # 计算新帧序列的 RGB 均值趋势
        new_means = new_frames.mean(dim=(1, 2))  # (N, 3)
        slopes = _detect_drift_slope(new_means)
        max_slope = max(abs(s) for s in slopes)
        info_lines.append(f"段内漂移斜率: R={slopes[0]:+.8f}/帧 G={slopes[1]:+.8f}/帧 B={slopes[2]:+.8f}/帧 max_abs={max_slope:.8f}")

        # ========== auto 模式：自动决策 ==========
        if mode == "auto":
            # --- 1. 段间跳变检测（接缝处精确检测） ---
            # V4.1: 统一在接缝处检测，不依赖多帧平均也不依赖 prev_frames 配对
            # 数据布局：overlap_frames 是图像帧拷贝 = prev_frames[-overlap_count:]
            # 跳变位置在 overlap_frames[-1] → new_frames[0] 之间
            last_overlap = overlap_frames[-1:]   # 重叠区域最后一帧
            first_new = new_frames[:1]           # 新帧第一帧
            ref_mean = last_overlap.mean(dim=(0, 1, 2))
            curr_mean = first_new.mean(dim=(0, 1, 2))
            jump_vec = curr_mean - ref_mean
            info_lines.append("段间跳变检测(接缝处): 重叠末帧 vs 新帧首帧")
            info_lines.append(f"  重叠末帧均值: R={ref_mean[0]:.4f} G={ref_mean[1]:.4f} B={ref_mean[2]:.4f}")
            info_lines.append(f"  新帧首帧均值: R={curr_mean[0]:.4f} G={curr_mean[1]:.4f} B={curr_mean[2]:.4f}")

            local_jump = max(abs(float(jump_vec[i])) for i in range(3))
            info_lines.append(f"  跳变量: R={float(jump_vec[0]):+.6f} G={float(jump_vec[1]):+.6f} B={float(jump_vec[2]):+.6f} max={local_jump:.6f}")

            jump_threshold = 0.0005
            has_jump = (local_jump > jump_threshold)
            info_lines.append(f"跳变判定: {'有跳变' if has_jump else '无跳变'} (阈值={jump_threshold})")

            # 自动接缝强度：有跳变时按幅度折算
            if has_jump:
                auto_seam = min(1.0, max(0.3, local_jump * 200))
                seam_strength = auto_seam
                info_lines.append(f"auto: 跳变={local_jump:.6f}, 自动接缝强度={seam_strength:.2f}")
            else:
                seam_strength = 0.0
                info_lines.append(f"auto: 无显著跳变, seam_strength=0（跳过接缝对齐）")

            # --- 2. 段内趋势强度：基于斜率自动决策 ---
            if max_slope > 0.00003:
                auto_trend = min(0.8, max(0.2, max_slope * 4000))
                drift_trend_strength = auto_trend
                info_lines.append(f"auto: 段内漂移明显({max_slope:.6f}), 自动趋势校正强度={drift_trend_strength:.2f}")
            else:
                drift_trend_strength = 0.0
                info_lines.append(f"auto: 段内漂移微弱({max_slope:.6f}), 趋势校正=0")

        # ========== 1. 接缝对齐计算 ==========
        has_prev = prev_frames is not None
        base_offset = torch.zeros(3, device=images.device)
        ref_type = "none"

        if has_prev:
            if prev_frames.shape[0] >= overlap_count:
                # V4.1: 接缝对齐在接缝处精确计算
                # 用 fade_frames 帧做小范围均值，比单帧更抗噪，又不会像 overlap_count 帧那样稀释跳变
                prev_tail = overlap_frames[-fade_frames:]
                curr_head = new_frames[:fade_frames]
                prev_mean = prev_tail.mean(dim=(0, 1, 2))
                curr_mean = curr_head.mean(dim=(0, 1, 2))
                base_offset = curr_mean - prev_mean
                ref_type = "prev_frame"
                info_lines.append(f"基准: prev_frames (重叠尾{fade_frames}帧 vs 新帧头{fade_frames}帧)")
                info_lines.append(f"重叠尾均值: R={prev_mean[0]:.4f} G={prev_mean[1]:.4f} B={prev_mean[2]:.4f}")
                info_lines.append(f"新帧头均值: R={curr_mean[0]:.4f} G={curr_mean[1]:.4f} B={curr_mean[2]:.4f}")
                info_lines.append(f"段间偏移: R={base_offset[0]:+.6f} G={base_offset[1]:+.6f} B={base_offset[2]:+.6f}")
            else:
                info_lines.append(f"prev_frames不足({prev_frames.shape[0]}<{overlap_count})，跳过")

        elif seam_strength > 0:
            last_overlap = overlap_frames[-1:]
            first_new = new_frames[:1]
            last_mean = last_overlap.mean(dim=(0, 1, 2))
            first_mean = first_new.mean(dim=(0, 1, 2))
            base_offset = first_mean - last_mean
            ref_type = "seam_align"
            info_lines.append("基准: 接缝对齐 (重叠末帧 vs 新帧首帧)")
            info_lines.append(f"重叠末帧均值: R={last_mean[0]:.4f} G={last_mean[1]:.4f} B={last_mean[2]:.4f}")
            info_lines.append(f"新帧首帧均值: R={first_mean[0]:.4f} G={first_mean[1]:.4f} B={first_mean[2]:.4f}")
            info_lines.append(f"接缝偏移: R={base_offset[0]:+.6f} G={base_offset[1]:+.6f} B={base_offset[2]:+.6f}")

        trend_enabled = (drift_trend_strength > 0)

        if ref_type == "none" and not trend_enabled:
            info_lines.append("基准: 无对齐 (输出原始帧)")
            return (images, "\n".join(info_lines))

        # ========== 2. 钳制接缝偏移 ==========
        if ref_type != "none":
            # 分通道独立钳制
            base_clamped = torch.clamp(base_offset, -max_offset, max_offset)
            # V4.1 FIX: 统一乘以 seam_strength，不再区分 has_prev 分支
            base_applied = base_clamped * seam_strength
            info_lines.append(f"钳制后偏移(max_offset={max_offset}): R={base_applied[0]:+.6f} G={base_applied[1]:+.6f} B={base_applied[2]:+.6f}")
        else:
            base_applied = torch.zeros(3, device=images.device)
            fade_frames = 0

        # ========== 3. 应用校正 ==========
        result = images.clone()
        frames_to_correct = result[overlap_count:]

        # 接缝对齐（基础校正）
        if base_applied.abs().sum() > 1e-8:
            if fade_frames > 0:
                fade_weights = torch.arange(1, fade_frames + 1, device=images.device, dtype=images.dtype) / fade_frames
                fade_weights = fade_weights.view(-1, 1, 1, 1)
                for i in range(min(fade_frames, N)):
                    frames_to_correct[i] -= base_applied.view(1, 1, 3) * fade_weights[i]
                if N > fade_frames:
                    frames_to_correct[fade_frames:] -= base_applied.view(1, 1, 3)
                info_lines.append(f"接缝校正: 前{fade_frames}帧淡入 + {N-fade_frames}帧完整应用")

        # ========== 段内趋势校正（线性补偿） ==========
        if trend_enabled:
            # 用段内斜率生成线性补偿曲线（逐帧增长的反向补偿）
            t = torch.arange(N, device=images.device, dtype=images.dtype).view(-1, 1)  # (N, 1)
            slope_tensor = torch.tensor([[slopes[0]], [slopes[1]], [slopes[2]]], 
                                        device=images.device, dtype=images.dtype).view(1, 3)  # (1, 3)
            # 线性补偿 = t * 斜率 * 强度，方向与漂移相反
            linear_compensation = t * slope_tensor * drift_trend_strength  # (N, 3)
            for i in range(N):
                frames_to_correct[i] -= linear_compensation[i].view(1, 1, 3)
            peak_val = max(abs(slopes[i] * N * drift_trend_strength) for i in range(3))
            info_lines.append(f"趋势校正: 斜率=[{slopes[0]:+.8f},{slopes[1]:+.8f},{slopes[2]:+.8f}] "
                             f"强度={drift_trend_strength:.2f} 末端补偿峰值为{peak_val:.4f}")

        frames_to_correct = torch.clamp(frames_to_correct, 0.0, 1.0)
        result[overlap_count:] = frames_to_correct

        info_lines.append(f"校正帧: [{overlap_count+1}-{B}] ({N}帧)")

        # ========== 输出校正信息 ==========
        trend_peak = max(abs(slopes[i] * N) for i in range(3)) if trend_enabled else 0.0
        drift_data = {
            "mode": mode,
            "v4_mode": ref_type if ref_type != "none" else "trend_only",
            "seam_strength": seam_strength if ref_type != "none" else 0,
            "drift_trend_strength": drift_trend_strength if trend_enabled else 0,
            "frames_corrected": N,
            "drift_slope_rgb": [slopes[0], slopes[1], slopes[2]],
            "drift_peak_rgb": [slopes[0] * N, slopes[1] * N, slopes[2] * N],
        }
        if ref_type != "none":
            drift_data["offset_rgb"] = [float(base_offset[i]) for i in range(3)]
        if base_applied.abs().sum() > 1e-8:
            drift_data["offset_clamped_rgb"] = [float(base_applied[i]) for i in range(3)]
            drift_data["fade_frames"] = fade_frames
        if trend_peak > 0:
            drift_data["trend_peak"] = round(trend_peak, 6)

        info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
        return (result, info)


NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction V4.1 (自适应)",
}