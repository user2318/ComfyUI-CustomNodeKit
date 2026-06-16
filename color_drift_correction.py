"""
AutoColorDriftCorrection - 自动色彩漂移校正节点

原理：利用接续生成中重叠帧（前5帧与上一段末尾5帧相同）的差异，
计算出本段发生的色彩偏移量，然后对本段所有新生成帧做全局统一的校正。

优势：
- 全局统一校正 → 无帧间闪烁
- 自动检测偏移量 → 零参数也可用
- PyTorch 直算 → GPU毫秒级处理
- 渐进应用 → 无接缝感

使用方式：
1) 接 prev_frames → 以上一段末尾帧为基准，校正本段的段间累积偏移（推荐）
2) 不接 prev_frames → 以本段前 overlap_count 帧为自参考，
   校正本段内部的漂移趋势（适用于无法提供上一段帧的场景）

校正模式：
- subtract（减法模式）→ pixel -= delta * weight，对亮部和暗部同等处理
- scale（缩放模式）→ pixel *= (1 - delta/128 * weight)，对暗部影响小，更自然
"""

import torch
import json


class AutoColorDriftCorrection:
    """自动色彩漂移校正节点
    
    检测每段接续生成中发生的色彩偏移并自动纠正。
    """

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
                    "tooltip": "重叠帧数量，默认5（与工作流中的片段间重叠帧数一致）。"
                }),
            },
            "optional": {
                "prev_frames": ("IMAGE", {
                    "tooltip": "（可选）上一段末尾的重叠帧。接入后以上一段为基准校正段间累积偏移。不接则以本段前N帧为自参考。"
                }),
                "correction_mode": (["subtract", "scale"], {
                    "default": "subtract",
                    "tooltip": "subtract=减法校正(对所有亮度同等处理)；scale=缩放校正(暗部影响小，更自然)。"
                }),
                "max_correction": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "单通道最大相对校正幅度（相对于像素值128）。0.05=最大移动6.4个像素值。防止意外场景差异导致过度校正。"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "校正强度。1.0=完整校正，0.5=一半校正，0.0=不校正。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "利用重叠帧自动检测色彩漂移并做全局统一校正，无帧间闪烁。"

    def correct(self, images, overlap_count=5,
                prev_frames=None, correction_mode="subtract",
                max_correction=0.05, strength=1.0):
        info_lines = []
        info_lines.append(f"输入帧数: {images.shape[0]}")
        info_lines.append(f"重叠帧数设定: {overlap_count}")
        info_lines.append(f"校正模式: {correction_mode}")

        # ==================== 安全检查 ====================
        B, H, W, C = images.shape
        if C != 3:
            info_lines.append(f"警告: 输入图像通道数={C}，期望3通道RGB")
            return (images, "\n".join(info_lines))

        # 检查 images 是否至少有 overlap_count+1 帧（至少1帧新内容）
        if images.shape[0] <= overlap_count:
            info_lines.append(
                f"警告: 当前段帧数({images.shape[0]}) <= 重叠帧数({overlap_count})，没有需要校正的新帧"
            )
            return (images, "\n".join(info_lines))

        # ==================== 模式选择 & 计算偏移量 ====================
        has_prev_frames = prev_frames is not None
        # images 取前 overlap_count 帧
        curr_head = images[:overlap_count]  # (overlap_count, H, W, C)

        if has_prev_frames:
            # 模式1：以上一段为基准
            if prev_frames.shape[0] < overlap_count:
                info_lines.append(
                    f"警告: 上一段帧数({prev_frames.shape[0]}) < 重叠帧数({overlap_count})，跳过校正"
                )
                return (images, "\n".join(info_lines))
            
            # prev_frames 取最后 overlap_count 帧
            prev_tail = prev_frames[-overlap_count:]
            prev_mean = prev_tail.mean(dim=(0, 1, 2))
            curr_mean = curr_head.mean(dim=(0, 1, 2))
            
            # 偏移量: 当前段 - 上一段
            delta = curr_mean - prev_mean
            ref_description = f"上一段末尾{overlap_count}帧"
            ref_type = "prev_frames"
            
            info_lines.append(
                f"{ref_description}均值: R={prev_mean[0]:.2f} G={prev_mean[1]:.2f} B={prev_mean[2]:.2f}"
            )
        else:
            # 模式2：以本段前 overlap_count 帧为自参考
            curr_mean = curr_head.mean(dim=(0, 1, 2))
            
            # 取本段最后 overlap_count 帧作为"偏移后的"状态
            segment_tail = images[-overlap_count:]
            tail_mean = segment_tail.mean(dim=(0, 1, 2))
            
            # 漂移 = 尾帧均值 - 头帧均值（本段内部的漂移趋势）
            delta = tail_mean - curr_mean
            
            ref_description = f"本段前{overlap_count}帧（自参考）"
            ref_type = "self_reference"
            
            info_lines.append(
                f"{ref_description}均值: R={curr_mean[0]:.2f} G={curr_mean[1]:.2f} B={curr_mean[2]:.2f}"
            )
            info_lines.append(
                f"本段尾{overlap_count}帧均值: R={tail_mean[0]:.2f} G={tail_mean[1]:.2f} B={tail_mean[2]:.2f}"
            )

        info_lines.append(
            f"当前段前{overlap_count}帧均值: R={curr_mean[0]:.2f} G={curr_mean[1]:.2f} B={curr_mean[2]:.2f}"
        )
        info_lines.append(
            f"检测偏移量: R={delta[0]:+.2f} G={delta[1]:+.2f} B={delta[2]:+.2f}"
        )

        # ==================== 钳制和强度 ====================
        max_abs_delta = max_correction * 128.0
        delta_clamped = torch.clamp(delta, -max_abs_delta, max_abs_delta)
        delta_applied = delta_clamped * strength

        info_lines.append(
            f"校正后偏移量(钳制后): R={delta_applied[0]:+.2f} G={delta_applied[1]:+.2f} B={delta_applied[2]:+.2f}"
        )
        if has_prev_frames:
            br_drift = (curr_mean[2]/curr_mean[0] - prev_mean[2]/prev_mean[0]) * 100
        else:
            br_drift = 0.0
        info_lines.append(f"B/R比率偏移: {br_drift:+.2f}%")

        # ==================== 应用校正 ====================
        result = images.clone()

        # 前 overlap_count 帧不动（重叠帧，不需要校正）
        # 从 overlap_count 到最后一帧应用校正
        num_correct_frames = B - overlap_count

        if num_correct_frames > 0:
            # 生成渐进权重：从0线性渐变到1
            weights = torch.linspace(0, 1, num_correct_frames,
                                     device=images.device, dtype=images.dtype)
            # reshape for broadcasting: (N, 1, 1, 1)
            weights = weights.view(-1, 1, 1, 1)

            # 提取需要校正的帧
            frames_to_correct = result[overlap_count:]  # (N, H, W, C)
            
            # delta_applied shape: (C,) → reshape to (1, 1, 1, 3)
            delta_reshaped = delta_applied.view(1, 1, 1, 3)

            if correction_mode == "scale":
                # 缩放模式: pixel *= (1 - delta/128 * weight)
                # 将 delta_applied 转换为相对于128的比例
                # 例如 delta_applied = [-3, 0, +2] 表示 R要+3/128=+2.3%, B要-2/128=-1.6%
                scale_factors = 1.0 - (delta_reshaped / 128.0) * weights
                corrected = frames_to_correct * scale_factors
                info_lines.append(f"缩放校正因子范围: R=[{1 - abs(float(delta_applied[0])/128):.4f}, 1.0] "
                                  f"B=[{1 - abs(float(delta_applied[2])/128):.4f}, 1.0]")
            else:
                # 减法模式: pixel = pixel - delta * weight（默认）
                corrected = frames_to_correct - delta_reshaped * weights

            # 钳制到 [0, 1] （ComfyUI IMAGE 是 float32 [0,1] 范围）
            corrected = torch.clamp(corrected, 0.0, 1.0)

            # 写回
            result[overlap_count:] = corrected

            info_lines.append(
                f"校正帧范围: [{overlap_count+1}-{B}] (共{num_correct_frames}帧)"
            )
            info_lines.append(
                f"渐进权重: 从{overlap_count+1}帧(权重0)渐变到{B}帧(权重1)"
            )
        else:
            info_lines.append("没有需要校正的新帧")

        # 构建偏移量JSON
        drift_data = {
            "mode": ref_type,
            "correction_type": correction_mode,
            "drift_r": float(delta[0]),
            "drift_g": float(delta[1]),
            "drift_b": float(delta[2]),
            "drift_r_applied": float(delta_applied[0]),
            "drift_g_applied": float(delta_applied[1]),
            "drift_b_applied": float(delta_applied[2]),
            "overlap_count": overlap_count,
            "strength": strength,
            "frames_corrected": num_correct_frames,
            "max_correction_pct": max_correction * 100,
        }
        drift_json = json.dumps(drift_data, indent=2)

        info = "\n".join(info_lines)
        info += f"\n\nDrift JSON:\n{drift_json}"

        return (result, info)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction (自动色彩漂移校正)",
}