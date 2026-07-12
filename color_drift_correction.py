"""
AutoColorDriftCorrection V4.3 - 两遍校正：全局偏移 + 残余增量微调

核心策略：
1. 第一遍（全局偏移）：多帧均值检测跳变，统一校正所有新帧，消除跳变主体
2. 第二遍（残余微调）：增量逐帧跟踪残余漂移，小钳制上限防噪声
3. 不再需要 EMA 平滑（全局偏移已很准，残余量极小）

V4.3 变更：
- 两遍校正替代纯增量递推：先全局对齐，再小幅度微调
- 移除 EMA：全局偏移足够准，残余微调幅度极小，无需平滑
- auto 模式：多帧检测的 jump_vec 直接作为第一遍偏移向量
- 新增 residual_max_offset 参数控制第二遍钳制（默认 max_offset×0.2）

V4.2 变更：
- 移除淡入（fade-in）：重叠帧不动，新帧第一帧直接全量对齐
- 新增增量逐帧校正（被 V4.3 的两遍方案替代）
"""

import torch
import json


class AutoColorDriftCorrection:
    OUTPUT_NODE = False

    def __init__(self):
        pass

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
                    "tooltip": "第一遍全局偏移强度。0=禁用第一遍。auto模式下自动决策。"
                }),
                "max_offset": ("FLOAT", {
                    "default": 0.02, "min": 0.001, "max": 0.05, "step": 0.001,
                    "tooltip": "第一遍最大单通道偏移钳制值。0.02≈5.1像素值。"
                }),
                "residual_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "第二遍残余微调强度（相对max_offset的比例）。0=禁用第二遍。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "V4.3: 全局偏移 + 残余增量微调。两遍校正消除跳变。"

    def correct(self, images, mode="standard", overlap_count=5,
                prev_frames=None, seam_strength=0.0, max_offset=0.02,
                residual_strength=0.2):
        info_lines = []
        B, H, W, C = images.shape
        info_lines.append(f"V4.3 输入帧数: {B}, 重叠帧数: {overlap_count}, 模式: {mode}")

        # ========== off 模式：完全旁路 ==========
        if mode == "off":
            info_lines.append("模式: off，跳过所有处理")
            drift_data = {"mode": "off", "v4_mode": "bypass", "frames_total": B}
            info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
            return (images, info)

        if C != 3:
            info_lines.append(f"警告: 通道数={C}，期望3")
            return (images, "\n".join(info_lines))
        if B <= overlap_count:
            info_lines.append(f"警告: 帧数({B}) <= 重叠帧数({overlap_count})，无新生成帧")
            return (images, "\n".join(info_lines))

        result = images.clone()

        # ========== Pass 0: prev_frames 对齐（整段偏移） ==========
        # 当接入 prev_frames 时，将整段 73 帧对齐到上一段真实输出
        if prev_frames is not None and prev_frames.shape[0] >= overlap_count:
            ref_prev = prev_frames[-overlap_count:].mean(dim=(0, 1, 2))
            cur_overlap = result[:overlap_count].mean(dim=(0, 1, 2))
            pass0_offset = ref_prev - cur_overlap
            pass0_clamped = torch.clamp(pass0_offset, -max_offset, max_offset)
            result += pass0_clamped.view(1, 1, 3)
            info_lines.append(f"Pass0(prev对齐): 偏移=[{float(pass0_clamped[0]):+.6f},{float(pass0_clamped[1]):+.6f},{float(pass0_clamped[2]):+.6f}] 基准来自 prev_frames[-{overlap_count}:]")

        new_frames = result[overlap_count:]          # (N, H, W, 3)
        overlap_frames = result[:overlap_count]      # (overlap, H, W, 3)
        N = new_frames.shape[0]
        fade_frames = min(overlap_count, 5)          # 多帧均值抗噪

        corrected_new = result[overlap_count:]       # 视图

        # ========== 第一遍：检测全局偏移 ==========
        base_offset = torch.zeros(3, device=images.device)
        ref_type = "none"

        if mode == "auto":
            # auto 模式：多帧均值检测
            last_overlap = overlap_frames[-fade_frames:]
            first_new = new_frames[:fade_frames]
            ref_mean = last_overlap.mean(dim=(0, 1, 2))
            curr_mean = first_new.mean(dim=(0, 1, 2))
            jump_vec = curr_mean - ref_mean
            local_jump = max(abs(float(jump_vec[i])) for i in range(3))

            info_lines.append("第一遍(全局): auto多帧均值检测 (各{}帧)".format(fade_frames))
            info_lines.append(f"  重叠尾均值: R={ref_mean[0]:.4f} G={ref_mean[1]:.4f} B={ref_mean[2]:.4f}")
            info_lines.append(f"  新帧头均值: R={curr_mean[0]:.4f} G={curr_mean[1]:.4f} B={curr_mean[2]:.4f}")
            info_lines.append(f"  跳变量: R={float(jump_vec[0]):+.6f} G={float(jump_vec[1]):+.6f} B={float(jump_vec[2]):+.6f} max={local_jump:.6f}")

            jump_threshold = 0.0005
            has_jump = (local_jump > jump_threshold)
            info_lines.append(f"跳变判定: {'有跳变' if has_jump else '无跳变'} (阈值={jump_threshold})")

            if has_jump:
                base_offset = jump_vec  # 直接使用多帧检测的偏移向量
                ref_type = "auto_seam"
                info_lines.append(f"auto: 直接使用检测偏移向量作为全局校正")
            else:
                seam_strength = 0.0
                info_lines.append(f"auto: 无显著跳变, 跳过第一遍")

        else:  # standard 模式
            if seam_strength > 0:
                # 多帧均值检测偏移
                last_overlap = overlap_frames[-fade_frames:]
                first_new = new_frames[:fade_frames]
                ref_mean = last_overlap.mean(dim=(0, 1, 2))
                curr_mean = first_new.mean(dim=(0, 1, 2))
                base_offset = curr_mean - ref_mean
                ref_type = "seam_align"
                info_lines.append("第一遍(全局): standard多帧均值检测 (各{}帧)".format(fade_frames))
                info_lines.append(f"  重叠尾均值: R={ref_mean[0]:.4f} G={ref_mean[1]:.4f} B={ref_mean[2]:.4f}")
                info_lines.append(f"  新帧头均值: R={curr_mean[0]:.4f} G={curr_mean[1]:.4f} B={curr_mean[2]:.4f}")
                info_lines.append(f"  段间偏移: R={float(base_offset[0]):+.6f} G={float(base_offset[1]):+.6f} B={float(base_offset[2]):+.6f}")

        # ========== 应用第一遍：全局偏移 ==========
        if ref_type != "none" and base_offset.abs().sum() > 1e-8:
            # 分通道钳制
            max_global = max_offset * (seam_strength if mode != "auto" else 1.0)
            base_clamped = torch.clamp(base_offset, -max_global, max_global)
            # 全部新帧统一减去 base_clamped
            corrected_new -= base_clamped.view(1, 1, 3)
            info_lines.append(f"  全局偏移施加: R={float(base_clamped[0]):+.6f} G={float(base_clamped[1]):+.6f} B={float(base_clamped[2]):+.6f} (max={max_global:.4f})")
        else:
            base_clamped = torch.zeros(3, device=images.device)
            info_lines.append("  第一遍跳过 (无偏移)")

        # ========== 第二遍：残余偏移增补 ==========
        # 每帧独立向基准靠拢：offset = ref_mean - cur_mean
        # 非增量递推，每帧不依赖前一帧，无误差传递
        if residual_strength > 0 and N >= 1:
            residual_max = max_offset * residual_strength  # 钳制上限
            info_lines.append(f"第二遍(偏移增补): 钳制上限={residual_max:.5f} (residual_strength={residual_strength})")

            # 参考基准：重叠末帧均值
            ref_mean = overlap_frames[-1].mean(dim=(0, 1))         # (3,)

            max_residual = 0.0
            total_residual = 0.0

            for i in range(N):
                cur_mean = corrected_new[i].mean(dim=(0, 1))       # (3,)
                # 向基准靠拢的偏移量（独立计算，不依赖前一帧）
                residual_offset = ref_mean - cur_mean
                residual_clamped = torch.clamp(residual_offset, -residual_max, residual_max)
                corrected_new[i] += residual_clamped.view(1, 1, 3)

                dev = float(residual_clamped.abs().max())
                if dev > max_residual:
                    max_residual = dev
                total_residual += dev

                if dev > 0.0003 or i < 2 or i == N - 1:
                    info_lines.append(f"  帧{overlap_count+1+i}: offset=[{float(residual_clamped[0]):+.6f},{float(residual_clamped[1]):+.6f},{float(residual_clamped[2]):+.6f}] "
                                     f"dev={dev:.6f}")

            avg_residual = total_residual / N
            info_lines.append(f"  偏移增补统计: 最大={max_residual:.6f}, 均值={avg_residual:.6f}")
        else:
            max_residual = 0.0
            avg_residual = 0.0
            info_lines.append("  第二遍跳过 (residual_strength=0 或 N<1)")

        # ========== 钳位到有效范围 ==========
        corrected_new.clamp_(0.0, 1.0)
        info_lines.append(f"校正帧: [{overlap_count+1}-{B}] ({N}帧)")

        # ========== 输出校正信息 ==========
        drift_data = {
            "mode": mode,
            "v4_mode": "two_pass",
            "seam_strength": seam_strength if mode != "auto" else 1.0,
            "max_offset": max_offset,
            "residual_strength": residual_strength,
            "frames_corrected": N,
            "global_offset_rgb": [float(base_clamped[i]) for i in range(3)],
            "max_global_offset": round(float(base_clamped.abs().max()), 6),
            "max_residual_offset": round(float(max_residual), 6),
            "avg_residual_offset": round(float(avg_residual), 6),
        }

        info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
        return (result, info)


NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction V4.3 (两遍校正)",
}