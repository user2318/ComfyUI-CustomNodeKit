"""
AutoColorDriftCorrection V3.1 - 漂移检测 + 实例级模板

核心策略：
1. 接缝对齐（seam_strength）：每段都启用，修正段间跳变 —— 基础校正
2. 隆起校正（bump_strength）：从第2段启用，修正段内启动震荡 —— 增强校正
3. 自适应学习：每段校正后提取残余偏差，EMA更新隆起模板

V3.1 改进：
- 移除类级静态变量，改用实例变量，解决跨工作流污染
- 新增 auto 模式：先检测漂移，再决定是否修复，避免对正常画面误修复
"""

import torch
import json
import math


# 初始隆起模板（指数衰减模型）
# 公式: value = A * t^a * exp(-t/b) + C
# 每通道4个参数，共12个浮点数，安全不发散（t→∞时归零）
_BUMP_PARAMS = {
    "R": {"A": -0.008, "a": 0.25, "b": 25, "C": -0.001},
    "G": {"A":  0.009, "a": 0.20, "b": 30, "C":  0.002},
    "B": {"A":  0.017, "a": 0.20, "b": 28, "C":  0.002},
}


def _bump_value(t, length=76):
    """
    计算位置t (0..length-1) 的 [R, G, B] 隆起偏差值。
    使用指数衰减模型，安全不发散。
    """
    x = t / (length - 1) * 75.0 if length > 1 else 0.0
    result = []
    for ch in ["R", "G", "B"]:
        p = _BUMP_PARAMS[ch]
        val = p["A"] * (x ** p["a"]) * math.exp(-x / p["b"]) + p["C"]
        result.append(val)
    return result


def _make_initial_bump(length=76):
    """生成初始隆起模板（指数衰减模型）"""
    return [_bump_value(i, length) for i in range(length)]


def _fit_bump_curve(new_frame_means, ref_rgb, length=76):
    """
    从当前段校正后的新帧数据提取隆起偏差曲线。
    """
    N = min(len(new_frame_means), length)
    curve = []
    for i in range(N):
        dev = new_frame_means[i] - ref_rgb
        curve.append([float(dev[0]), float(dev[1]), float(dev[2])])
    while len(curve) < length:
        curve.append([0.0, 0.0, 0.0])
    return curve


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
    # 线性回归斜率: slope = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)^2)
    numerator = ((x - x_mean).unsqueeze(1) * (frame_means - y_mean.unsqueeze(0))).sum(dim=0)
    denominator = ((x - x_mean) ** 2).sum()
    if denominator < 1e-8:
        return (0.0, 0.0, 0.0)
    slopes = numerator / denominator
    return (float(slopes[0]), float(slopes[1]), float(slopes[2]))


class AutoColorDriftCorrection:
    OUTPUT_NODE = False

    def __init__(self):
        # 实例级变量，每次节点重新加载时重置，不会跨工作流污染
        self._learned_bump = None    # 隆起模板 [76 x [R,G,B]]
        self._segment_count = 0      # 段计数器

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
                    "tooltip": "接缝对齐强度（基础校正）。0=禁用。auto模式下自动计算。"
                }),
                "bump_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "段内隆起校正强度（增强校正，第2段起生效）。0=禁用。auto模式下自动计算。"
                }),
                "bump_learning_rate": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "自适应学习率。0=不学习。0.3=平稳更新。1.0=完全替换。"
                }),
                "max_offset": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.02, "step": 0.001,
                    "tooltip": "最大单通道偏移钳制值。0.01≈2.6像素值。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "drift_info")
    FUNCTION = "correct"
    CATEGORY = "CustomNodes/Video"
    DESCRIPTION = "V3.1: 漂移检测 + 实例级模板。auto模式自动识别漂移，无漂移则原样输出。"

    def correct(self, images, mode="standard", overlap_count=5,
                prev_frames=None, seam_strength=0.0, bump_strength=0.0,
                bump_learning_rate=0.3, max_offset=0.01):
        info_lines = []
        B, H, W, C = images.shape
        info_lines.append(f"V3.1 输入帧数: {B}, 重叠帧数: {overlap_count}, 模式: {mode}")

        # ========== off 模式：完全旁路 ==========
        if mode == "off":
            info_lines.append(f"模式: off，跳过所有处理")
            drift_data = {
                "mode": "off",
                "v3_mode": "bypass",
                "frames_total": B,
            }
            info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
            return (images, info)

        # ========== 段计数（实例级） ==========
        self._segment_count += 1
        current_seg = self._segment_count
        info_lines.append(f"段号: {current_seg}")

        if C != 3:
            info_lines.append(f"警告: 通道数={C}，期望3")
            return (images, "\n".join(info_lines))
        if B <= overlap_count:
            info_lines.append(f"警告: 帧数({B}) <= 重叠帧数({overlap_count})，无新生成帧")
            return (images, "\n".join(info_lines))

        new_frames = images[overlap_count:]  # (N, H, W, 3)
        overlap_frames = images[:overlap_count]  # (overlap, H, W, 3)
        N = new_frames.shape[0]

        # ========== auto 模式：段内跳变检测 ==========
        if mode == "auto":
            # 检测：用段内自有重叠帧的尾帧 vs 新帧的首帧
            # 如果内容相同但色彩不同，说明有漂移，需要校正
            # 不依赖 prev_frames，每段独立检测
            last_overlap = overlap_frames[-1:]    # 重叠区域最后一帧（帧5）
            first_new = new_frames[:1]            # 新帧第一帧（帧6）
            last_mean = last_overlap.mean(dim=(0, 1, 2))
            first_mean = first_new.mean(dim=(0, 1, 2))
            local_jump_vec = first_mean - last_mean
            local_jump = max(abs(float(local_jump_vec[i])) for i in range(3))

            info_lines.append(f"段内跳变检测: 重叠末帧 vs 新帧首帧")
            info_lines.append(f"  重叠末帧均值: R={last_mean[0]:.4f} G={last_mean[1]:.4f} B={last_mean[2]:.4f}")
            info_lines.append(f"  新帧首帧均值: R={first_mean[0]:.4f} G={first_mean[1]:.4f} B={first_mean[2]:.4f}")
            info_lines.append(f"  跳变量: R={float(local_jump_vec[0]):+.6f} G={float(local_jump_vec[1]):+.6f} B={float(local_jump_vec[2]):+.6f} max={local_jump:.6f}")

            # 决策：跳变是否超过阈值？
            jump_threshold = 0.0005  # 约0.13像素值/通道
            has_drift = (local_jump > jump_threshold)

            info_lines.append(f"漂移判定: {'有漂移' if has_drift else '无漂移'} (阈值={jump_threshold})")

            if not has_drift:
                info_lines.append(f"auto: 无显著段内跳变，原样输出")
                drift_data = {
                    "mode": "auto",
                    "v3_mode": "bypass",
                    "segment": current_seg,
                    "drift_detected": False,
                    "local_jump": local_jump,
                    "jump_rgb": [float(local_jump_vec[i]) for i in range(3)],
                    "frames_total": B,
                    "frames_skipped": N,
                }
                info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
                return (images, info)

            # 有跳变：自动计算校正强度
            # 跳变越大，强度越高
            # 跳变 0.0005→0.3(最小), 0.003→0.5, 0.01→1.0(最大)
            auto_seam = min(1.0, max(0.3, local_jump * 200))
            seam_strength = auto_seam
            info_lines.append(f"auto: 跳变={local_jump:.6f}, 自动接缝强度={seam_strength:.2f}")

            # 隆起强度：跳变较大时启用适中的隆起校正
            if local_jump > 0.002:
                auto_bump = min(0.8, max(0.3, local_jump * 150))
                bump_strength = auto_bump
                info_lines.append(f"auto: 跳变较大({local_jump:.6f}), 自动隆起强度={bump_strength:.2f}")
            else:
                bump_strength = 0.0
                info_lines.append(f"auto: 跳变较小，隆起禁用")

        # ========== 初始化实例级隆起模板 ==========
        if self._learned_bump is None:
            self._learned_bump = _make_initial_bump()
            info_lines.append(f"隆起初始模板: 指数衰减模型 (峰值≈0.016)")

        current_template = self._learned_bump

        # ========== 隆起校正是否启用？==========
        # 第1段不做隆起校正（无上一段数据参考）
        # 第2段起启用（已学到上一段的隆起规律）
        bump_enabled = (bump_strength > 0 and current_seg > 1)
        if bump_strength > 0 and not bump_enabled:
            info_lines.append(f"隆起校正: 跳过第1段（第2段起生效）")

        # ========== 1. 接缝对齐计算 ==========
        has_prev = prev_frames is not None
        base_offset = torch.zeros(3, device=images.device)
        ref_type = "none"
        fade_frames = 0

        if has_prev:
            if prev_frames.shape[0] >= overlap_count:
                prev_slice = prev_frames[-overlap_count:]
                curr_slice = new_frames[:overlap_count]
                prev_mean = prev_slice.mean(dim=(0, 1, 2))
                curr_mean = curr_slice.mean(dim=(0, 1, 2))
                base_offset = curr_mean - prev_mean
                ref_type = "prev_frame"
                fade_frames = min(overlap_count, 3)
                info_lines.append(f"基准: prev_frames (上一段尾{overlap_count}帧 vs 当前段前{overlap_count}帧)")
                info_lines.append(f"prev均值: R={prev_mean[0]:.4f} G={prev_mean[1]:.4f} B={prev_mean[2]:.4f}")
                info_lines.append(f"当前均值: R={curr_mean[0]:.4f} G={curr_mean[1]:.4f} B={curr_mean[2]:.4f}")
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
            fade_frames = min(overlap_count, 3)
            info_lines.append(f"基准: 接缝对齐 (重叠末帧 vs 新帧首帧)")
            info_lines.append(f"重叠末帧均值: R={last_mean[0]:.4f} G={last_mean[1]:.4f} B={last_mean[2]:.4f}")
            info_lines.append(f"新帧首帧均值: R={first_mean[0]:.4f} G={first_mean[1]:.4f} B={first_mean[2]:.4f}")
            info_lines.append(f"接缝偏移: R={base_offset[0]:+.6f} G={base_offset[1]:+.6f} B={base_offset[2]:+.6f}")

        if ref_type == "none" and not bump_enabled:
            info_lines.append(f"基准: 无对齐 (输出原始帧)")
            return (images, "\n".join(info_lines))

        # ========== 2. 钳制接缝偏移 ==========
        if ref_type != "none":
            base_clamped = torch.clamp(base_offset, -max_offset, max_offset)
            if has_prev:
                base_applied = base_clamped
            else:
                base_applied = base_clamped * seam_strength
            info_lines.append(f"钳制后偏移(max_offset={max_offset}): R={base_applied[0]:+.6f} G={base_applied[1]:+.6f} B={base_applied[2]:+.6f}")
        else:
            base_applied = torch.zeros(3, device=images.device)
            fade_frames = 0

        # ========== 3. 应用校正 ==========
        result = images.clone()
        frames_to_correct = result[overlap_count:]

        # 接缝对齐（基础校正，每段都做）
        if base_applied.abs().sum() > 1e-8:
            fade_weights = torch.arange(1, fade_frames + 1, device=images.device, dtype=images.dtype) / fade_frames
            fade_weights = fade_weights.view(-1, 1, 1, 1)
            for i in range(min(fade_frames, N)):
                frames_to_correct[i] -= base_applied.view(1, 1, 3) * fade_weights[i]
            if N > fade_frames:
                frames_to_correct[fade_frames:] -= base_applied.view(1, 1, 3)
            info_lines.append(f"接缝校正: 前{fade_frames}帧淡入 + {N-fade_frames}帧完整应用")

        # 隆起校正（增强校正，第2段起生效）
        if bump_enabled:
            template_tensor = torch.tensor(current_template, device=images.device, dtype=images.dtype)
            if N <= 76:
                bump_curve = template_tensor[:N]
            else:
                bump_curve = torch.zeros(N, 3, device=images.device, dtype=images.dtype)
                bump_curve[:76] = template_tensor

            for i in range(N):
                frames_to_correct[i] -= bump_curve[i].view(1, 1, 3) * bump_strength

            info_lines.append(f"隆起校正: 强度={bump_strength}, 模板峰值={max(abs(v) for row in current_template for v in row):.4f}")

        frames_to_correct = torch.clamp(frames_to_correct, 0.0, 1.0)
        result[overlap_count:] = frames_to_correct

        info_lines.append(f"校正帧: [{overlap_count+1}-{B}] ({N}帧)")

        # ========== 4. 自适应学习 ==========
        if bump_learning_rate > 0 and N >= 5:
            ref_rgb = frames_to_correct[-5:].mean(dim=(0, 1, 2))
            new_means = frames_to_correct.mean(dim=(1, 2))
            observed = _fit_bump_curve(new_means, ref_rgb, 76)

            lr = bump_learning_rate
            updated = []
            for i in range(76):
                old = current_template[i]
                new = observed[i]
                updated.append([
                    old[0] * (1 - lr) + new[0] * lr,
                    old[1] * (1 - lr) + new[1] * lr,
                    old[2] * (1 - lr) + new[2] * lr,
                ])

            self._learned_bump = updated
            max_v = max(abs(v) for row in updated for v in row)
            info_lines.append(f"自适应学习: lr={lr}, 模板峰值={max_v:.4f}")

        drift_data = {
            "mode": mode,
            "v3_mode": ref_type if ref_type != "none" else "bump_only",
            "segment": current_seg,
            "seam_strength": seam_strength,
            "bump_strength": bump_strength if bump_enabled else 0,
            "bump_learning_rate": bump_learning_rate,
            "frames_corrected": N,
        }
        if ref_type != "none":
            drift_data["offset_rgb"] = [float(base_offset[i]) for i in range(3)]
        if base_applied.abs().sum() > 1e-8:
            drift_data["offset_clamped_rgb"] = [float(base_applied[i]) for i in range(3)]
            drift_data["fade_frames"] = fade_frames
        if current_template:
            drift_data["bump_peak"] = float(max(abs(v) for row in current_template for v in row))

        info = "\n".join(info_lines) + f"\n\n{json.dumps(drift_data, indent=2)}"
        return (result, info)


NODE_CLASS_MAPPINGS = {
    "AutoColorDriftCorrection": AutoColorDriftCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorDriftCorrection": "Auto Color Drift Correction V3.1 (漂移检测)",
}