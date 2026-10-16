"""
WanSCAIL2PhaseSampler — 独立两阶段采样节点。

完整复刻 SCAIL2LoopSampler 的两阶段采样核心逻辑。
用像素帧（IMAGE）作为锚定输入，内部完成 VAE 编码、锚定帧回写、noise_mask 管理。

与 WanSCAILToVideoMultiRef 配合使用时，直接接在 conditioning 节点下游替代 KSampler。
previous_frames 来自上一个 chunk 解码后的尾部像素帧。

两阶段采样核心逻辑基于 ComfyUI-Scail2-Sampler-Helper 项目：
https://github.com/checknickname/ComfyUI-Scail2-Sampler-Helper
感谢作者 checknickname 的开源贡献。
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import logging


logger = logging.getLogger("WanSCAIL2PhaseSampler")


# ==================== σ 阶梯工具（外部 SIGMAS 支持） ====================

def _fmt_sigmas(sigmas, max_items=24):
    """把 σ 阶梯打成紧凑字符串，便于日志核对 / 手填比对。"""
    vals = [float(v) for v in sigmas.reshape(-1).tolist()]
    if len(vals) <= max_items:
        return ", ".join("%g" % v for v in vals)
    half = max_items // 2
    return "%s, ... , %s" % (", ".join("%g" % v for v in vals[:half]),
                             ", ".join("%g" % v for v in vals[-half:]))


def _external_sigmas(sigmas, steps, denoise):
    """校验并规整外部 SIGMAS（可来自 ManualSigmas / BasicScheduler / SplitSigmas 等）。

    约定（与 comfy 官方一致）：len(sigmas) = 步数 + 1，末尾元素为 0。
    返回 (CPU float tensor, 实际步数)；未提供或不可用时返回 None（回退到 steps/scheduler）。
    """
    if sigmas is None:
        return None
    try:
        t = sigmas.detach().to("cpu").float().reshape(-1)
    except AttributeError:
        try:
            t = torch.as_tensor(sigmas, dtype=torch.float32).reshape(-1)
        except Exception as exc:
            logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 无法解析（%s）=> 已忽略，改用 steps/scheduler", exc)
            return None
    if int(t.numel()) < 2:
        logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 只有 %d 个值（至少 2 个）=> 已忽略，改用 steps/scheduler",
                       int(t.numel()))
        return None

    vals = [float(v) for v in t.tolist()]
    if abs(vals[-1]) > 1e-6:
        logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 末尾 %.6g ≠ 0 => 已自动补 0"
                       "（否则最后一步不会完全去噪）", vals[-1])
        t = torch.cat([t, torch.zeros(1, dtype=t.dtype)])
        vals.append(0.0)
    if min(vals) < 0.0:
        logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 含负值（最小 %.6g）=> 结果不可预期，请检查输入", min(vals))
    rises = [(i, vals[i], vals[i + 1]) for i in range(len(vals) - 1) if vals[i + 1] > vals[i] + 1e-9]
    if rises:
        i, a, b = rises[0]
        logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 不是单调递减：第 %d→%d 个 %g → %g"
                       "（共 %d 处上升）=> 建议按 1 → 0 递减", i + 1, i + 2, a, b, len(rises))
    if max(vals) > 1.5:
        logger.warning("[WanSCAIL2PhaseSampler] 外部 SIGMAS 最大 %.6g > 1.5 => 看起来是 SD/SVD 的 σ"
                       "（本模型是 flow，σ 通常在 0~1）=> 已按原样使用，请确认", max(vals))
    if abs(vals[0] - 1.0) > 1e-3:
        logger.info("[WanSCAIL2PhaseSampler] 外部 SIGMAS 首项 σ=%.4g（≠1.0 => 起点不是纯噪声，"
                    "等价于降低了最大去噪强度）", vals[0])

    n_steps = int(t.numel()) - 1
    if int(steps) != n_steps:
        logger.warning("[WanSCAIL2PhaseSampler] 控件 steps=%s 与外部 SIGMAS 不符 => steps 已忽略，实际按 %d 步执行",
                       steps, n_steps)
    if float(denoise) < 0.9999:
        logger.warning("[WanSCAIL2PhaseSampler] 控件 denoise=%.3f 已忽略（外部 SIGMAS 已含起点；"
                       "如需截断请在阶梯里截）", float(denoise))
    logger.info("[WanSCAIL2PhaseSampler] 使用外部 SIGMAS：%d 步，σ = %s", n_steps, _fmt_sigmas(t))
    return t, n_steps


def _build_ladder(model1, steps, sampler_name, scheduler, denoise):
    """按官方 KSampler 生成 σ 阶梯（CPU tensor，len = 步数 + 1），并打进日志便于事后核对。"""
    s = comfy.samplers.KSampler(
        model1, steps=int(steps), device=model1.load_device,
        sampler=sampler_name, scheduler=scheduler,
        denoise=denoise, model_options=model1.model_options)
    t = s.sigmas.detach().to("cpu").float()
    del s
    logger.info("[WanSCAIL2PhaseSampler] σ 阶梯（%d 步，scheduler=%s，denoise=%.2f）：%s",
                int(t.numel()) - 1, scheduler, float(denoise), _fmt_sigmas(t))
    return t


class WanSCAIL2PhaseSampler:
    """
    两阶段采样节点。
    
    输入要求：latent 必须从 WanSCAILToVideoMultiRef 等 conditioning 节点来。
    previous_frames 是上一个 chunk 解码后的尾部像素帧。
    
    两阶段条件：split_step > 0 AND split_step < steps AND 有 previous_frames。

    可选输入 sigmas（SIGMAS）：接上外部 σ 阶梯后 steps/scheduler/denoise 被旁路，
    实际步数 = len(sigmas) - 1（末尾元素应为 0，缺失会自动补 0）；其余机制
    （锚定回写 / phase_schedule / color_align）完全不变。
    
    流程：
    1. 将 previous_frames 尾部 previous_frame_count 帧 VAE 编码，写入 latent 前段作锚定
    2. 创建 noise_mask：锚定区=phase1_noise，其余=1
    3. Phase 1：正常噪声 + noise_mask → 采样到 split_step
    4. 回写：用原始锚定帧 latent 强制覆盖 Phase 1 输出中的锚定区
    5. 更新 noise_mask：锚定区=phase2_noise，其余=1
    6. Phase 2：空噪声 + noise_mask → 采样到结束
    7. 输出 latent + 第二阶段 noise_mask
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL", {"tooltip": "Phase 1 模型。Phase 1 model."}),
                "positive": ("CONDITIONING", {"tooltip": "正向条件。Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件。Negative conditioning."}),
                "vae": ("VAE", {"tooltip": "VAE，用于锚定帧编解码。VAE, for anchor frame encode/decode."}),
                "latent": ("LATENT", {"tooltip": "输入的 latent（来自 conditioning 节点）。Input latent (from conditioning node)."}),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_step": ("INT", {"default": 2, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "分割步数。0=单阶段；>0且<steps且接入了previous_frames时启用两阶段。Split step. 0=single-phase; >0 and <steps with previous_frames enables two-phase."}),
                "phase1_noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Phase 1 锚定帧 noise_mask 值。0=冻结，1=自由。Phase 1 anchor noise_mask value. 0=frozen, 1=free."}),
                "phase2_noise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Phase 2 锚定帧 noise_mask 值。0=冻结，1=自由。Phase 2 anchor noise_mask value. 0=frozen, 1=free."}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 4,
                    "tooltip": "从 previous_frames 尾部取多少像素帧作为锚定（单位：像素帧，会经 VAE 编码为 latent）。How many tail pixel frames of previous_frames to use as anchor (pixel frames; VAE-encoded to latent)."}),
                "phase_schedule": ("BOOLEAN", {"default": False,
                    "tooltip": "逐步插值回写：每步一个阶段（忽略 split_step），锚定区 noise_mask 从 phase1_noise 插值到 phase2_noise，每阶段结束强制回写锚定区。Per-step interpolated anchor write-back: one stage per step (ignores split_step); anchor noise_mask interpolates from phase1_noise to phase2_noise; anchor region force written back after every stage."}),
                "schedule_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "mask 插值曲线形状：s=t^exponent。<1=先陡后缓（前几步高自由度，随后强锚定）；1.0=线性；>1=先缓后陡。Mask interpolation curve shape: s=t^exponent. <1=steep first then flat; 1.0=linear; >1=flat first then steep."}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"optional": True,
                    "tooltip": "可选：外部 σ 阶梯（SIGMAS，如 ManualSigmas / BasicScheduler / SplitSigmas）。接上后 steps/scheduler/denoise 被忽略，实际步数=长度-1，末尾元素应为 0（缺则自动补 0）。Optional external sigma ladder; steps/scheduler/denoise are bypassed."}),
                "model2": ("MODEL", {"optional": True,
                    "tooltip": "Phase 2 使用的模型，不提供时使用 model1。Model used for Phase 2; falls back to model1 if not provided."}),
                "previous_frames": ("IMAGE", {"optional": True,
                    "tooltip": "上一段解码后的尾部像素帧。不接时退化为单阶段。Tail pixel frames of previous decoded segment. If not connected, falls back to single-phase."}),
                "color_align": ("BOOLEAN", {"default": True,
                    "tooltip": "接续一致性色彩/亮度校正：采样结束后，用采样输出锚定区与接续帧 latent 的 per-channel mean/std 偏差，对整段做色彩+亮度校正。仅在有接续帧时生效。Continuity color/brightness correction: aligns whole sampled latent to the anchor-frame latent by per-channel mean/std. Only active when anchor frames exist."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    EXPERIMENTAL = True
    DESCRIPTION = "SCAIL-2 完整两阶段采样。内部完成锚定帧 VAE 编码、回写、noise_mask 管理。 (optional per-step interpolated write-back via phase_schedule)"

    def sample(self, model1, positive, negative, vae, latent, width, height,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               split_step=0, phase1_noise=0.5, phase2_noise=0.1,
               previous_frame_count=5, model2=None, previous_frames=None,
               color_align=True,
               phase_schedule=False, schedule_exponent=1.0, sigmas=None):

        samples = latent["samples"]
        B, C, T, Hl, Wl = samples.shape

        # ----- 处理 previous_frames（像素帧 → VAE 编码 → latent 锚定）-----
        encoded_prev = None
        prev_latent_frames = 0
        if previous_frames is not None and previous_frames.shape[0] > 0:
            # 取尾部 previous_frame_count 帧
            prev_trimmed = previous_frames[-previous_frame_count:]
            # 先 upscale 到目标分辨率再 VAE 编码
            pf = comfy.utils.common_upscale(
                prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center"
            ).movedim(1, -1)
            encoded_prev = vae.encode(pf[:, :, :, :3])
            prev_latent_frames = min(encoded_prev.shape[2], T)

            # 防御：锚定 latent 空间尺寸与输入 latent 不一致时 resize 对齐（避免赋值广播崩溃）
            if (encoded_prev.shape[-1] != samples.shape[-1]
                    or encoded_prev.shape[-2] != samples.shape[-2]):
                logger.warning(
                    "[WanSCAIL2PhaseSampler] previous_frames 编码后 latent 空间尺寸 %s 与输入 latent 尺寸 %s 不一致，已 resize 对齐",
                    tuple(encoded_prev.shape[-2:]), tuple(samples.shape[-2:]))
                encoded_prev = comfy.utils.common_upscale(
                    encoded_prev, samples.shape[-1], samples.shape[-2], "bicubic", "center")

            # 写入 latent 前段（锚定区域）
            samples = samples.clone()
            samples[:, :, :prev_latent_frames] = encoded_prev[:, :, :prev_latent_frames].to(
                device=samples.device, dtype=samples.dtype)


        # ----- 噪声生成（GPU）-----
        noise = comfy.sample.prepare_noise(samples, seed)

        # ----- σ 阶梯：外部 SIGMAS 优先（接上后 steps/scheduler/denoise 被旁路）-----
        ext = _external_sigmas(sigmas, steps, denoise)
        if ext is not None:
            full_sigmas_cpu, steps = ext
            if split_step > 0 and split_step >= steps:
                logger.warning(
                    "[WanSCAIL2PhaseSampler] split_step=%d >= 实际步数 %d => 不满足两阶段条件，本次按单阶段执行",
                    split_step, steps)
        else:
            full_sigmas_cpu = _build_ladder(model1, steps, sampler_name, scheduler, denoise)
        sampling_sigmas = None if ext is None else full_sigmas_cpu.to(model1.load_device)

        pbar = comfy.utils.ProgressBar(steps)

        def _mk_cb(offset=0):
            def cb(step, denoised, x, total_steps):
                pbar.update_absolute(offset + step + 1, steps)
            return cb

        # ========== phase_schedule mode (N stages) ==========
        if phase_schedule and encoded_prev is not None:
            logger.info("[WanSCAIL2PhaseSampler] phase_schedule enabled: %d stages, anchor noise_mask %.2f -> %.2f",
                        steps, phase1_noise, phase2_noise)
            samples_out = self._sample_phase_schedule(
                model1, model2, positive, negative, samples, prev_latent_frames,
                encoded_prev, noise, steps, cfg, sampler_name, scheduler,
                denoise, seed, phase1_noise, phase2_noise,
                schedule_exponent, make_cb=_mk_cb, full_sigmas=full_sigmas_cpu)
            out_noise_mask = torch.ones((1, 1, T, Hl, Wl), device=samples.device, dtype=samples.dtype)
            out_noise_mask[:, :, :prev_latent_frames] = phase2_noise

        # ========== 两阶段模式 ==========
        elif split_step > 0 and split_step < steps and encoded_prev is not None:
            # --- Phase 1 noise_mask ---
            mask1 = torch.ones((1, 1, T, Hl, Wl), device=samples.device, dtype=samples.dtype)
            mask1[:, :, :prev_latent_frames] = phase1_noise

            # --- Phase 2 noise_mask ---
            mask2 = torch.ones((1, 1, T, Hl, Wl), device=samples.device, dtype=samples.dtype)
            mask2[:, :, :prev_latent_frames] = phase2_noise

            # --- KSampler → sigma 分割 ---
            s1 = comfy.samplers.KSampler(
                model1, steps=steps, device=model1.load_device,
                sampler=sampler_name, scheduler=scheduler,
                denoise=denoise, model_options=model1.model_options)
            full_sigmas = full_sigmas_cpu.to(model1.load_device)
            high_sigmas = full_sigmas[:split_step + 1]
            low_sigmas = full_sigmas[split_step:]

            # ---------- Phase 1 ----------
            samples_out = s1.sample(
                noise, positive, negative, cfg=cfg, latent_image=samples,
                denoise_mask=mask1, sigmas=high_sigmas, seed=seed,
                callback=_mk_cb(0))

            del s1, mask1, high_sigmas
            comfy.model_management.soft_empty_cache()

            # --- 锚定帧回写（用原始锚定帧 latent 强制覆盖 Phase 1 输出中的锚定区）---
            samples_out = samples_out.clone()
            samples_out[:, :, :prev_latent_frames] = encoded_prev[:, :, :prev_latent_frames].to(
                device=samples_out.device, dtype=samples_out.dtype)

            # ---------- Phase 2 ----------
            empty_noise = torch.zeros_like(noise)
            s2_model = model2 if model2 is not None else model1
            s2 = comfy.samplers.KSampler(
                s2_model, steps=steps, device=s2_model.load_device,
                sampler=sampler_name, scheduler=scheduler,
                denoise=denoise, model_options=s2_model.model_options)

            samples_out = s2.sample(
                empty_noise, positive, negative, cfg=cfg, latent_image=samples_out,
                denoise_mask=mask2, sigmas=low_sigmas, seed=seed,
                callback=_mk_cb(split_step))

            del s2, empty_noise, low_sigmas, full_sigmas
            comfy.model_management.soft_empty_cache()

            out_noise_mask = mask2

        # ========== 单阶段 + 锚定帧模式 ==========
        elif encoded_prev is not None:
            noise_mask = torch.ones((1, 1, T, Hl, Wl), device=samples.device, dtype=samples.dtype)
            noise_mask[:, :, :prev_latent_frames] = phase2_noise

            samples_out = comfy.sample.sample(
                model=model1, noise=noise, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative=negative,
                latent_image=samples, denoise=denoise,
                noise_mask=noise_mask, seed=seed, sigmas=sampling_sigmas,
                callback=_mk_cb())
            out_noise_mask = noise_mask

        # ========== 纯单阶段（无锚定帧）==========
        else:
            samples_out = comfy.sample.sample(
                model=model1, noise=noise, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative=negative,
                latent_image=samples, denoise=denoise,
                noise_mask=None, seed=seed, sigmas=sampling_sigmas,
                callback=_mk_cb())
            out_noise_mask = None

        # ---------- Continuity color/brightness correction ----------
        if color_align and encoded_prev is not None and prev_latent_frames > 0:
            refs = encoded_prev[:, :, :prev_latent_frames].to(samples_out.device, samples_out.dtype)
            curs = samples_out[:, :, :prev_latent_frames]
            p_mean = refs.mean(dim=(0, 2, 3, 4), keepdim=True)
            p_std = refs.std(dim=(0, 2, 3, 4), keepdim=True)
            c_mean = curs.mean(dim=(0, 2, 3, 4), keepdim=True)
            c_std = curs.std(dim=(0, 2, 3, 4), keepdim=True)
            scale = p_std / (c_std + 1e-5)
            shift = p_mean - scale * c_mean
            samples_out = samples_out * scale + shift
            # ★ 紧凑：默认只报"有没有实质校正"（旧实现打两串 16 个浮点 ⇒ 一行很长、还常被截断）
            _sc = [float(v) for v in scale.reshape(-1).tolist()]
            _sh = [float(v) for v in shift.reshape(-1).tolist()]
            _dev = max([abs(v - 1.0) for v in _sc] + [0.0])
            _shmax = max([abs(v) for v in _sh] + [0.0])
            if _dev < 1e-4 and _shmax < 1e-4:
                logger.info("[WanSCAIL2PhaseSampler] color_align：≈恒等（max|scale-1|=%.1e，max|shift|=%.1e）"
                            " => 接续帧与本段输出本就一致，未做实质校正", _dev, _shmax)
            else:
                logger.info("[WanSCAIL2PhaseSampler] color_align：已校正（前 4 通道 scale=%s shift=%s）",
                            ["%.4f" % v for v in _sc[:4]], ["%+.4f" % v for v in _sh[:4]])

        out = {"samples": samples_out}
        if out_noise_mask is not None:
            out["noise_mask"] = out_noise_mask

        return (out,)

    # ========== phase_schedule: per-step interpolated anchor write-back ==========
    def _sample_phase_schedule(self, model1, model2, positive, negative, latent,
                               anchor_len, anchor_content, noise, steps, cfg,
                               sampler_name, scheduler, denoise, seed,
                               phase1_noise, phase2_noise,
                                schedule_exponent, make_cb, full_sigmas=None):
        """Extend two-phase to N stages (one stage per sampling step).

        Continuity mechanism is identical to the two-phase mode:
        - stage 1: full noise + latent_image=latent (anchor content in latent head)
        - stage 2..N: empty noise + latent_image=previous stage output
          (noise_scaling: x0 = 0*sigma_i + prev_out = prev_out, so x stays continuous)
        - per-stage sigmas share endpoints: [sigma_i, sigma_{i+1}]
        - anchor noise_mask interpolates from phase1_noise to phase2_noise via s=t^schedule_exponent
          (schedule_exponent<1 => steep first then flat);
        - anchor region is force written back after every stage (like two-phase Phase 1)

        model2, if provided, is used for stages 2..N (same as two-phase Phase 2).
        """
        B, C, T, Hl, Wl = latent.shape
        device = model1.load_device

        # σ 阶梯由调用方（sample）解析：外部 SIGMAS 优先，否则官方 KSampler 生成
        # （显式 sigmas ⇒ denoise 不会被应用两次）
        if full_sigmas is None:
            full_sigmas = _build_ladder(model1, steps, sampler_name, scheduler, denoise)
        full_sigmas = full_sigmas.to(device)

        n_stages = len(full_sigmas) - 1  # number of stages = number of steps
        out = latent
        for i in range(n_stages):
            t = i / max(n_stages - 1, 1)  # progress [0,1]
            s = t ** schedule_exponent  # curve shaping: <1 => steep first, then flat
            alpha = phase1_noise + (phase2_noise - phase1_noise) * s

            mask = torch.ones((B, 1, T, Hl, Wl), device=latent.device, dtype=latent.dtype)
            if anchor_len > 0:
                mask[:, :, :anchor_len] = alpha

            if i == 0:
                stage_noise = noise
                lat_in = latent
            else:
                stage_noise = torch.zeros_like(noise)
                lat_in = out

            stage_model = model2 if (i > 0 and model2 is not None) else model1
            s = comfy.samplers.KSampler(
                stage_model, steps=steps, device=stage_model.load_device,
                sampler=sampler_name, scheduler=scheduler,
                denoise=denoise, model_options=stage_model.model_options)
            out = s.sample(
                stage_noise, positive, negative, cfg=cfg, latent_image=lat_in,
                denoise_mask=mask, sigmas=full_sigmas[i:i + 2], seed=seed,
                callback=make_cb(i))
            del s, mask, stage_noise, lat_in

            # force write-back of the anchor region (original anchor content)
            if anchor_len > 0:
                out = out.clone()
                out[:, :, :anchor_len] = anchor_content[:, :, :anchor_len].to(out.device, out.dtype)

        del full_sigmas
        comfy.model_management.soft_empty_cache()
        return out



NODE_CLASS_MAPPINGS = {
    "WanSCAIL2PhaseSampler": WanSCAIL2PhaseSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAIL2PhaseSampler": "Wan SCAIL-2 Phase Sampler",
}