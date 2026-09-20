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


class WanSCAIL2PhaseSampler:
    """
    两阶段采样节点。
    
    输入要求：latent 必须从 WanSCAILToVideoMultiRef 等 conditioning 节点来。
    previous_frames 是上一个 chunk 解码后的尾部像素帧。
    
    两阶段条件：split_step > 0 AND split_step < steps AND 有 previous_frames。
    
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
               phase_schedule=False, schedule_exponent=1.0):

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
                schedule_exponent, make_cb=_mk_cb)
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
            full_sigmas = s1.sigmas.to(model1.load_device)
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
                noise_mask=noise_mask, seed=seed,
                callback=_mk_cb())
            out_noise_mask = noise_mask

        # ========== 纯单阶段（无锚定帧）==========
        else:
            samples_out = comfy.sample.sample(
                model=model1, noise=noise, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative=negative,
                latent_image=samples, denoise=denoise,
                noise_mask=None, seed=seed,
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
            logger.info("[WanSCAIL2PhaseSampler] color_align: per-channel scale=%s shift=%s", scale.squeeze().tolist(), shift.squeeze().tolist())

        out = {"samples": samples_out}
        if out_noise_mask is not None:
            out["noise_mask"] = out_noise_mask

        return (out,)

    # ========== phase_schedule: per-step interpolated anchor write-back ==========
    def _sample_phase_schedule(self, model1, model2, positive, negative, latent,
                               anchor_len, anchor_content, noise, steps, cfg,
                               sampler_name, scheduler, denoise, seed,
                               phase1_noise, phase2_noise,
                                schedule_exponent, make_cb):
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

        # full sigma sequence (from model1; explicit sigmas means denoise is not applied twice)
        s_ref = comfy.samplers.KSampler(
            model1, steps=steps, device=device,
            sampler=sampler_name, scheduler=scheduler,
            denoise=denoise, model_options=model1.model_options)
        full_sigmas = s_ref.sigmas.to(device)
        del s_ref

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