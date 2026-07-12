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
import nodes
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
                "model1": ("MODEL", {"tooltip": "Phase 1 模型。"}),
                "positive": ("CONDITIONING", {"tooltip": "正向条件。"}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件。"}),
                "vae": ("VAE", {"tooltip": "VAE，用于锚定帧编解码。"}),
                "latent": ("LATENT", {"tooltip": "输入的 latent（来自 conditioning 节点）。"}),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_step": ("INT", {"default": 2, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "分割步数。0=单阶段；>0且<steps且接入了previous_frames时启用两阶段。"}),
                "phase1_noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Phase 1 锚定帧 noise_mask 值。0=冻结，1=自由。"}),
                "phase2_noise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Phase 2 锚定帧 noise_mask 值。0=冻结，1=自由。"}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 4,
                    "tooltip": "从 previous_frames 尾部取多少帧作为锚定。"}),
            },
            "optional": {
                "model2": ("MODEL", {"optional": True,
                    "tooltip": "Phase 2 使用的模型，不提供时使用 model1。"}),
                "previous_frames": ("IMAGE", {"optional": True,
                    "tooltip": "上一段解码后的尾部像素帧。不接时退化为单阶段。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    EXPERIMENTAL = True
    DESCRIPTION = "SCAIL-2 完整两阶段采样。内部完成锚定帧 VAE 编码、回写、noise_mask 管理。"

    def sample(self, model1, positive, negative, vae, latent, width, height,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               split_step=0, phase1_noise=0.75, phase2_noise=0.25,
               previous_frame_count=5, model2=None, previous_frames=None):

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

        # ========== 两阶段模式 ==========
        if split_step > 0 and split_step < steps and encoded_prev is not None:
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
            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()

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

        out = {"samples": samples_out}
        if out_noise_mask is not None:
            out["noise_mask"] = out_noise_mask

        return (out,)


NODE_CLASS_MAPPINGS = {
    "WanSCAIL2PhaseSampler": WanSCAIL2PhaseSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAIL2PhaseSampler": "Wan SCAIL-2 Phase Sampler",
}