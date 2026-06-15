"""
SCAIL Sparse Attention & Causal Attention node.
Patches WanSelfAttention to apply attention masks that limit interactions
between pose/ref/main tokens, reducing degradation in long video generation.

V2: Mask caching — builds mask once per denoising step, reuses across all 32 layers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import logging

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher


# ==============================================================================
#  Attention Mask Builders
# ==============================================================================

def build_mask_none(S: int, **kwargs) -> torch.Tensor | None:
    return None


def build_mask_pose_no_main(S: int, *, pose_start: int, **kwargs) -> torch.Tensor | None:
    """Pose tokens cannot attend to main tokens (only ref & each other)."""
    mask = torch.ones(1, 1, S, S, dtype=torch.bool)
    mask[:, :, pose_start:, :pose_start] = False
    return mask


def build_mask_ref_no_late_main(S: int, *, ref_len: int, main_start: int, main_end: int, **kwargs) -> torch.Tensor | None:
    """Ref tokens cannot attend to late half of main tokens."""
    mask = torch.ones(1, 1, S, S, dtype=torch.bool)
    main_mid = main_start + (main_end - main_start) // 2
    mask[:, :, :ref_len, main_mid:] = False
    return mask


def build_mask_fast_video(S: int, *, ref_len: int, main_start: int, main_end: int, pose_start: int, **kwargs) -> torch.Tensor | None:
    """Combined fast-sparse mask — vectorized:
    - Ref → only first 1/4 of main
    - Main → ref + local window around self
    - Pose → ref + pose only"""
    window = kwargs.get("window_size", 16)
    mask = torch.zeros(1, 1, S, S, dtype=torch.bool)

    main_quarter = main_start + (main_end - main_start) // 4
    mask[:, :, :ref_len, :ref_len] = True
    mask[:, :, :ref_len, main_start:main_quarter] = True

    main_len = main_end - main_start
    idx = torch.arange(main_len, device=mask.device)
    local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= window
    mask[:, :, main_start:main_end, main_start:main_end] = local_mask.unsqueeze(0).unsqueeze(0)

    mask[:, :, main_start:main_end, :ref_len] = True
    mask[:, :, pose_start:, :ref_len] = True
    mask[:, :, pose_start:, pose_start:] = True
    return mask


def build_mask_causal_with_global(S: int, *, ref_len: int, causal_window: int = -1, **kwargs) -> torch.Tensor | None:
    """Prefix-causal: ref is global (bidirectional), main+pose are causal.
    Suitable for single-segment generation or previous_frames chaining."""
    mask = torch.zeros(1, 1, S, S, dtype=torch.bool)
    mask[:, :, :ref_len, :] = True  # ref → all

    if causal_window > 0:
        d = causal_window - 1
        band = torch.triu(torch.ones(1, 1, S - ref_len, S - ref_len, dtype=torch.bool),
                          diagonal=-d)
        mask[:, :, ref_len:, ref_len:] = band
    else:
        mask[:, :, ref_len:, ref_len:] = torch.tril(
            torch.ones(1, 1, S - ref_len, S - ref_len, dtype=torch.bool))

    mask[:, :, ref_len:, :ref_len] = True  # → ref
    return mask


MASK_BUILDERS = {
    "none": build_mask_none,
    "pose_no_main": build_mask_pose_no_main,
    "ref_no_late_main": build_mask_ref_no_late_main,
    "fast_video": build_mask_fast_video,
    "causal_with_global_prefix": build_mask_causal_with_global,
}

MASK_OPTIONS = [
    "none",
    "pose_no_main",
    "ref_no_late_main",
    "fast_video",
    "causal_with_global_prefix",
]

MASK_LABELS = {
    "none": "无 (全注意力)",
    "pose_no_main": "姿态不看主帧 (Pose→No Main)",
    "ref_no_late_main": "参考不看后半段主帧 (Ref→No Late Main)",
    "fast_video": "快速视频 (Fast Video)",
    "causal_with_global_prefix": "因果+全局前缀 (Causal+Prefix)",
}


# ==============================================================================
#  Token Boundary — read from transformer_options (injected by SCAILWanModel patch)
# ==============================================================================

def _get_boundaries(transformer_options: dict, x) -> dict:
    """Return {S, ref_len, main_start, main_end, pose_start, pose_len}."""
    ref_len = transformer_options.get("_sparse_ref_len", 0)
    main_len = transformer_options.get("_sparse_main_len", 0)
    pose_len = transformer_options.get("_sparse_pose_len", 0)

    if ref_len > 0 or main_len > 0 or pose_len > 0:
        return dict(
            S=ref_len + main_len + pose_len,
            ref_len=ref_len,
            main_start=ref_len,
            main_end=ref_len + main_len,
            pose_start=ref_len + main_len,
            pose_len=pose_len,
        )

    # Fallback: no boundary info available
    grid = transformer_options.get("grid_sizes")
    if grid is not None:
        t, h, w = grid
        main_tokens = t * h * w
        return dict(S=main_tokens, ref_len=0, main_start=0, main_end=main_tokens,
                    pose_start=main_tokens, pose_len=0)

    return dict(S=x.shape[1], ref_len=0, main_start=0, main_end=x.shape[1],
                pose_start=x.shape[1], pose_len=0)


# ==============================================================================
#  Global patch state
# ==============================================================================

_PATCH_APPLIED = False
_SPARSE_CONFIG: dict = {}

_CACHE_KEY = "_sparse_mask_cache"


def _get_tokens_per_frame(transformer_options: dict) -> int:
    """Get number of tokens per frame from grid_sizes in transformer_options."""
    grid = transformer_options.get("grid_sizes", None)
    if grid is not None:
        _, h, w = grid
        return h * w
    return 1  # fallback — no meaningful conversion


def _build_mask_once(transformer_options: dict, x) -> torch.Tensor | None:
    """Build the sparse mask once and cache it in transformer_options."""
    cached = transformer_options.get(_CACHE_KEY)
    if cached is not None:
        return cached

    cfg = _SPARSE_CONFIG
    builder = MASK_BUILDERS.get(cfg.get("mask_type"))
    if builder is None:
        return None

    bounds = _get_boundaries(transformer_options, x)
    if bounds["S"] == 0:
        return None

    kw = dict(cfg)
    kw.pop("enabled", None)
    kw.pop("mask_type", None)

    tpf = _get_tokens_per_frame(transformer_options)
    if "window_size" in kw:
        kw["window_size"] = kw["window_size"] * tpf
    if "causal_window" in kw and kw["causal_window"] > 0:
        kw["causal_window"] = kw["causal_window"] * tpf

    kw.update(bounds)
    S = kw.pop("S", 0)

    mask = builder(S=S, **kw)
    if mask is not None:
        transformer_options[_CACHE_KEY] = mask.to(x.device, dtype=torch.bool)
    return mask


def _apply_all_patches():
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        import comfy.ldm.wan.model as wan_mod
        import comfy.ldm.modules.attention as attn_mod

        _orig_self_attn = wan_mod.WanSelfAttention.forward

        def _patched_self_attn(self, x, freqs, transformer_options={}):
            if not _SPARSE_CONFIG.get("enabled"):
                return _orig_self_attn(self, x, freqs, transformer_options)

            attn_mask = _build_mask_once(transformer_options, x)
            if attn_mask is None:
                return _orig_self_attn(self, x, freqs, transformer_options)

            transformer_options["_sparse_mask"] = attn_mask
            try:
                return _orig_self_attn(self, x, freqs, transformer_options)
            finally:
                transformer_options.pop("_sparse_mask", None)

        wan_mod.WanSelfAttention.forward = _patched_self_attn

        _orig_opt_attn = attn_mod.optimized_attention

        def _patched_opt_attn(q, k, v, heads=8, mask=None, transformer_options={}):
            sparse_mask = transformer_options.get("_sparse_mask")
            if sparse_mask is not None:
                fm = torch.full(sparse_mask.shape, float('-inf'),
                                dtype=q.dtype, device=q.device)
                fm.masked_fill_(sparse_mask, 0.0)
                mask = fm if mask is None else torch.where(mask < 0, mask, fm)
            return _orig_opt_attn(q, k, v, heads=heads, mask=mask,
                                  transformer_options=transformer_options)

        attn_mod.optimized_attention = _patched_opt_attn

        try:
            _chain_to_fwd = wan_mod.SCAILWanModel._forward

            def _scail_fwd(self, x, timestep, context, clip_fea=None,
                           time_dim_concat=None, transformer_options={},
                           pose_latents=None, ref_mask_latents=None,
                           sam_latents=None, **kwargs):
                transformer_options.pop(_CACHE_KEY, None)

                ref_lat = kwargs.get("reference_latent", None)
                if ref_lat is not None:
                    rpt = (ref_lat.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]
                    rph = (ref_lat.shape[3] + (self.patch_size[1] // 2)) // self.patch_size[1]
                    rpw = (ref_lat.shape[4] + (self.patch_size[2] // 2)) // self.patch_size[2]
                    transformer_options["_sparse_ref_len"] = rpt * rph * rpw

                mpt = (x.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]
                mph = (x.shape[3] + (self.patch_size[1] // 2)) // self.patch_size[1]
                mpw = (x.shape[4] + (self.patch_size[2] // 2)) // self.patch_size[2]
                transformer_options["_sparse_main_len"] = mpt * mph * mpw

                if pose_latents is not None:
                    pp = pose_latents.shape[-3:]
                    ppt = (pp[0] + (self.patch_size[0] // 2)) // self.patch_size[0]
                    pph = (pp[1] + (self.patch_size[1] // 2)) // self.patch_size[1]
                    ppw = (pp[2] + (self.patch_size[2] // 2)) // self.patch_size[2]
                    transformer_options["_sparse_pose_len"] = ppt * pph * ppw
                else:
                    transformer_options["_sparse_pose_len"] = 0

                return _chain_to_fwd(
                    self, x, timestep, context, clip_fea=clip_fea,
                    time_dim_concat=time_dim_concat,
                    transformer_options=transformer_options,
                    pose_latents=pose_latents,
                    ref_mask_latents=ref_mask_latents,
                    sam_latents=sam_latents,
                    **kwargs
                )

            wan_mod.SCAILWanModel._forward = _scail_fwd
        except Exception:
            logging.warning("[SparseAttn] SCAILWanModel._forward not available")

        _PATCH_APPLIED = True
        logging.info("[SparseAttn] Patches applied (mask cached across layers)")

    except Exception as e:
        logging.warning("[SparseAttn] Patch failed: %s", e)


# ==============================================================================
#  Node
# ==============================================================================

class WanSCAILSparseAttention:
    """
    为 SCAIL 模型应用稀疏注意力掩码，在 attention 计算时限制
    pose/ref/main 各部分之间的交互，以减轻续接劣化并节省计算。

    无论是否使用 WanSCAILContextWindows 均可独立工作。

    模式说明:
    - 无 (none): 不修改注意力，等同于 bypass
    - 姿态不看主帧 (pose_no_main): 只限制 pose token 不 attend 主帧。不改变帧间关系，适合上下文采样。
    - 参考不看后半段主帧 (ref_no_late_main): 限制 ref token 不 attend 后半段主帧。轻量，可配合其他模式。
    - 快速视频 (fast_video): 组合约束 — ref只看前1/4, main局部窗口, pose不看main。
    - 因果+全局前缀 (causal_with_global_prefix): ref 全局双向可见，main+pose causal。
      适合不需要上下文窗口、用 previous_frames 接续的场景。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要应用稀疏注意力的 SCAIL 模型。The SCAIL model to apply sparse attention to."}),
                "mask_type": (MASK_OPTIONS, {
                    "default": "pose_no_main",
                    "tooltip": "注意力掩码模式:\n"
                               "  无 (全注意力): 不修改注意力\n"
                               "  姿态不看主帧 (Pose→No Main): pose token 不 attend 主帧。推荐配合上下文采样。\n"
                               "  参考不看后半段主帧 (Ref→No Late Main): ref 不 attend 后半段主帧\n"
                               "  快速视频 (Fast Video): 组合局部窗口+约束\n"
                               "  因果+全局前缀 (Causal+Prefix): ref 全局,其余 causal。推荐配合 previous_frames 接续。\n"
                               "Attention mask type:\n"
                               "  none: don't modify attention\n"
                               "  pose_no_main: pose tokens don't attend main. Recommended with context sampling.\n"
                               "  ref_no_late_main: ref doesn't attend late half of main\n"
                               "  fast_video: combined local window + constraints\n"
                               "  causal_with_global_prefix: ref global, rest causal. Recommended with previous_frames chaining.",
                }),
            },
            "optional": {
                "causal_window": ("INT", {
                    "default": -1, "min": -1, "max": 512,
                    "tooltip": "Causal 滑窗大小，单位：帧 (仅 causal_with_global_prefix 模式有效。-1=严格 causal, >0=允许回头看 N 帧)。Causal window size in frames (only effective in causal_with_global_prefix mode. -1=strict causal, >0=allow looking back N frames).",
                }),
                "local_window": ("INT", {
                    "default": 1, "min": 1, "max": 512,
                    "tooltip": "局部窗口大小，单位：帧（仅 fast_video 模式使用。1=前后各看 1 帧，2=前后各看 2 帧）。Local window size in frames (only used in fast_video mode. 1=look 1 frame ahead/behind, 2=look 2 frames).",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model/attention"
    EXPERIMENTAL = True
    DESCRIPTION = "为 SCAIL 模型应用稀疏注意力掩码，减轻视频续接劣化并节省计算。Apply sparse attention masks to the SCAIL model, reducing video continuation degradation and saving computation."

    def apply(self, model, mask_type, causal_window=-1, local_window=1):
        _apply_all_patches()

        global _SPARSE_CONFIG
        _SPARSE_CONFIG = {
            "enabled": True,
            "mask_type": mask_type,
            "causal_window": causal_window,
            "window_size": local_window,
        }

        model = model.clone()
        model.model_options["sparse_attn_config"] = dict(_SPARSE_CONFIG)

        logging.info("[SparseAttn] Applied: mask_type=%s causal_window=%d local_window=%d",
                     mask_type, causal_window, local_window)
        return (model,)


# ==============================================================================
#  ComfyUI Mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "WanSCAILSparseAttention": WanSCAILSparseAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAILSparseAttention": "Wan SCAIL Sparse Attention",
}