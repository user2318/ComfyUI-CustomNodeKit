"""SCAIL-specific context window node.
Handles the special SCAIL conditioning fields:
- driving_mask_28ch: time in dim=1, flat tensor (B, T, 28, H, W)
- ref_mask_28ch: time in dim=1, (B, N+T, 28, H, W) - keep first N frames
- reference_latents: list in conditioning, keep untouched
- pose_video_latent: wrapped in dict with .cond, time in dim=2

No prefix support — SCAIL's reference_latents / ref_mask_28ch[N] are global
conditions that are automatically retained across all windows via the handler.

Reuses the core infrastructure from Custom_context.py (IndexListContextWindow etc.)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import torch
import logging
import nodes

from .Custom_context import (
    IndexListContextWindow, IndexListContextHandler,
    ContextSchedule, ContextFuseMethod, ContextFuseMethods,
    ContextSchedules,
    get_matching_context_schedule, get_matching_fuse_method,
    get_shape_for_dim, match_weights_to_dim,
    create_prepare_sampling_wrapper, create_sampler_sample_wrapper,
)

import comfy.patcher_extension

if TYPE_CHECKING:
    from comfy.model_base import BaseModel


# ===== Model forward patch: slices SCAIL conditioning at model entry =====
# Conditioning dict items like ref_mask_28ch come in at full length (N+T_full),
# but the model receives window-sliced x. ComfyUI also moves tensor dimensions
# around (28→dim=1). This patch slices ref_mask_latents at model forward entry
# to match the windowed x.
_SCAIL_PATCH_APPLIED = False

def _apply_scail_model_patch():
    global _SCAIL_PATCH_APPLIED
    if _SCAIL_PATCH_APPLIED:
        return
    try:
        import comfy.ldm.wan.model as _mod
        _orig_forward = _mod.SCAILWanModel._forward

        def _patched_forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None,
                             transformer_options={}, pose_latents=None, ref_mask_latents=None,
                             sam_latents=None, **kwargs):
            window = transformer_options.get("context_window", None) if transformer_options else None
            ref_latent_kw = kwargs.get("reference_latent", None)

            # ComfyUI renames conditioning dict key "ref_mask_28ch" → model named param
            # "ref_mask_latents" via Python function argument binding.
            # So ref_mask_latents is already populated and NOT in kwargs.
            # We slice it here to match the windowed x.
            if ref_mask_latents is not None and window is not None and hasattr(window, "index_list"):
                # ref_mask_latents: (B, 28, N+T_full, H, W) — ComfyUI movedim'ed to channel-first
                n_ref = ref_latent_kw.shape[2] if ref_latent_kw is not None else 0
                indices = window.index_list
                ref_part = ref_mask_latents[:, :, :n_ref]
                video_part = ref_mask_latents[:, :, n_ref:]
                logging.info(
                    "[SCAIL_DEBUG] ref_mask_latents: input=%s, n_ref=%d, indices=%s, mean=%.6f",
                    list(ref_mask_latents.shape), n_ref, indices[:5],
                    ref_mask_latents.mean().item()
                )
                # --- 重叠区追踪日志 ---
                if hasattr(window, "original_indices") and len(video_part.shape) >= 3:
                    oi = window.original_indices
                    if oi and len(oi) > 1:
                        first_frame_val = video_part[:, :, oi[0]].mean().item()
                        last_frame_val = video_part[:, :, oi[-1]].mean().item()
                        logging.info(
                            "[SCAIL_OVERLAP] ref_mask overlap boundary: window orig=[%d..%d], "
                            "first_frame_mean=%.6f, last_frame_mean=%.6f",
                            oi[0], oi[-1], first_frame_val, last_frame_val
                        )
                # --- 结束重叠区追踪 ---
                video_part = video_part[:, :, indices]
                ref_mask_latents = torch.cat([ref_part, video_part], dim=2)
                logging.info(
                    "[SCAIL_DEBUG] ref_mask_latents: sliced=%s, mean=%.6f",
                    list(ref_mask_latents.shape),
                    ref_mask_latents.mean().item()
                )

            # driving_mask_28ch is NOT a named param of _forward, stays in kwargs
            _driving_from_kw = kwargs.pop("driving_mask_28ch", None)
            if _driving_from_kw is not None and window is not None and hasattr(window, "index_list"):
                logging.info(
                    "[SCAIL_DEBUG] driving_mask_28ch: input=%s, indices=%s, mean=%.6f",
                    list(_driving_from_kw.shape), window.index_list[:5],
                    _driving_from_kw.mean().item()
                )
                _driving_from_kw = _driving_from_kw[:, window.index_list]
                logging.info(
                    "[SCAIL_DEBUG] driving_mask_28ch: sliced=%s, mean=%.6f",
                    list(_driving_from_kw.shape),
                    _driving_from_kw.mean().item()
                )

            ref_mask_flag = kwargs.pop("ref_mask_flag", None)

            return _orig_forward(self, x, timestep, context, clip_fea=clip_fea,
                                  time_dim_concat=time_dim_concat,
                                  transformer_options=transformer_options,
                                  pose_latents=pose_latents,
                                  ref_mask_latents=ref_mask_latents,
                                  sam_latents=sam_latents,
                                  driving_mask_28ch=_driving_from_kw,
                                  ref_mask_flag=ref_mask_flag,
                                  **kwargs)

        _mod.SCAILWanModel._forward = _patched_forward
        _SCAIL_PATCH_APPLIED = True
        logging.info("[SCAILContext] Model patch applied")
    except Exception as e:
        logging.warning("[SCAILContext] Failed to patch SCAIL model: %s", e)


class SCAILContextHandler(IndexListContextHandler):
    """Context handler with SCAIL-aware conditioning slicing.

    SCAIL conditioning field time-dimension layout:
    - ref_mask_28ch:   (B, N+T, 28, H, W), time in dim=1  (N ref + T video)
    - driving_mask_28ch: (B, T, 28, H, W), time in dim=1
    - pose_video_latent: wrapped .cond (B, C, T, H, W), time in dim=2
    - reference_latents: list[Tensor], keep untouched (global reference)
    - ref_mask_flag: bool, keep untouched
    - clip_vision_output: object, keep untouched
    """

    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor,
                         window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None
        resized_cond = []
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            logging.info(
                "SCAIL context: splitting conds to windows; using region %d for window "
                "(original %d-%d)", region, window.original_indices[0], window.original_indices[-1]
            )
            cond_in = [cond_in[region]]
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    # --- Level 1: plain tensors ---
                    if isinstance(cond_item, torch.Tensor):
                        # SCAIL fields: pass through — model patch handles slicing
                        if key in ("ref_mask_28ch", "driving_mask_28ch"):
                            resized_actual_cond[key] = cond_item.to(device)
                        # Standard: time in dim=self.dim (=2)
                        elif (self.dim < cond_item.ndim
                              and cond_item.size(self.dim) == x_in.size(self.dim)):
                            resized_actual_cond[key] = window.get_tensor(
                                cond_item, device, retain_index_list=self.prefix_indices
                            )
                        # Fallback dim=1
                        elif cond_item.ndim >= 2 and cond_item.size(1) == x_in.size(self.dim):
                            resized_actual_cond[key] = window.get_tensor(
                                cond_item, device, dim=1,
                                retain_index_list=self.prefix_indices
                            )
                        else:
                            resized_actual_cond[key] = cond_item.to(device)

                    # --- Level 2: control objects ---
                    elif key == "control":
                        resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)

                    # --- Level 3: dict items (where SCAIL .cond fields live) ---
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        for cond_key, cond_value in new_cond_item.items():
                            # ----- pose_video_latent (time in dim=2, .cond wrapped) -----
                            if (cond_key == "pose_video_latent"
                                    and hasattr(cond_value, "cond")
                                    and isinstance(cond_value.cond, torch.Tensor)):
                                # CRITICAL: Use range(self.context_length) not window.original_indices!
                                # Conditioning tensors are already offset by video_frame_offset,
                                # so their frame 0 = current chunk's frame 0.
                                # window.original_indices like [5,6,...,25] can exceed the
                                # tensor length (21), causing silent corruption via IndexError
                                # in the try/finally that swallows the exception.
                                indices = range(self.context_length)
                                if cond_value.cond.ndim == 5:
                                    idx = (slice(None), slice(None), list(indices),
                                           slice(None), slice(None))
                                elif cond_value.cond.ndim == 4:
                                    idx = (slice(None), list(indices), slice(None), slice(None))
                                else:
                                    idx = (slice(None), list(indices))
                                sliced = cond_value.cond[idx]
                                logging.info(
                                    "[SCAILContext] pose_video_latent: input=%s, indices=%s, result=%s",
                                    list(cond_value.cond.shape),
                                    f"0..{self.context_length-1}",
                                    list(sliced.shape)
                                )
                                new_cond_item[cond_key] = cond_value._copy_with(sliced)
                                continue

                            # ----- SCAIL dict-level fields: pass through, model patch handles -----
                            if cond_key in ("driving_mask_28ch", "ref_mask_28ch"):
                                if isinstance(cond_value, torch.Tensor):
                                    new_cond_item[cond_key] = cond_value.to(device)
                                    continue
                                elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                    new_cond_item[cond_key] = cond_value
                                    continue

                            # ----- clip_vision_output: passthrough -----

                            # ----- Callback hooks -----
                            handled = False
                            for callback in comfy.patcher_extension.get_all_callbacks(
                                "resize_cond_item", self.callbacks
                            ):
                                result = callback(cond_key, cond_value, window, x_in, device,
                                                  new_cond_item)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                                    break
                            if handled:
                                continue

                            # ----- face_pixel_values (time in pixel frames) -----
                            if (cond_key == "face_pixel_values"
                                    and hasattr(cond_value, "cond")
                                    and isinstance(cond_value.cond, torch.Tensor)):
                                pixel_tensor = cond_value.cond
                                oi = window.original_indices if hasattr(window, "original_indices") and window.original_indices else window.index_list
                                pixel_start = max(oi[0] * 4, 0)
                                pixel_end = min((oi[-1] + 1) * 4, pixel_tensor.shape[2])
                                sliced_pixels = pixel_tensor[:, :, pixel_start:pixel_end, :, :] if pixel_start < pixel_tensor.shape[2] else torch.zeros_like(pixel_tensor[:, :, :0, :, :])
                                new_cond_item[cond_key] = cond_value._copy_with(sliced_pixels)
                                continue

                            # ----- Generic dict-cond handler -----
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim
                                        and cond_value.size(self.dim) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(
                                        cond_value, device,
                                        retain_index_list=self.prefix_indices
                                    )
                            elif hasattr(cond_value, "cond") and isinstance(
                                    cond_value.cond, torch.Tensor):
                                cond_t = cond_value.cond
                                if (self.dim < cond_t.ndim
                                        and cond_t.size(self.dim) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = cond_value._copy_with(
                                        window.get_tensor(cond_t, device,
                                                          retain_index_list=self.prefix_indices)
                                    )
                            elif cond_key == "num_video_frames":
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                                new_cond_item[cond_key].cond = window.context_length
                        resized_actual_cond[key] = new_cond_item

                    # --- Level 4: anything else (pass through) ---
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item
            resized_cond.append(resized_actual_cond)
        return resized_cond


# ==================== Node Definition ====================

class WanSCAILContextWindowsNode:
    """SCAIL 上下文窗口节点。自动处理 conditioning 中不同字段的时间维差异。

    SCAIL 的 reference_latents / ref_mask_28ch[N 帧] 作为全局条件自动保留，
    不需要 prefix 机制。
    """

    @classmethod
    def INPUT_TYPES(s):
        schedule_options = [
            ContextSchedules.UNIFORM_LOOPED,
            ContextSchedules.UNIFORM_STANDARD,
            ContextSchedules.STATIC_STANDARD,
            ContextSchedules.BATCHED,
        ]
        fuse_options = ContextFuseMethods.LIST_STATIC
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "应用上下文窗口的模型（SCAIL 模型）。The model to apply context windows to (SCAIL model)."
                }),
                "context_length": ("INT", {
                    "default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "上下文窗口长度（像素帧）。建议设为 SCAIL 的 length 参数值（如81）。Context window length (pixel frames). Recommended to match SCAIL's length parameter (e.g. 81)."
                }),
                "context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": nodes.MAX_RESOLUTION,
                    "tooltip": "窗口间重叠帧数（像素帧）。Overlap between windows (pixel frames)."
                }),
                "context_schedule": (schedule_options, {
                    "tooltip": "窗口生成策略。static_standard=固定窗口（推荐）。Window generation schedule. static_standard=fixed windows (recommended)."
                }),
                "context_stride": ("INT", {
                    "default": 1, "min": 1, "max": 10,
                    "tooltip": "窗口步进（仅对 uniform 策略有效）。Window stride (only effective for uniform schedules)."
                }),
                "closed_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否闭环窗口（仅 looped 策略有效）。Whether to close the window loop (only effective for looped schedules)."
                }),
                "fuse_method": (fuse_options, {
                    "default": ContextFuseMethods.PYRAMID,
                    "tooltip": "窗口融合方法。Window fusion method."
                }),
                "freenoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "FreeNoise 噪声混洗，改善窗口间融合。FreeNoise noise shuffling, improves inter-window blending."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_context_windows"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "SCAIL 上下文窗口节点。自动处理 conditioning 中不同字段的时间维差异。\n"
        "- reference_latents: 保留全部 N 帧（全局参考，不切片）\n"
        "- pose_video_latent: 按 latent 窗口在 dim=2 切片\n"
        "- driving_mask_28ch: 按 latent 窗口在 dim=1 切片\n"
        "- ref_mask_28ch: ref 部分(N帧)保留 + video 部分按窗口切片"
    )

    def apply_context_windows(self, model, context_length, context_overlap,
                              context_schedule, context_stride, closed_loop,
                              fuse_method, freenoise):
        context_length = max(((context_length - 1) // 4) + 1, 1)
        context_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        _apply_scail_model_patch()

        model = model.clone()
        handler = SCAILContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=2,
            freenoise=freenoise,
            prefix_latent_len=0,
            split_conds_to_windows=False,
            causal_window_fix=False,
        )
        model.model_options["context_handler"] = handler

        create_prepare_sampling_wrapper(model)
        if freenoise:
            create_sampler_sample_wrapper(model)

        logging.info(
            "[SCAILContext] 已应用上下文窗口: len=%d_latent overlap=%d_latent "
            "schedule=%s fuse=%s",
            context_length, context_overlap,
            context_schedule, fuse_method
        )
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanSCAILContextWindows": WanSCAILContextWindowsNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAILContextWindows": "Wan SCAIL Context Windows",
}