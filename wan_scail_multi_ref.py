"""Multi-reference SCAIL / SCAIL-2 nodes: WanSCAILToVideoMultiRef conditioning node
with support for multiple reference images and masks."""

from typing_extensions import override

import torch
import torch.nn.functional as F
import logging

import nodes
import node_helpers
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.sam3.tracker import unpack_masks

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


# Model was trained on these exact colors; deviating degrades multi-identity quality.
DEFAULT_PALETTE = [
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 1.0, 0.0),  # Yellow
]


_SCAIL_ROPE_PATCH_APPLIED = False


def _apply_rope_downsample_patch():
    """Patch WanModel.rope_encode + SCAILWanModel.rope_encode to implement
    'full-res RoPE first, then avg_pool2d downsample' for the pose branch only.

    Original SCAIL-2 path:
    1. Build full-resolution RoPE grid at 2x the pose latent spatial extent
    2. avg_pool2d(2,2) on real/imag parts to downsample the RoPE field
    3. Recombine into final half-res pose RoPE

    Only the pose branch gets the downsampled RoPE — main/reserved frames
    use the standard non-downsampled RoPE.
    """
    global _SCAIL_ROPE_PATCH_APPLIED
    if _SCAIL_ROPE_PATCH_APPLIED:
        return
    try:
        import comfy.ldm.wan.model as _mod

        # ---- Patch WanModel.rope_encode (base class) ----
        # Adds avg_pool2d logic when _scail_rope_downsample flag is set in transformer_options
        _orig_wan_rope = _mod.WanModel.rope_encode

        def _patched_wan_rope(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None,
                              device=None, dtype=None, transformer_options={}):
            need_downsample = transformer_options.get("_scail_rope_downsample", False)
            if not need_downsample:
                return _orig_wan_rope(self, t, h, w, t_start, steps_t, steps_h, steps_w,
                                       device, dtype, transformer_options)

            # --- Full-resolution RoPE generation (2x spatial) ---
            h2 = h * 2
            w2 = w * 2
            steps_h2 = steps_h * 2 if steps_h is not None else None
            steps_w2 = steps_w * 2 if steps_w is not None else None

            freqs_2x = _orig_wan_rope(self, t, h2, w2, t_start, steps_t, steps_h2, steps_w2,
                                       device, dtype, transformer_options)
            # --- Infer spatial decomposition ---
            patch_size = self.patch_size
            t_p = steps_t if steps_t is not None else ((t + patch_size[0] // 2) // patch_size[0])
            h_p_2x = steps_h2 if steps_h2 is not None else ((h2 + patch_size[1] // 2) // patch_size[1])
            w_p_2x = steps_w2 if steps_w2 is not None else ((w2 + patch_size[2] // 2) // patch_size[2])

            # --- avg_pool2d downsample on spatial dimensions ---
            d_rope = freqs_2x.shape[3]
            freqs_grid = freqs_2x.reshape(1, t_p, h_p_2x, w_p_2x, d_rope, 2, 2)

            cos = freqs_grid[..., 0, 0].contiguous()
            sin = freqs_grid[..., 1, 0].contiguous()

            B, Tp, H2, W2, D = cos.shape
            cos_4d = cos.permute(0, 1, 4, 2, 3).reshape(1, Tp * D, H2, W2)
            sin_4d = sin.permute(0, 1, 4, 2, 3).reshape(1, Tp * D, H2, W2)

            cos_pooled = F.avg_pool2d(cos_4d, 2, 2)
            sin_pooled = F.avg_pool2d(sin_4d, 2, 2)

            H, W = cos_pooled.shape[2], cos_pooled.shape[3]

            cos_5d = cos_pooled.reshape(1, Tp, D, H, W).permute(0, 1, 3, 4, 2)
            sin_5d = sin_pooled.reshape(1, Tp, D, H, W).permute(0, 1, 3, 4, 2)

            freqs_out_grid = torch.stack([
                torch.stack([cos_5d, -sin_5d], dim=-1),
                torch.stack([sin_5d,  cos_5d], dim=-1),
            ], dim=-2)

            return freqs_out_grid.reshape(1, Tp * H * W, 1, d_rope, 2, 2)

        _mod.WanModel.rope_encode = _patched_wan_rope

        # ---- Patch SCAILWanModel.rope_encode to inject flag ONLY into pose branch ----
        _orig_scail_rope = _mod.SCAILWanModel.rope_encode

        def _patched_scail_rope(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None,
                                 device=None, dtype=None, pose_latents=None, reference_latent=None,
                                 ref_mask_flag=None, transformer_options={}):
            if pose_latents is None:
                # No pose → no downsample needed, pass through
                return _orig_scail_rope(self, t, h, w, t_start, steps_t, steps_h, steps_w,
                                         device, dtype, pose_latents=None,
                                         reference_latent=reference_latent,
                                         ref_mask_flag=ref_mask_flag,
                                         transformer_options=transformer_options)

            F_pose, H_pose, W_pose = pose_latents.shape[-3], pose_latents.shape[-2], pose_latents.shape[-1]
            ref_t_patches = 0

            # --- Replacement mode path ---
            if ref_mask_flag is not None and not bool(ref_mask_flag):
                REF_ROPE_H = 120.0
                POSE_ROPE_W = 120.0
                if reference_latent is not None:
                    ref_t_patches = (reference_latent.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]
                main_t_patches = t - ref_t_patches
                video_t_start = max(ref_t_patches - 1, 0)

                parts = []
                if ref_t_patches > 0:
                    ref_tf = {"rope_options": {"shift_y": REF_ROPE_H, "shift_x": 0.0, "scale_y": 1.0, "scale_x": 1.0}}
                    parts.append(_mod.WanModel.rope_encode(self, ref_t_patches, h, w, t_start=0, device=device, dtype=dtype, transformer_options=ref_tf))
                if main_t_patches > 0:
                    parts.append(_mod.WanModel.rope_encode(self, main_t_patches, h, w, t_start=video_t_start, device=device, dtype=dtype, transformer_options=transformer_options))
                if F_pose > 0:
                    # Pose branch: generate full-resolution RoPE then avg_pool2d downsample.
                    # No scale/shift on coordinates — dense 0..N-1 grid with x-offset=120.0,
                    # matching WanAnimatePlus commit 22555324 (Original SCAIL-2 path).
                    pose_tf = {"rope_options": {"shift_y": 0.0, "shift_x": 120.0, "scale_y": 1.0, "scale_x": 1.0},
                               "_scail_rope_downsample": True}
                    parts.append(_mod.WanModel.rope_encode(self, F_pose, H_pose, W_pose, t_start=video_t_start, device=device, dtype=dtype, transformer_options=pose_tf))
                return torch.cat(parts, dim=1)

            # --- Animation mode path (default) ---
            if reference_latent is not None:
                ref_t_patches = (reference_latent.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]

            # Main frames: NO downsample
            main_freqs = _mod.WanModel.rope_encode(self, t, h, w, t_start=t_start, steps_t=steps_t, steps_h=steps_h, steps_w=steps_w,
                                                    device=device, dtype=dtype, transformer_options=transformer_options)

            # Pose frames: WITH downsample (only if there are pose frames)
            if F_pose > 0:
                # No scale/shift — dense 0..N-1 grid with x-offset=120.0 (Original SCAIL-2 path)
                pose_tf = {"rope_options": {"shift_y": 0.0, "shift_x": 120.0, "scale_y": 1.0, "scale_x": 1.0},
                           "_scail_rope_downsample": True}
                pose_freqs = _mod.WanModel.rope_encode(self, F_pose, H_pose, W_pose, t_start=t_start + ref_t_patches,
                                                        device=device, dtype=dtype, transformer_options=pose_tf)
                return torch.cat([main_freqs, pose_freqs], dim=1)

            return main_freqs

        _mod.SCAILWanModel.rope_encode = _patched_scail_rope

        _SCAIL_ROPE_PATCH_APPLIED = True
        logging.info("[WanSCAIL_MultiRef] RoPE pose-only downsample patch applied")
    except Exception as e:
        logging.warning("[WanSCAIL_MultiRef] Failed to patch RoPE downsample: %s", e)


def _unpack(track_data):
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_frame_cx_area(masks_bool):
    first = masks_bool[0].float()
    H, W = first.shape[-2], first.shape[-1]
    n_pixels = H * W
    grid_x = torch.arange(W, device=first.device, dtype=first.dtype).view(1, W)
    area = first.sum(dim=(-1, -2)).clamp_(min=1)
    cx = (first * grid_x).sum(dim=(-1, -2)) / area
    return (cx / W).tolist(), (area / n_pixels).tolist()


def _subset_track_data(track_data, obj_indices):
    out = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not obj_indices:
        out["packed_masks"] = None
        if "scores" in out:
            out["scores"] = []
        return out
    out["packed_masks"] = packed[:, obj_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        out["scores"] = [scores[i] for i in obj_indices if i < len(scores)]
    return out


def _render_colored_masks(track_data, background="black"):
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device, dtype=dtype)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(N_obj)],
        device=device, dtype=dtype,
    )
    masks_full = unpack_masks(packed.to(device)).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = F.interpolate(
        masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest"
    ).view(T, N_obj, H, W) > 0.5
    any_mask = masks_full.any(dim=1)
    obj_idx_map = masks_full.to(torch.uint8).argmax(dim=1)
    color_overlay = colors[obj_idx_map]
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


def _extract_mask_to_28ch(rgb_video):
    """Colored RGB mask (T, H, W, 3) in [0, 1] -> SCAIL-2 28-channel binary latent
    (1, T_lat, 28, H_lat, W_lat). 7 per-color binary channels (white/r/g/b/y/m/c)
    threshold-extracted at 225/255, 8x spatial downsample, 4-frame temporal stacking."""
    T, H, W, _ = rgb_video.shape
    _ON_THRESH = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B,    # white
        R * nG * nB,  # red
        nR * G * nB,  # green
        nR * nG * B,  # blue
        R * G * nB,   # yellow
        R * nG * B,   # magenta
        nR * G * B,   # cyan
    ], dim=1)
    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = torch.nn.functional.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')
    T_latent = (T - 1) // 4 + 1
    padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, 28, H_lat, W_lat)
    return out.unsqueeze(0)


class WanSCAILToVideoMultiRef(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILToVideoMultiRef",
            category="model/conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True, tooltip="用于姿态 conditioning 的视频，将降低分辨率至主视频的一半。Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="仅 SCAIL-2。与 pose_video 同分辨率的 SAM3 逐人彩色遮罩视频。SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="仅 SCAIL-2。False=动画模式(pose_video_mask应为黑色背景)；True=替换模式(pose_video_mask应为白色背景)。SCAIL-2 only. False = Animation Mode (black bg). True = Replacement Mode (white bg)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="姿态 latent 的强度。Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="姿态 conditioning 的起始步。Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="姿态 conditioning 的结束步。End step of the pose conditioning."),
                io.Combo.Input("ref_encoding_mode", options=["1+4n批量", "逐帧编码", "混合编码"], default="1+4n批量", tooltip="参考图编码模式。1+4n批量=第1张×1其余×4后一次编码（默认,更接近训练）；逐帧编码=每张图独立编码后时间维拼接（高保真）；混合编码=前N-1张1+4n批量+最后1张逐帧编码（兼顾清晰度与减轻段间跳变）。Reference encoding mode: 1+4n batch=1st×1 rest×4 then batch encode (default, closer to training); per-frame=independent encode then temporal concat (high fidelity); hybrid=first N-1 as 1+4n batch + last frame independently."),
                io.Image.Input("reference_image", optional=True, tooltip="参考图输入，多张参考图以批次形式传入 (N, H, W, 3)。每张图独立编码后作为单独的 reference_latents 添加。Reference image(s). For multiple references, pass a batch of images (N, H, W, 3). Each image is independently encoded and added as a separate reference_latents."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="仅 SCAIL-2。与 reference_image 同分辨率的彩色参考遮罩。应与 reference_image 数量匹配。SCAIL-2 only. Colored reference mask(s) at the same resolution as reference_image. Should match the reference_image count."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="用于 conditioning 的 CLIP 视觉特征。模型使用拉伸缩放至宽高比进行训练。CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="当前块开始的累计输出帧偏移。从上一块的 video_frame_offset 输出接入。Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="用于锚定的上一块尾帧数。SCAIL-2 训练使用 5（81帧块，76帧步长）。Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="仅 SCAIL-2。上一块的完整解码输出。仅最后 previous_frame_count 帧用作扩展锚定。SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="生成尺寸的空 latent 张量。Empty latent of the generation size."),
                io.Int.Output(display_name="video_frame_offset", tooltip="调整后的偏移量 + 长度。接入下一块。Adjusted offset + length. Wire into the next chunk."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, ref_encoding_mode="1+4n批量",
                reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None) -> io.NodeOutput:
        # RoPE downsample patch: full-res RoPE first, then avg_pool2d downsample.
        # Matches WanAnimatePlus commit 22555324 (Original SCAIL-2 path).
        _apply_rope_downsample_patch()

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        noise_mask = None

        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        prev_trimmed = None
        if previous_frames is not None and previous_frames.shape[0] > 0:
            prev_trimmed = previous_frames[-previous_frame_count:]
            video_frame_offset -= prev_trimmed.shape[0]
            video_frame_offset = max(0, video_frame_offset)

        # ----- Multi-reference image handling -----
        # Two encoding modes:
        #   "1+4n批量" : 1st ref ×1, each additional ×4 → batch → single VAE encode
        #   "逐帧编码"  : each ref independently upsample + VAE encode → cat in time dim
        concat_ref_latent = None
        if reference_image is not None:
            num_refs = reference_image.shape[0]
            if ref_encoding_mode == "1+4n批量":
                # 1+4n batch encoding
                ref_pixel_parts = [reference_image[0:1]]
                if num_refs > 1:
                    for i in range(1, num_refs):
                        ref_pixel_parts.append(reference_image[i:i+1].repeat(4, 1, 1, 1))
                ref_pixels = torch.cat(ref_pixel_parts, dim=0)  # (4N-3, H, W, 3)

                # Replacement Mode mask compositing
                if replacement_mode and reference_image_mask is not None:
                    mask_parts = [reference_image_mask[0:1]]
                    if reference_image_mask.shape[0] > 1:
                        for i in range(1, min(num_refs, reference_image_mask.shape[0])):
                            mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                    while len(mask_parts) < len(ref_pixel_parts):
                        mask_parts.append(mask_parts[-1])
                    ref_masks = torch.cat(mask_parts, dim=0)
                    rm = comfy.utils.common_upscale(
                        ref_masks.movedim(-1, 1), width, height, "nearest-exact", "center"
                    ).movedim(1, -1)
                    is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_pixels.dtype)
                    ref_pixels = ref_pixels * is_char

                # Batch upsample + VAE encode (bicubic for upsampling, matching official node)
                ref_pixels = comfy.utils.common_upscale(
                    ref_pixels.movedim(-1, 1), width, height, "bicubic", "center"
                ).movedim(1, -1)
                concat_ref_latent = vae.encode(ref_pixels[:, :, :, :3])
                # concat_ref_latent: (1, 16, 4N-3, H/8, W/8)
            elif ref_encoding_mode == "逐帧编码":
                ref_latent_parts = []
                for i in range(num_refs):
                    single_ref = comfy.utils.common_upscale(
                        reference_image[i:i+1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    if replacement_mode and reference_image_mask is not None:
                        mask_idx = min(i, reference_image_mask.shape[0] - 1)
                        rm = comfy.utils.common_upscale(
                            reference_image_mask[mask_idx:mask_idx+1].movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(single_ref.dtype)
                        single_ref = single_ref * is_char
                    ref_latent = vae.encode(single_ref[:, :, :, :3])
                    ref_latent_parts.append(ref_latent)
                concat_ref_latent = torch.cat(ref_latent_parts, dim=2)
                # concat_ref_latent: (1, 16, N, H/8, W/8)
            else:  # "混合编码" — 前N-1张1+4n批量 + 最后1张逐帧编码
                if num_refs == 1:
                    # 只有1张参考图时退化为逐帧编码
                    single_ref = comfy.utils.common_upscale(
                        reference_image[0:1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    concat_ref_latent = vae.encode(single_ref[:, :, :, :3])
                else:
                    # 前 N-1 张: 1+4n 批量
                    batch_parts = [reference_image[0:1]]
                    for i in range(1, num_refs - 1):
                        batch_parts.append(reference_image[i:i+1].repeat(4, 1, 1, 1))
                    ref_pixels_batch = torch.cat(batch_parts, dim=0)
                    if replacement_mode and reference_image_mask is not None:
                        mask_parts = [reference_image_mask[0:1]]
                        if reference_image_mask.shape[0] > 1:
                            for i in range(1, min(num_refs - 1, reference_image_mask.shape[0])):
                                mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                        while len(mask_parts) < len(batch_parts):
                            mask_parts.append(mask_parts[-1])
                        ref_masks = torch.cat(mask_parts, dim=0)
                        rm = comfy.utils.common_upscale(
                            ref_masks.movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_pixels_batch.dtype)
                        ref_pixels_batch = ref_pixels_batch * is_char
                    ref_pixels_batch = comfy.utils.common_upscale(
                        ref_pixels_batch.movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    batch_latent = vae.encode(ref_pixels_batch[:, :, :, :3])
                    # 最后1张: 逐帧编码
                    last_ref = comfy.utils.common_upscale(
                        reference_image[-1:].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    if replacement_mode and reference_image_mask is not None:
                        mask_idx = min(num_refs - 1, reference_image_mask.shape[0] - 1)
                        rm = comfy.utils.common_upscale(
                            reference_image_mask[mask_idx:mask_idx+1].movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(last_ref.dtype)
                        last_ref = last_ref * is_char
                    last_latent = vae.encode(last_ref[:, :, :, :3])
                    # 拼接: (1, 16, N-1) + (1, 16, 1) = (1, 16, N)
                    concat_ref_latent = torch.cat([batch_latent, last_latent], dim=2)
                # concat_ref_latent: (1, 16, N, H/8, W/8)

        if concat_ref_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [concat_ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [concat_ref_latent]}, append=True)

        if clip_vision_output is not None:
            logging.info(f"[DEBUG wan_scail_multi_ref] clip_vision_output type: {type(clip_vision_output)}")
            if hasattr(clip_vision_output, 'penultimate_hidden_states'):
                hs = clip_vision_output.penultimate_hidden_states
                logging.info(f"[DEBUG wan_scail_multi_ref] penultimate_hidden_states shape: {hs.shape}, dtype: {hs.dtype}, mean={hs.mean().item():.6f}, std={hs.std().item():.6f}, min={hs.min().item():.6f}, max={hs.max().item():.6f}")
            if hasattr(clip_vision_output, 'last_hidden_state'):
                hs = clip_vision_output.last_hidden_state
                logging.info(f"[DEBUG wan_scail_multi_ref] last_hidden_state shape: {hs.shape}, mean={hs.mean().item():.6f}, std={hs.std().item():.6f}")
            if hasattr(clip_vision_output, 'image_embeds'):
                ie = clip_vision_output.image_embeds
                logging.info(f"[DEBUG wan_scail_multi_ref] image_embeds shape: {ie.shape}, mean={ie.mean().item():.6f}, std={ie.std().item():.6f}")
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]
        if pose_video_mask is not None:
            if pose_video_mask.shape[0] <= video_frame_offset:
                pose_video_mask = None
            else:
                pose_video_mask = pose_video_mask[video_frame_offset:]

        # Truncate pose+mask jointly to the shorter of the two, capped at length.
        ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if ts:
            T_kept = ((min(min(ts), length) - 1) // 4) * 4 + 1
            if pose_video is not None:
                pose_video = pose_video[:T_kept]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[:T_kept]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            logging.info(
                "[WanSCAIL_COND] driving_mask_28ch: shape=%s, val_range=[%.6f, %.6f], mean=%.6f",
                list(driving_mask_28ch.shape),
                driving_mask_28ch.min().item(), driving_mask_28ch.max().item(),
                driving_mask_28ch.mean().item()
            )
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        # ----- Multi-reference mask handling (ref_mask_28ch) -----
        # Each ref mask after _extract_mask_to_28ch: (1, 1, 28, H_lat, W_lat)
        # Concatenate N masks in temporal dim (dim=1) → (1, N, 28, H_lat, W_lat)
        # Then append zeros for the T_lat video frames → (1, N+T_lat, 28, H_lat, W_lat)
        # In "1+4n批量" mode, the mask should be expanded at pixel level (1st ×1, rest ×4)
        # before _extract_mask_to_28ch, to match how the reference image is expanded.
        # Both paths produce N frames of latent mask because _extract_mask_to_28ch does 4→1 compression:
        #   "1+4n批量": (4N-3) pixel frames → 4→1 → N latent frames
        #   "逐帧编码": N pixel frames → 4→1 → N latent frames
        if reference_image_mask is not None:
            if ref_encoding_mode == "1+4n批量":
                # 1+4n pixel-level expansion: 1st mask ×1, rest ×4
                mask_parts = [reference_image_mask[0:1]]
                if reference_image_mask.shape[0] > 1:
                    for i in range(1, reference_image_mask.shape[0]):
                        mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                masks_expanded = torch.cat(mask_parts, dim=0)  # (4N-3, H, W, 3)
                # Upsample all together, then single _extract_mask_to_28ch call gets (4N-3)
                # frames → internal 4→1 compression → N latent frames
                ref_mask_hw = comfy.utils.common_upscale(
                    masks_expanded.movedim(-1, 1), width, height, "bicubic", "center"
                ).movedim(1, -1)
                ref_mask_concat = _extract_mask_to_28ch(ref_mask_hw)  # (1, N, 28, H_lat, W_lat)
            else:  # "逐帧编码"
                ref_mask_t_parts = []
                for i in range(reference_image_mask.shape[0]):
                    ref_mask_hw = comfy.utils.common_upscale(
                        reference_image_mask[i:i+1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw)  # (1, 1, 28, H_lat, W_lat)
                    ref_mask_t_parts.append(ref_mask_1f)
                # N pixel frames → N single-frame encodes → concat in dim=1 → (1, N, 28, H_lat, W_lat)
                ref_mask_concat = torch.cat(ref_mask_t_parts, dim=1)

            # Pad with very small value to avoid zero boundary in ref_mask_28ch
            T_lat = latent.shape[2]
            zeros = torch.full(
                (1, T_lat, 28, ref_mask_concat.shape[-2], ref_mask_concat.shape[-1]),
                fill_value=0.001,
                device=ref_mask_concat.device, dtype=ref_mask_concat.dtype
            )
            ref_mask_full = torch.cat([ref_mask_concat, zeros], dim=1)  # (1, N_ref_mask+T_lat, 28, H_lat, W_lat)

            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_full})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_full})

        logging.info(
            "[WanSCAIL_COND] video_frame_offset=%d, length=%d, latent_len=%d, pose_video=%s, pose_video_mask=%s",
            video_frame_offset, length, latent.shape[2],
            list(pose_video.shape) if pose_video is not None else "None",
            list(pose_video_mask.shape) if pose_video_mask is not None else "None",
        )

        if prev_trimmed is not None:
            pf = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            prev_latent = vae.encode(pf[:, :, :, :3])
            prev_latent_frames  = min(prev_latent.shape[2], latent.shape[2])
            latent[:, :, :prev_latent_frames] = prev_latent[:, :, :prev_latent_frames].to(latent.dtype)
            noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
                                    device=latent.device, dtype=latent.dtype)
            noise_mask[:, :, :prev_latent_frames] = 0.0
            logging.info(
                "[WanSCAIL] hard cut at latent frame %d",
                prev_latent_frames
            )
        out_latent = {"samples": latent}
        if noise_mask is not None:
            out_latent["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_latent, video_frame_offset + length)


class SCAIL2ColoredMaskMultiRef(io.ComfyNode):
    """渲染 SAM3 追踪数据为彩色遮罩（支持多参考图）。
    与 WanSCAILToVideoMultiRef 配合使用。

    reference_image_mask 始终为黑底（模型约定）。
    pose_video_mask 背景色按 replacement_mode 变化：
    - False(动画模式)=黑底
    - True(替换模式)=白底
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMaskMultiRef_CK",
            display_name="Create SCAIL-2 Colored Mask (MultiRef)",
            category="conditioning/video_models/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data", tooltip="驱动姿态视频的 SAM3 追踪数据。将渲染为 pose_video_mask 输出。SAM3 track of the driving pose video."),
                SAM3TrackData.Input("ref_track_data", optional=True, tooltip="参考图的 SAM3 追踪数据。SAM3 track of the reference image."),
                io.String.Input("object_indices", default="",
                                tooltip="逗号分隔的人物索引列表（如 '0,2,3'）。同时应用于参考图和姿态视频遮罩。空=全部。Comma-separated person indices (e.g. '0,2,3'). Applied to both ref and pose masks. Empty=all."),
                io.Combo.Input("sort_by", options=["none", "left_to_right", "area"], default="left_to_right",
                               tooltip="调色板分配到追踪对象的顺序（同时应用于参考图和姿态视频，确保同一个人物颜色一致）。left_to_right=由左至右（按首帧质心）；area=按面积从大到小；none=保持 SAM3 原始顺序。"),
                io.Boolean.Input("replacement_mode", default=False,
                                 tooltip="False=动画模式(pose_video_mask=黑底, reference_image_mask=白底)；True=替换模式(pose_video_mask=白底, reference_image_mask=黑底)。"),
            ],
            outputs=[
                io.Image.Output("pose_video_mask"),
                io.Image.Output("reference_image_mask"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, driving_track_data, object_indices, sort_by, replacement_mode, ref_track_data=None):
        def _prep(td):
            masks_bool = _unpack(td)
            if sort_by != "none" and masks_bool is not None:
                cx, area = _first_frame_cx_area(masks_bool)
                if sort_by == "left_to_right":
                    order = sorted(range(len(cx)), key=lambda i: cx[i])
                else:  # "area"
                    order = sorted(range(len(area)), key=lambda i: -area[i])
                td = _subset_track_data(td, order)
            if object_indices.strip():
                indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
                packed = td.get("packed_masks")
                n_obj = packed.shape[1] if packed is not None else 0
                indices = [i for i in indices if 0 <= i < n_obj]
                td = _subset_track_data(td, indices)
            return td

        drv = _prep(driving_track_data)
        # Animation: driving=black, ref=white. Replacement: driving=white, ref=black.
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")
        ref_bg = "black" if replacement_mode else "white"

        if ref_track_data is not None:
            ref = _prep(ref_track_data)
            reference_image_mask = _render_colored_masks(ref, ref_bg)
        else:
            H, W = drv["orig_size"]
            fill_value = 1.0 if ref_bg == "white" else 0.0
            reference_image_mask = torch.full((1, H, W, 3), fill_value, device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())

        return io.NodeOutput(mask_video, reference_image_mask)


class SCAILExtensionMultiRef(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanSCAILToVideoMultiRef,
            SCAIL2ColoredMaskMultiRef,
        ]


async def comfy_entrypoint() -> SCAILExtensionMultiRef:
    return SCAILExtensionMultiRef()


NODE_CLASS_MAPPINGS = {
    "WanSCAILToVideoMultiRef": WanSCAILToVideoMultiRef,
    "SCAIL2ColoredMaskMultiRef_CK": SCAIL2ColoredMaskMultiRef,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAILToVideoMultiRef": "Wan SCAIL To Video (Multi Ref)",
    "SCAIL2ColoredMaskMultiRef_CK": "Create SCAIL-2 Colored Mask (MultiRef)",
}
