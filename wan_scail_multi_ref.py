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
                io.Image.Input("pose_video", optional=True, tooltip="Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="SCAIL-2 only. False = Animation Mode (pose_video_mask should have black background). True = Replacement Mode (pose_video_mask should have white background)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="End step of the pose conditioning."),
                io.Combo.Input("ref_encoding_mode", options=["1+4n批量", "逐帧编码"], default="1+4n批量", tooltip="参考图编码模式。1+4n批量=第1张×1其余×4后一次编码（默认,更接近训练）；逐帧编码=每张图独立编码后时间维拼接（高保真）。"),
                io.Image.Input("reference_image", optional=True, tooltip="Reference image(s). For multiple references, pass a batch of images (N, H, W, 3). Each image is independently encoded and added as a separate reference_latent."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="SCAIL-2 only. Colored reference mask(s) at the same resolution as reference_image. Should match the reference_image count."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Empty latent of the generation size."),
                io.Int.Output(display_name="video_frame_offset", tooltip="Adjusted offset + length. Wire into the next chunk."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, ref_encoding_mode="1+4n批量",
                reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None) -> io.NodeOutput:
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

                # Batch upsample + VAE encode
                ref_pixels = comfy.utils.common_upscale(
                    ref_pixels.movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                concat_ref_latent = vae.encode(ref_pixels[:, :, :, :3])
                # concat_ref_latent: (1, 16, 4N-3, H/8, W/8)
            else:  # "逐帧编码"
                ref_latent_parts = []
                for i in range(num_refs):
                    single_ref = comfy.utils.common_upscale(
                        reference_image[i:i+1].movedim(-1, 1), width, height, "area", "center"
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

        if concat_ref_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [concat_ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [concat_ref_latent]}, append=True)

        if clip_vision_output is not None:
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
    """Render SAM3 tracks for the driving pose video and (optionally) the reference
    image(s) into the colored masks WanSCAILToVideoMultiRef consumes. Shared `sort_by`
    across both outputs guarantees identity K maps to the same color on both
    sides, for multi-person workflow consistency.
    reference_image_mask is always rendered black-bg (model convention)
    pose_video_mask bg follows replacement_mode: black = Animation Mode, white = Replacement Mode
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMaskMultiRef",
            display_name="Create SCAIL-2 Colored Mask (Multi-Ref)",
            category="conditioning/video_models/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data", tooltip="SAM3 track of the driving pose video. Will be rendered into the pose_video_mask output."),
                SAM3TrackData.Input("ref_track_data", optional=True,
                                    tooltip="SAM3 track of the reference image(s). If multiple references, pass multiple ref_track_data inputs (future)."),
                io.String.Input("object_indices", default="",
                                tooltip="Comma-separated list of person indices to include (e.g. '0,2,3'). Applied to both reference and pose video masks. Empty = all."),
                io.Combo.Input("sort_by", options=["none", "left_to_right", "area"], default="left_to_right",
                               tooltip="Order in which palette colors are assigned to the tracked objects (applied to both reference and pose video so each identity keeps the same color). left_to_right = leftmost object (by first-frame centroid) gets the first color; area = biggest object (by first-frame mask area) gets the first color; none = keep SAM3's order."),
                io.Boolean.Input("replacement_mode", default=False,
                                 tooltip="False = mask_video has black bg (Animation Mode). True = white bg (Replacement Mode). Set the matching replacement_mode on WanSCAILToVideoMultiRef. reference_image_mask is always black-bg regardless."),
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
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")

        if ref_track_data is not None:
            ref = _prep(ref_track_data)
            reference_image_mask = _render_colored_masks(ref, "black")
        else:
            H, W = drv["orig_size"]
            reference_image_mask = torch.zeros(1, H, W, 3, device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())

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
    "SCAIL2ColoredMaskMultiRef": SCAIL2ColoredMaskMultiRef,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAILToVideoMultiRef": "Wan SCAIL To Video (Multi Ref)",
    "SCAIL2ColoredMaskMultiRef": "Create SCAIL-2 Colored Mask (Multi Ref)",
}