# ==============================
# WanAnimate2ToVideoCustom 节点（官方 WanAnimate2ToVideo + 一代 WanAnimateToVideoCustom 融合版）
#
# 保留一代特性：
#   - 多参考图（逐帧独立编码，兼容 EverAnimate LoRA；原模式 1+4n 已移除）
#   - fix/legacy/vanilla 掩码处理、黑帧检测+过渡、尾帧分段掩码、中性灰混合
#   - concat_latent 输出（与 Custom_context 衔接）
#
# 保留官方二代特性：
#   - pose_strength / pose_start_percent / pose_end_percent（pose 时间窗口，窗口外跳过 pose 分支）
#   - reference_image_strength（参考帧身份保真度）
#   - positive_pose / clip_vision_output_pose（pose 分支独立条件）
#   - 单通道 concat mask（二代 WAN21.concat_cond 会自动扩展为 4 通道）
#
# 去掉（二代架构不适用/不需要）：
#   - face_video / face_strength、background_video、character_mask、prev_latent 潜变量直传
#
# 注意：pose 分支采用一代的「前补 trim_latent-1 帧零 latent」逻辑，
#       以满足二代模型 pose_latents.shape[2] == f_gen - 1 的帧数校验（多参考图时必需）。
# ==============================

import torch
import logging
import node_helpers
import nodes
import comfy.utils
import comfy.model_management


class WanAnimate2ToVideoCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "正向提示词 conditioning。Positive prompt conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "负向提示词 conditioning。Negative prompt conditioning."}),
                "vae": ("VAE", {"tooltip": "Wan 模型的 VAE，用于编码参考图和窗口帧。VAE for the Wan model, used to encode reference images and window frames."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "生成视频的宽度（像素），必须是 16 的倍数。Width of the generated video (pixels), must be a multiple of 16."}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "生成视频的高度（像素），必须是 16 的倍数。Height of the generated video (pixels), must be a multiple of 16."}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "生成视频的帧数（像素帧），必须是 4 的倍数。Number of frames for the generated video (pixel frames), must be a multiple of 4."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "批量大小，通常保持为 1。Batch size, usually keep at 1."}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "从上一块携带的最大帧数（RGB 图像），用于块间接续。Maximum frames carried from the previous chunk (RGB images), used for inter-chunk continuity."}),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "当前 chunk 的帧偏移量，从上一块的 video_frame_offset 输出接入。Frame offset of the current chunk, connected from the previous chunk's video_frame_offset output."}),
                "transition_width": ("INT", {"default": 0, "min": 0, "max": 128, "step": 4, "tooltip": "fix 模式下黑帧区域的过渡区宽度，0=禁用过渡。Transition zone width for black frame areas in fix mode, 0=disable transition."}),
                "mode": (["fix", "legacy", "vanilla"], {"default": "vanilla", "tooltip": "掩码模式：vanilla=官方行为(fix/legacy不动)，fix=黑帧检测+过渡，legacy=尾帧渐变。Mask mode: vanilla=official behavior, fix=black frame detection+transition, legacy=tail frame fade."}),
                "tail_frame_count": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "legacy 模式下尾帧处理帧数，0=禁用。Number of tail frames to process in legacy mode, 0=disabled."}),
                "tail_start_strength": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式尾帧起始强度。Tail frame start strength in legacy mode."}),
                "tail_end_strength": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式尾帧结束强度。Tail frame end strength in legacy mode."}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP Vision 输出，用于参考图语义理解。CLIP Vision output for reference image semantic understanding."}),
                "reference_image": ("IMAGE", {"tooltip": "参考图输入。接参考图选择器的 selected_images（排序后的原始图片，节点内部自动逐帧编码）。Reference image input. Connect to selected_images from ReferenceImageSelector (node encodes each frame independently)."}),
                "pose_video": ("IMAGE", {"tooltip": "姿态视频帧序列，用于姿态引导。Pose video frame sequence for pose guidance."}),
                "positive_pose": ("CONDITIONING", {"tooltip": "姿态分支专属提示词（描述动作而非角色），默认回退到 positive。Prompt for the pose-video branch, describing the motion rather than the character. Defaults to positive."}),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "姿态视频首帧的 CLIP Vision 输出，默认回退到 clip_vision_output。CLIP vision of the pose video's first frame. Defaults to clip_vision_output."}),
                "continue_motion": ("IMAGE", {"tooltip": "上一块的末尾 RGB 帧，用于块间运动接续。Last RGB frames of the previous chunk for inter-chunk motion continuity."}),
                "mid_frame": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "legacy 模式中间帧锚点位置，-1=不使用。Mid-frame anchor position in legacy mode, -1=not used."}),
                "mid_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式中间帧锚点强度。Mid-frame anchor strength in legacy mode."}),
                "neutral_mix_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "legacy 模式掩码=0时的中性灰混合比例。Neutral gray mix ratio when mask=0 in legacy mode."}),
                "neutral_mix_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "legacy 模式掩码=1时的中性灰混合比例。Neutral gray mix ratio when mask=1 in legacy mode."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "缩放姿态视频对运动的影响强度。1.0=训练行为；0.0=静音但不会完全移除；>1=增强。Scales the pose video's influence on the motion. 1.0 is the trained behavior; 0.0 mutes it but does not fully remove it; above amplifies."}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "姿态影响的起始采样百分比，窗口外整个姿态分支被跳过（可提速）。Sampling percent at which the pose influence starts. Outside the window the pose branch is skipped entirely."}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "姿态影响的结束采样百分比，例如 0.7 可在保留舞蹈编排的同时放松细节。Sampling percent at which the pose influence ends. Motion is mostly established early, so e.g. 0.7 can loosen fine detail while keeping the choreography."}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "生成帧对参考图 latent 帧的注意力强度。<1 放松身份/外观一致性，>1 收紧。Scales how strongly generated frames attend to the reference image's latent frame. Below 1.0 loosens identity/appearance adherence, above tightens it against drift."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent", "trim_image", "video_frame_offset", "concat_latent")
    FUNCTION = "process"
    CATEGORY = "WanLoop/整合节点"

    def process(self, positive, negative, vae, width, height, length, batch_size,
                continue_motion_max_frames, video_frame_offset, transition_width,
                mode, tail_frame_count, tail_start_strength, tail_end_strength,
                pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0,
                reference_image_strength=1.0,
                clip_vision_output=None, reference_image=None,
                pose_video=None, positive_pose=None, clip_vision_output_pose=None,
                continue_motion=None,
                mid_frame=-1, mid_strength=0.5,
                neutral_mix_min=0.0, neutral_mix_max=1.0):

        if pose_start_percent > pose_end_percent:
            raise ValueError("pose_start_percent ({}) must not be greater than pose_end_percent ({}).".format(
                pose_start_percent, pose_end_percent))

        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0
        ref_motion_latent_length = 0

        # ==================== 参考图像处理（逐帧独立编码，兼容 EverAnimate LoRA） ====================
        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        # 逐帧独立 VAE 编码（原模式 1+4n 批量编码在 animate2 上表现不佳，已移除）
        encoded_list = []
        num_ref_frames = reference_image.shape[0]
        for i in range(num_ref_frames):
            single_img = reference_image[i:i + 1]
            single_img = comfy.utils.common_upscale(
                single_img.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            single_latent = vae.encode(single_img[:, :, :, :3])
            encoded_list.append(single_latent)
        concat_latent_image = torch.cat(encoded_list, dim=2)
        trim_latent += concat_latent_image.shape[2]
        # 构造占位 image 用于后续（生成区，不会与参考区混用）
        image = torch.ones((length, height, width, 3)) * 0.5

        # ==================== continue_motion 处理（多帧续帧） ====================
        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset = max(0, video_frame_offset - continue_motion.shape[0])
            continue_motion = comfy.utils.common_upscale(
                continue_motion[-length:].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            image = torch.ones((length, height, width, continue_motion.shape[-1]),
                               device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        # ==================== clip_vision 处理 ====================
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # ==================== reference_image_strength（官方二代） ====================
        if reference_image_strength != 1.0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_strength": reference_image_strength})
            negative = node_helpers.conditioning_set_values(negative, {"reference_strength": reference_image_strength})

        # ==================== pose_video 处理（前补 trim_latent-1 帧零对齐参考 anchor） ====================
        pose_values = {}
        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            if pose_video.shape[0] < length:
                pose_video = torch.cat([pose_video, pose_video[-1:].repeat(length - pose_video.shape[0], 1, 1, 1)], dim=0)
            else:
                pose_video = pose_video[:length]
            pose_video = comfy.utils.common_upscale(
                pose_video.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3])

            # 前补 latent 帧以对齐 concat_latent_image 的 anchor 部分（多参考图必需）
            delta = trim_latent - 1
            if delta > 0:
                zero_pad = torch.zeros_like(pose_video_latent[:, :, :1, :, :]).repeat(1, 1, delta, 1, 1)
                pose_video_latent = torch.cat([zero_pad, pose_video_latent], dim=2)
                logging.info("[WanAnimate2ToVideoCustom] pose 分支已前补 %d 帧零 latent，总长度 = %d",
                             delta, pose_video_latent.shape[2])
            pose_values["pose_video_latent"] = pose_video_latent

        pose_clip = clip_vision_output_pose if clip_vision_output_pose is not None else clip_vision_output
        if pose_clip is not None:
            pose_values["clip_vision_output_pose"] = pose_clip

        pose_cond = positive_pose if positive_pose is not None else positive
        if len(pose_cond) > 0:
            pose_values["cross_attn_pose"] = pose_cond[0][0]

        if pose_strength != 1.0:
            pose_values["pose_strength"] = pose_strength

        if pose_start_percent > 0.0 or pose_end_percent < 1.0:
            # 通过 cond 时间窗口控制 pose 分支启停：窗口外完全跳过 pose 分支（可提速）
            def windowed(cond):
                parts = node_helpers.conditioning_set_values(cond, {**pose_values, "start_percent": pose_start_percent, "end_percent": pose_end_percent})
                if pose_start_percent > 0.0:
                    parts = parts + node_helpers.conditioning_set_values(cond, {"start_percent": 0.0, "end_percent": pose_start_percent})
                if pose_end_percent < 1.0:
                    parts = parts + node_helpers.conditioning_set_values(cond, {"start_percent": pose_end_percent, "end_percent": 1.0})
                return parts
            positive = windowed(positive)
            negative = windowed(negative)
        else:
            positive = node_helpers.conditioning_set_values(positive, pose_values)
            negative = node_helpers.conditioning_set_values(negative, pose_values)

        # ==================== image 区像素处理（fix/legacy 会修改 image） ====================
        float_mask = None
        if continue_motion is not None and mode != "vanilla":
            N = continue_motion.shape[0]
            is_black = (continue_motion.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0] < 1e-6)
            float_mask = is_black.float().clone()

            if mode == "fix" and transition_width > 0:
                trans = transition_width
                black_regions = []
                i = 0
                while i < N:
                    if is_black[i]:
                        start = i
                        while i < N and is_black[i]:
                            i += 1
                        end = i - 1
                        black_regions.append((start, end))
                    else:
                        i += 1

                for start, end in black_regions:
                    left_start = max(0, start - trans)
                    left_end = start - 1
                    for j in range(left_start, left_end + 1):
                        dist = start - j
                        value = max(0.0, 1.0 - (dist - 1) / trans)
                        float_mask[j] = max(float_mask[j].item(), value)

                    right_start = end + 1
                    right_end = min(N - 1, end + trans)
                    for j in range(right_start, right_end + 1):
                        dist = j - end
                        value = max(0.0, 1.0 - (dist - 1) / trans)
                        float_mask[j] = max(float_mask[j].item(), value)

                # fix 模式：纯黑帧硬替换为中性灰
                for i in range(N):
                    if is_black[i]:
                        image[i] = 0.5

            if mode == "legacy" and tail_frame_count > 0:
                tail_len = min(tail_frame_count, N)
                if tail_len > 0:
                    use_mid = (mid_frame >= 1 and mid_frame <= tail_len)
                    if use_mid:
                        mid_idx = mid_frame - 1
                        strengths_first = torch.linspace(tail_start_strength, mid_strength, mid_idx + 1, device=float_mask.device)
                        strengths_second = torch.linspace(mid_strength, tail_end_strength, tail_len - mid_idx, device=float_mask.device)
                        strengths = torch.cat([strengths_first, strengths_second[1:]])
                        logging.info("[WanAnimate2ToVideoCustom][legacy模式] 使用中间帧锚点：第%d帧强度=%.2f",
                                     mid_frame, mid_strength)
                    else:
                        strengths = torch.linspace(tail_start_strength, tail_end_strength, tail_len, device=float_mask.device)

                    for i in range(tail_len):
                        frame_idx = N - tail_len + i
                        float_mask[frame_idx] = max(float_mask[frame_idx].item(), strengths[i].item())

                    if neutral_mix_min != 0.0 or neutral_mix_max != 1.0:
                        logging.info("[WanAnimate2ToVideoCustom][legacy模式] 中性灰混合比例范围：掩码0时=%.2f, 掩码1时=%.2f",
                                     neutral_mix_min, neutral_mix_max)
                    for i in range(tail_len):
                        frame_idx = N - tail_len + i
                        mask_strength = strengths[i].item()
                        mix_alpha = neutral_mix_min + (neutral_mix_max - neutral_mix_min) * mask_strength
                        if mix_alpha > 0:
                            original = image[frame_idx]
                            neutral = torch.full_like(original, 0.5)
                            image[frame_idx] = original * (1 - mix_alpha) + neutral * mix_alpha

        # ==================== image 区编码 + 二代 concat（参考区 + image 区） ====================
        image_latent = vae.encode(image[:, :, :, :3])
        concat_latent_image = torch.cat((concat_latent_image, image_latent), dim=2)

        # ==================== 二代 1 通道 mask（0=已知，1=未知） ====================
        mask = torch.ones((1, 1, latent_length + trim_latent, latent_height, latent_width),
                          device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        mask[:, :, :trim_latent + ref_motion_latent_length] = 0.0

        # ==================== fix/legacy 掩码渐变（像素帧粒度 -> latent 帧粒度重映射） ====================
        if float_mask is not None:
            for k in range(ref_motion_latent_length):
                p0 = k * 4
                p1 = min(p0 + 4, N)
                if p0 >= N:
                    break
                strength = float_mask[p0:p1].max().item()
                if strength > 0.0:
                    mask[0, 0, trim_latent + k, :, :] = torch.maximum(
                        mask[0, 0, trim_latent + k, :, :],
                        torch.full((latent_height, latent_width), strength,
                                   device=mask.device, dtype=mask.dtype))

        concat_meta = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask,
        }
        positive = node_helpers.conditioning_set_values(positive, concat_meta)
        negative = node_helpers.conditioning_set_values(negative, concat_meta)

        latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
                             device=comfy.model_management.intermediate_device())
        out_latent = {"samples": latent}

        # ----- 对外输出 trim 值（兼容模式原逻辑） -----
        out_trim_latent = trim_latent
        out_trim_image = max(0, ref_motion_latent_length * 4 - 3)

        return (positive, negative, out_latent, out_trim_latent, out_trim_image, video_frame_offset + length,
                {"samples": concat_latent_image})


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "WanAnimate2ToVideoCustom": WanAnimate2ToVideoCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimate2ToVideoCustom": "WanAnimate2 To Video (自定义)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
