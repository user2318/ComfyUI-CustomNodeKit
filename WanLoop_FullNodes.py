# ==============================
# WanAnimateToVideoCustom 节点（独立精简版）
# 支持 pose/face 强度控制、尾帧分段掩码、中性灰混合、上下文模式等
# ==============================

import torch
import logging
import node_helpers
import nodes
import comfy.utils


def _wan_4n1_video_to_latent(video_frames):
    """视频帧数转潜在空间帧数（Wan 编码器每4帧压缩为1帧）"""
    if video_frames < 1:
        return 1
    return (video_frames - 1) // 4 + 1


class WanAnimateToVideoCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "正向提示词 conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "负向提示词 conditioning"}),
                "vae": ("VAE", {"tooltip": "Wan 模型的 VAE，用于编码参考图和窗口帧"}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "生成视频的宽度（像素），必须是 16 的倍数"}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "生成视频的高度（像素），必须是 16 的倍数"}),
                "length": ("INT", {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "生成视频的帧数（像素帧），必须是 4 的倍数"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "批量大小，通常保持为 1"}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "从上一块携带的最大帧数（RGB 图像），用于块间接续"}),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "当前 chunk 的帧偏移量，从上一块的 video_frame_offset 输出接入"}),
                "transition_width": ("INT", {"default": 0, "min": 0, "max": 128, "step": 4, "tooltip": "fix 模式下黑帧区域的过渡区宽度，0=禁用过渡"}),
                "mode": (["fix", "legacy", "vanilla"], {"default": "vanilla", "tooltip": "掩码模式：vanilla=官方行为(fix/legacy不动)，fix=黑帧检测+过渡，legacy=尾帧渐变"}),
                "tail_frame_count": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "legacy 模式下尾帧处理帧数，0=禁用"}),
                "tail_start_strength": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式尾帧起始强度"}),
                "tail_end_strength": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式尾帧结束强度"}),
                "ref_mode": (["原模式", "兼容模式"], {"default": "原模式", "tooltip": "原模式=内部1+4n排列后批量编码(接selected_images)；兼容模式=逐帧独立编码(接selected_images，兼容EverAnimate LoRA)"}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP Vision 输出，用于参考图语义理解"}),
                "reference_image": ("IMAGE", {"tooltip": "参考图输入。接参考图选择器的 selected_images（排序后的原始图片，节点内部自动处理1+4n或逐帧编码）"}),
                "face_video": ("IMAGE", {"tooltip": "面部视频帧序列，用于面部引导"}),
                "pose_video": ("IMAGE", {"tooltip": "姿态视频帧序列，用于姿态引导"}),
                "continue_motion": ("IMAGE", {"tooltip": "上一块的末尾 RGB 帧，用于块间运动接续"}),
                "background_video": ("IMAGE", {"tooltip": "背景视频帧序列，用于替换/增强背景"}),
                "character_mask": ("MASK", {"tooltip": "角色遮罩，用于精确控制角色区域的保护"}),
                "yaw_angles": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "tooltip": "偏航角序列（像素帧级别）"}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "面部引导强度，0=关闭面部引导"}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "姿态引导强度，0=关闭姿态引导，>1=增强姿态影响"}),
                "mid_frame": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "legacy 模式中间帧锚点位置，-1=不使用"}),
                "mid_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy 模式中间帧锚点强度"}),
                "neutral_mix_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "legacy 模式掩码=0时的中性灰混合比例"}),
                "neutral_mix_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "legacy 模式掩码=1时的中性灰混合比例"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT", "LATENT", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent", "trim_image", "video_frame_offset", "concat_latent", "latent_yaw_angles")
    FUNCTION = "process"
    CATEGORY = "WanLoop/整合节点"

    def process(self, positive, negative, vae, width, height, length, batch_size,
                continue_motion_max_frames, video_frame_offset, transition_width,
                mode, tail_frame_count, tail_start_strength, tail_end_strength,
                ref_mode,
                clip_vision_output=None, reference_image=None,
                face_video=None, pose_video=None, continue_motion=None,
                background_video=None, character_mask=None,
                face_strength=1.0, pose_strength=1.0,
                mid_frame=-1, mid_strength=0.5,
                neutral_mix_min=0.0, neutral_mix_max=1.0,
                yaw_angles=None):

        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0
        ref_motion_latent_length = 0

        # ----- 参考图像处理（根据模式走不同路径） -----
        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        if ref_mode == "兼容模式":
            # 兼容模式：逐帧独立 VAE 编码（模拟 EverAnimate 单帧编码方式）
            encoded_list = []
            num_ref_frames = reference_image.shape[0]
            for i in range(num_ref_frames):
                single_img = reference_image[i:i+1]
                single_img = comfy.utils.common_upscale(
                    single_img.movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                single_latent = vae.encode(single_img[:, :, :, :3])
                encoded_list.append(single_latent)
            concat_latent_image = torch.cat(encoded_list, dim=2)
            mask = torch.zeros((1, 4, concat_latent_image.shape[-3],
                                concat_latent_image.shape[-2],
                                concat_latent_image.shape[-1]),
                               device=concat_latent_image.device, dtype=concat_latent_image.dtype)
            trim_latent += concat_latent_image.shape[2]
            # 构造占位 image 用于后续兼容（不会被实际用于编码）
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            # 原模式：内部做 1+4n 排列后再批量编码
            # 第1张参考图 ×1 + 其余参考图(每张×4)
            ref_batch_parts = [reference_image[0:1]]
            if reference_image.shape[0] > 1:
                for i in range(1, reference_image.shape[0]):
                    ref_batch_parts.append(reference_image[i:i+1].repeat(4, 1, 1, 1))
            ref_batch = torch.cat(ref_batch_parts, dim=0)

            # 限制到 length 帧
            ref_batch = ref_batch[:length]
            image = comfy.utils.common_upscale(ref_batch.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.zeros((1, 4, concat_latent_image.shape[-3], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                               device=concat_latent_image.device, dtype=concat_latent_image.dtype)
            trim_latent += concat_latent_image.shape[2]

        # ----- continue_motion 处理 -----
        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(continue_motion[-length:].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            image = torch.ones((length, height, width, continue_motion.shape[-1]), device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        # ----- clip_vision 处理 -----
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # ----- pose_video 处理 -----
        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            ref_pixel_frames = reference_image.shape[0] if reference_image is not None else 1
            ref_pixel_frames = min(ref_pixel_frames, length)

            if pose_video.shape[0] < length:
                pose_video = torch.cat([
                    pose_video,
                    pose_video[-1:].repeat(length - pose_video.shape[0], 1, 1, 1)
                ], dim=0)
            else:
                pose_video = pose_video[:length]

            pose_video = comfy.utils.common_upscale(
                pose_video.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3])

            # 前补 latent 帧以对齐 concat_latent_image 的 anchor 部分
            delta = trim_latent - 1

            if delta > 0:
                zero_pad = torch.zeros_like(pose_video_latent[:, :, :1, :, :]).repeat(1, 1, delta, 1, 1)
                pose_video_latent = torch.cat([zero_pad, pose_video_latent], dim=2)
                logging.info("[PoseVideo] 已在开头补 %d 帧零 latent，总长度 = %d", delta, pose_video_latent.shape[2])

            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        # ----- face_video 处理 -----
        if face_video is not None:
            if face_video.shape[0] <= video_frame_offset:
                face_video = None
            else:
                face_video = face_video[video_frame_offset:]
        if face_video is not None:
            if reference_image is not None:
                # 前补像素帧以对齐 concat_latent_image 的 anchor 部分
                prefix_frames_to_add = max(0, (trim_latent - 1) * 4)
                if prefix_frames_to_add > 0:
                    neutral_pad = torch.full((prefix_frames_to_add, *face_video.shape[1:]), -1.0, device=face_video.device, dtype=face_video.dtype)
                    face_video = torch.cat([neutral_pad, face_video], dim=0)
                    logging.info("[FaceVideo] 已在开头补 %d 帧 -1.0，总帧数 = %d", prefix_frames_to_add, face_video.shape[0])

            face_video = comfy.utils.common_upscale(face_video.movedim(-1, 1), 512, 512, "area", "center") * 2.0 - 1.0
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            if face_strength != 1.0:
                face_video = face_video * face_strength
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

            logging.info("[FaceVideo] 最终传给模型的总帧数: %d", face_video.shape[2])

        # ----- 背景和遮罩处理 -----
        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        # EverAnimate 兼容的保护帧数（character_mask/background_video 偏移量用）
        continue_motion_latents = ref_motion_latent_length
        continue_motion_frames = continue_motion.shape[0] if continue_motion is not None else 0
        effective_frame_offset = max(0, int(video_frame_offset) - continue_motion_frames)
        protected_frames = ref_motion_latent_length * 4
        if background_video is not None:
            if background_video.shape[0] > effective_frame_offset:
                background_video = background_video[effective_frame_offset:]
                background_video = comfy.utils.common_upscale(background_video[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]

        # ----- 基于 continue_motion 内容生成掩码 -----
        mask_refmotion = torch.ones((1, 1, latent_length * 4, latent_height, latent_width),
                                     device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        if continue_motion is not None:
            N = continue_motion.shape[0]

            if mode == "vanilla":
                # vanilla 模式：完全复刻官方行为，无条件 mask=0
                ref_motion_latent_length = ((N - 1) // 4) + 1
                mask_refmotion[:, :, :ref_motion_latent_length * 4, :, :] = 0.0
            else:
                # fix / legacy 模式：保留自定义的黑帧检测和尾帧处理
                is_black = (continue_motion.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0] < 1e-6)
                binary_mask = is_black.float()
                float_mask = binary_mask.clone().float()

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
                            float_mask[j] = max(float_mask[j], value)

                        right_start = end + 1
                        right_end = min(N - 1, end + trans)
                        for j in range(right_start, right_end + 1):
                            dist = j - end
                            value = max(0.0, 1.0 - (dist - 1) / trans)
                            float_mask[j] = max(float_mask[j], value)

                    # fix 模式：纯黑帧硬替换为中性灰
                    for i in range(N):
                        if binary_mask[i] == 1:
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
                            logging.info("[legacy模式] 使用中间帧锚点：第%d帧强度=%.2f，尾帧强度序列长度=%d", mid_frame, mid_strength, tail_len)
                        else:
                            strengths = torch.linspace(tail_start_strength, tail_end_strength, tail_len, device=float_mask.device)
                            logging.info("[legacy模式] 未启用中间帧锚点，使用线性插值")

                        for i in range(tail_len):
                            frame_idx = N - tail_len + i
                            float_mask[frame_idx] = max(float_mask[frame_idx].item(), strengths[i].item())

                        if neutral_mix_min != 0.0 or neutral_mix_max != 1.0:
                            logging.info("[legacy模式] 中性灰混合比例范围：掩码0时=%.2f, 掩码1时=%.2f", neutral_mix_min, neutral_mix_max)
                        for i in range(tail_len):
                            frame_idx = N - tail_len + i
                            mask_strength = strengths[i].item()
                            mix_alpha = neutral_mix_min + (neutral_mix_max - neutral_mix_min) * mask_strength
                            if mix_alpha > 0:
                                original = image[frame_idx]
                                neutral = torch.full_like(original, 0.5)
                                image[frame_idx] = original * (1 - mix_alpha) + neutral * mix_alpha

                float_mask = float_mask.to(device=concat_latent_image.device)
                expanded_float_mask = float_mask.view(1, 1, N, 1, 1)
                mask_refmotion[:, :, :N, :, :] = expanded_float_mask

        # ----- character_mask 处理 -----
        if character_mask is not None:
            if character_mask.shape[0] > effective_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
                else:
                    character_mask = character_mask[effective_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1)
                    character_mask = character_mask.movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(character_mask[:, :, :length],
                                                            concat_latent_image.shape[-1],
                                                            concat_latent_image.shape[-2],
                                                            "nearest-exact", "center")
                if character_mask.shape[2] > protected_frames:
                    mask_refmotion[:, :, protected_frames:character_mask.shape[2]] = character_mask[:, :, protected_frames:]

        # ----- 拼接最终的 concat_latent_image -----
        concat_latent_image = torch.cat((concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2)

        mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
                             device=comfy.model_management.intermediate_device())
        out_latent = {"samples": latent}

        trim_image = max(0, ref_motion_latent_length * 4 - 3)

        # ----- pose_strength 强度控制（conditioning 层缩放） -----
        if pose_strength != 1.0:
            pos_list = []
            for cond in positive:
                c = cond[0]
                cond_dict = cond[1].copy()
                if "pose_video_latent" in cond_dict:
                    cond_dict["pose_video_latent"] = cond_dict["pose_video_latent"] * pose_strength
                    logging.info(f"[WanAnimateToVideoCustom] pose_strength={pose_strength}, "
                               f"pose_video_latent_shape={cond_dict['pose_video_latent'].shape}")
                pos_list.append([c, cond_dict])
            positive = pos_list

            neg_list = []
            for cond in negative:
                c = cond[0]
                cond_dict = cond[1].copy()
                if "pose_video_latent" in cond_dict:
                    cond_dict["pose_video_latent"] = cond_dict["pose_video_latent"] * pose_strength
                neg_list.append([c, cond_dict])
            negative = neg_list

        # ----- latent_yaw_angles 下采样 -----
        if yaw_angles is not None:
            latent_yaw_angles = self._downsample_yaw_to_latent(yaw_angles, latent_length + trim_latent)
        else:
            latent_yaw_angles = None

        return (positive, negative, out_latent, trim_latent, trim_image, video_frame_offset + length, {"samples": concat_latent_image}, latent_yaw_angles)

    # ==================== 辅助方法 ====================

    def _downsample_yaw_to_latent(self, yaw_angles, total_latent_len):
        """将像素帧偏航角下采样为 latent 帧偏航角（每4帧取首值）"""
        if yaw_angles is None:
            return [0.0] * total_latent_len

        # 展平为 list[float]
        if isinstance(yaw_angles, (int, float)):
            yaw_list = [float(yaw_angles)]
        elif isinstance(yaw_angles, list):
            yaw_list = [float(v) for v in yaw_angles]
        elif isinstance(yaw_angles, torch.Tensor):
            yaw_list = yaw_angles.flatten().tolist()
        else:
            yaw_list = []

        if len(yaw_list) == 0:
            return [0.0] * total_latent_len

        # 每 4 个像素帧取一个 latent 帧的值（取第一个像素帧的角度）
        latent_yaw = []
        for i in range(total_latent_len):
            pixel_idx = i * 4
            if pixel_idx < len(yaw_list):
                latent_yaw.append(float(yaw_list[pixel_idx]))
            else:
                latent_yaw.append(float(yaw_list[-1]))

        return latent_yaw


# ==============================
# 节点注册（仅保留此节点）
# ==============================
NODE_CLASS_MAPPINGS = {
    "WanAnimateToVideoCustom": WanAnimateToVideoCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimateToVideoCustom": "WanAnimate To Video (自定义)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]