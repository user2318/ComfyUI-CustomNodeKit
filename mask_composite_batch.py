import torch
import numpy as np
from PIL import Image


class MaskCompositeRef:
    """遮罩批次合成节点 — 用于 WanSCAILToVideo 的 reference_image 预处理

    本节点专门处理参考图（reference_image）及其对应的遮罩图（reference_image_mask），
    输出可直接接入 WanSCAILToVideo 的 reference_image 和 reference_image_mask 输入。

    根据 replacement_mode 分为两种工作模式：

    **替换模式（replacement_mode=True）：**
    - 遮罩为黑底彩色（人物非黑），检测黑色区域作为背景
    - 忽略 backgrounds 输入
    - 自动剔除遮罩全黑的帧（原图+遮罩同时剔除）
    - 将剩余原图的背景替换为纯黑
    - 输出保留原始彩色遮罩

    **动作迁移模式（replacement_mode=False）：**
    - 遮罩为白底彩色（人物非白），检测白色区域作为背景
    - 支持 backgrounds 背景图合成（第1张用于合成，其余插入序列）
    - 批次不一致时自动截断并给出警告
    - 支持首尾帧跳过控制
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "原图批次 [B, H, W, C]。输入参考图序列，顺序需与 masks 一一对应（节点不校验排序一致性）。Source image batch [B, H, W, C]. The order must correspond one-to-one with masks (node does not validate ordering)."
                }),
                "replacement_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True=替换模式：遮罩为黑底彩色(检测纯黑区域为背景)，忽略背景图输入，自动剔除遮罩全黑帧。False=动作迁移模式：遮罩为白底彩色(检测纯白区域为背景)，与背景图合成。Replacement mode: True=mask has black bg, ignores backgrounds, auto-removes all-black mask frames; False=animation mode: mask has white bg, composites with backgrounds."
                }),
                "process_first_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅动作迁移模式生效。是否对第1张原图也进行遮罩合成。False时第1张保留不动。Only valid in animation mode. Whether to apply mask composite to the first image. When False, the first image is kept unchanged."
                }),
                "process_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "仅动作迁移模式生效。True时最后1张原图参与合成。Only valid in animation mode. Whether to apply mask composite to the last image. When True, the last image participates in compositing."
                }),
                "remove_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "替换模式：关闭时仅做全黑帧剔除+透传，不做背景替换合成。动作迁移模式：关闭时仅做尺寸对齐和背景插入，不做遮罩合成。Whether to remove background. Replacement mode: when disabled, only removes all-black mask frames and passes through. Animation mode: when disabled, only aligns sizes and inserts backgrounds."
                }),
            },
            "optional": {
                "masks": ("IMAGE", {
                    "tooltip": "遮罩批次 [B, H, W, C]，顺序必须与 images 一一对应（节点不校验排序一致性）。替换模式：黑底彩色，纯黑(RGB<10/255)区域为背景，全黑帧会被自动剔除。动作迁移模式：白底彩色，纯白(RGB>245/255)区域为背景。未连接时透传原图，遮罩输出根据模式为全黑(替换模式)或全白(动作迁移模式)。Mask batch [B, H, W, C]. Order must correspond to images one-to-one (node does not validate ordering). Replacement mode: black bg, black areas(RGB<10/255)=background, all-black frames auto-removed. Animation mode: white bg, white areas(RGB>245/255)=background. When disconnected, passes through original images with all-black(all-white in animation mode) mask output."
                }),
                "backgrounds": ("IMAGE", {
                    "tooltip": "背景图批次 [B, H, W, C]。仅动作迁移模式有效，替换模式下此输入被忽略。第1张用于遮罩合成填充背景，其余插入到合成序列(倒数第2和倒数第1之间)。未连接时用纯黑背景(仅做剔除效果)。Background image batch [B, H, W, C]. Only valid in animation mode; ignored in replacement mode. First image used for compositing background, rest inserted before the last frame. When disconnected, uses pure black background (removal only)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "composite"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def _is_black(self, mask_tensor, threshold=10.0 / 255.0):
        """判断mask中哪些像素是黑色（RGB三通道均低于阈值）"""
        return torch.all(mask_tensor < threshold, dim=-1, keepdim=True)

    def _is_white(self, mask_tensor, threshold=10.0 / 255.0):
        """判断mask中哪些像素是白色（RGB三通道均高于1-threshold）"""
        return torch.all(mask_tensor > (1.0 - threshold), dim=-1, keepdim=True)

    def _is_all_black(self, mask_tensor, ratio=0.999):
        """判断mask是否整张接近全黑（用于替换模式的全黑帧剔除）"""
        black = self._is_black(mask_tensor)
        return black.float().mean().item() >= ratio

    def _resize_to(self, img_tensor, target_h, target_w):
        """将单帧 [H, W, C] 缩放到目标尺寸"""
        img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil_resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.array(pil_resized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)

    def _warn(self, msg):
        """统一打印带节点标识的警告信息"""
        print(f"[MaskCompositeBatch] Warning: {msg}")

    def composite(self, images, replacement_mode, process_first_frame, process_last_frame,
                  remove_background=True, masks=None, backgrounds=None):
        if replacement_mode:
            return self._composite_replacement(images, masks, remove_background)
        else:
            return self._composite_animation(
                images, masks, backgrounds,
                process_first_frame, process_last_frame,
                remove_background
            )

    # ==============================
    # 替换模式
    # ==============================
    def _composite_replacement(self, images, masks, remove_background):
        batch_n = images.shape[0]

        # masks 未连接 → 透传原图 + 全黑遮罩
        if masks is None:
            h, w = images[0].shape[0], images[0].shape[1]
            zeros = torch.zeros((batch_n, h, w, 3), dtype=torch.float32)
            return (images.clone(), zeros)

        mask_n = masks.shape[0]
        target_h, target_w = images[0].shape[0], images[0].shape[1]

        # 尺寸不一致时发出警告
        if mask_n > 0:
            m0 = masks[0]
            if m0.shape[0] != target_h or m0.shape[1] != target_w:
                self._warn(
                    f"Mask resolution {m0.shape[0]}x{m0.shape[1]} "
                    f"differs from target {target_h}x{target_w}, resizing masks."
                )

        # 1. 剔除全黑遮罩对应的帧
        valid_indices = []
        for i in range(mask_n):
            m = masks[i]
            if m.shape[0] != target_h or m.shape[1] != target_w:
                m = self._resize_to(m, target_h, target_w)
            if not self._is_all_black(m):
                valid_indices.append(i)

        if len(valid_indices) == 0:
            # 所有帧都被剔除 → 返回空批次
            return (
                torch.empty((0, target_h, target_w, 3), dtype=torch.float32),
                torch.empty((0, target_h, target_w, 3), dtype=torch.float32),
            )

        filtered_images = images[valid_indices]
        filtered_masks = masks[valid_indices]

        if not remove_background:
            # 仅剔除，不做合成 → 透传
            return (filtered_images.clone(), filtered_masks.clone())

        # 2. 合成：将背景替换为纯黑
        bg0 = torch.zeros((target_h, target_w, 3), dtype=torch.float32)
        result_frames = []
        result_masks = []

        n = filtered_images.shape[0]
        for i in range(n):
            img = filtered_images[i]
            m = filtered_masks[i]

            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = self._resize_to(img, target_h, target_w)
            if m.shape[0] != target_h or m.shape[1] != target_w:
                m = self._resize_to(m, target_h, target_w)

            background_mask = self._is_black(m).float()
            composed = img * (1.0 - background_mask) + bg0 * background_mask
            result_frames.append(composed)
            result_masks.append(m)  # 保留原始彩色遮罩

        result_batch = torch.stack(result_frames, dim=0)
        result_mask_batch = torch.stack(result_masks, dim=0)
        return (result_batch, result_mask_batch)

    # ==============================
    # 动作迁移模式
    # ==============================
    def _composite_animation(self, images, masks, backgrounds,
                             process_first_frame, process_last_frame,
                             remove_background):
        batch_n = images.shape[0]
        mask_n = masks.shape[0] if masks is not None else 0

        # masks 未连接 → 透传原图 + 全白遮罩
        if masks is None:
            h, w = images[0].shape[0], images[0].shape[1]
            ones = torch.ones((batch_n, h, w, 3), dtype=torch.float32)
            return (images.clone(), ones)

        # === Step 1: 批次截断（输入后第一顺位操作） ===
        n = min(batch_n, mask_n)
        if batch_n != mask_n:
            self._warn(
                f"images count {batch_n} != masks count {mask_n}, "
                f"truncating to {n}"
            )

        images = images[:n]
        masks = masks[:n]

        # 参考图=0 特殊处理
        if n == 0:
            has_bg = backgrounds is not None and backgrounds.shape[0] > 0
            if has_bg:
                bg_h, bg_w = backgrounds[0].shape[0], backgrounds[0].shape[1]
                ones = torch.ones((backgrounds.shape[0], bg_h, bg_w, 3), dtype=torch.float32)
                return (backgrounds.clone(), ones)
            else:
                return (
                    torch.empty((0,), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.float32),
                )

        # 目标尺寸：以 images 为准
        target_h, target_w = images[0].shape[0], images[0].shape[1]

        # 尺寸警告
        if n > 0:
            m0 = masks[0]
            if m0.shape[0] != target_h or m0.shape[1] != target_w:
                self._warn(
                    f"Mask resolution {m0.shape[0]}x{m0.shape[1]} "
                    f"differs from target {target_h}x{target_w}, resizing masks."
                )

        # 背景图
        has_backgrounds = backgrounds is not None and backgrounds.shape[0] > 0
        if has_backgrounds:
            bg0 = backgrounds[0]
            if bg0.shape[0] != target_h or bg0.shape[1] != target_w:
                bg0 = self._resize_to(bg0, target_h, target_w)
        else:
            bg0 = torch.zeros((target_h, target_w, 3), dtype=torch.float32)

        white_mask_tpl = torch.ones((target_h, target_w, 3), dtype=torch.float32)

        # === Step 2: 首尾帧跳过（基于截断后的结果） ===
        skip_flags = []
        for i in range(n):
            skip = False
            if i == 0 and not process_first_frame:
                skip = True
            if i == n - 1 and not process_last_frame:
                skip = True
            skip_flags.append(skip)

        # === Step 3: 参考图=1 特殊处理 ===
        if n == 1:
            img = images[0]
            m = masks[0]
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = self._resize_to(img, target_h, target_w)
            if m.shape[0] != target_h or m.shape[1] != target_w:
                m = self._resize_to(m, target_h, target_w)

            if has_backgrounds and backgrounds.shape[0] > 0:
                # 首尾 = 原图，中间 = 所有背景图
                all_bgs = []
                for i in range(backgrounds.shape[0]):
                    bg = backgrounds[i]
                    if bg.shape[0] != target_h or bg.shape[1] != target_w:
                        bg = self._resize_to(bg, target_h, target_w)
                    all_bgs.append(bg)

                result_frames = [img] + all_bgs + [img]
                bg_mask_count = len(all_bgs)
                result_masks = [m] + [white_mask_tpl.clone()] * bg_mask_count + [m]
            else:
                result_frames = [img]
                result_masks = [m]

            result_batch = torch.stack(result_frames, dim=0)
            result_mask_batch = torch.stack(result_masks, dim=0)
            return (result_batch, result_mask_batch)

        # === Step 4: 常规循环处理 ===
        result_frames = []
        result_masks = []

        for i in range(n):
            img = images[i]
            m = masks[i]

            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = self._resize_to(img, target_h, target_w)
            if m.shape[0] != target_h or m.shape[1] != target_w:
                m = self._resize_to(m, target_h, target_w)

            if remove_background and not skip_flags[i]:
                background_mask = self._is_white(m).float()
                composed = img * (1.0 - background_mask) + bg0 * background_mask
                result_frames.append(composed)
            else:
                # 跳过帧 或 remove_background=False → 透传原图
                result_frames.append(img)

            # 遮罩：原帧保留原始彩色遮罩（与有无背景图无关）
            result_masks.append(m)

        # === Step 5: 插入所有背景图（第1张既用于合成，也插入序列） ===
        if has_backgrounds:
            extra_bgs = []
            for i in range(backgrounds.shape[0]):
                bg = backgrounds[i]
                if bg.shape[0] != target_h or bg.shape[1] != target_w:
                    bg = self._resize_to(bg, target_h, target_w)
                extra_bgs.append(bg)

            if len(result_frames) <= 1:
                result_frames = result_frames + extra_bgs
                result_masks = result_masks + [white_mask_tpl.clone()] * len(extra_bgs)
            else:
                # 插入到倒数第2和倒数第1之间
                result_frames = result_frames[:-1] + extra_bgs + result_frames[-1:]
                result_masks = (
                    result_masks[:-1]
                    + [white_mask_tpl.clone()] * len(extra_bgs)
                    + result_masks[-1:]
                )

        result_batch = torch.stack(result_frames, dim=0)
        result_mask_batch = torch.stack(result_masks, dim=0)
        return (result_batch, result_mask_batch)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "MaskCompositeRef": MaskCompositeRef,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCompositeRef": "遮罩批次合成 (Ref)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
