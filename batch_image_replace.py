import torch
import numpy as np
from PIL import Image


class BatchImageReplace:
    """图像批次替换节点 — 在指定位置替换图像批次中的部分图像

    主输入是一个图像批次，可选输入是替换用的图像批次。
    支持从指定索引位置开始替换指定数量的图像。
    当替换图像数量不匹配时，提供多种策略处理。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFF_FFFF,
                    "step": 1,
                    "tooltip": "从主批次的哪个位置开始替换（0-based）。Starting position for replacement (0-based)."
                }),
                "replace_count": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xFFFF_FFFF,
                    "step": 1,
                    "tooltip": "要替换多少张图像。 -1 = 全部替换（从start_index到末尾），0 = 不替换。实际可替换数量受主批次剩余长度限制。Number of images to replace. -1=replace all (from start_index to end), 0=no replacement. Actual count limited by remaining batch length."
                }),
                "overflow_mode": (["take_front", "take_back"], {
                    "default": "take_front",
                    "tooltip": "替换图像批次数量多于需替换张数时的处理方式。take_front=从前面截取所需张数；take_back=从后面截取所需张数。When replace_images has MORE frames than needed: take_front=use first N; take_back=use last N."
                }),
                "underflow_mode": (["as_is", "repeat_begin", "repeat_end"], {
                    "default": "as_is",
                    "tooltip": "替换图像批次数量少于需替换张数时的处理方式。as_is=按实际张数替换（不扩展）；repeat_begin=用第1张重复填充前面（如[A,B]→[A,A,A,B]）；repeat_end=用最后1张重复填充后面（如[A,B]→[A,B,B,B]）。When replace_images has FEWER frames than needed: as_is=use actual count; repeat_begin=fill front with first image; repeat_end=fill end with last image."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "主图像批次 [B, H, W, C]。将要在此批次中替换指定位置的图像。Main image batch [B, H, W, C]. Images will be replaced at specified positions."
                }),
                "replace_images": ("IMAGE", {
                    "tooltip": "替换用图像批次 [B, H, W, C]。替换来源图像，未连接时直接透传主批次。Replacement image batch [B, H, W, C]. When disconnected, passes through original images unchanged."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def _resize_to(self, img_tensor, target_h, target_w):
        """将单帧 [H, W, C] 缩放到目标尺寸"""
        img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil_resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.array(pil_resized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)

    def _warn(self, msg):
        """统一打印带节点标识的警告信息"""
        print(f"[BatchImageReplace] Warning: {msg}")

    def replace(self, images, start_index, replace_count, overflow_mode,
                underflow_mode="as_is", replace_images=None):
        if images is None or images.shape[0] == 0:
            return (None,)

        B = images.shape[0]

        # 未连接替换图 → 直接返回原批次
        if replace_images is None:
            return (images.clone(),)

        # 替换张数为0 → 直接返回原批次
        if replace_count == 0:
            return (images.clone(),)

        R = replace_images.shape[0]
        target_h, target_w = images[0].shape[0], images[0].shape[1]

        # 计算实际可替换数量（受主批次剩余长度限制）
        max_available = B - start_index
        if max_available <= 0:
            self._warn(
                f"start_index={start_index} >= batch_size={B}, "
                f"no valid replacement range, returning original batch."
            )
            return (images.clone(),)

        # 替换张数为-1 → 替换从start_index到末尾全部
        if replace_count == -1:
            replace_count = max_available

        actual_count = min(replace_count, max_available)

        # 准备替换序列
        if R >= actual_count:
            # 替换图多于或等于实际需替换张数 → 截取
            if overflow_mode == "take_front":
                selected = replace_images[:actual_count]
            else:  # take_back
                selected = replace_images[-actual_count:]
        else:
            # 替换图少于实际需替换张数 → 按策略处理
            if underflow_mode == "as_is":
                # 按实际张数替换，不扩展
                actual_count = R
                selected = replace_images
            elif underflow_mode == "repeat_begin":
                # 扩展开头：用第1张重复填充前面，最后一张在末尾
                frames = []
                # 先填充 (actual_count - R) 张用第1张
                for _ in range(actual_count - R):
                    frames.append(replace_images[0].clone())
                # 再按原顺序添加 R 张
                for i in range(R):
                    frames.append(replace_images[i])
                selected = torch.stack(frames, dim=0)
            else:  # repeat_end
                # 扩展结尾：第1张保持，后面用最后1张重复填充
                frames = []
                # 先放第1张
                frames.append(replace_images[0].clone())
                # 再用最后1张填充剩余 (actual_count - 1) 张
                last_img = replace_images[-1].clone()
                for _ in range(actual_count - 1):
                    frames.append(last_img.clone())
                selected = torch.stack(frames, dim=0)

        # 尺寸缩放（如有需要）
        resized_selected = []
        for i in range(selected.shape[0]):
            img = selected[i]
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = self._resize_to(img, target_h, target_w)
            resized_selected.append(img)
        if len(resized_selected) > 0:
            selected = torch.stack(resized_selected, dim=0)

        # 执行替换
        result = images.clone()
        result[start_index: start_index + actual_count] = selected

        return (result,)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "BatchImageReplace": BatchImageReplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageReplace": "图像批次替换 (Batch Replace)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]