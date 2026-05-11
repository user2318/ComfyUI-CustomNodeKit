import folder_paths
from server import PromptServer
from aiohttp import web
import os
import torch
import numpy as np
from PIL import Image
import logging


class PathCollectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "paths": ("STRING", {"default": "", "multiline": True, "tooltip": "输入多个路径，每行一个。可通过多文件选择器前端控件批量选择"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_paths",)
    FUNCTION = "combine_paths"
    CATEGORY = "Custom Nodes"
    OUTPUT_NODE = False

    def combine_paths(self, paths=""):
        # 直接返回多行文本，去除首尾空白行
        return (paths.strip(),)


class IndexSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_string": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "hidden": True,         # 隐藏输入控件，仅显示连接点
                    "tooltip": "来自路径收集器或其他来源的多行路径字符串，每行一个路径"
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "选择第几条路径（从0开始）。超出范围时输出为空"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("selected_path", "file_name", "last_folder")
    FUNCTION = "select_by_index"
    CATEGORY = "Custom Nodes"
    OUTPUT_NODE = False

    def select_by_index(self, combined_string, index):
        parts = combined_string.replace("\r\n", "\n").split("\n")
        if index < 0 or index >= len(parts):
            selected = ""
            file_name = ""
            last_folder = ""
        else:
            selected = parts[index]
            basename = os.path.basename(selected)
            file_name = os.path.splitext(basename)[0] if selected else ""

            # 计算最后一级文件夹名
            last_folder = ""
            if selected:
                # 判断是文件还是文件夹：优先按实际存在判断，否则启发式
                is_file = False
                if os.path.exists(selected):
                    is_file = os.path.isfile(selected)
                elif selected.endswith(os.sep) or selected.endswith("/"):
                    is_file = False
                else:
                    dot_index = basename.rfind('.')
                    if dot_index >= 0 and dot_index < len(basename) - 1:
                        is_file = True

                if is_file:
                    # 文件：取其父目录的 basename
                    parent_dir = os.path.dirname(selected)
                    last_folder = os.path.basename(parent_dir)
                else:
                    # 文件夹：取自身 basename
                    last_folder = basename
        return (selected, file_name, last_folder)


# 后端路由：文件/文件夹选择接口（支持多选）
@PromptServer.instance.routes.post("/multi_file_picker/select")
async def select_file(request):
    try:
        data = await request.json()
        mode = data.get("mode", "file")
        multi = data.get("multi", False)

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()

        if mode == "file":
            if multi:
                file_paths = filedialog.askopenfilenames()
                root.destroy()
                paths = list(file_paths) if file_paths else []
                return web.json_response({"paths": paths})
            else:
                file_path = filedialog.askopenfilename()
                root.destroy()
                return web.json_response({"paths": [file_path] if file_path else []})
        else:  # directory
            # tkinter 的 askdirectory 不支持多选，所以始终单选
            dir_path = filedialog.askdirectory()
            root.destroy()
            return web.json_response({"paths": [dir_path] if dir_path else []})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


class IntegerSnapNode:
    """整数对齐节点：将输入值修正到 start + k*step 的最接近值，等距时选离 start 更近的。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "step": 1, "tooltip": "对齐基准值，对齐后的值将为 start + k*step"}),
                "step": ("INT", {"default": 1, "min": 1, "step": 1, "tooltip": "对齐步长，必须大于等于1"}),
                "value": ("INT", {"default": 0, "step": 1, "tooltip": "需要对齐的原始值，将被修正到最近的 start + k*step"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("snapped",)
    FUNCTION = "snap"
    CATEGORY = "utils"

    def snap(self, start, step, value):
        k = (value - start) // step
        lower = start + k * step
        upper = lower + step

        dist_lower = abs(lower - value)
        dist_upper = abs(upper - value)

        if dist_lower < dist_upper:
            snapped = lower
        elif dist_upper < dist_lower:
            snapped = upper
        else:
            # 等距，选绝对值较小的
            snapped = lower if abs(lower) <= abs(upper) else upper
        return (snapped,)


class PathValidatorNode:
    """
    自动判断文件/文件夹的路径验证与操作节点
    - 自动识别路径是文件还是文件夹
    - 检查路径是否存在
    - 可选为相对路径添加前缀
    - 可选自动创建缺失的文件夹（仅对文件夹类型生效）
    - 输出：是否存在、存在的原路径、文件夹内指定扩展名文件数量
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False, "tooltip": "待验证的文件或文件夹路径，支持相对路径和绝对路径"}),
                "create_if_missing": ("BOOLEAN", {"default": False, "tooltip": "当路径为文件夹且不存在时，是否自动创建该文件夹"}),
                "path_prefix": ("STRING", {"default": "", "multiline": False, "tooltip": "相对路径的前缀目录。若path为相对路径且此前缀不为空，将拼接为完整路径"}),
                "extension_filter": ("STRING", {"default": "", "multiline": False, "tooltip": "统计文件数量时的扩展名过滤。多个扩展名用英文逗号分隔，如 'png,jpg'。留空则统计所有文件"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("exists", "path_out", "file_count")
    FUNCTION = "validate_path"
    CATEGORY = "Custom Nodes"
    OUTPUT_NODE = False

    def _detect_type(self, path, actual_path):
        """自动判断路径类型：已存在则根据实际，否则启发式"""
        if os.path.exists(actual_path):
            return "directory" if os.path.isdir(actual_path) else "file"
        # 启发式：以分隔符结尾 → 目录
        if path.endswith(os.sep):
            return "directory"
        # 有扩展名 → 文件，否则目录
        basename = os.path.basename(path)
        _, ext = os.path.splitext(basename)
        if ext:
            return "file"
        return "directory"

    def validate_path(self, path, create_if_missing, path_prefix, extension_filter):
        # 1. 处理相对路径前缀
        if not os.path.isabs(path) and path_prefix:
            actual_path = os.path.normpath(os.path.join(path_prefix, path))
        else:
            actual_path = os.path.normpath(path)

        # 2. 自动判断类型
        mode = self._detect_type(path, actual_path)

        # 3. 检查是否存在
        if mode == "directory":
            exists = os.path.isdir(actual_path)
        else:
            exists = os.path.isfile(actual_path)

        # 4. 自动创建文件夹（仅对目录模式生效）
        if mode == "directory" and not exists and create_if_missing:
            try:
                os.makedirs(actual_path, exist_ok=True)
                exists = True
            except Exception:
                exists = False

        # 5. 输出路径：存在则返回完整路径（包含前缀）
        path_out = actual_path if exists else ""

        # 6. 计算文件夹内指定扩展名文件数
        if mode == "directory" and os.path.isdir(actual_path):
            if extension_filter.strip():
                exts = [
                    f".{ext.strip().lstrip('.').lower()}"
                    for ext in extension_filter.split(",") if ext.strip()
                ]
            else:
                exts = None
            count = 0
            try:
                for entry in os.listdir(actual_path):
                    full_entry = os.path.join(actual_path, entry)
                    if os.path.isfile(full_entry):
                        if exts is None:
                            count += 1
                        else:
                            _, ext = os.path.splitext(entry)
                            if ext.lower() in exts:
                                count += 1
            except Exception:
                count = -1
        else:
            count = -1   # 非文件夹或无法访问

        return (exists, path_out, count)


class FolderImageLoaderNode:
    """文件夹图片载入器
    从指定文件夹中按文件名升序读取图片，支持跳过、数量限制和尺寸处理。
    路径不存在、文件夹为空等情况不抛出异常，仅输出控制台警告和空结果。
    """
    _IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "包含图片文件的文件夹路径。支持绝对路径，或相对于ComfyUI根目录的相对路径。仅读取本目录下的图片，不会检索子目录。"
                }),
                "size_mode": (["resize_to_first", "filter_same_size"], {
                    "default": "resize_to_first",
                    "tooltip": "图片尺寸处理方式：resize_to_first — 所有图片统一缩放至第一张图的宽高；filter_same_size — 仅加载与第一张图尺寸完全相同的图片，尺寸不同的会被跳过并在控制台提示。"
                }),
                "skip_first_n": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "跳过文件夹排序靠前的N张图片。默认0表示不跳过。"
                }),
                "load_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "最多载入的图片数量。设置为0表示不限制，载入全部符合条件的图片。如果该值大于文件夹内实际图片数量，则全量输出。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load_images"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def _resolve_path(self, folder_path: str) -> str:
        """将输入路径解析为绝对路径
        解析优先级：
        1. 原样路径直接可用 → 直接返回
        2. 相对于 ComfyUI 根目录拼接 → 可用则返回
        3. 都不存在 → 返回原样路径（后续 os.path.isdir 检查会触发 None 输出）
        """
        path = os.path.normpath(folder_path)

        # 1. 原样路径直接可用
        if os.path.isdir(path):
            return path

        # 2. 相对于 ComfyUI 根目录
        root_path = os.path.normpath(folder_paths.base_path)
        candidate = os.path.normpath(os.path.join(root_path, folder_path))
        if os.path.isdir(candidate):
            return candidate

        # 都不存在，返回原样路径
        return path

    def _get_image_files(self, folder_path: str) -> list:
        """获取文件夹内所有图片文件（按文件名升序），不包含子目录"""
        try:
            entries = os.listdir(folder_path)
        except Exception:
            return []

        files = []
        for entry in entries:
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                _, ext = os.path.splitext(entry)
                if ext.lower() in self._IMAGE_EXTENSIONS:
                    files.append(entry)
        files.sort()
        return files

    def _load_image_tensor(self, file_path: str) -> torch.Tensor:
        """加载单张图片并转为 [H, W, C] 的 0-1 范围张量"""
        img = Image.open(file_path)
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)

    def _resize_tensor(self, img_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """将图片张量缩放至目标尺寸 (H, W)"""
        img_np = (img_tensor.numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)

    def load_images(self, folder_path: str, size_mode: str, skip_first_n: int, load_count: int):
        # ==================== 1. 路径解析与验证 ====================
        if not folder_path or not folder_path.strip():
            print("[FolderImageLoader] 警告: 未提供文件夹路径")
            return (None, 0)

        actual_path = self._resolve_path(folder_path.strip())

        if not os.path.isdir(actual_path):
            print(f"[FolderImageLoader] 警告: 文件夹不存在或不是有效目录: {actual_path}")
            return (None, 0)

        # ==================== 2. 获取图片文件列表 ====================
        image_files = self._get_image_files(actual_path)

        if len(image_files) == 0:
            print(f"[FolderImageLoader] 警告: 文件夹内无图片文件: {actual_path}")
            return (None, 0)

        # ==================== 3. 跳过前 N 张 ====================
        if skip_first_n >= len(image_files):
            print(f"[FolderImageLoader] 警告: skip_first_n={skip_first_n} 超出图片总数 {len(image_files)}")
            return (None, 0)

        if skip_first_n > 0:
            image_files = image_files[skip_first_n:]

        # ==================== 4. 应用数量限制 ====================
        if load_count > 0 and load_count < len(image_files):
            image_files = image_files[:load_count]

        # ==================== 5. 加载第一张图，确定目标尺寸 ====================
        try:
            first_file_path = os.path.join(actual_path, image_files[0])
            first_tensor = self._load_image_tensor(first_file_path)
        except Exception as e:
            print(f"[FolderImageLoader] 警告: 无法读取第一张图片 {image_files[0]}: {e}")
            return (None, 0)

        target_h, target_w = first_tensor.shape[0], first_tensor.shape[1]

        # ==================== 6. 按模式加载剩余图片 ====================
        loaded_tensors = [first_tensor]
        for fname in image_files[1:]:
            file_path = os.path.join(actual_path, fname)
            try:
                img_tensor = self._load_image_tensor(file_path)
            except Exception as e:
                print(f"[FolderImageLoader] 警告: 无法读取图片 {fname}: {e}，已跳过")
                continue

            h, w = img_tensor.shape[0], img_tensor.shape[1]

            if size_mode == "resize_to_first":
                if h != target_h or w != target_w:
                    img_tensor = self._resize_tensor(img_tensor, target_h, target_w)
                loaded_tensors.append(img_tensor)
            else:  # filter_same_size
                if h == target_h and w == target_w:
                    loaded_tensors.append(img_tensor)
                else:
                    print(f"[FolderImageLoader] 提示: 跳过尺寸不匹配的图片 {fname} (尺寸: {w}x{h})")

        # ==================== 7. 构建批次 ====================
        if len(loaded_tensors) == 0:
            print("[FolderImageLoader] 警告: 没有有效图片可加载")
            return (None, 0)

        batch = torch.stack(loaded_tensors, dim=0)
        return (batch, batch.shape[0])


class ImageBatchConcatNode:
    """图像批次拼接节点
    将两个图像批次沿第一个维度拼接。支持任一输入为空时的透传。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images_a": ("IMAGE", {
                    "tooltip": "第一个图像批次。images_b将拼接到此批次后方。如果仅此路有输入，将直接透传输出。"
                }),
                "images_b": ("IMAGE", {
                    "tooltip": "第二个图像批次。将被拼接到images_a后方构成完整批次。如果仅此路有输入，将直接透传输出。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "concat"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def concat(self, images_a=None, images_b=None):
        # 判断两路输入的有效性
        a_valid = images_a is not None and isinstance(images_a, torch.Tensor) and images_a.shape[0] > 0
        b_valid = images_b is not None and isinstance(images_b, torch.Tensor) and images_b.shape[0] > 0

        if a_valid and b_valid:
            result = torch.cat([images_a, images_b], dim=0)  # type: ignore
            return (result,)
        elif a_valid:
            result = images_a.clone()  # type: ignore
            return (result,)
        elif b_valid:
            result = images_b.clone()  # type: ignore
            return (result,)
        else:
            # 两路都无有效输入
            return (None,)


class ImageBatchResizeNode:
    """图像批次缩放节点
    将图像批次缩放到指定宽高。支持按方向裁剪以保持宽高比。
    输入无效时输出 None。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "step": 1,
                    "tooltip": "目标宽度（像素）"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "step": 1,
                    "tooltip": "目标高度（像素）"
                }),
                "crop_mode": (["disabled", "center", "top", "bottom", "left", "right"], {
                    "default": "disabled",
                    "tooltip": "裁剪模式。disabled：直接拉伸不保持比例；center/top/bottom/left/right：先等比缩放覆盖目标再按方向裁剪。"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "输入图像批次。无输入或输入无效时输出None。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "resize_batch"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def _tensor_to_pil(self, img_tensor: torch.Tensor) -> Image.Image:
        """[H, W, C] 0-1 张量 → PIL Image"""
        img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """PIL Image → [H, W, C] 0-1 张量"""
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)

    def _crop(self, pil_img: Image.Image, target_w: int, target_h: int, mode: str) -> Image.Image:
        """按方向裁剪到目标尺寸"""
        src_w, src_h = pil_img.size
        if mode == "center":
            left = (src_w - target_w) // 2
            top = (src_h - target_h) // 2
        elif mode == "top":
            left = (src_w - target_w) // 2
            top = 0
        elif mode == "bottom":
            left = (src_w - target_w) // 2
            top = src_h - target_h
        elif mode == "left":
            left = 0
            top = (src_h - target_h) // 2
        elif mode == "right":
            left = src_w - target_w
            top = (src_h - target_h) // 2
        else:
            return pil_img
        return pil_img.crop((left, top, left + target_w, top + target_h))

    def resize_batch(self, width: int, height: int, crop_mode: str, images=None):
        # 输入有效性检查
        if images is None or not isinstance(images, torch.Tensor) or images.shape[0] == 0:
            return (None,)

        batch_size = images.shape[0]
        result_tensors = []

        for i in range(batch_size):
            img_tensor = images[i]  # [H, W, C]
            pil_img = self._tensor_to_pil(img_tensor)
            src_w, src_h = pil_img.size

            if crop_mode == "disabled":
                # 直接拉伸缩放
                pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
            else:
                # 计算覆盖缩放比
                scale = max(width / src_w, height / src_h)
                scaled_w = max(1, round(src_w * scale))
                scaled_h = max(1, round(src_h * scale))
                pil_img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                # 裁剪到目标尺寸
                pil_img = self._crop(pil_img, width, height, crop_mode)

            result_tensors.append(self._pil_to_tensor(pil_img))

        batch = torch.stack(result_tensors, dim=0)
        return (batch,)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "PathCollectorNode": PathCollectorNode,
    "IndexSelectorNode": IndexSelectorNode,
    "PathValidatorNode": PathValidatorNode,
    "IntegerSnapNode": IntegerSnapNode,
    "FolderImageLoaderNode": FolderImageLoaderNode,
    "ImageBatchConcatNode": ImageBatchConcatNode,
    "ImageBatchResizeNode": ImageBatchResizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathCollectorNode": "路径收集器",
    "IndexSelectorNode": "索引选择器",
    "PathValidatorNode": "路径验证器",
    "IntegerSnapNode": "Integer Snap",
    "FolderImageLoaderNode": "文件夹图像载入器",
    "ImageBatchConcatNode": "图像批次合并",
    "ImageBatchResizeNode": "图像批次缩放",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
