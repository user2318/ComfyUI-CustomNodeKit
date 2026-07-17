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
                "paths": ("STRING", {"default": "", "multiline": True, "tooltip": "输入多个路径，每行一个。可通过多文件选择器前端控件批量选择。Enter multiple paths, one per line. Supports batch selection via the multi-file picker frontend control."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "选择要输出的行号（从1开始）。0=输出所有行；1=输出第1行；2=输出第2行；以此类推。超出范围时输出空字符串。Select which line number to output (starting from 1). 0=output all lines; 1=output line 1; 2=output line 2; etc. Outputs empty string if out of range."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paths_out",)
    FUNCTION = "combine_paths"
    CATEGORY = "Custom Nodes"
    OUTPUT_NODE = False

    def combine_paths(self, paths="", index=0):
        paths = paths.strip()
        if index == 0:
            # 输出所有行（原行为）
            return (paths,)
        
        # 按行分割（支持 \r\n 和 \n）
        lines = paths.replace("\r\n", "\n").split("\n")
        
        # 从1开始索引，但内部用0-based
        if 1 <= index <= len(lines):
            return (lines[index - 1],)
        else:
            # 超出范围 → 输出空字符串
            return ("",)


class IndexSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_string": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "hidden": True,         # 隐藏输入控件，仅显示连接点
                    "tooltip": "来自路径收集器或其他来源的多行路径字符串，每行一个路径。Multi-line path string from the path collector, one path per line."
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "选择第几条路径（从0开始）。超出范围时输出为空。Select which path by index (starting from 0). Outputs empty if out of range."}),
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

    except ImportError:
        return web.json_response({"error": "tkinter 不可用。请运行 install.py 安装 tkinter 组件，或重启 ComfyUI 后重试。"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


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
                "path": ("STRING", {"default": "", "multiline": False, "tooltip": "待验证的文件或文件夹路径，支持相对路径和绝对路径。File or folder path to validate, supports relative and absolute paths."}),
                "create_if_missing": ("BOOLEAN", {"default": False, "tooltip": "当路径为文件夹且不存在时，是否自动创建该文件夹。Whether to automatically create the folder if it does not exist."}),
                "path_prefix": ("STRING", {"default": "", "multiline": False, "tooltip": "相对路径的前缀目录。若path为相对路径且此前缀不为空，将拼接为完整路径。Prefix directory for relative paths. If path is relative and prefix is not empty, they will be joined into a full path."}),
                "extension_filter": ("STRING", {"default": "", "multiline": False, "tooltip": "统计文件数量时的扩展名过滤。多个扩展名用英文逗号分隔，如 'png,jpg'。留空则统计所有文件。Extension filter for file counting. Separate multiple extensions with commas, e.g. 'png,jpg'. Leave empty to count all files."}),
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
                    "tooltip": "包含图片文件的文件夹路径。支持绝对路径，或相对于ComfyUI根目录的相对路径。仅读取本目录下的图片，不会检索子目录。Folder path containing image files. Supports absolute paths or relative paths to the ComfyUI root directory. Only reads images from this directory, not subdirectories."
                }),
                "size_mode": (["resize_to_first", "filter_same_size"], {
                    "default": "resize_to_first",
                    "tooltip": "图片尺寸处理方式：resize_to_first — 所有图片统一缩放至第一张图的宽高；filter_same_size — 仅加载与第一张图尺寸完全相同的图片，尺寸不同的会被跳过并在控制台提示。Size mode: resize_to_first — resize all images to the first image's dimensions; filter_same_size — only load images exactly matching the first image's dimensions."
                }),
                "skip_first_n": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "跳过文件夹排序靠前的N张图片。默认0表示不跳过。Skip the first N images in the folder. Default 0 means no skip."
                }),
                "load_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "最多载入的图片数量。设置为0表示不限制，载入全部符合条件的图片。如果该值大于文件夹内实际图片数量，则全量输出。Maximum number of images to load. Set to 0 for unlimited (loads all matching images). If the value exceeds the actual count, all images are output."
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
                    "tooltip": "第一个图像批次。images_b将拼接到此批次后方。如果仅此路有输入，将直接透传输出。First image batch. images_b will be concatenated after this batch. If only this input is provided, it will be passed through directly."
                }),
                "images_b": ("IMAGE", {
                    "tooltip": "第二个图像批次。将被拼接到images_a后方构成完整批次。如果仅此路有输入，将直接透传输出。Second image batch. It will be concatenated after images_a to form the complete batch. If only this input is provided, it will be passed through directly."
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
                    "tooltip": "目标宽度（像素）。Target width (pixels)."
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "step": 1,
                    "tooltip": "目标高度（像素）。Target height (pixels)."
                }),
                "crop_mode": (["disabled", "center", "top", "bottom", "left", "right"], {
                    "default": "disabled",
                    "tooltip": "裁剪模式。disabled：直接拉伸不保持比例；center/top/bottom/left/right：先等比缩放覆盖目标再按方向裁剪。Crop mode: disabled — stretch directly without aspect ratio; center/top/bottom/left/right — scale to cover then crop from the specified direction."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "输入图像批次。无输入或输入无效时输出None。Input image batch. Outputs None when no input or invalid input."
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


class SingleFrameVAEEncode:
    """逐帧独立 VAE 编码节点
    
    将输入图像批次中的每张图单独送入 VAE 编码（每次 1 帧），
    避免 Wan 3D VAE 的 temporal attention 产生跨帧影响，
    然后将所有独立编码的 latent 在时间维度拼接输出。
    
    行为模拟 EverAnimate 节点的单帧编码 + 复制方式：
    - 输入 N 张图 → 输出 N 帧 latent
    - 每帧 latent 不受其他帧的时序上下文影响
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "输入图像批次，每帧将单独编码。Input image batch, each frame will be encoded independently."}),
                "vae": ("VAE", {"tooltip": "VAE 模型，用于将图像编码为潜在空间。VAE model used to encode images into latent space."}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "WanLoop/工具节点"
    
    def encode(self, images, vae):
        batch_size = images.shape[0]
        if batch_size == 0:
            raise ValueError("输入图像为空")
        
        encoded_list = []
        
        for i in range(batch_size):
            # 每次只取 1 帧，保持 [1, H, W, 3] 形状
            single_image = images[i:i+1]
            
            # 单帧 VAE 编码（无跨帧时序上下文）
            single_latent = vae.encode(single_image[:, :, :, :3])
            
            # single_latent 形状为 [1, 16, 1, H/8, W/8]
            encoded_list.append(single_latent)
        
        # 在时间维度（dim=2）拼接所有 latent
        # 最终形状 [1, 16, N, H/8, W/8]
        concat_latent = torch.cat(encoded_list, dim=2)
        
        out = {"samples": concat_latent}
        
        return (out,)


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


class BatchFrameReplicate:
    """图像批次指定帧复制节点 — 在图像批次中复制指定图像并插入到其后

    功能：
    - 两路输入（image_a, image_b），均为可选
    - 通过 image_index 选择要复制的图像（支持负索引：-1=最后一张）
    - 通过 copy_count 控制复制份数
    - 每路独立处理、独立输出

    例如：输入 [A, B, C, D, E]，image_index=0, copy_count=2
         → 取 A，复制 2 份插入其后
         → 输出 [A, A', A', B, C, D, E]

    使用场景：延长首/尾帧时长、在关键帧上做停留效果等。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "总开关。关闭时各路直接透传，不做任何复制。Master switch. When disabled, all channels pass through unchanged."
                }),
                "image_index": ("INT", {
                    "default": 0,
                    "min": -0xFFFF_FFFF,
                    "max": 0xFFFF_FFFF,
                    "step": 1,
                    "tooltip": "要复制的图像索引。0=第一张，-1=最后一张，-2=倒数第二张，以此类推。越界时自动限制到有效范围。Index of the image to replicate. 0=first, -1=last, -2=second-to-last. Clamped to valid range automatically."
                }),
                "copy_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFF_FFFF,
                    "step": 1,
                    "tooltip": "复制多少份。0=不复制，直接透传。Number of copies to make. 0=pass through (no replication)."
                }),
            },
            "optional": {
                "image_a": ("IMAGE", {
                    "tooltip": "第一路图像批次输入 [B, H, W, C]，未连接时对应输出为 None。Channel A image batch input [B, H, W, C]. When disconnected, output is None."
                }),
                "image_b": ("IMAGE", {
                    "tooltip": "第二路图像批次输入 [B, H, W, C]，未连接时对应输出为 None。Channel B image batch input [B, H, W, C]. When disconnected, output is None."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_a", "image_b")
    FUNCTION = "replicate"
    CATEGORY = "image"
    OUTPUT_NODE = False

    def _resolve_index(self, idx, batch_size):
        """解析索引，支持负索引，越界时 clamp 到有效范围"""
        if batch_size <= 0:
            return None
        if idx < 0:
            resolved = batch_size + idx
        else:
            resolved = idx
        # clamp 到 [0, batch_size - 1]
        if resolved < 0:
            resolved = 0
        if resolved >= batch_size:
            resolved = batch_size - 1
        return resolved

    def _process_single(self, images, image_index, copy_count):
        """处理单路图像批次

        Args:
            images: 图像批次 [B, H, W, C] 或 None
            image_index: 要复制的图像索引
            copy_count: 复制份数

        Returns:
            处理后的图像批次，或 None（当输入为 None 时）
        """
        if images is None:
            return None

        B = images.shape[0]
        if B == 0:
            return images.clone()

        # 复制数为 0 → 透传
        if copy_count == 0:
            return images.clone()

        # 解析索引
        src_idx = self._resolve_index(image_index, B)
        if src_idx is None:
            return images.clone()

        # 取出要复制的图像并复制 N 份
        src_image = images[src_idx:src_idx + 1]  # 保持 4D [1, H, W, C]
        copies = src_image.repeat(copy_count, 1, 1, 1)  # [copy_count, H, W, C]

        # 在原索引位置后插入副本
        # 分成三部分: [0:src_idx+1], [copies], [src_idx+1:]
        result = torch.cat([images[:src_idx + 1], copies, images[src_idx + 1:]], dim=0)
        return result

    def replicate(self, enabled, image_index, copy_count, image_a=None, image_b=None):
        """主函数

        Args:
            enabled: 是否生效
            image_index: 复制目标索引
            copy_count: 复制份数
            image_a: 第一路图像批次（可选）
            image_b: 第二路图像批次（可选）

        Returns:
            tuple: (image_a, image_b)
        """
        # 未启用或复制数为 0 → 各路直接透传
        if not enabled or copy_count == 0:
            return (
                image_a.clone() if image_a is not None else None,
                image_b.clone() if image_b is not None else None,
            )

        # 正常处理
        result_a = self._process_single(image_a, image_index, copy_count)
        result_b = self._process_single(image_b, image_index, copy_count)

        return (result_a, result_b)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "PathCollectorNode": PathCollectorNode,
    "IndexSelectorNode": IndexSelectorNode,
    "PathValidatorNode": PathValidatorNode,
    "FolderImageLoaderNode": FolderImageLoaderNode,
    "ImageBatchConcatNode": ImageBatchConcatNode,
    "ImageBatchResizeNode": ImageBatchResizeNode,
    "SingleFrameVAEEncode": SingleFrameVAEEncode,
    "BatchImageReplace": BatchImageReplace,
    "BatchFrameReplicate": BatchFrameReplicate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathCollectorNode": "路径收集器",
    "IndexSelectorNode": "索引选择器",
    "PathValidatorNode": "路径验证器",
    "FolderImageLoaderNode": "文件夹图像载入器",
    "ImageBatchConcatNode": "图像批次合并",
    "ImageBatchResizeNode": "图像批次缩放",
    "SingleFrameVAEEncode": "逐帧独立 VAE 编码",
    "BatchImageReplace": "图像批次替换 (Batch Replace)",
    "BatchFrameReplicate": "图像批次指定帧复制 (Frame Replicate)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
