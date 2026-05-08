import folder_paths
from server import PromptServer
from aiohttp import web
import os


class PathCollectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "paths": ("STRING", {"default": "", "multiline": True}),
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
                    "hidden": True          # 隐藏输入控件，仅显示连接点
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("selected_path", "file_name", "last_folder")
    FUNCTION = "select_by_index"
    CATEGORY = "Custom Nodes"
    OUTPUT_NODE = False

    def select_by_index(self, combined_string, index):
        parts = combined_string.split("\n")
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
                # 判断是文件还是文件夹：末段含扩展名则为文件
                dot_index = basename.rfind('.')
                if dot_index > 0 and dot_index < len(basename) - 1:
                    # 文件：取其父目录的 basename
                    parent_dir = os.path.dirname(selected)
                    last_folder = os.path.basename(parent_dir)
                else:
                    # 文件夹或无扩展名：取自身 basename
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
                "start": ("INT", {"default": 0, "step": 1}),
                "step": ("INT", {"default": 1, "min": 1, "step": 1}),
                "value": ("INT", {"default": 0, "step": 1}),
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

        dist_lower = abs(lower - start)
        dist_upper = abs(upper - start)

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
                "path": ("STRING", {"default": "", "multiline": False}),
                "create_if_missing": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "extension_filter": ("STRING", {"default": "", "multiline": False}),
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

    def validate_path(self, path, create_if_missing, prefix, extension_filter):
        # 1. 处理相对路径前缀
        if not os.path.isabs(path) and prefix:
            actual_path = os.path.normpath(os.path.join(prefix, path))
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

        # 5. 输出路径：存在则返回原始传入路径
        path_out = path if exists else ""

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


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "PathCollectorNode": PathCollectorNode,
    "IndexSelectorNode": IndexSelectorNode,
    "PathValidatorNode": PathValidatorNode,
    "IntegerSnapNode": IntegerSnapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PathCollectorNode": "路径收集器",
    "IndexSelectorNode": "索引选择器",
    "PathValidatorNode": "路径验证器",
    "IntegerSnapNode": "Integer Snap",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
