import torch
import numpy as np
import os
import json
import logging
import time
from collections import defaultdict, OrderedDict
from PIL import Image, ImageColor
from server import PromptServer
from aiohttp import web

# 每批处理的图片数量，可根据显存大小调整
BATCH_SIZE = 32
CACHE_MAX_SIZE = 50  # output_cache 最多保留 50 个节点的数据
output_cache = OrderedDict()  # OrderedDict 实现 LRU 淘汰

# ============= 安全工具 =============

# 请求频率限制：防止路由被恶意高频调用
_request_timestamps = defaultdict(list)  # IP -> [timestamp, ...]
RATE_LIMIT_REQUESTS = 120   # 最大请求次数
RATE_LIMIT_WINDOW = 60      # 时间窗口（秒）


def _check_rate_limit(request) -> bool:
    """检查请求是否超过频率限制，放行返回 True，拦截返回 False"""
    ip = request.remote or "127.0.0.1"
    now = time.time()
    timestamps = _request_timestamps[ip]
    # 清理过期记录
    cutoff = now - RATE_LIMIT_WINDOW
    _request_timestamps[ip] = [t for t in timestamps if t > cutoff]
    if len(_request_timestamps[ip]) >= RATE_LIMIT_REQUESTS:
        logging.warning(f"[Security] 请求频率限制已触发: {ip}")
        return False
    _request_timestamps[ip].append(now)
    return True

# 允许的图片格式白名单（用于格式验证）
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "TIFF", "TIF"}


def _validate_image_format(path: str) -> bool:
    """验证文件是否为允许的图片格式"""
    try:
        with Image.open(path) as img:
            return img.format is not None and img.format.upper() in ALLOWED_IMAGE_FORMATS
    except Exception:
        return False


def _parse_fill_color(fill_color: str, device: torch.device):
    """将颜色名称或十六进制值转为归一化 RGB 张量 [1,1,3]"""
    if fill_color == "black":
        fill = torch.tensor([0.0, 0.0, 0.0])
    elif fill_color == "white":
        fill = torch.tensor([1.0, 1.0, 1.0])
    elif fill_color == "gray":
        fill = torch.tensor([0.5, 0.5, 0.5])
    else:
        try:
            rgb = ImageColor.getrgb(fill_color)
            fill = torch.tensor([c / 255.0 for c in rgb])
        except Exception:
            fill = torch.tensor([0.0, 0.0, 0.0])
    return fill.to(device).view(1, 1, 3)


class InteractiveBatchCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_params": ("STRING", {"default": "", "multiline": False, "hidden": True}),
                "node_id": ("STRING", {"default": "", "multiline": False, "hidden": True}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "target_size": ("STRING", {"default": "", "multiline": False, "hidden": True}),
                "max_preview": ("INT", {"default": 10, "min": 1, "max": 500}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Custom Nodes/Interactive"
    OUTPUT_NODE = True

    # 从文件列表逐批加载并处理，返回完整张量
    def _batch_process_files(self, paths, target_size, params, device):
        """分批加载图片并应用裁剪，返回合成后的张量"""
        if not paths:
            raise RuntimeError("没有有效图片")
        # 获取参考尺寸
        ref_path = paths[0]
        with Image.open(ref_path) as im:
            if target_size:
                w, h = map(int, target_size.split("x"))
                if im.size != (w, h):
                    raise RuntimeError(f"第一张图片尺寸 {im.size} 与目标 {target_size} 不符")
                ref_w, ref_h = w, h
            else:
                ref_w, ref_h = im.size
        # 构建 grid (如果裁剪参数有效)
        has_crop = params is not None
        if has_crop:
            crop_x = params["crop_x"]
            crop_y = params["crop_y"]
            crop_w = int(params["crop_w"])
            crop_h = int(params["crop_h"])
            img_x, img_y = params["img_x"], params["img_y"]
            fill_color = params["fill_color"]
            fill = _parse_fill_color(fill_color, device)
            # 生成归一化坐标网格
            y_coords = torch.arange(crop_h, device=device)
            x_coords = torch.arange(crop_w, device=device)
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            canvas_x = crop_x + gx
            canvas_y = crop_y + gy
            src_x = canvas_x - img_x
            src_y = canvas_y - img_y
            grid_x = (src_x / ((ref_w - 1) / 2.0)) - 1.0
            grid_y = (src_y / ((ref_h - 1) / 2.0)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # 1, H, W, 2

        # 分批处理
        outputs = []
        for start in range(0, len(paths), BATCH_SIZE):
            chunk_paths = paths[start:start + BATCH_SIZE]
            # 加载当前批次
            tensors = []
            for p in chunk_paths:
                try:
                    with Image.open(p) as img_file:
                        img = img_file.convert("RGB")
                    if img.size != (ref_w, ref_h):
                        logging.warning(f"跳过尺寸不一致的图片: {p}")
                        continue
                    arr = np.array(img).astype(np.float32) / 255.0
                    tensors.append(torch.from_numpy(arr))
                except Exception as e:
                    logging.warning(f"跳过无法读取的图片: {p} ({e})")
            if not tensors:
                continue
            img_batch = torch.stack(tensors, dim=0).to(device)  # B, H, W, C

            if not has_crop:
                outputs.append(img_batch.cpu())
            else:
                # 将单张网格广播到批次
                batch_grid = grid.expand(img_batch.shape[0], -1, -1, -1)
                img_batch_perm = img_batch.permute(0, 3, 1, 2).float()
                sampled = torch.nn.functional.grid_sample(
                    img_batch_perm, batch_grid, mode='bilinear', padding_mode='zeros', align_corners=True
                )  # B, C, H_out, W_out
                sampled = sampled.permute(0, 2, 3, 1)  # B, H_out, W_out, C
                # 蒙版
                src_x_broadcast = src_x.unsqueeze(0).expand(img_batch.shape[0], -1, -1)
                src_y_broadcast = src_y.unsqueeze(0).expand(img_batch.shape[0], -1, -1)
                mask = ((src_x_broadcast >= 0) & (src_x_broadcast < ref_w) &
                        (src_y_broadcast >= 0) & (src_y_broadcast < ref_h)).float().unsqueeze(-1)
                fill_batch = fill.expand(img_batch.shape[0], crop_h, crop_w, -1)
                result = sampled * mask + fill_batch * (1 - mask)
                outputs.append(result.cpu())
        if not outputs:
            raise RuntimeError("没有有效图片可处理")
        return torch.cat(outputs, dim=0)

    def process(self, crop_params="", images=None, path="", node_id="",
                target_size="", max_preview=10):
        # 确定图像源
        if images is not None:
            batch_input = images
            source_type = "tensor"
            image_paths = []
        elif path.strip():
            if os.path.isdir(path):
                # 文件夹模式：获取符合条件的路径列表
                exts = ('.png','.jpg','.jpeg','.bmp','.webp')
                files = sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])
                paths_all = [os.path.join(path, f) for f in files]
                if target_size:
                    w, h = map(int, target_size.split("x"))
                    image_paths = []
                    for p in paths_all:
                        if os.path.isfile(p):
                            try:
                                with Image.open(p) as im:
                                    if im.size == (w, h):
                                        image_paths.append(p)
                            except:
                                pass
                else:
                    image_paths = paths_all
                source_type = "folder"
            else:
                # 多文件模式
                paths_all = [p.strip() for p in path.split("|") if p.strip()]
                if target_size:
                    w, h = map(int, target_size.split("x"))
                    image_paths = []
                    for p in paths_all:
                        if os.path.isfile(p):
                            try:
                                with Image.open(p) as im:
                                    if im.size == (w, h):
                                        image_paths.append(p)
                            except:
                                pass
                else:
                    image_paths = paths_all
                source_type = "files"
            if not image_paths:
                raise RuntimeError("没有找到符合要求的图片")
        else:
            raise RuntimeError("必须提供 images 或有效的 path")

        # 解析裁剪参数
        try:
            params = json.loads(crop_params) if crop_params.strip() else None
        except:
            params = None

        # 更新缓存（带 LRU 淘汰）
        if node_id:
            # 如果已存在，先删除（后面重新插入以更新 LRU 顺序）
            if node_id in output_cache:
                del output_cache[node_id]
            # 超过上限则淘汰最旧的条目
            while len(output_cache) >= CACHE_MAX_SIZE:
                output_cache.popitem(last=False)

            entry = {
                "batch": batch_input if source_type == "tensor" else None,
                "source_type": source_type,
                "image_paths": image_paths if source_type != "tensor" else [],
                "max_preview": max_preview,
            }
            output_cache[node_id] = entry

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if source_type == "tensor":
            # tensor 模式：分批处理传入的张量
            if params is None:
                return (batch_input,)
            crop_x = params["crop_x"]
            crop_y = params["crop_y"]
            crop_w = int(params["crop_w"])
            crop_h = int(params["crop_h"])
            img_x = params["img_x"]
            img_y = params["img_y"]
            fill_color = params["fill_color"]
            fill = _parse_fill_color(fill_color, device)
            _, H, W, C = batch_input.shape
            # 生成网格
            y_coords = torch.arange(crop_h, device=device)
            x_coords = torch.arange(crop_w, device=device)
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            canvas_x = crop_x + gx
            canvas_y = crop_y + gy
            src_x = canvas_x - img_x
            src_y = canvas_y - img_y
            grid_x = (src_x / ((W - 1) / 2.0)) - 1.0
            grid_y = (src_y / ((H - 1) / 2.0)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # 1, H_out, W_out, 2

            outputs = []
            total = batch_input.shape[0]
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                chunk = batch_input[start:end].to(device)
                batch_grid = grid.expand(chunk.shape[0], -1, -1, -1)
                chunk_perm = chunk.permute(0, 3, 1, 2).float()
                sampled = torch.nn.functional.grid_sample(
                    chunk_perm, batch_grid, mode='bilinear', padding_mode='zeros', align_corners=True
                )
                sampled = sampled.permute(0, 2, 3, 1)
                mask = ((src_x.unsqueeze(0) >= 0) & (src_x.unsqueeze(0) < W) &
                        (src_y.unsqueeze(0) >= 0) & (src_y.unsqueeze(0) < H)).float().unsqueeze(-1)
                fill_chunk = fill.expand(end-start, crop_h, crop_w, -1)
                result_chunk = sampled * mask + fill_chunk * (1 - mask)
                outputs.append(result_chunk.cpu())
            result = torch.cat(outputs, dim=0)
            return (result,)

        else:
            # 文件类模式：分批加载并处理
            result = self._batch_process_files(image_paths, target_size, params, device)
            return (result,)


# ============= 合并后的路由（2个统一入口替代原来的5个） =============

@PromptServer.instance.routes.get("/interactive_crop/get_source_info")
async def get_source_info(request):
    """
    统一入口：获取文件源的尺寸分布和总数
    合并自：get_folder_sizes / get_files_sizes（含 filter_paths_by_size 的统计部分）
    
    参数：
        source: "folder" | "files"
        path: 文件夹路径 或 以 | 分隔的文件路径列表
        size_filter: 可选，只统计匹配尺寸（格式 "WxH"）
    
    返回：
        { sizes: [{size: "WxH", count: N}, ...], total: N }
    """
    source = request.query.get("source", "folder")
    path = request.query.get("path", "")
    size_filter = request.query.get("size_filter", "")

    # 安全检测：频率限制
    if not _check_rate_limit(request):
        return web.json_response({"error": "Too many requests"}, status=429)

    if not path:
        return web.json_response({"error": "Missing path"}, status=400)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    if source == "folder":
        if not os.path.isdir(path):
            return web.json_response({"error": "Invalid folder"}, status=400)
        try:
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])
        except Exception:
            return web.json_response({"error": "Cannot read folder"}, status=400)
        all_paths = [os.path.join(path, f) for f in files]
    else:  # files
        all_paths = [p.strip() for p in path.split("|") if p.strip()]
        if not all_paths:
            return web.json_response({"error": "No valid paths"}, status=400)

    # 统计尺寸分布
    size_counts = {}
    for p in all_paths:
        try:
            if os.path.isfile(p):
                with Image.open(p) as im:
                    s = f"{im.width}x{im.height}"
                    size_counts[s] = size_counts.get(s, 0) + 1
        except Exception:
            pass

    sizes = [{"size": s, "count": c} for s, c in size_counts.items()]
    sizes.sort(key=lambda x: x["count"], reverse=True)

    total = sum(c for _, c in size_counts.items())
    if size_filter:
        total = size_counts.get(size_filter, 0)

    return web.json_response({"sizes": sizes, "total": total})


@PromptServer.instance.routes.get("/interactive_crop/get_preview")
async def get_preview(request):
    """
    统一入口：获取指定索引的图片预览（base64）
    合并自：get_image_by_path / get_folder_preview / filter_paths_by_size（预览部分）

    参数：
        source: "folder" | "files" | "tensor"
        path:  文件夹路径 或 |分隔的文件列表（tensor 模式不需要）
        index: 图片索引（默认 0）
        size_filter: 可选，文件夹模式只匹配指定尺寸
        node_id: tensor 模式需要

    返回：
        { image: "data:image/png;base64,...", total: N }
    """
    source = request.query.get("source", "folder")
    path = request.query.get("path", "")
    index = int(request.query.get("index", "0"))
    size_filter = request.query.get("size_filter", "")
    node_id = request.query.get("node_id", "")

    # 安全检测：频率限制
    if not _check_rate_limit(request):
        return web.json_response({"error": "Too many requests"}, status=429)

    from io import BytesIO
    import base64

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    if source == "tensor":
        # tensor 模式：从缓存读取（不操作文件系统）
        if not node_id or node_id not in output_cache:
            return web.json_response({"error": "Node not executed yet"}, status=404)
        cache = output_cache[node_id]
        if cache["source_type"] != "tensor":
            return web.json_response({"error": "Not tensor mode"}, status=400)
        batch = cache["batch"]
        if index < 0 or index >= batch.shape[0]:
            return web.json_response({"error": "Index out of range"}, status=400)
        import torchvision.transforms.functional as TF
        img_tensor = batch[index].permute(2, 0, 1)
        img_pil = TF.to_pil_image(img_tensor.clamp(0, 1))
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return web.json_response({"image": f"data:image/png;base64,{b64}", "total": batch.shape[0]})

    # 文件类模式：解析路径列表
    if source == "folder":
        if not path or not os.path.isdir(path):
            return web.json_response({"error": "Invalid folder"}, status=400)
        try:
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])
        except Exception:
            return web.json_response({"error": "Cannot read folder"}, status=400)
        all_paths = [os.path.join(path, f) for f in files]
    else:  # files
        if not path:
            return web.json_response({"error": "Missing path"}, status=400)
        all_paths = [p.strip() for p in path.split("|") if p.strip()]

    if not all_paths:
        return web.json_response({"error": "No valid images found"}, status=400)

    # 如果指定尺寸过滤，筛选
    if size_filter:
        try:
            w, h = map(int, size_filter.split("x"))
        except ValueError:
            return web.json_response({"error": "Invalid size_filter format"}, status=400)
        filtered_paths = []
        for p in all_paths:
            if os.path.isfile(p):
                try:
                    with Image.open(p) as im:
                        if im.size == (w, h):
                            filtered_paths.append(p)
                except Exception:
                    pass
        all_paths = filtered_paths

    if index < 0 or index >= len(all_paths):
        return web.json_response({"error": "Index out of range"}, status=400)

    try:
        with Image.open(all_paths[index]) as img:
            img_rgb = img.convert("RGB")
            buf = BytesIO()
            img_rgb.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
        return web.json_response({"image": f"data:image/png;base64,{b64}", "total": len(all_paths)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ============= 保留的独立路由 =============

# tensor 预览（单独保留，只读缓存，不操作文件系统）
@PromptServer.instance.routes.get("/interactive_crop/get_tensor_preview")
async def get_tensor_preview(request):
    """保留：仅读内存缓存，不接触文件系统"""
    node_id = request.query.get("node_id")
    index = int(request.query.get("index","0"))
    if not node_id or node_id not in output_cache:
        return web.json_response({"error":"Node not executed yet"}, status=404)
    cache = output_cache[node_id]
    if cache["source_type"] != "tensor":
        return web.json_response({"error":"Not tensor mode"}, status=400)
    batch = cache["batch"]
    if index<0 or index>=batch.shape[0]:
        return web.json_response({"error":"Index out of range"}, status=400)
    import torchvision.transforms.functional as TF
    from io import BytesIO
    import base64
    img_tensor = batch[index].permute(2,0,1)
    img_pil = TF.to_pil_image(img_tensor.clamp(0,1))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return web.json_response({"image": f"data:image/png;base64,{b64}", "total": batch.shape[0]})

# tkinter 文件夹选择弹窗（需用户交互）
@PromptServer.instance.routes.post("/interactive_crop/select_folder")
async def select_folder(request):
    """只弹窗选择单个目录（单选），不循环多选"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()

        dir_path = filedialog.askdirectory(title="选择图片文件夹")
        root.destroy()

        return web.json_response({"path": dir_path if dir_path else ""})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# tkinter 文件选择弹窗（需用户交互）
@PromptServer.instance.routes.post("/interactive_crop/select_files")
async def select_files(request):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_paths = filedialog.askopenfilenames(title="选择图片文件（可多选）")
        root.destroy()
        if not file_paths:
            return web.json_response({"paths": ""})
        paths_str = "|".join(file_paths)
        return web.json_response({"paths": paths_str, "count": len(file_paths)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "InteractiveBatchCrop": InteractiveBatchCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveBatchCrop": "交互式批量裁剪",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
