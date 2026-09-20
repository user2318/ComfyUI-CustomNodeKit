import torch
import numpy as np
import os
import json
import logging
import time
import asyncio
import threading
from collections import defaultdict, OrderedDict
from PIL import Image, ImageColor
from server import PromptServer
from aiohttp import web
import comfy.model_management

# 每批处理的图片数量，可根据显存大小调整
BATCH_SIZE = 32
CACHE_MAX_SIZE = 50  # output_cache 最多保留 50 个节点的数据
CACHE_MAX_BYTES = 2 << 30  # output_cache 所有条目合计内存预算（2GB），超出按 LRU 淘汰
output_cache = OrderedDict()  # OrderedDict 实现 LRU 淘汰


def _entry_bytes(entry) -> int:
    """估算单条缓存占用的字节数（仅统计预览帧张量）"""
    frames = entry.get("frames")
    if frames is None:
        return 0
    try:
        return int(frames.numel()) * int(frames.element_size())
    except Exception:
        return 0


def _cache_total_bytes() -> int:
    return sum(_entry_bytes(e) for e in output_cache.values())

# 前端预览取帧规则（见 js/interactive_batch_crop.js:223-240）：
#   total <= max_preview                -> 取全部帧
#   max_preview < total < 2*max_preview -> 弹窗自选一段连续区间（最多 2*max_preview-1 帧）
#   total >= 2*max_preview              -> step = ceil(total / max_preview)，等距采样
# 所以预览只需要"前端真正会请求的那几帧"，没必要把整批张量常驻内存
# （2600 帧 @480×832 float32 整批约 12.5GB，按此规则只有 max_preview 帧量级）
# 前端预览取帧规则（见 js/interactive_batch_crop.js:223-240）：
#   total <= max_preview                -> 取全部帧
#   max_preview < total < 2*max_preview -> 弹窗自选一段连续区间（最多 2*max_preview-1 帧）
#   total >= 2*max_preview              -> step = ceil(total / max_preview)，等距采样
# 所以预览只需要"前端真正会请求的那几帧"，没必要把整批张量常驻内存
# （2600 帧 @480×832 float32 整批约 12.5GB，按此规则只有 max_preview 帧量级）
PREVIEW_CACHE_MAX_BYTES = 1 << 30  # 预览缓存内存预算（1GB），按分辨率自适应换算成帧数


def _build_preview_cache(batch, max_preview: int):
    """按前端预览规则抽取帧子集，返回 (frames, index_map, total)

    index_map 记录每个缓存帧在原批次中的真实索引，路由据此精确还原请求的帧。

    常规路径（total >= 2*max_preview）采样步距与前端完全一致，因此逐帧精确命中；
    仅当"弹窗自选区间"且整批超出内存预算时才会出现近似（回退到最接近的缓存帧）。
    """
    total = int(batch.shape[0])
    if total <= 0:
        return batch[:0], [], total
    per_frame = max(int(batch[0].numel()) * int(batch.element_size()), 1)
    byte_budget_frames = max(PREVIEW_CACHE_MAX_BYTES // per_frame, 1)
    # 可缓存帧数上限：不超过"弹窗自选区间"所需（2*max_preview），也不超过字节预算
    limit = min(max(int(max_preview) * 2, 1), byte_budget_frames)
    if total <= limit:
        idx = list(range(total))
    else:
        step = max(1, (total + int(max_preview) - 1) // int(max_preview))
        idx = list(range(0, total, step))
        if len(idx) > byte_budget_frames:
            step = (total + byte_budget_frames - 1) // byte_budget_frames
            idx = list(range(0, total, step))
    return batch[idx].cpu(), idx, total


def _resolve_cached_preview(cache, index: int):
    """在预览缓存中定位请求的帧。

    精确命中（常规路径）时返回真实帧；未命中时回退到最接近的已缓存帧，
    避免返回错误框（例如用户在前端自选了采样步距之外的区间）。
    返回 (frame, total, exact)；frame 为 None 表示索引越界。
    """
    total = int(cache.get("total", 0))
    if index < 0 or index >= total:
        return None, total, False
    frames = cache.get("frames")
    index_map = cache.get("index_map") or []
    if frames is None or len(index_map) == 0:
        return None, total, False
    if index in index_map:
        return frames[index_map.index(index)], total, True
    pos = min(range(len(index_map)), key=lambda i: abs(index_map[i] - index))
    return frames[pos], total, False

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
                "path": ("STRING", {"default": "", "multiline": False, "tooltip": "图片文件夹路径或用 | 分隔的多个文件路径。Folder path or multiple file paths separated by |."}),
                "target_size": ("STRING", {"default": "", "multiline": False, "hidden": True}),
                "max_preview": ("INT", {"default": 10, "min": 1, "max": 500, "tooltip": "前端预览中显示的最大图片数量。Maximum number of images shown in the frontend preview."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "直接传入图片张量作为裁剪源（与 path 二选一）。Image tensor input as the crop source (alternative to path)."}),
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
        # 预分配输出后逐块写入，避免"攒 outputs + torch.cat"的双倍峰值
        # 容量按路径总数分配；若因尺寸不符/损坏跳过了部分图片，末尾截断
        # （调用方通常已按 target_size 过滤过路径，正常情况下不会明显浪费容量）
        capacity = len(paths)
        out_h = crop_h if has_crop else ref_h
        out_w = crop_w if has_crop else ref_w
        result_all = torch.empty((capacity, out_h, out_w, 3), dtype=torch.float32)
        written = 0
        for start in range(0, capacity, BATCH_SIZE):
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
                n_chunk = img_batch.shape[0]
                result_all[written:written + n_chunk] = img_batch.cpu()
                written += n_chunk
            else:
                # 将单张网格广播到批次
                batch_grid = grid.expand(img_batch.shape[0], -1, -1, -1)
                img_batch_perm = img_batch.permute(0, 3, 1, 2).float()
                sampled = torch.nn.functional.grid_sample(
                    img_batch_perm, batch_grid, mode='bilinear', padding_mode='border', align_corners=True
                )  # B, C, H_out, W_out
                sampled = sampled.permute(0, 2, 3, 1)  # B, H_out, W_out, C
                # 蒙版
                src_x_broadcast = src_x.unsqueeze(0).expand(img_batch.shape[0], -1, -1)
                src_y_broadcast = src_y.unsqueeze(0).expand(img_batch.shape[0], -1, -1)
                mask = ((src_x_broadcast >= 0) & (src_x_broadcast < ref_w) &
                        (src_y_broadcast >= 0) & (src_y_broadcast < ref_h)).float().unsqueeze(-1)
                fill_batch = fill.expand(img_batch.shape[0], crop_h, crop_w, -1)
                chunk_result = sampled * mask + fill_batch * (1 - mask)
                n_chunk = chunk_result.shape[0]
                result_all[written:written + n_chunk] = chunk_result.cpu()
                written += n_chunk
        if written == 0:
            raise RuntimeError("没有有效图片可处理")
        return result_all if written == capacity else result_all[:written]

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
                            except Exception:
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
                            except Exception:
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
        except Exception:
            params = None

        # 更新缓存（LRU：条数上限 + 全局字节预算）
        if node_id:
            # 如果已存在，先删除（后面重新插入以更新 LRU 顺序）
            if node_id in output_cache:
                del output_cache[node_id]

            if source_type == "tensor":
                # 只缓存前端预览真正会请求的帧（规则见 _build_preview_cache），
                # 避免整批图像张量常驻内存（长序列可达数 GB，且会阻碍缓存回收）
                preview_frames, preview_index_map, preview_total = _build_preview_cache(
                    batch_input, max_preview
                )
            else:
                preview_frames, preview_index_map, preview_total = None, [], 0

            entry = {
                "frames": preview_frames,
                "index_map": preview_index_map,
                "total": preview_total,
                "source_type": source_type,
                "image_paths": image_paths if source_type != "tensor" else [],
                "max_preview": max_preview,
            }
            entry["bytes"] = _entry_bytes(entry)

            # 淘汰最旧条目：条数超上限，或加上新条目后会超出总字节预算
            # （单条本身超过预算时允许保留，避免全部淘汰后预览完全不可用）
            while len(output_cache) >= CACHE_MAX_SIZE or (
                    output_cache and _cache_total_bytes() + entry["bytes"] > CACHE_MAX_BYTES):
                output_cache.popitem(last=False)

            output_cache[node_id] = entry

        # 统一走 ComfyUI 的设备管理（兼容 AMD/Intel 后端与 --cpu/--novram 配置）
        device = comfy.model_management.get_torch_device()

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

            # 预分配输出后逐块写入，避免"攒 outputs + torch.cat"的双倍峰值
            total = batch_input.shape[0]
            result = torch.empty((total, crop_h, crop_w, 3), dtype=torch.float32)
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                chunk = batch_input[start:end].to(device)
                batch_grid = grid.expand(chunk.shape[0], -1, -1, -1)
                chunk_perm = chunk.permute(0, 3, 1, 2).float()
                sampled = torch.nn.functional.grid_sample(
                    chunk_perm, batch_grid, mode='bilinear', padding_mode='border', align_corners=True
                )
                sampled = sampled.permute(0, 2, 3, 1)
                mask = ((src_x.unsqueeze(0) >= 0) & (src_x.unsqueeze(0) < W) &
                        (src_y.unsqueeze(0) >= 0) & (src_y.unsqueeze(0) < H)).float().unsqueeze(-1)
                fill_chunk = fill.expand(end-start, crop_h, crop_w, -1)
                result_chunk = sampled * mask + fill_chunk * (1 - mask)
                result[start:end] = result_chunk.cpu()
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
        frame, total, _exact = _resolve_cached_preview(cache, index)
        if frame is None:
            return web.json_response({"error": "Index out of range"}, status=400)
        import torchvision.transforms.functional as TF
        img_tensor = frame.permute(2, 0, 1)
        img_pil = TF.to_pil_image(img_tensor.clamp(0, 1))
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return web.json_response({"image": f"data:image/png;base64,{b64}", "total": total})

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
    frame, total, _exact = _resolve_cached_preview(cache, index)
    if frame is None:
        return web.json_response({"error":"Index out of range"}, status=400)
    import torchvision.transforms.functional as TF
    from io import BytesIO
    import base64
    img_tensor = frame.permute(2,0,1)
    img_pil = TF.to_pil_image(img_tensor.clamp(0,1))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return web.json_response({"image": f"data:image/png;base64,{b64}", "total": total})

# 文件/文件夹选择弹窗
# 注意: 依赖 tkinter（Python 标准 GUI 库）。
# 如果使用 ComfyUI 官方 Release 便携版，嵌入式 Python 可能缺少 tkinter。
# 解决方案: 运行 install.py 或重启 ComfyUI 后会自动从 NuGet 下载缺失组件。
#
# tkinter 模态对话框是阻塞调用：直接写在 async 路由里会在弹窗期间卡住整个 aiohttp
# 事件循环（队列/进度/其它前端请求全部无响应）。这里放到工作线程执行并加锁串行化；
# 若运行环境要求 Tk 主线程（如 macOS），自动回退为同步调用（等同改动前行为）。
_DIALOG_LOCK = threading.Lock()


def _run_tk_dialog(pick_fn):
    """在工作线程中执行 tkinter 对话框"""
    with _DIALOG_LOCK:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        try:
            return pick_fn(root)
        finally:
            root.destroy()


async def _exec_dialog(pick_fn):
    """异步执行对话框：优先线程池，失败回退同步"""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_tk_dialog, pick_fn)
    except Exception:
        return _run_tk_dialog(pick_fn)


@PromptServer.instance.routes.post("/interactive_crop/select_folder")
async def select_folder(request):
    """只弹窗选择单个目录（单选），不循环多选"""
    try:
        def pick(root):
            from tkinter import filedialog
            return filedialog.askdirectory(title="选择图片文件夹")

        dir_path = await _exec_dialog(pick)
        return web.json_response({"path": dir_path if dir_path else ""})
    except ImportError:
        return web.json_response({
            "error": "tkinter 不可用。请重启 ComfyUI，本插件将在启动时自动下载并安装 tkinter 组件。"
        }, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# tkinter 文件选择弹窗（需用户交互）
@PromptServer.instance.routes.post("/interactive_crop/select_files")
async def select_files(request):
    try:
        def pick(root):
            from tkinter import filedialog
            return filedialog.askopenfilenames(title="选择图片文件（可多选）")

        file_paths = await _exec_dialog(pick)
        if not file_paths:
            return web.json_response({"paths": ""})
        paths_str = "|".join(file_paths)
        return web.json_response({"paths": paths_str, "count": len(file_paths)})
    except ImportError:
        return web.json_response({
            "error": "tkinter 不可用。请重启 ComfyUI，本插件将在启动时自动下载并安装 tkinter 组件。"
        }, status=500)
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
