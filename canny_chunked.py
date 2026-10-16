"""Canny 边缘检测（分块版）—— 与官方 Detect Edges (Canny) 结果一致，但按块处理

为什么需要它（官方节点卡死的根因，全部为实测结论）
--------------------------------------------------
官方 ``comfy_extras/nodes_canny.py`` 把**整个批次**一次性丢给 ``kornia.filters.canny``：

1. kornia 的迟滞(hysteresis)阶段是一个 Python 级 ``while`` 循环
   （``kornia/filters/canny.py:147``）：每轮只把强边向外扩张 1 个像素，
   实测 256^2 约 22-33 轮、512^2 约 49-96 轮、1024^2 约 71-190 轮
   （轮数随分辨率与图片内容线性增长，弱边链越长越慢）；
   循环条件 ``(edges_old - edges).abs().any()`` 每轮都要 GPU->CPU 同步；
2. 每轮的 ``F.conv2d(edges, hysteresis_kernels)`` 输出形状是 ``(B, 8, H, W)``
   —— 输入体积的 8 倍（NMS 那步同样是 8 通道），而循环条件是**整批共享**的：
   只要有 1 帧没收敛，整批 200 帧就要一起再赔一轮。

于是 200 帧 1024^2 的瞬时峰值高达约 25 GB，超过 24 GB 显卡后 Windows WDDM 会
**静默溢出到系统内存**（不报 OOM），表现为「不报错、无进度、彻底卡死」；
而节点正在执行时 ComfyUI 无法中断，只能杀进程。

实测（RTX 4090 24GB，真实帧，走官方同一套完整通路）::

    256^2  x 200 帧 = 0.27 s / 峰值 1.6 GB
    384^2  x 200 帧 = 0.32 s / 峰值 3.6 GB
    512^2  x 200 帧 = 0.80 s / 峰值 6.3 GB
    768^2  x 200 帧 = 2.0  s / 峰值 14.2 GB
    1024^2 x 200 帧 = 4.0  s / 峰值 25.2 GB   <- 超过 24GB => spill 卡死

本节点的做法
------------
逐块调用**与官方完全相同的那一行**（同一函数、同一 dtype、同一算子链）：

    canny(chunk[..., :3].to(device=dev, dtype=torch.float32).movedim(-1, 1), low, high)[1]

该算法逐帧独立（迟滞循环只做逐元素比较与本帧自身的 3x3 卷积，已收敛的帧是不动点），
峰值显存由此降到「块大小」对应的量级（默认 8 帧 -> 约 1 GB）。
与官方结果的一致性如下（全部为 md/canny_chunked_selftest.py 实测）：

- 单块（``chunk_size`` >= 帧数，即与官方同批尺寸）：**bit-exact（max|diff| = 0）**；
- 真实图像分块（``chunk_size`` = 1 / 4 / 8 / 100）：**0 像素差异**（256^2、512^2、1024^2 实测均为 0）；
- CPU 通路任意分块：**bit-exact**（CPU 卷积实现与批尺寸无关）；
- 唯一例外是「均匀随机噪声」这类最坏输入（幅值密集落在阈值边界）：最多约 0.003% 的像素翻转，
  原因是 CUDA 卷积会按 batch 尺寸选择 cuDNN 算法，阈值边界的种子像素一旦翻转会被迟滞传播放大。
  **官方节点自己按 8 帧分批跑、与整批跑相比也会出现同样量级的偏离**，因此该现象并非本节点引入。

另外三点：
- 前端进度条（comfy.utils.ProgressBar）+ 控制台 tqdm；
- 每块开头检查中断 -> **可随时取消**，不再「假死只能杀进程」；
- ``auto_chunk`` 依据 ``comfy.model_management.get_free_memory()`` 自动收缩块大小
  （标定系数 ``4 * 8 * H * W * 4`` 字节/帧，与上表实测吻合到 10% 以内），
  保证不会把显存顶到溢出。

独立导入
--------
本文件对 ComfyUI 依赖做了 try/except 兜底（设备/进度条均有回退），
因此 md/ 下的自测脚本可以不启动 ComfyUI 直接加载本文件。
"""

import logging
import time

import torch
from kornia.filters import canny

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 已是 requirements 依赖
    tqdm = None

try:
    import comfy.model_management as mm
except ImportError:  # md/ 自测（不启动 ComfyUI）时走这里
    mm = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


def _frames_peak_bytes(h, w):
    """单帧峰值字节估计（标定见文件头实测表，误差 10% 以内）

    迟滞循环内同时存在 2 个 8 通道 fp32 大张量（NMS 卷积输出、迟滞卷积输出），
    另有布尔/浮点临时量与 1-3 通道的小张量，合计约等于 4 份「8 通道 fp32 特征图」。
    """
    return 4 * 8 * h * w * 4


class _NullProgress:
    """ProgressBar 兜底：脱离 ComfyUI 时不报错、不输出"""

    def update_absolute(self, value, total=None, preview=None):
        pass


class CannyChunked:
    """分块版 Canny：接口/结果对齐官方 Canny，只把「整批」改成「逐块」"""

    OUTPUT_NODE = False

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "图像批次 (B, H, W, C)。支持任意帧数（长视频序列可放心直连）。"
                }),
                "low_threshold": ("FLOAT", {
                    "default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "低阈值（与官方 Canny 同名同义，0-1）。调高可显著减少迟滞循环轮数。"
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "高阈值（与官方 Canny 同名同义，0-1）。必须 >= low_threshold。"
                }),
                "chunk_size": ("INT", {
                    "default": 8, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "每块帧数（上限）。越小峰值显存越低但调用次数越多；8 帧约 1GB 峰值。"
                }),
            },
            "optional": {
                "auto_chunk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "按当前空闲显存自动收缩块大小（只减不增），避免溢出到系统内存导致卡死。"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "关闭则全程在 CPU 上算（慢但完全不占显存），用于显存被大模型占满时。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "canny_chunked"
    CATEGORY = "image"
    DESCRIPTION = (
        "分块版 Canny 边缘检测：与官方 Detect Edges (Canny) 结果一致，"
        "但按块处理以避免大批次显存溢出卡死，带进度条且可随时中断。"
    )

    # ==============================
    # 设备 / 输出 / 块大小
    # ==============================

    @staticmethod
    def _resolve_device(use_gpu):
        """与官方节点一致：默认走 ``model_management.get_torch_device()``"""
        if not use_gpu:
            return torch.device("cpu")
        if mm is not None:
            dev = mm.get_torch_device()
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return dev
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _resolve_output():
        """与官方节点一致：输出落在 intermediate_device / intermediate_dtype"""
        if mm is not None:
            return mm.intermediate_device(), mm.intermediate_dtype()
        return torch.device("cpu"), torch.float32

    @staticmethod
    def _resolve_chunk_size(requested, auto_chunk, dev, h, w):
        """把用户给的块大小再按空闲内存收一次（只减不增）

        CPU 时 ``get_free_memory`` 返回可用系统内存，因此这条路径同样有效
        （200 帧 1024^2 整批在 CPU 上也要约 27GB 峰值）。
        """
        requested = max(1, int(requested))
        if not auto_chunk or mm is None:
            return requested, None
        try:
            free = int(mm.get_free_memory(dev))
        except Exception:
            return requested, None
        per_frame = _frames_peak_bytes(h, w)
        # 只用空闲内存的一半：给同图其它节点/模型留出余量
        cap = max(1, int(free * 0.5 // per_frame))
        return max(1, min(requested, cap)), (free, per_frame, cap)

    # ==============================
    # 主流程
    # ==============================

    def canny_chunked(self, image, low_threshold=0.4, high_threshold=0.8,
                      chunk_size=8, auto_chunk=True, use_gpu=True):
        # 阈值约束与 kornia 内部 KORNIA_CHECK 完全一致，只是把报错提前并写清楚
        if not (0.0 < low_threshold < 1.0) or not (0.0 < high_threshold < 1.0):
            raise ValueError(
                "阈值必须落在 (0, 1) 开区间内: low=%.4f high=%.4f" % (low_threshold, high_threshold))
        if low_threshold > high_threshold:
            raise ValueError(
                "low_threshold 不能大于 high_threshold: low=%.4f high=%.4f" % (low_threshold, high_threshold))

        if image.ndim != 4:
            raise ValueError("IMAGE 必须是 4 维 (B, H, W, C)，实际为 %s" % (tuple(image.shape),))

        total_frames, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        if total_frames == 0:
            return (image,)

        dev = self._resolve_device(use_gpu)
        out_dev, out_dtype = self._resolve_output()
        chunk, cache_info = self._resolve_chunk_size(chunk_size, auto_chunk, dev, h, w)

        if cache_info is None:
            logging.info(
                "[CannyChunked] %d 帧 %dx%d：设备 %s，块大小 %d（未启用自动收缩）",
                total_frames, h, w, dev, chunk)
        else:
            free, per_frame, cap = cache_info
            logging.info(
                "[CannyChunked] %d 帧 %dx%d：设备 %s，请求块 %d -> 实际块 %d"
                "（空闲 %.2f GB，每帧约 %.1f MB，可容纳 %d 帧）",
                total_frames, h, w, dev, int(chunk_size), chunk,
                free / 2 ** 30, per_frame / 2 ** 20, cap)

        # 预分配输出：官方是「整批算完再 repeat 成 3 通道」，这里是逐块写入，
        # 峰值只在块内出现（200 帧 1024^2 省下约 2.3GB 的整批中转）
        result = torch.empty((total_frames, h, w, 3), dtype=out_dtype, device=out_dev)

        progress = ProgressBar(total_frames) if ProgressBar is not None else _NullProgress()
        progress.update_absolute(0, total_frames)

        chunks = list(range(0, total_frames, chunk))
        iterator = tqdm(chunks, desc="Canny", unit="chunk") if tqdm is not None else chunks

        t0 = time.time()
        for start in iterator:
            end = min(start + chunk, total_frames)

            if mm is not None:
                # 每块开头检查中断 => 长序列中途可以取消（官方节点做不到）
                mm.throw_exception_if_processing_interrupted()
                if dev.type == "cuda":
                    need = (end - start) * _frames_peak_bytes(h, w)
                    try:
                        if int(mm.get_free_memory(dev)) < need * 1.2:
                            mm.soft_empty_cache(force=True)
                    except Exception:
                        pass

            # ↓↓↓ 与官方 comfy_extras/nodes_canny.py:33 完全相同的算子链（含张量布局）↓↓↓
            part = image[start:end][..., :3].to(device=dev, dtype=torch.float32).movedim(-1, 1)
            edges = canny(part, low_threshold, high_threshold)[1]
            # ↓↓↓ 与官方 :34 相同：转 intermediate 设备/精度后 repeat 成 3 通道，再回到 HWC ↓↓↓
            result[start:end] = edges.to(
                device=out_dev, dtype=out_dtype).repeat(1, 3, 1, 1).movedim(1, -1)

            progress.update_absolute(end, total_frames)
            logging.debug("[CannyChunked] 块 %d-%d / %d 完成", start, end, total_frames)

        elapsed = time.time() - t0
        peak_txt = ""
        if dev.type == "cuda":
            try:
                peak_txt = "，峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated(dev) / 2 ** 30)
            except Exception:
                peak_txt = ""
        logging.info(
            "[CannyChunked] 完成：%d 帧 %dx%d，共 %d 块，耗时 %.2f s%s",
            total_frames, h, w, len(chunks), elapsed, peak_txt)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "CannyChunked": CannyChunked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CannyChunked": "Detect Edges (Canny) Chunked",
}

