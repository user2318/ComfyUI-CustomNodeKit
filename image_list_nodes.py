"""图像 List（列表）相关节点

本文件专门存放与「图像列表（IMAGE list）」相关的节点，后续新增的 list 类节点都放这里。

列表语义（与普通 IMAGE 批次的关键差异）：
- 本文件节点使用 ``INPUT_IS_LIST = True``：节点在一次执行中拿到**整个列表**，列表元素是各自
  独立的图像张量，**尺寸可以互不相同**（普通批次 ``[B, H, W, C]`` 要求批内尺寸一致）。
- 上游如果给的是普通批次，缓存里的值是 ``[batch]``（只有 1 个元素），本文件会把它**拆成 B 个
  单张元素**（batch -> list），所以「列表」和「批次」两种上游都能接。
- 空输出一律返回 ``None`` —— 与「这根线根本没接」完全等价：
  下游把该输入声明为 optional 时走 Python 默认值（按"这一路没有"处理：透传/忽略），
  下游声明为 required 时会在执行期报错（等价于未连接时的 ``Required input is missing``，只是报错从提交挪到执行）。
  注意 **不能**用空列表 ``[]`` 当空输出：ComfyUI 的 ``slice_dict`` 会对空列表取 ``[-1]``，
  直接 IndexError（所以节点 1 不用 ``OUTPUT_IS_LIST``，只有节点 2 的输出是列表）。
- **不要**改成 ``ExecutionBlocker(None)``：那会让下游节点"整块不执行"并静默连坐（下游的其它输入也一起失效、
  再往下游全部停摆，界面上毫无提示），与"这一路没接"的直觉不符。
"""

import logging
import math

import numpy as np
import torch
from PIL import Image


MAX_OUTPUTS = 16


# ==============================
# 工具函数
# ==============================

def _empty_output():
    """单值输出的「空」：返回 None —— 下游效果等同于「这根线没接」"""
    return None


def is_empty_output(value):
    """判断某个输出是否为「空」（自测脚本也用它）"""
    return value is None


def _empty_list_output():
    """列表输出（OUTPUT_IS_LIST）的「空」

    有 ``OUTPUT_IS_LIST`` 时整个返回值必须是可迭代的列表（ComfyUI 会对它做 ``extend``），
    所以这里返回 ``([None],)``：下游拿到的同样是 ``None``，效果同样等同于「这根线没接」。
    """
    return ([None],)


def _scalar(value, default, cast=int):
    """取出标量：INPUT_IS_LIST 下 widget 值会以 ``[值]`` 的形式传进来"""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        value = value[0]
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _as_single_image(item):
    """把列表元素规范化为单帧 4D 张量 ``[1, H, W, C]``；无法识别时返回 None"""
    if not isinstance(item, torch.Tensor):
        return None
    if item.ndim == 3:      # [H, W, C] -> [1, H, W, C]
        return item.unsqueeze(0)
    if item.ndim == 4:      # [1, H, W, C]（多帧元素则原样保留）
        return item
    return None


def normalize_image_list(value, expand_single_batch=True):
    """把输入值规范化为「图像元素列表」

    - ``None`` / 空 => ``[]``
    - 真列表（列表语义）=> 逐元素规范化，元素尺寸允许不同
    - 普通批次（``[B, H, W, C]``，作为唯一元素传入，即 ``[batch]``）=> 拆成 B 个单张元素

    注意：当列表**只有 1 个元素且该元素是多帧批次**时，会按「批次」理解并拆开（batch -> list），
    这是「两种上游都能吃」的代价，详见 README。
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        items = list(value)
        if (expand_single_batch and len(items) == 1 and isinstance(items[0], torch.Tensor)
                and items[0].ndim == 4 and items[0].shape[0] > 1):
            batch = items[0]
            items = [batch[i:i + 1] for i in range(batch.shape[0])]
        result = []
        for item in items:
            img = _as_single_image(item)
            if img is not None:
                result.append(img)
        return result

    if isinstance(value, torch.Tensor):
        if value.ndim == 4 and value.shape[0] > 1:
            return [value[i:i + 1] for i in range(value.shape[0])]
        img = _as_single_image(value)
        return [img] if img is not None else []

    return []


def to_single_images(value):
    """规范化为「单张 3D 图 ``[H, W, C]``」列表

    ``normalize_image_list`` 返回的元素可能带 batch 维度（真列表元素通常是 ``[1,H,W,C]``），
    这里统一压成 3D；若某个元素本身是多帧批次（``[B,H,W,C]``，B>1）则拆成 B 张。
    """
    result = []
    for item in normalize_image_list(value):
        if item.ndim == 4:
            for i in range(int(item.shape[0])):
                result.append(item[i])
        else:
            result.append(item)
    return result


def _to_rgb(img):
    """把单张图规范化为 3 通道 RGB 张量 ``[H, W, 3]``（0-1）

    - 1 通道（灰度）-> 复制成 RGB
    - 2 通道（灰度+alpha）-> 灰度转 RGB（丢弃 alpha）
    - 3 通道 -> 原样
    - >=4 通道 -> 取前 3 通道（丢弃 alpha）
    """
    if not isinstance(img, torch.Tensor):
        return None
    c = int(img.shape[-1])
    if c == 1:
        return img.repeat(1, 1, 3)
    if c == 2:
        return img[..., 0:1].repeat(1, 1, 3)
    if c == 3:
        return img
    if c >= 4:
        return img[..., :3]
    return None


def _to_rgba(img):
    """把单张图规范化为 RGBA 张量 ``[H, W, 4]``（0-1）

    - >=4 通道 -> 取前 4 通道（保留其 alpha）
    - 2 通道（灰度+alpha）-> 灰度转 RGB，保留 alpha
    - 1 / 3 通道 -> 灰度/RGB 补 alpha=1
    """
    if not isinstance(img, torch.Tensor):
        return None
    c = int(img.shape[-1])
    if c >= 4:
        return img[..., :4].contiguous()
    h, w = int(img.shape[0]), int(img.shape[1])
    alpha = img[..., 1:2] if c == 2 else torch.ones((h, w, 1), dtype=img.dtype)
    return torch.cat([_to_rgb(img), alpha], dim=-1)


def _resize_image(img, new_w, new_h):
    """把单张图 ``[H, W, C]``（0-1）缩放到 ``new_w × new_h``（PIL LANCZOS，与仓库其它节点一致）"""
    if int(img.shape[-1]) == 2:      # PIL 不支持 2 通道，先转 RGBA
        img = _to_rgba(img)
    arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    channels = arr.shape[-1]
    mode = "RGBA" if channels == 4 else ("L" if channels == 1 else "RGB")
    pil = Image.fromarray(arr, mode=mode)
    pil = pil.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS)
    out = np.array(pil).astype(np.float32) / 255.0
    if out.ndim == 2:
        out = out[..., None]
    return torch.from_numpy(out)


def _pick_reference_index(items):
    """选出基准图（决定输出尺寸与比例）

    评分：``score = 面积 × 方正度 = (w·h) × (短边/长边) = 短边²``
    ⇒ 等价于「短边最长」；并列时取更方正（长边更小），仍并列取更早出现的。
    这样"又大又方"的图胜出：正常情况＝面积最大的那张说了算，混入超宽/超高长条图时不会被它撑爆画布。
    """
    best_key, best_index = None, 0
    for i, img in enumerate(items):
        h, w = int(img.shape[0]), int(img.shape[1])
        short, long_ = (w, h) if w <= h else (h, w)
        key = (short, -long_)
        if best_key is None or key > best_key:
            best_key, best_index = key, i
    return best_index


# ==============================
# 节点 1：图像列表分路输出
# ==============================

class ImageListSplitOutputs:
    """图像列表分路输出节点 — 把图像列表按指定路数拆成多个独立输出

    功能：
    - 输入一个图像列表（可选），从 ``start_index`` 开始，依次把列表元素输出到第 1..N 路
    - 输出路数（``output_count``）可设置 1~16，**界面上显示的路数是前端动态增删的**
      （后端声明 16 路作为上限，真正显示几路由 ``js/image_list_nodes.js`` 按 widget 值裁剪）
    - 从 ``start_index`` 到列表结尾不足 ``output_count`` 时，多出来的那几路为空（不输出内容）
    - 输入未连接时：不报错，也不输出任何内容

    空输出返回 ``None``：下游效果等同于「这一路没接线」——下游把该输入声明为 optional 时按
    "没有这一路"处理（透传/忽略），声明为 required 时会在该下游报错（等价于未连接）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": MAX_OUTPUTS,
                    "step": 1,
                    "tooltip": f"输出路数（1~{MAX_OUTPUTS}）。节点界面上的输出口会随该值即时增减。"
                               f"从 start_index 到列表结尾不足该数量时，多出来的路不输出内容。"
                               f"Number of output channels (1-{MAX_OUTPUTS}). The node's output slots are "
                               f"added/removed live with this value. Channels beyond the remaining items are empty."
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFF_FFFF,
                    "step": 1,
                    "tooltip": "从列表第几个元素开始输出（0 = 第 1 个，1 = 第 2 个，以此类推）。"
                               "List index to start outputting from (0 = first item)."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "图像列表输入（列表语义：元素尺寸可以互不相同；未连接时不报错也不输出）。"
                               "普通 IMAGE 批次（同一尺寸的多张图）也可以接，会自动拆成单张元素。"
                               "Image list input (list semantics: elements may have different sizes; "
                               "if not connected, nothing is output and no error is raised). "
                               "A plain IMAGE batch is also accepted and will be split into single-frame items."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",) * MAX_OUTPUTS
    RETURN_NAMES = tuple(f"image_{i + 1}" for i in range(MAX_OUTPUTS))
    FUNCTION = "split_outputs"
    CATEGORY = "image"
    OUTPUT_NODE = False
    INPUT_IS_LIST = True
    DESCRIPTION = (f"把图像列表按指定路数（1~{MAX_OUTPUTS}）拆成多个独立输出；"
                   "从 start_index 开始依次输出，不足的路为空（返回 None，效果等同于该路未接线），"
                   "未连接输入时不报错。")

    def split_outputs(self, output_count=2, start_index=0, images=None):
        count = max(1, min(MAX_OUTPUTS, _scalar(output_count, 2)))
        start = max(0, _scalar(start_index, 0))

        items = normalize_image_list(images)
        available = max(0, len(items) - start)

        if len(items) > 0 and available == 0:
            logging.warning(f"[ImageListSplitOutputs] Warning: start_index={start} >= 列表长度={len(items)}，"
                  f"所有输出为空 / no valid range, all outputs are empty.")

        outputs = []
        for i in range(MAX_OUTPUTS):
            if i < count and i < available:
                outputs.append(items[start + i])
            else:
                # 未连接输入 / 超出剩余元素 / 超出设置的路数 -> 空输出
                # （返回 None：下游效果等同于「这一路没接线」）
                outputs.append(_empty_output())
        return tuple(outputs)


# ==============================
# 节点 2：图像列表位置替换
# ==============================

class ImageListReplaceAt:
    """图像列表位置替换节点 — 用替换列表覆盖目标列表中从指定位置开始的一段

    功能：
    - ``images``：要被替换的图像列表（可选）
    - ``replace_images``：用来替换的图像列表（可选）
    - 从 ``start_index`` 开始，按替换列表的长度覆盖对应位置
      （第 ``start_index + i`` 个元素 = 替换列表第 i 个元素）
    - **绝不超出原有列表长度**：替换列表比剩余空间长时，多出来的替换图直接丢弃
    - 逐元素整体替换，**不做任何缩放/补齐/扩展**，列表长度不变（元素尺寸允许互不相同）

    与既有 ``BatchImageReplace`` 的区别：无 overflow/underflow 模式、不缩放、不改变列表长度，
    并且输出为列表（``OUTPUT_IS_LIST``，下游节点按元素逐个执行）。
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
                    "tooltip": "从列表第几个元素开始替换（0 = 第 1 个，1 = 第 2 个，以此类推）。"
                               "List index to start replacing at (0 = first item)."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "要被替换的图像列表（列表语义：元素尺寸可不同）。"
                               "未连接时不报错也不输出。Target image list "
                               "(list semantics: elements may have different sizes). "
                               "If not connected, nothing is output and no error is raised."
                }),
                "replace_images": ("IMAGE", {
                    "tooltip": "用来替换的图像列表，按顺序填入 start_index 之后的对应位置；"
                               "比剩余空间多出来的部分会被丢弃（不扩展列表）。未连接时原样透传目标列表。"
                               "Replacement source list; items are written into the positions after "
                               "start_index. Extra items are dropped (the list is never extended). "
                               "If not connected, the target list is passed through unchanged."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "replace_at"
    CATEGORY = "image"
    OUTPUT_NODE = False
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = ("用替换列表覆盖目标图像列表中从 start_index 开始的一段（长度由替换列表决定，"
                   "绝不超过原列表长度）；逐元素整体替换、不缩放、长度不变，输出为列表。")

    def replace_at(self, start_index=0, images=None, replace_images=None):
        start = max(0, _scalar(start_index, 0))

        target = normalize_image_list(images)
        if len(target) == 0:
            # 未连接 / 空列表：不报错，也不输出内容（None ⇒ 下游效果等同于「这一路没接线」）
            return _empty_list_output()

        source = normalize_image_list(replace_images)
        if len(source) == 0:
            # 没有替换图：原样透传
            return (target,)

        remaining = len(target) - start
        if remaining <= 0:
            logging.warning(f"[ImageListReplaceAt] Warning: start_index={start} >= 列表长度={len(target)}，"
                  f"原样返回 / no valid range, returning original list.")
            return (target,)

        count = min(len(source), remaining)
        result = list(target)
        for i in range(count):
            result[start + i] = source[i]
        return (result,)


# ==============================
# 节点 3：图像列表转批次（RGBA 透明补边）
# ==============================

class ImageListToBatch:
    """图像列表 → 批次（单轴对称补边，不裁剪、不拉伸）

    问题：普通批次要求所有图同尺寸；用缩放统一尺寸又会拉伸/裁剪，丢内容。

    做法（单轴对称补边）：
    1. **基准图** = 列表里「面积 × 方正度」最大（= 短边最长；并列取更方正）的那张。
       它的比例就是输出比例，**它的原始尺寸就是输出尺寸**（因此它是唯一零重采样的）。
    2. 其余每张：比基准更宽 → **只补上下**；更高 → **只补左右**（各补一半，绝不同时四面补）。
    3. 等比缩放到基准尺寸后堆成批次。

    实现上先按"最终内容尺寸"缩放、再居中贴进画布（等价于"先补边再整体缩放"，
    但不会让补边被 LANCZOS 采样出内容边缘暗边），尺寸已匹配的图完全跳过重采样。

    ``output_mode``：
    - ``RGB``（**默认**，3 通道，补边纯黑）：**最兼容**——VL/视觉编码器、VAE、保存/预览都能吃。
    - ``RGBA``（4 通道，补边 alpha=0）：真透明，适合后续合成；但**只吃 3 通道的下游会出错**——
      例：Qwen VL 视觉塔 ``Qwen35VisionPatchEmbed.forward`` 固定 ``x.view(-1, 3, 2, 16, 16)``
      （``in_channels=3``），收到 4 通道会让 patch 数多 1/3，与 grid 位置编码数不匹配
      （``RuntimeError: tensor a (4608) vs tensor b (3456)``）。要用 RGBA 又必须喂 3 通道节点时，
      中间先接核心节点 ``Split Image with Alpha``（IMAGE 输出即 ``i[:,:,:3]``）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_mode": (["RGB", "RGBA"], {
                    "default": "RGB",
                    "tooltip": "输出通道。RGB=3 通道（补边纯黑，**最兼容**：VL/视觉编码器、VAE、保存/预览都能吃）；"
                               "RGBA=4 通道（补边 alpha=0，真透明，适合合成；但只吃 3 通道的下游——如 Qwen VL "
                               "视觉编码器——会报 token 数不匹配，需改用 RGB 或中间接 Split Image with Alpha）。"
                               "Output channels: RGB (3ch, black padding, works everywhere) or RGBA (4ch, "
                               "alpha=0 padding, compositing only; 3-channel-only consumers such as Qwen VL "
                               "vision encoders will fail unless alpha is stripped first)."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "图像列表输入（元素尺寸可互不相同；普通 IMAGE 批次也可以接，会先拆成单张）。"
                               "输出为统一尺寸的批次，补边只出现在某一个轴上。未连接时不报错、不输出。"
                               "Image list input (items may have different sizes; a plain IMAGE batch is "
                               "accepted too and is split first). Output is a uniform-size batch padded on a "
                               "single axis. When not connected: no error, no output."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "to_batch"
    CATEGORY = "image"
    OUTPUT_NODE = False
    INPUT_IS_LIST = True
    DESCRIPTION = ("把尺寸不同的图像列表转成同尺寸批次：以「短边最长（面积×方正度最大）」的那张为基准，"
                   "其余图只在一个轴上对称补边后等比缩放，绝不裁剪/拉伸。"
                   "output_mode=RGB（默认，3 通道黑边，最兼容）/ RGBA（4 通道 alpha=0 透明补边，适合合成）。")

    def to_batch(self, output_mode="RGB", images=None):
        items = to_single_images(images)
        if len(items) == 0:
            # 未连接 / 空列表：不报错，也不输出
            return (None,)

        mode = str(_scalar(output_mode, "RGB", cast=str)).upper()
        use_alpha = mode == "RGBA"
        channels = 4 if use_alpha else 3
        convert = _to_rgba if use_alpha else _to_rgb

        ref = _pick_reference_index(items)
        Rw, Rh = int(items[ref].shape[1]), int(items[ref].shape[0])
        divisor = math.gcd(Rw, Rh) or 1
        aligned = (Rw % 8 == 0 and Rh % 8 == 0)

        if len(items) <= 12:
            logging.debug("[ImageListToBatch] 基准候选: " + " | ".join(
                f"#{i} {int(it.shape[1])}×{int(it.shape[0])} min={min(int(it.shape[1]), int(it.shape[0]))}"
                for i, it in enumerate(items)) + f"  -> 选定 #{ref}")
        else:
            logging.info(f"[ImageListToBatch] 基准候选: 共 {len(items)} 张（略），"
                  f"选定 #{ref} {Rw}×{Rh} min={min(Rw, Rh)}")
        logging.info(f"[ImageListToBatch] 输出尺寸 = {Rw}×{Rh}"
              f"（基准 #{ref}，比例 {Rw // divisor}:{Rh // divisor}），共 {len(items)} 张，"
              f"通道 = {'RGBA(4)' if use_alpha else 'RGB(3)'}")
        if use_alpha:
            logging.info("[ImageListToBatch] 提示：RGBA(4 通道) 只适合支持 alpha 的下游（合成 / 存 PNG）。"
                  "若下游只吃 3 通道（如 Qwen VL 视觉编码器），请改用 RGB，"
                  "或在中间接 Split Image with Alpha（IMAGE 输出即 3 通道），否则会报 token 数不匹配。")
        if not aligned:
            logging.warning(f"[ImageListToBatch] 提示：输出尺寸 {Rw}×{Rh} 不是 8 的倍数，"
                  "若下游要过 VAE 编码（下采样 8/16），请改用 8/16 倍数尺寸或先做对齐。")

        batch = torch.zeros((len(items), Rh, Rw, channels), dtype=torch.float32)
        for i, img in enumerate(items):
            h, w = int(img.shape[0]), int(img.shape[1])
            if w * Rh > h * Rw:          # 比基准更宽 → 只补上下
                new_w, new_h = Rw, max(1, round(h * Rw / w))
            elif w * Rh < h * Rw:        # 比基准更高 → 只补左右
                new_w, new_h = max(1, round(w * Rh / h)), Rh
            else:                        # 比例相同 → 不补边
                new_w, new_h = Rw, Rh

            if (new_w, new_h) == (w, h):
                content = convert(img)                            # 尺寸已匹配 → 零重采样
            else:
                content = convert(_resize_image(img, new_w, new_h))

            ox, oy = (Rw - new_w) // 2, (Rh - new_h) // 2
            batch[i, oy:oy + new_h, ox:ox + new_w, :] = content
        return (batch,)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "ImageListSplitOutputs": ImageListSplitOutputs,
    "ImageListReplaceAt": ImageListReplaceAt,
    "ImageListToBatch": ImageListToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageListSplitOutputs": "图像列表分路输出 (Image List Split)",
    "ImageListReplaceAt": "图像列表位置替换 (Image List Replace)",
    "ImageListToBatch": "图像列表转批次 (Image List To Batch)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
