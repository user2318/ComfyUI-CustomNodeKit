# ==============================
# Context Window Anchor —— 上下文窗口「统一基准锚定（拼接式） + 双侧延伸」（实验性）
#
# 一句话：把「你自己提供的几个基准帧」和「窗口前后的真实相邻帧」作为**额外帧**拼进每个窗口参与采样，
#         采样后裁掉；它们只做上下文/基准，不参与窗口融合。
#
# 生成的切片结构（x 坐标系，P = 参考前缀帧数）：
#   [参考前缀] [锚定帧(你提供的基准)] [前延伸帧(紧邻)] [本窗口帧(参与融合)] [后延伸帧]
#
# 两个可独立开关的机制（参数为 0 / 不接图即关闭，完全等价于不装本节点）：
#   1) 锚定（拼接式）：输入 `anchor_frames`（驱动/pose 视频的若干像素帧，VAE 编码；SCAIL-2 另可接
#      `anchor_mask_frames` 提供同帧号的 SAM3 彩色遮罩）。**提供多少就用多少**，没有数量参数。
#      它们拼在参考前缀之后、窗口帧之前；首窗同样拼接（基准是外部素材，不依赖窗口 0）。
#   2) 双侧延伸（halo）：把窗口前后各 N 个 latent 帧的真实相邻帧纳入窗口视野但不参与融合，
#      让每个「被拥有」的帧都处在窗口内部，而不是永远落在两个窗口的边缘。
#
# 为什么需要（背景）：animate2 / SCAIL-2 无骨骼等显式空间锚，且窗口化后 RoPE 每窗恒为 0..L-1
#   （不携带绝对时间）；Wan-Animate-2 论文 §3.3/§3.4 证实「驱动帧 ↔ 生成帧」是**逐帧刚性对齐**
#   （frame-wise token concat + 每帧只 attend 时间上对应的参考 token），并没有任何机制保证
#   两个窗口建立出**同一个**「参考图↔驱动」映射。于是每个窗口都会从自己的开头重新推断基准：
#   起点不同 ⇒ 同一全局帧结果不同（重叠区重影 / 动作对不上）；起点难 ⇒ 整段结构崩。
#   实测：只把统一开头拼到两段头部，两段在同一全局帧上就大幅趋同（仅剩细节/空间位置残差）。
#
# 注入点（全部逐窗、绝不污染全局 x）：
#   - x 侧：包装 execute 里的 calc_cond_batch，把锚帧 latent 插在**参考前缀之后**
#     （animate2 的 slice[0] 是 reference slot，`x_input = x[:,:,1:]`，必须保持在最前）；
#   - 条件侧：包装 get_resized_cond，给 `pose_video_latent`（两模型同键名）前插对应锚帧；
#     逐帧流（长度 == x 长度的那些，如 animate2 的 concat_latent_image / concat_mask）按同一位置前插；
#   - SCAIL-2 掩码：在 wan_scail_context.py 的模型 patch 完成窗口切片之后插入
#     （driving_mask_28ch/sam_latents 前插、ref_mask_latents 插在 N 帧参考之后）；
#   - 融合前裁剪：Custom_context.py 的 combine_context_window_results 按
#     「前缀 + 锚定帧数 + 头段」/「尾段」裁掉，锚帧预测不进入融合结果。
#
# 缓存与预算：VAE 编码按 (类型, 目标 latent 空间尺寸) 缓存；锚定帧数 = ((T-1)//4)+1。
#   窗口总长（context_length + 锚定 + 头段 + 尾段）建议 ≤ ≈21 latent（模型训练长度）。
#
# 必须接在上下文窗口节点（Custom Context Windows (Manual) / Wan SCAIL Context Windows）之后。
# 实现方式：浅拷贝 handler 实例并包装 get_context_windows / get_resized_cond / execute
#           （不做 __class__ 置换 —— 3.13 下 abc.ABC 派生类会报 layout differs）。
# 注意：锚定素材/VAE 记在模块级（同进程内以最后一次 apply 为准），每个工作流接一个本节点即可。
# ==============================

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch


# 原始 conditioning 里承载「驱动/pose latent」的键名（animate2 与 SCAIL-2 相同）
ANCHOR_POSE_KEY = "pose_video_latent"
# 长度与 x 相同、需要一起前插的掩码类键（值语义 0=已知）：锚定帧填 0
ANCHOR_ZERO_KEYS = ("concat_mask",)


@dataclass
class AnchorConfig:
    halo_head: int = 0
    halo_tail: int = 0
    anchor_frames: object = None        # (T,H,W,C) 像素帧 或 None
    anchor_mask_frames: object = None   # (T,H,W,C) 彩色遮罩帧 或 None
    vae: object = None
    log_debug: bool = False


# ---------------- 模块级锚定池（按最后一次 apply_anchor 生效） ----------------
_POOL_CACHE: dict = {}
_POOL_SOURCE = {"frames": None, "mask_frames": None, "vae": None}
_ANCHOR_COUNT = 0
_WARNED = set()


def reset_pools(frames, mask_frames, vae):
    """设置锚定素材并清空编码缓存；返回锚定 latent 帧数。"""
    global _ANCHOR_COUNT
    _POOL_CACHE.clear()
    _WARNED.clear()
    _POOL_SOURCE.update(frames=frames, mask_frames=mask_frames, vae=vae)
    _ANCHOR_COUNT = latent_frames_of(frames.shape[0]) if frames is not None else 0
    return _ANCHOR_COUNT


def anchor_count() -> int:
    return int(_ANCHOR_COUNT)


# ---------------- 基础工具 ----------------

def _as_image_batch(value):
    """IMAGE 输入 → (T,H,W,C)；空/非法返回 None。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] < 1:
        return None
    return value


def latent_frames_of(pixel_frames: int) -> int:
    """像素帧数 → latent 帧数（Wan 的 4x 时间压缩：1+4n）。"""
    t = int(pixel_frames)
    return max(1, ((t - 1) // 4) + 1) if t > 0 else 0


def _resize_frames(frames, target_h, target_w):
    """(T,H,W,C) 缩放到目标像素尺寸（area/center，与仓库其它节点一致）。"""
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    import comfy.utils
    return comfy.utils.common_upscale(frames.movedim(-1, 1).float(), int(target_w), int(target_h),
                                      "area", "center").movedim(1, -1)


def _match_channels(pool, target_channels: int):
    """SCAIL-2 的 pose_latents 可能 = 16ch latent + 4ch ones mask，这里按目标通道补齐。"""
    if pool is None or pool.shape[1] == target_channels:
        return pool
    if pool.shape[1] + 4 == target_channels:
        return torch.cat([pool, torch.ones_like(pool[:, :4])], dim=1)
    return pool


def _fit_frames(tensor, dim: int, count: int):
    """把 tensor 的时间维调整为恰好 count 帧（多则截断，少则重复末帧）。"""
    if tensor is None or count <= 0:
        return tensor
    cur = tensor.shape[dim]
    if cur == count:
        return tensor
    if cur > count:
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(0, count)
        return tensor[tuple(idx)]
    last = [slice(None)] * tensor.ndim
    last[dim] = slice(cur - 1, cur)
    reps = [1] * tensor.ndim
    reps[dim] = count - cur
    return torch.cat([tensor, tensor[tuple(last)].repeat(*reps)], dim=dim)


def _temporal_dim(tensor, default: int = 2) -> int:
    if tensor.ndim == 5:
        return 2
    if tensor.ndim == 4:
        return 1
    return max(0, tensor.ndim - 3)


def _insert_at(tensor, extra, dim: int, pos: int):
    """把 extra 沿 dim 插到 tensor 的 pos 位置（形状不匹配则原样返回并告警）。"""
    if extra is None or tensor is None:
        return tensor
    if tensor.ndim != extra.ndim:
        return tensor
    for d in range(tensor.ndim):
        if d != dim and tensor.shape[d] != extra.shape[d]:
            logging.warning("[ContextAnchor] 锚定帧形状 %s 与目标 %s 不匹配（dim=%d），跳过该张量",
                            list(extra.shape), list(tensor.shape), dim)
            return tensor
    pos = max(0, min(int(pos), tensor.shape[dim]))
    extra = extra.to(device=tensor.device, dtype=tensor.dtype)
    if extra.shape[0] == 1 and tensor.shape[0] > 1:
        reps = [1] * extra.ndim
        reps[0] = tensor.shape[0]
        extra = extra.repeat(*reps)
    if pos <= 0:
        return torch.cat([extra, tensor], dim=dim)
    if pos >= tensor.shape[dim]:
        return torch.cat([tensor, extra], dim=dim)
    head = [slice(None)] * tensor.ndim
    head[dim] = slice(0, pos)
    tail = [slice(None)] * tensor.ndim
    tail[dim] = slice(pos, None)
    return torch.cat([tensor[tuple(head)], extra, tensor[tuple(tail)]], dim=dim)


def _zero_like_frames(tensor, dim: int, count: int):
    """取 tensor 的前 count 帧并置 0（不够则重复末帧），用于「已知/未知」类掩码流。"""
    if tensor is None or count <= 0:
        return None
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(0, min(count, tensor.shape[dim]))
    z = tensor[tuple(idx)].clone()
    z.zero_()
    if z.shape[dim] < count:
        reps = [1] * tensor.ndim
        reps[dim] = count - z.shape[dim]
        last = [slice(None)] * tensor.ndim
        last[dim] = slice(z.shape[dim] - 1, z.shape[dim])
        z = torch.cat([z, z[tuple(last)].repeat(*reps)], dim=dim)
    return z


def _cond_tensor(value):
    """取出 cond 值里的张量（支持 .cond 包装）；没有则 None。"""
    if isinstance(value, torch.Tensor):
        return value
    inner = getattr(value, "cond", None)
    return inner if isinstance(inner, torch.Tensor) else None


def _cond_replace(value, tensor):
    """把张量放回 cond 值（保持 CONDRegular 等包装）。"""
    if hasattr(value, "_copy_with"):
        return value._copy_with(tensor)
    return tensor


# ---------------- 锚定池：编码（带缓存） ----------------

def _encode_pool(kind: str, h: int, w: int):
    src = _POOL_SOURCE
    try:
        target_h, target_w = int(h) * 8, int(w) * 8
        if kind == "mask":
            if src["mask_frames"] is None:
                if "no_mask" not in _WARNED:
                    _WARNED.add("no_mask")
                    logging.warning("[ContextAnchor] 未接 anchor_mask_frames：驱动遮罩不锚定，"
                                    "姿态与遮罩的时间基准可能不一致（SCAIL-2 建议一起接）")
                return None
            from .wan_scail_multi_ref import _extract_mask_to_28ch
            frames = _resize_frames(src["mask_frames"], target_h, target_w)
            pool = _extract_mask_to_28ch(frames[:, :, :, :3])      # (1,T',28,h,w)，时间在 dim=1
            return _fit_frames(pool, 1, _ANCHOR_COUNT)
        vae = src["vae"]
        if vae is None:
            if "no_vae" not in _WARNED:
                _WARNED.add("no_vae")
                logging.warning("[ContextAnchor] 锚定帧需要 VAE，但未提供（锚定已跳过）")
            return None
        frames = _resize_frames(src["frames"], target_h, target_w)
        pool = vae.encode(frames[:, :, :, :3])
        return _fit_frames(pool, 2, _ANCHOR_COUNT)
    except Exception:
        logging.exception("[ContextAnchor] 锚定池编码失败（kind=%s, spatial=(%d,%d)）", kind, h, w)
        return None


def _get_pool(kind: str, spatial_h: int, spatial_w: int, target_channels=None, device=None, dtype=None):
    """按目标 latent 空间尺寸取锚定池（kind ∈ x/pose/mask），带缓存。"""
    if _POOL_SOURCE["frames"] is None:
        return None
    key = (kind, int(spatial_h), int(spatial_w))
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = _encode_pool(kind, int(spatial_h), int(spatial_w))
    pool = _POOL_CACHE[key]
    if pool is None:
        return None
    if target_channels is not None and pool.shape[1] != int(target_channels):
        pool = _match_channels(pool, int(target_channels))
    if device is not None or dtype is not None:
        pool = pool.to(device=device or pool.device, dtype=dtype or pool.dtype)
    return pool


# ---------------- 注入：x 切片（插在参考前缀之后） ----------------

def inject_anchor_x(model_options, x_slice, dim: int = 2):
    """把锚帧 latent 插到窗口 x 切片的「参考前缀之后」；返回新切片。"""
    if _POOL_SOURCE["frames"] is None or x_slice is None:
        return x_slice
    window = None
    try:
        window = ((model_options or {}).get("transformer_options") or {}).get("context_window", None)
    except Exception:
        window = None
    if window is None or getattr(window, "_anchor_x_applied", False):
        return x_slice
    pool = _get_pool("x", x_slice.shape[-2], x_slice.shape[-1],
                     target_channels=x_slice.shape[1] if x_slice.ndim > 1 else None,
                     device=x_slice.device, dtype=x_slice.dtype)
    if pool is None:
        return x_slice
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    new_x = _insert_at(x_slice, pool, dim, prefix_len)
    if new_x is not x_slice:
        window._anchor_x_applied = True
        window.anchor_latent_count = int(pool.shape[dim])
        window.anchor_latent = pool
        logging.info("[ContextAnchor] x 侧已插入锚定 %d 帧（位置=%d）：%s → %s",
                     pool.shape[dim], prefix_len, list(x_slice.shape), list(new_x.shape))
    return new_x


# ---------------- 注入：条件流 ----------------

# 掩码型通道（1/3/4ch）：锚定帧填 0。SCAIL-2 的 c_concat = 1 - mask（1=clean anchor），
# 所以锚帧的 mask 取 0 才能让模型把锚帧当作 clean anchor；同样的道理适用于 denoise_mask / concat_mask。
ANCHOR_ZERO_CHANNELS = (1, 3, 4)


def _window_lengths(window, dim: int = 2):
    """返回 (slice_len, frames_len)：含参考前缀的窗口切片长度 / 不含前缀的窗口帧长度。"""
    index_list = getattr(window, "index_list", None) or []
    try:
        frames_len = len(window.frame_indices())
    except Exception:
        frames_len = len(getattr(window, "original_indices", None) or index_list)
    return len(index_list), frames_len


def _prepend_one_cond(key, tensor, window, slice_len, frames_len, prefix_len, stats=None):
    """按键类型把锚帧插进某个条件张量；返回新张量（不改则原样返回）。

    长度判据必须是「窗口级」长度（slice_len / frames_len），**不能用全局 chunk 长度** ——
    否则窗口内部的逐帧流会被静默漏插，随后在 comfy 的 c_concat 通道拼接处以形状不匹配报错。
    """
    if stats is None:
        stats = {"extended": [], "skipped": []}
    if tensor is None or _ANCHOR_COUNT <= 0:
        return tensor
    t_dim = _temporal_dim(tensor)
    cur = int(tensor.shape[t_dim])
    ch = int(tensor.shape[1]) if tensor.ndim > 1 else 1

    # 1) pose / 驱动 latent 流：一定前插（窗口帧坐标系，不含参考前缀）
    if key == ANCHOR_POSE_KEY:
        pool = _get_pool("pose", tensor.shape[-2], tensor.shape[-1], target_channels=ch,
                         device=tensor.device, dtype=tensor.dtype)
        if pool is None:
            stats["skipped"].append(key)
            return tensor
        window.anchor_pose_latent = pool
        new_t = _insert_at(tensor, pool, t_dim, 0)
        if new_t is not tensor:
            stats["extended"].append(f"{key}(pose池)")
            logging.info("[ContextAnchor] pose 流已插入锚定 %d 帧：%s → %s",
                         _ANCHOR_COUNT, list(tensor.shape), list(new_t.shape))
        return new_t

    # 2) 与窗口切片 / 窗口帧等长的逐帧流：按通道决定填什么
    if cur == slice_len or cur == frames_len:
        pos = prefix_len if cur == slice_len else 0
        if key in ANCHOR_ZERO_KEYS or ch in ANCHOR_ZERO_CHANNELS:
            extra = _zero_like_frames(tensor, t_dim, _ANCHOR_COUNT)
            what = f"填0({ch}ch)"
        else:
            extra = _get_pool("x", tensor.shape[-2], tensor.shape[-1], target_channels=ch,
                              device=tensor.device, dtype=tensor.dtype)
            what = f"x池({ch}ch)"
        if extra is None:
            stats["skipped"].append(f"{key}({ch}ch,长{cur})")
            return tensor
        new_t = _insert_at(tensor, extra, t_dim, pos)
        if new_t is not tensor:
            stats["extended"].append(f"{key}({what},pos={pos})")
            logging.info("[ContextAnchor] 逐帧流 '%s'（%dch, 长 %d）已插入锚定 %d 帧：%s，位置=%d",
                         key, ch, cur, _ANCHOR_COUNT, what, pos)
        return new_t
    return tensor


def inject_anchor_conds(window, resized_cond, x_in=None, dim: int = 2):
    """把锚帧插进窗口的条件流（就地更新 resized_cond 中的条目）。"""
    if _POOL_SOURCE["frames"] is None or resized_cond is None:
        return resized_cond
    slice_len, frames_len = _window_lengths(window, dim)
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    stats = {"extended": [], "skipped": []}
    for actual in resized_cond:
        if not isinstance(actual, dict):
            continue
        for key, value in list(actual.items()):
            if isinstance(value, dict):
                for sub_key, sub_value in list(value.items()):
                    t = _cond_tensor(sub_value)
                    if t is None:
                        continue
                    new_t = _prepend_one_cond(sub_key, t, window, slice_len, frames_len, prefix_len, stats)
                    if new_t is not t:
                        value[sub_key] = _cond_replace(sub_value, new_t)
            else:
                t = _cond_tensor(value)
                if t is None:
                    continue
                new_t = _prepend_one_cond(key, t, window, slice_len, frames_len, prefix_len, stats)
                if new_t is not t:
                    actual[key] = _cond_replace(value, new_t)
    if stats["extended"] or stats["skipped"]:
        orig = getattr(window, "original_indices", None) or []
        logging.info("[ContextAnchor] 窗口帧 %s..%s（切片长 %d / 帧长 %d）：已插=%s；等长未插=%s",
                     orig[0] if orig else None, orig[-1] if orig else None,
                     slice_len, frames_len, stats["extended"] or "无", stats["skipped"] or "无")
        for name in stats["skipped"]:
            tag = ("skip", name)
            if tag not in _WARNED:
                _WARNED.add(tag)
                logging.warning("[ContextAnchor] 条件流 '%s' 长度等于窗口长度但未插入锚定帧："
                                "若它在 comfy 内部参与通道拼接会以形状不匹配报错，请检查。", name)
    return resized_cond


def prepend_anchor_mask(window, tensor, t_dim: int, insert_at=None):
    """供 wan_scail_context.py 的模型 patch 调用：把锚定遮罩帧插入「已切片的」掩码张量。

    t_dim=1 → 原始 driving_mask_28ch 布局 (B,T,28,H,W)：直接前插；
    t_dim=2 → sam_latents / ref_mask_latents 布局 (B,28,T,H,W)：permute 后插入（ref 掩码插在 N 帧参考之后）。
    """
    if tensor is None or window is None or _POOL_SOURCE["frames"] is None or _ANCHOR_COUNT <= 0:
        return tensor
    pool = _get_pool("mask", tensor.shape[-2], tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
    if pool is None:
        return tensor
    if int(t_dim) == 1:
        extra = _fit_frames(pool, 1, _ANCHOR_COUNT)
    else:
        extra = _fit_frames(pool.permute(0, 2, 1, 3, 4), 2, _ANCHOR_COUNT).contiguous()
    pos = 0 if insert_at is None else int(insert_at)
    new_t = _insert_at(tensor, extra, int(t_dim), pos)
    if new_t is not tensor:
        window.anchor_mask_latent = pool
        logging.info("[ContextAnchor] 掩码流已插入锚定 %d 帧（dim=%d, pos=%d）：%s → %s",
                     _ANCHOR_COUNT, int(t_dim), pos, list(tensor.shape), list(new_t.shape))
    return new_t


# ---------------- 双侧延伸（halo）：改写窗口 index_list ----------------

def _window_original(window) -> list:
    return list(getattr(window, "original_indices", None) or [])


def _window_prefix(window) -> list:
    """窗口的参考前缀帧索引（位于 index_list 最前、且不属于窗口帧）。"""
    original = set(_window_original(window))
    return [i for i in (getattr(window, "index_list", None) or []) if i not in original]


def apply_window_extras(handler, windows, cfg: AnchorConfig) -> None:
    """就地改写每个窗口的 index_list 与头/尾额外段（只做双侧延伸；锚定帧走注入路径）。"""
    if not windows:
        return
    halo_head = max(0, int(cfg.halo_head))
    halo_tail = max(0, int(cfg.halo_tail))
    if halo_head == 0 and halo_tail == 0:
        return
    ref_prefix_len = int(getattr(handler, "prefix_latent_len", 0) or 0)
    w0 = windows[0]
    x_len = int(getattr(w0, "total_frames", 0) or 0)
    if x_len <= 0:
        x_len = max([max(getattr(w, "index_list", None) or [0]) for w in windows] or [0]) + 1

    for window in windows:
        original = _window_original(window)
        if not original:
            continue
        prefix = _window_prefix(window)
        used = set(prefix) | set(original)
        lo, hi = min(original), max(original)

        head_halo = []
        if halo_head > 0:
            start = max(ref_prefix_len, lo - halo_head)
            for i in range(start, lo):
                if i not in used:
                    head_halo.append(i)
                    used.add(i)

        tail_halo = []
        if halo_tail > 0:
            for i in range(hi + 1, min(x_len, hi + 1 + halo_tail)):
                if i not in used:
                    tail_halo.append(i)
                    used.add(i)

        window.head_extra_indices = head_halo
        window.tail_extra_indices = tail_halo
        window.index_list = list(prefix) + head_halo + original + tail_halo
        window.context_length = len(window.index_list)

        if cfg.log_debug:
            logging.info(
                "[ContextAnchor] 窗口帧 %d..%d → 切片帧 %d..%d | 前缀=%d 前延伸=%d 后延伸=%d（原始长度=%d→%d）",
                lo, hi,
                window.index_list[0] if window.index_list else -1,
                window.index_list[-1] if window.index_list else -1,
                len(prefix), len(head_halo), len(tail_halo),
                len(original), len(window.index_list))


# ---------------- 安装 ----------------

def _install(handler, cfg: AnchorConfig):
    """把「锚定注入 + 双侧延伸」挂到 handler 实例上（实例属性遮蔽原方法）。重复调用只更新配置。"""
    if getattr(handler, "_cwa_anchor_installed", False):
        handler._anchor_cfg = cfg
        return handler

    dim = int(getattr(handler, "dim", 2) or 2)
    orig_get_context_windows = getattr(handler, "get_context_windows", None)
    orig_get_resized_cond = getattr(handler, "get_resized_cond", None)
    orig_execute = getattr(handler, "execute", None)

    def get_context_windows(model, x_in, model_options):
        windows = orig_get_context_windows(model, x_in, model_options)
        try:
            apply_window_extras(handler, windows, cfg)
        except Exception:
            logging.exception("[ContextAnchor] 双侧延伸失败，本步回退为原始窗口")
        return windows

    def get_resized_cond(cond_in, x_in, window, device=None):
        resized = orig_get_resized_cond(cond_in, x_in, window, device)
        if cfg.anchor_frames is None:
            return resized
        try:
            return inject_anchor_conds(window, resized, x_in, dim=dim)
        except Exception:
            logging.exception("[ContextAnchor] 锚定帧条件前插失败，本次跳过")
            return resized

    def execute(calc_cond_batch, model, conds, x_in, timestep, model_options):
        if cfg.anchor_frames is None:
            return orig_execute(calc_cond_batch, model, conds, x_in, timestep, model_options)

        def calc_cond_batch_wrapped(m, c, sub_x, ts, mo, **kw):
            return calc_cond_batch(m, c, inject_anchor_x(mo, sub_x, dim=dim), ts, mo, **kw)

        return orig_execute(calc_cond_batch_wrapped, model, conds, x_in, timestep, model_options)

    if orig_get_context_windows is not None:
        handler.get_context_windows = get_context_windows
    if orig_get_resized_cond is not None and orig_execute is not None:
        handler.get_resized_cond = get_resized_cond
        handler.execute = execute
    handler._cwa_anchor_installed = True
    handler._anchor_cfg = cfg
    return handler


# ---------------- 节点 ----------------

class WanContextWindowAnchorNode:
    """上下文窗口「统一基准锚定（拼接式）+ 双侧延伸」节点。

    两个机制都可独立关闭（halo=0 且不接锚定帧时，完全等价于不装本节点）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（`Custom Context Windows (Manual)` 或 `Wan SCAIL-2 Context Windows` 的 model 输出）。Model that already has a context windows node applied."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关；关闭时模型直接透传。Master switch."}),
                "halo_frames_head": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "【双侧延伸·前】把窗口之前的 N 个 latent 帧纳入窗口视野（只做上下文、不参与融合）；0=关闭。首窗自动为 0（前面是参考前缀区）。\nHead halo: preceding latent frames the window can see (context only, not fused). 0=off."}),
                "halo_frames_tail": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "【双侧延伸·后】把窗口之后的 N 个 latent 帧纳入窗口视野（只做上下文、不参与融合）；0=关闭。末窗自动截断。\nTail halo: following latent frames the window can see (context only, not fused). 0=off."}),
            },
            "optional": {
                "anchor_frames": ("IMAGE", {"tooltip": "【锚定基准帧】驱动/pose 视频里你想当统一基准的那几帧（例如整段视频开头几帧、或与参考图姿态最接近的一段）。**提供多少就用多少**（没有数量参数），会 VAE 编码后拼在每个窗口的参考前缀之后，采样后裁掉；不接=关闭锚定。\nAnchor basis frames (driving/pose video). How many you feed is how many are used; trimmed after sampling. Leave unconnected to disable."}),
                "anchor_mask_frames": ("IMAGE", {"tooltip": "【可选，仅 SCAIL-2】与 anchor_frames **同帧号**的 SAM3 彩色遮罩帧；接上才能让驱动遮罩与姿态保持同一时间基准（掩码不需要 VAE）。\nOptional SCAIL-2 mask frames matching anchor_frames (no VAE needed)."}),
                "vae": ("VAE", {"tooltip": "【接 anchor_frames 时必接】与 conditioning 节点同一个 Wan VAE，用于把锚定帧编码成 latent（SCAIL-2/近分辨率的 pose 分支会按各自 latent 尺寸自动缩放）。\nSame Wan VAE as the conditioning node; required when anchor_frames is connected."}),
                "log_debug": ("BOOLEAN", {"default": False,
                    "tooltip": "打印每个窗口的延伸/锚定明细。Log per-window halo/anchor details."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_anchor"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "窗口「统一基准锚定（拼接式）+ 双侧延伸」节点。⚠️ 实验性：有效性与副作用需自行开/关对比验证。\n"
        "1) 锚定：把你提供的基准帧（anchor_frames，可选配 SCAIL-2 遮罩）VAE 编码后作为**额外帧**拼在"
        "每个窗口的参考前缀之后参与采样，采样后裁掉（提供多少用多少）。\n"
        "2) 双侧延伸：把窗口前后各 N 帧真实相邻帧纳入视野但不参与融合，让每帧都处在窗口内部。\n"
        "切片结构：[参考前缀][锚定帧][前延伸帧][本窗口帧(参与融合)][后延伸帧]。\n"
        "两者都可关闭（halo=0 且不接锚定帧 ⇒ 完全等价于不装本节点）。必须接在上下文窗口节点之后。\n"
        "推荐：双侧延伸 + 上下文节点 overlap=0 → 每个全局帧只有一个拥有者，重影从机制上消失。\n"
        "Splice-style unified-basis anchoring + two-sided halo for context windows (experimental)."
    )

    def apply_anchor(self, model, enabled=True, halo_frames_head=0, halo_frames_tail=0,
                     anchor_frames=None, anchor_mask_frames=None, vae=None, log_debug=False):
        frames = _as_image_batch(anchor_frames)
        mask_frames = _as_image_batch(anchor_mask_frames)
        halo_head = max(0, int(halo_frames_head))
        halo_tail = max(0, int(halo_frames_tail))

        if not enabled or (frames is None and halo_head <= 0 and halo_tail <= 0):
            reset_pools(None, None, None)
            logging.info("[ContextAnchor] 未启用（enabled=%s, anchor=%s, halo=%d/%d），模型直接透传",
                         enabled, "未接" if frames is None else f"{frames.shape[0]} 像素帧",
                         halo_head, halo_tail)
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[ContextAnchor] 模型上没有 context_handler：请把本节点接在 "
                "WAN Context Windows (Manual Custom) / Wan SCAIL Context Windows 之后。")
        if frames is not None and vae is None:
            raise ValueError(
                "[ContextAnchor] 接了 anchor_frames 但没有接 vae：锚定帧必须用（与 conditioning 相同的）"
                "Wan VAE 编码成 latent。")
        if frames is not None and mask_frames is not None and mask_frames.shape[0] != frames.shape[0]:
            logging.warning("[ContextAnchor] anchor_mask_frames 帧数(%d) 与 anchor_frames 帧数(%d) 不一致，"
                            "遮罩会按自己的帧数编码", mask_frames.shape[0], frames.shape[0])

        cfg = AnchorConfig(halo_head=halo_head, halo_tail=halo_tail, anchor_frames=frames,
                           anchor_mask_frames=mask_frames, vae=vae, log_debug=bool(log_debug))
        k = reset_pools(frames, mask_frames, vae)

        model = model.clone()
        new_handler = copy.copy(handler)
        _install(new_handler, cfg)
        model.model_options["context_handler"] = new_handler

        ctx_len = int(getattr(handler, "context_length", 0) or 0)
        budget = ctx_len + k + halo_head + halo_tail
        logging.info(
            "[ContextAnchor] 已应用：锚定 %d 帧（%s）+ 双侧延伸 %d/%d latent 帧；窗口总长 = %d + %d + %d/%d = %d latent",
            k, "未接" if frames is None else f"输入 {frames.shape[0]} 像素帧", halo_head, halo_tail,
            ctx_len, k, halo_head, halo_tail, budget)
        if budget > 21:
            logging.warning("[ContextAnchor] 窗口总长 %d latent 超过 ≈21（81 像素帧，模型训练长度）："
                            "可能影响质量，建议减小锚定帧数或 halo。", budget)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanContextWindowAnchorNode": WanContextWindowAnchorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowAnchorNode": "Context Window Anchor + Halo (窗口锚定/双侧延伸)",
}



