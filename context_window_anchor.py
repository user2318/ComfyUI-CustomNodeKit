# ==============================
# Context Window Anchor —— 上下文窗口「统一基准锚定（拼接式） + 双侧延伸」（实验性）
#
# 一句话：把「你自己提供的几个基准帧」和「窗口前后的真实相邻帧」作为**额外帧**拼进每个窗口参与采样，
#         采样后裁掉；它们只做上下文/基准，不参与窗口融合。
#
# 生成的切片结构（x 坐标系，P = 参考前缀帧数；halo 关闭时中间两段为空）：
#   [参考前缀] [锚定帧(已知基准)] [前延伸(内部·默认关)] [本窗口帧(参与融合)] [后延伸(内部·默认关)]
#
# 节点接口（极简）：只需要三根可选线
#   - anchor_frames（IMAGE）：要当基准的那几帧，**给多少用多少**（没有数量参数）；
#   - vae（VAE）：与 conditioning 节点同一个 Wan VAE；
#   - anchor_mask_frames（IMAGE，仅 SCAIL-2）：与 anchor_frames 同帧号的驱动遮罩。
# 这些帧会被当作「已知帧（干净历史）」拼在每个窗口最前面：驱动流 + 驱动遮罩 + 待生成序列，
# 采样后裁掉、不参与融合；不接 anchor_frames 时完全等价于不装本节点。
# （内部的 apply_window_extras 仍保留"双侧延伸"能力，但暂不暴露，默认 0=关闭。）
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


# 「驱动/pose latent」的键名：原始键 + 模型侧键（animate2 与 SCAIL-2 都是这两个）
ANCHOR_POSE_KEYS = ("pose_video_latent", "pose_latents")
# 「已知/未知」标记类流的填法（锚帧当作**普通帧**——与「手动拼接一段开头」的分布一致）：
#   concat_mask / denoise_mask（输入掩码，1 = 去噪/待生成）        → 锚帧填 1
#   c_concat（SCAIL-2 的 4 通道 history mask，1 = 干净(已知)锚帧）→ 锚帧填 0
ANCHOR_ONE_KEYS = ("concat_mask", "denoise_mask")
ANCHOR_ZERO_KEYS = ("c_concat",)
# 掩码型通道（按通道数识别的兜底：未知的掩码类流按「普通帧」填 0 并只告警一次）
MASK_LIKE_CHANNELS = (1, 3, 4)


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
_LOGGED_ONCE = set()


def _log_once(tag, msg, *args):
    """同一条消息只打一次（避免每个采样步 × 每个窗口刷屏）。"""
    if tag in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(tag)
    logging.info(msg, *args)


def reset_pools(frames, mask_frames, vae):
    """设置锚定素材并清空编码缓存；返回锚定 latent 帧数。"""
    global _ANCHOR_COUNT
    _POOL_CACHE.clear()
    _WARNED.clear()
    _LOGGED_ONCE.clear()
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


def _zero_like_frames(tensor, dim: int, count: int, value: float = 0.0):
    """取 tensor 的前 count 帧并整体置为 value（不够则重复末帧），用于「已知/未知」标记类流。"""
    if tensor is None or count <= 0:
        return None
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(0, min(count, tensor.shape[dim]))
    z = tensor[tuple(idx)].clone()
    if value == 0.0:
        z.zero_()
    else:
        z.fill_(float(value))
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

def inject_anchor_x(model_options, x_slice, dim: int = 2, sigma: float = 0.0):
    """把锚帧 latent（按当前 σ 加噪、当作普通帧）插到窗口 x 切片的「参考前缀之后」。

    σ 取自采样循环当前的 timestep：x_t = (1-σ)·锚帧 + σ·噪声 —— 这样模型看到的就是
    「剪辑开头多了几帧普通帧」，而不是「噪声帧里夹着干净帧」的分布外输入。
    """
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
    sigma = float(sigma)
    anchor = pool
    if sigma > 0.0:
        step = int(getattr(window, "_anchor_x_applied_step", 0) or 0)
        gen = torch.Generator(device="cpu").manual_seed(0x5EED + 7919 * (step + 1))
        noise = torch.randn(pool.shape, generator=gen, device="cpu", dtype=torch.float32)
        noise = noise.to(device=pool.device, dtype=pool.dtype)
        anchor = pool * (1.0 - sigma) + noise * sigma
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    new_x = _insert_at(x_slice, anchor, dim, prefix_len)
    if new_x is not x_slice:
        window._anchor_x_applied = True
        window.anchor_latent_count = int(pool.shape[dim])
        window.anchor_latent = pool
        _log_once(("x", round(sigma, 3)),
                  "[ContextAnchor] 待生成序列前已插入锚定 %d 帧（位置=%d, 按 σ=%.3f 加噪当普通帧）：%s → %s",
                  pool.shape[dim], prefix_len, sigma, list(x_slice.shape), list(new_x.shape))
    return new_x


# ---------------- 注入：条件流 ----------------


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

    长度判据必须是「窗口级」长度（slice_len / frames_len），不能用全局 chunk 长度 ——
    否则窗口内部的逐帧流会被静默漏插，随后在 comfy 的 c_concat 通道拼接处以形状不匹配报错。
    """
    if stats is None:
        stats = {"extended": [], "skipped": []}
    if tensor is None or _ANCHOR_COUNT <= 0:
        return tensor
    t_dim = _temporal_dim(tensor)
    cur = int(tensor.shape[t_dim])
    ch = int(tensor.shape[1]) if tensor.ndim > 1 else 1

    def _pool(kind):
        return _get_pool(kind, tensor.shape[-2], tensor.shape[-1], target_channels=ch,
                         device=tensor.device, dtype=tensor.dtype)

    # 1) 驱动/pose latent 流：一定前插（窗口帧坐标系，不含参考前缀）
    if key in ANCHOR_POSE_KEYS:
        pool = _pool("pose")
        if pool is None:
            stats["skipped"].append(key)
            return tensor
        window.anchor_pose_latent = pool
        new_t = _insert_at(tensor, pool, t_dim, 0)
        if new_t is not tensor:
            stats["extended"].append(f"{key}(驱动池)")
            _log_once(("pose", key), "[ContextAnchor] 已把锚定 %d 帧插入驱动流 '%s'：%s → %s",
                      _ANCHOR_COUNT, key, list(tensor.shape), list(new_t.shape))
        return new_t

    if cur != slice_len and cur != frames_len:
        return tensor
    pos = prefix_len if cur == slice_len else 0

    # 2) SCAIL-2 的 28 通道掩码由模型 patch 前插（位置更精细：ref 掩码要插在 N 帧参考之后），此处不动
    if ch == 28:
        logging.debug("[ContextAnchor] 条件流 '%s'(28ch) 由模型 patch 处理，锚定注入跳过", key)
        return tensor

    # 3) 「已知/未知」标记类流：输入掩码填 1（= 待生成/普通帧），history mask 填 0
    if key in ANCHOR_ONE_KEYS:
        extra, what = _zero_like_frames(tensor, t_dim, _ANCHOR_COUNT, value=1.0), "填1(待生成)"
    elif key in ANCHOR_ZERO_KEYS or ch in MASK_LIKE_CHANNELS:
        extra, what = _zero_like_frames(tensor, t_dim, _ANCHOR_COUNT, value=0.0), "填0(普通帧)"
    # 4) 20 通道 = 16 latent + 4 ones（SCAIL 的驱动格式）→ 用驱动池
    elif ch == 20:
        extra, what = _pool("pose"), "驱动池"
    # 5) 其余（16ch 等）→ 用 x 池
    else:
        extra, what = _pool("x"), "x池"

    if extra is None:
        stats["skipped"].append(f"{key}({ch}ch,长{cur})")
        return tensor
    new_t = _insert_at(tensor, extra, t_dim, pos)
    if new_t is not tensor:
        stats["extended"].append(f"{key}({what})")
        _log_once(("stream", key, what),
                  "[ContextAnchor] 已把锚定 %d 帧插入条件流 '%s'（%dch, 长 %d → %d，%s）",
                  _ANCHOR_COUNT, key, ch, cur, int(new_t.shape[t_dim]), what)
    return new_t


def inject_anchor_conds(window, resized_cond, x_in=None, dim: int = 2):
    """把锚帧作为额外帧前插到窗口的条件流（驱动流 / 逐帧流 / 标记流），就地更新条目。

    长度判据必须用「窗口级」长度（slice_len / frames_len），不能用全局 chunk 长度。
    """
    if _POOL_SOURCE["frames"] is None or resized_cond is None:
        return resized_cond
    slice_len, frames_len = _window_lengths(window, dim)
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    stats = {"extended": [], "skipped": []}

    def _act(k, t):
        return _prepend_one_cond(k, t, window, slice_len, frames_len, prefix_len, stats)

    for actual in resized_cond:
        if not isinstance(actual, dict):
            continue
        for key, value in list(actual.items()):
            if isinstance(value, dict):
                for sub_key, sub_value in list(value.items()):
                    t = _cond_tensor(sub_value)
                    if t is None:
                        continue
                    new_t = _act(sub_key, t)
                    if new_t is not t:
                        value[sub_key] = _cond_replace(sub_value, new_t)
            else:
                t = _cond_tensor(value)
                if t is None:
                    continue
                new_t = _act(key, t)
                if new_t is not t:
                    actual[key] = _cond_replace(value, new_t)
    if stats["extended"]:
        _log_once(("summary",), "[ContextAnchor] 锚定生效：本窗口已插入的流 = %s",
                  ", ".join(stats["extended"]))
    for name in stats["skipped"]:
        tag = ("skip", name)
        if tag not in _WARNED:
            _WARNED.add(tag)
            logging.warning("[ContextAnchor] 条件流 '%s' 长度等于窗口长度但未插入锚定帧："
                            "若它参与 comfy 内部的通道拼接会以形状不匹配报错，请检查。", name)
    return resized_cond


def prepend_anchor_mask(window, tensor, t_dim: int, insert_at=None):
    """供 wan_scail_context.py 的模型 patch 调用：把锚帧遮罩插入「已切片的」掩码张量。

    t_dim=1 → 原始 driving_mask_28ch 布局 (B,T,28,H,W)；t_dim=2 → sam_latents / ref_mask_latents
    布局 (B,28,T,H,W)（ref 掩码用 insert_at=N 插在 N 帧参考之后）。
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
        _log_once(("mask_prepend", int(t_dim), pos),
                  "[ContextAnchor] 掩码流已插入锚定 %d 帧（dim=%d, pos=%d）：%s → %s",
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
    """把「锚定注入」挂到 handler 实例上（实例属性遮蔽原方法）。重复调用只更新配置。"""
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
            logging.exception("[ContextAnchor] 窗口改写失败，本步回退为原始窗口")
        return windows

    def get_resized_cond(cond_in, x_in, window, device=None):
        resized = orig_get_resized_cond(cond_in, x_in, window, device)
        if cfg.anchor_frames is None:
            return resized
        try:
            return inject_anchor_conds(window, resized, x_in, dim=dim)
        except Exception:
            logging.exception("[ContextAnchor] 锚定帧处理失败，本次跳过")
            return resized

    def execute(calc_cond_batch, model, conds, x_in, timestep, model_options):
        if cfg.anchor_frames is None:
            return orig_execute(calc_cond_batch, model, conds, x_in, timestep, model_options)
        try:
            sigma = float(timestep.reshape(-1)[0])
        except Exception:
            sigma = 0.0

        def calc_cond_batch_wrapped(m, c, sub_x, ts, mo, **kw):
            return calc_cond_batch(m, c, inject_anchor_x(mo, sub_x, dim=dim, sigma=sigma), ts, mo, **kw)

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
    """上下文窗口「统一基准锚定」节点（实验性）。

    只需要三根可选线：anchor_frames（要当基准的那几帧）、vae（同一个 Wan VAE）、
    SCAIL-2 再加 anchor_mask_frames（同帧号的驱动遮罩）。其余行为内部定死，无需任何调参。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（`Custom Context Windows (Manual)` 或 `Wan SCAIL-2 Context Windows` 的 model 输出）。Model that already has a context windows node applied."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关；关闭时模型直接透传。Master switch."}),
            },
            "optional": {
                "anchor_frames": ("IMAGE", {"tooltip": "【基准帧】驱动/pose 视频里你想当统一基准的那几帧（例如整段视频开头几帧）。**给多少就用多少**（没有数量参数）：会 VAE 编码成 latent，作为「已知帧」拼在每个窗口最前面（驱动流 + 驱动遮罩 + 待生成序列），采样后裁掉、不参与融合。不接=关闭本节点。\nAnchor basis frames (driving/pose video): how many you feed is how many are used. Leave unconnected to disable."}),
                "anchor_mask_frames": ("IMAGE", {"tooltip": "【仅 SCAIL-2】与 anchor_frames 同帧号的彩色遮罩帧（内部转成 28 通道，不需要 VAE）。建议一起接，否则驱动遮罩与驱动视频的时间基准可能不一致。\nSCAIL-2 only: colored mask frames matching anchor_frames (no VAE needed)."}),
                "vae": ("VAE", {"tooltip": "【接了 anchor_frames 时必接】与 conditioning 节点**同一个** Wan VAE（用于把基准帧编码成 latent；接错 VAE 会得到无意义的条件）。\nSame Wan VAE as the conditioning node; required when anchor_frames is connected."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_anchor"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "上下文窗口「统一基准锚定」节点（⚠️ 实验性：有效性与副作用需自行开/关对比验证）。\n"
        "给每个窗口的**驱动视频（含驱动遮罩）开头接一段全局锚定帧**，并把同样的帧插到待生成序列最前面"
        "（按当前 σ 加噪、当普通帧），采样后连同预测一起裁掉——等价于手动「把整段视频开头拼到片段开头再生成」的做法。\n"
        "只需三根线：anchor_frames（给多少用多少）、vae（同一个 Wan VAE）、SCAIL-2 再加 anchor_mask_frames。\n"
        "不接 anchor_frames ⇒ 完全等价于不装本节点。必须接在上下文窗口节点之后。\n"
        "Global-basis anchoring for context windows (experimental)."
    )

    def apply_anchor(self, model, enabled=True,
                     anchor_frames=None, anchor_mask_frames=None, vae=None):
        frames = _as_image_batch(anchor_frames)
        mask_frames = _as_image_batch(anchor_mask_frames)

        if not enabled or frames is None:
            reset_pools(None, None, None)
            logging.info("[ContextAnchor] 未启用（enabled=%s, anchor=%s），模型直接透传",
                         enabled, "未接" if frames is None else f"{frames.shape[0]} 像素帧")
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[ContextAnchor] 模型上没有 context_handler：请把本节点接在 "
                "WAN Context Windows (Manual Custom) / Wan SCAIL Context Windows 之后。")
        if vae is None:
            raise ValueError(
                "[ContextAnchor] 接了 anchor_frames 但没有接 vae：基准帧必须用（与 conditioning 相同的）"
                "Wan VAE 编码成 latent。")
        if mask_frames is not None and mask_frames.shape[0] != frames.shape[0]:
            logging.warning("[ContextAnchor] anchor_mask_frames 帧数(%d) 与 anchor_frames 帧数(%d) 不一致，"
                            "遮罩会按自己的帧数编码", mask_frames.shape[0], frames.shape[0])

        cfg = AnchorConfig(halo_head=0, halo_tail=0, anchor_frames=frames,
                           anchor_mask_frames=mask_frames, vae=vae, log_debug=False)
        k = reset_pools(frames, mask_frames, vae)

        model = model.clone()
        new_handler = copy.copy(handler)
        _install(new_handler, cfg)
        model.model_options["context_handler"] = new_handler

        logging.info("[ContextAnchor] 已启用：%d 像素帧 → %d latent 帧；给每个窗口的驱动视频（含遮罩）"
                     "开头接上这段锚帧，并把它按当前 σ 加噪后插到待生成序列最前，采样后裁掉", frames.shape[0], k)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanContextWindowAnchorNode": WanContextWindowAnchorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowAnchorNode": "Context Window Anchor + Halo (窗口锚定/双侧延伸)",
}



