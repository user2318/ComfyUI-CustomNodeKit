# ==============================
# Context Window Anchor —— 上下文窗口首帧锚定（实验性 / 未充分验证的测试性节点）
#
# 背景（要解决的问题）：
#   animate2 / SCAIL-2 的「参考图 ↔ 驱动视频」身体比例与空间位置对齐由模型内部逐窗口推导。
#   上下文窗口采样时，每个窗口的驱动视频 latent 只包含本窗口对应的帧，窗口首帧姿态各不相同，
#   于是每个窗口的对齐基准不同 → 窗口融合后出现比例/位置跳变（接缝错位）。
#
# 本节点思路：
#   把窗口 w>0 的驱动视频 latent 的前 K 个 latent 帧，替换为「当前片段窗口 0」的同名张量前 K 帧，
#   让所有窗口以同一个基准对齐（SCAIL-2 的驱动遮罩 sam_latents 同理一起锚定）。
#
# 生效位置：
#   采样期 calc_cond_batch 的 per-window conditioning 重切分之后（在 handler.get_resized_cond
#   的返回值上做后处理），属于 conditioning 层：
#     - 每个采样步都会生效（get_resized_cond 每步都会被调用）；
#     - 每个 chunk 的窗口 0 即该段首窗口，快照在窗口 0 处刷新 → 天然「不跨 chunk」；
#     - 不触碰 x/latent 层，因此与融合权重 / overlap / FreeNoise 无交互。
#
# 锚定的 cond 键（只处理 model_conds 内、模型实际消费的那套键）：
#   - animate2 : pose_latents (dim=2) ← 节点输入 pose_video
#   - SCAIL-2  : pose_latents (dim=2) ← 节点输入 pose_video
#                sam_latents  (dim=2) ← 节点输入 mask_video（驱动遮罩 driving_mask_28ch）
#   只改数值、不改长度，故 animate2 的 pose_latents.shape[2] == f_gen - 1 校验不受影响。
#
# 锚点位置（anchor_offset，默认 -1=自动）：
#   上下文节点会把前缀索引（chunk 的 0..P-1 帧）拼到每个窗口 index_list 的前面，而 pose/遮罩切片用的
#   正是 index_list → 每个窗口的切片头部都会多出 P-1 帧"前缀派生帧"，且它们对所有窗口完全相同
#   （多参考图时是零填充帧；单参考图时是 chunk 开头几帧的驱动内容）。锚在这些帧上等于空操作，
#   故自动偏移取 max(0, 前缀帧数 - 1)，即从"窗口自身的第一个驱动帧"开始锚定：
#     - 无前缀 / 前缀=1 ：偏移 0（与最初实现一致，本来就正确）
#     - 前缀=6（6 张参考图）：偏移 5
#
# 混合锚定（可选，接了才生效）：
#   默认锚定源是「当前片段窗口 0 的首帧」。若把 global_pose_frames（以及 SCAIL-2 的
#   global_mask_frames）接上，则会把"全局素材"编码成 latent，用来替换锚定带的前 M 帧：
#     hybrid = [全局池前 M 帧] + [本段窗口0 锚的其余 K−M 帧]
#   - 给多少像素帧就用多少（不设"取多少帧"参数）：M = ((T-1)//4)+1；
#   - 给的帧数超出锚定带 → 只取前 K 帧，等价于"完全替换"；不足 → "部分替换"；
#   - 不接 → 完全等价于纯本段锚定（默认行为零变化）；
#   - 分辨率由采样期 latent 空间尺寸反推（latent×8），无需额外接线；
#   - mask 池复用 wan_scail_multi_ref._extract_mask_to_28ch（无需 VAE）。
#
# 重要：必须接在上下文窗口节点（Custom Context Windows (Manual) /
#       Wan SCAIL-2 Context Windows）之后 —— handler 是那两个节点挂在
#       model.model_options["context_handler"] 上的。
# ==============================

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch


# 节点暴露给用户的“锚定目标”预设（下拉框）。内部值仍是 (键名, 时间维) 列表。
ANCHOR_TARGET_PRESETS = {
    "驱动视频 + 驱动遮罩 (auto)": [("pose_latents", 2), ("sam_latents", 2)],
    "仅驱动视频 (pose only)": [("pose_latents", 2)],
    "仅驱动遮罩 (mask only)": [("sam_latents", 2)],
}
DEFAULT_ANCHOR_TARGET = "驱动视频 + 驱动遮罩 (auto)"

# 分类遮罩键：0/1（或按身份分色）语义，禁止线性混合 → blend 会被强制为硬替换
MASK_COND_KEYS = {"sam_latents", "driving_mask_28ch", "ref_mask_latents", "ref_mask_28ch"}
# 各 cond 键对应的"全局池"来源：mask 类走 global_mask_frames，其余走 global_pose_frames
GLOBAL_POOL_SOURCE = {"sam_latents": "mask", "driving_mask_28ch": "mask"}
DEFAULT_POOL_SOURCE = "pose"


@dataclass
class AnchorConfig:
    """锚定配置（每个模型链一份）。"""

    keys: list = field(default_factory=list)   # [(cond_key, temporal_dim), ...]
    frames: int = 1                            # 锚定前 K 个 latent 帧
    offset: int = -1                           # 切片内起始帧偏移；-1=自动（跳过前缀派生帧 = max(0, 前缀帧数-1)）
    steps: int = -1                            # 只在前 N 个采样步锚定；-1=全程（默认），0=不锚定
    blend: float = 1.0                         # 1.0=完全替换；<1 与窗口自身首帧线性混合
    apply_to_negative: bool = True             # 是否也作用于第 2 个及之后的 cond 批次（CFG 的 negative）
    log_debug: bool = False                    # 打印每窗口命中的键与首帧信息（探针用）


def _parse_anchor_keys(spec: str) -> list:
    """解析 "key:dim, key:dim" 规格；缺省 dim=2，非法项忽略并告警。"""
    parsed = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, _, dim_text = part.rpartition(":")
            name = name.strip()
            try:
                dim = int(dim_text)
            except ValueError:
                logging.warning("[ContextAnchor] 无法解析时间维 '%s'，已忽略该项", part)
                continue
        else:
            name, dim = part, 2
        if not name or dim < 1:
            logging.warning("[ContextAnchor] 无效的锚定键 '%s'，已忽略该项", part)
            continue
        parsed.append((name, dim))
    return parsed


# 允许通过“旧字符串值”指定的键（仅用于兼容历史工作流；正常使用请用上面的下拉预设）
KNOWN_ANCHOR_KEYS = {"pose_latents", "sam_latents"}


def _resolve_targets(value) -> list:
    """把节点的下拉选项解析成 [(cond_key, temporal_dim), ...]；兼容旧工作流里存的原始字符串规格。"""
    if value in ANCHOR_TARGET_PRESETS:
        return list(ANCHOR_TARGET_PRESETS[value])
    parsed = [item for item in _parse_anchor_keys(value if isinstance(value, str) else "")
              if item[0] in KNOWN_ANCHOR_KEYS]
    if not parsed:
        logging.warning("[ContextAnchor] 无法识别的锚定目标 %r，已回退到默认 %s", value, DEFAULT_ANCHOR_TARGET)
        return list(ANCHOR_TARGET_PRESETS[DEFAULT_ANCHOR_TARGET])
    return parsed


def _get_tensor(value):
    """取出 cond 值里的张量；返回 (tensor, 是否 .cond 包装)。"""
    if isinstance(value, torch.Tensor):
        return value, False
    inner = getattr(value, "cond", None)
    if isinstance(inner, torch.Tensor):
        return inner, True
    return None, False


def _copy_with_tensor(value, tensor):
    """把替换后的张量放回 cond 值（保持 CONDRegular 等包装类型）。"""
    if hasattr(value, "_copy_with"):
        return value._copy_with(tensor)
    return tensor


def _slice_frames(tensor: torch.Tensor, dim: int, start: int, count: int) -> torch.Tensor:
    """沿 dim 取 [start, start+count) 帧（越界自动裁剪）。"""
    start = max(0, min(int(start), tensor.shape[dim]))
    count = max(0, min(int(count), tensor.shape[dim] - start))
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(start, start + count)
    return tensor[tuple(index)]


def _anchor_tensor(tensor: torch.Tensor, dim: int, anchor: torch.Tensor, blend: float,
                   start: int = 0) -> torch.Tensor:
    """把 tensor 的 [start, start+K) 帧替换为 anchor（blend<1 时线性混合）；形状不符则原样返回。"""
    start = max(0, min(int(start), tensor.shape[dim]))
    count = min(anchor.shape[dim], tensor.shape[dim] - start)
    if count <= 0:
        return tensor
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(start, start + count)
    index = tuple(index)

    anchor = anchor.to(device=tensor.device, dtype=tensor.dtype)
    if anchor.shape != tensor[index].shape:
        logging.warning("[ContextAnchor] 锚定源形状 %s 与目标 %s 不一致，跳过该键",
                        list(anchor.shape), list(tensor[index].shape))
        return tensor

    new_tensor = tensor.clone()
    if blend >= 1.0:
        new_tensor[index] = anchor
    else:
        alpha = max(0.0, float(blend))
        new_tensor[index] = tensor[index] * (1.0 - alpha) + anchor * alpha
    return new_tensor


def _as_image_batch(value):
    """把 IMAGE 输入统一成 (T, H, W, C) 张量；空输入返回 None。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] < 1:
        return None
    return value


def _resize_frames(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """把 (T,H,W,C) 帧缩放到 (target_h, target_w)（与节点内 pose/mask 分支一致：area/center）。"""
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    import comfy.utils
    return comfy.utils.common_upscale(frames.movedim(-1, 1).float(), int(target_w), int(target_h),
                                      "area", "center").movedim(1, -1)


def _match_channels(pool: torch.Tensor, target_channels: int) -> torch.Tensor:
    """SCAIL-2 的 pose_latents = 16ch latent + 4ch ones mask，模型侧就是这么拼的，这里照做。"""
    if pool.shape[1] == target_channels:
        return pool
    if pool.shape[1] + 4 == target_channels:
        return torch.cat([pool, torch.ones_like(pool[:, :4])], dim=1)
    return pool


def _build_global_pool(handler, key: str, tensor: torch.Tensor, dim: int):
    """按目标张量的空间尺寸构建「全局池」（每个键只算一次）；返回 (1,C,T',H,W) 或 None。"""
    cache = handler._anchor_global_cache
    spatial = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    cache_key = (key, spatial)
    if cache_key in cache:
        return cache[cache_key]

    source = GLOBAL_POOL_SOURCE.get(key, DEFAULT_POOL_SOURCE)
    frames = _as_image_batch(getattr(handler, "_anchor_global_mask", None) if source == "mask"
                             else getattr(handler, "_anchor_global_pose", None))
    if frames is None:
        cache[cache_key] = None
        return None

    target_h, target_w = spatial[0] * 8, spatial[1] * 8   # latent×8 = 该模型 pose/mask 分支的像素分辨率
    pool = None
    try:
        frames = _resize_frames(frames, target_h, target_w)
        if source == "mask":
            from .wan_scail_multi_ref import _extract_mask_to_28ch
            pool = _extract_mask_to_28ch(frames[:, :, :, :3])
        else:
            vae = getattr(handler, "_anchor_vae", None)
            if vae is None:
                logging.warning("[ContextAnchor] 接了 global_pose_frames 但没接 vae，无法编码全局池，已忽略")
                cache[cache_key] = None
                return None
            pool = vae.encode(frames[:, :, :, :3])
        pool = _match_channels(pool, int(tensor.shape[1]))
        if pool.shape[1] != tensor.shape[1]:
            logging.warning("[ContextAnchor] 全局池通道数 %d 与目标 %d 不一致，忽略该池（key=%s）",
                            pool.shape[1], tensor.shape[1], key)
            pool = None
        else:
            logging.info("[ContextAnchor] 全局池已构建：key=%s, 来源=%s, 输入 %d 像素帧 → %d latent 帧, 空间=%s",
                         key, source, int(frames.shape[0]), int(pool.shape[dim]), list(pool.shape[-2:]))
    except Exception:
        logging.exception("[ContextAnchor] 构建全局池失败（key=%s），本次忽略", key)
        pool = None
    cache[cache_key] = pool
    return pool


def _compose_hybrid_anchor(anchor: torch.Tensor, pool, dim: int):
    """用全局池替换锚的前 M 帧（M ≥ K 时等价于完全替换）；返回 (混合锚, M)。"""
    if pool is None or pool.shape[dim] <= 0:
        return anchor, 0
    k = int(anchor.shape[dim])
    m = min(int(pool.shape[dim]), k)
    pool = pool.to(device=anchor.device, dtype=anchor.dtype)
    hybrid = torch.cat([_slice_frames(pool, dim, 0, m), _slice_frames(anchor, dim, m, k - m)], dim=dim)
    return hybrid, m


def _install_anchor(handler, cfg: AnchorConfig):
    """把「窗口首帧锚定」后处理挂到 handler 实例上（实例属性遮蔽原方法，不改动其类）。

    说明：不采用 `handler.__class__ = 子类` 的方式 —— Python 3.13 下 abc.ABC 派生类的实例
    布局与普通 Mixin 不一致，会抛 "__class__ assignment: object layout differs"。
    实例属性包装对任意 handler 实现都可靠，且 super()/原方法链路不受影响。
    重复调用只会更新配置。
    """
    if getattr(handler, "_cwa_anchor_installed", False):
        handler._anchor_cfg = cfg
        return handler

    orig_get_resized_cond = handler.get_resized_cond
    orig_execute = handler.execute

    def execute(*args, **kwargs):
        # 记录本次 execute 的 cond 批次（CFG 下为 [positive, negative]）以便区分 side
        conds = args[2] if len(args) > 2 else kwargs.get("conds", None)
        handler._anchor_side_ids = [id(c) for c in conds] if conds is not None else []
        # 快照在每个采样步的窗口 0 处重建，这里清空上一步残留
        handler._anchor_cache = {}
        handler._anchor_step_skip_logged = None
        return orig_execute(*args, **kwargs)

    def get_resized_cond(cond_in, x_in, window, device=None):
        resized = orig_get_resized_cond(cond_in, x_in, window, device)
        try:
            return _anchor_resized_cond(handler, resized, window, cond_in)
        except Exception:
            logging.exception("[ContextAnchor] 窗口锚定失败，本次已跳过（原 conditioning 不变）")
            return resized

    handler.execute = execute
    handler.get_resized_cond = get_resized_cond
    handler._cwa_anchor_installed = True
    handler._anchor_cfg = cfg
    handler._anchor_cache = {}
    handler._anchor_side_ids = []
    handler._anchor_global_cache = {}
    return handler


def _resolve_window_state(handler, window):
    """返回 (window_idx, is_anchor_source)；is_anchor_source 表示这是窗口 0（快照来源）。"""
    window_idx = getattr(handler, "_current_window_idx", None)
    if isinstance(window_idx, int):
        return window_idx, window_idx == 0
    indices = getattr(window, "original_indices", None) or getattr(window, "index_list", None)
    if indices:
        return None, int(min(indices)) == 0
    return None, False


def _anchor_resized_cond(handler, resized_cond, window, cond_in):
    cfg = getattr(handler, "_anchor_cfg", None)
    if cfg is None or not cfg.keys or cfg.frames <= 0 or resized_cond is None:
        return resized_cond

    # ---- 采样步门控：只在前 cfg.steps 个采样步锚定（-1 = 全程，默认；0 = 不锚定）----
    # 思路：幅度混合会喂"合成中间态"，时间门控则每步都给真实信号（要么锚、要么本段），
    # 让早期步决定布局/对齐、后期步回归真实运动。
    if int(cfg.steps) >= 0:
        cur_step = getattr(handler, "_step", None)
        if isinstance(cur_step, int) and cur_step >= int(cfg.steps):
            if cfg.log_debug and getattr(handler, "_anchor_step_skip_logged", None) is None:
                logging.info("[ContextAnchor] 采样步 %d >= anchor_steps=%d：本步起停止锚定",
                             cur_step, int(cfg.steps))
                handler._anchor_step_skip_logged = cur_step
            return resized_cond

    side = 0
    for i, cond_id in enumerate(getattr(handler, "_anchor_side_ids", None) or []):
        if cond_id == id(cond_in):
            side = i
            break
    if side > 0 and not cfg.apply_to_negative:
        return resized_cond

    window_idx, is_anchor_source = _resolve_window_state(handler, window)
    index_list = getattr(window, "index_list", None) or []
    max_len = len(index_list) + 1   # 合理长度上限：窗口帧数（+1 容差），用于排除全序列张量
    cache = handler._anchor_cache

    # 有效偏移：跳过"前缀派生帧"。
    # 上下文节点把前缀索引（chunk 的 0..P-1 帧）拼到每个窗口的 index_list 前面，而 pose/遮罩切片
    # 用的是 index_list，于是切片头部会多出 P-1 帧（多参考图时是零填充，单参考图时是 chunk 开头几帧），
    # 它们对所有窗口都相同 → 锚这些帧等于空操作。故默认从切片第 max(0, P-1) 帧开始锚定。
    prefix_len = int(getattr(handler, "prefix_latent_len", 0) or 0)
    auto_offset = max(0, prefix_len - 1)
    offset = auto_offset if int(cfg.offset) < 0 else max(0, int(cfg.offset))

    for actual_cond in resized_cond:   # 元素是 handler 返回的浅拷贝，可就地更新
        if not isinstance(actual_cond, dict):
            continue
        model_conds = actual_cond.get("model_conds", None)
        if not isinstance(model_conds, dict):
            continue

        updated = None
        for key, dim in cfg.keys:
            if key not in model_conds:
                continue
            value = model_conds[key]
            tensor, _ = _get_tensor(value)
            if tensor is None or tensor.ndim <= dim:
                continue
            length = tensor.shape[dim]
            if length < 1 or (max_len > 0 and length > max_len):
                # 不是按窗口切分出来的张量（例如整段条件），不处理
                continue

            cache_key = (side, key)
            if offset >= length:
                if cfg.log_debug:
                    logging.info("[ContextAnchor] 窗口%s(side=%d) %s 切片长度=%d <= 偏移=%d，跳过"
                                 "（前缀帧数=%d）", window_idx, side, key, length, offset, prefix_len)
                continue
            if is_anchor_source:
                cache[cache_key] = _slice_frames(tensor, dim, offset, cfg.frames).detach().clone()
                if cfg.log_debug:
                    head_n = min(offset + 1, length, 12)
                    head_means = [round(tensor.select(dim, i).float().mean().item(), 4)
                                  for i in range(head_n)]
                    logging.info(
                        "[ContextAnchor] 窗口%s(side=%d) 记录锚定源 %s: shape=%s, dim=%d, 前缀帧数=%d, "
                        "偏移=%d(自动=%d), 取帧[%d..%d), 前%d帧mean=%s",
                        window_idx, side, key, list(tensor.shape), dim, prefix_len,
                        offset, auto_offset, offset, offset + cfg.frames, head_n, head_means)
                continue

            anchor = cache.get(cache_key, None)
            if anchor is None:
                continue

            # ---- 混合锚定：接了全局素材则用全局池替换锚的前 M 帧（M ≥ K = 完全替换）----
            pool = _build_global_pool(handler, key, tensor, dim)
            anchor, head_global = _compose_hybrid_anchor(anchor, pool, dim)

            # ---- per-key 的 blend 规则：分类遮罩禁止线性混合（模型没见过混合值）----
            blend = float(cfg.blend)
            if key in MASK_COND_KEYS and blend < 1.0:
                blend = 1.0

            new_tensor = _anchor_tensor(tensor, dim, anchor, blend, start=offset)
            if updated is None:
                updated = dict(model_conds)
            updated[key] = _copy_with_tensor(value, new_tensor)
            if cfg.log_debug:
                k_eff = int(anchor.shape[dim])
                logging.info(
                    "[ContextAnchor] 窗口%s(side=%d) 锚定 %s: 替换帧[%d..%d) ← 头部 %d 帧来自全局池 + "
                    "%d 帧来自本段窗口0, blend=%.2f, shape=%s",
                    window_idx, side, key, offset, offset + k_eff, head_global, k_eff - head_global,
                    blend, list(new_tensor.shape))

        if updated is not None:
            actual_cond["model_conds"] = updated

    return resized_cond


class WanContextWindowAnchorNode:
    """把每个上下文窗口的驱动视频（及 SCAIL-2 驱动遮罩）首帧锚定到当前片段窗口 0 的首帧。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（Custom Context Windows (Manual) 或 "
                                               "Wan SCAIL-2 Context Windows 的输出）。Model already wrapped by a "
                                               "context windows node."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "是否启用锚定；关闭时模型直接透传。Whether to enable anchoring; "
                                                   "when off the model is passed through unchanged."}),
                "anchor_targets": (list(ANCHOR_TARGET_PRESETS.keys()),
                                   {"default": DEFAULT_ANCHOR_TARGET,
                                    "tooltip": "要锚定的目标（下拉选择）。驱动视频=pose 分支（animate2 / SCAIL-2 都是 "
                                               "pose_latents）；驱动遮罩=SCAIL-2 的 driving_mask（sam_latents）。"
                                               "Targets to anchor: the driving video (pose branch) and/or the SCAIL-2 "
                                               "driving mask."}),
                "anchor_frames": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1,
                                          "tooltip": "每个窗口要锚定的 latent 帧数，0=关闭。建议先 1，再试 2-3。"
                                                     "Number of leading latent frames to anchor per window; 0 disables."}),
                "anchor_offset": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1,
                                          "tooltip": "-1=自动：跳过切片头部的“前缀派生帧”（=max(0, 前缀帧数-1)），"
                                                     "从窗口自身的第一个驱动帧开始锚定；填 0/1/… 可手动指定起始帧做实验。"
                                                     "-1 = auto: skip prefix-derived frames at the slice head."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "1.0=完全替换为窗口 0 的首帧；<1.0 与窗口自身首帧线性混合（用于缓解"
                                               "首帧姿态回跳造成的抖动）。1.0 fully replaces with the anchor; below "
                                               "1.0 linearly blends with the window's own frames."}),
                "apply_to_negative": ("BOOLEAN", {"default": True,
                                                  "tooltip": "CFG 下是否同时锚定 negative（通常两者传入同一份驱动"
                                                             "条件，锚定结果一致，不会造成 CFG 偏斜）。Also anchor the "
                                                             "negative conditioning under CFG."}),
                "log_debug": ("BOOLEAN", {"default": False,
                                          "tooltip": "打印每个窗口命中的键、shape 与首帧数值，用于验证锚定是否生效。"
                                                     "Log per-window hits, shapes and first-frame stats for debugging."}),
                "anchor_steps": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1,
                                         "tooltip": "只在前 N 个采样步锚定：-1=全程（默认）；0=不锚定；N>0=前 N 步。"
                                                    "时间门控（每步给真实信号，而非幅度混合的合成中间态）。"
                                                    "-1 = all steps (default); N > 0 = only the first N sampling steps."}),
                "global_pose_frames": ("IMAGE", {"tooltip": "【可选】驱动/pose 视频开头的若干像素帧（给多少用多少，不设帧数参数）。"
                                                            "编码成 latent 后替换锚定带的前 M 帧（M=((T-1)//4)+1）：给多了=完全替换，"
                                                            "给少了=部分替换；不接=纯本段锚定。Optional global driving-video head frames."}),
                "global_mask_frames": ("IMAGE", {"tooltip": "【可选，SCAIL-2】彩色遮罩视频开头的若干像素帧，"
                                                            "按 _extract_mask_to_28ch 编码后替换遮罩锚的前 M 帧。"
                                                            "Optional global colored-mask head frames (SCAIL-2)."}),
                "vae": ("VAE", {"tooltip": "【可选】仅用于把 global_pose_frames 编码成 latent（接同一个 Wan VAE）；"
                                           "遮罩池不需要 VAE。VAE used to encode global_pose_frames."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_anchor"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = ("⚠️ 实验性 / 测试性节点：有效性与副作用尚未充分验证，仅供测试参考，请自行做开/关对比后再使用。Experimental / test-only node: effectiveness and side effects are not fully validated — testing reference only. 把上下文窗口采样中每个窗口的驱动视频 latent 首帧（animate2: pose_latents；SCAIL-2: "
                   "pose_latents + 驱动遮罩 sam_latents）锚定为当前片段窗口 0 的首帧，使各窗口的内部对齐基准一致，"
                   "缓解窗口融合处的比例/位置跳变。必须接在上下文窗口节点之后。Anchors the leading driving-video "
                   "latent frame(s) of every context window to window 0's frame so that all windows share the same "
                   "internal alignment reference. Must be placed after a context windows node.")

    def apply_anchor(self, model, enabled=True, anchor_targets=DEFAULT_ANCHOR_TARGET, anchor_frames=1,
                     anchor_offset=-1, blend=1.0, apply_to_negative=True, log_debug=False,
                     anchor_steps=-1, global_pose_frames=None, global_mask_frames=None, vae=None):
        handler = model.model_options.get("context_handler", None)
        if handler is None:
            logging.warning("[ContextAnchor] 模型上没有 context_handler —— 请把本节点接在上下文窗口节点"
                            "（Custom Context Windows (Manual) / Wan SCAIL-2 Context Windows）之后。模型已直接透传。")
            return (model,)

        if not enabled or int(anchor_frames) <= 0:
            logging.info("[ContextAnchor] 未启用（enabled=%s, anchor_frames=%s），模型直接透传",
                         enabled, anchor_frames)
            return (model,)

        parsed_keys = _resolve_targets(anchor_targets)
        if not parsed_keys:
            logging.warning("[ContextAnchor] 锚定目标为空，模型直接透传")
            return (model,)

        # 克隆模型 + 浅拷贝 handler：不改动上游节点持有的对象
        model = model.clone()
        new_handler = copy.copy(handler)
        _install_anchor(new_handler, AnchorConfig(
            keys=parsed_keys,
            frames=int(anchor_frames),
            offset=int(anchor_offset),
            steps=int(anchor_steps),
            blend=float(blend),
            apply_to_negative=bool(apply_to_negative),
            log_debug=bool(log_debug),
        ))
        model.model_options["context_handler"] = new_handler

        new_handler._anchor_global_pose = global_pose_frames
        new_handler._anchor_global_mask = global_mask_frames
        new_handler._anchor_vae = vae
        new_handler._anchor_global_cache = {}

        global_note = []
        if _as_image_batch(global_pose_frames) is not None:
            global_note.append("pose 帧 %d" % int(_as_image_batch(global_pose_frames).shape[0]))
        if _as_image_batch(global_mask_frames) is not None:
            global_note.append("mask 帧 %d" % int(_as_image_batch(global_mask_frames).shape[0]))

        logging.info("[ContextAnchor] 已启用：目标=%s → keys=%s, frames=%d, offset=%s, steps=%s, blend=%.2f, "
                     "apply_to_negative=%s, 全局池=%s (handler=%s)", anchor_targets, parsed_keys,
                     int(anchor_frames), "auto" if int(anchor_offset) < 0 else int(anchor_offset),
                     "全程" if int(anchor_steps) < 0 else ("不锚定" if int(anchor_steps) == 0
                                                          else "前 %d 步" % int(anchor_steps)),
                     float(blend), bool(apply_to_negative),
                     ("混合锚定：" + ", ".join(global_note)) if global_note else "未接入（纯本段锚定）",
                     type(handler).__name__)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "WanContextWindowAnchor": WanContextWindowAnchorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowAnchor": "Context Window Anchor (窗口首帧锚定)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
