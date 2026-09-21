# ==============================
# Context Window Anchor —— 上下文窗口「统一基准锚定（拼接式） + 双侧延伸」（实验性）
#
# 背景（要解决的问题）：
#   animate2 / SCAIL-2 都没有骨骼这类显式空间锚，「参考图 ↔ 驱动」的对齐是 in-context 推断出来的；
#   而窗口化之后 RoPE 每个窗口都恒为 0..L-1（不携带绝对时间信息，见 comfy/ldm/wan/model.py 与
#   model_animate2.py），模型区分窗口身份只能靠 conditioning。于是：
#     - 每个窗口都「从自己的开头重新推断对齐基准」→ 起点不同则同一全局帧的生成结果不同
#       → 相邻窗口在重叠区不一致（重影 / 动作对不上）；
#     - 起点那一帧的姿态若与参考图差得远 → 推断失败 → 整段结构与动作崩坏。
#
# 实测依据（用户用「两段有重叠的片段各自单窗生成」做的代理实验）：
#   - 只把「统一开头」拼到两段头部，两段在同一全局帧上就大幅趋同（大姿势一致，仅剩细节/空间位置残差）；
#   - 不拼，则同一全局帧差别巨大，其中一段结构扭曲；
#   ⇒ 对齐基准可被「头部内容」锁定，且残差来自「重叠帧永远落在两窗的边缘（一窗无未来、一窗无过去）」。
#
# 本节点（一个节点，两个可独立开关的机制）：
#   1) 锚定（拼接式，anchor_frames > 0）：把「统一基准帧」作为**额外帧**拼在参考前缀与窗口帧之间
#      参与采样，采样后裁掉。**不再用替换**——替换会把重叠区里窗口自己的驱动内容改掉，
#      反而制造不一致（旧实现「只有一点点效果」的原因之一）。
#      基准源：本段窗口0开头 / 本段开头（都在 x 内 → 纯索引，零拷贝）。
#      首窗自动跳过：基准源本来就在首窗内，拼上去等于自我重复。
#   2) 双侧延伸（halo_frames_head / halo_frames_tail > 0）：把窗口前后各 b 帧真实相邻帧纳入窗口
#      的「视野」，但不参与融合（采样后裁掉）。这样每个「被拥有」的帧都处在窗口内部、前后都有
#      上下文，而不是永远落在两窗的边缘。
#
# 生成的切片结构（x 坐标系）：
#   [参考前缀] [锚定帧(统一基准)] [前延伸帧(紧邻)] [本窗口帧(参与融合)] [后延伸帧]
#   前缀、头段、尾段只作为采样时的上下文，融合前由 combine_context_window_results 裁掉。
#
# 典型用法：
#   - 只做双侧延伸：halo_head/tail = 4~8，并把上下文节点的 overlap 设为 0（窗口平铺，
#     每个全局帧只有一个「拥有者」→ 现有加权平均自动退化为「owner 说了算」，重影从机制上消失）
#   - 只做统一基准：anchor_frames = 1~2（量级对齐训练期「首 2 个 latent 帧 = 历史」）
#   - 两者叠加：先 halo（保证每帧前后有上下文），再 anchor（保证跨窗基准一致）
#
# 生效位置：采样期 per-window 条件重切分之前（改写 window.index_list /
#          head_extra_indices / tail_extra_indices），对三条切片路径同时生效：
#          - 插件 handler 的 get_resized_cond（含 SCAIL 专用 handler）
#          - 模型级 model.resize_cond_for_context_window
#          - 核心 slice_cond（按 index_list + temporal_offset）
#
# 重要：必须接在上下文窗口节点（Custom Context Windows (Manual) /
#       Wan SCAIL Context Windows）之后 —— handler 挂在 model.model_options["context_handler"]。
# 实现方式：浅拷贝 handler 实例并包装 get_context_windows（不做 __class__ 置换 —— 3.13 下 abc.ABC
#           派生类会报 layout differs）。
# ==============================

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch  # noqa: F401  (保持与本仓库其它节点一致的导入习惯/未来扩展)


# 锚定基准源（都在 x 内，纯索引实现）
ANCHOR_SOURCES = ["本段窗口0开头", "本段开头"]


@dataclass
class AnchorConfig:
    """锚定 + 双侧延伸配置（每条模型链一份）。"""

    anchor_frames: int = 0                     # 拼接式锚定帧数（0=关闭）
    anchor_source: str = ANCHOR_SOURCES[0]     # 基准源
    halo_head: int = 0                         # 窗口向前延伸的 latent 帧数（0=关闭）
    halo_tail: int = 0                         # 窗口向后延伸的 latent 帧数（0=关闭）
    log_debug: bool = False                    # 打印每个窗口的拼接明细


def _window_original(window) -> list:
    return list(getattr(window, "original_indices", None) or [])


def _window_prefix(window) -> list:
    """窗口的参考前缀帧索引（位于 index_list 最前、且不属于窗口帧）。"""
    original = set(_window_original(window))
    return [i for i in (getattr(window, "index_list", None) or []) if i not in original]


def apply_window_extras(handler, windows, cfg: AnchorConfig) -> None:
    """按配置就地改写每个窗口的 index_list 与 extra 段（不改 original_indices）。"""
    if not windows:
        return

    # 参考前缀帧数：以 handler 的 prefix_latent_len 为准（也就是 x 最前面那 P 帧）
    ref_prefix_len = int(getattr(handler, "prefix_latent_len", 0) or 0)

    w0 = windows[0]
    w0_original = _window_original(w0)
    x_len = int(getattr(w0, "total_frames", 0) or 0)
    if x_len <= 0:
        x_len = max([max(getattr(w, "index_list", None) or [0]) for w in windows] or [0]) + 1

    anchor_frames = max(0, int(cfg.anchor_frames))
    halo_head = max(0, int(cfg.halo_head))
    halo_tail = max(0, int(cfg.halo_tail))

    # 锚定源池（都不含参考前缀帧）
    pool_w0 = [i for i in w0_original if i >= ref_prefix_len][:anchor_frames]
    pool_chunk = [i for i in range(ref_prefix_len, x_len)][:anchor_frames]

    for window in windows:
        original = _window_original(window)
        if not original:
            continue
        prefix = _window_prefix(window)
        used = set(prefix) | set(original)
        lo, hi = min(original), max(original)

        # ---- 1) 锚定帧（拼接式统一基准；只取窗口之前的帧，首窗自动为空）----
        anchor = []
        if anchor_frames > 0 and lo > ref_prefix_len:
            pool = pool_w0 if cfg.anchor_source == ANCHOR_SOURCES[0] else pool_chunk
            for i in pool:
                if i < lo and i not in used:
                    anchor.append(i)
                    used.add(i)

        # ---- 2) 前延伸（真实相邻帧；不越过参考前缀区）----
        head_halo = []
        if halo_head > 0:
            start = max(ref_prefix_len, lo - halo_head)
            for i in range(start, lo):
                if i not in used:
                    head_halo.append(i)
                    used.add(i)

        # ---- 3) 后延伸（真实相邻帧；越过 x 末尾自动截断）----
        tail_halo = []
        if halo_tail > 0:
            for i in range(hi + 1, min(x_len, hi + 1 + halo_tail)):
                if i not in used:
                    tail_halo.append(i)
                    used.add(i)

        head_extra = anchor + head_halo          # 时间顺序：锚定（更早）→ 紧邻的前延伸
        window.head_extra_indices = head_extra
        window.tail_extra_indices = tail_halo
        window.index_list = list(prefix) + head_extra + original + tail_halo
        window.context_length = len(window.index_list)

        if cfg.log_debug:
            logging.info(
                "[ContextAnchor] 窗口帧 %d..%d → 切片帧 %d..%d | 前缀=%d 锚定=%d 前延伸=%d 后延伸=%d（原始长度=%d→%d）",
                lo, hi,
                window.index_list[0] if window.index_list else -1,
                window.index_list[-1] if window.index_list else -1,
                len(prefix), len(anchor), len(head_halo), len(tail_halo),
                len(original), len(window.index_list))


def _install(handler, cfg: AnchorConfig):
    """把「锚定 + 双侧延伸」挂到 handler 实例上（实例属性遮蔽原方法，不改动其类）。重复调用只更新配置。"""
    if getattr(handler, "_cwa_anchor_installed", False):
        handler._anchor_cfg = cfg
        return handler

    orig_get_context_windows = handler.get_context_windows

    def get_context_windows(model, x_in, model_options):
        windows = orig_get_context_windows(model, x_in, model_options)
        try:
            apply_window_extras(handler, windows, cfg)
        except Exception:
            logging.exception("[ContextAnchor] 窗口锚定/双侧延伸失败，本步回退为原始窗口")
        return windows

    handler.get_context_windows = get_context_windows
    handler._cwa_anchor_installed = True
    handler._anchor_cfg = cfg
    return handler


class WanContextWindowAnchorNode:
    """上下文窗口「统一基准锚定（拼接式）+ 双侧延伸」节点。

    两个机制都可独立关闭（参数为 0 即为关闭，行为完全等价于未安装本节点）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关。关闭时模型直接透传。Master switch; when off the model passes through unchanged."}),
                "anchor_frames": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "【拼接式锚定】把「统一基准帧」作为额外帧拼在窗口开头（采样后裁掉）；0=关闭。建议 1~2（对齐训练期「首 2 个 latent 帧=历史」的量级）。\nConcat-style anchor: prepend N latent frames of the shared basis to every window (trimmed after sampling). 0=off."}),
                "anchor_source": (ANCHOR_SOURCES, {"default": ANCHOR_SOURCES[0],
                    "tooltip": "基准源：本段窗口0开头 = 当前片段第一个窗口的开头若干帧（首窗自动跳过，避免自我重复）；本段开头 = 当前片段最前的若干视频帧。二者都在 x 内，纯索引实现、零拷贝。\nAnchor basis source; both live inside x (index-only, no copies)."}),
                "halo_frames_head": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "【双侧延伸·前】把窗口之前的 N 个 latent 帧纳入窗口视野（只做上下文、不参与融合）；0=关闭。首窗自动为 0（前面是参考前缀区）。\nHead halo: preceding latent frames the window can see (context only, not fused). 0=off."}),
                "halo_frames_tail": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "【双侧延伸·后】把窗口之后的 N 个 latent 帧纳入窗口视野（只做上下文、不参与融合）；0=关闭。末窗自动截断。\nTail halo: following latent frames the window can see (context only, not fused). 0=off."}),
            },
            "optional": {
                "log_debug": ("BOOLEAN", {"default": False,
                    "tooltip": "打印每个窗口的拼接明细（前缀/锚定/前延伸/后延伸帧数）。Log per-window splice details."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_anchor"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "窗口「统一基准锚定（拼接式）+ 双侧延伸」节点。⚠️ 实验性：有效性与副作用需自行做开/关对比验证。\n"
        "1) 锚定：把统一基准帧作为额外帧拼在窗口开头（不再替换窗口自己的帧，避免破坏重叠区的逐帧对应）。\n"
        "2) 双侧延伸：把窗口前后各 N 帧真实相邻帧纳入视野但不参与融合，让每个帧都不再落在两个窗口的边缘。\n"
        "切片结构：[参考前缀][锚定帧][前延伸帧][本窗口帧(参与融合)][后延伸帧]。\n"
        "参数为 0 即关闭对应机制（完全等价于不装本节点）。必须接在上下文窗口节点之后。\n"
        "推荐：双侧延伸与「上下文节点 overlap=0」配合 → 每个全局帧只有一个拥有者，重影从机制上消失。\n"
        "Splice-style unified-basis anchoring + two-sided halo for context windows (experimental)."
    )

    def apply_anchor(self, model, enabled=True, anchor_frames=0, anchor_source=ANCHOR_SOURCES[0],
                     halo_frames_head=0, halo_frames_tail=0, log_debug=False):
        anchor_frames = max(0, int(anchor_frames))
        halo_frames_head = max(0, int(halo_frames_head))
        halo_frames_tail = max(0, int(halo_frames_tail))

        if not enabled or (anchor_frames <= 0 and halo_frames_head <= 0 and halo_frames_tail <= 0):
            logging.info(
                "[ContextAnchor] 未启用（enabled=%s, anchor=%d, halo=%d/%d），模型直接透传",
                enabled, anchor_frames, halo_frames_head, halo_frames_tail)
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[ContextAnchor] 模型上没有 context_handler：请把本节点接在 "
                "WAN Context Windows (Manual Custom) / Wan SCAIL Context Windows 之后。")

        cfg = AnchorConfig(
            anchor_frames=anchor_frames,
            anchor_source=anchor_source if anchor_source in ANCHOR_SOURCES else ANCHOR_SOURCES[0],
            halo_head=halo_frames_head,
            halo_tail=halo_frames_tail,
            log_debug=bool(log_debug),
        )

        model = model.clone()
        new_handler = copy.copy(handler)
        _install(new_handler, cfg)
        model.model_options["context_handler"] = new_handler

        logging.info(
            "[ContextAnchor] 已应用：锚定 %d 帧（源=%s）+ 双侧延伸 %d/%d latent 帧。"
            "提示：若启用双侧延伸，建议把上下文节点的 overlap 设为 0，让每个全局帧只有一个拥有者。",
            anchor_frames, cfg.anchor_source, halo_frames_head, halo_frames_tail)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanContextWindowAnchorNode": WanContextWindowAnchorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowAnchorNode": "Context Window Anchor + Halo (窗口锚定/双侧延伸)",
}

