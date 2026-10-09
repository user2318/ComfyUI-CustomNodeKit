"""Wan SCAIL-2 滚动窗口（实验性 / S2）—— **步内 x 前移**。

要解决的问题
------------
上下文窗口 + 融合（Gen-L-Video 式：每步各窗独立 forward，再按权重平均）在相邻窗
「对同一全局帧给出两份不同 x0」时会平均成**重影 / 动作发软**。`overlap_scope=single`
（重叠只读化）已经能让每帧只有一个作者，但**后窗看到的仍是「上一步的融合值」**，
它在本步的 forward 里并不知道前窗**本步**已经决定了什么。

本节点做的事
------------
在**同一个去噪步内**，让后面的窗口看到前面窗口**本步刚产出的 x0**：

    窗 k forward 之前：
        对本窗的「只读帧」i（已被前窗写过）：
            x_sub[i] <- (1-β)·x_sub[i] + β·( (1-σ)·x0_prev[i] + σ·ε )
        其中 x0_prev[i] = 前窗**本步**对帧 i 的 x0 预测；ε = 确定性噪声（roll_seed 决定）
    窗 k forward 之后：
        记录本窗**拥有**的帧（window.original_indices）在本步的 x0，供后续窗口使用

关键事实（本机代码已验证）
--------------------------
· `calc_cond_batch` 返回的是 **x0（denoised）**，不是 v/ε（`model_base.py:258` 已做换算）
  => 不需要自己反推 x0。
· ComfyUI 侧 `handler.execute(calc_cond_batch, ...)` 的这个 `calc_cond_batch` 参数
  **就是「单个窗口的 forward」**，每窗调用一次 => 所有逐窗干预都在这里挂钩。
· `window.get_tensor()` 用列表索引 => 返回**副本**，改 `sub_x` 不会污染 `x_in`。

为什么必须带 σ·ε
----------------
直接把这帧换成干净的 x0 属于**分布外输入**（模型期望 x_σ = (1-σ)·x0 + σ·ε；
喂干净帧就是社区说的 "too clean previous frames" 漂移来源）。所以用**同一 σ**再噪。

为什么建议配 overlap_scope=single
--------------------------------
本节点只改「本窗只读的帧」（`roll_scope=readonly`，默认）=> 永远不动本窗拥有的帧，
所有权不变、不引入新的平均。若重叠区仍是「两窗共写 + 加权平均」（`average`），
本步提交的效果会被平均削弱（此时可把 `roll_scope` 设成 `both` 让它影响本窗写回值）。

与 `head_context_frames` 的区别
-------------------------------
`head_context_frames` 让窗口**看得到**前面几帧，但看到的是 `x_in` 里**上一步**的值；
本节点让那几帧变成**本步**的值 => 多出一整步的信息流动（4 步采样下这很可观）。

依赖与接线
----------
必须接在上下文窗口节点（`Wan SCAIL-2 Context Windows` / `Custom Context Windows (Manual)`）
**之后**；与 `Context Window Anchor` 可共存（都通过 handler_wrap 包装链挂载，互不覆盖）。
`roll_mode=off` 或 `roll_beta=0` 或 `enabled=关` => 完全透传（等价于没装本节点）。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch

try:
    from .handler_wrap import install as _wrap_install, describe as _wrap_describe
except ImportError:                                     # pragma: no cover - 顶层导入（自测脚本）
    from handler_wrap import install as _wrap_install, describe as _wrap_describe


# 包装链里的 owner 标识（与 ContextWindowAnchor 平级，互不覆盖）
_WRAP_OWNER = "RollingWindows"


@dataclass
class RollingConfig:
    mode: str = "intra_step_x"      # off / intra_step_x
    beta: float = 0.5               # 提交强度：0=透传，1=完全用前窗本步 x0 再噪
    scope: str = "readonly"         # readonly=只改本窗只读帧 / both=也改本窗拥有的重叠帧
    seed: int = 0                   # 再噪用的确定性噪声种子


def _current_window(model_options):
    """取当前正在评估的窗口（由 base execute 在调用 calc_cond_batch 前写入 transformer_options）。"""
    try:
        return (model_options or {}).get("transformer_options", {}).get("context_window", None)
    except Exception:
        return None


def _frame_positions(window, sub_x, dim: int):
    """全局帧索引 -> sub_x 的局部位置。

    offset = sub_x 帧数 - len(index_list)：模型 patch / 锚定节点可能在本窗切片**之外**
    追加了额外帧（SCAIL-2 的锚帧就是前插的），用长度差自动对齐，比约定「前插几帧」稳健。
    """
    idx_list = [int(i) for i in (getattr(window, "index_list", None) or [])]
    offset = int(sub_x.shape[dim]) - len(idx_list)
    if offset < 0:
        return {}
    return {i: offset + n for n, i in enumerate(idx_list)}


def _sigma_step_key(sigma: float) -> int:
    """用 σ 当「步」的标识：同一步内所有窗口一致，跨运行稳定（不需要访问 handler 的 _step）。"""
    return int(round(float(sigma) * 1_000_000))


def _seeded_noise_like(ref: torch.Tensor, seed: int, step_key: int, idx: int) -> torch.Tensor:
    """确定性噪声：同一 (seed, σ, 帧) => 逐位一致（保证 A/B 可复现）。

    在 CPU 上按固定种子生成再搬到目标设备 => 跨设备也一致。
    只在「需要提交的帧」上按需生成，开销 = 每步 * 重叠帧数 个单帧随机数。
    """
    h = int(seed) * 1000003 + int(step_key) * 9176 + int(idx)
    g = torch.Generator(device="cpu").manual_seed(h & 0x7FFFFFFF)
    noise = torch.randn(tuple(ref.shape), generator=g, dtype=torch.float32)
    return noise.to(device=ref.device, dtype=ref.dtype)


def _frame_slice(tensor: torch.Tensor, dim: int, pos: int):
    sl = [slice(None)] * tensor.ndim
    sl[dim] = slice(pos, pos + 1)
    return tuple(sl)


def _apply_commit(sub_x, pos_of, own, state, sigma, cfg: RollingConfig):
    """把「已被前面窗口写过的帧」在本窗输入里换成 (1-σ)·x0_prev + σ·ε（按 β 混合）。"""
    committed = state["committed"]
    if not committed:
        return sub_x
    targets = [i for i in pos_of
               if i in committed and (cfg.scope == "both" or i not in own)]
    if not targets:
        return sub_x
    out = sub_x.clone()
    for i in targets:
        sl = _frame_slice(out, state["dim"], pos_of[i])
        ref = out[sl]
        eps = _seeded_noise_like(ref, cfg.seed, state["step"], i)
        target = (1.0 - sigma) * committed[i].to(ref.device, ref.dtype) + sigma * eps
        out[sl] = (1.0 - cfg.beta) * ref + cfg.beta * target
        state["hits"] += 1
    return out


def _record_commit(out, pos_of, own, state):
    """记录本窗**拥有**的帧在本步的 x0（供后面的窗口使用）。

    `calc_cond_batch` 返回 list（每个 cond 一路）；与 `_capture_anchor_x0` 一致取第 0 路
    —— lightx2v 4 步蒸馏常用 cfg=1.0，此时 CFG 输出恰等于 cond 路（`cfg_function` 在
    cond_scale=1 时退化为 cond_pred），所以取 out[0] 是准确的。
    """
    tensor = out
    if isinstance(out, (list, tuple)):
        tensor = out[0] if len(out) else None
    if not torch.is_tensor(tensor):
        return
    dim = state["dim"]
    if tensor.ndim <= dim:
        return
    for i in own:
        p = pos_of.get(i)
        if p is None or p >= tensor.shape[dim]:
            continue
        state["committed"][i] = tensor[_frame_slice(tensor, dim, p)].detach()


def _make_execute(cfg: RollingConfig, dim: int):
    """工厂：在 `execute` 外层包一圈「步内 x 前移」。

    注意 `calc_cond_batch` 参数**就是单窗 forward** —— base execute 会按窗口顺序逐次调用它，
    所以在这里包它即可拿到「每窗 + 每步」的粒度，且不需要改动 `Custom_context.py`。
    """
    def factory(next_call):
        def execute(calc_cond_batch, model, conds, x_in, timestep, model_options):
            if cfg.mode == "off" or float(cfg.beta) <= 0.0:
                return next_call(calc_cond_batch, model, conds, x_in, timestep, model_options)
            try:
                sigma = float(timestep.reshape(-1)[0])
            except Exception:
                sigma = 0.0
            # 步内状态：每次 execute（= 每个去噪步）重置一次
            state = {"step": _sigma_step_key(sigma), "dim": dim,
                     "committed": {}, "windows": 0, "hits": 0}

            def rolling_cb(m, c, sub_x, ts, mo, **kw):
                window = _current_window(mo)
                if window is None or not torch.is_tensor(sub_x):
                    return calc_cond_batch(m, c, sub_x, ts, mo, **kw)
                pos_of = _frame_positions(window, sub_x, dim)
                if not pos_of:
                    return calc_cond_batch(m, c, sub_x, ts, mo, **kw)
                own = set(int(i) for i in (getattr(window, "original_indices", None) or []))
                sub_x = _apply_commit(sub_x, pos_of, own, state, sigma, cfg)
                out = calc_cond_batch(m, c, sub_x, ts, mo, **kw)
                _record_commit(out, pos_of, own, state)
                state["windows"] += 1
                return out

            result = next_call(rolling_cb, model, conds, x_in, timestep, model_options)
            logging.info("[RollingWindows] σ=%.3f 步完成：%d 窗，提交命中 %d 帧（mode=%s beta=%.2f scope=%s）",
                         sigma, state["windows"], state["hits"], cfg.mode, cfg.beta, cfg.scope)
            return result
        return execute
    return factory


# ---------------- 节点 ----------------

class WanSCAIL2RollingWindowsNode:
    """SCAIL-2 滚动窗口（步内 x 前移）。必须接在上下文窗口节点之后。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（`Wan SCAIL-2 Context Windows` 或 `Custom Context Windows (Manual)` 的 model 输出）。Model that already has a context windows node applied."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关；关闭时模型直接透传。Master switch."}),
                "roll_mode": (["off", "intra_step_x"], {"default": "intra_step_x",
                    "tooltip": "off = 透传（等价于没装本节点）；\n"
                               "intra_step_x = 步内 x 前移：后面的窗口在【本步】就能看到前面窗口刚产出的 x0。\n"
                               "off = passthrough; intra_step_x = later windows see the previous window's current-step x0."}),
                "roll_beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "提交强度 β：0 = 完全不改（透传）；1 = 该帧的输入完全换成「前窗本步 x0 按当前 σ 再噪」。\n"
                               "建议先试 0.3~0.6；若出现噪点感/闪烁就往回收。\n"
                               "Commit strength. 0 = passthrough, 1 = full replace with re-noised previous-window x0."}),
                "roll_scope": (["readonly", "both"], {"default": "readonly",
                    "tooltip": "readonly（推荐）= 只改【本窗只读的帧】（已被前窗写过），永不动本窗拥有的帧\n"
                               "    => 所有权不变、不引入新的平均；\n"
                               "both = 连本窗拥有的重叠帧也改（仅在 overlap_scope=average 时有区别；single 下两者等价）。"}),
                "roll_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                    "tooltip": "再噪用的确定性噪声种子（同一 seed/σ/帧 => 逐位一致，便于 A/B 对照）。换这个值会改变结果。\n"
                               "Deterministic noise seed for re-noising."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_rolling"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "SCAIL-2 滚动窗口：在同一步内把「已被前窗写过的帧」换成 (1-σ)·前窗本步 x0 + σ·ε，\n"
        "让后面的窗口看到前面窗口刚产出的内容（而不是上一步的融合值）。\n"
        "需接在上下文窗口节点之后；建议配 overlap_scope=single（重叠只读化）。\n"
        "roll_mode=off / roll_beta=0 / enabled=关 => 完全透传。"
    )

    def apply_rolling(self, model, enabled=True, roll_mode="intra_step_x", roll_beta=0.5,
                      roll_scope="readonly", roll_seed=0):
        if not enabled or str(roll_mode) == "off" or float(roll_beta) <= 0.0:
            logging.info("[RollingWindows] 未启用（enabled=%s, mode=%s, beta=%.3f），模型直接透传",
                         enabled, roll_mode, roll_beta)
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[RollingWindows] 模型上没有 context_handler：请把本节点接在上下文窗口节点"
                "（`Wan SCAIL-2 Context Windows` 或 `Custom Context Windows (Manual)`）之后。")

        cfg = RollingConfig(mode=str(roll_mode), beta=float(roll_beta),
                            scope=str(roll_scope), seed=int(roll_seed))

        model = model.clone()
        # 浅拷贝：包装链会为副本重建并重放既有包装（handler_wrap 保证原 handler 不被污染）
        new_handler = copy.copy(handler)
        dim = int(getattr(new_handler, "dim", 2) or 2)
        applied = _wrap_install(new_handler, _WRAP_OWNER, {"execute": _make_execute(cfg, dim)})
        model.model_options["context_handler"] = new_handler

        logging.info("[RollingWindows] 已启用：mode=%s beta=%.2f scope=%s seed=%d（dim=%d）",
                     cfg.mode, cfg.beta, cfg.scope, cfg.seed, dim)
        logging.info("[RollingWindows] 包装链：%s（本次安装 %s）", _wrap_describe(new_handler), applied)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanSCAIL2RollingWindows": WanSCAIL2RollingWindowsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAIL2RollingWindows": "Wan SCAIL-2 Rolling Windows (步内滚动)",
}
