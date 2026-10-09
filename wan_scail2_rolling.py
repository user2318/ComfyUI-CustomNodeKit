"""Wan SCAIL-2 滚动窗口（实验性 / S2）—— **步内 x 前移**。

要解决的问题
------------
上下文窗口 + 融合（Gen-L-Video 式：每步各窗独立 forward，再按权重平均）在相邻窗
「对同一全局帧给出两份不同 x0」时会平均成**重影 / 动作发软**。`overlap_scope=single`
（重叠只读化）已经能让每帧只有一个作者，但**后窗看到的仍是「上一步的融合值」**，
它在本步的 forward 里并不知道前窗**本步**已经决定了什么。

本节点做的事（S2.1：`noise=content_delta` 为默认）
------------------------------------------------
在**同一个去噪步内**，让后面的窗口看到前面窗口**本步刚产出的 x0**：

    窗 k forward 之前（对待提交的帧 i）：
        Δ_i        = x0_prev[i] - (上一步的融合 x0)[i]
        x_sub[i]  <- x_sub[i] + A · β · Δ_i        ← 只平移「内容」，噪声一字不动
        其中 x0_prev[i] = 前窗**本步**对帧 i 的 x0 预测
             A = dx/dx0（auto：用最小二乘从 x_in 与上一步 x0 直接量出来）
    窗 k forward 之后：
        记录本窗**拥有**的帧（`window.original_indices`）在本步的 x0，供后续窗口使用
        execute 结束时，把「本步的融合 x0」缓存到 handler 上，供【下一步】当基准

**为什么不沿用"带新噪声的再噪"**（旧行为，现由 `roll_noise=legacy_fresh` 保留）
旧公式 `(1-β)·x + β·[(1-σ)·x0_prev + σ·ε_new]` 有两个副作用：
  1. 把该帧的噪声换成「混合噪声」，幅值压到 √((1-β)^2+β^2) 倍（β=0.5 时最低 0.707）；
  2. σ→1 时 `(1-σ)→0`，提交退化成「纯新噪声」—— 既没传内容、又打乱了噪声，
     而第 1 步恰恰是 4 步蒸馏里最重要的一步。
后果：「动作连贯」与「背景色一致」互相牺牲（实测 β=0.3 动作好/背景差，β=1 背景好/动作散）。
`content_delta` 把两者**解耦**：内容对齐与噪声共享可以同时 100% 拿到。

天然零改动的三种情况
--------------------
· σ=1（片段第 1 步）：A 里含 (1-σ) 因子（auto 量出来就是 0）=> 不动
· 片段首步 / 新 phase：σ 回跳 => 跨步缓存自动失效 => 无基准 => 不动
  （这条保证长流程，例如「25 段 × 4 步」，不会跨片段串味）
· `roll_beta=0` / `mode=off` / `enabled=关` => 完全透传

关键事实（本机代码已验证）
--------------------------
· `calc_cond_batch` 返回的是 **x0（denoised）**，不是 v/ε（`model_base.py:258` 已做换算）
  => 不需要自己反推 x0。
· ComfyUI 侧 `handler.execute(calc_cond_batch, ...)` 的这个 `calc_cond_batch` 参数
  **就是「单个窗口的 forward」**，每窗调用一次 => 所有逐窗干预都在这里挂钩。
· `window.get_tensor()` 用列表索引 => 返回**副本**，改 `sub_x` 不会污染 `x_in`。
· `handler.execute()` 的**返回值就是全长的融合 x0**（`conds_final`）=> 跨步基准不用另算。
· `sample_lcm` 每步是 `x = (1-σ)·x0 + σ·新噪声`（`CONST.noise_scaling` 证实）
  => 单阶段下 A = 1-σ；而 `WanSCAIL2PhaseSampler` 的 phase_schedule 第 2..N 段用
  `noise=0` + `latent_image`，实测 A ≈ (1-σ)^2 => 所以 A 必须**量出来**而不是硬编码。

与 `overlap_scope` 的关系
------------------------
· `single`（重叠只读化）：本窗只读帧 = 前窗拥有帧，`roll_scope=readonly` 即可，每帧唯一作者。
· `average`（重叠共写 + 加权平均）：重叠帧同时属于两窗 => 必须 `roll_scope=both` 才有效；
  此时 rolling 让两窗**先对齐**，平均就从「破坏性（半透明）」变成
  「建设性（降噪 + 平滑过渡）」。实测 `average + rolling` 优于 `single + rolling`
  （后者没有过渡 => 颜色/曝光会硬跳）。

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
import math
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
    beta: float = 0.5               # 提交强度：0=透传，1=内容完全用前窗本步 x0
    scope: str = "readonly"         # readonly=只改本窗只读帧 / both=也改本窗拥有的重叠帧
    seed: int = 0                   # 再噪用的确定性噪声种子（仅 legacy_fresh 用）
    noise: str = "content_delta"    # content_delta（默认：不引入任何新噪声）/ legacy_fresh（旧行为）
    content_scale: str = "auto"     # auto（最小二乘估计 dx/dx0）/ one_minus_sigma / one


# 跨步 x0 缓存挂在 handler 上（与 Context Window Anchor 的 x0 缓存同一套路）
_X0_CACHE_ATTR = "_roll_x0_cache"


def _handler_of(model_options):
    try:
        return (model_options or {}).get("context_handler", None)
    except Exception:
        return None


def _read_base_x0(model_options, sigma: float):
    """取「上一步的融合 x0」（= 采样器真正拿去构造当前 x 的那个基准）。

    σ 在一个片段内单调下降；**新片段 / 新 phase 会让 σ 跳回高位** =>
    用 `sigma >= cached_sigma` 判定缓存过期。这条规则保证长流程（例如 25 段 × 4 步）
    里「跨片段不串味」—— 每段第一步必然失效 => 那一步天然零改动。
    """
    h = _handler_of(model_options)
    if h is None:
        return None
    cache = getattr(h, _X0_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        return None
    cs = cache.get("sigma")
    x0 = cache.get("x0")
    if cs is None or x0 is None:
        return None
    try:
        if float(sigma) >= float(cs):
            return None
    except Exception:
        return None
    return x0


def _write_base_x0(model_options, sigma: float, result) -> None:
    """把本次 execute 的融合 x0（= 采样器构造下一步 x 所用的值）缓存到 handler 上。"""
    h = _handler_of(model_options)
    if h is None:
        return
    tensor = result
    if isinstance(result, (list, tuple)):
        tensor = result[0] if len(result) else None
    if not torch.is_tensor(tensor):
        return
    try:
        setattr(h, _X0_CACHE_ATTR, {"sigma": float(sigma), "x0": tensor.detach()})
    except Exception:
        logging.debug("[RollingWindows] 缓存上一步 x0 失败（忽略）", exc_info=True)


def _estimate_content_scale(x_in, base, sigma: float, cfg: RollingConfig):
    """估计 A = dx/dx0，使 `x_in ≈ A·base + 噪声`。

    为什么不能硬编码 `1-σ`：`x` 与 `x0` 的关系**依赖采样器与阶段** ——
      · euler / lcm（单阶段、真噪声）：`x = (1-σ)·x0 + σ·ε`            => A = 1-σ
      · `WanSCAIL2PhaseSampler` 的 phase_schedule 第 2..N 段
        （`noise=0` + `latent_image=上一段输出`）：`x = (1-σ)^2·x0 + ...` => A = (1-σ)^2
    所以这里用**最小二乘投影**直接量出来：

        A ≈ <x_in, base> / <base, base>

    噪声与 base 独立，且全长序列有 400 万+ 个元素 => 估计误差约 1e-3（实际等于精确）。
    这样「换了采样器 / 换了阶段」都不用改公式。
    """
    mode = str(getattr(cfg, "content_scale", "auto") or "auto")
    if mode == "one_minus_sigma":
        return max(0.0, 1.0 - float(sigma))
    if mode == "one":
        return 1.0
    try:
        if not (torch.is_tensor(x_in) and torch.is_tensor(base)):
            raise ValueError("x_in/base 不是张量")
        if tuple(x_in.shape) != tuple(base.shape):
            raise ValueError("x_in/base 形状不一致")
        a = x_in.detach().float()
        b = base.detach().float()
        denom = float((b * b).sum())
        if denom <= 1e-12:
            raise ValueError("base 范数为 0")
        est = float((a * b).sum()) / denom
        if math.isnan(est) or math.isinf(est) or est <= 0.0:
            raise ValueError("估计值非法：%r" % (est,))
        return min(1.0, est)           # A 不可能大于 1（x 里 x0 的系数）
    except Exception:
        logging.debug("[RollingWindows] content_scale 自动估计失败，回退 1-σ", exc_info=True)
        return max(0.0, 1.0 - float(sigma))


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


def _global_frame(tensor, dim: int, idx: int):
    """取全长张量在全局帧 `idx` 上的切片（保持 dim 维）。越界/非张量 => None。"""
    if not torch.is_tensor(tensor) or tensor.ndim <= dim:
        return None
    if idx < 0 or idx >= tensor.shape[dim]:
        return None
    return tensor[_frame_slice(tensor, dim, idx)]


def _apply_commit(sub_x, pos_of, own, state, sigma, cfg: RollingConfig):
    """把「已被前面窗口写过的帧」在本窗输入里做提交。两种模式：

    · `noise="content_delta"`（**默认**）：
          Δ = x0_prev - 上一步融合 x0
          x ← x + A · β · Δ          （A = dx/dx0，自动估计）
      **不引入任何新噪声** —— 帧原有的噪声实现被完整保留（σ·ε 一字不动），
      只把「内容」朝前窗平移。σ=1（或 A≈0）时天然零改动。
    · `noise="legacy_fresh"`（保留旧行为，供 A/B）：
          x ← (1-β)·x + β·[(1-σ)·x0_prev + σ·ε_new]
      代价：该帧噪声被换成"混合噪声"，幅值被压到 √((1-β)^2+β^2) 倍
      （β=0.3 → 0.762；β=0.5 → 0.707；β=1 → 1.0）。
    """
    committed = state["committed"]
    if not committed:
        return sub_x
    targets = [i for i in pos_of
               if i in committed and (cfg.scope == "both" or i not in own)]
    if not targets:
        return sub_x
    dim = state["dim"]
    base = state.get("base_x0")
    scale = state.get("content_scale")
    if cfg.noise == "content_delta":
        if base is None or scale is None or float(scale) <= 0.0:
            return sub_x          # 第一步 / 新片段 / 拿不到基准 => 天然零改动
    out = sub_x.clone()
    for i in targets:
        sl = _frame_slice(out, dim, pos_of[i])
        ref = out[sl]
        prev = committed[i].to(ref.device, ref.dtype)
        if cfg.noise == "content_delta":
            b = _global_frame(base, dim, i)
            if b is None:
                continue
            delta = prev - b.to(ref.device, ref.dtype)
            out[sl] = ref + (float(scale) * cfg.beta) * delta
        else:
            eps = _seeded_noise_like(ref, cfg.seed, state["step"], i)
            target = (1.0 - sigma) * prev + sigma * eps
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
            # 跨步基准：上一步的融合 x0（σ 回跳 => 新片段 => 自动失效）
            base = _read_base_x0(model_options, sigma)
            # A = dx/dx0；每步只算一次（全长张量上统计量最稳），与窗口数无关
            scale = _estimate_content_scale(x_in, base, sigma, cfg) if base is not None else None
            # 步内状态：每次 execute（= 每个去噪步）重置一次
            state = {"step": _sigma_step_key(sigma), "dim": dim, "committed": {},
                     "windows": 0, "hits": 0, "base_x0": base, "content_scale": scale}

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
            # 把本次的融合 x0 存起来，供【下一步】当基准（下一片段第一步会因 σ 回跳而自动失效）
            _write_base_x0(model_options, sigma, result)
            logging.info("[RollingWindows] σ=%.3f 步完成：%d 窗，提交命中 %d 帧"
                         "（mode=%s noise=%s beta=%.2f scope=%s A=%s%s）",
                         sigma, state["windows"], state["hits"],
                         cfg.mode, cfg.noise, cfg.beta, cfg.scope,
                         ("%.4f" % scale) if scale is not None else "n/a",
                         "（无基准：片段首步 / 缓存已失效）" if base is None else "")
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
                    "tooltip": "再噪用的确定性噪声种子（**仅 noise=legacy_fresh 时有效**）。\n"
                               "Deterministic noise seed (only used when noise=legacy_fresh)."}),
                "roll_noise": (["content_delta", "legacy_fresh"], {"default": "content_delta",
                    "tooltip": "content_delta（默认，推荐）= 只把「内容」朝前窗平移，**帧原有的噪声一字不动**；\n"
                               "    σ=1、片段首步、拿不到基准时天然零改动\n"
                               "    => 「内容对齐」与「噪声共享」可以同时拿到（两者不再互相牺牲）。\n"
                               "legacy_fresh = 旧行为：把该帧换成 (1-σ)·前窗x0 + σ·新噪声\n"
                               "    （代价：该帧噪声被换成混合噪声，幅值压到 0.76~0.71 倍）。\n"
                               "content_delta = move content only, never touch the noise (recommended)."}),
                "roll_content_scale": (["auto", "one_minus_sigma", "one"], {"default": "auto",
                    "tooltip": "x 里 x0 的系数 A（即 dx/dx0），决定「内容平移」该按多大比例施加。\n"
                               "auto（推荐）= 用最小二乘从 x_in 与上一步 x0 直接量出来\n"
                               "    => 自动适配单阶段采样器（A=1-σ）与 phase_schedule 后续阶段（A=(1-σ)^2）；\n"
                               "one_minus_sigma = 固定 1-σ；one = 固定 1.0。\n"
                               "Auto-estimated content coefficient."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_rolling"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "SCAIL-2 滚动窗口：在同一步内让后面的窗口看到前面窗口【本步】刚产出的 x0。\n"
        "默认 noise=content_delta —— 只平移「内容」、**完全不动该帧的噪声**；\n"
        "σ=1 与片段首步天然零改动（因此跨片段不会串味）。\n"
        "需接在上下文窗口节点之后；与 overlap_scope=average 搭配时建议 roll_scope=both。\n"
        "roll_mode=off / roll_beta=0 / enabled=关 => 完全透传。"
    )

    def apply_rolling(self, model, enabled=True, roll_mode="intra_step_x", roll_beta=0.5,
                      roll_scope="readonly", roll_seed=0,
                      roll_noise="content_delta", roll_content_scale="auto"):
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
                            scope=str(roll_scope), seed=int(roll_seed),
                            noise=str(roll_noise), content_scale=str(roll_content_scale))

        model = model.clone()
        # 浅拷贝：包装链会为副本重建并重放既有包装（handler_wrap 保证原 handler 不被污染）
        new_handler = copy.copy(handler)
        # 新运行必须从零开始：清掉可能被浅拷贝带过来的旧「上一步 x0」基准
        try:
            delattr(new_handler, _X0_CACHE_ATTR)
        except AttributeError:
            pass
        dim = int(getattr(new_handler, "dim", 2) or 2)
        applied = _wrap_install(new_handler, _WRAP_OWNER, {"execute": _make_execute(cfg, dim)})
        model.model_options["context_handler"] = new_handler

        logging.info("[RollingWindows] 已启用：mode=%s noise=%s beta=%.2f scope=%s seed=%d scale=%s（dim=%d）",
                     cfg.mode, cfg.noise, cfg.beta, cfg.scope, cfg.seed, cfg.content_scale, dim)
        logging.info("[RollingWindows] 包装链：%s（本次安装 %s）", _wrap_describe(new_handler), applied)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanSCAIL2RollingWindows": WanSCAIL2RollingWindowsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAIL2RollingWindows": "Wan SCAIL-2 Rolling Windows (步内滚动)",
}
