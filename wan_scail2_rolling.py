"""Wan SCAIL-2 滚动窗口（实验性 / S2）—— **步内 x 前移**。

要解决的问题
------------
上下文窗口 + 融合（Gen-L-Video 式：每步各窗独立 forward，再按权重平均）在相邻窗
「对同一全局帧给出两份不同 x0」时会平均成**重影 / 动作发软**。`overlap_scope=single`
（重叠只读化）已经能让每帧只有一个作者，但**后窗看到的仍是「上一步的融合值」**，
它在本步的 forward 里并不知道前窗**本步**已经决定了什么。

本节点做的事（步内 x 前移；**2026-10-11 精简：参数 11 -> 5；2026-10-15 面板再精简为 3 个控件**）
------------------------------------------------------------------------
在**同一个去噪步内**，让后面的窗口看到前面窗口**本步刚产出的 x0**：

    窗 k forward 之前（对待提交的帧 i）：
        target_i  = (1 - σ) · x0_prev[i] + σ · ε(seed, σ, i)   ← ε 按种子确定 => 两窗共享同一份噪声
        x_sub[i]  <- (1 - β_eff) · x_sub[i] + β_eff · target_i
        其中 x0_prev[i] = 前窗**本步**对帧 i 的 x0 预测（不是上一步的融合值）
             β_eff     = roll_beta（本窗拥有的帧）/ roll_beta_inherited（继承帧），再乘 curve 衰减
    窗 k forward 之后：
        记录本窗**拥有**的帧（`window.original_indices`）在本步的 x0，供后面的窗口使用

σ→1（片段首步）时 (1-σ)→0 => 提交值退化成「两窗共享的那块噪声」（内容项天然为 0）。
「动作连贯」与「背景色一致」的取舍由 roll_beta / roll_beta_inherited 调（实测 0.45 + 0.6 最健康）。

历史（已删除的实现，仅备查）：`content_delta`（只平移内容、不换噪声，A=dx/dx0 由最小二乘自动估计）
曾作为默认实现（提交 02366db），同 seed 实测更差 => 已整体回退删除。随之删除的还有
`mode` / `noise` / `content_scale` / `w_content` / `w_noise` / `roll_seed` / `roll_beta_step1` 等控件，
以及只喂日志、**从不参与计算**的「跨步 x0 缓存」。

天然零改动的情况
----------------
· σ=1（片段第 1 步）：(1-σ)=0 => 提交值只剩「两窗共享的那块噪声」，内容项天然为 0
· `roll_beta=0` 且 `roll_beta_inherited=-1` => 完全透传（原样返回同一个 model，不 clone、不装包装器）
· `enabled=关` => 完全透传
· β=0 但 `roll_beta_inherited>=0` => **按设计仍生效**：只把「继承帧（只读前文）」单作者化，本窗拥有帧不动

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
`enabled=关`、或（`roll_beta=0` 且 `roll_beta_inherited=-1`）=> 完全透传（等价于没装本节点）。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch

try:
    from .handler_wrap import install as _wrap_install
except ImportError:                                     # pragma: no cover - 顶层导入（自测脚本）
    from handler_wrap import install as _wrap_install


# 包装链里的 owner 标识（与 ContextWindowAnchor 平级，互不覆盖）
_WRAP_OWNER = "RollingWindows"


@dataclass
class RollingConfig:
    """滚动提交配置。

    ⚠️ 这里的默认值是**内部 / 自测**用的兜底档（`curve="flat"` = 不做随步衰减）；
       节点 `WanSCAIL2RollingWindowsNode` 每次都显式传入实测档（`scope="both"` / `curve="late"`）
       覆盖它们 => 改这里的默认值只影响直接构造本对象的调用方（含自测），不影响节点行为。
    """
    beta: float = 0.45              # 提交强度：0=透传；1=该帧输入完全换成「前窗本步 x0 按当前 σ 再噪」
    beta_inherited: float = -1.0    # 继承帧（= 本窗的只读前文）的单独强度；-1 = 跟随 beta
    curve: str = "flat"             # 随步 β 曲线：flat（兜底=恒 β）/ late（早期弱、后期强）/ mid（前半程不提交）
    scope: str = "both"             # readonly=只改本窗只读帧 / both=也改本窗拥有的重叠帧
    seed: int = 0                   # 再噪用的确定性噪声种子（内部固定，不再暴露成参数）

# 2026-10-15 删除：跨步 x0 缓存（`_X0_CACHE_ATTR` / `_handler_of` / `_read_base_x0` / `_write_base_x0`）。
#   它被 `execute` 读出并写进 `state["base_x0"]`，但**从未参与任何计算**（只喂「（无基准…）」那条日志），
#   真正传递信息的「前窗本步 x0」走的是 `state["committed"]`（见 _record_commit / _apply_commit）。
#   删掉它 = 少一次「每步一个全长张量 detach + 挂在 handler 上」的开销，逐位不改结果（自测已覆盖）。


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
    """把「已被前面窗口写过的帧」在本窗输入里做提交。

        target = (1 - sigma) * x0_prev + sigma * eps     # eps 按 (seed, sigma, 帧) 确定
        x      = (1 - beta) * x + beta * target

    · sigma 接近 1（片段首步）：target 基本等于 eps => 两窗共享同一份噪声（内容项天然为 0）。
    · beta_eff：本窗拥有的帧用 `roll_beta`；「继承帧（只读前文）」在 `roll_beta_inherited >= 0` 时单独取值。
    · 再乘 `curve` 衰减（late = σ 大时不提交、σ<=0.5 满量）。
    """
    committed = state["committed"]
    if not committed:
        return sub_x
    targets = [i for i in pos_of
               if i in committed and (cfg.scope == "both" or i not in own)]
    if not targets:
        return sub_x
    dim = state["dim"]
    out = sub_x.clone()
    for i in targets:
        sl = _frame_slice(out, dim, pos_of[i])
        ref = out[sl]
        prev = committed[i].to(ref.device, ref.dtype)
        beta_i = float(cfg.beta)
        if float(getattr(cfg, "beta_inherited", -1.0)) >= 0.0 and i not in own:
            beta_i = float(cfg.beta_inherited)
            state["hits_inherited"] = int(state.get("hits_inherited", 0)) + 1
        # ★ roll_beta_curve：早期少干预（预测还糙，融合只会抹细节）→ 后期让两窗对齐
        _curve = str(getattr(cfg, "curve", "flat"))
        if _curve == "late":
            beta_i *= max(0.0, min(1.0, (1.0 - float(sigma)) / 0.5))   # σ>=1 => 0；σ<=0.5 => 满量
        elif _curve == "mid":
            beta_i *= 0.0 if float(sigma) > 0.5 else 1.0               # 前半程不提交
        if beta_i <= 0.0:
            continue
        eps = _seeded_noise_like(ref, cfg.seed, state["step"], i)
        target = (1.0 - sigma) * prev + sigma * eps
        out[sl] = (1.0 - beta_i) * ref + beta_i * target
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
    """工厂：在 `execute` 外层包一圈「步内 x 前移」（`calc_cond_batch` 就是单窗 forward）。"""
    def factory(next_call):
        def execute(calc_cond_batch, model, conds, x_in, timestep, model_options):
            if float(cfg.beta) <= 0.0 and float(cfg.beta_inherited) < 0.0:
                return next_call(calc_cond_batch, model, conds, x_in, timestep, model_options)
            try:
                sigma = float(timestep.reshape(-1)[0])
            except Exception:
                sigma = 0.0
            state = {"step": _sigma_step_key(sigma), "dim": dim, "committed": {},
                     "windows": 0, "hits": 0, "hits_inherited": 0}

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
            logging.info("[RollingWindows] σ=%.3f 步完成：%d 窗，提交命中 %d 帧（其中继承帧 %d）"
                         "（beta=%.2f beta_inherited=%.2f scope=%s curve=%s）",
                         sigma, state["windows"], state["hits"], state["hits_inherited"],
                         cfg.beta, cfg.beta_inherited, cfg.scope, cfg.curve)
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
                    "tooltip": "总开关；关闭时模型直接透传（等价于不装本节点）。Master switch: off = exactly like not installing this node."}),
                "roll_beta": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "提交强度 β（**主力旋钮**）：0 = 透传；1 = 该帧输入完全换成「前窗本步 x0 按当前 σ 再噪」。"
                               "实测推荐 0.45~0.50（0.30 / 0.60 / 1.00 都出现过缝上 26~40px 瞬移）。Commit strength; measured sweet spot 0.45~0.50."}),
                "roll_beta_inherited": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "【只作用于「继承帧」（本窗的只读前文）的提交强度】-1 = 跟随 roll_beta；>=0 = 单独指定。同 seed 实测：0.6 => 质心 10.3px；-1 => 21.3px => 0.6 明显更好（默认 0.6）。\n"
                               "Commit strength for inherited frames only (measured: 0.6 much better than -1)."}),
                # 2026-10-15：roll_scope / roll_beta_curve 两个控件的**控件已移除**，值固定在 `apply_rolling`
                #   的签名默认里（both / late，均为实测档）。==> 面板现只有 enabled / roll_beta / roll_beta_inherited。
                # 删掉的两个控件（原值备查）：
                #   roll_scope      = ["readonly", "both"]，默认 both（readonly = 只改本窗只读帧）
                #   roll_beta_curve = ["flat", "late", "mid"]，默认 late（flat = 恒 β / mid = 前半程不提交）
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_rolling"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "SCAIL-2 滚动窗口：在同一步内让后面的窗口看到前面窗口【本步】刚产出的 x0。",
        "提交公式：x = (1-β)·x + β·[(1-σ)·前窗本步x0 + σ·ε]（ε 按 seed 确定 ⇒ 两窗共享同一份噪声）。",
        "2026-10-11 精简：参数 11 -> 5（删掉 mode / noise / content_scale / w_content / w_noise / beta_step1；",
        "content_delta 与 w_noise<1 实测更差 => 固定「再噪」公式，且「第 1 步不提交」由 (1-σ) 天然给出）。",
        "2026-10-15 面板再精简：只留 enabled / roll_beta / roll_beta_inherited"
        "（roll_scope 固定 both、roll_beta_curve 固定 late；参数保留在签名里，去掉控件不改行为）。",
        "需接在上下文窗口节点之后；roll_scope=both 与 overlap_scope=average 搭配。",
        "enabled=关 /（roll_beta=0 且 roll_beta_inherited=-1）=> 完全透传。"
    )

    # 2026-10-15：roll_scope / roll_beta_curve 的**控件已从面板移除**，值固定在下方的签名默认里
    #   （both / late，均为实测档）。理由：overlap_scope 已在锚定侧硬固定为 average => 只有 both 有意义；
    #   curve=late 是正收益（内容异常 -29%）且早已是默认。参数本身保留 => 代码/自测仍可按任意档调用，
    #   节点行为与「去掉控件前」逐位一致。
    def apply_rolling(self, model, enabled=True, roll_beta=0.45,
                      roll_scope="both", roll_beta_inherited=0.6, roll_beta_curve="late"):
        b_inh = float(roll_beta_inherited) if roll_beta_inherited is not None else -1.0
        if not enabled or (float(roll_beta) <= 0.0 and b_inh < 0.0):
            logging.info("[RollingWindows] 未启用（enabled=%s beta=%.3f beta_inherited=%.2f），模型直接透传",
                         enabled, roll_beta, b_inh)
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[RollingWindows] 模型上没有 context_handler：请把本节点接在上下文窗口节点"
                "（`Wan SCAIL-2 Context Windows` 或 `Custom Context Windows (Manual)`）之后。")

        cfg = RollingConfig(beta=float(roll_beta), scope=str(roll_scope),
                            beta_inherited=b_inh, seed=0,
                            curve=str(roll_beta_curve))
        model = model.clone()
        # 浅拷贝：包装链会为副本重建并重放既有包装（handler_wrap 保证原 handler 不被污染）
        new_handler = copy.copy(handler)
        dim = int(getattr(new_handler, "dim", 2) or 2)
        applied = _wrap_install(new_handler, _WRAP_OWNER, {"execute": _make_execute(cfg, dim)})
        model.model_options["context_handler"] = new_handler

        logging.info("[RollingWindows] 已启用：beta=%.2f beta_inherited=%.2f scope=%s noise_seed=%d（dim=%d）",
                     cfg.beta, cfg.beta_inherited, cfg.scope, cfg.seed, dim)
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanSCAIL2RollingWindows": WanSCAIL2RollingWindowsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSCAIL2RollingWindows": "步内滚动",
}
