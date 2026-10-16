"""锚定 + 步内滚动（极简整合节点）—— 给「不想了解原理」的人用的一键节点。

它做的事
--------
把「窗口锚定」（context_window_anchor.py）与「步内滚动」（wan_scail2_rolling.py）
按**固定的实测档**串起来，面板上**只留一个 `enabled` 开关**：

    [模型] -> 窗口锚定（锚定帧 / 只读前文 / 长程记忆组）
           -> 步内滚动（步内 x 前移）
           -> 采样器

· 实现是**组合**，不复制逻辑：内部调用两个节点的 `apply_anchor` / `apply_rolling`
  => 包装链顺序与手工接两个节点完全一致（`[ContextWindowAnchor -> RollingWindows]`），
     以后两个节点各自的修复会自动继承过来。
· 所有设置项固定在下面的 `FIXED_ANCHOR` / `FIXED_ROLL` 里；要改行为就改那两处
  （改完**重启 ComfyUI**才生效：锚定的 RoPE 位置补丁是进程级 monkey patch）。
· `enabled=False` => 等价于「锚定节点 enabled=关」：清空锚定池 + 模型原样透传。

为什么可以这么极简
------------------
被固定掉的每一项，台账里都是「实测更差」或「与默认同级」（依据写在每个值旁边；
`md/_FACTS.md` §K = 实际在跑的基线，`md/SCAIL_Rolling_Handoff.md` §10 = 逐项实测）。
真正还值得调的四个量（`head_context_frames` / `memory_frames` / `roll_beta` /
`roll_beta_inherited`）仍可在老的两个节点上调 —— 它们保留了全部控件：

    **本节点 = 日常生产；老节点 = 调参实验。**

依赖与接线
----------
· 必须接在上下文窗口节点（`Wan SCAIL-2 Context Windows` / `Custom Context Windows (Manual)`）之后。
· 锚定素材是**模块级状态** => 一个工作流里只接一个锚定类节点（本节点与老锚定节点二选一）。
· 非 SCAIL 模型会自动降级（记忆组自动关闭、锚帧改走「同时注入待生成序列」），与老节点一致。
· 诊断日志：本节点没有 debug 控件 —— 在节点目录放一个 `CWA_DEBUG.on` 空文件即可开启（免重启）；
  更详细的逐流/逐帧日志再设环境变量 `CWA_DEBUG_VERBOSE=1`（需重启）。
"""

from __future__ import annotations

import logging

try:
    from .context_window_anchor import WanContextWindowAnchorNode
    from .wan_scail2_rolling import WanSCAIL2RollingWindowsNode
except ImportError:                                     # pragma: no cover - 顶层导入（自测脚本）
    from context_window_anchor import WanContextWindowAnchorNode
    from wan_scail2_rolling import WanSCAIL2RollingWindowsNode


# ============================== 固定档（唯一的行为定义处） ==============================
# 出处：md/_FACTS.md §K（作者实际在跑的基线）+ md/SCAIL_Rolling_Handoff.md §10（逐项实测）。
# 改这里 = 改所有工作流的行为；改完重启 ComfyUI。

FIXED_ANCHOR = {
    # ---- 三个真旋钮的基线值 ----
    "head_context_frames": 2,       # 只读前文 2 latent 帧（6 帧会让 token 达 27 > 训练 21 并出现肢体变形）
    "memory_frames": 4,             # 长程记忆组 4 帧
    "memory_stride": 0,             # 0 = 自动（两段式：先贴窗口，再覆盖片段开头）
    # ---- 已判决的开关（都是「实测更差」或「与默认同级」）----
    "anchor_scope": "fresh_only",   # 跳过含「接续帧钉住区」的窗口（它有真实已知前缀，再叠头会互相干扰）
    "drive_memory": "off",          # 16ch 驱动流不收记忆帧：坏种子好 4.9px、好种子差 0.7px（噪声级）
    "anchor_slot": "aligned",       # 锚帧时间码语义；目视偏好，与 context 同级
    "driving_only": False,          # 锚帧前插驱动流 + 待生成序列（逐帧配对）；作者实测 True 更差
    "anchor_fixed": False,          # 锚帧随窗口各自演化（与 anchor_image_blank 同时开时本就无意义）
    "anchor_image_blank": True,     # 待生成序列里的锚帧只放噪声占位（不泄漏源人物画面）：19.9 -> 15.7px
    "shared_head": "off",           # 各窗各自生成锚帧内容
    "overlap_scope": "average",     # 重叠共写 + 加权平均（模块内也已硬固定，改这里无效）
    "memory_position": "head",      # 规范物理顺序 [锚][记忆][本窗]
    "memory_order": "sparse",       # 记忆帧排布（当前仅一档）
    "x_prefix": "anchor_only",      # x 侧只放锚帧前缀
    "debug_log": False,             # 不开控件也能开诊断：节点目录放 CWA_DEBUG.on 即可（免重启）
}

FIXED_ROLL = {
    "roll_beta": 0.45,              # 提交强度：实测最健康档（0.30 / 0.60 / 1.00 都出现缝上 26~40px 瞬移）
    "roll_beta_inherited": 0.6,     # 继承帧（只读前文）单独取 0.6：10.3px vs 跟随 beta 的 21.3px
    "roll_scope": "both",           # overlap_scope 固定 average => 只有 both 有意义（共写重叠帧也要改）
    "roll_beta_curve": "late",      # σ 大时不提交、σ<=0.5 满量：内容异常 -29%
}


class WanAnchorRollingNode:
    """锚定 + 步内滚动（极简）。必须接在上下文窗口节点之后。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（`Wan SCAIL-2 Context Windows` 或 `Custom Context Windows (Manual)` 的输出）。Model that already has a context windows node applied."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关；关闭时模型直接透传（并清空锚定池）。Master switch.\n"
                               "本节点只有一个设置项：只读前文 2 / 记忆组 4（自动间距）/ 锚定帧数量跟随输入 /\n"
                               "提交强度 0.45 / 继承帧 0.6，全部按实测基线固定。\n"
                               "需要调这些量时，请改用老节点「窗口锚定」/「步内滚动」（保留全部控件）。"}),
            },
            "optional": {
                "anchor_frames": ("IMAGE", {"tooltip": "【基准帧】驱动/pose 视频里你想当统一基准的那几帧（例如整段视频开头几帧）。**给多少就用多少**（没有数量参数）：会 VAE 编码成 latent，给每个窗口一份共同的「前文/基准」运动。不接=不做锚定（仍有只读前文与长程记忆组）。\nAnchor basis frames (driving/pose video): how many you feed is how many are used. Leave unconnected to skip anchoring."}),
                "anchor_mask_frames": ("IMAGE", {"tooltip": "【仅 SCAIL-2】与 anchor_frames 同帧号的彩色遮罩帧（内部转成 28 通道，不需要 VAE）。SCAIL-2 下建议一起接，否则驱动遮罩与驱动视频的时间基准可能不一致。\nSCAIL-2 only: colored mask frames matching anchor_frames (no VAE needed)."}),
                "vae": ("VAE", {"tooltip": "【接了 anchor_frames 时必接】与 conditioning 节点**同一个** Wan VAE（用于把基准帧编码成 latent；接错 VAE 会得到无意义的条件）。\nSame Wan VAE as the conditioning node; required when anchor_frames is connected."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "「窗口锚定 + 步内滚动」的一键整合节点（面板只有 enabled 一个设置项）。\n"
        "内部 = 窗口锚定（只读前文 2 / 长程记忆组 4 / 锚定帧跟随 anchor_frames）"
        " -> 步内滚动（beta 0.45 / 继承帧 0.6 / 曲线 late / scope both），"
        "全部按实测基线固定；要调就用老节点「窗口锚定」/「步内滚动」。\n"
        "必须接在上下文窗口节点之后；一个工作流只接一个锚定类节点。\n"
        "One-click Anchor + Intra-step Rolling with a single switch (all settings fixed to the measured baseline)."
    )

    def apply(self, model, enabled=True, anchor_frames=None, anchor_mask_frames=None, vae=None):
        if not enabled:
            logging.info("[AnchorRolling] 未启用（enabled=False），模型直接透传（锚定池已清空）")
            # 复用锚定节点的「关」分支：清空锚定池 + 原样返回同一个 model（不 clone、不装包装器）
            return WanContextWindowAnchorNode().apply_anchor(model, enabled=False)

        (model,) = WanContextWindowAnchorNode().apply_anchor(
            model, enabled=True, **FIXED_ANCHOR,
            anchor_frames=anchor_frames, anchor_mask_frames=anchor_mask_frames, vae=vae)
        (model,) = WanSCAIL2RollingWindowsNode().apply_rolling(model, enabled=True, **FIXED_ROLL)

        logging.info("[AnchorRolling] 已启用：锚定（只读前文=%d / 记忆组=%d / 间距=%s）"
                     " + 滚动（beta=%.2f / 继承帧=%.2f / scope=%s / 曲线=%s）",
                     FIXED_ANCHOR["head_context_frames"], FIXED_ANCHOR["memory_frames"],
                     "自动" if FIXED_ANCHOR["memory_stride"] == 0 else FIXED_ANCHOR["memory_stride"],
                     FIXED_ROLL["roll_beta"], FIXED_ROLL["roll_beta_inherited"],
                     FIXED_ROLL["roll_scope"], FIXED_ROLL["roll_beta_curve"])
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanAnchorRolling": WanAnchorRollingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnchorRolling": "锚定+步内滚动（极简）",
}
