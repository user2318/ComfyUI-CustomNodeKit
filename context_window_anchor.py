# ==============================
# Context Window Anchor —— 上下文窗口「统一基准锚定 + 只读上下文 + 长程记忆组」（实验性）
#
# 背景（为什么需要）：
#   animate2 / SCAIL-2 没有骨骼这类显式空间锚；窗口化后每个窗口的 RoPE 都从 0 重新计数
#   （不含绝对时间），而且 SCAIL-2 的「驱动帧 <-> 生成帧」是按 RoPE 同位**逐帧刚性配对**的
#   （comfy/ldm/wan/model.py:1821-1838：pose j 与 video j 的 t 相同）。
#   于是每个窗口只能靠「自己那段驱动内容」自己推断基准 => 相邻窗口在重叠区给出两份不同的
#   动作预测，而融合是**加权平均** => 重影 / 动作发软 / 个别窗口不准。
#
# 本节点给上下文窗口补三样东西（都是「内容层」的干预，位置编码补不了这些）：
#   1) 锚定帧（anchor_frames，可选）：所有窗口共享同一段「前文/基准」运动。
#      ⚠️ anchor_frames 来自**驱动视频**（源画面）。它在待生成序列里只当「占位」用，内容由
#         模型自己长出来（见 anchor_fixed 说明）；若把它当「已知干净帧」钉住，就会把
#         **驱动者形象**带进画面（即历史文档里的「驱动者形象伪影」）。
#      - driving_only=开（默认，SCAIL-2）：锚帧作为**额外的驱动 token 追加在驱动流尾部**，
#        再由 RoPE 补丁把它们放到「本窗口之前」的位置 => 既有几何完全不变、**不产生帧错位**
#        （旧实现把锚帧前插在驱动流最前，会让整段生成相对驱动视频滞后 K 帧 —— 已修）；
#      - driving_only=关：锚帧前插到驱动流 + 待生成序列（与视频帧逐帧配对，采样后裁掉），
#        最接近训练语义（训练时每块头部是上一块的输出）。
#   2) 只读上下文（head_context_frames，**单位 = latent 帧**）：每个窗口头部各看几帧真实邻居，
#      参与注意力但**不参与融合**（融合前被裁掉）。配 overlap=0 时每一帧只有一个「作者」，
#      从根上避免「两份预测被平均」。
#      （尾段 tail_context_frames 已移除：它的预测永远被裁掉、且那些帧归后面的窗口负责
#       => 机制上不产生可见效果；内部能力保留但恒为 0。）
#   3) 长程记忆组（memory_frames，**单位 = latent 帧**）：取本窗口**之前**的历史驱动帧作为额外
#      token 追加；采样间距由 memory_stride 决定（0=自动：让 N 个点覆盖「片段开头 → 本窗口
#      之前」的全部历史；>0=固定）。RoPE 位置如实编码那个真实间距（不会被当成相邻帧）。
#
# ⚠️ 帧单位：anchor_frames 是**像素帧**（给多少用多少，内部按 ((T-1)//4)+1 转 latent）；
#    head_context_frames / memory_frames / memory_stride 都是**latent 帧**。
#
# 其它：halo 机制（apply_window_extras）负责改写 index_list 与头/尾只读段；
#       融合前裁剪由 Custom_context.combine_context_window_results 按
#       「前缀 + 锚定 + 头段」/「尾段」执行。
#       锚定素材/配置是模块级状态，**一个工作流接一个本节点**。
# 不接 anchor_frames、也不开 2)/3) => 完全等价于不装本节点。
#
# ⚠️ latent 空间（2026-10 修）：锚定池来自 `vae.encode()`（**原始** latent），而模型真正消费的是
#    `model_conds` 里的模型空间键（`pose_latents`）与采样中的 x —— 两者都是**归一化** latent
#    （`process_latent_in` / `Wan21.process_in = (x-mean)/std`）。此前直接把原始池插进归一化流
#    ⇒ 锚帧 = 「幅度 2~3 倍 + 每通道直流偏移」的畸形帧 ⇒ 结构上不可能成为「共同基准」
#    （实证：旧日志 `pose_latents` 首帧 mean=0.061764 == mean(-latents_mean/latents_std)）。
#    现在：x / pose 池一律先归一化；mask（28ch）不归一化。
# 诊断：节点参数 `debug_log`（推荐，**免重启**）/ 哨兵文件 `CWA_DEBUG.on` / 环境变量 `CWA_DEBUG=1`，
#    打印 空间自检 / 每窗条件流布局 / 跨窗共享头 / **两窗一致性（level + 动作两个口径）**
#    （= 重影的直接量纲；旧 devlog §10 基线：正常素材 0.22~0.35）。
# 实验开关（背后就是「为什么一直改不对」的三件事，默认 = 手动测试的语义）：
#   · `anchor_scope=fresh_only`（默认）：**跳过含「接续帧钉住区」（全局帧 0）的窗口** —— 它已经有
#     真实已知前缀，再叠合成头会互相干扰；`all` = 每个窗口都插（旧行为）。
#   · `shared_head=off`（默认）：每个窗口**各自生成**锚帧内容（= 手测里两段各自生成的等价物）；
#     `leader` / `consensus` = 跨窗共享一份（更强，但可能引入偏置/混合）。
#   · `overlap_scope=average`（默认）：重叠帧被相邻两窗**共写**并加权平均（= 现状，也是**动作被
#     抵消/发软**的根源：两个方向不同的动作一平均就互相削弱）；
#     `single` = **重叠只读化** —— 已被前窗写完的重叠段在本窗只读（参与注意力、不写回）
#     ⇒ 每帧只有一个作者、无平均；切片长度/显存/token 预算**都不变**，且下一窗仍看得到前窗真实输出。
# 必须接在上下文窗口节点（Custom Context Windows (Manual) / Wan SCAIL Context Windows）之后。
# 实现方式：浅拷贝 handler 实例并包装 get_context_windows / get_resized_cond / execute
#           （不做 __class__ 置换 —— 3.13 下 abc.ABC 派生类会报 layout differs）。
#
# ⚠️ RoPE 位置补丁是**进程级 monkey patch**（改 SCAILWanModel.rope_encode 类属性）：
#    只有重启 ComfyUI 才真正还原。但功能上被「运行期标记 _anchor_active + 窗口上的
#    _anchor_plan」双重门控 —— 关掉/绕过本节点时补丁自动走原实现，等价于没打。
#    补丁自带「热重载自愈」：每次应用前检查当前函数是不是本补丁，不是就重装（避免静默失效）。
# ==============================

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass

import torch

# handler 包装链（多节点共用同一个 handler 时的唯一包装入口，见 handler_wrap.py）。
# 注意：本模块会被两种方式导入 —— ComfyUI 以「包」加载（相对导入可用），
# 而 md/ 下的自测脚本把本目录加进 sys.path 后按「顶层模块」导入（相对导入会 ImportError）。
try:
    from .handler_wrap import install as _wrap_install, describe as _wrap_describe
except ImportError:                                     # pragma: no cover - 顶层导入（自测脚本）
    from handler_wrap import install as _wrap_install, describe as _wrap_describe


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
    driving_only: bool = True           # True=只接驱动流+驱动遮罩（SCAIL-2）；animate2 会强制为 False
    memory_frames: int = 0              # 低帧率「历史驱动」帧数（0=关闭；单位 = latent 帧）
    memory_stride: int = 0              # 历史帧间距：0=自动（覆盖全历史）/ >0=固定（latent 帧）
    anchor_fixed: bool = False          # True=锚帧 latent 在窗口间逐位相同（真正恒定基准）
    anchor_image_blank: bool = False    # True=待生成序列里的锚帧只放噪声占位（不写锚帧画面内容）
    log_debug: bool = False


# ---------------- 模块级锚定池（按最后一次 apply_anchor 生效） ----------------
_POOL_CACHE: dict = {}
_POOL_SOURCE = {"frames": None, "mask_frames": None, "vae": None}
_ANCHOR_COUNT = 0
_DRIVING_ONLY = True
_MEMORY_FRAMES = 0
_MEMORY_STRIDE = 0      # 记忆组的抽样间距（latent 帧）：0 = 自动（覆盖「片段开头 → 本窗口之前」全部历史）
_ANCHOR_FIXED = False
_ANCHOR_IMAGE_BLANK = False   # True = 待生成序列里的锚帧只放噪声占位（不写入 anchor_frames 的画面内容）
_WARNED = set()
_LOGGED_ONCE = set()

# ---- 实验开关（由节点面板参数写入；默认 = 「手动测试」的语义）----
#   anchor_scope : fresh_only = 跳过「含接续帧钉住区（全局帧 0）」的窗口（它已有真实已知前缀）;
#                  all        = 每个窗口都插（= 旧行为）
#   shared_head  : off       = 每个窗口各自生成锚帧内容（= 手测里两段各自生成的等价物）;
#                  leader    = 跨窗共享：本步第一个窗口产出的头 → 下一步全员共用；
#                  consensus = 跨窗共享：各窗头内容取平均（无偏置，但是混合值）
#   overlap_scope: average   = 现状（重叠帧被两窗共写 + 融合加权平均）;
#                  single    = 重叠只读化：已被前窗写完的重叠段改为「只读前文」⇒ 每帧只有一个作者
_ANCHOR_SCOPE = "fresh_only"
_SHARED_HEAD_MODE = "off"
_OVERLAP_SCOPE = "average"

# 锚帧「占位噪声」的种子：固定值 => 跨窗口 / 跨运行**逐位一致**（作者的「共同基准」意图）。
# 可用环境变量 CWA_ANCHOR_NOISE_SEED 覆盖（十进制或 0x 前缀），便于做「换一块噪声会不会影响结果」
# 的对比实验；也可据此判断它是否会与周围帧的噪声产生不连续。
# 注意：它只影响「锚帧那几帧」的噪声；其他帧的噪声仍由你在采样器里设的 seed 决定。
_ANCHOR_NOISE_SEED = int(os.environ.get("CWA_ANCHOR_NOISE_SEED", "24301"), 0)   # 24301 = 0x5EED

# ---- 诊断开关（默认关）：CWA_DEBUG=1 时输出「空间 / 布局 / 两窗重叠区分歧」日志 ----
_DEBUG_ENV = str(os.environ.get("CWA_DEBUG", "")).strip() not in ("", "0", "false", "False", "no")
# 哨兵文件（存在即视为开启；放在节点目录下，已加入 .gitignore）
_DEBUG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CWA_DEBUG.on")
_DEBUG = _DEBUG_ENV      # 实际开关：每次节点执行时由 refresh_debug() 刷新


def refresh_debug(force_on: bool = False) -> bool:
    """刷新诊断开关，优先级：**节点参数 `debug_log`** > 哨兵文件 `CWA_DEBUG.on` > 环境变量 `CWA_DEBUG`。

    ⇒ 用节点上的 `debug_log` 开关就能**免重启**开/关诊断日志（环境变量只是兜底）。
    `debug_log=False` 时不强制关闭（否则会让「文件/环境变量」这两条路失效）。
    """
    global _DEBUG
    on = bool(force_on)
    if not on:
        try:
            on = os.path.exists(_DEBUG_FILE)
        except Exception:
            on = False
    if not on:
        on = _DEBUG_ENV
    _DEBUG = on
    return _DEBUG
# 模型的 latent 归一化函数（`BaseModel.process_latent_in`）。
# ⚠️ 为什么必须有它：节点的锚定池来自 `vae.encode()`（**原始** latent），而模型真正消费的是
#   `model_conds` 里的**模型空间键**（如 `pose_latents`）与采样中的 x —— 两者都是**归一化空间**
#   （`comfy/model_base.py:1973-1978` 对 pose/reference 做 `process_latent_in`；
#     `comfy/samplers.py:1224` 对 latent_image 同样处理；`Wan21.process_in = (x-mean)/std`）。
#   不归一化 = 把「幅度 ~2~3 倍 + 每通道直流偏移」的畸形帧喂给模型 ⇒ 锚帧不可能成为「共同基准」。
#   实证：旧日志里 `pose_latents` 首帧 mean=0.061764 == mean(-latents_mean/latents_std)。
_PROCESS_LATENT_IN = None
_PROCESS_LATENT_IN_NAME = "未捕获"
_DIAG = {}      # 诊断用临时状态（每步清空）：前一个窗口的输出 / 已打印标记
# 「原始空间」的 latent 键（模型侧会用 extra_conds 自行归一化 ⇒ 节点应保持原始）：
#   其余（如模型空间键 `pose_latents`）视为**模型空间**（值已归一化）⇒ 锚定池必须先归一化再插入。
RAW_LATENT_KEYS = (
    "pose_video_latent", "concat_latent_image", "reference_latents", "ref_latents",
    "time_dim_concat", "concat_mask", "denoise_mask", "c_concat",
)

# ---- 跨窗「共享头」：同一步内所有窗口看到**逐位相同**的头部块（每一步只算一次）----
# 为什么：手动测试「两段各自在开头拼上同一段视频开头」里，两段的头是**同一份内容** ⇒ 进入重叠区时
#   的基准/相位一致 ⇒ 分歧小。节点此前按「窗口帧索引」分别缓存上一步 x0 ⇒ 各窗的头各自演化 ⇒
#   「共同基准」实际不成立。
# 现在：每步由**第一个被评估的窗口（leader）**产生头内容（写 `_SHARED_HEAD_KEY`），步边界把它
#   「轮转」成 `_SHARED_HEAD_PREV_KEY` —— 下一个采样步里**所有窗口（含 leader）都读这一份**
#   ⇒ 同一步内逐位相同，且头每步只演化一次（= 手动测试里那个「共同的头」）。
#   （`anchor_fixed=关` 时生效；`anchor_fixed=开` 是更强的版本：整个采样过程恒定不变。）
_SHARED_HEAD_KEY = "__cwa_shared_head__"            # 本步 leader 刚产出的头 = (leader 帧索引, tensor)
_SHARED_HEAD_PREV_KEY = "__cwa_shared_head_prev__"  # 上一步的头：本步全员读它（逐位相同）


def _advance_shared_head(handler) -> None:
    """步边界轮转共享头：本步各窗产出的 → 下一步全员共用的那一份（并清空本步槽位）。

    由包装的 `get_context_windows` 每步调用一次（窗口对象每步重建 ⇒ 那里就是「步」的边界）；
    自测也直接调它来模拟步边界。

    `_SHARED_HEAD_MODE`：
      off       → 不做共享（清空槽位；每个窗口用自己的上一步 x0，= 手动测试的等价物）
      leader    → 本步第一个被评估的窗口产出的头（`(key, tensor)`）
      consensus → 本步各窗头内容的平均（`("consensus(n)", mean)`）
    """
    try:
        if handler is None:
            return
        cache = _x0_cache(handler)
        cur = cache.get(_SHARED_HEAD_KEY, None)
        if _SHARED_HEAD_MODE == "off":
            cache.pop(_SHARED_HEAD_PREV_KEY, None)
        elif _SHARED_HEAD_MODE == "consensus":
            blocks = []
            if isinstance(cur, list):
                blocks = [it[1] for it in cur
                          if isinstance(it, tuple) and len(it) == 2 and torch.is_tensor(it[1])]
            if blocks:
                try:
                    mean = torch.stack(blocks, dim=0).mean(dim=0)
                    cache[_SHARED_HEAD_PREV_KEY] = (f"consensus({len(blocks)})", mean)
                except Exception:
                    cache[_SHARED_HEAD_PREV_KEY] = cur[0]
        else:                                   # leader
            if cur is not None:
                cache[_SHARED_HEAD_PREV_KEY] = cur
        cache.pop(_SHARED_HEAD_KEY, None)
    except Exception:
        logging.debug("[ContextAnchor] 共享头轮转失败（忽略）", exc_info=True)


def _anchor_on(window) -> bool:
    """只有被锚定节点包装过的窗口（运行期打了 _anchor_active 标记）才允许注入。

    模型 patch 是进程级常驻的、而锚帧素材/配置是模块级的：若不校验标记，上一次运行残留的配置
    会污染「绕过锚定节点」的运行（曾导致 ref_mask 比 x 多 K 帧而崩）。
    """
    if getattr(window, "_anchor_skip", False):
        return False          # anchor_scope=fresh_only：该窗口已有「接续帧钉住区」⇒ 不再叠合成头
    return bool(getattr(window, "_anchor_active", False))


def _x0_cache(handler) -> dict:
    """锚帧「上一步 x0」缓存（挂在 handler 上 => 跨采样步存活）。

    ⚠️ 不能挂在 window 上：窗口对象每个采样步都会重建
    （Custom_context.get_context_windows 每步 new 一批），挂窗口 = 每步丢失 =>
    锚帧退化成「每步都从干净锚帧重来」，后果有两个：
      1) `anchor_fixed` 开/关完全等价（用户实测确认过）；
      2) 后期步（σ→0）锚帧 ≈ 一张干净的驱动者画面，被模型顺着续写
         => 出现「驱动者形象 / 开头动作叠在当前动作上」的伪影。
    handler 在同一次模型应用/整段采样内共享，正好适合放这份「跨步」缓存。
    """
    cache = getattr(handler, "_cwa_anchor_x0", None)
    if cache is None:
        cache = {}
        handler._cwa_anchor_x0 = cache
    return cache


def _x0_cache_key(window):
    """窗口的稳定标识：即使每步重建，同一个窗口的帧索引也相同。"""
    orig = getattr(window, "original_indices", None) or []
    if orig:
        return tuple(int(i) for i in orig)
    return tuple(int(i) for i in (getattr(window, "index_list", None) or []))


def _log_once(tag, msg, *args):
    """同一条消息只打一次（避免每个采样步 × 每个窗口刷屏）。"""
    if tag in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(tag)
    logging.info(msg, *args)


def _dbg(msg, *args):
    """诊断日志（仅 CWA_DEBUG=1 时输出）；调用方自行按 (步, 窗口) 去重。"""
    if _DEBUG:
        logging.info(msg, *args)


def _capture_latent_normalizer(model):
    """捕获模型的 latent 归一化函数（`BaseModel.process_latent_in`）供锚定池转「模型空间」用。

    采样空间 / 模型空间的条件流都是**归一化** latent，而 `vae.encode()` 产出的是**原始** latent
    ⇒ 没有这一步，锚帧就是分布外的畸形帧（见文件顶部 `_PROCESS_LATENT_IN` 的说明）。
    """
    global _PROCESS_LATENT_IN, _PROCESS_LATENT_IN_NAME
    _PROCESS_LATENT_IN = None
    _PROCESS_LATENT_IN_NAME = "未捕获"
    try:
        inner = getattr(model, "model", model)          # ModelPatcher → BaseModel
        fn = getattr(inner, "process_latent_in", None)
        if callable(fn):
            _PROCESS_LATENT_IN = fn
            _PROCESS_LATENT_IN_NAME = "model.process_latent_in"
            return
    except Exception:
        pass
    try:
        import comfy.latent_formats
        _PROCESS_LATENT_IN = comfy.latent_formats.Wan21().process_in
        _PROCESS_LATENT_IN_NAME = "Wan21.process_in（兜底）"
    except Exception:
        logging.warning("[ContextAnchor] 无法捕获 latent 归一化函数：锚定池将按**原始**空间使用"
                        "（锚帧可能无法起到「共同基准」作用）")


def _space_norm(tensor, kind: str = "latent"):
    """把**原始 VAE latent** 转到模型/采样空间（归一化）；mask 类不要调用本函数。"""
    if tensor is None or _PROCESS_LATENT_IN is None:
        return tensor
    try:
        return _PROCESS_LATENT_IN(tensor)
    except Exception:
        logging.exception("[ContextAnchor] %s 归一化失败（按原始空间继续）", kind)
        return tensor


def _stat_str(tensor):
    """一行统计（诊断用）：shape + mean/std。"""
    if tensor is None or not torch.is_tensor(tensor):
        return "None"
    try:
        t = tensor.detach().float()
        return (f"{list(t.shape)} ch_mean={float(t.mean()):+.4f} "
                f"ch_std={float(t.std()):.4f} abs_mean={float(t.abs().mean()):.4f}")
    except Exception:
        return str(getattr(tensor, "shape", None))


def _slice_frames(tensor, dim: int, sl: slice):
    """沿 dim 取帧切片（诊断用；dim 为模型时间维）。"""
    return tensor[tuple([slice(None)] * int(dim) + [sl])]


def _win_range(window) -> str:
    """窗口的全局帧范围（日志/诊断用）。"""
    orig = list(getattr(window, "original_indices", None) or [])
    return f"{orig[0]}..{orig[-1]}" if orig else "?"


def _per_frame_abs_mean(tensor, dim: int):
    """逐帧的绝对值均值（列表）——把时间维移到最前再摊平其余维度。"""
    try:
        x = torch.movedim(tensor, int(dim), 0)
        return x.reshape(x.shape[0], -1).mean(dim=1).tolist()
    except Exception:
        return []



def describe_windows(windows) -> str:
    """一行描述窗口边界（latent 全局索引 + 头/尾只读段帧数），便于把截图里的"缝"对到帧号。

    latent i（i≥1）对应像素帧 4i-3..4i => 缝就落在相邻两窗 original_indices 的交界处。
    """
    parts = []
    for i, w in enumerate(windows or []):
        orig = list(getattr(w, "original_indices", None) or [])
        if not orig:
            continue
        pre = int(getattr(w, "prefix_length", 0) or 0)
        head = len(getattr(w, "head_extra_indices", None) or [])
        tail = len(getattr(w, "tail_extra_indices", None) or [])
        parts.append(f"#{i} {orig[0]}-{orig[-1]}(前缀{pre} 头{head} 尾{tail})")
    return " | ".join(parts)


def _sigma_step_str(handler, model_options) -> str:
    """σ +「步号」：双阶段采样（WanSCAIL2PhaseSampler）里 Phase 2 会重新从步号 0 计数，看 σ 才准。"""
    step = int(getattr(handler, "_step", 0) or 0)
    try:
        sigmas = model_options["transformer_options"]["sample_sigmas"]
        if 0 <= step < len(sigmas):
            return f"σ={float(sigmas[step]):.3f}（步号 {step}）"
    except Exception:
        pass
    return f"步号 {step}"


def reset_pools(frames, mask_frames, vae, driving_only: bool = True,
                memory_frames: int = 0, memory_stride: int = 0,
                anchor_fixed: bool = False, anchor_image_blank: bool = False, model=None,
                anchor_scope: str = "fresh_only", shared_head: str = "off",
                overlap_scope: str = "average"):
    """设置锚定素材并清空编码缓存；返回锚定 latent 帧数。"""
    global _ANCHOR_COUNT, _DRIVING_ONLY, _MEMORY_FRAMES, _MEMORY_STRIDE, _ANCHOR_FIXED, _ANCHOR_IMAGE_BLANK
    global _ANCHOR_SCOPE, _SHARED_HEAD_MODE, _OVERLAP_SCOPE
    _POOL_CACHE.clear()
    _DIAG.clear()
    _capture_latent_normalizer(model)
    _WARNED.clear()
    _LOGGED_ONCE.clear()
    _DRIVING_ONLY = bool(driving_only)
    _MEMORY_FRAMES = max(0, int(memory_frames or 0))
    # 0 = 自动（不要用 `or 4` 兜底：0 是合法值，会被 falsy 判断吞掉）
    _MEMORY_STRIDE = max(0, int(memory_stride if memory_stride is not None else 0))
    _ANCHOR_FIXED = bool(anchor_fixed)
    _ANCHOR_IMAGE_BLANK = bool(anchor_image_blank)
    _ANCHOR_SCOPE = str(anchor_scope or "fresh_only") if str(anchor_scope) in ("fresh_only", "all") else "fresh_only"
    _SHARED_HEAD_MODE = (str(shared_head) if str(shared_head) in ("off", "leader", "consensus") else "off")
    _OVERLAP_SCOPE = (str(overlap_scope) if str(overlap_scope) in ("average", "single") else "average")
    _POOL_SOURCE.update(frames=frames, mask_frames=mask_frames, vae=vae)
    _ANCHOR_COUNT = latent_frames_of(frames.shape[0]) if frames is not None else 0
    return _ANCHOR_COUNT


def anchor_active() -> bool:
    """本节点是否处于「有实际作用」的状态（有锚帧 或 有头/尾只读上下文 或 有历史记忆组）。"""
    return _ANCHOR_COUNT > 0 or _MEMORY_FRAMES > 0


def anchor_count() -> int:
    return int(_ANCHOR_COUNT)


# ---------------- 基础工具 ----------------

def _needs_driving_mask(model) -> bool:
    """SCAIL-2 必须同时锚定驱动遮罩（28ch，逐 token 加进驱动流 → 必须与驱动同帧数）。"""
    try:
        return "SCAIL" in type(model.model).__name__.upper()
    except Exception:
        return False


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
    try:
        import comfy.utils
        return comfy.utils.common_upscale(frames.movedim(-1, 1).float(), int(target_w), int(target_h),
                                          "area", "center").movedim(1, -1)
    except Exception:
        # 独立环境（无 comfy）兜底：面积插值，便于自测直接跑
        x = frames.movedim(-1, 1).float()
        return torch.nn.functional.interpolate(x, size=(int(target_h), int(target_w)),
                                               mode="area").movedim(1, -1)


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

def _mask_latents_any_t(frames):
    """(T,H,W,3) 彩色遮罩（0..1）→ (1,T_lat,28,H/8,W/8)，**任意 T 都能编码**。

    约定与 wan_scail_multi_ref._extract_mask_to_28ch 完全一致（所以 T=1+4n 时结果不变）：
      latent 0 = 第 0 帧（重复 4 次）；latent i(i≥1) = 第 4i-3 .. 4i 帧，组内不足 4 帧时用该组最后一帧补足。
    """
    t = int(frames.shape[0])
    t_lat = max(1, (t - 1) // 4 + 1)
    thr = 225.0 / 255.0
    m = frames.movedim(-1, 1).float()
    r = (m[:, 0:1] > thr).float()
    g = (m[:, 1:2] > thr).float()
    b = (m[:, 2:3] > thr).float()
    nr, ng, nb = 1 - r, 1 - g, 1 - b
    seven = torch.cat([r * g * b, r * ng * nb, nr * g * nb, nr * ng * b,
                       r * g * nb, r * ng * b, nr * g * b], dim=1)     # (T,7,H,W)
    h_lat, w_lat = int(seven.shape[2]), int(seven.shape[3])
    for _ in range(3):
        h_lat = (h_lat + 1) // 2
        w_lat = (w_lat + 1) // 2
    seven = torch.nn.functional.interpolate(seven, size=(h_lat, w_lat), mode="area")

    groups = []
    for i in range(t_lat):
        if i == 0:
            lo, need, repeat0 = 0, 1, True
        else:
            lo, need, repeat0 = 4 * i - 3, 4, False
        lo = max(0, min(lo, t - 1))
        hi = min(t, lo + need)
        grp = seven[lo:hi]
        if grp.shape[0] < need:                       # 组内帧数不足 → 用该组最后一帧补足
            grp = torch.cat([grp, grp[-1:].repeat(need - grp.shape[0], 1, 1, 1)], dim=0)
        if repeat0:                                   # latent 0 = 第 0 帧重复 4 次（旧约定）
            grp = grp.repeat(4, 1, 1, 1)
        groups.append(grp)
    stacked = torch.cat(groups, dim=0)                # (4*T_lat, 7, H_lat, W_lat)
    return stacked.view(t_lat, 28, h_lat, w_lat).unsqueeze(0)


def _encode_pool(kind: str, h: int, w: int):
    """编码一个锚定池（失败时**抛异常**，由 _get_pool 统一降级）。"""
    src = _POOL_SOURCE
    target_h, target_w = int(h) * 8, int(w) * 8
    if kind == "mask":
        if src["mask_frames"] is None:
            raise RuntimeError("未接 anchor_mask_frames（SCAIL-2 的驱动遮罩无法锚定）")
        frames = _resize_frames(src["mask_frames"], target_h, target_w)
        pool = _mask_latents_any_t(frames[:, :, :, :3])          # (1,T',28,h,w)，时间在 dim=1
        return _fit_frames(pool, 1, _ANCHOR_COUNT)
    vae = src["vae"]
    if vae is None:
        raise RuntimeError("没有 VAE，无法把锚帧编码成 latent")
    frames = _resize_frames(src["frames"], target_h, target_w)
    pool = vae.encode(frames[:, :, :, :3])
    return _fit_frames(pool, 2, _ANCHOR_COUNT)


def _get_pool(kind: str, spatial_h: int, spatial_w: int, target_channels=None, device=None, dtype=None,
              model_space: bool = True):
    """按目标 latent 空间尺寸取锚定池（kind ∈ x/pose/mask），带缓存；编码失败则返回 None（只报一次）。"""
    if _POOL_SOURCE["frames"] is None:
        return None
    # normalized：目标流是不是「模型空间」（= 值已由 extra_conds 归一化）。
    #   x / pose 池插入的是 latent 流（模型空间 或 采样空间）⇒ 必须归一化；
    #   mask 池（28ch）**永远不归一化**。
    normalized = bool(model_space) and kind in ("x", "pose")
    key = (kind, int(spatial_h), int(spatial_w), normalized)
    if key not in _POOL_CACHE:
        try:
            _POOL_CACHE[key] = _encode_pool(kind, int(spatial_h), int(spatial_w))
            if normalized:
                _before = _stat_str(_POOL_CACHE[key])
                _POOL_CACHE[key] = _space_norm(_POOL_CACHE[key], kind)
                _log_once(("pool_space", kind),
                          "[ContextAnchor] 锚定池已转到模型空间（%s，原始→归一化）：%s → %s（归一化器=%s）",
                          kind, _before, _stat_str(_POOL_CACHE[key]), _PROCESS_LATENT_IN_NAME)
        except Exception:
            logging.exception("[ContextAnchor] 锚定池编码失败（kind=%s, spatial=(%d,%d)）", kind, spatial_h, spatial_w)
            _POOL_CACHE[key] = None
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
    if window is None or not _anchor_on(window) or getattr(window, "_anchor_x_applied", False):
        return x_slice
    pool = _get_pool("x", x_slice.shape[-2], x_slice.shape[-1],
                     target_channels=x_slice.shape[1] if x_slice.ndim > 1 else None,
                     model_space=True,      # 采样中的 x = 归一化空间（samplers.py:1224）
                     device=x_slice.device, dtype=x_slice.dtype)
    if pool is None:
        return x_slice
    sigma = float(sigma)
    gen = torch.Generator(device="cpu").manual_seed(_ANCHOR_NOISE_SEED)
    noise = torch.randn(pool.shape, generator=gen, device="cpu", dtype=torch.float32)
    noise = noise.to(device=pool.device, dtype=pool.dtype)
    # 锚帧内容（按优先级）：
    #   anchor_image_blank=True → 纯噪声占位：base 保持 None => 锚帧 = 噪声本身。
    #        （锚帧素材来自**驱动视频**=源人物画面，写进画面侧会泄漏源形象；开这个可避免。
    #          仍然保留 K 帧的位置 => 模型自己在那 K 帧里完成「参考图 → 当前姿态」的迁移。）
    #   anchor_fixed=True       → 恒定基准：base 永远是干净锚帧，x_t=(1-σ)·锚帧+σ·固定噪声
    #   （默认）                → 自己/共同的生成轨迹（σ=1 时退化为纯噪声）：
    #        ① **跨窗共享头**（同一步内 leader 窗口生成的 x0，所有窗口逐位相同）—— 默认行为，
    #           对应手动测试里「两段拼同一段开头」；
    #        ② 回退：本窗上一步的 x0；③ 再回退：干净锚帧（缓存未命中）。
    base = None
    base_src = ""
    if not _ANCHOR_IMAGE_BLANK:
        if _ANCHOR_FIXED:
            base, base_src = pool, "干净锚帧（恒定基准）"
        else:
            handler = (model_options or {}).get("context_handler", None)
            if handler is not None:
                cache = _x0_cache(handler)
                for _k, _label in ((_SHARED_HEAD_PREV_KEY, "跨窗共享头（上一步 leader 产出）"),):
                    entry = cache.get(_k, None)
                    if isinstance(entry, tuple) and len(entry) == 2:
                        leader, shared = entry
                        if torch.is_tensor(shared) and tuple(shared.shape) == tuple(pool.shape):
                            base, base_src = shared, f"{_label}，leader={leader}"
                            break
                if base is None:
                    own = cache.get(_x0_cache_key(window), None)
                    if torch.is_tensor(own) and tuple(own.shape) == tuple(pool.shape):
                        _why = "" if _SHARED_HEAD_MODE == "off" else "（共享头未就绪）"
                        base, base_src = own, f"本窗上一步 x0{_why}"
            if base is None:
                base, base_src = pool, "干净锚帧（缓存未命中回退）"
    anchor = noise if base is None else base * (1.0 - sigma) + noise * sigma
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    new_x = _insert_at(x_slice, anchor, dim, prefix_len)
    if new_x is not x_slice:
        window._anchor_x_applied = True
        window.anchor_latent_count = int(pool.shape[dim])
        window.anchor_latent = pool
        if _ANCHOR_IMAGE_BLANK:
            src = "纯噪声占位（不写锚帧画面）"
        else:
            src = base_src or "上一步 x0"
        _log_once(("x",),
                  "[ContextAnchor] 待生成序列前已插入锚定 %d 帧（位置=%d, σ=%.3f, 内容=%s）：%s → %s",
                  pool.shape[dim], prefix_len, sigma, src, list(x_slice.shape), list(new_x.shape))
        _dbg("[ContextAnchor][diag] σ=%.3f 窗 %s：x 前插锚 %d 帧（内容=%s, 锚块 %s）",
             sigma, _win_range(window), pool.shape[dim], src, _stat_str(anchor))
    return new_x


def _capture_anchor_x0(model_options, extended, sub_x, out, dim):
    """缓存模型对锚帧位置的 x0 预测，作为下一步锚帧的内容（让它像真实帧一样被生成）。

    缓存挂在 handler 上（跨步存活）；key = 窗口帧索引（共享头未就绪时回退用），另外维护**跨窗共享头**：
    本步第一个被评估的窗口（leader）把它的 x0 写进 `_SHARED_HEAD_KEY`，步边界由 `_advance_shared_head`
    轮转成 `_SHARED_HEAD_PREV_KEY` ⇒ **下一个采样步里所有窗口（含 leader）都读同一份**（逐位相同）
    ⇒ 对应手动测试里「两段各自拼上同一段视频开头」的共同的头，共同基准/相位才成立。
    """
    try:
        if extended is sub_x or out is None:
            return
        if isinstance(out, (list, tuple)):        # CFG：取正样本那一路
            out = out[0] if len(out) > 0 else None
        if not torch.is_tensor(out):
            return
        window = ((model_options or {}).get("transformer_options") or {}).get("context_window", None)
        if window is None or not _anchor_on(window):
            return
        handler = (model_options or {}).get("context_handler", None)
        if handler is None:
            return
        k = int(getattr(window, "anchor_latent_count", 0) or 0)
        if k <= 0 or out.ndim != extended.ndim or out.shape[dim] != extended.shape[dim]:
            return
        start = int(getattr(window, "prefix_length", 0) or 0)
        sl = [slice(None)] * out.ndim
        sl[dim] = slice(start, start + k)
        block = out[tuple(sl)].detach().clone()
        key = _x0_cache_key(window)
        cache = _x0_cache(handler)
        cache[key] = block                         # 本窗自己的（共享头未就绪时的回退）
        if _SHARED_HEAD_MODE == "off":
            _dbg("[ContextAnchor][diag] 窗 %s 产出的头仅本窗使用（shared_head=off，%s）",
                 _win_range(window), _stat_str(block))
        elif _SHARED_HEAD_MODE == "consensus":
            lst = cache.get(_SHARED_HEAD_KEY)
            if not isinstance(lst, list):
                lst = []
                cache[_SHARED_HEAD_KEY] = lst
            lst.append((key, block))
            _dbg("[ContextAnchor][diag] 窗 %s 的头已计入本步共识（%d 份，%s）",
                 _win_range(window), len(lst), _stat_str(block))
        else:                                      # leader
            if _SHARED_HEAD_KEY not in cache:
                cache[_SHARED_HEAD_KEY] = (key, block)
                _dbg("[ContextAnchor][diag] 本步共享头由窗 %s 产生（leader，%s）",
                     _win_range(window), _stat_str(block))
            else:
                entry = cache.get(_SHARED_HEAD_KEY)
                leader = entry[0] if isinstance(entry, tuple) and len(entry) == 2 else "?"
                _dbg("[ContextAnchor][diag] 窗 %s 已产生自己的头；本步其余窗口复用 leader=%s 的头",
                     _win_range(window), leader)
    except Exception:
        logging.debug("[ContextAnchor] 锚帧 x0 缓存失败（忽略）", exc_info=True)


# ---------------- 注入：条件流 ----------------


def _find_full_stream(cond_in, keys):
    """在（尚未切片的）条件字典里找第一个匹配 key 的张量。

    用途：取「全量驱动流」来切出本窗口之前的低帧率历史帧（记忆组）。
    优先 model_conds 子字典（模型侧键，如 pose_latents），其次顶层原始键（如 pose_video_latent）。
    """
    if not cond_in:
        return None
    for nested_first in (True, False):
        for actual in cond_in:
            if not isinstance(actual, dict):
                continue
            for k, v in actual.items():
                if nested_first and isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if sub_k in keys:
                            t = _cond_tensor(sub_v)
                            if t is not None:
                                return t
                elif not nested_first and not isinstance(v, dict) and k in keys:
                    t = _cond_tensor(v)
                    if t is not None:
                        return t
    return None


def _memory_stride_for(lo: int) -> int:
    """本次记忆组使用的抽样间距（latent 帧）。

    _MEMORY_STRIDE = 0（自动，默认）：让 _MEMORY_FRAMES 个点刚好覆盖「片段开头 → 本窗口
        之前」的全部历史 => 间隔 = max(1, 窗口起点 // 点数)。片段越长 / 窗口越靠后，间隔越大
        （时间分辨率降低，但**始终覆盖得到开头**；固定间距在长片段上会够不到开头，
        失去「长程」的意义）。
    _MEMORY_STRIDE > 0：固定间隔（最小 1）。
    """
    n = max(1, int(_MEMORY_FRAMES))
    if int(_MEMORY_STRIDE) > 0:
        return max(1, int(_MEMORY_STRIDE))
    return max(1, int(lo) // n)


def _memory_tokens(window, full_pose, tensor, t_dim, ch):
    """从【全量】驱动流里取「本窗口之前」的低帧率历史帧。

    返回 (tensor, 实际取到的帧数, 本次使用的间距)。帧序 = 从旧到新（与 RoPE 位置升序一致）。
    窗口前面没有可用历史（例如窗口 0）时返回 (None, 0, stride)。
    """
    if full_pose is None or _MEMORY_FRAMES <= 0:
        return None, 0, max(1, int(_MEMORY_STRIDE) or 1)
    # 基准取「窗口自身的起点」：不能用 frame_indices()（它含头/尾只读段），
    # 否则开了 head_context_frames 会把记忆组的取点位置一起往前推 —— 两个开关互相干扰。
    indices = list(getattr(window, "original_indices", None) or [])
    if not indices:
        try:
            indices = list(window.frame_indices())
        except Exception:
            indices = list(getattr(window, "index_list", None) or [])
    if not indices:
        return None, 0, max(1, int(_MEMORY_STRIDE) or 1)
    lo = min(indices)
    stride = _memory_stride_for(lo)
    full_len = int(full_pose.shape[t_dim])
    idx = sorted(i for i in (lo - stride * (m + 1) for m in range(int(_MEMORY_FRAMES)))
                 if 0 <= i < full_len)
    if not idx:
        return None, 0, stride
    sl = [slice(None)] * full_pose.ndim
    sl[t_dim] = idx
    mem = full_pose[tuple(sl)]
    if int(mem.shape[1]) != int(ch):
        mem = _match_channels(mem, int(ch))
    if mem is None or int(mem.shape[1]) != int(ch):
        return None, 0, stride
    _log_once(("memory_span",),
              "[ContextAnchor] 记忆组：窗口起点 %d，间距 %d latent，取到 %d 帧（覆盖到第 %d 帧）",
              lo, stride, len(idx), (idx[0] if idx else -1))
    return mem.to(device=tensor.device, dtype=tensor.dtype), len(idx), stride


def _set_anchor_plan(window, head: int, tail_pool: int, tail_zero: int,
                     prehistory: int, memory: int, stride: int) -> None:
    """把「本窗口怎么扩流」的精确方案记到窗口上。

    单一事实来源：条件流注入写它，SCAIL 掩码注入（prepend_anchor_mask）读它，
    RoPE 位置补丁（_apply_scail_anchor_rope_patch）也读它 —— 三处必须一致才不会错位。
      head        : 前插到驱动流最前的锚帧数（0 = 不前插）
      tail_pool   : 追加到驱动流尾部的「锚帧」数（对应尾部 token，用遮罩池）
      tail_zero   : 追加到驱动流尾部的「历史记忆帧」数（掩码填 0：无身份信息）
      prehistory  : RoPE 需要放在窗口之前的锚帧 token 数（= driving_only 时的 head→tail）
      memory      : RoPE 需要放在窗口之前的低帧率历史 token 数
    """
    window._anchor_plan = {"head": int(head), "tail_pool": int(tail_pool),
                           "tail_zero": int(tail_zero)}
    window.anchor_head_count = int(head)
    window.anchor_tail_count = int(tail_pool) + int(tail_zero)
    window.anchor_prehistory_count = int(prehistory)
    window.anchor_memory_count = int(memory)
    window.anchor_memory_stride = int(stride)


def _window_lengths(window, dim: int = 2):
    """返回 (slice_len, frames_len)：含参考前缀的窗口切片长度 / 不含前缀的窗口帧长度。"""
    index_list = getattr(window, "index_list", None) or []
    try:
        frames_len = len(window.frame_indices())
    except Exception:
        frames_len = len(getattr(window, "original_indices", None) or index_list)
    return len(index_list), frames_len


def _prepend_one_cond(key, tensor, window, slice_len, frames_len, prefix_len, stats=None, only_pose=False,
                      full_pose=None):
    """按键类型把锚帧（+ 低帧率历史帧）接进某个条件张量；返回新张量（不改则原样返回）。

    长度判据必须是「窗口级」长度（slice_len / frames_len），不能用全局 chunk 长度 ——
    否则窗口内部的逐帧流会被静默漏插，随后在 comfy 的 c_concat 通道拼接处以形状不匹配报错。
    """
    if stats is None:
        stats = {"extended": [], "skipped": []}
    if tensor is None or (_ANCHOR_COUNT <= 0 and _MEMORY_FRAMES <= 0):
        return tensor
    t_dim = _temporal_dim(tensor)
    cur = int(tensor.shape[t_dim])
    ch = int(tensor.shape[1]) if tensor.ndim > 1 else 1

    def _pool(kind, model_space: bool = True):
        return _get_pool(kind, tensor.shape[-2], tensor.shape[-1], target_channels=ch,
                         model_space=model_space,
                         device=tensor.device, dtype=tensor.dtype)

    # 1) 驱动/pose latent 流
    #    - driving_only=True  → 锚帧 + 低帧率历史帧【追加在尾部】，再由 RoPE 补丁把它们放在
    #      「本窗口之前」的位置 => 不与视频帧 1:1 配对、**不会把整段动作错位 K 帧**（旧行为会）；
    #    - driving_only=False → 锚帧【前插】，与视频帧逐帧配对（最接近训练语义，采样后裁掉）；
    #      此时低帧率历史帧仍追加在尾部（同样由 RoPE 补丁放到窗口之前）。
    if key in ANCHOR_POSE_KEYS:
        # 模型空间键（`pose_latents`）= 归一化空间；原始键（`pose_video_latent`）= 原始空间。
        # 空间错了 ⇒ 锚帧是畸形帧 ⇒ 不可能成为「共同基准」（本文件顶部 _PROCESS_LATENT_IN 说明）。
        pool = _pool("pose", str(key) not in RAW_LATENT_KEYS) if _ANCHOR_COUNT > 0 else None
        if pool is not None:
            window.anchor_pose_latent = pool
        mem, mem_n, stride = _memory_tokens(window, full_pose, tensor, t_dim, ch)
        if pool is None and mem is None:
            stats["skipped"].append(key)
            return tensor
        if _DRIVING_ONLY:
            k_pre = int(pool.shape[t_dim]) if pool is not None else 0
            new_t = tensor
            if k_pre > 0:
                new_t = _insert_at(new_t, pool, t_dim, int(new_t.shape[t_dim]))
            if mem is not None:
                new_t = _insert_at(new_t, mem, t_dim, int(new_t.shape[t_dim]))
            _set_anchor_plan(window, head=0, tail_pool=k_pre, tail_zero=mem_n,
                             prehistory=k_pre, memory=mem_n, stride=stride)
            if new_t is not tensor:
                stats["extended"].append(f"{key}(尾部:锚{k_pre}+历史{mem_n})")
                _log_once(("pose_tail", key),
                          "[ContextAnchor] 驱动流 '%s' 尾部追加 %d 帧 token（锚定 %d + 低帧率历史 %d，间距 %d；"
                          "RoPE 会放在窗口之前 => 不再产生 K 帧错位）：%s → %s",
                          key, k_pre + mem_n, k_pre, mem_n, stride, list(tensor.shape), list(new_t.shape))
            return new_t
        new_t = _insert_at(tensor, pool, t_dim, 0) if pool is not None else tensor
        if mem is not None:
            new_t = _insert_at(new_t, mem, t_dim, int(new_t.shape[t_dim]))
        k_head = int(pool.shape[t_dim]) if pool is not None else 0
        _set_anchor_plan(window, head=k_head, tail_pool=0, tail_zero=mem_n,
                         prehistory=0, memory=mem_n, stride=stride)
        if new_t is not tensor:
            stats["extended"].append(f"{key}(前插锚{k_head}+历史{mem_n})")
            _log_once(("pose_head", key),
                      "[ContextAnchor] 驱动流 '%s' 前插锚定 %d 帧 + 尾部低帧率历史 %d 帧：%s → %s",
                      key, k_head, mem_n, list(tensor.shape), list(new_t.shape))
        return new_t

    if only_pose:
        return tensor
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
        extra, what = _pool("x", str(key) not in RAW_LATENT_KEYS), "x池"

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


def _log_anchor_layout(window, resized_cond, dim: int) -> None:
    """诊断：打印本窗口的条件流布局 + 「锚段 vs 窗口自身帧」统计（**空间自检**）。

    关键读数：**锚段与窗口自身驱动帧应当同尺度**（mean/std 同数量级）。
    若锚段 std 是窗口帧的 2~3 倍且带直流偏移 ⇒ 锚定池没走到模型空间 ⇒ 锚帧无法当「共同基准」。
    """
    plan = getattr(window, "_anchor_plan", None) or {}
    kx = int(getattr(window, "anchor_latent_count", 0) or 0)
    orig = list(getattr(window, "original_indices", None) or [])
    _dbg("[ContextAnchor][diag] 窗 %s：x 侧锚=%d 帧 | plan=%s | 头只读=%d | index_list=%s / 共 %d 帧",
         _win_range(window), kx, plan,
         len(getattr(window, "head_extra_indices", None) or []),
         [int(i) for i in (getattr(window, "index_list", None) or [])][:3],
         len(getattr(window, "index_list", None) or []))
    if not resized_cond:
        return
    for actual in resized_cond:
        if not isinstance(actual, dict):
            continue
        for key, value in actual.items():
            if not isinstance(value, dict):
                continue
            for sk, sv in value.items():
                t = _cond_tensor(sv)
                if t is None or t.ndim < 3:
                    continue
                td = _temporal_dim(t)
                n = int(t.shape[td])
                _dbg("[ContextAnchor][diag]   流 '%s': len=%d ch=%d（窗口帧=%d, 额外=%+d）%s",
                     sk, n, int(t.shape[1]), len(orig), n - len(orig), _stat_str(t))
                if sk in ANCHOR_POSE_KEYS and kx > 0 and n > kx:
                    _dbg("[ContextAnchor][diag]     ↳ 锚段自检 '%s'：前 %d 帧 %s ｜ 后 %d 帧 %s",
                         sk, kx, _stat_str(_slice_frames(t, td, slice(0, kx))),
                         kx, _stat_str(_slice_frames(t, td, slice(n - kx, n))))


def inject_anchor_conds(window, resized_cond, x_in=None, dim: int = 2, only_pose: bool = False, full_pose=None):
    """把锚帧（+ 低帧率历史帧）接到窗口的条件流（驱动流 / 逐帧流 / 标记流），就地更新条目。

    长度判据必须用「窗口级」长度（slice_len / frames_len），不能用全局 chunk 长度。
    full_pose = 尚未切片的「全量驱动流」（用于切出低帧率历史帧，见 _memory_tokens）。
    """
    if resized_cond is None or not _anchor_on(window) or not anchor_active():
        if _DEBUG and getattr(window, "_anchor_skip", False):
            _dbg("[ContextAnchor][diag] 窗 %s 已跳过锚定（anchor_scope=fresh_only：含接续帧钉住区）",
                 _win_range(window))
        return resized_cond
    slice_len, frames_len = _window_lengths(window, dim)
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    stats = {"extended": [], "skipped": []}

    def _act(k, t):
        return _prepend_one_cond(k, t, window, slice_len, frames_len, prefix_len, stats,
                                 only_pose=only_pose, full_pose=full_pose)

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
    if _DEBUG:
        try:                       # 布局诊断放在**注入之后** ⇒ 「额外=+N」才反映锚帧/记忆组实际插了几帧
            _tag = ("layout", tuple(int(i) for i in (getattr(window, "original_indices", None) or [])))
            if _tag not in _DIAG:
                _DIAG[_tag] = True
                _log_anchor_layout(window, resized_cond, dim)
        except Exception:
            logging.debug("[ContextAnchor] 布局诊断失败（忽略）", exc_info=True)
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


def _mask_pool_frames(tensor, t_dim: int, count: int):
    """把遮罩池取成与目标张量同布局的 count 帧（t_dim=1: (B,T,28,H,W) / t_dim=2: (B,28,T,H,W)）。"""
    pool = _get_pool("mask", tensor.shape[-2], tensor.shape[-1],
                     device=tensor.device, dtype=tensor.dtype)
    if pool is None or count <= 0:
        return None
    if int(t_dim) == 1:
        return _fit_frames(pool, 1, int(count))
    return _fit_frames(pool.permute(0, 2, 1, 3, 4), 2, int(count)).contiguous() \
        .to(device=tensor.device, dtype=tensor.dtype)


def _tail_mask_extra(window, tensor, t_dim: int):
    """驱动流「尾部额外 token」的掩码帧：锚帧用遮罩池，低帧率历史帧填 0（无身份信息）。"""
    plan = getattr(window, "_anchor_plan", None) or {}
    tail_pool = int(plan.get("tail_pool", 0) or 0)
    tail_zero = int(plan.get("tail_zero", 0) or 0)
    if tail_pool + tail_zero <= 0:
        return None
    parts = []
    if tail_pool > 0:
        extra = _mask_pool_frames(tensor, t_dim, tail_pool)
        if extra is not None:
            parts.append(extra)
    if tail_zero > 0:
        shape = list(tensor.shape)
        shape[int(t_dim)] = tail_zero
        parts.append(torch.zeros(shape, device=tensor.device, dtype=tensor.dtype))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=int(t_dim))


def prepend_anchor_mask(window, tensor, t_dim: int, insert_at=None, is_driving_mask: bool = True):
    """供 wan_scail_context.py 的模型 patch 调用：把锚帧/历史帧的遮罩接进「已切片的」掩码张量。

    扩流方案由窗口上的 `_anchor_plan`（条件流注入时写入）决定，保证与驱动流 token 数一致：
      - head > 0      → 前插 head 帧遮罩（driving_only=False 时与驱动流前插配对）；
      - tail_pool > 0 → 尾部追加遮罩池帧（driving_only=True 时的锚帧）；
      - tail_zero > 0 → 尾部追加全 0 帧（低帧率历史帧，无身份信息）。
    t_dim=1 → 原始 driving_mask_28ch 布局 (B,T,28,H,W)；t_dim=2 → sam_latents / ref_mask_latents
    布局 (B,28,T,H,W)（ref 掩码用 insert_at=N 插在 N 帧参考之后）。
    """
    if tensor is None or window is None or not _anchor_on(window):
        return tensor
    if _DRIVING_ONLY and not is_driving_mask:
        return tensor          # 只接驱动条件模式：与待生成序列配对的那路掩码不插
    plan = getattr(window, "_anchor_plan", None) or {}
    head_n = int(plan.get("head", 0) or 0)
    out = tensor
    if head_n > 0:
        extra = _mask_pool_frames(tensor, t_dim, head_n)
        if extra is not None:
            pos = 0 if insert_at is None else int(insert_at)
            out = _insert_at(out, extra, int(t_dim), pos)
    # 尾部额外 token 只属于「驱动流」（与驱动 token 一一对应）；
    # ref 掩码是与视频帧对齐的流，不能带尾部历史帧（否则长度对不上会崩）。
    tail = _tail_mask_extra(window, out, t_dim) if is_driving_mask else None
    if tail is not None:
        out = _insert_at(out, tail, int(t_dim), int(out.shape[t_dim]))
    if out is not tensor:
        pool = _get_pool("mask", tensor.shape[-2], tensor.shape[-1],
                         device=tensor.device, dtype=tensor.dtype)
        if pool is not None:
            window.anchor_mask_latent = pool
        _log_once(("mask_inject", int(t_dim)),
                  "[ContextAnchor] 掩码流已按方案扩流（dim=%d, 前插=%d, 尾部锚=%d, 尾部历史=%d）：%s → %s",
                  int(t_dim), head_n,
                  int(plan.get("tail_pool", 0) or 0), int(plan.get("tail_zero", 0) or 0),
                  list(tensor.shape), list(out.shape))
    return out


# ---------------- RoPE 位置补丁：把「锚帧 / 历史记忆帧」token 放到窗口之前 ----------------

_SCAIL_ANCHOR_ROPE_PATCH_APPLIED = False
_SCAIL_POSE_SHIFT_X = 120.0     # 与 wan_scail_multi_ref / 官方 SCAIL-2 的姿态列偏移保持一致


def _window_rope_plan(window):
    """窗口上的扩流方案（prehistory, memory, stride）；没有额外 token 时返回 None。"""
    if window is None:
        return None
    pre = int(getattr(window, "anchor_prehistory_count", 0) or 0)
    mem = int(getattr(window, "anchor_memory_count", 0) or 0)
    stride = max(1, int(getattr(window, "anchor_memory_stride", 1) or 1))
    if pre <= 0 and mem <= 0:
        return None
    return pre, mem, stride


def _scail_rope_is_patched(mod) -> bool:
    """当前的 SCAILWanModel.rope_encode 是不是本补丁（用于热重载后自愈）。"""
    fn = getattr(getattr(mod, "SCAILWanModel", None), "rope_encode", None)
    return bool(getattr(fn, "_cwa_rope_patch", False))


def _apply_scail_anchor_rope_patch():
    """给 SCAILWanModel.rope_encode 打补丁：把驱动流里的「额外 token」放到窗口之前的位置。

    为什么必须打这个补丁：
      SCAIL-2 的「驱动帧 <-> 生成帧」是按 RoPE 的 t **同位配对**的（pose j <-> video j）。
      若把锚帧/历史帧【前插】在驱动流最前面，整段生成就会相对驱动视频**滞后 K 帧**；
      【追加】到尾部又会被模型当成「窗口之后的未来驱动」。
      本补丁的做法：token 顺序 = [本窗驱动帧][锚帧][低帧率历史帧]，而位置 =
        [N_ref .. N_ref+L-1]（与视频帧同位 => 1:1 配对不变）
        + [N_ref-K .. N_ref-1]（紧邻本窗之前 => 语义 = 进入本窗之前的运动）
        + [更靠前的 stride 间距位置]（低帧率历史摘要，间距如实编码）。
      => 既有几何（视频 / 参考 / 本窗驱动）**完全不变**，额外 token 只作为注意力上下文。

    仅对 SCAIL 的 animation 模式生效；其余情况（replacement 模式 / 无额外 token / 无 window）原样回退。

    ⚠️ 这是**进程级 monkey patch**（改的是类属性）=> 影响所有 SCAIL 模型与工作流，
    且没有保存原函数引用 => 只能靠重启 ComfyUI 还原。
    自愈设计：每次应用前检查「当前 rope_encode 是不是本补丁」，不是就（重新）安装 ——
    这样即使 comfy 核心被热重载冲掉补丁，也能在下次使用时自动恢复，而不是静默失效
    （静默失效会让这些额外 token 被当成「窗口之后的未来驱动」=> 位置错、效果变差，却无任何报错）。
    """
    global _SCAIL_ANCHOR_ROPE_PATCH_APPLIED
    try:
        import comfy.ldm.wan.model as _mod
    except Exception as e:
        logging.warning("[ContextAnchor] 无法导入 comfy.ldm.wan.model，RoPE 位置补丁未应用：%s", e)
        return

    if _scail_rope_is_patched(_mod):
        if not _SCAIL_ANCHOR_ROPE_PATCH_APPLIED:
            _SCAIL_ANCHOR_ROPE_PATCH_APPLIED = True
            logging.info("[ContextAnchor] SCAIL RoPE 位置补丁已在位（无需重复安装）")
        return

    reinstalled = bool(_SCAIL_ANCHOR_ROPE_PATCH_APPLIED)
    try:
        base_rope = _mod.WanModel.rope_encode            # 基础实现（含 _scail_rope_downsample 支持）
        _orig_scail_rope = _mod.SCAILWanModel.rope_encode

        def _call_orig(args):
            self, t, h, w, t_start, steps_t, steps_h, steps_w, device, dtype, pose_latents, \
                reference_latent, ref_mask_flag, transformer_options = args
            return _orig_scail_rope(self, t, h, w, t_start=t_start, steps_t=steps_t, steps_h=steps_h,
                                    steps_w=steps_w, device=device, dtype=dtype, pose_latents=pose_latents,
                                    reference_latent=reference_latent, ref_mask_flag=ref_mask_flag,
                                    transformer_options=transformer_options)

        def _patched_scail_rope(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None,
                                device=None, dtype=None, pose_latents=None, reference_latent=None,
                                ref_mask_flag=None, transformer_options={}):
            args = (self, t, h, w, t_start, steps_t, steps_h, steps_w, device, dtype,
                    pose_latents, reference_latent, ref_mask_flag, transformer_options)
            plan = _window_rope_plan((transformer_options or {}).get("context_window", None))
            if plan is None or pose_latents is None or t_start != 0 \
                    or (ref_mask_flag is not None and not bool(ref_mask_flag)):
                return _call_orig(args)
            try:
                pre, mem, stride = plan
                F_pose = int(pose_latents.shape[-3])
                F_main = F_pose - pre - mem
                if F_main <= 0:
                    raise ValueError(f"驱动流帧数 {F_pose} 不足以容纳额外 token（锚{pre}+历史{mem}）")
                H_pose, W_pose = int(pose_latents.shape[-2]), int(pose_latents.shape[-1])
                ref_t_patches = 0
                if reference_latent is not None:
                    ref_t_patches = (reference_latent.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]

                main_freqs = base_rope(self, t, h, w, t_start=t_start, steps_t=steps_t, steps_h=steps_h,
                                       steps_w=steps_w, device=device, dtype=dtype,
                                       transformer_options=transformer_options)
                pose_tf = {"rope_options": {"shift_y": 0.0, "shift_x": _SCAIL_POSE_SHIFT_X,
                                            "scale_y": 1.0, "scale_x": 1.0},
                           "_scail_rope_downsample": True}
                parts = [base_rope(self, F_main, H_pose, W_pose, t_start=t_start + ref_t_patches,
                                   device=device, dtype=dtype, transformer_options=pose_tf)]
                if pre > 0:
                    # 紧邻本窗之前：N_ref-K .. N_ref-1（步长 1）
                    parts.append(base_rope(self, pre, H_pose, W_pose,
                                           t_start=t_start + ref_t_patches - pre, steps_t=pre,
                                           device=device, dtype=dtype, transformer_options=pose_tf))
                if mem > 0:
                    # 更早的低帧率历史：间距 = stride（不占用任何视频帧的位置）
                    span = stride * mem - (stride - 1)
                    start = t_start + ref_t_patches - pre - stride * mem
                    parts.append(base_rope(self, span, H_pose, W_pose, t_start=start, steps_t=mem,
                                           device=device, dtype=dtype, transformer_options=pose_tf))
                _log_once(("rope_slots",),
                          "[ContextAnchor] RoPE 位置补丁生效：本窗驱动 %d 帧同位配对；"
                          "锚帧 %d 帧放在窗口之前（t=%d..%d）；低帧率历史 %d 帧（间距 %d，t=%d..%d）",
                          F_main, pre,
                          ref_t_patches - pre, ref_t_patches - 1,
                          mem, stride,
                          ref_t_patches - pre - stride * mem, ref_t_patches - pre - stride)
                return torch.cat([main_freqs] + parts, dim=1)
            except Exception:
                logging.exception("[ContextAnchor] RoPE 位置补丁失败，本窗口回退为默认位置")
                return _call_orig(args)

        _mod.SCAILWanModel.rope_encode = _patched_scail_rope
        _patched_scail_rope._cwa_rope_patch = True       # 自愈标记：识别「这就是本补丁」
        _SCAIL_ANCHOR_ROPE_PATCH_APPLIED = True
        if reinstalled:
            logging.warning("[ContextAnchor] RoPE 位置补丁此前被覆盖/丢失，现已重新安装。"
                            "（如非预期，建议重启 ComfyUI 排除其它补丁干扰）")
        else:
            logging.info("[ContextAnchor] SCAIL RoPE 位置补丁已应用（额外驱动 token 放在「窗口之前」）")
    except Exception as e:
        logging.warning("[ContextAnchor] 无法应用 RoPE 位置补丁：%s", e)


# ---------------- 双侧延伸（halo）：改写窗口 index_list ----------------

def _window_original(window) -> list:
    return list(getattr(window, "original_indices", None) or [])


def _window_prefix(window) -> list:
    """窗口的参考前缀帧索引（位于 index_list 最前、且不属于窗口帧）。"""
    original = set(_window_original(window))
    return [i for i in (getattr(window, "index_list", None) or []) if i not in original]


def apply_window_extras(handler, windows, cfg: AnchorConfig) -> None:
    """就地改写每个窗口的 index_list / 头尾只读段 / 写入范围。三件事互相独立：

      1) halo（`head_context_frames`）：在切片最前面再补几帧额外只读前文（越过窗口范围向前取）；
      2) `overlap_scope=single`：**重叠只读化** —— 把「已被前面窗口写完的重叠段」从
         `original_indices` 移到 `head_extra_indices`（只读）⇒ **每帧只有一个作者**（消除共写平均）。
         切片长度不变（那些帧本来就在切片里）⇒ token 预算不变、显存不变；
      3) `anchor_scope=fresh_only`：给「含接续帧钉住区（全局帧 0）」的窗口打 `_anchor_skip`
         ⇒ 它已有真实已知前缀，不再叠合成头（`_anchor_on` 会直接跳过该窗口的所有注入）。
    """
    if not windows:
        return
    halo_head = max(0, int(cfg.halo_head))
    halo_tail = max(0, int(cfg.halo_tail))
    single = (_OVERLAP_SCOPE == "single")
    skip_head = (_ANCHOR_SCOPE == "fresh_only")
    if halo_head == 0 and halo_tail == 0 and not single and not skip_head:
        return
    ref_prefix_len = int(getattr(handler, "prefix_latent_len", 0) or 0)
    w0 = windows[0]
    x_len = int(getattr(w0, "total_frames", 0) or 0)
    if x_len <= 0:
        x_len = max([max(getattr(w, "index_list", None) or [0]) for w in windows] or [0]) + 1

    # 安全网快照（single 校验失败时整体回滚）+ 改写前「被拥有的帧」并集
    snapshot = []
    covered_old = set()
    for w in windows:
        snapshot.append((w, list(getattr(w, "index_list", None) or []),
                         list(getattr(w, "head_extra_indices", None) or []),
                         list(getattr(w, "tail_extra_indices", None) or []),
                         list(getattr(w, "original_indices", None) or []),
                         int(getattr(w, "context_length", 0) or 0)))
        covered_old |= set(getattr(w, "original_indices", None) or [])

    claimed = set()
    for window in windows:
        original = _window_original(window)
        if not original:
            continue
        prefix = _window_prefix(window)
        used = set(prefix) | set(original)
        lo, hi = min(original), max(original)

        # (2) 重叠只读化：本窗里「已被前面窗口写完」的帧 → 只读前文
        read_only_overlap = []
        if single:
            read_only_overlap = sorted(i for i in original if i in claimed)
            keep = [i for i in original if i not in claimed]
            if read_only_overlap and keep and max(read_only_overlap) < min(keep):
                original = keep
            else:
                if read_only_overlap:
                    logging.warning("[ContextAnchor] overlap_scope=single：窗 %s 的只读重叠段不连续"
                                    "（%s…）⇒ 本窗保持共写（保守回退）",
                                    _win_range(window), read_only_overlap[:8])
                read_only_overlap = []

        # (1) halo 头（在「写入段」之前取，且与只读重叠段去重 ⇒ 不增加 token）
        head_halo = []
        if halo_head > 0:
            start = max(ref_prefix_len, min(original) - halo_head)
            for i in range(start, min(original)):
                if i not in used and i not in read_only_overlap:
                    head_halo.append(i)
                    used.add(i)

        tail_halo = []
        if halo_tail > 0:
            for i in range(hi + 1, min(x_len, hi + 1 + halo_tail)):
                if i not in used:
                    tail_halo.append(i)
                    used.add(i)

        window.head_extra_indices = sorted(head_halo + read_only_overlap)
        window.tail_extra_indices = tail_halo
        window.original_indices = list(original)
        window.index_list = list(prefix) + window.head_extra_indices + list(original) + tail_halo
        window.context_length = len(window.index_list)
        # (3) fresh_only：含「接续帧钉住区」（全局帧 0）的窗口 ⇒ 跳过锚定
        window._anchor_skip = bool(skip_head and original and min(original) == 0)

        if single:
            claimed |= set(original) | set(read_only_overlap)

        if cfg.log_debug:
            logging.info(
                "[ContextAnchor] 窗口帧 %d..%d → 写入 %s（切片 %s）| 前缀=%d 只读前文=%d 后延伸=%d",
                lo, hi, f"{min(original)}..{max(original)}",
                f"{window.index_list[0]}..{window.index_list[-1]}",
                len(prefix), len(window.head_extra_indices), len(tail_halo))

    # 安全网：single 模式下「被写入的帧」必须与改写前被拥有的帧**不重不漏**
    if single:
        covered_new, dup = set(), False
        for window in windows:
            for i in window.original_indices:
                if i in covered_new:
                    dup = True
                covered_new.add(i)
        if dup or covered_new != covered_old:
            logging.warning("[ContextAnchor] overlap_scope=single 写入范围校验失败"
                            "（漏=%s 多=%s 重复=%s）⇒ 本步回退为 average（重叠共写）",
                            sorted(covered_old - covered_new)[:8],
                            sorted(covered_new - covered_old)[:8], dup)
            for window, idx, head, tail, orig, clen in snapshot:
                window.index_list, window.head_extra_indices = idx, head
                window.tail_extra_indices, window.original_indices = tail, orig
                window.context_length = clen
                window._anchor_skip = False


# ---------------- 诊断：两窗「重叠区分歧」审计（只读；CWA_DEBUG=1） ----------------

# 回调键（等价于 Custom_context.IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS；
# 这里用字面量以避免与 Custom_context 产生 import 依赖）
_CB_COMBINE = "combine_context_window_results"


def _trim_like_combine(window, tensor, dim: int):
    """按 `combine_context_window_results` 的裁法，把窗口输出裁到 original_indices（诊断用）。

    裁不掉（长度不符）时返回 None —— 说明条件流/方案的账目和融合端不一致。
    """
    prefix_len = int(getattr(window, "prefix_length", 0) or 0)
    anchor_extra = int(getattr(window, "anchor_latent_count", 0) or 0)
    head_extra = len(getattr(window, "head_extra_indices", None) or [])
    tail_extra = len(getattr(window, "tail_extra_indices", None) or [])
    expected = len(getattr(window, "original_indices", None) or [])
    front = prefix_len + anchor_extra + head_extra
    total = int(tensor.shape[dim])
    stop = total - tail_extra if total - tail_extra > front else front
    out = _slice_frames(tensor, dim, slice(front, stop))
    if expected <= 0 or int(out.shape[dim]) != expected:
        return None
    return out


def _diag_combine(handler, x_in, sub_conds_out, sub_conds, window, window_idx,
                  total_windows, timestep, conds_final, counts_final, biases_final):
    """两窗一致性审计（只读；`debug_log=1` 时输出）——**两个口径**：

      · **level**：两窗对**同一全局帧**的预测差异（外观/位置差多少）；
      · **动作**：两窗各自的**逐帧动作**（本窗 x0 的相邻帧差）之差，并用「每帧动作量」归一化
        ⇒ 这才是"人的动作变了没有"的量纲（level 差很小时动作仍可能完全不同）。

    `overlap_scope=single` 时，后一窗对这些帧是"只读前文"⇒ 同一比较就是**续写一致性**。

    位置映射：模型输出长度 = 锚帧数 + len(window.index_list)（锚帧前插在最前）⇒
    `pos(f) = k_anchor + index_list.index(f)`（对「共写」与「只读」两种模式都成立）。
    """
    if not _DEBUG:
        return
    try:
        dim = int(getattr(handler, "dim", 2) or 2)
        out = sub_conds_out[0] if isinstance(sub_conds_out, (list, tuple)) and len(sub_conds_out) else sub_conds_out
        if not torch.is_tensor(out):
            return
        idx_list = [int(i) for i in (getattr(window, "index_list", None) or [])]
        if not idx_list:
            return
        k_anchor = int(getattr(window, "anchor_latent_count", 0) or 0)
        total = int(out.shape[dim])
        if total != k_anchor + len(idx_list):
            _dbg("[ContextAnchor][diag] 窗#%d 输出长度 %d ≠ 锚%d+切片%d，跳过分歧审计",
                 window_idx, total, k_anchor, len(idx_list))
            return
        cur = out.detach().float().cpu()
        pos = {f: k_anchor + i for i, f in enumerate(idx_list)}
        prev = _DIAG.get("prev_out")
        _DIAG["prev_out"] = (window_idx, pos, cur)
        if prev is None:
            return
        p_idx, p_pos, p_out = prev
        shared = sorted(set(pos) & set(p_pos))
        if not shared:
            return
        try:
            sigma = float(timestep.reshape(-1)[0])
        except Exception:
            sigma = 0.0

        def _one(t, p):
            return _slice_frames(t, dim, slice(p, p + 1))

        dl, per_frame = [], []
        v_cur, v_prev, per_frame_mot = [], [], []
        for f in shared:
            a, b = _one(cur, pos[f]), _one(p_out, p_pos[f])
            dl.append((a - b).abs())
            per_frame.append(float(dl[-1].mean()))
            if (f - 1) in pos and (f - 1) in p_pos:
                ma = _one(cur, pos[f]) - _one(cur, pos[f - 1])
                mbm = _one(p_out, p_pos[f]) - _one(p_out, p_pos[f - 1])
                v_cur.append(ma)
                v_prev.append(mbm)
                per_frame_mot.append(float((ma - mbm).abs().mean()))
        if not dl:
            return
        d = torch.cat(dl, dim=dim)
        level_mean = float(d.mean())
        level_p95 = float(d.flatten().quantile(0.95)) if d.numel() else 0.0
        ref_val = float(torch.cat([_one(cur, pos[f]) for f in shared], dim=dim).abs().mean()) + 1e-6
        motion_txt = "（动作：本窗无相邻帧可算）"
        if v_cur:
            ma_all = torch.cat(v_cur, dim=dim)
            mb_all = torch.cat(v_prev, dim=dim)
            dm = float((ma_all - mb_all).abs().mean())
            mref = 0.5 * (float(ma_all.abs().mean()) + float(mb_all.abs().mean())) + 1e-6
            motion_txt = (f"动作:|Δ动作|均={dm:.4f}（每帧动作量={mref:.4f}，比值={dm / mref:.3f}）"
                          f" 逐帧={[round(v, 3) for v in per_frame_mot]}")
        mode = "只读前文→续写一致性" if _OVERLAP_SCOPE == "single" else "重叠区共写"
        _dbg("[ContextAnchor][diag] σ=%.3f 窗#%d↔#%d 共 %d 帧（%s）%d..%d："
             "level:|Δ|均=%.4f p95=%.4f 相对=%.3f 逐帧=%s | %s",
             sigma, p_idx, window_idx, len(shared), mode, shared[0], shared[-1],
             level_mean, level_p95, level_mean / ref_val,
             [round(v, 3) for v in per_frame], motion_txt)
    except Exception:
        logging.debug("[ContextAnchor] 重叠区诊断失败（忽略）", exc_info=True)


def _register_overlap_diag(handler) -> None:
    """把重叠区诊断挂到 handler 的 callbacks（结构见 comfy.patcher_extension.get_all_callbacks：
    `{"callbacks": {call_type: {name: [fn, ...]}}}`）；只读，不影响采样。"""
    cbs = getattr(handler, "callbacks", None)
    if isinstance(cbs, dict) and "callbacks" in cbs:
        new = {"callbacks": {k: dict(v) for k, v in (cbs.get("callbacks") or {}).items()}}
    else:
        if cbs:
            logging.debug("[ContextAnchor] handler.callbacks 结构未知（%s），诊断回调将重建该字段", type(cbs))
        new = {"callbacks": {}}
    slot = new["callbacks"].setdefault(_CB_COMBINE, {})
    slot["cwa_overlap_diag"] = [_diag_combine]
    handler.callbacks = new


# ---------------- 安装（S1：改为 handler_wrap 包装链） ----------------
# 旧实现（已删）直接在实例上 `handler.execute = execute` 并自存 `_cwa_originals`；
# 第二个节点也想包 handler 时会互相覆盖/嵌套。现在统一走 `handler_wrap.install()`，
# 参见下方 `_make_*` 工厂 + `_install`。

# 包装链里的 owner 标识：同一 owner 重复安装 = 原地替换（不叠加），换 cfg 重装即生效
_WRAP_OWNER = "ContextWindowAnchor"


def _make_get_context_windows(handler, cfg: AnchorConfig):
    """工厂：把「每步窗口规划 + 锚定/只读/记忆组改写」包在下一层之外。

    这是「每步」的分界点：窗口对象每步重建，故在此清空诊断状态并轮转跨窗共享头。
    """
    def factory(next_call):
        def get_context_windows(model, x_in, model_options):
            _DIAG.clear()          # 每步清空诊断状态（窗口对象每步重建 => 这里就是「步」的边界）
            # 步边界轮转「跨窗共享头」：上一步 leader 产出的头 -> 本步全员（含 leader）共用同一份
            _advance_shared_head((model_options or {}).get("context_handler", None))
            windows = next_call(model, x_in, model_options)
            try:
                for w in windows:
                    w._anchor_active = True    # 运行期标记：绕过本节点时窗口上没有该标记
                apply_window_extras(handler, windows, cfg)
                logging.info("[ContextAnchor] %s｜窗口边界: %s",
                             _sigma_step_str(handler, model_options), describe_windows(windows))
            except Exception:
                logging.exception("[ContextAnchor] 窗口改写失败，本步回退为原始窗口")
            return windows
        return get_context_windows
    return factory


def _make_get_resized_cond(cfg: AnchorConfig, dim: int):
    """工厂：把「锚定帧 / 记忆组注入条件流」包在下一层之外。"""
    cond_active = cfg.anchor_frames is not None or int(cfg.memory_frames or 0) > 0

    def factory(next_call):
        def get_resized_cond(cond_in, x_in, window, device=None):
            resized = next_call(cond_in, x_in, window, device)
            if not cond_active:
                return resized
            try:
                full_pose = None
                if int(cfg.memory_frames or 0) > 0:
                    # 记忆组需要「尚未切片的全量驱动流」来切出本窗之前的历史帧
                    full_pose = _find_full_stream(cond_in, ANCHOR_POSE_KEYS)
                return inject_anchor_conds(window, resized, x_in, dim=dim,
                                           only_pose=cfg.driving_only, full_pose=full_pose)
            except Exception:
                logging.exception("[ContextAnchor] 锚定帧处理失败，本次跳过")
                return resized
        return get_resized_cond
    return factory


def _make_execute(cfg: AnchorConfig, dim: int):
    """工厂：把「待生成序列头部的锚帧注入 + x0 捕获」包在下一层之外。

    仅在 `driving_only=关` 且有锚帧时介入（SCAIL 的 `driving_only=开` 只走驱动流，不碰 x）。
    """
    def factory(next_call):
        def execute(calc_cond_batch, model, conds, x_in, timestep, model_options):
            if cfg.anchor_frames is None or cfg.driving_only:
                return next_call(calc_cond_batch, model, conds, x_in, timestep, model_options)
            try:
                sigma = float(timestep.reshape(-1)[0])
            except Exception:
                sigma = 0.0

            def calc_cond_batch_wrapped(m, c, sub_x, ts, mo, **kw):
                extended = inject_anchor_x(mo, sub_x, dim=dim, sigma=sigma)
                out = calc_cond_batch(m, c, extended, ts, mo, **kw)
                _capture_anchor_x0(mo, extended, sub_x, out, dim)
                return out

            return next_call(calc_cond_batch_wrapped, model, conds, x_in, timestep, model_options)
        return execute
    return factory


def _install(handler, cfg: AnchorConfig):
    """把「锚定注入」挂到 handler 的**包装链**上（见 handler_wrap.py）。

    与旧实现的差别（按设计行为不变，只换了挂载方式）：
      · 不再自存 `_cwa_originals`：真原始由包装链统一解析（优先取**类**方法）
        => 浅拷贝（`copy.copy(handler)`）与重复安装都不会叠加包装层；
      · 同一 owner 重复安装 = 原地替换，闭包里的 cfg 随之更新
        （旧实现靠每次用 `_cwa_originals` 重绑达到同样效果）；
      · 三个方法**互相独立**地安装（旧实现在 get_resized_cond / execute 任一缺失时两个都不装）。
    """
    dim = int(getattr(handler, "dim", 2) or 2)
    applied = _wrap_install(handler, _WRAP_OWNER, {
        "get_context_windows": _make_get_context_windows(handler, cfg),
        "get_resized_cond": _make_get_resized_cond(cfg, dim),
        "execute": _make_execute(cfg, dim),
    })
    logging.debug("[ContextAnchor] 已挂载包装链：%s（本次安装 %s）", _wrap_describe(handler), applied)
    # ---- 诊断（只读）：两窗重叠区分歧审计；CWA_DEBUG=1 时输出 ----
    try:
        _register_overlap_diag(handler)
    except Exception:
        logging.exception("[ContextAnchor] 注册重叠区诊断失败（忽略）")
    _log_once(("space",), "[ContextAnchor] 锚定池空间：%s（latent 流=模型/采样空间，mask 不归一化）；"
                          "诊断日志：%s", _PROCESS_LATENT_IN_NAME,
              "开" if _DEBUG else "关（设 CWA_DEBUG=1 后重启）")
    handler._cwa_anchor_installed = True
    handler._anchor_cfg = cfg
    return handler


# ---------------- 节点 ----------------

class WanContextWindowAnchorNode:
    """上下文窗口「统一基准锚定 + 只读上下文 + 长程记忆组」节点（实验性）。

    可选输入：anchor_frames（要当基准的那几帧，**像素帧**，来自驱动视频）、vae（同一个 Wan VAE）、
    SCAIL-2 再加 anchor_mask_frames（同帧号的驱动遮罩）。
    只读上下文（head_context_frames）与低帧率历史记忆组（memory_frames / memory_stride）
    **不需要任何输入**，纯参数即可；这三个参数的**单位都是 latent 帧**。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已接入上下文窗口节点的模型（`Custom Context Windows (Manual)` 或 `Wan SCAIL-2 Context Windows` 的 model 输出）。Model that already has a context windows node applied."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "总开关；关闭时模型直接透传。Master switch."}),
                "driving_only": ("BOOLEAN", {"default": True,
                    "tooltip": "只把锚帧接到【驱动流 + 驱动遮罩】，不注入待生成序列（不会渗出锚帧画面、不会有伪影）。\n"
                               "开=锚帧作为「额外的驱动 token」追加在驱动流尾部，并由 RoPE 放在【本窗口之前】(不产生帧错位)；\n"
                               "关=锚帧前插到驱动流与待生成序列（与视频帧逐帧配对，最接近训练语义，采样后裁掉）。\n"
                               "animate2 会因模型硬校验自动改回「否」。\nTrue = anchors are extra driving tokens placed *before* the window (no frame shift). animate2 forces this to False."}),
                "head_context_frames": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1,
                    "tooltip": "每个窗口头部纳入几帧「只读前文」（单位 = **latent 帧**）：参与注意力、不参与融合。\n"
                               "内容 = 共享 latent `x` 上那几帧：在 overlap=0（单作者）下其作者就是相邻窗口。\n"
                               "实测：2 帧正常；6 帧会让总 token 达 27（训练长度 21）并出现肢体变形 => 建议 ≤2~3。0=关闭。\n"
                               "（尾段已移除：它只影响注意力、不改变结果。）\n"
                               "How many read-only preceding frames each window may see, in LATENT frames. 0=off."}),
                "memory_stride": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "【长程记忆组的抽样间距】(单位 = latent 帧)\n"
                               "0 = 自动（推荐）：让 memory_frames 个点刚好覆盖「片段开头 → 本窗口之前」的全部历史\n"
                               "     => 片段/窗口再长也覆盖得到开头（间隔自动变大、分辨率变粗）。\n"
                               ">0 = 固定间距（4 ≈ 每 4 latent / 约 16 像素帧取 1 帧）。\n"
                               "Sampling stride (latent frames) for the memory group. 0 = auto (spread over the whole history)."}),
                "memory_frames": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "【长程记忆组】取本窗口【之前】几个历史驱动帧（单位 = **latent 帧**，不是 token 数）：\n"
                               "作为额外驱动 token 追加，由 RoPE 放在「窗口之前」并按真实间距编码（不与任何视频帧配对）。\n"
                               "0=关闭；间距见 memory_stride（默认 0=自动覆盖全部历史）。\n"
                               "How many low-frame-rate history LATENT frames to append. 0=off."}),
                "anchor_fixed": ("BOOLEAN", {"default": False,
                    "tooltip": "仅 driving_only=关 时有效。开=锚帧 latent 在所有窗口/所有步上都是同一个值（真正的「恒定共同基准」）；"
                               "关=锚帧走自己的生成轨迹（用上一步模型对这几帧的预测继续演化，与其他帧同构）。\n"
                               "Only for driving_only=False. True = identical anchor latent across all windows."}),
                "anchor_image_blank": ("BOOLEAN", {"default": False,
                    "tooltip": "【仅 driving_only=关 时有意义】待生成序列里那几帧锚帧要不要写入 anchor_frames 的画面内容。\n"
                               "开 = 只放【纯噪声占位】（不写锚帧画面）：锚帧素材来自驱动视频=源人物画面，写进画面侧会泄漏源形象；\n"
                               "     开这个可避免，同时仍保留那几帧的位置（模型自己在那 K 帧里完成「参考图 → 当前姿态」的迁移）。\n"
                               "关 = 保持现状（锚帧画面内容会被写入）。开时 anchor_fixed 失去意义（会被忽略）。\n"
                               "Blank the anchor frames inside the generation sequence (noise placeholder only)."}),
                "debug_log": ("BOOLEAN", {"default": False,
                    "tooltip": "【诊断日志】开 = 打印「锚定池空间 / 每窗条件流布局 / 锚段自检 / 跨窗共享头 / "
                               "两窗重叠区预测分歧（= 重影的直接量纲）」；\n"
                               "关 = 由哨兵文件 CWA_DEBUG.on 或环境变量 CWA_DEBUG 决定（默认全关，零影响）。\n"
                               "⇒ 做开/关 A/B 时直接切这个开关即可，**不用重启 ComfyUI**。\n"
                               "Diagnostic logs (pool space / per-window layout / shared head / overlap divergence)."}),
                "anchor_scope": (["fresh_only", "all"], {"default": "fresh_only",
                    "tooltip": "【锚定范围】fresh_only（默认）= **跳过含「接续帧钉住区」（全局帧 0）的窗口**——"
                               "它已经有真实已知前缀（上一段的尾部），再叠合成头会互相干扰；\n"
                               "all = 每个窗口都插（旧行为，头会排在接续帧之前）。\n"
                               "Whether to skip the window that already owns the pinned continuation frames."}),
                "shared_head": (["off", "leader", "consensus"], {"default": "off",
                    "tooltip": "【锚帧内容的产生方式】(仅 driving_only=关 有意义)\n"
                               "off（默认）= **每个窗口各自生成**自己的锚帧内容（= 手动测试里两段各自生成的等价物）；\n"
                               "leader = 跨窗共享：本步第一个窗口产出的头 → 下一步全员共用；\n"
                               "consensus = 跨窗共享：各窗头内容取平均（无偏置但是混合值）。\n"
                               "How the anchor block's content is produced across windows."}),
                "overlap_scope": (["average", "single"], {"default": "average",
                    "tooltip": "【重叠帧的写入方式】average（默认）= 现状：重叠帧被相邻两窗**共写**，融合时加权平均；\n"
                               "single = **重叠只读化**：已被前一个窗口写完的重叠段在本窗改为「只读前文」"
                               "（仍参与注意力、不写回）⇒ **每帧只有一个作者**，消除「两份动作被平均」；\n"
                               "切片长度/token 预算不变（那些帧本来就在切片里）。\n"
                               "average: frames co-written by both windows. single: read-only context only (one author per frame)."}),
            },
            "optional": {
                "anchor_frames": ("IMAGE", {"tooltip": "【基准帧】驱动/pose 视频里你想当统一基准的那几帧（例如整段视频开头几帧）。**给多少就用多少**（没有数量参数）：会 VAE 编码成 latent，给每个窗口一份共同的「前文/基准」运动。不接=不做锚定（仍可使用只读上下文与长程记忆组）。\nAnchor basis frames (driving/pose video): how many you feed is how many are used. Leave unconnected to skip anchoring."}),
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
        "上下文窗口「统一基准锚定 + 只读上下文 + 长程记忆组」节点（⚠️ 实验性：有效性与副作用需自行开/关对比验证）。\n"
        "1) 锚定帧：给每个窗口一份共同的「基准/前文」运动（anchor_frames，可选）；\n"
        "     driving_only=开时作为额外驱动 token 追加，并由 RoPE 放在「窗口之前」=> 不产生帧错位；\n"
        "     driving_only=关时前插到驱动流与待生成序列（逐帧配对，采样后裁掉）。\n"
        "2) 只读上下文：head_context_frames —— 每个窗口头部各看几帧真实邻居（单位 = latent 帧；参与注意力、不参与融合），\n"
        "     配合 overlap=0 可让每一帧只有一个「作者」，从根上避免融合平均造成的重影。\n"
        "     （尾段已移除：它只影响注意力、不改变结果。）\n"
        "3) 长程记忆组：memory_frames —— 取本窗【之前】的历史驱动帧作为额外 token（单位 = latent 帧）；\n"
        "     间距由 memory_stride 决定（0=自动，覆盖「片段开头 → 本窗口之前」的全部历史）。\n"
        "不接锚帧、也不开 2)/3) => 完全等价于不装本节点。必须接在上下文窗口节点之后。\n"
        "Global-basis anchoring + read-only context + long-range motion memory for context windows (experimental)."
    )

    def apply_anchor(self, model, enabled=True, driving_only=True, head_context_frames=0,
                     memory_stride=0, memory_frames=0, anchor_fixed=False, anchor_image_blank=False,
                     debug_log=False, anchor_scope="fresh_only", shared_head="off", overlap_scope="average",
                     anchor_frames=None, anchor_mask_frames=None, vae=None):
        refresh_debug(bool(debug_log))   # 诊断开关：节点参数 > 哨兵文件 > 环境变量（免重启）
        frames = _as_image_batch(anchor_frames)
        mask_frames = _as_image_batch(anchor_mask_frames)
        halo_head = max(0, int(head_context_frames or 0))
        # 尾段（halo_tail）已从面板移除：机制上它只影响注意力、不改变结果
        # （其预测在融合前被裁掉，且这些帧的「作者」是后面的窗口）。内部能力保留，恒为 0。
        # 注意：新参数 memory_stride 是**占用原 tail_context_frames 的槽位**加进来的
        # => 老工作流的该位置值（通常是 0）会落到 memory_stride 上，而 0 恰好 = 自动，语义兼容。
        halo_tail = 0
        memory = max(0, int(memory_frames or 0))
        m_stride = max(0, int(memory_stride if memory_stride is not None else 0))
        # 画面侧锚帧是否只用噪声占位（不写锚帧画面内容）。开启时 anchor_fixed 失去意义。
        blank_image = bool(anchor_image_blank)
        if blank_image and bool(anchor_fixed):
            logging.info("[ContextAnchor] anchor_image_blank=开 => 画面侧锚帧固定为噪声占位，"
                         "anchor_fixed 已失去意义（本次按忽略处理）")

        if not enabled or (frames is None and halo_head == 0 and halo_tail == 0 and memory == 0
                           and str(overlap_scope) != "single"):
            reset_pools(None, None, None)
            logging.info("[ContextAnchor] 未启用（enabled=%s, 锚帧=%s, 头只读=%d, 记忆组=%d, "
                         "overlap_scope=%s），模型直接透传",
                         enabled, "未接" if frames is None else f"{frames.shape[0]} 像素帧",
                         halo_head, memory, overlap_scope)
            return (model,)

        handler = model.model_options.get("context_handler", None)
        if handler is None:
            raise ValueError(
                "[ContextAnchor] 模型上没有 context_handler：请把本节点接在 "
                "WAN Context Windows (Manual Custom) / Wan SCAIL Context Windows 之后。")
        if frames is not None and vae is None:
            raise ValueError(
                "[ContextAnchor] 接了 anchor_frames 但没有接 vae：基准帧必须用（与 conditioning 相同的）"
                "Wan VAE 编码成 latent。")
        if frames is not None and mask_frames is not None and mask_frames.shape[0] != frames.shape[0]:
            logging.warning("[ContextAnchor] anchor_mask_frames 帧数(%d) 与 anchor_frames 帧数(%d) 不一致，"
                            "遮罩会按自己的帧数编码", mask_frames.shape[0], frames.shape[0])
        if frames is not None and mask_frames is None and _needs_driving_mask(model):
            raise ValueError(
                "[ContextAnchor] SCAIL-2 必须同时接 anchor_mask_frames（与 anchor_frames 同帧号的驱动彩色遮罩）；"
                "否则驱动流与驱动遮罩帧数不一致，会在模型内部以形状不匹配报错。")

        # driving_only 只对 SCAIL 型模型有意义：SCAIL 的驱动遮罩是逐 token 加进驱动流的，必须与驱动同帧数，
        # 所以「锚帧只走驱动侧」在 SCAIL 上成立；而 animate2 有硬校验 pose = f_gen - 1，
        # 只接驱动侧会让帧数不符 => 必须同时注入待生成序列。
        # （此前这里写反了：`and not _needs_driving_mask(model)` —— 结果是 SCAIL 被强制成前插、animate2 反而错，
        #   与日志/文档描述正好相反。）
        use_driving_only = bool(driving_only) and _needs_driving_mask(model)
        if bool(driving_only) and not use_driving_only:
            logging.info("[ContextAnchor] 当前模型不是 SCAIL（animate2 等有 pose=生成帧-1 的硬校验），"
                         "已自动改为「同时注入待生成序列」。")

        # 长程记忆组走的是 SCAIL 的驱动分支 + 专门的位置槽 → 其它模型不支持
        if memory > 0 and not _needs_driving_mask(model):
            logging.warning("[ContextAnchor] 长程记忆组只对 SCAIL 的驱动流生效（当前模型不是 SCAIL），已自动关闭。")
            memory = 0

        k = reset_pools(frames, mask_frames, vae, driving_only=use_driving_only,
                        memory_frames=memory, memory_stride=m_stride,
                        anchor_fixed=bool(anchor_fixed) and not blank_image,
                        anchor_image_blank=blank_image, model=model,
                        anchor_scope=anchor_scope, shared_head=shared_head, overlap_scope=overlap_scope)

        # 编码自检（极小尺寸试编）：失败就在采样前给出明确错误，而不是在模型深处以形状不匹配崩掉
        if frames is not None:
            for kind in ("x", "pose") + (("mask",) if mask_frames is not None else ()):
                try:
                    _encode_pool(kind, 4, 4)
                except Exception as e:
                    raise ValueError(f"[ContextAnchor] 锚帧编码自检失败（{kind}）：{e}") from e

        cfg = AnchorConfig(halo_head=halo_head, halo_tail=halo_tail, anchor_frames=frames,
                           anchor_mask_frames=mask_frames, vae=vae, driving_only=use_driving_only,
                           memory_frames=memory, memory_stride=m_stride,
                           anchor_fixed=bool(anchor_fixed) and not blank_image,
                           anchor_image_blank=blank_image, log_debug=bool(_DEBUG))

        model = model.clone()
        new_handler = copy.copy(handler)
        _install(new_handler, cfg)
        model.model_options["context_handler"] = new_handler

        # 驱动流有额外 token（锚帧按「窗口之前」放置 / 长程记忆组）→ 需要 RoPE 位置补丁
        if _needs_driving_mask(model) and ((use_driving_only and k > 0) or memory > 0):
            _apply_scail_anchor_rope_patch()

        logging.info("[ContextAnchor] 已启用：锚帧 %s（%d latent，%s）；头只读=%d；"
                     "长程记忆组=%d（间距=%s）；锚帧恒定=%s；画面侧锚帧=%s",
                     "未接" if frames is None else f"{frames.shape[0]} 像素帧", k,
                     "只接驱动流（放在窗口之前，不错位）" if use_driving_only else "前插驱动流+待生成序列（逐帧配对）",
                     halo_head, memory,
                     "自动（覆盖全历史）" if m_stride == 0 else f"固定 {m_stride} latent",
                     bool(anchor_fixed) and not blank_image,
                     "纯噪声占位（不写锚帧画面）" if blank_image else "写入锚帧画面内容")
        logging.info("[ContextAnchor] 实验开关：anchor_scope=%s；shared_head=%s；overlap_scope=%s"
                     "（single=重叠只读化/每帧一个作者；fresh_only=跳过含接续帧的窗口）",
                     _ANCHOR_SCOPE, _SHARED_HEAD_MODE, _OVERLAP_SCOPE)
        logging.info("[ContextAnchor] 锚帧占位噪声：种子=%s（固定值 => 跨窗口/跨运行逐位一致；"
                     "可用环境变量 CWA_ANCHOR_NOISE_SEED 覆盖后重启，做「换一块噪声是否影响结果」的对比实验）",
                     hex(_ANCHOR_NOISE_SEED))
        return (model,)


# ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "WanContextWindowAnchorNode": WanContextWindowAnchorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowAnchorNode": "Context Window Anchor + Halo (窗口锚定/双侧延伸)",
}



