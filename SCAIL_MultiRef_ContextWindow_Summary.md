# SCAIL 多参考图 + 上下文窗口节点 开发总结

## 一、任务目标

为 ComfyUI-Wan 的 SCAIL-2 模型实现：
1. **多参考图支持**（`wan_scail_multi_ref.py`）— 已有，基于 `nodes_scail.py` 官方节点扩写
2. **上下文窗口支持**（`wan_scail_context.py`）— 本次新建，解决 SCAIL conditioning 在长视频采样中的窗口切分问题

---

## 二、文件结构

| 文件 | 作用 | 来源 |
|------|------|------|
| `wan_scail_multi_ref.py` | SCAIL 多参考图 conditioning 节点 | 基于官方 `nodes_scail.py` 扩写 |
| `wan_scail_context.py` | SCAIL 上下文窗口控制节点 | 本次新建 |
| `Custom_context.py` | 通用上下文窗口基础设施 | 已有，基于 AnimateDiff 的 `context.py` |
| `__init__.py` | 节点注册 | 本次修改增加 `SC` 导入 |

---

## 三、SCAIL conditioning 字段的时间维布局

这是整个任务中**最核心的知识点**。SCAIL 的不同 conditioning 字段时间维位置不一致：

| 字段 | shape | 时间维 | 含义 | 在 conditioning 中的位置 |
|------|-------|--------|------|------------------------|
| `reference_latents` | `(1, 16, N, H/8, W/8)` | **dim=2** | N 张参考图，**全局保留** | 顶层 list[Tensor] |
| `pose_video_latent` | `(B, 16, T, H/16, W/16)` | **dim=2** | 每帧 pose 骨架 | dict → `.cond` wrapped |
| `driving_mask_28ch` | `(1, T, 28, H, W)` | **dim=1** | 驱动视频的 28通道 mask | 顶层 tensor 或 dict 中 |
| `ref_mask_28ch` | `(1, N+T, 28, H, W)` | **dim=1** | 前 N 帧参考 mask + 后 T 帧零 | 顶层 tensor 或 dict 中 |
| `ref_mask_flag` | bool | 无 | replacement_mode 标志 | dict 中 |
| `clip_vision_output` | 对象 | 无 | CLIP 视觉特征 | dict 中 |

**关键观察**：`driving_mask_28ch` 和 `ref_mask_28ch` 的时间维在 dim=1，而其他视频相关字段在 dim=2。这是因为 `_extract_mask_to_28ch()` 函数输出的是 `(1, T, 28, H, W)` 格式——28 通道编码了 7种颜色×4帧压缩的信息，没有独立的 C 维。

---

## 四、WAN VAE encode 的行为

**非常重要**：WAN 的 `WanVAE.encode()` 是**逐帧编码**的，没有 4:1 时间维压缩。

```python
# WanVAE.encode (comfy/ldm/wan/vae.py)
def encode(self, x):
    # 按 4N+1 取整后，拆成 1 + 2×(N-1)/2 组逐次编码
    t = x.shape[2]
    t = 1 + ((t - 1) // 4) * 4
    iter_ = 1 + (t - 1) // 2
    # ...
```

输入 `(4N-3, H, W, 3)` → 输出 `(1, 16, 2N-1, H/8, W/8)`。**不是 N**。

两种编码模式的区别：
- **"1+4n批量"**：像素帧展开为 `4N-3` 帧整体编码 → latent = `2N-1` 帧
- **"逐帧编码"**：每帧独立编码 → latent = `N` 帧

实测证实两者 latent 时间维都是 N（用户实测）。代码注释标注 `4N-3` 是**错的**。

---

## 五、上下文窗口架构

基于 `Custom_context.py` 中的 `IndexListContextHandler` 框架，通过 `model.model_options["context_handler"]` 注入到 KSampler 内部。

### 核心流程

```
KSampler → calc_cond_batch → handler.execute()
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
         get_resized_cond()       evaluate_context_windows()
         切片 conditioning         切片 x_in，逐个窗口推理
                    │                       │
                    ▼                       ▼
          model patch _patched_forward()   combine_context_window_results()
          切片 ref_mask_latents            加权融合各窗口结果
```

### 切片分工

| 阶段 | 处理字段 | 说明 |
|------|---------|------|
| `get_resized_cond` | `pose_video_latent`（dim=2，`.cond` wrapped） | 用 `range(context_length)` 切片 |
| `_patched_forward`（模型 patch） | `ref_mask_latents`（dim=2，ComfyUI movedim 后） | 用 `window.index_list` 切片 |
| `_patched_forward` | `driving_mask_28ch`（从 kwargs pop） | 用 `window.index_list` 切片 |
| 模型 forward 自动 | `reference_latent`（kwargs） | **不切片**，全局保留 |

### `ref_mask_latents` 的双重映射

conditioning 中的 key 是 `ref_mask_28ch`，但模型 `_forward()` 的命名参数是 `ref_mask_latents`。ComfyUI 会自动匹配同名参数（去掉后缀），所以 `ref_mask_latents` 在 `_forward` 中作为命名参数存在，不在 `**kwargs` 中。

---

## 六、Bug 记录

### Bug 1: ref_mask_28ch 切片未生效（"光点"问题）

**症状**：画面出现随机光点、马赛克、色彩偏移。

**根因**：模型 patch 中检查了命名参数 `ref_mask_latents`，但最初错误的版本检查了 `kwargs["ref_mask_28ch"]`，永远为 None。同时 `get_resized_cond` 中的切片被改成了透传（为了去双重切片），导致两处都没切片。

**修复**：patch 中检查 `ref_mask_latents` 命名参数。

### Bug 2: pose_video_latent 切片越界（累积瑕疵 + 光点 + 偏色）

**症状**：第二段开头（保护帧附近）偏亮、光点、抖动；50 多帧出现马赛克；色彩逐段漂移。

**根因**：`pose_video_latent` 用 `window.original_indices` 切片。第二段的窗口1 `original_indices=[5,6,...,25]`，但 `pose_video_latent` 已被 `video_frame_offset` 截断为 21 帧。索引 25 越界，在 `try/finally` 中被静默忽略，产生错误条件。

**修复**：改为 `range(self.context_length)`（始终 `[0,...,20]`）。

**教训**：conditioning 中视频字段的长度已被 `WanSCAILToVideoMultiRef` 用 `video_frame_offset` 截断过，它们的第 0 帧就是当前 chunk 的第 0 帧。不应该用全局的窗口索引（`original_indices`）去切已经被截断的 tensor。

### Bug 3: reference_latent 被误 pop（推理中断）

**症状**：模型 forward 中 `reference_latent=None`，x 不拼接，维度不匹配崩溃。

**根因**：patch 中 `pop("reference_latent")` 导致变量被移除。

**修复**：改为 `get()` 不 pop。

---

## 七、关于 prefix 机制的结论

**SCAIL 不需要 prefix。**

- WanAnimate 用 prefix 把参考图重复拼到每个窗口前
- SCAIL 的 `reference_latents` 是全局条件**不切片**，每个窗口自动完整可见
- SCAIL 的 `ref_mask_28ch` 前 N 帧保留不切，后 T 帧才切片
- prefix 机制在 SCAIL 中反而会造成 `original_indices != index_list`，引发维度不匹配

因此 `WanSCAILContextWindowsNode` 已移除 `prefix_latent_num` 参数，`prefix_latent_len=0` 固定。

---

## 八、段间过渡问题（排查记录）

### 问题背景

`previous_frames`（前一段尾帧）写入 latent 前几帧，通过 `noise_mask=0` 保护。但保护帧和生成帧之间 `noise_mask` 直接从 0 跳到 1，导致第二段开头出现偏亮、偏蓝、光点、抖动。

### 尝试过的方案（全部无效）

| # | 方案 | 修改层面 | 结果 |
|---|------|---------|------|
| 1 | 增大保护帧长度（5→17帧） | 工作流参数 | 无效，问题推迟到保护帧后 |
| 2 | noise_mask 线性渐变（中间值） | `wan_scail_multi_ref.py` | 画面崩坏（混合分布模型不识别） |
| 3 | protect_frame_noise（0.03~0.10） | `wan_scail_multi_ref.py` | 无改善 |
| 4 | 像素域整体 VAE 编码（中性灰画布） | `wan_scail_multi_ref.py` | 无改善 |
| 5 | ref_mask_28ch 长度压缩 | `wan_scail_multi_ref.py` | ComfyUI crash（维度不匹配） |
| 6 | 前端每段增加过渡帧 | 工作流层面 | 无改善 |
| 7 | 保护帧区域 conditioning 置零 | `wan_scail_multi_ref.py` | 光点乱飞 |
| 8 | differential diffusion（渐进释放） | 采样器层面 | 理论分析发现 4-6 步不生效（未实施） |

### 最终结论

**分段采样的本质缺陷**：前一段尾帧是已完整去噪的最终输出，后一段开头是纯随机噪声，两段初始状态"干净程度"不同。即使通过 noise_mask 保护，模型在边界处也必然产生不连续响应。

一次性上下文窗口采样（不分段）则没有这个问题——所有帧同时生成，窗口融合只是推理时做的切分，不影响最终一致性。

### 当前文件状态

- `wan_scail_multi_ref.py`（`WanSCAILToVideoMultiRef`）：
  - `previous_frames` 机制保留（硬切 noise_mask=0/1）
  - 调试日志保留（`[WanSCAIL_COND]`、`[WanSCAIL]`）
  - 无 `protect_frame_noise`、`noise_mask_gradient`、`mask_video_length` 等额外参数
  - conditioning 置零已移除
- `wan_scail_context.py`：调试日志保留（`[SCAIL_DEBUG]`）
- 所有 `.bak` 备份和 `Diagnostics-*.log` 已清理

---

## 九、当前已知问题

1. **段间过渡偏亮/偏蓝**：分段采样固有的边界不连续问题，代码层面无法完全消除。可通过后处理亮度校正吸收。
2. **context_window 融合权重**：当前使用 pyramid/relative 权重，窗口融合区可能有亮度/纹理不连续。可通过调整 overlap 或 fuse_method 改善。
3. **双重切片仍有一处冗余**：`ref_mask_28ch` 在 `get_resized_cond` 中透传后在模型 patch 中切片，但 `get_resized_cond` 中仍有一次未生效的透传路径。语义正确但留有历史代码尾巴。
4. **模型 patch 是全局 monky-patch**：会修改 `SCAILWanModel._forward` 类方法，影响所有实例。当前通过 `_SCAIL_PATCH_APPLIED` 防重复，但重启 ComfyUI 前 patch 不会撤销。

---

## 十、调试方法总结

### 常用的日志埋点

在本次调试中使用的日志前缀：

| 前缀 | 位置 | 用途 |
|------|------|------|
| `[SCAILContext]` | `wan_scail_context.py` | 正常 INFO 日志 |
| `[SCAIL_DEBUG]` | `wan_scail_context.py`（模型 patch） | ref_mask_latents、driving_mask_28ch 切片情况 |
| `[WanSCAIL_COND]` | `wan_scail_multi_ref.py` | conditioning 源头 shape/数值统计 |
| `[WanSCAIL]` | `wan_scail_multi_ref.py` | noise_mask 信息 |

### 有效的调试策略

1. **从 conditioning 源头看 shape**：在 `get_resized_cond` 入口打印 x_in.shape、window.index_list 长度、各字段原始 shape
2. **在模型入口验证**：在 `_forward`/`forward_orig` 入口打印实际传入的参数 shape，比对 conditioning 切片后的结果和模型实际收到的是否一致
3. **对比有无上下文窗口**：关闭上下文窗口先跑通 baseline，再用上下文窗口对比差异
4. **关注段间边界**：问题最容易在第二段开头（保护帧交界处）和不同窗口的重叠区暴露

---

## 十一、文件状态（截至 2026-06-11）

| 文件 | 状态 | 备注 |
|------|------|------|
| `wan_scail_context.py` | 可用 | ~360行，含模型 patch + 调试日志 |
| `wan_scail_multi_ref.py` | 可用 | ~430行，含调试日志，干净硬切版本 |
| `Custom_context.py` | 未修改 | 复用已有代码 |
| `__init__.py` | 已修改 | 增加 SC 导入 |

---

## 十二、使用说明

### 工作流连接

```
WanSCAILToVideoMultiRef → WanSCAILContextWindows → KSampler
```

### 推荐参数

- `context_length`: SCAIL 的 length 参数值（如 81）
- `context_overlap`: 30 像素帧（太大增加计算量，太小可能看到接缝）
- `context_schedule`: `static_standard`（稳定可预测）
- `fuse_method`: `pyramid`（推荐）或 `relative`（平滑但慢）
- `prefix_latent_num`: **不要设置**（SCAIL 不需要）
- `freenoise`: 可选，改善窗口间随机性

### 与 `CustomWanContextWindowsManualNode` 的对比

| | 通用版（CustomWanContextWindows） | SCAIL 专用版（WanSCAILContextWindows） |
|---|---|---|
| conditioning 切片 | 通用 dim=2 | 智能识别 SCAIL 字段不同维度 |
| ref_mask_28ch | 会按 dim=2 错误切片 | 透传后在模型 patch 中正确切片 |
| driving_mask_28ch | 无特殊处理 | 从 kwargs pop 后切片 |
| prefix | 支持 | 不支持（不需要） |
| 模型 patch | 无 | 有（必须） |