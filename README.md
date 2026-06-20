# ComfyUI Custom Node Kit

一套为 ComfyUI 设计的自定义节点工具集，涵盖视频生成、姿态处理、图像交互操作等常见工作流需求。

当前版本：**1.5.6** — 新增 AutoColorDriftCorrection V3.1 色彩漂移校正节点、新增 CLIP Vision Multi-Ref Switch 多参考图 CLIP 特征合并节点、新增 WanSparseAttention 稀疏注意力节点。

## 目录

- [安装](#安装)
- [节点列表](#节点列表)
  - [WanLoop 视频生成](#wanloop-视频生成)
  - [SCAIL/SCAIL-2 多参考图生成](#scailscail-2-多参考图生成)
  - [SDPose 姿态系统](#sdpose-姿态系统)
  - [视频工具](#视频工具)
  - [交互式工具](#交互式工具)
  - [通用工具](#通用工具)
  - [图像批次工具](#图像批次工具)
  - [上下文工具](#上下文工具)
- [复杂节点详解](#复杂节点详解)
  - [WanAnimateToVideoCustom](#wananimatetovideocustom)
  - [Custom Context Windows (Manual)](#custom-context-windows-manual)
  - [Reference Image Selector](#reference-image-selector)
  - [SDPose 空帧修复](#sdpose-空帧修复)
  - [SDPoseDrawKeypointsV2 脚部绘制模式](#sdposedrawkeypointsv2-脚部绘制模式)
  - [Folder Image Loader](#folder-image-loader)
  - [Image Batch Concat](#image-batch-concat)
  - [Image Batch Resize](#image-batch-resize)
  - [WanAnimate Uni3C 运镜控制](#wananimate-uni3c-运镜控制)
  - [Interactive Batch Crop (交互式批量裁剪)](#interactive-batch-crop-交互式批量裁剪)
  - [MaskCompositeRef (遮罩批次合成 Ref)](#maskcompositeref-遮罩批次合成-ref)
- [SCAIL-2 节点详解](#scail-2-节点详解)
  - [Wan SCAIL To Video (Multi Ref)](#wan-scail-to-video-multi-ref)
  - [Create SCAIL-2 Colored Mask (Multi Ref)](#create-scail-2-colored-mask-multi-ref)
  - [Wan SCAIL-2 Context Windows](#wan-scail-2-context-windows)
- [依赖](#依赖)
- [工作流](#工作流)
  - [长视频姿态检测工作流](#长视频姿态检测工作流)
  - [WanAnimate多参考图生成长视频工作流](#wananimate多参考图生成长视频工作流)
  - [WanSCAIL2 多参考图生成长视频工作流](#wanscail2-多参考图生成长视频工作流)
- [使用示例](#使用示例)
- [许可证](#许可证)

---

## 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录下：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/user2318/ComfyUI-CustomNodeKit.git
```

2. 安装 Python 依赖：

```bash
cd ComfyUI-CustomNodeKit
pip install -r requirements.txt
```

3. 重启 ComfyUI。

> **注意**：请确保你的 ComfyUI 环境已安装 PyTorch 与相关核心依赖。

---

## 节点列表

### WanLoop 视频生成

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **WanAnimateToVideoCustom** | `WanLoop/整合节点` | WanAnimate 视频生成核心节点，支持 pose/face 控制、尾帧分段掩码、中性灰混合、上下文模式、prev_latent 潜变量接续等完整参数 |
| **单帧独立 VAE 编码** | `WanLoop/工具节点` | 将 N 张图片逐帧独立 VAE 编码后在时间维度拼接为 latent，为 ComfyUI-EverAnimate 项目提供锚点 latent |
| **WanUni3CLoader** | `WanVideo/Control` | 加载 Uni3C ControlNet 模型，支持 fp32/bf16/fp16 精度选择（可选） |
| **WanUni3CApply** | `WanVideo/Control` | 在 KSampler 中注入 Uni3C 运镜控制，支持 strength/start_percent/end_percent 等参数（可选） |

### SCAIL/SCAIL-2 多参考图生成

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Wan SCAIL To Video (Multi Ref)** | `model/conditioning/video_models` | SCAIL/SCAIL-2 多参考图视频生成 conditioning 节点。支持多张参考图自动编码、SCAIL-2 多身份 colored mask 注入、pose 视频引导、运动接续（prev_latent）等 |
| **Create SCAIL-2 Colored Mask (Multi Ref)** | `conditioning/video_models/scail` | 渲染 SAM3 追踪数据为 SCAIL-2 使用的 colored mask。支持参考图和驱动视频的共享调色板排序（left_to_right/area），确保多人物场景下同一身份颜色一致 |
| **Wan SCAIL Sparse Attention** | `conditioning/video_models/scail` | SCAIL 稀疏注意力节点，通过注意力掩码限制 pose/ref/main token 之间的交互，减少长视频生成中的退化现象。支持多种注意力掩码策略 |
| **CLIP Vision Multi-Ref Switch** | `conditioning/video_models/scail` | 多参考图 CLIP 特征合并节点。将批次中 N 张图的 CLIP Vision 特征拼接到 token 维度，使所有参考图都参与 conditioning |
| **Auto Color Drift Correction V3.1** | `CustomNodes/Video` | 自对齐色彩漂移校正节点 V3.1。三层校正：① 段内漂移检测（auto模式自动识别跳变，无漂移则旁路输出）；② 接缝对齐消除段间跳变；③ 隆起校正修正段内震荡 + 自适应EMA学习模板 |

### SDPose 姿态系统

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Draw SDPose Keypoints (V2)** | `SDPose` | 将姿态关键点数据渲染为可视化图像，支持全身骨骼、手部、面部、脚部的分层绘制，并根据偏航角自动调整骨骼粗细与遮挡顺序。新增 `foot_mode` 参数支持 `dots`（圆点）和 `line`（踝→大趾线条）两种脚部绘制模式 |
| **Save SDPose Keypoints as JSON** | `SDPose` | 保存姿态关键点数据为 JSON 文件，支持覆盖模式与自动递增编号 |
| **Load SDPose JSON** | `SDPose` | 加载 JSON 姿态文件，支持按目标帧率自动抽帧/补帧（线性插值或复制），支持空帧自动修复 |
| **Slice SDPose Keypoints** | `SDPose` | 对姿态序列进行时间切片（按起始帧与帧数截取） |
| **Concat SDPose Keypoints** | `SDPose` | 将两段姿态序列前后拼接 |
| **Estimate Yaw (Simple)** | `SDPose` | 简化版偏航角估计，从姿态关键点推算人物朝向角度（核心参数：置信度阈值、是否解缠、肩部权重） |
| **Estimate Yaw (Advanced)** | `SDPose` | 完整版偏航角估计，可调节所有底层参数（平滑窗口、EMA alpha、角度限制、侧向校准等），并输出详细调试表格 |
| **Resize SDPose Keypoints** | `SDPose` | 缩放姿态关键点坐标并更新画布尺寸，支持保持宽高比、智能裁剪（基于关键点包围盒） |
| **Resample SDPose Keypoints** | `SDPose` | 对姿态关键点序列进行帧率重采样（抽帧/补帧），独立于 JSON 载入逻辑，支持空帧自动修复 |
| **Load GroundingDINO Model** | `SDPose/GD` | 加载 GroundingDINO 模型（全局缓存，常驻 GPU），为官方 SDPose 提供 bbox 检测基础 |
| **GD BBox Detect** | `SDPose/GD` | GroundingDINO 一站式检测：文本检测 → 多模式筛选 → 输出 bbox 给官方 SDPoseOODProcessor |
| **Reference Image Selector** | `CustomNodes/SDPose` | 参考图选择器：根据偏航角范围自动筛选和排序参考图批次 |

### 视频工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **VideoFrameCounter** | `video` | 获取输入视频/图像序列的帧数统计信息与帧率 |
| **ImageSequenceToVideo** | `video` | 使用 ffmpeg 将图片序列合成为视频文件，支持 CRF、音频合成、帧偏移等参数 |

### 交互式工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Interactive Batch Crop** | `Interactive` | 交互式批量裁剪节点，配合前端 JS 实现图形化区域选择与批量处理 |

### 通用工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Path Collector** | `Utility` | 从指定目录收集文件路径 |
| **Index Selector** | `Utility` | 根据索引从列表中选取元素，支持 `last_folder` 输出 |
| **Path Validator** | `Utility` | 验证文件路径是否存在 |
| **Integer Setting (整数设置)** | `CustomNodes/Utils` | 将整数对齐到目标步长（start + step×n），支持负整数 |

### 图像批次工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Folder Image Loader** | `image` | 从文件夹按文件名升序载入图片批次，支持跳过、数量限制与尺寸同步 |
| **Image Batch Concat** | `image` | 将两个图像批次拼接，任意一路无输入时透传另一路 |
| **Image Batch Resize** | `image` | 缩放图像批次到指定宽高，可选按方向裁剪保持宽高比 |
| **MaskCompositeRef** | `image` | 遮罩批次合成节点 — 用于 WanSCAILToVideo 的 reference_image 预处理。支持替换模式和动作迁移模式，自动剔除全黑遮罩帧、背景合成 |

### 上下文工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Custom Context Windows (Manual)** | `context` | 通用上下文窗口调度节点。将长序列拆分为滑动窗口逐段处理，支持参考帧前缀、噪声混洗、多种调度策略（均匀/静态/循环/批处理）与融合模式（金字塔/线性叠加/相对加权）。新增 `causal_window_fix` 因果窗口修复功能，用于保留跨窗口的交互细节（如脚印） |
| **Wan SCAIL-2 Context Windows** | `context` | SCAIL-2 专用上下文窗口节点。自动处理 SCAIL-2 的特殊 conditioning 字段（ref_mask_28ch、driving_mask_28ch、pose_video_latent），无需前缀参考帧和 causal_window_fix。与 WanSCAILToVideoMultiRef 配合使用 |

---

## 复杂节点详解

### WanAnimateToVideoCustom

WanAnimate 视频生成的核心整合节点，将参考图、姿视频、面部视频、背景视频、角色遮罩等多路输入统一编码后拼接为合适的 conditioning 和 latent 格式，直接送入 KSampler 即可生成视频。

#### 参数说明

**必选参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `positive` | CONDITIONING | 正向提示词条件 |
| `negative` | CONDITIONING | 负向提示词条件 |
| `vae` | VAE | 用于编码图像为 latent 的 VAE 模型 |
| `width` / `height` | INT | 输出分辨率（默认 832×480，步长 16） |
| `length` | INT | 总输出帧数（像素帧，步长 4） |
| `batch_size` | INT | 批大小（通常为 1） |
| `continue_motion_max_frames` | INT | 运动接续的最大参考帧数（从上一段视频尾部取帧） |
| `video_frame_offset` | INT | 视频帧偏移量（多段拼接时递增） |
| `transition_width` | INT | fix 模式下黑帧过渡宽度（0-128，步长 4） |
| `mode` | 枚举 | `vanilla`=官方行为 / `legacy`=旧版尾帧分段掩码模式 / `fix`=硬替换中性灰+过渡扩散模式 |
| `tail_frame_count` | INT | legacy 模式下尾帧掩码覆盖的帧数 |
| `tail_start_strength` | FLOAT | legacy 模式下尾帧掩码起始强度 |
| `tail_end_strength` | FLOAT | legacy 模式下尾帧掩码结束强度 |
| `ref_mode` | 枚举 | `原模式`=内部1+4n排列后批量编码(接selected_images)；`兼容模式`=逐帧独立编码(接selected_images，兼容EverAnimate LoRA) |

**可选参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `clip_vision_output` | CLIP_VISION_OUTPUT | 用于注入 CLIP Vision 特征 |
| `reference_image` | IMAGE | 参考图像批次（首帧×1 + 其余各×4 拼接） |
| `face_video` | IMAGE | 面部视频输入（自动缩放到 512×512，归一化到 [-1, 1]） |
| `pose_video` | IMAGE | 姿态视频输入（编码为 pose_video_latent） |
| `continue_motion` | IMAGE | 上一段视频的尾部帧，用于运动接续 |
| `background_video` | IMAGE | 背景视频，覆盖参考图批次中运动接续帧之后的区域 |
| `character_mask` | MASK | 角色遮罩，在运动接续帧之后替换 concat_mask |
| `prev_latent` | LATENT | 前一段输出的完整 latent（concat_latent），接入后忽略 continue_motion，直接用前一段潜变量替换中性灰帧编码后的 latent |
| `face_strength` | FLOAT | 面部视频强度系数（0-1，默认 1.0） |
| `mid_frame` | INT | legacy 模式下中间帧锚点位置（-1=禁用） |
| `mid_strength` | FLOAT | legacy 模式下中间帧锚点强度 |
| `neutral_mix_min` / `neutral_mix_max` | FLOAT | legacy 模式下中性灰混合范围（掩码→中性灰的混合比例） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `positive` | CONDITIONING | 已注入 concat_latent_image、concat_mask 等条件的正向条件 |
| `negative` | CONDITIONING | 已注入对应条件的负向条件 |
| `latent` | LATENT | 空的 latent 占位（samples 形状已匹配总帧数） |
| `trim_latent` | INT | 需要跳过的首部 latent 帧数（用于后续 VAE Decode 时裁剪） |
| `trim_image` | INT | 需要跳过的首部像素帧数 |
| `video_frame_offset` | INT | 累积帧偏移（用于多段拼接时传递给下一段） |
| `concat_latent` | LATENT | 拼接后的完整 latent 图像（用于解码输出） |

#### 典型工作流

```
[上一段视频尾部帧] → continue_motion ─┐
[参考图批次]        → reference_image ─┤
[姿态视频]          → pose_video ──────┤
[面部视频]          → face_video ──────┤
[正向提示词]        → positive ────────┤
[负向提示词]        → negative ────────┤
                                        ↓
                               WanAnimateToVideoCustom
                                        ↓
                     positive / negative / latent → [KSampler]
                                                        ↓
                                                [VAE Decode]
                                                        ↓
                                                   输出视频
```

---

### Custom Context Windows (Manual)

用于 WAN 类视频生成模型的上下文窗口调度节点。当视频帧数超过模型单次处理上限时，将长序列拆分为多个滑动窗口逐段处理，再通过融合策略合并结果。此节点对模型进行封装，配合后续的 KSampler 使用即可。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | MODEL | 待封装的模型 |
| `context_length` | INT | 上下文窗口长度（像素帧数，自动转换为 latent 帧数，每 4 帧→1） |
| `context_overlap` | INT | 窗口重叠长度（像素帧数，自动转换） |
| `context_schedule` | 枚举 | 窗口调度策略（见下文） |
| `context_stride` | INT | 均匀调度策略中的步长指数上限（1-10） |
| `closed_loop` | BOOLEAN | 是否闭合窗口循环（仅循环调度有效） |
| `fuse_method` | 枚举 | 窗口融合方式（见下文） |
| `freenoise` | BOOLEAN | 是否启用 FreeNoise 噪声混洗（改善窗口间连续性） |
| `prefix_latent_num` | INT | 前缀参考帧的 latent 数量。接参考图选择器 raw_reference_images 的图片数量即可（每张图片编码为1个 latent）。这些帧会作为稳定参考拼接到每个窗口前 |
| `split_conds_to_windows` | BOOLEAN | 是否将多个 conditioning 按区域索引分配给各窗口 |
| `causal_window_fix` | BOOLEAN | 因果窗口修复（默认 True）。开启后在每个窗口前补上一窗口的 denoised 末帧，保留脚印等交互细节。会牺牲并行性（每次 denoise step 内窗口串行执行） |

#### 调度策略 (context_schedule)

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `looped_uniform` | 均匀循环调度：窗口在时间轴上循环分布，步长按指数递增 | 需要多次迭代覆盖的场景 |
| `standard_uniform` | 标准均匀调度：与 looped 类似但不循环，去除重复窗口 | 大多数通用场景的首选 |
| `standard_static` | 标准静态调度：固定步长的滑动窗口（步长 = context_length - context_overlap） | 需要简单分段处理的长视频 |
| `batched` | 批处理调度：按 context_length 直接切分，无重叠 | 不需要过渡融合的快速批处理 |

#### 融合方式 (fuse_method)

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| `pyramid` | 金字塔权重：窗口中心权重最大，向两侧递减 | 通用场景，过渡自然 |
| `flat` | 平坦权重：所有帧权重相同 | 简单平均融合 |
| `overlap-linear` | 线性重叠：重叠区域线性渐变（首帧 1→0，末帧 0→1） | 需要明确边界过渡的场景 |
| `relative` | 相对加权运行平均：基于窗口中心距离的动态加权，平滑分批采样累积 | 需要精细权重控制的高级场景 |

#### 使用要点

1. **帧数转换**：`context_length`、`context_overlap` 均以像素帧为单位输入，节点内部自动转换为 latent 帧（每 4 像素帧 → 1 latent 帧）。
2. **前缀参考帧**：设置 `prefix_latent_num > 0` 后，该数量的参考图 latent 会被追加到每个窗口开头作为稳定参考，适合需要全局上下文信息的生成任务。
3. **freenoise**：启用后会混洗噪声以改善窗口间的纹理连续性，建议在总帧数较长且重叠较小时开启。
4. **causal_window_fix**：启用后会在每个窗口前补入上一窗口 denoised 输出的末帧作为 anchor，有效保留脚印、地面交互等跨窗口连续性细节。注意这会强制窗口串行执行，降低并行度。

#### 典型工作流

```
[加载模型] → model ──┐
                      ↓
           Custom Context Windows (Manual)
                      ↓
               model (已封装) → [KSampler] → [VAE Decode] → 输出
```

---

### Reference Image Selector

参考图选择器，用于根据视频片段的偏航角范围自动筛选最合适的参考图批次。典型应用：WanAnimate 视频生成中根据人物朝向（yaw）从多角度参考图中选取最佳匹配。也可用于 SCAIL-2 工作流（通过 `raw_reference_images` 配合 Custom Context Windows 动态前缀）。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_images` | IMAGE | ✅ | 参考图批次（N 张不同视角的参考图） |
| `angle_map` | STRING | ✅ | JSON 格式的角度映射，如 `[-90, -45, 0, 45, 90]`，数组长度必须与参考图数量一致 |
| `yaw_angles` | FLOAT | ❌ | 目标视频片段的偏航角序列（每帧一个值），不接入时输出全部参考图 |
| `background_images` | IMAGE | ❌ | 背景图批次（原始图片直接插入到最后一张辅助参考图之前）。仅改变输出顺序，不做遮罩合成 |
| `select_references` | BOOLEAN | ❌ | `True`（默认）=按角度筛选+排序，只保留匹配的参考图；`False`=仅排序不筛选（全部参考图都使用） |
| `allow_switch_main` | BOOLEAN | ❌ | `True`（默认）=允许更换第一张主参考图；`False`=固定第一张不变（当第一张不在候选集时会强制加入） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `selected_images` | IMAGE | 筛选排序后的参考图批次（原始图片直接输出，不做 1+4n 拼接）。主参考图排第 0 位 + 辅助参考图 + 背景图块 + 最后一张辅助参考图 |
| `raw_reference_images` | IMAGE | 未排序的原始参考图（用于接入 Custom Context Windows 的 `prefix_latent_num`，因为上下文窗口需要重新筛选） |
| `info` | STRING | 调试信息（角度范围、候选索引、排序结果等） |
| `reference_angle_map` | STRING | 验证后的角度映射 JSON 字符串（用于 Custom Context Windows 动态前缀） |

#### 工作原理

1. **yaw_angles 未接入 或 angle_map 无效/数量不匹配**：直接输出全部参考图。
2. **有 yaw_angles 输入（筛选+排序模式）**：
   - 计算偏航角范围 [min, max]（处理 -180°/180° 环绕）
   - 筛选所有在此范围内的参考图 + 左右各最邻近一张
   - 从候选中选出覆盖帧数最多的参考图作为**主参考图**（排在第 0 位）
   - 辅助参考图按与首帧偏航角的偏差从大到小排列
   - 若主参考图恰好最贴合首帧偏航角，末尾追加主参考图副本
   - 背景图插入到辅助参考图的最后一张之前
3. **仅排序模式（select_references=False）**：所有参考图都参与，不筛除，仅执行排序和输出构建。
4. **输出格式**：原始图片直出，1+4n 拼接由下游节点（WanAnimateToVideoCustom 或 Wan SCAIL To Video）处理。

#### 与 Custom Context Windows 的配合

WanAnimate 使用 `Custom Context Windows (Manual)` 时，需要将 `raw_reference_images`（未排序的原始参考图）接入其 `prefix_latent_num`（传入图片数量）。上下文窗口内部会根据每段视频的实际偏航角重新调用选择逻辑，实现动态参考图分配。同时将 `reference_angle_map` 传入上下文窗口节点即可。

#### 典型工作流

```
[多角度参考图] → reference_images ─┐
[角度映射 JSON] → angle_map ───────┤
[偏航角序列]    → yaw_angles ─────┤
                                    ↓
                         Reference Image Selector
                                    ↓
      selected_images → [WanAnimateToVideoCustom] reference_image
      raw_reference_images + reference_angle_map → [Custom Context Windows] prefix
```

---

### SDPose 空帧修复

在 SDPose 姿态处理流水线中，某些帧可能因检测失败导致 `people` 列表为空（产生黑帧）。新增的 `fix_empty_frames` 功能可自动检测并修复这些空帧。

#### 适用节点

- **Load SDPose JSON**：加载 JSON 时修复（参数 `fix_empty_frames`）
- **Resample SDPose Keypoints**：重采样后修复（参数 `fix_empty_frames`）

#### 修复逻辑

1. 扫描所有帧，标记 `people` 为空或所有关键点坐标为零的帧为"无效帧"
2. 对每个无效帧：
   - **双侧有有效帧**：线性插值填充（若左右帧有突变则直接复制较近侧）
   - **仅一侧有有效帧**：复制该侧数据
3. 修复过程不修改原帧列表，返回新列表

#### 使用建议

- 在姿态数据来源不够稳定的情况下建议开启
- 配合 `Estimate Yaw` 节点使用时可有效避免因空帧导致的角度跳变

---

### SDPoseDrawKeypointsV2 脚部绘制模式

从 v1.4.0 开始，SDPoseDrawKeypointsV2 新增 `foot_mode` 参数，提供两种脚部绘制方式：

| 模式 | 说明 |
|------|------|
| `dots` | 默认模式。在脚部关键点位置绘制彩色圆点（与之前版本行为一致） |
| `line` | 线条模式。绘制踝关节→大脚趾的连线，更清晰地表现脚部朝向和姿态，不绘制圆点 |

两种模式下都会根据 `bottom_side` / `top_side` 参数进行前后侧分层绘制，确保遮挡关系正确。

---

### Folder Image Loader

从指定文件夹按文件名升序读取图片批次。遇到路径错误、文件夹为空等情况不会抛出异常，仅输出空结果并在控制台打印警告。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `folder_path` | STRING | ✅ | 图片文件夹路径，支持绝对路径或相对于 ComfyUI 根目录的相对路径 |
| `size_mode` | 枚举 | ✅ | `resize_to_first`：所有图片缩放至第一张的尺寸；`filter_same_size`：仅加载与第一张尺寸相同的图片 |
| `skip_first_n` | INT | ✅ | 跳过前 N 张图片（默认 0） |
| `load_count` | INT | ✅ | 最多加载 N 张图片（0=不限制，加载全部） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 载入的图片批次。路径无效或无图片时输出 None |
| `count` | INT | 实际载入的图片数量。异常情况输出 0 |

#### 容错行为

- 文件夹路径为空 → 输出 `(None, 0)`，控制台打印警告
- 文件夹不存在 → 输出 `(None, 0)`，控制台打印警告
- 文件夹内无图片 → 输出 `(None, 0)`，控制台打印警告
- 跳过数超过图片总数 → 输出 `(None, 0)`，控制台打印警告
- 单张图片读取失败 → 跳过该图，继续加载其余图片

---

### Image Batch Concat

将两个图像批次沿 batch 维度拼接（images_b 追加到 images_a 后方）。一路或两路无输入时，透传有输入的那一路；两路都无输入时输出 None。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `images_a` | IMAGE | ❌ | 第一个图像批次 |
| `images_b` | IMAGE | ❌ | 第二个图像批次，拼接到 images_a 后方 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 拼接后的图像批次。仅一路有输入时透传该路；两路均无输入时输出 None |

#### 典型工作流

```
[FolderImageLoader A] → images_a ─┐
[FolderImageLoader B] → images_b ─┤
                                    ↓
                           Image Batch Concat
                                    ↓
                               images (合并批次)
```

---

### Image Batch Resize

将图像批次缩放到指定宽高。可选择按方向裁剪以保持宽高比。输入无效时输出 None。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `width` | INT | ✅ | 目标宽度（默认 512） |
| `height` | INT | ✅ | 目标高度（默认 512） |
| `crop_mode` | 枚举 | ✅ | 裁剪模式。`disabled`=直接拉伸；`center`/`top`/`bottom`/`left`/`right`=先等比缩放至覆盖目标再按方向裁剪 |
| `images` | IMAGE | ❌ | 输入图像批次。无输入时输出 None |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 缩放后的图像批次。输入无效时输出 None |

#### 缩放算法

统一使用 Pillow 的 LANCZOS 算法，兼顾缩小和放大的高质量效果。

#### 裁剪模式图解

```
原图 1920×1080 → 目标 512×512
disabled：直接拉伸，画面变形
center：  等比缩放到 911×512，左右各裁 199px
top：     等比缩放到 911×512，裁底部 399px
bottom：  等比缩放到 911×512，裁顶部 399px
left：    等比缩放到 512×288，裁右侧 176px（需再放大到 512×512）
right：   等比缩放到 512×288，裁左侧 176px（需再放大到 512×512）
```

> 裁剪模式始终先缩放至覆盖目标尺寸（取 max 缩放比），再按方向裁剪多余内容，保证最终输出尺寸精确等于目标宽高。

---

### WanAnimate Uni3C 运镜控制

Uni3C 运镜控制为 WanAnimate 视频生成提供可选的相机运动控制能力。它通过在 KSampler 的采样过程中注入 ControlNet 控制信号，引导生成结果按照指定的相机轨迹运动。**这是一个可选功能**，不使用时不影响正常生成流程。

#### 使用前提

1. 需要下载 Uni3C ControlNet 模型文件（`.safetensors`）放入 `ComfyUI/models/controlnet` 目录
2. 需要安装 `diffusers` 和 `accelerate` Python 包

#### WanUni3CLoader — 模型加载

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | COMBO | ✅ | Uni3C ControlNet 模型文件，自动扫描 controlnet 目录 |
| `base_precision` | 枚举 | ❌ | 基础精度，可选 `fp32` / `bf16` / `fp16`（默认 `fp16`） |

**输出**：`uni3c_controlnet`（UNI3C_CONTROLNET）— 加载后的 ControlNet 模型，供 WanUni3CApply 使用。

#### WanUni3CApply — 控制注入

在 KSampler 中注入 Uni3C 运镜控制信号。节点通过 monkey-patch 模型内部的 `forward_orig` 方法，在每个 denoising step 中计算并注入控制信号。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | MODEL | ✅ | 待注入控制的模型 |
| `uni3c_controlnet` | UNI3C_CONTROLNET | ✅ | 由 WanUni3CLoader 加载的 ControlNet 模型 |
| `render_latent` | LATENT | ✅ | 预渲染的参考视频潜空间张量（B, C, T, H, W），作为运镜控制的视觉参考 |
| `strength` | FLOAT | ✅ | 控制强度（0.0-10.0，默认 1.0） |
| `start_percent` | FLOAT | ✅ | 控制生效起始百分比（0.0-1.0，默认 0.0） |
| `end_percent` | FLOAT | ✅ | 控制生效结束百分比（0.0-1.0，默认 1.0） |
| `render_mask` | MASK | ❌ | 可选的渲染遮罩（实验性功能） |
| `trim_latent` | INT | ❌ | 参考图占用的 latent 帧数，从 WanAnimateToVideoCustom 的 `trim_latent` 输出接入，用于时序对齐 |
| `positive` | CONDITIONING | ❌ | 可选：传入 conditioning 自动提取 `concat_latent_image`，替代 WanAnimateChannelPack 节点 |
| `negative` | CONDITIONING | ❌ | 同上，负条件 |

**输出**：`model`（MODEL）— 已注入控制的模型，可直接连接 KSampler。

#### 核心原理

1. **v4 架构**：采用 monkey-patch 方式修改模型 `forward_orig` 方法，在 block 循环中精确控制注入时机
2. **时序对齐**：通过 `trim_latent` 参数自动对齐 render_latent 与生成视频的时序偏移
3. **异步预取**：控制信号采用 CPU→GPU 异步预取机制，与 block 计算重叠，最小化性能开销
4. **step 控制**：通过 `start_percent` / `end_percent` 控制控制信号在 denoising 过程中的生效范围

#### 典型工作流

```
[WanUni3CLoader] → uni3c_controlnet ─┐
[渲染好的 latent] → render_latent ────┤
                                       ↓
[模型] → WanUni3CApply → model (已注入控制) → [KSampler] → [VAE Decode] → 输出
```

> **注意**：WanUni3CApply 节点内置了 WanAnimateChannelPack 逻辑，如果传入了 `positive`/`negative` conditioning，会自动从其中提取 `concat_latent_image`，无需额外连接 WanAnimateChannelPack 节点。

---

### Interactive Batch Crop (交互式批量裁剪)

通用交互式裁剪节点，提供图形化界面进行区域选择和批量裁剪。不绑定特定工作流，可在任何需要批量裁剪图片的场景中使用。典型应用场景如 360° 图像批次预处理模块中，从 360° 渲染图中裁剪出人物区域。

#### 输入方式

节点支持两种图片源：

| 方式 | 说明 |
|------|------|
| **路径模式** | 在 `path` 参数中输入文件夹路径或用 `|` 分隔的多个文件路径。点击节点上的"选择目录"或"多选文件"按钮可调出系统文件选择器 |
| **张量模式** | 将上游节点的图片张量连接到 `images` 可选输入。优先级高于路径模式（连接后 path 参数被忽略） |

#### 操作流程

1. **设置图片源**：填入文件夹路径 / 多文件路径，或连接 `images` 输入
2. **运行一次**：首次执行工作流以建立图片缓存（节点需要知道图片数量用于预览索引）
3. **打开裁剪器**：点击节点上的 **✂ 打开裁剪器** 按钮，弹出图形化裁剪界面
4. **调整裁剪框**：在画布上拖动/缩放裁剪框，设置合适的裁剪区域和填充色
5. **确认**：点击 ✅ 确认，裁剪参数自动保存到节点的 `crop_params`（隐藏）
6. **再次运行**：再次执行工作流，节点根据保存的裁剪参数批量处理所有图片

> **提示**：路径模式下，如果文件夹中有多种分辨率的图片，裁剪器会先弹出尺寸选择对话框，选择统一处理的尺寸。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | STRING | ❌ | 图片文件夹路径，或用 `|` 分隔的多个文件路径。与 `images` 二选一 |
| `max_preview` | INT | ❌ | 前端预览中显示的最大图片数量（默认 10），图片过多时会自动抽样 |
| `images` | IMAGE | ❌ | 直接传入图片张量作为裁剪源（与 path 二选一） |

> `crop_params`、`node_id`、`target_size` 为隐藏参数，由前端 JS 自动管理，用户无需手动填写。

#### 裁剪器界面功能

| 功能 | 操作方式 |
|------|----------|
| **拖动裁剪框** | 鼠标在裁剪框内拖拽移动位置 |
| **缩放裁剪框** | 鼠标拖拽裁剪框的 8 个边缘/角点 |
| **移动图片** | `Shift` + 拖拽，调整图片在画布中的位置 |
| **平移裁剪框+图片** | `Ctrl` + 拖拽裁剪框 |
| **滚轮缩放** | 鼠标滚轮放大/缩小画布视图 |
| **翻页** | ◀ ▶ 按钮或键盘 ← → 方向键切换预览图片 |
| **缩略图导航** | 点击底部缩略图跳转到对应图片 |
| **吸附** | 裁剪框边缘靠近图片边缘时自动吸附对齐（可关闭） |
| **保持比例** | 勾选后锁定宽高比，拖动时自动维持比例 |
| **预设比例** | 快速选择 1:1、4:3、16:9 等预设比例，或选择"自由"模式 |
| **↕ 交换** | 交换裁剪框的宽和高 |
| **填充色** | 裁剪超出图片区域的填充颜色（黑/白/灰/自定义） |
| **重置** | 将裁剪框重置为覆盖整张图片 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 裁剪后的图片批次。裁剪区域超出原图部分用填充色填充 |

#### 典型工作流

```
[图片源（路径/张量）] → Interactive Batch Crop → 裁剪后的图片批次
```

---

### MaskCompositeRef (遮罩批次合成 Ref)

专门用于 **WanSCAILToVideo MultiRef** 节点的 reference_image 预处理节点。根据 `replacement_mode` 分为两种工作模式，对应 SCAIL-2 的两种生成策略。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `images` | IMAGE | ✅ | 原图批次 [B, H, W, C]。输入参考图序列，顺序必须与 masks 一一对应 |
| `replacement_mode` | BOOLEAN | ✅ | `False`=动作迁移模式（白底彩色遮罩，与背景图合成）；`True`=替换模式（黑底彩色遮罩，背景替换为纯黑，自动剔除全黑帧） |
| `process_first_frame` | BOOLEAN | ✅ | 仅动作迁移模式有效。`False` 时第 1 张原图保留不动不做合成 |
| `process_last_frame` | BOOLEAN | ✅ | 仅动作迁移模式有效。`True` 时最后 1 张原图参与合成 |
| `remove_background` | BOOLEAN | ✅ | 替换模式：关闭时仅做全黑帧剔除+透传，不做背景替换。动作迁移模式：关闭时仅做尺寸对齐和背景插入，不做遮罩合成 |
| `masks` | IMAGE | ❌ | 遮罩批次 [B, H, W, C]，顺序与 images 一一对应。未连接时透传原图 |
| `backgrounds` | IMAGE | ❌ | 背景图批次。仅动作迁移模式有效。第 1 张用于遮罩合成填充背景，其余插入到合成序列中 |

#### 两种模式详解

##### 替换模式 (replacement_mode=True)

用于替换背景的场景（如将人物从旧背景迁移到新环境中）。

- **遮罩要求**：黑底彩色（人物区域为彩色，纯黑 RGB<10/255 区域为背景）
- **处理逻辑**：
  1. 自动检测并剔除遮罩全黑的帧（原图和遮罩同时剔除）
  2. 将剩余原图的背景（黑色区域）替换为纯黑
  3. 保留原始彩色遮罩输出
- **backgrounds 输入**：被忽略
- **典型用途**：SCAIL-2 替换模式中，将参考图中的人物从原背景中分离出来

##### 动作迁移模式 (replacement_mode=False)

用于动作迁移的场景（将人物的动作迁移到新的背景中）。

- **遮罩要求**：白底彩色（人物区域为彩色，纯白 RGB>245/255 区域为背景）
- **处理逻辑**：
  1. 将原图的人物区域提取出来，与背景图合成到新的背景上
  2. 支持首尾帧跳过控制（`process_first_frame` / `process_last_frame`）
  3. 背景图批次：第 1 张用作遮罩合成的填充背景，多余的背景图会插入到倒数第 1 和第 2 帧之间
- **backgrounds 输入**：有效，第 1 张用于合成，其余插入序列
- **典型用途**：SCAIL-2 动画模式中，将人物动作迁移到新背景上

#### 自动容错

| 情况 | 行为 |
|------|------|
| masks 未连接 | 透传原图。替换模式输出全黑遮罩，动作迁移模式输出全白遮罩 |
| 遮罩尺寸不一致 | 自动缩放到原图尺寸（LANCZOS 算法），控制台打印警告 |
| 替换模式所有帧全黑 | 返回空批次（可用于条件判断跳过当前段） |
| 批次数量不一致 | 自动截断为较短的数量并打印警告 |
| 无背景图 | 使用纯黑作为背景（仅做剔除效果，人物区域无变化） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 合成后的图像批次 |
| `masks` | IMAGE | 处理后的遮罩批次 |

#### 典型工作流

```
[参考图](4 张) → images ───┐
[参考图遮罩]    → masks ───┤
[背景图]       → backgrounds ─┤
                             ↓
                    MaskCompositeRef
                             ↓
              images + masks → [Wan SCAIL To Video] reference_image + reference_image_mask
```

---

## SCAIL-2 节点详解

以下节点为本工具集对 **Wan SCAIL-2** 模型的支持部分。SCAIL-2 与 WanAnimate 共用 `Custom Context Windows (Manual)`？**不**，SCAIL-2 有专用的 `Wan SCAIL-2 Context Windows` 节点，因为其 conditioning 中包含 SCAIL-2 特有的多通道遮罩字段。

### Wan SCAIL To Video (Multi Ref)

SCAIL-2 视频生成的核心 conditioning 节点。将多张参考图、姿态视频、colored mask 等输入统一编码为模型所需的 conditioning 格式。

#### 参数说明

**必选参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `positive` / `negative` | CONDITIONING | 正/负向提示词条件 |
| `vae` | VAE | VAE 模型 |
| `width` / `height` | INT | 输出分辨率（步长 32） |
| `length` | INT | 输出帧数（像素帧，步长 4，推荐 81） |
| `batch_size` | INT | 批大小（通常为 1） |

**参考图参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_image` | IMAGE | ❌ | 多张参考图以批次形式传入 (N, H, W, 3)。每张图独立编码为单独的 reference_latent |
| `reference_image_mask` | IMAGE | ❌ | 仅 SCAIL-2。与 reference_image 同分辨率的彩色参考遮罩，数量需匹配 |
| `ref_encoding_mode` | 枚举 | ❌ | 参考图编码模式，见下文详解 |

**姿态参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pose_video` | IMAGE | ❌ | 姿态视频，将降低分辨率至主视频的一半后编码为 pose_video_latent |
| `pose_video_mask` | IMAGE | ❌ | 仅 SCAIL-2。与 pose_video 同分辨率的 SAM3 逐人彩色遮罩视频 |
| `replacement_mode` | BOOLEAN | ❌ | `False`=动画模式（默认）；`True`=替换模式 |
| `pose_strength` | FLOAT | ❌ | 姿态 latent 强度（默认 1.0） |
| `pose_start` / `pose_end` | FLOAT | ❌ | 姿态 conditioning 在 denoising 步中的生效范围（0.0~1.0） |

**接续参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `video_frame_offset` | INT | ❌ | 当前块开始的累计输出帧偏移，从上一块的输出接入（默认 0） |
| `previous_frame_count` | INT | ❌ | 用于锚定的上一块尾帧数（SCAIL-2 训练使用 5，即 81 帧块+76 帧步长） |
| `previous_frames` | IMAGE | ❌ | 仅 SCAIL-2。上一块的完整解码输出，仅最后 previous_frame_count 帧用作锚定 |

**可选输入：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `clip_vision_output` | CLIP_VISION_OUTPUT | CLIP 视觉特征注入 |

#### ref_encoding_mode 三种模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `1+4n批量` | 第 1 张×1 + 其余每张×4 → 批处理 VAE 编码（默认）。更接近模型训练时的分布 | 通用场景，推荐用于大多数情况 |
| `逐帧编码` | 每张参考图独立升采样+VAE 编码 → 在时间维拼接。每帧独立保真度更高 | 需要高保真参考图细节的场景 |
| `混合编码` | 前 N-1 张用 1+4n 批量编码 + 最后 1 张逐帧独立编码。兼顾主体清晰度与减轻段间跳变 | 需要平衡参考图清晰度和视觉连续性的场景 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `positive` | CONDITIONING | 已注入 reference_latents、pose_video_latent、ref_mask_28ch、driving_mask_28ch 等条件的正向输出 |
| `negative` | CONDITIONING | 同上，负向输出 |
| `latent` | LATENT | 生成尺寸的空 latent（samples 形状已匹配）。接入 previous_frames 时内置噪声遮罩实现硬切锚定 |
| `video_frame_offset` | INT | 调整后的偏移量 + 长度，接入下一块 |

#### 典型工作流

```
[多张参考图] → reference_image ────┤
[参考图遮罩]  → reference_image_mask ─┤
[驱动视频]    → pose_video ──────────┤
[驱动遮罩]    → pose_video_mask ─────┤
[条件]        → positive/negative ───┤
[接续帧]      → previous_frames ─────┤
                                     ↓
                      Wan SCAIL To Video (Multi Ref)
                                     ↓
                     positive / negative / latent
                                     ↓
               [Wan SCAIL-2 Context Windows] → [KSampler] → [VAE Decode] → 输出视频
```

---

### Create SCAIL-2 Colored Mask (Multi Ref)

渲染 SAM3 追踪数据为 SCAIL-2 使用的彩色遮罩。与 `WanSCAILToVideoMultiRef` 配合使用。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `driving_track_data` | SAM3_TRACK_DATA | ✅ | 驱动姿态视频的 SAM3 追踪数据，渲染为 `pose_video_mask` |
| `ref_track_data` | SAM3_TRACK_DATA | ❌ | 参考图的 SAM3 追踪数据，渲染为 `reference_image_mask`。未连接时自动生成纯色背景遮罩 |
| `object_indices` | STRING | ❌ | 逗号分隔的人物索引列表（如 `0,2,3`），只保留指定人物。空=保留全部 |
| `sort_by` | 枚举 | ❌ | 调色板分配到追踪对象的顺序（同时应用于参考图和驱动视频，确保同一个人物颜色一致） |
| `replacement_mode` | BOOLEAN | ❌ | 影响输出遮罩的背景色 |

#### sort_by 排序方式

| 方式 | 说明 |
|------|------|
| `left_to_right` | 按首帧质心的 x 坐标从左到右分配调色板索引，适用于多人物并排场景 |
| `area` | 按首帧人物面积从大到小分配调色板索引，适用于人物大小差异大的场景 |
| `none` | 保持 SAM3 原始顺序 |

#### 遮罩背景色规则

| 模式 | pose_video_mask 背景色 | reference_image_mask 背景色 |
|------|----------------------|--------------------------|
| 动画模式 (replacement_mode=False) | 黑色 | 白色 |
| 替换模式 (replacement_mode=True) | 白色 | 黑色 |

> 此背景色规则与 `MaskCompositeRef` 节点的遮罩检测逻辑完全对应。

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `pose_video_mask` | IMAGE | 渲染后的驱动视频彩色遮罩，接入 WanSCAILToVideoMultiRef 的 `pose_video_mask` |
| `reference_image_mask` | IMAGE | 渲染后的参考图彩色遮罩，接入 WanSCAILToVideoMultiRef 的 `reference_image_mask` |

#### 典型工作流

```
[SAM3 参考图追踪] → ref_track_data ─┐
[SAM3 驱动追踪]    → driving_data ───┤
[人物索引]        → object_indices ──┤
[排序方式]        → sort_by ─────────┤
                                     ↓
                     Create SCAIL-2 Colored Mask (Multi Ref)
                                     ↓
          pose_video_mask → [WanSCAILToVideoMultiRef] pose_video_mask
          reference_image_mask → [WanSCAILToVideoMultiRef] reference_image_mask
```

---

### Wan SCAIL-2 Context Windows

SCAIL-2 专用的上下文窗口节点，与通用的 `Custom Context Windows (Manual)` 有重要区别。

#### 与 Custom Context Windows (Manual) 的区别

| 特性 | Custom Context Windows (Manual) | Wan SCAIL-2 Context Windows |
|------|-------------------------------|----------------------------|
| conditioning 处理 | 通用处理，需要手动配置 prefix_latent_num | 自动处理 SCAIL-2 特殊字段（ref_mask_28ch、driving_mask_28ch、pose_video_latent） |
| 前缀参考帧 | 通过 `prefix_latent_num` 配置 | **不需要**，SCAIL-2 的 reference_latents / ref_mask_28ch 是全局保留的 |
| causal_window_fix | 支持 | **不支持**（SCAIL-2 通过 previous_frames 机制实现段间锚定） |
| 适用模型 | 通用 WAN 模型 | SCAIL-2 模型专用 |

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | MODEL | 待封装的 SCAIL-2 模型 |
| `context_length` | INT | 上下文窗口长度（像素帧）。建议设为 SCAIL-2 的 length 参数值（如 81） |
| `context_overlap` | INT | 窗口间重叠帧数（像素帧） |
| `context_schedule` | 枚举 | 窗口调度策略。`static_standard`（固定窗口）为推荐选项 |
| `context_stride` | INT | 窗口步进（仅对 uniform 策略有效） |
| `closed_loop` | BOOLEAN | 是否闭环窗口（仅 looped 策略有效） |
| `fuse_method` | 枚举 | 窗口融合方法（金字塔权重推荐） |
| `freenoise` | BOOLEAN | FreeNoise 噪声混洗，改善窗口间融合 |

#### 工作原理

1. **conditioning 自动分片**：节点内部自动识别 conditioning 中各字段的时间维布局，对 `ref_mask_28ch`（保留前 N 帧参考 + 按窗口切片视频部分）、`driving_mask_28ch`（按窗口切片）、`pose_video_latent`（按窗口切片）分别处理
2. **模型 patch**：在模型 forward 入口自动应用 patch，在每次前向时根据当前窗口索引切片 ref_mask_28ch 和 driving_mask_28ch
3. **无需 prefix**：SCAIL-2 的 reference_latents 和 ref_mask_28ch 的前 N 帧作为全局条件自动保留在所有窗口中

#### 典型工作流

```
[SCAIL-2 模型] → model ──┐
                           ↓
              Wan SCAIL-2 Context Windows
                           ↓
           model (已封装) → [KSampler] → [VAE Decode] → 输出视频
```

---

## 工作流

`workflow/` 目录下提供了多个完整的 ComfyUI 工作流 JSON 文件，可直接拖入 ComfyUI 界面使用。

### 长视频姿态检测工作流

**文件**：`Long_Pose_detection_SD_offical.json` / `Long_Pose_detection_SDpose.json`

**用途**：对长视频进行逐帧骨骼（body）和面部关键点检测，支持将检测骨骼与参考图人物的骨骼进行比例对齐（BodyRatioMapper），最终生成骨骼可视化视频、面部特征 JSON 和姿态 JSON 文件。适用于制作舞蹈骨骼动画、动作捕捉数据提取等场景。

- `Long_Pose_detection_SD_offical.json`：使用官方 SDPoseOODProcessor 进行姿态识别
- `Long_Pose_detection_SDpose.json`：使用 SDPose 流程进行姿态识别（配合自定义节点）

**简要使用说明**：

1. 在「参数设置」区域内设置分辨率（宽度/高度）、帧率（fps）、每片段帧数等；
2. 在「视频分段循环加载」区域内配置输入视频路径、视频分段参数；
3. 在「识别并保存面部图、面部特征点json、姿态json」区域内设置参考骨骼 JSON 路径和参考图，用于骨骼比例对齐；
4. 在「骨骼对齐、绘制，合成骨骼视频」区域调整骨骼绘制参数（线条宽度、面部/手部关键点大小等）；
5. **运行前请先清除工作目录中的过往文件，尤其是脸图（face）文件夹**，避免旧数据残留干扰；
6. 运行后自动分段循环处理长视频，逐段进行 SDPose 姿态识别、骨骼对齐、可视化绘制，最终合成骨骼视频（保留原视频音频）。

**视频合成方式**：将绘制好的骨骼帧通过 `ImageSequenceToVideo` 合成为视频文件，可同步附带原视频音频（通过 `VHS_LoadAudio` 加载）。

**本工作流使用到的第三方节点包**（不含本工作区 `ComfyUI-CustomNodeKit` 自身节点）：

| 节点包 | 主要用途 |
|--------|----------|
| `sdpose-ood` | SDPose 姿态识别模型加载（`SDPoseOODLoader`）与处理（`SDPoseOODProcessor`） |
| `ComfyUI-WanAnimatePreprocess` | ONNX 面部检测模型加载（`OnnxDetectionModelLoader`）与面部图像截取（`PoseAndFaceDetection`） |
| `comfyui-videohelpersuite` | 视频文件加载（`VHS_LoadVideoFFmpegPath`）与音频加载（`VHS_LoadAudio`） |
| `comfyui-kjnodes` | 常量节点（`INTConstant`、`FloatConstant`、`StringConstant`） |
| `comfyui-easy-use` | While 循环控制（`easy whileLoopStart`/`easy whileLoopEnd`）与缓存清理（`easy clearCacheAll`） |
| `comfyui-impact-pack` | 循环条件比较（`ImpactCompare`） |
| `comfyui-custom-scripts` | 数学表达式计算（`MathExpression\|pysssss`） |
| `ComfyUI-BodyRatioMapper` | 骨骼比例对齐（`BodyRatioMapperProportionTransfer`） |
| `ComfyUI-Addoor` | 临时文件清理（`AD_DeleteLocalAny`） |

### WanAnimate多参考图生成长视频工作流

**文件**：`WanAnimate_ref_pics_loop+context.json`

**用途**：基于 WanAnimate 模型，使用多张参考图生成带有上下文窗口的长视频。该工作流特别包含一个**360° 图像批次预处理模块**，可将单张参考图处理为 360° 环绕视角的图片批次（配套文件存放于输出目录中），用户可以从生成结果中挑选合适角度的图片作为多参考图输入，从而更好地控制长视频中人物的外观一致性。

**简要使用说明**：

1. 在「模型、LoRA及VAE加载」区域加载 WanAnimate UNet 模型、CLIP 模型、VAE 及所需 LoRA；
2. 在「设置主参考图」区域加载主参考图并缩放至 512×512；
3. 在「设置副参考图」区域通过 `PathCollectorNode` 指定副参考图目录（可放入 360° 预处理输出的多角度参考图）；
4. 在「提示词填写」区域输入正/负向提示词；
5. 在「视频生成参数区」设置每段生成帧数、片段间重叠帧数等参数；
6. 选择动作和面部数据加载方式：
   - **常规加载**：直接载入原视频，实时识别面部和姿态（不支持骨骼对齐）；
   - **快速加载**：直接载入预先导出的姿态 JSON 和面部 JSON，支持骨骼对齐；
7. 运行后通过循环控制区逐段生成视频帧，利用 `Custom Context Windows` 实现跨段上下文窗口调度；
8. **生成完成后**，手动启用「保存视频」区域，运行 `ImageSequenceToVideo` 将帧序列合成为视频文件（可附带原视频音频）。

**视频合成方式**：先通过 KSampler 生成各段视频帧（`VAE Decode`），再进行跨段淡入淡出融合（`CrossFadeImages`），最后通过 `ImageSequenceToVideo` 将完整帧序列合成为视频。

**360° 预处理模块说明**：

- 该模块位于工作流下半部分（通常 mute 状态），包含 `InteractiveBatchCrop` 交互式裁剪、`SDPoseLoadJson` 加载 360° 关键点 JSON、`SDPoseResizeKeypoints` 缩放关键点、`SDPoseDrawKeypointsV2` 绘制姿态视频；
- 配套文件（360° 关键点 JSON 等）需放置在 `output/video/360°` 目录下；
- 用户可运行预处理模块生成 360° 环绕视角姿态视频批次，从中挑选合适角度的帧作为副参考图；
- 复杂预处理建议使用主工作流（`Long_Pose_detection_SD_offical.json` 或 `Long_Pose_detection_SDpose.json`）进行。

**本工作流使用到的第三方节点包**（不含本工作区 `ComfyUI-CustomNodeKit` 自身节点）：

| 节点包 | 主要用途 |
|--------|----------|
| `sdpose-ood` | SDPose 姿态识别模型加载与处理 |
| `ComfyUI-WanAnimatePreprocess` | ONNX 面部检测模型加载与面部图像截取 |
| `comfyui-videohelpersuite` | 视频文件加载（`VHS_LoadVideoFFmpegPath`）、图像分割（`VHS_SplitImages`）与音频加载 |
| `comfyui-kjnodes` | 常量节点、`CrossFadeImages` 跨段融合、`ColorMatch` 色彩匹配、`PatchSageAttentionKJ` 等 |
| `comfyui-easy-use` | While 循环控制、IfElse 条件分支、缓存清理、`easy imageSwitch` |
| `comfyui-impact-pack` | 循环条件比较（`ImpactCompare`）、空值检测（`ImpactIfNone`） |
| `comfyui-custom-scripts` | 数学表达式计算（`MathExpression\|pysssss`） |
| `ComfyUI-BodyRatioMapper` | 骨骼比例对齐（`BodyRatioMapperProportionTransfer`） |
| `rgthree-comfy` | `Any Switch` 多路图像切换、`Fast Muter` 快速开关 |
| `pr-was-node-suite-comfyui-47064894` | `Image Filter Adjustments` 图像滤镜调整、`Random Number` 随机种子 |
| `comfyui_memory_cleanup` | `RAMCleanup` / `VRAMCleanup` 内存/显存清理 |
| `ComfyUI-EasyColorCorrector` | `BatchColorCorrection` 批量色彩校正 |

### WanSCAIL2 多参考图生成长视频工作流

**文件**：`WanSCAIL2_ref_pics_loop+context.json`

**用途**：基于 Wan SCAIL/SCAIL-2 模型，使用多张参考图生成带有上下文窗口的长视频。支持多身份识别（通过 SAM3 追踪 + colored mask），可实现动画模式（Animation Mode）和替换模式（Replacement Mode）两种生成策略。

**简要使用说明**：

1. 在「模型、LoRA及VAE加载」区域加载 SCAIL 模型及相关组件；
2. 加载参考图并通过 SAM3 生成参考图追踪数据；
3. 加载驱动视频并通过 SAM3 生成驱动视频追踪数据；
4. 使用 `Create SCAIL-2 Colored Mask (Multi Ref)` 节点生成 colored mask，选择合适的排序方式（left_to_right/area）确保多身份颜色一致性；
5. 将 colored mask 和驱动视频接入 `Wan SCAIL To Video (Multi Ref)` 节点；
6. 运行后逐段生成视频帧，利用上下文窗口实现长视频生成；
7. 生成完成后将帧序列合成为视频文件。

---

## 依赖

### Python 依赖

核心依赖（已在 `requirements.txt` 中声明）：

- `torch` — PyTorch 深度学习框架（通常由 ComfyUI 环境提供）
- `numpy` — 数值计算
- `Pillow` — 图像处理
- `aiohttp` — 异步 HTTP 通信
- `colorsys` — 颜色空间转换（Python 标准库）
- `tqdm` — 进度条显示
- `imageio-ffmpeg` — 视频合成（自动下载 ffmpeg，无需系统安装）

---

## 使用示例

### 1. WanAnimateVideo 视频生成

```
[Reference Image] → WanAnimateToVideoCustom
[Pose Video]       →  (pose_video 输入)
[Face Video]       →  (face_video 输入)
                       ↓
               positive / negative / latent → [KSampler] → [VAE Decode] → 输出视频
```

### 2. SDPose 姿态可视化流水线

```
[SDPose JSON 文件] → Load SDPose JSON
                       ↓
               [Estimate Yaw] → yaw_array
                       ↓
               [Resize Keypoints] → 统一尺寸
                       ↓
               [Draw Keypoints V2] → 可视化图像 (支持 foot_mode=line/dots)
```

### 3. 上下文窗口视频生成

```
[模型] → Custom Context Windows (Manual) → model (已封装)
                                               ↓
[prompt] → CLIP Text Encode → positive ──────┤
[WanAnimateToVideoCustom] → positive/negative/latent → [KSampler] → [VAE Decode] → 视频
```

### 4. 参考图自动选择 + 视频生成

```
[多角度参考图] → Reference Image Selector ← yaw_angles ← [Estimate Yaw]
                       ↓
               selected_images → [WanAnimateToVideoCustom] reference_image
```

### 5. SCAIL/SCAIL-2 多参考图视频生成

```
[SAM3 参考图追踪] → ref_track_data ─┐
[SAM3 驱动追踪]    → driving_data ──┤
                                    ↓
                   Create SCAIL-2 Colored Mask (Multi Ref)
                                    ↓
               pose_video_mask / reference_image_mask
                                    ↓
[多张参考图] → reference_image ────┤
[驱动视频]   → pose_video ────────┤
[条件]       → positive/negative ─┤
                                  ↓
                   Wan SCAIL To Video (Multi Ref)
                                  ↓
                  positive / negative / latent → [KSampler]
```

---

## 致谢

本项目的部分节点为以下第三方节点提供辅助服务（如 bbox 检测、姿态数据预处理等），感谢这些优秀项目的启发：

- sdpose-ood — SDPose 姿态系统
- ComfyUI-WanAnimatePreprocess & comfyui-kjnodes (Kijai) — WanAnimate 预处理与工具包
- GroundingDINO (ShilongLiu) — GD 检测模型接口
- ComfyUI-BodyRatioMapper — 骨骼比例对齐
- comfyui-videohelpersuite — 视频工具
- comfyui-custom-scripts — 实用节点

同时，本项目中的 WanAnimate 多参考图视频生成逻辑、偏航角估计算法、上下文窗口调度策略、骨骼绘制引擎、SCAIL/SCAIL-2 多参考图支持等核心逻辑均为自主实现。

---

## 许可证

MIT License