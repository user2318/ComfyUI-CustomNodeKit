# ComfyUI Custom Node Kit

一套为 ComfyUI 设计的自定义节点工具集，涵盖视频生成、姿态处理、图像交互操作等常见工作流需求。

## 目录

- [安装](#安装)
- [节点列表](#节点列表)
  - [WanLoop 视频生成](#wanloop-视频生成)
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
  - [Folder Image Loader](#folder-image-loader)
  - [Image Batch Concat](#image-batch-concat)
  - [Image Batch Resize](#image-batch-resize)
  - [WanAnimate Uni3C 运镜控制](#wananimate-uni3c-运镜控制)
- [依赖](#依赖)
- [工作流](#工作流)
  - [长视频姿态检测工作流](#长视频姿态检测工作流)
  - [WanAnimate多参考图生成长视频工作流](#wananimate多参考图生成长视频工作流)
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
| **WanAnimateToVideoCustom** | `WanLoop/整合节点` | WanAnimate 视频生成核心节点，支持 pose/face 控制、尾帧分段掩码、中性灰混合、上下文模式等完整参数 |
| **WanUni3CLoader** | `WanVideo/Control` | 加载 Uni3C ControlNet 模型，支持 fp32/bf16/fp16 精度选择（可选） |
| **WanUni3CApply** | `WanVideo/Control` | 在 KSampler 中注入 Uni3C 运镜控制，支持 strength/start_percent/end_percent 等参数（可选） |

### SDPose 姿态系统

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Draw SDPose Keypoints (V2)** | `SDPose` | 将姿态关键点数据渲染为可视化图像，支持全身骨骼、手部、面部、脚部的分层绘制，并根据偏航角自动调整骨骼粗细与遮挡顺序 |
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

### 上下文工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Custom Context Windows (Manual)** | `context` | 通用上下文窗口调度节点。将长序列拆分为滑动窗口逐段处理，支持参考帧前缀、噪声混洗、多种调度策略（均匀/静态/循环/批处理）与融合模式（金字塔/线性叠加/相对加权），适用于需要分段式上下文管理的视频/序列生成任务 |

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
| `mode` | 枚举 | `legacy`=旧版尾帧分段掩码模式 / `fix`=硬替换中性灰+过渡扩散模式 |
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

#### 两种模式的差异

**legacy 模式**（推荐用于需要精细控制尾帧过渡的场景）：
- 尾帧掩码强度由 `tail_start_strength` → `tail_end_strength` 线性过渡（可配合 `mid_frame` 设置中间锚点实现非线性过渡）
- 支持中性灰混合：尾帧区域的图像会按掩码强度混合中性灰（`neutral_mix_min` 为掩码=0 时的混合比例，`neutral_mix_max` 为掩码=1 时的比例）

**fix 模式**（推荐用于黑帧过渡场景）：
- 纯黑帧被直接替换为中性灰（0.5）
- 黑帧两侧的过渡区域按 `transition_width` 进行渐变扩散
- 注意：纯白区域也会被视为有效内容

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

参考图选择器，用于根据视频片段的偏航角范围自动筛选最合适的参考图批次。典型应用场景：WanAnimate 视频生成中，需要根据人物朝向（yaw）从多角度参考图中选取最佳匹配。

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_images` | IMAGE | ✅ | 参考图批次（N 张不同视角的参考图） |
| `angle_map` | STRING | ✅ | JSON 格式的角度映射，如 `[-90, -45, 0, 45, 90]`，数组长度必须与参考图数量一致 |
| `yaw_angles` | FLOAT | ❌ | 目标视频片段的偏航角序列（每帧一个值），不接入时输出全量参考图 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `selected_images` | IMAGE | 筛选排序后的参考图批次（主参考图×1 + 辅助参考图各×1，原始图片直接输出） |
| `raw_reference_images` | IMAGE | 原始参考图（未排序，用于接入 Custom Context Windows 的 prefix_latent_num） |
| `info` | STRING | 调试信息（角度范围、候选索引、排序结果等） |
| `reference_angle_map` | STRING | 参考图角度映射 JSON（用于下游节点） |

#### 工作原理

1. **未接入 yaw_angles 或 angle_map 无效**：直接输出全部参考图（第 0 张×1 + 其余各×4）。
2. **有 yaw_angles 输入**：
   - 计算偏航角范围 [min, max]
   - 筛选所有在此范围内的参考图 + 左右各最邻近一张
   - 选出覆盖帧数最多的参考图作为**主参考图**（排在第 0 位）
   - 辅助参考图按与首帧偏航角的偏差从大到小排列
   - 若主参考图恰好最贴合首帧偏航角，末尾追加主参考图副本
3. **输出批次格式**：第 0 张（主参考图）×1，其余辅助参考图各×4（符合 WanAnimate 的 concat 格式要求）

#### 典型工作流

```
[多角度参考图] → reference_images ─┐
[角度映射 JSON] → angle_map ───────┤
[偏航角序列]    → yaw_angles ─────┤
                                   ↓
                        Reference Image Selector
                                   ↓
                          selected_images → [WanAnimateToVideoCustom] reference_image
```

> **提示**：`angle_map` 输入可直接连接前端 JS 节点 `Angle Map Config` 来可视化配置角度映射。

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

## 工作流

`workflow/` 目录下提供了两个完整的 ComfyUI 工作流 JSON 文件，可直接拖入 ComfyUI 界面使用。

### 长视频姿态检测工作流

**文件**：`Long_Pose_detection_while_new.json`

**用途**：对长视频进行逐帧骨骼（body）和面部关键点检测，支持将检测骨骼与参考图人物的骨骼进行比例对齐（BodyRatioMapper），最终生成骨骼可视化视频、面部特征 JSON 和姿态 JSON 文件。适用于制作舞蹈骨骼动画、动作捕捉数据提取等场景。

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
- 复杂预处理建议使用主工作流（`Long_Pose_detection_while_new.json`）进行。

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
              [Draw Keypoints V2] → 可视化图像
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

---

## 致谢

本项目的部分节点为以下第三方节点提供辅助服务（如 bbox 检测、姿态数据预处理等），感谢这些优秀项目的启发：

- sdpose-ood — SDPose 姿态系统
- ComfyUI-WanAnimatePreprocess & comfyui-kjnodes (Kijai) — WanAnimate 预处理与工具包
- GroundingDINO (ShilongLiu) — GD 检测模型接口
- ComfyUI-BodyRatioMapper — 骨骼比例对齐
- comfyui-videohelpersuite — 视频工具
- comfyui-custom-scripts — 实用节点

同时，本项目中的 WanAnimate 多参考图视频生成逻辑、偏航角估计算法、上下文窗口调度策略、骨骼绘制引擎等核心逻辑均为自主实现。

---

## 许可证

MIT License
