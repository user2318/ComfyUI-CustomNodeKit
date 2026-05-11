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
- [依赖](#依赖)
- [使用示例](#使用示例)
- [许可证](#许可证)

---

## 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录下：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/user2318/ComfyUI-CustomNodeKit.git ComfyUI-Custom_Tools
```

2. 安装 Python 依赖：

```bash
cd ComfyUI-Custom_Tools
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
| **Reference Image Selector** | `CustomNodes/SDPose` | 参考图选择器：根据偏航角范围自动筛选和排序参考图批次 |

### 视频工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Get Frame Count** | `VideoHelper` | 获取输入图像/视频帧数统计信息 |
| **Image Sequence to Video (ffmpeg)** | `VideoHelper` | 使用 ffmpeg 将图片序列合成为视频文件，支持自定义帧率、编码器参数 |

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
| **Integer Aligner** | `Utility` | 将整数对齐到目标步长的倍数 |

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
| `context_mode` | BOOLEAN | 启用上下文模式后，pose/face 视频将自动填充至与总 latent 长度匹配 |

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
| `prefix_frames` | INT | 前缀参考帧数（像素帧，自动转换）。这些帧会被固定追加到每个窗口开头，提供跨窗口的稳定上下文参考 |
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

1. **帧数转换**：`context_length`、`context_overlap`、`prefix_frames` 均以像素帧为单位输入，节点内部自动转换为 latent 帧（每 4 像素帧 → 1 latent 帧）。
2. **前缀参考帧**：设置 `prefix_frames > 0` 后，前 N 帧的 latent 会被追加到每个窗口开头作为稳定参考，适合需要全局上下文信息的生成任务。
3. **freenoise**：启用后会混洗噪声以改善窗口间的纹理连续性，建议在总帧数较长且重叠较小时开启。
4. **与 WanAnimateToVideoCustom 配合**：将本节点输出的 model 连接到 KSampler，同时将 WanAnimateToVideoCustom 的 context_mode 设为 `True` 以确保 pose/face 视频长度匹配。

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
| `selected_images` | IMAGE | 筛选排序后的参考图批次（主参考图×1 + 辅助参考图各×4） |
| `info` | STRING | 调试信息（角度范围、候选索引、排序结果等） |

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

## 依赖

核心依赖（已在 `requirements.txt` 中声明）：

- `torch` — PyTorch 深度学习框架
- `numpy` — 数值计算
- `colorsys` — 颜色空间转换（Python 标准库）
- `tqdm` — 进度条显示
- `ffmpeg` — 视频合成（需要系统安装 ffmpeg 命令行工具）

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

## 许可证

MIT License
