# ComfyUI Custom Node Kit

一套为 ComfyUI 设计的自定义节点工具集，涵盖视频生成、姿态处理、图像交互操作等常见工作流需求。

当前版本：**1.9.0**（2026-10-16）— 新增 **锚定 + 步内滚动（极简）**、**步内滚动** 与 **Detect Edges (Canny) Chunked**（分块版 Canny 边缘检测）：修复官方 Detect Edges (Canny) 在 200+ 帧批次上显存溢出导致「不报错、彻底卡死」的问题，实测 200 帧 1024² 峰值显存由约 29.7GB（外推）降到 1.31GB（详见 [更新日志](#更新日志)）；以下是 1.8.0 起的既有简介：**1.8.0** — 新增 WanAnimate2ToVideoCustom 节点（Wan2.1-I2V 底座 + 二代 pose 分支 K/V 注入，支持多参考图）；WanUni3CApply 重构为 render_video 输入（内置 VAE 编码 + 参考帧偏移，修复 render_latent 归一化偏差问题）；移除已退役的 WanSCAILSparseAttention 节点（长序列下掩码为 O(S²) 稠密矩阵，导致内存爆掉，实测在 SCAIL 上无收益）；另外还有一个**实验性 / 测试性节点 Context Window Anchor（上下文窗口锚定）——其有效性与副作用尚未充分验证，仅供测试参考；不接 `anchor_frames` 时既有行为完全不变**（详见 [Context Window Anchor](#context-window-anchor-上下文窗口锚定)）。

本版另修复了 SDPose 偏航角在短片段素材下的必崩问题、BatchImageReplace 的 repeat_end 填充逻辑（3 张替换 4 张由 `0,2,2,2` 修正为 `0,1,2,2`），并对智能视频裁剪、交互式批量裁剪等节点做了内存/性能优化（后者预览缓存从 2600 帧约 12.5GB 降到约 48MB）。**升级前请先阅读文末的 [更新日志](#更新日志)**（其中含不兼容变更说明）。

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
  - [WanAnimate2ToVideoCustom](#wananimate2tovideocustom)
  - [WanAnimateToVideoCustom](#wananimatetovideocustom)
  - [Custom Context Windows (Manual)](#custom-context-windows-manual)
  - [Context Window Anchor (上下文窗口锚定)](#context-window-anchor-上下文窗口锚定)
  - [Reference Image Selector](#reference-image-selector)
  - [SDPose 空帧修复](#sdpose-空帧修复)
  - [SDPoseDrawKeypointsV2 脚部绘制模式](#sdposedrawkeypointsv2-脚部绘制模式)
  - [Folder Image Loader](#folder-image-loader)
  - [Image Batch Concat](#image-batch-concat)
  - [Image Batch Resize](#image-batch-resize)
  - [WanAnimate Uni3C 运镜控制](#wananimate-uni3c-运镜控制)
  - [Interactive Batch Crop (交互式批量裁剪)](#interactive-batch-crop-交互式批量裁剪)
  - [MaskCompositeRef (遮罩批次合成 Ref)](#maskcompositeref-遮罩批次合成-ref)
  - [SmartVideoCrop (智能视频裁剪)](#smartvideocrop-智能视频裁剪)
  - [VideoGridCompose (视频网格拼接)](#videogridcompose-视频网格拼接)
  - [BatchImageReplace (图像批次替换)](#batchimagereplace-图像批次替换)
  - [ImageListSplitOutputs (图像列表分路输出)](#imagelistsplitoutputs-图像列表分路输出)
  - [ImageListReplaceAt (图像列表位置替换)](#imagelistreplaceat-图像列表位置替换)
  - [ImageListToBatch (图像列表转批次)](#imagelisttobatch-图像列表转批次)
- [SCAIL-2 节点详解](#scail-2-节点详解)
  - [Wan SCAIL To Video (Multi Ref)](#wan-scail-to-video-multi-ref)
  - [Create SCAIL-2 Colored Mask (Multi Ref)](#create-scail-2-colored-mask-multi-ref)
  - [Wan SCAIL-2 Context Windows](#wan-scail-2-context-windows)
- [工作流](#工作流)
  - [长视频姿态检测工作流](#长视频姿态检测工作流)
  - [WanAnimate多参考图生成长视频工作流](#wananimate多参考图生成长视频工作流)
  - [WanSCAIL2 多参考图生成长视频工作流](#wanscail2-多参考图生成长视频工作流)
- [更新日志](#更新日志)
- [依赖](#依赖)
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
| **WanAnimate To Video (自定义)** | `WanLoop/整合节点` | WanAnimate 视频生成核心节点，支持 pose/face 控制、尾帧分段掩码、中性灰混合、上下文模式、prev_latent 潜变量接续等完整参数 |
| **WanAnimate2 To Video (自定义)** | `WanLoop/整合节点` | WanAnimate2（官方二代架构）视频生成核心节点。保留一代多参考图（逐帧独立编码）与 fix/legacy/vanilla 掩码处理、中性灰混合、yaw 角度输出；叠加官方二代 pose_strength / pose_start_percent / pose_end_percent / reference_image_strength / positive_pose / clip_vision_output_pose。已适配二代 pose 分支帧数校验（多参考前补 latent 帧） |
| **单帧独立 VAE 编码** | `WanLoop/工具节点` | 将 N 张图片逐帧独立 VAE 编码后在时间维度拼接为 latent，为 ComfyUI-EverAnimate 项目提供锚点 latent |
| **WanVideo Uni3C ControlNet Loader (Custom)** | `WanVideo/Control` | 加载 Uni3C ControlNet 模型，支持 fp32/bf16/fp16 精度选择（可选） |
| **WanVideo Uni3C Apply (for KSampler)** | `WanVideo/Control` | 在 KSampler 中注入 Uni3C 运镜控制，支持 strength/start_percent/end_percent 等参数（可选） |

### SCAIL/SCAIL-2 多参考图生成

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **Wan SCAIL To Video (Multi Ref)** | `model/conditioning/video_models` | SCAIL/SCAIL-2 多参考图视频生成 conditioning 节点。支持多张参考图自动编码、SCAIL-2 多身份 colored mask 注入、pose 视频引导、运动接续（prev_latent）等 |
| **Create SCAIL-2 Colored Mask (MultiRef)** | `conditioning/video_models/scail` | 渲染 SAM3 追踪数据为 SCAIL-2 使用的 colored mask。支持参考图和驱动视频的共享调色板排序（left_to_right/area），确保多人物场景下同一身份颜色一致 |
| **Wan SCAIL-2 Phase Sampler** | `sampling` | 独立两阶段采样节点。完整复刻 SCAIL2LoopSampler 的两阶段采样核心逻辑。用像素帧（IMAGE）作为锚定输入，内部完成 VAE 编码、锚定帧回写、noise_mask 管理 |
| **CLIP Vision Multi-Ref Switch** | `conditioning/video_models/scail` | 多参考图 CLIP 特征合并节点。将批次中 N 张图的 CLIP Vision 特征拼接到 token 维度，使所有参考图都参与 conditioning |
| **Auto Color Drift Correction V4.3 (两遍校正)** | `CustomNodes/Video` | 自对齐色彩漂移校正节点 V4.3。两遍校正：① 第一遍全局偏移——多帧均值检测段间跳变后统一校正所有新帧；② 第二遍残余微调——在钳制上限内逐帧向基准靠拢。支持 standard/auto/off 三种模式，`max_offset` 控制单通道偏移上限 |

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
| **Reference Image Selector (参考图选择器)** | `CustomNodes/SDPose` | 参考图选择器：根据偏航角范围自动筛选和排序参考图批次 |

### 视频工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **视频帧数统计** | `video` | 获取输入视频/图像序列的帧数统计信息与帧率 |
| **图片序列合成视频** | `video` | 使用 ffmpeg 将图片序列合成为视频文件，支持 CRF、音频合成、帧偏移等参数 |
| **智能视频裁剪 (Smart Video Crop)** | `video` | 基于 SAM3 追踪数据的智能视频裁剪：按人物质心做水平跟随裁剪（EMA 平滑 + 切镜检测），固定保留全高后按输出宽高比缩放。支持多 ID 追踪、空 mask 回退策略（保持最后位置/回到中心）、可选第二路图像同步裁剪 |
| **视频网格拼接** | `video` | 把 1~4 路视频/图片按布局拼成网格（水平排列 / 垂直排列 / 田字格2×2 / 左一右N / 上一+下N），支持自适应填满或保持比例、时长对齐（超长截断、不足冻结补帧）、音频来源选择；全程 ffmpeg 流式处理，不占内存 |

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
| **Folder Image Loader** | `image` | 从文件夹按文件名升序载入图片；输出同尺寸批次 `images` + 张数 `count` + 逐张列表 `images_list`（列表语义，`size_mode=no_resize` 时列表保持原尺寸）。支持跳过、数量限制与尺寸处理 |
| **Detect Edges (Canny) Chunked** | `image` | 分块版 Canny 边缘检测（解决官方 Detect Edges (Canny) 在 200+ 帧批次上「不报错、无进度、卡死」的问题）：逐块调用同一算子链，结果一致（真实图像 0 像素差异），峰值显存从约 25GB 降到约 1GB；带进度条、可随时中断，`auto_chunk` 按空闲显存自动收缩。 |
| **Image Batch Concat** | `image` | 将两个图像批次拼接，任意一路无输入时透传另一路 |
| **Image Batch Resize** | `image` | 缩放图像批次到指定宽高，可选按方向裁剪保持宽高比 |
| **Batch Image Replace** | `image` | 批量图像替换节点。在图像批次中按起始索引和数量替换指定位置的图像。支持溢出模式和下溢模式，替换源可选连接 |
| **Batch Frame Replicate** | `image` | 图像批次指定帧复制节点。在图像批次中复制指定图像并插入到其后。支持负索引和双路独立输入。用于延长首/尾帧时长、关键帧停留效果 |
| **Image List Split (图像列表分路输出)** | `image` | 把**图像列表**按指定路数（1~16，界面输出口随该值即时增减）拆成多个独立输出。从 `start_index` 开始依次输出，不足的路为空（返回 `None`，**效果等同该路未接线**：下游 optional 输入按"没有这一路"处理），输入未连接时同样不报错也不输出。列表元素尺寸可互不相同；接普通批次会自动拆成单张 |
| **Image List Replace (图像列表位置替换)** | `image` | 用替换列表从 `start_index` 开始覆盖目标图像列表的一段，长度由替换列表决定且**绝不超过原列表长度**（多余的替换图丢弃）；逐元素整体替换、**不缩放**、长度不变，输出为列表（下游按元素逐个执行） |
| **Image List To Batch (图像列表转批次)** | `image` | 把尺寸不同的图像列表转成同尺寸批次（**绝不裁剪/拉伸**）：基准取「短边最长（＝面积×方正度最大）」的那张，**其原始尺寸即输出尺寸**（零重采样），其余图**只在一个轴上对称补边**后等比缩放。`output_mode` 默认 `RGB`（3 通道、补边纯黑、最兼容 VL/VAE），可切 `RGBA`（4 通道、补边 alpha=0、适合合成） |
| **遮罩批次合成 (Ref)** | `image` | 遮罩批次合成节点 — 用于 WanSCAILToVideo 的 reference_image 预处理。支持替换模式和动作迁移模式，自动剔除全黑遮罩帧、背景合成 |


### 上下文工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **WAN Context Windows (Manual Custom)** | `context` | 通用上下文窗口调度节点。将长序列拆分为滑动窗口逐段处理，支持参考帧前缀、噪声混洗、多种调度策略（均匀/静态/循环/批处理）与融合模式（金字塔/线性叠加/相对加权） |
| **Wan SCAIL Context Windows** | `context` | SCAIL-2 专用上下文窗口节点。自动处理 SCAIL-2 的特殊 conditioning 字段（ref_mask_28ch、driving_mask_28ch、pose_video_latent），无需前缀参考帧。与 WanSCAILToVideoMultiRef 配合使用 |
| **窗口锚定** | `context` | 【实验性·测试性】给每个窗口的**驱动视频（含 SCAIL-2 驱动遮罩）开头接一段全局锚定帧**，并把同样的帧按当前 σ 加噪后放到待生成序列最前（当普通帧），采样后裁掉——等价于手动「把整段视频开头拼到片段开头再生成」。接口极简：`anchor_frames`(IMAGE) + `vae`（同一个 Wan VAE）+ SCAIL-2 的 `anchor_mask_frames`；**给多少用多少**。不接 `anchor_frames` 即等价于不装本节点。必须接在上下文窗口节点之后 |

---

### 补充列表（2026-10-15 按代码校对）

> 下面这些节点在原表里没有单列，按代码现状补齐（顺序/归类可自行调整）。

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **锚定+步内滚动（极简）** | `context` | 「窗口锚定 + 步内滚动」的一键整合节点（面板只有 enabled 一个设置项） |
| **索引选择器** | `Custom Nodes` |  |
| **路径验证器** | `Custom Nodes` |  |
| **文件夹图像载入器** | `image` | 从文件夹按文件名升序载入图片：images 为同尺寸批次；images_list 为逐张列表（元素 [1,H,W,C]，尺寸可互不相同，下游按元素逐个执行） |
| **图像批次合并** | `image` |  |
| **图像批次指定帧复制 (Frame Replicate)** | `image` |  |
| **步内滚动** | `context` | SCAIL-2 滚动窗口：在同一步内让后面的窗口看到前面窗口【本步】刚产出的 x0 |

## 复杂节点详解

### WanAnimate2ToVideoCustom

基于官方 `WanAnimate2ToVideo`（Wan2.1-I2V 底座 + pose 分支 K/V 注入）与一代 `WanAnimateToVideoCustom` 融合的二代 animate 节点。保留一代的多参考图（逐帧独立编码，兼容 EverAnimate LoRA；原模式 1+4n 批量编码在 animate2 上表现不佳已移除）、fix/legacy/vanilla 掩码处理、黑帧检测+过渡、尾帧分段掩码、中性灰混合、yaw_angles→latent_yaw_angles 输出；叠加官方二代的 `pose_strength` / `pose_start_percent` / `pose_end_percent`（pose 时间窗口，窗口外跳过 pose 分支可提速）/ `reference_image_strength` / `positive_pose` / `clip_vision_output_pose`。pose 分支按二代模型 `pose_latents.shape[2] == f_gen - 1` 的帧数校验要求前补 latent 帧，多参考图场景可直接使用。已去掉二代架构不支持的 face/background/character_mask/prev_latent。

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
| `freenoise` | BOOLEAN | 是否启用 FreeNoise 噪声混洗（改善窗口间连续性）。⚠️ **2026-10 修复**：覆写块现在按**真实窗口起点**对齐（旧实现假定步长均匀 `delta=C−OV`，在 `standard_even` 这类**非均匀起点**的调度下会错位 ⇒「噪声池→新帧」的对应关系碎掉）；等步长调度下与旧实现**逐位一致** |
| `prefix_latent_num` | INT | 前缀参考帧的 latent 数量。接参考图选择器 raw_reference_images 的图片数量即可（每张图片编码为1个 latent）。这些帧会作为稳定参考拼接到每个窗口前 |
| `split_conds_to_windows` | BOOLEAN | 是否将多个 conditioning 按区域索引分配给各窗口 |
| `fuse_sharpen` | FLOAT | **融合权重锐化（幂次）**，默认 1.0=不锐化（历史行为）。设 3~6 可把重叠区的交叉淡化带压窄：任一瞬间基本只取「更靠窗中心」的那一方 ⇒ **既保留连续性（仍是渐变）又不出现两个角色叠在一起的重影**。窗口重叠越大、两窗分歧越大时越有用 |

#### 调度策略 (context_schedule)

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `looped_uniform` | 均匀循环调度：窗口在时间轴上循环分布，步长按指数递增 | 需要多次迭代覆盖的场景 |
| `standard_uniform` | 标准均匀调度：与 looped 类似但不循环，去除重复窗口 | 大多数通用场景的首选 |
| `standard_static` | 标准静态调度：固定步长的滑动窗口（步长 = context_length - context_overlap）。⚠️ 末窗越界时会**回头对齐到序列尾**，使末缝重叠远大于设定值（会告警） | 需要简单分段处理的长视频 |
| `standard_even` | **等长窗口 + 末窗不回退**：窗口起点在 `[0, 总长-窗口长]` 上均匀分布，重叠被摊平（不再出现"末缝超大重叠"） | 不想改窗口长度、又想修掉末缝重影 |
| `batched` | 批处理调度：按 context_length 直接切分，**无重叠**（末窗变短） | 配合 `Context Window Anchor` 的 head/tail 只读上下文做"每帧只有一个作者" |

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
4. **重叠与重影**：重叠区会被加权平均（`fuse_method`），两个窗口对尺度/位置判断不同时会出现"两个角色叠在一起"。三条应对：① `fuse_sharpen=3~6`（把淡化带压窄，仍连续）；② 换 `standard_even` 修掉"末缝超大重叠"；③ 用 `batched` + `Context Window Anchor` 的 `head/tail_context_frames` 做到"每帧只有一个作者"。
5. **重叠总量是被几何锁死的**：`总重叠 = 窗口数 × 窗口长 − 总长`。想让重叠变小，必须用**更短 + 更多**的窗口（例如 43 latent 用 11 latent × 4 窗），否则调度再怎么换也躲不掉。

#### 典型工作流

```
[加载模型] → model ──┐
                      ↓
           Custom Context Windows (Manual)
                      ↓
               model (已封装) → [KSampler] → [VAE Decode] → 输出
```

---

### Context Window Anchor (上下文窗口锚定 / 只读上下文 / 长程记忆组)
> ⚠️ **实验性 / 测试性节点（未充分验证）**：其**有效性**与**副作用**（画质、人物一致性、采样稳定性）尚未系统评估，请先用**同一 seed** 做「接 / 不接」对比再决定是否长期使用。**只接本节点、又不填任何参数时完全等价于不装本节点。**

**要解决的问题**：animate2 / SCAIL-2 没有骨骼这类显式空间锚；窗口化后每个窗口的 RoPE 都从 0 重新计数（不含绝对时间），而 SCAIL-2 的「驱动帧 ↔ 生成帧」是按 RoPE **同位逐帧刚性配对**的。于是每个窗口只能靠自己那段驱动内容推断基准 ⇒ 相邻窗口在重叠区给出**两份不同的动作预测**，而融合是**加权平均** ⇒ 重影 / 动作发软 / 个别窗口不准。另外 `standard_static` 调度在末窗越界时会「回头对齐」，会让**末缝重叠**远大于设定值（例如 20/6 latent 下变成 11 latent），那一段最容易重影。

**本节点给上下文窗口补三样东西（都是「内容层」干预 —— 位置编码补不了这些）：**

1. **锚定帧**（`anchor_frames`，可选）：所有窗口共享同一段「前文/基准」运动。
   - `driving_only = 开`（默认，SCAIL-2）：锚帧作为**额外的驱动 token 追加在驱动流尾部**，再由 RoPE 补丁把它们放到「**本窗口之前**」的位置（`t = N_ref-K .. N_ref-1`，不占用任何视频帧的位置，也不与视频帧配对）⇒ **既有几何完全不变、不会把动作整体错位 K 帧**（旧实现前插在驱动流最前，会让整段生成相对驱动视频**滞后 K latent = 4K 像素帧**，已修）。
   - `driving_only = 关`：锚帧前插到驱动流 **与待生成序列**（与视频帧逐帧配对，采样后连同预测一起裁掉），最接近训练语义（训练时每块头部是上一块的输出）。
2. **只读上下文**（`head_context_frames`，**单位 = latent 帧**）：每个窗口头部各看几帧**真实邻居**（参与注意力、**不参与融合**，融合前被裁掉）。配上 `overlap = 0` 时，**每一帧只由一个窗口生成**（不再有「两份预测取平均」）。
   - 内容 = 共享 latent `x` 上那几帧，即**上一步迭代后的值**（不是本步刚算出来的结果）：采样早期（σ≈1）它们基本就是噪声、没有信息量，只有后期步才逐渐有用。
   - ⚠️ **尾段 `tail_context_frames` 已移除**：机制上不产生可见效果 —— 尾段的预测在融合前就被裁掉，且那些帧的「作者」是后面的窗口（你只是「偷看一眼」，改不了结果）。
3. **长程记忆组**（`memory_frames`，**单位 = latent 帧**）：取本窗口**之前**的历史驱动帧作为额外 token，追加在驱动流尾部；RoPE 位置**如实编码那个间距与先后顺序**（不会被当成相邻帧，也不会被当成「未来驱动」）。
   - **间距由 `memory_stride` 决定**：`0`（默认）= 自动 —— 让 `memory_frames` 个点刚好覆盖「**片段开头 → 本窗口之前**」的全部历史 ⇒ 片段再长、窗口再靠后也覆盖得到开头（间隔自动变大、时间分辨率变粗）；`>0` = 固定间距（`4` ≈ 每 16 像素帧取 1 帧）。

> ⚠️ **帧单位别搞混**：`anchor_frames`（IMAGE）是**像素帧**（给多少用多少，内部按 `((T-1)//4)+1` 转成 latent）；而 `head_context_frames` / `memory_frames` / `memory_stride` 都是 **latent 帧**。

#### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | MODEL | — | 已接入上下文窗口节点的模型（`Custom Context Windows (Manual)` 或 `Wan SCAIL-2 Context Windows` 的输出） |
> 2026-10-15 起 `driving_only` / `anchor_fixed` / `anchor_image_blank` / `anchor_scope` / `shared_head` / `overlap_scope` / `memory_position` 已从面板移除（值固定为实测档）；需要 A/B 时改代码里的签名默认即可。

| `enabled` | BOOLEAN | true | 总开关；关闭时模型直接透传 |
| `head_context_frames` | INT | 0 | 每窗头部纳入 N 帧「只读前文」（参与注意力、不参与融合）；配 `overlap=0` 用 |
| `memory_stride` | INT | 0 | **长程记忆组的抽样间距**（latent 帧）：`0`=自动（推荐）——让 `memory_frames` 个点覆盖「片段开头 → 本窗口之前」的全部历史；`>0`=固定间距（`4` ≈ 每 16 像素帧取 1 帧）。⚠️ 本参数**占用原 `tail_context_frames` 的槽位**：老工作流该位置的值（通常是 0）会落到这里，而 `0` 恰好 = 自动，语义兼容 |
| `memory_frames` | INT | 0 | 长程记忆组：取本窗之前 **N 个历史 latent 帧**作为额外驱动 token（0=关闭，仅 SCAIL 有效）；间距见 `memory_stride`。自动模式下 `N` 就是**覆盖预算**（`N=4` = 用 4 个点概括本窗之前的全部历史） |
| `debug_log` | BOOLEAN | false | **诊断日志**（默认关，对结果零影响）：开=打印「锚定池空间 / 每窗条件流布局 / 锚段自检 / 跨窗共享头 / **两窗一致性（level + 动作）**」；关=由哨兵文件 `CWA_DEBUG.on` 或环境变量 `CWA_DEBUG` 决定。**切它无需重启 ComfyUI**，适合做开/关 A/B。🆕 **2026-10 日志分级**：默认档已大幅精简（几何 / 流布局 **每个片段只打一次**；每步只留「每缝一行」的关键比率）；**逐流统计 / 逐帧列表**移入详细档 —— 设环境变量 **`CWA_DEBUG_VERBOSE=1`**（重启生效）才输出 |
| `anchor_frames` | IMAGE（可选） | 不接=不做锚定 | **基准帧**：驱动/pose 视频里你想当统一基准的那几帧（例如整段视频开头几帧）。**给多少用多少** |
| `anchor_mask_frames` | IMAGE（可选，仅 SCAIL-2） | 不接 | 与 `anchor_frames` **同帧号**的彩色遮罩帧（内部转 28 通道，不需要 VAE） |
| `vae` | VAE（接锚帧时必接） | 不接 | 与 conditioning 节点**同一个** Wan VAE；接了 `anchor_frames` 却没接 `vae` 会**直接报错** |

> 只读上下文（`head_context_frames`）与长程记忆组（`memory_frames` / `memory_stride`）**不需要任何输入连线**，纯参数即可；长程记忆组连 VAE 都不需要。
>
> ⚠️ `anchor_frames` 来自**驱动视频的画面**。它在待生成序列里只当「占位」用（内容由模型自己生成）；如果把它当「已知干净帧」钉住，就会把**驱动者形象**带进画面（历史文档里的「驱动者形象伪影」）。

> ✅ **2026-10 修复与新增（两处，都直接决定「锚定有没有效」）**：
> 1. **latent 空间**：锚定池来自 `vae.encode()`（**原始** latent），而模型真正消费的「模型空间键」（如 `pose_latents`）与采样中的 x 都是**归一化** latent（`process_latent_in` / `Wan21.process_in`）。此前直接把原始池插进归一化流 ⇒ 锚帧 = 「幅度 2~3 倍 + 每通道直流偏移」的**畸形帧** ⇒ 结构上不可能成为「共同基准」。现在 **x / pose 池一律先归一化，mask（28ch）不归一化**。
> 2. **跨窗共享头**：`driving_only=关 + anchor_fixed=关` 时，头**每步只演化一次** —— 由本步第一个被评估的窗口（leader）产出，步边界轮转后**下一步所有窗口（含 leader）逐位相同**地使用它 ⇒ 对应手动测试里「两段各自拼上同一段视频开头」的共同基准。旧行为是各窗按自己的上一步 x0 各自演化 ⇒「共同基准」实际不成立（回退链：共享头 → 本窗上一步 x0 → 干净锚帧）。
>
> 🔍 **诊断**：把节点上的 **`debug_log` 打开**（默认关、对结果零影响、**不用重启 ComfyUI**）即打印 ① 锚定池空间 + 每窗条件流的长度/额外 token/锚段自检 + 共享头 leader；② **两窗重叠区预测分歧**（`|Δ|均 / p95 / 相对 / 逐帧`）——它正是「重影」的直接量纲（旧 devlog §10 基线：正常素材 0.22~0.35）。也可用哨兵文件 `CWA_DEBUG.on`（放在节点目录）或环境变量 `CWA_DEBUG=1` 打开。

#### 推荐 A/B 清单（同 seed，一次只改一个变量）

| 组 | 关键设置 | 想回答什么 |
|---|---|---|
| **A（基线）** | `overlap_scope=average` + `shared_head=off` + `anchor_scope=fresh_only` + 锚定头（`driving_only=关`、`anchor_fixed=关`、`anchor_image_blank=关`、锚帧 = 视频开头 5~9px + 同帧号 mask、`head_context_frames=0`、`memory_frames=0`） | 记录 `level 相对` / `动作 比值` + 画面 |
| **B（每帧一个作者）** | 只把 `overlap_scope` 改 **`single`** | 动作是否不再被"平均"掉（`动作 比值` 应下降） |
| **C（首窗也插头）** | 只把 `anchor_scope` 改 **`all`** | 第一道缝（首窗 ↔ 次窗）是否变好 |
| **D（更强共享头）** | 只把 `shared_head` 改 **`consensus`**（或 `leader`） | 跨窗共同基准能否把两窗再拉近 |

> 判据：日志里的 `level:…相对=` 与 `动作:…比值=`（**动作口径**才是"人的动作变了没有"；level 小 ≠ 动作对），再加画面。
> **每窗 token = 窗口长 + 锚帧数 + 只读前文 ≤ 21**（训练长度）。

#### 典型接线

```
[驱动/pose 视频] --ImageFromBatch(start=0, length=9)--> anchor_frames
[同一个 Wan VAE] -------------------------------------> vae
[驱动遮罩视频]   --ImageFromBatch(start=0, length=9)--> anchor_mask_frames   (仅 SCAIL-2)

[模型] → Custom Context Windows / Wan SCAIL Context Windows → Context Window Anchor → KSampler
```

#### 推荐尝试顺序（一次只改一个变量）

1. **只读上下文**：`overlap = 0`（调度改 `batched`，或 `standard_even` 让重叠摊平）+ `head_context_frames = 2`（latent 帧；让「自身帧数 + 头段」保持在 21~23 latent 以内）；
2. **锚定帧**：`anchor_frames` 取整段开头 5~9 像素帧（SCAIL-2 记得同帧号 `anchor_mask_frames`），`driving_only` 开/关各跑一次对比；
3. **长程记忆组**：`memory_frames = 4 → 8`（`memory_stride` 保持 `0` = 自动）。

> **多 seed 配对 A/B**：本节点内部唯一的随机源是**写死种子的噪声**，所以「同一 seed + 同一配置」的结果是**确定性的**。这意味着「某些 seed 有效、某些无效」是**真实差异**（而非随机噪声）⇒ 请固定若干 seed，每个 seed 都跑「开 / 关」两遍，统计成功率，而不要用单次结果下结论。

> 未接在上下文窗口节点之后时，节点会直接报错提示（不会静默产出错误结果）。

#### 已知边界

- 锚定帧必须与「驱动/pose 视频」同源同帧号（SCAIL-2 还要同帧号的遮罩）。
- 长程记忆组的 token 没有身份掩码（遮罩填 0 = 无身份信息）；它只提供**运动上下文**，不提供身份。
- 记忆组与「锚帧放在窗口之前」依赖 SCAIL 的 RoPE 位置补丁（`_apply_scail_anchor_rope_patch`）；replacement 模式与无额外 token 时补丁自动回退、不改变原有几何。
- 旧版本的**替换式锚定**与 `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug` 均已移除。
- ⚠️ **旧版本 `driving_only` 的判定条件写反了**（`and not _needs_driving_mask(model)`）：结果是 **SCAIL-2 上 `driving_only=开` 实际走了「前插（对齐）」那条路**（锚帧前插驱动流 + 待生成序列，逐帧配对），而 `driving_only=关` 反而只接驱动侧；animate2 侧则相反（会撞 `pose = f_gen − 1` 硬校验）。现已修正为与文档一致。⇒ **若你之前用 `driving_only=开` 得到过某个基线，那条基线对应的是「前插」路径；修复后 `driving_only=开` = 「尾部 + 放在窗口之前」路径。两者都不错位，可直接 A/B（想复现旧行为就把 `driving_only` 关掉）。**

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
| `size_mode` | 枚举 | ✅ | `resize_to_first`：所有图片缩放至第一张的尺寸；`filter_same_size`：仅加载与第一张尺寸相同的图片；`no_resize`：**只对列表输出 `images_list` 有效** —— 列表保持原尺寸（不缩放、不按尺寸过滤），批次输出无法容纳不同尺寸，会自动回退为 `resize_to_first` 并在控制台提示 |
| `skip_first_n` | INT | ✅ | 跳过前 N 张图片（默认 0） |
| `load_count` | INT | ✅ | 最多加载 N 张图片（0=不限制，加载全部） |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 同尺寸图片批次 `[B,H,W,C]`。路径无效或无图片时输出 None |
| `count` | INT | 实际载入的图片数量（= 批次张数）。异常情况输出 0 |
| `images_list` | IMAGE（列表语义） | 逐张列表输出，元素为 `[1,H,W,C]`，**下游按元素逐个执行**，尺寸可互不相同 |

#### 批次输出与列表输出的关系

- 列表语义沿用本仓库既有约定（同 `image_list_nodes.py`）：空输出一律为 `[None]`（单个 None 元素）——
  **不能**是 `None`（ComfyUI `execution.py::merge_result_data` 会对列表输出做 `extend`，`extend(None)` 直接 TypeError），
  也**不能**是 `[]`（下游 `slice_dict` 会取空列表的 `[-1]` → IndexError）。`[None]` 传给下游等价于「这根线没接」。
- `resize_to_first` / `filter_same_size`：列表就是批次的逐张视图（**零拷贝**，不额外占像素内存），内容与 `images` 完全一致。
- `no_resize`：列表保持各自原尺寸且**不做尺寸过滤**（尺寸不同的图只出现在列表里、不进入批次），此时 `images` 自动回退为 `resize_to_first`。
  内存方面只有「尺寸与第一张不同」的图会额外保留原图；尺寸一致的原图直接复用批次行（同一块内存），
  所以常见的「文件夹内尺寸一致」场景是**零额外开销**。
- 把 `images_list` 接到普通 IMAGE 输入时，ComfyUI 会对该下游**按元素逐个执行**（列表传播语义），这是预期行为。

#### 容错行为

- 文件夹路径为空 / 文件夹不存在 / 文件夹内无图片 / 跳过数超过图片总数 → 输出 `(None, 0, [None])`，控制台打印警告
- 单张图片读取失败 → 跳过该图，继续加载其余图片
- 第一张图片读取失败 → 走上面的空输出（既有行为，保持不变）

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
| `render_video` | IMAGE | ✅ | 运镜参考视频（像素帧），节点内部 VAE 编码后作为运镜控制的视觉参考 |
| `vae` | VAE | ✅ | 用于编码 render_video 的 VAE |
| `strength` | FLOAT | ✅ | 控制强度（0.0-10.0，默认 1.0） |
| `start_percent` | FLOAT | ✅ | 控制生效起始百分比（0.0-1.0，默认 0.0） |
| `end_percent` | FLOAT | ✅ | 控制生效结束百分比（0.0-1.0，默认 1.0） |
| `trim_latent` | INT | ❌ | 参考图占用的 latent 帧数，从 WanAnimateToVideoCustom 的 `trim_latent` 输出接入，用于时序对齐 |

**输出**：`model`（MODEL）— 已注入控制的模型，可直接连接 KSampler。

#### 核心原理

1. **v4 架构**：采用 monkey-patch 方式修改模型 `forward_orig` 方法，在 block 循环中精确控制注入时机
2. **参考帧偏移**：注入时跳过参考帧 token（与官方 `img_offset` 语义一致），避免参考帧被污染及控制信号时序错位；支持 WanAnimate concat / ref_conv / SCAIL / context window 场景
3. **latent 归一化**：render_video 内部 VAE 编码后应用 `latent_format.process_in`，与采样中已归一化的 x 尺度对齐，避免控制信号偏向 render 内容
4. **step 控制**：通过 `start_percent` / `end_percent` 控制控制信号在 denoising 过程中的生效范围

#### 典型工作流

```
[WanUni3CLoader] → uni3c_controlnet ─┐
[运镜参考视频] → render_video ────────┤
[VAE] → vae ────────────────────────┤
                                       ↓
[模型] → WanUni3CApply → model (已注入控制) → [KSampler] → [VAE Decode] → 输出
```

> **注意**：旧版提供过 `positive`/`negative` 条件接口和 `render_mask`，实际为无效逻辑（写入的键从未被读取），已在 v4 清理移除。

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

### SmartVideoCrop (智能视频裁剪)

基于 SAM3 追踪数据的智能视频裁剪：以人物质心为基准做**水平跟随裁剪**，固定保留原始画面全高，再按输出宽高比缩放到目标尺寸。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `frames` | IMAGE | 当前段原始视频帧批次 (B, H, W, C) |
| `track_data` | SAM3_TRACK_DATA | SAM3_VideoTrack 的追踪数据（接 `driving_track_data`） |
| `output_width` / `output_height` | INT | 输出尺寸；裁剪宽度按 `H × output_width / output_height` 推算 |
| `object_ids` | STRING | 要追踪的人物 ID，逗号分隔（如 `1` 或 `1,2`）；空 = 默认 ID 1 |
| `smoothing` | FLOAT | 帧间 EMA 平滑（0 = 完全跟随、抖动明显；1 = 几乎静止） |
| `mask_fallback` | 枚举 | mask 为空时：`hold_last` 保持最后有效位置 / `center` 回到画面中心 |
| `scene_cut_threshold` | FLOAT | 切镜检测阈值（归一化）：连续有 mask 的帧间质心跳变超过该值即判定切镜并重置平滑；0 = 关闭 |
| `fallback_center` | FLOAT | `mask_fallback=center` 时的回退水平中心比例（0=最左, 0.5=居中, 1=最右） |
| `mask_frames` | IMAGE | 可选第二路图像（如遮罩视频），与 `frames` 共用同一裁剪窗口同步输出 |

#### 输出

- `cropped_frames`：裁剪并缩放后的帧批次
- `cropped_mask`：第二路的同步裁剪结果（未接时为空批次）
- `debug_info`：每帧裁剪位置等 JSON 调试信息

> 输入比目标更窄时会自动切换为"垂直居中裁剪"（用全宽、按比例算高），因此不会裁掉画面内容。
> 内部按 **8 帧分块批处理** grid_sample，长序列比逐帧处理更快（裁剪结果与原逐帧实现逐位一致）。

---

### VideoGridCompose (视频网格拼接)

把 1~4 路视频/图片拼成一个网格视频，**全程由 ffmpeg 流式处理**，不会把帧读进内存（长视频同样安全）。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `video_paths` | STRING | 多行路径，每行一个（最多 4 路，自动去重去空行）；某一行是**图片**时会自动以 `-loop 1 -t <目标时长>` 当作静止视频流处理 |
| `layout` | 枚举 | `水平排列` / `垂直排列` / `田字格2×2`（需 4 路）/ `左一右N` / `上一+下N`（后两者需 ≥3 路）；路数不足时自动降级为水平排列 |
| `audio_source` | 枚举 | `无音频` 或 `视频1`~`视频4`（取对应输入的音频轨） |
| `match_method` | 枚举 | `自适应(填满)`=拉伸铺满 / `自适应(保持比例)`=等比缩放+黑边 / `统一到最大` / `统一到最小` |
| `fps` | FLOAT | 输出帧率；0 = 取输入中的最大帧率 |
| `crf` | INT | x264 质量（越小越好，建议 18~28） |
| `duration_mode` | 枚举 | 时长基准：`自动(最短)` / `自动(最长)` / `视频1`~`视频4`；比基准长的截断、短的冻结最后一帧补齐，音频同步 atrim/apad |
| `output_path_prefix` | STRING | 输出路径前缀（相对 ComfyUI 的 output 目录） |

输出 `video_path`（STRING），文件写入 `ComfyUI/output/<output_path_prefix>_0001.mp4`（自动递增避免覆盖）。

---

### BatchImageReplace (图像批次替换)

在主图像批次中，从 `start_index` 开始按 `replace_count` 替换若干帧。替换源（`replace_images`）的数量可以与需求不一致，由两个模式决定如何处理。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_index` | INT | 从主批次哪个位置开始替换（0-based） |
| `replace_count` | INT | 替换多少张；`-1` = 从 start_index 一直替换到末尾；`0` = 不替换（原样透传） |
| `overflow_mode` | 枚举 | **替换源多于需求**时：`take_front` 取前 N 张 / `take_back` 取后 N 张 |
| `underflow_mode` | 枚举 | **替换源少于需求**时：`as_is` 只替换源里实际有的张数（不扩展）/ `repeat_begin` 前面用第 1 张补齐 / `repeat_end` 按原顺序保留全部、末尾用最后 1 张补齐 |
| `images` | IMAGE | 主图像批次（未连接时输出为空） |
| `replace_images` | IMAGE | 替换源批次（未连接时原样透传主批次） |

#### underflow 语义示例（替换源 3 张、需替换 4 张）

| 模式 | 替换结果 |
|------|----------|
| `as_is` | 只替换 3 帧（第 4 帧保持原样） |
| `repeat_begin` | `0,0,1,2`（第 1 张重复到最前面） |
| `repeat_end` | `0,1,2,2`（按原顺序保留，末尾补最后 1 张） |

> `repeat_end` 在 v1.8.0 修正过一次：旧实现会丢弃中间帧得到 `0,2,2,2`，升级后行为与上表一致。

#### 其它说明

- 替换源与主批次分辨率不一致时会自动缩放到主批次尺寸
- `start_index` 超出批次长度时打印警告并原样返回主批次

---

### ImageListSplitOutputs (图像列表分路输出)

把**图像列表**按指定路数拆成多个独立输出：列表第 `start_index` 个元素送到第 1 路，第 `start_index + 1` 个送到第 2 路，以此类推。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_count` | INT | 输出路数（1~16，默认 2）。**画布上的输出口会随该值即时增减**，界面只会显示设定的路数 |
| `start_index` | INT | 从列表第几个元素开始输出（0 = 第 1 个，1 = 第 2 个） |
| `images` | IMAGE | 图像列表输入（可选）。未连接时不报错、也不输出任何内容 |

#### 输出规则

| 情况 | 结果 |
|------|------|
| 列表 5 张、`start_index=1`、`output_count=8` | 第 1~4 路输出第 2~5 张；第 5~8 路为空（返回 `None`，等同该路未接线） |
| `start_index` ≥ 列表长度 | 所有路为空（并打印一条 Warning） |
| `images` 未连接 / 列表为空 | 所有路为空，**不报错** |
| `output_count` 超出 1~16 | 自动钳制到 1 / 16 |

#### 关于"空路"与列表语义（重要）

- **空路返回 `None`，效果完全等同于"这根线没接"**：
  - 下游把该输入声明为 **optional** 时，它会走 Python 默认值，节点**照常执行**并按"这一路没有"处理
    （例如 `ImageBatchConcat` 透传另一路、`BatchImageReplace`/`MaskCompositeRef` 忽略它）——**不会连坐**其它输入；
  - 下游把该输入声明为 **required** 时，会在该下游节点报 `NoneType` 相关错误（等价于未连接时提交期的
    `Required input is missing`，只是报错从"提交"挪到"执行"，位置在下游节点上）。
- 实现上**不能**用空列表 `[]`（ComfyUI 的 `slice_dict` 会对空列表取 `[-1]` → IndexError），
  **也不要**用 `ExecutionBlocker(None)`：那会让下游节点"整块不执行"并**静默连坐**（下游的其它输入一起失效、
  再往下游全部停摆、界面毫无提示），与"这一路没接"的直觉不符。
- 节点使用 **列表语义**（`INPUT_IS_LIST`）：一次执行拿到整个列表，**元素尺寸可以互不相同**（普通批次要求批内尺寸一致）。
  上游如果给的是普通 IMAGE 批次，会自动拆成单张元素（batch → list）；列表元素若是 3D `[H,W,C]` 会自动补成 `[1,H,W,C]`。
- 每个输出口输出的是**单张图**（`[1,H,W,C]`），下游节点正常执行一次。
- 后端固定声明 16 路是为了通过 ComfyUI 的 prompt 类型校验，**显示的槽位数量由前端 `js/image_list_nodes.js` 真实增删**
  （`addOutput/removeOutput`），所以节点不会变成"一长条 16 路"。

---

### ImageListReplaceAt (图像列表位置替换)

用替换列表覆盖目标图像列表中从 `start_index` 开始的一段：第 `start_index + i` 个元素被替换列表第 `i` 个元素整体替换。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `start_index` | INT | 从列表第几个元素开始替换（0 = 第 1 个，1 = 第 2 个） |
| `images` | IMAGE | 要被替换的图像列表（可选）。未连接时不报错、也不输出任何内容 |
| `replace_images` | IMAGE | 用来替换的图像列表（可选）。未连接时原样透传目标列表 |

#### 替换规则

- 替换数量 = `min(替换列表长度, 目标列表长度 − start_index)` ⇒ **绝不超出原有列表长度**
  - 例：目标列表 5 张、`start_index=2`，替换列表 4 张 → 只替换第 3、4、5 张（第 4 张替换图被丢弃）
  - 例：目标列表 5 张、`start_index=2`，替换列表 2 张 → 替换第 3、4 张，第 5 张保持原样
- `start_index` ≥ 目标列表长度 → 打印警告并原样返回目标列表
- **不做任何缩放/补齐/扩展**：列表元素尺寸允许互不相同（这是列表语义与"批次替换"最大的区别）

#### 与 [BatchImageReplace](#batchimagereplace-图像批次替换) 的区别

| | ImageListReplaceAt | BatchImageReplace |
|---|---|---|
| 数据单位 | 列表元素（尺寸可不同） | 批次内同尺寸图像 |
| 替换长度 | 由替换列表长度决定，**不超出原长度** | `replace_count` 指定（-1 = 到末尾） |
| 数量不匹配 | 多余的替换图直接丢弃 | `overflow_mode` / `underflow_mode` 多种策略 |
| 尺寸不一致 | **不缩放**，逐元素整体替换 | 自动缩放到主批次尺寸 |
| 输出 | 列表（下游按元素逐个执行） | 单个批次 |

---

### ImageListToBatch (图像列表转批次)

把**尺寸各不相同**的图像列表转成同尺寸批次（供只接受批次的节点使用）。**不裁剪、不拉伸**：多出来的部分补在一个轴上（默认**纯黑**，可选 **alpha=0 透明**），每张图的完整内容都保留。

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `output_mode` | 枚举 | `RGB`（**默认**）= 3 通道，补边纯黑，**最兼容**（VL/视觉编码器、VAE、保存/预览都能吃）；`RGBA` = 4 通道，补边 `alpha=0`（真透明，适合后续合成，但**只吃 3 通道的下游会报错**，见下方"注意"） |
| `images` | IMAGE | 图像列表输入（可选，列表语义）。未连接时不报错、不输出 |

输出 `images`（批次 `[B, Rh, Rw, C]`，C = 3 或 4）。

#### 规则

1. **基准图** = 列表里 `min(w, h)` 最大的那张（＝「面积 × 方正度」最大，也就是"**又大又方**"）；
   并列时取更方正（长边更小）的那张。基准图的**比例**就是输出比例，**基准图的原始尺寸就是输出尺寸**。
   - 例：`[1024×1024, 1920×1080, 800×1200]` → 短边 1024 / **1080** / 800 → 基准 = `1920×1080`
   - 例：`[4000×1000（超宽条）, 1024×1024]` → 短边 1000 / **1024** → 基准 = `1024×1024`
     （超宽/超高长条图**不会**把画布撑到 4000 宽，批量内存与耗时可控）
2. 基准图**零重采样**（像素逐位不变；`RGBA` 模式下只多出 alpha 通道）。
3. 其余每张：比基准**更宽** → **只补上下**；比基准**更高** → **只补左右**（都是居中、两侧等分）；
   比例相同 → 不补边。受限的那一轴取精确尺寸，因此**不会出现"四面都补"**。
4. 补边按 `output_mode`：`RGB` → 纯黑（`0,0,0`）；`RGBA` → `alpha=0`、内容区 `alpha=1`
   （输入自带 alpha 则保留原 alpha），3 通道输入自动补 alpha。
5. 尺寸本来就匹配的图**完全跳过重采样**（不经过 numpy/PIL）。

#### 示例（基准 = 1920×1080，16:9）

| 输入 | 内容最终尺寸 | 补边（默认纯黑 / RGBA 模式为 alpha=0） |
|------|--------------|----------------------------------------|
| 1920×1080（基准） | 1920×1080 | 无（零重采样） |
| 1024×1024（更高） | 1080×1080 | 左右各 420 |
| 2000×1000（更宽） | 1920×960 | 上下各 60 |

#### 运行日志

每次执行会打印基准选择、输出尺寸与通道模式（选错基准一眼可见）：

```
[ImageListToBatch] 基准候选: #0 1920×1080 min=1080 | #1 1024×1024 min=1024 | #2 800×1200 min=800  -> 选定 #0
[ImageListToBatch] 输出尺寸 = 1920×1080（基准 #0，比例 16:9），共 3 张，通道 = RGB(3)
```

尺寸不是 8 的倍数时还会多打一条提示（VAE 下采样是 8/16，过 VAE 前需要对齐）：
`[ImageListToBatch] 提示：输出尺寸 876×1024 不是 8 的倍数，...`

#### 注意

- **为什么默认是 RGB**：4 通道（RGBA）会踩到一些"写死 3 通道"的下游。典型例子是 Qwen VL 视觉塔
  `Qwen35VisionPatchEmbed.forward` → `x.view(-1, 3, 2, 16, 16)`（`in_channels=3`）：收到 4 通道会让 patch 数多 1/3
  （1024×864 时 `8·H·W/1536 = 4608`，而位置编码数是 `6·H·W/1536 = 3456`），直接报
  `RuntimeError: The size of tensor a (4608) must match the size of tensor b (3456)`。
  所以喂 VL 文本编码器 / VAE / 大多数节点时用默认 `RGB` 最稳。
- 确实需要"透明补边"（后续合成、存 PNG 保留透明）时才切 `RGBA`；**此时若要再接 3 通道节点**，
  中间加一个核心节点 **`Split Image with Alpha`**：它的 IMAGE 输出就是 `i[:,:,:3]`（3 通道 ✓），
  MASK 输出是"补边区域=1"（`1 - alpha`，做合成/贴回正好能直接用）。
- 输出尺寸取自**基准图的原始尺寸**，因此可能不是 8/16 的倍数（例：`876×1024`）。要过 VAE 编码
  （下采样 8/16）时请先对齐到 8/16 的倍数——节点会在日志里提示这种情况。
- 与 [ImageBatchResize](#image-batch-resize-图像批次缩放) 的区别：那个是"缩放到**你指定**的尺寸（可选裁剪/拉伸）"，
  本节点是"**按列表自身**选基准 + 单轴补边、绝不丢内容"。
- 实现上**不是**"先补成大画布再整体缩放"，而是"先把内容缩到它在最终画布里的精确尺寸，再居中贴进画布"——
  数学等价，但补边不会被 LANCZOS 采样出内容边缘的暗边（halo）。

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
| 适用模型 | 通用 WAN 模型 | SCAIL-2 模型专用 |

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | MODEL | 待封装的 SCAIL-2 模型 |
| `context_length` | INT | 上下文窗口长度（像素帧）。建议设为 SCAIL-2 的 length 参数值（如 81） |
| `context_overlap` | INT | 窗口间重叠帧数（像素帧） |
| `context_schedule` | 枚举 | 窗口调度策略。`static_standard`（固定窗口，但末窗会回头对齐）；`standard_even`（等长、末窗不回退，重叠摊平）；`batched`（零重叠、末窗变短，配合锚定节点的 head/tail 只读上下文） |
| `context_stride` | INT | 窗口步进（仅对 uniform 策略有效） |
| `closed_loop` | BOOLEAN | 是否闭环窗口（仅 looped 策略有效） |
| `fuse_method` | 枚举 | 窗口融合方法（金字塔权重推荐） |
| `freenoise` | BOOLEAN | FreeNoise 噪声混洗，改善窗口间融合。⚠️ 2026-10 修复：覆写块改用**真实窗口起点**（旧实现假定步长均匀，在 `standard_even` 下错位）；等步长时逐位回归 |
| `fuse_sharpen` | FLOAT | 融合权重锐化（幂次），默认 1.0=不锐化。设 3~6 把重叠区交叉淡化带压窄 ⇒ 任一瞬间基本只取「更靠窗中心」的那一方，**既连续又不出现两个角色叠在一起的重影** |

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

### 步内滚动 (In-step Rolling)
> 配合上下文窗口使用：让**后面的窗口在同一步里就能看到前面窗口刚产出的内容**，减少缝上的重影/动作发软。

**接线**：`上下文窗口节点 → 步内滚动 → 采样器`

| 参数 | 默认 | 说明 |
|------|------|------|
| `enabled` | 开 | 关掉 = 等价于不装本节点 |
| `roll_beta` | 0.45 | 提交强度（主力旋钮）：0 = 透传，1 = 完全采用前窗当前步的结果；实测 0.45~0.50 最健康 |
| `roll_beta_inherited` | 0.6 | 只对「只读前文」帧单独生效的强度；-1 = 跟随 `roll_beta` |

> 已移除的控件（值固定）：`roll_scope=both`、`roll_beta_curve=late`。日志每个去噪步 1 行摘要，细节用 `--verbose` 查看。

### 锚定+步内滚动（极简）
> **只有一个开关的整合节点**：把「窗口锚定 + 步内滚动」按实测基线串起来，适合不想调参直接出片的情况。

**接线**：`上下文窗口节点 → 锚定+步内滚动（极简） → 采样器`；可选输入 `anchor_frames` / `anchor_mask_frames` / `vae`（与窗口锚定节点相同）。

| 设置项 | 值 | 说明 |
|--------|----|------|
| `enabled` | 开 | **唯一的设置项**；关掉 = 两功能都关（并清空锚定池） |

固定档写在 `context_anchor_rolling.py` 顶部的 `FIXED_ANCHOR` / `FIXED_ROLL`（只读前文 2、记忆组 4、自动间距、提交强度 0.45、继承帧 0.6、`roll_scope=both`、`curve=late`、`anchor_slot=aligned`、`drive_memory=off`、`anchor_scope=fresh_only` …）。需要调这些量时请改用上面两个「老」节点（它们保留全部控件）。



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

## 更新日志

### v1.9.0（2026-10-16）

**新增**
- **Detect Edges (Canny) Chunked**（`CannyChunked`，类别 `image`）：分块版 Canny 边缘检测，
  修掉官方 `Detect Edges (Canny)` 在 200+ 帧批次上「不报错、无进度、彻底卡死」的问题。

  根因（实测）：官方把**整批**一次性交给 `kornia.filters.canny`，而 kornia 的迟滞阶段是一个
  Python 级 `while` 循环（每轮只把强边向外扩张 1 个像素，512² 约 96 轮、1024² 约 190 轮），
  循环里每轮的 `F.conv2d(edges, hysteresis_kernels)` 输出是 `(B, 8, H, W)`（输入体积的 8 倍），
  且循环条件是**整批共享**的（只要有 1 帧没收敛，200 帧一起再赔一轮）⇒ 200 帧 1024² 的瞬时峰值
  约 29.7GB，超过 24GB 显卡后 Windows WDDM 静默溢出到系统内存 ⇒ 表现为卡死，且节点执行中无法取消。

  本节点逐块调用**与官方完全相同的那一行**（只把「整批」换成「逐块」）：

  | 场景（RTX 4090，真实帧） | 官方 Canny | CannyChunked |
  |---|---|---|
  | 200 帧 1024² 峰值显存 | ~29.7GB（超 24GB → spill 卡死） | **1.31GB** |
  | 200 帧 1024² 耗时 | 卡死（数十分钟无响应） | **约 5.7s** |
  | 中断 | 节点内无法取消 | 每块检查，可随时取消 |
  | 进度 | 无 | 前端进度条 + 控制台 tqdm |

  参数与一致性：`low_threshold` / `high_threshold` 与官方同名同义（可直接替换节点）；单块
  （`chunk_size` ≥ 帧数）与官方 **bit-exact**；真实图像分块实测 **0 像素差异**（256²/512²/1024²）；
  CPU 通路任意分块 bit-exact。`auto_chunk` 依 `model_management.get_free_memory()` 自动收缩块大小
  （只减不增），`use_gpu` 关掉即走 CPU（零显存，用于显存被大模型占满时）。
  一致性口径提示：CUDA 卷积会按 batch 尺寸选择 cuDNN 算法，在「均匀随机噪声」这类最坏输入下
  分块与整批最多约 0.003% 像素翻转（**官方自己分批跑也会同样偏离**，非本节点引入）。
  自测脚本 `md/canny_chunked_selftest.py`（40 项断言）可脱离 ComfyUI 直接运行。

#### 同版其它新增（2026-10-15 起）
- **锚定 + 步内滚动（极简）**：面板**只有一个开关**的一键节点，把「窗口锚定 + 步内滚动」按实测基线串起来（只读前文 2 / 记忆组 4 / 提交强度 0.45 / 继承帧 0.6 / `curve=late`…）。设置项集中在 `context_anchor_rolling.py` 顶部的 `FIXED_ANCHOR` / `FIXED_ROLL`，改那里即改行为。
- **步内滚动**（`Wan SCAIL-2 Rolling Windows`）：同一步内让后面的窗口看到前面窗口**本步**刚产出的结果，减少缝上重影/动作发软。

**精简（面板控件）**
- 窗口锚定 **8 → 5**：移除 `anchor_scope` / `drive_memory` / `anchor_slot`（值固定为实测档 `fresh_only` / `off` / `aligned`）。
  ⚠️ **兼容性提示**：`anchor_slot` 的默认值由 `context` 改为 **`aligned`**（与「已移除控件」的原默认一致）；老工作流请重存一次节点，或自行核对这几个值。
- 步内滚动 **5 → 3**：移除 `roll_scope` / `roll_beta_curve`（值固定 `both` / `late`）。

**其它**
- 滚动提交公式回退为「再噪」式（此前试过的 `content_delta` 实测更差，已删除）：
  `x = (1-β)·x + β·[(1-σ)·前窗本步 x0 + σ·ε]`（ε 由种子确定 ⇒ 两窗共享同一份噪声）。
- **日志降噪**：每步/每窗触发的日志（如 `SCAILContext sam_latents`、`窗口边界`、`FreeNoise`…）降为 `debug` 档（实测省约 12% 日志行）；需要时用 `--verbose` 启动 ComfyUI 即可看全量。`WanUni3C` 的逐步统计改为「首次 info + 之后 debug」。
- 日志/打印文本 **GBK 安全化**（避免 cp936 控制台抛 `UnicodeEncodeError`）；`print` 全部改为 `logging.*`；`FolderImageLoader` 的逐文件告警改为汇总一行。
- 修正「末窗重叠」告警文案（在 `standard_even` 下不再自相矛盾地建议改用 `standard_even`，改为建议缩短 `context_length` / 用 `batched`）。

#### v1.8.1 内容（曾未单独发布，并入本版）

**修复**
- **`anchor_fixed` 之前完全失效（真 bug）**：锚帧「上一步 x0 预测」的缓存被挂在**窗口对象**上，而窗口对象**每个采样步都会重建**（`Custom_context.get_context_windows` 每步 new 一批）⇒ 缓存每步丢失，代码永远退回「干净锚帧」这一条路。后果有两个：
  ① `anchor_fixed` 开/关**逐位相同**（完全看不出效果，用户实测确认）；
  ② 采样后期（σ→0）锚帧 ≈ 一张**干净的驱动者画面**，被模型顺着续写 ⇒ **「驱动者形象伪影」**（即 v1.8.0 里声称「已修」的那个问题，实际并未修掉）。
  现在缓存改挂在 **handler** 上（同一次采样内共享，跨步存活），并优先取 CFG 正样本那一路的输出；新增自测用例覆盖「窗口重建后仍能读到上一步缓存」。
- **RoPE 位置补丁可能静默失效**：该补丁是进程级 monkey patch（改 `SCAILWanModel.rope_encode` 类属性），若 comfy 核心被热重载冲掉补丁、而 `_SCAIL_ANCHOR_ROPE_PATCH_APPLIED` 标记仍是 `True` ⇒ 补丁**再也不会重装**，且**不报任何错**，导致额外 token 被当成「窗口之后的未来驱动」（位置错、效果变差）。现在补丁带**自愈**：每次应用前检查「当前函数是不是本补丁」，不是就重装，并在日志里明确写「已应用 / 已在位 / 被覆盖，已重新安装」。
- **GBK 安全**：`context_window_anchor.py` 有 2 处**日志文本**含 `⇒`(U+21D2) 与 `−`(U+2212)，在 cp936 控制台下会直接抛 `UnicodeEncodeError`（项目里记录过的坑）。已统一替换为 ASCII 等价写法，自测脚本也一并安全化。

**变更**
- **移除 `tail_context_frames`（尾段只读上下文）**：机制上不产生可见效果 —— 尾段的预测在融合前会被裁掉，且那些帧的「作者」是后面的窗口（看一眼改不了结果）。内部能力（`halo_tail` / `tail_extra_indices` / 融合前裁剪）保留但恒为 0。
- **新增 `memory_stride`（`0` = 自动）**，并**占用原 `tail_context_frames` 的参数槽位**（避免老工作流 widget 值整体错位；该位置原本几乎总是 0，而 0 恰好 = 自动，语义兼容）。
  `0` = 让 `memory_frames` 个点覆盖「片段开头 → 本窗口之前」的**全部历史**（解决长片段 / 靠后窗口「够不到开头」的问题）；`>0` = 固定间距（`4` ≈ 每 16 像素帧取 1 帧）。
- **`memory_frames` 与 `head_context_frames` 解耦**：记忆组的取点基准改用「窗口自身的起点」，不再因为开了头段而被一起往前推（两个开关不再互相干扰）。
- **文档口径以代码为准**：修正「`c_concat` 填 1」等**属于已撤销实验的过时描述**；新增「帧单位」说明（`anchor_frames` = 像素帧；`head_context_frames` / `memory_frames` / `memory_stride` = latent 帧）。

**自测**
- `md/context_window_anchor_selftest.py`：**19 项全过**（新增/改写：x0 缓存跨步存活、记忆组自动间距与固定间距、记忆组与头段解耦、RoPE 补丁热重载自愈、接口含 `memory_stride` 且不含 `tail_context_frames`、**`memory_position`（tail/head + 驱动流/掩码同步）**、**RoPE 补丁归属与每步自愈 + 每步从 cfg 刷新全局（免疫节点缓存）**、**驱动流/驱动遮罩 token 数一致性自检**）；脚本已 GBK 安全化，可直接在本机 cp936 控制台运行。

### 2026-10-10 修复与新增（按"静默故障"优先）

| 项 | 内容 |
|---|---|
| **RoPE 补丁争抢** | `SCAILWanModel.rope_encode` 被 anchor 与 `WanSCAIL_MultiRef` **同时抢**（都是直接赋值 ⇒ 谁最后打谁赢）。后果：anchor 补丁被顶掉时，额外 token 被当成「窗口之后的未来驱动」⇒ **位置错、无报错**（且是否发生取决于节点执行顺序/缓存命中 ⇒ **间歇性**）。现在：① MultiRef 检测到 anchor 补丁在位就**跳过 SCAIL 级覆盖**；② anchor 侧**每个采样步检查并重装**；③ 告警里直接打印**"被顶掉时的所有者"**（一眼看出凶手） |
| **参数"全局 + 节点缓存"串味** | 4 个开关（`anchor_scope`/`shared_head`/`overlap_scope`/`memory_position`）原先**只存在模块全局**；节点被 ComfyUI 缓存时 `reset_pools` 不执行 ⇒ 采样期读到上一次运行的残留值。现在：这 4 个字段进 `AnchorConfig`，**每个采样步开头从 cfg 刷新全局**（池源变了就清池缓存） |
| **`memory_position` 默认改 `head`** | `tail` 摆放只有在 RoPE 补丁生效时才正确；`head` 对"补丁失效"**免疫**（最多被当成相邻帧）。补丁正常时两者**逐位相同** |
| **驱动流/驱动遮罩一致性自检** | SCAIL 的驱动遮罩是**逐 token** 加进驱动流的 ⇒ 必须同帧数；现在不一致会**告警**（以前是静默错位） |
| **FreeNoise 的 stride 假设** | 覆写块改用**真实窗口起点**（旧实现假定步长均匀 ⇒ 在 `standard_even` 下与真实窗口边界不重合 ⇒ 越往后噪声来源越碎） |
| **掩码日志去重键** | `sam_latents` / `ref_mask_latents` 两路（同为 `t_dim=2`）此前会**互相吞日志** ⇒ 排查必然漏看一路；现在 key 里带"哪一路流" |
| **`roll_beta_inherited`**（滚动节点）| 按「帧的归属」分档：**只把"只读前文"单作者化、本窗拥有的帧一字不动**（默认 -1 = 跟随 `roll_beta` = 逐位回归）。用于压低"后两缝"的窗间分歧 |
| ~~**`roll_w_content` / `roll_w_noise`**~~ **已删除（2026-10-11 精简）** | 实测：`w_noise<1` 把缝上瞬移从 11.4px 恶化到 31~40px 并脱地（接地面比 0.47）；`w_content` 无单调效果 ⇒ 两者删除，提交公式固定为「legacy 再噪」，界面参数 11 -> 5 |
| ~~**`roll_noise_seed`**~~ **已隐藏（2026-10-11）** | ε（提交帧噪声种子）**内部固定 0**：统计上任意值等价，而实测换 ε 会让缝上瞬移在 11.4~23.3px 之间变化（抽签）⇒ 不再暴露。**同时精简锚定节点**：删掉 `anchor_fixed`（节点自注「开/关完全等价，用户实测确认过」）与 `shared_head`（小杠杆），锚帧占位噪声种子默认 0x5EED -> 0 |
| **锚定节点精简（2026-10-11）** | 删掉 4 个控件、值锁定为实测最优/你的习惯：`driving_only` → **False**（你实测 True 效果差；锚帧前插并逐帧配对）、`anchor_image_blank` → **True**（锚帧在待生成序列里只放噪声占位，避免源人物画面泄漏；同 seed 对照 19.9 → 15.7px）、`anchor_fixed`（节点自注「开/关完全等价，用户实测确认过」）、`shared_head`（文档记为「小杠杆」）| 锚定控件 12 → **8** |
| **〔已回退，见上方「顺序规范化回退」〕锚定顺序规范化（2026-10-11）** | 统一为 **`[锚][记忆][本窗]`**，且**条件流 / 掩码 / RoPE 分块三处同序**（自测断言守护）：锚帧 = 「紧邻窗口之前」的 `pre` 块；记忆 = 「更早、按 stride」的 `mem` 块。`memory_position` 控件删除（固定 head）；`prehistory` 改为 = 锚帧数 K（此前是 0 ⇒ RoPE 会把锚帧当成「序列开头」，是**静默错位**风险）| `context_window_anchor.py` |
| **B + `x_prefix`（2026-10-11）** | ① **B**：`_match_channels` 补齐 **20→16 降通道** ⇒ `pose_video_latent`（16ch）此前**永远拿不到记忆帧**（20ch 记忆转 16ch 被丢弃）；现在两条 pose 流结构一致 ✓ ② **`x_prefix`**（新控件，默认 `anchor_only` = 零回归）：`anchor_memory` ⇒ x 也镜像驱动流 `[锚][记忆空白][本窗]`（掩码同步补 0）；`none` ⇒ x 不加前缀。⚠️ 已知限制：x 侧额外 token 仍走基类 RoPE 的「顺序编号」⇒ 若本项有效，再做 x 侧时间码补丁 | `context_window_anchor.py` |
| **`x_prefix` 崩溃修复（2026-10-11）** | `anchor_memory` / `none` 一跑就崩（`Sizes of tensors must match... Expected 25 but got 21`）：**与 x 逐帧对齐的流**（`concat_mask` / `denoise_mask` / **`c_concat`** / 1·3·4ch 掩码 / 16ch x 池流）仍在用 `_ANCHOR_COUNT`，没跟随 `x_prefix`。修法：新增唯一事实来源 **`_x_side_prefix_len(window)`**（none=>0 / anchor_only=>K / anchor_memory=>K+M）⇒ 上述流全部改用它（含 `prepend_anchor_mask` 里 ref 掩码那路）；并新增**跨流一致性运行时告警**（x 侧实际额外帧 ≠ 应为值 ⇒ 立刻告警，不再等到 comfy 崩）| `context_window_anchor.py` |
| **新档安全门 + 观测修复（2026-10-11）** | ① `shifted`/`aligned` 的 x 段拆分现在**透传 `steps_h`/`steps_w`**（此前只有窗口段传，锚/记忆段用默认网格 ⇒ 网格不一致）；② 档位因「调用方传了 `steps_t`」或「本窗无锚帧」而**退回 `context`** 时**打显式日志**（此前静默 ⇒ 实验会白跑且看不出）| `context_window_anchor.py` |
| **裁剪偏移漏算 x 记忆空白（2026-10-11 修复）** | `x_prefix=anchor_memory` 时 x 侧多插 M 个记忆空白槽，而融合前裁剪只算「前缀+锚+头段」⇒ 每窗输出**错位 M 帧**（症状：动作错乱/反复动作；日志 `窗口N 裁剪后长度 21 != 窗口帧数 17`）。修法：新增唯一来源 `Custom_context._x_prefix_mem_slots(window)`（复用 `_x_memory_slots`）并纳入 `front_trim`；同时修好分歧审计的位置映射（`base = 锚 + M`），重影审计恢复可用。回归用例24 | `Custom_context.py`、`context_window_anchor.py`、`md/context_window_anchor_selftest.py` |
| **FreeNoise 实现（2026-10-11 审计：代码级事实）** | 与 comfy 官方 `apply_freenoise` 同语义（噪声池 / randperm / 覆写位置一致），另修官方两处：①改用**真实窗口起点**（官方假定均匀步长 ⇒ `even_standard` 这类非均匀起点下覆写块错位；selftest 用例7 守护）；②`dynamic_seed` 固定 = `seed`（官方残留的 `seed + _step*997` 在原 wrapper 里 `_step` 恒 0，无效）。**作用范围：每次 `sample()` 只对初始噪声置换一次**（wrapper 挂在 `SAMPLER_SAMPLE`，非每去噪步）⇒ 对缝合处的单帧跳变作用天然有限。已知限制：**不支持多模态 noise**（官方会按模态缩放 C/OV）；wrapper key 与官方同名（两者同时存在时只有后装者生效，且无告警）| `Custom_context.py`、`md/context_windows_schedule_selftest.py` |
| **`anchor_slot`（2026-10-11 新增控件）** | 两档锚帧时间码语义：`context`（默认）= 物理 `[记忆][锚][本窗]` + `prehistory=0`；`leading` = 物理 `[锚][记忆][本窗]` + `prehistory=K`，并在**同一个 RoPE 函数内**把 x 段拆成 `[x锚][x记忆][x窗]`（锚 = `ref−K..ref−1`、记忆按 stride、窗本体 = `ref..`）。不改 `WanModel` 基类 ⇒ 对其他模型无影响。切换后重跑即可、无需重启。自测：用例3（控件顺序/默认值）、用例13b（`prehistory=K`、总 token 数不变、base 调用 ≥5）。❓**待测**：`leading` vs `context` 的同 seed 对比（各 2 个种子）尚无数据 | `context_window_anchor.py`、`md/context_window_anchor_selftest.py` |
| **`drive_memory` 控件（2026-10-11）** | 16ch 驱动流（`pose_video_latent`）是否接收记忆帧：默认 `off` = 旧行为（不把 20ch 裁成 16ch ⇒ 该流不接收记忆帧）/ `on` = 补齐记忆帧。切换后重跑即可、无需重启；环境变量 `CWA_DRIVE_MEMORY=0` 为兜底。**实测（各 1 段，幅度不可信、方向暂取 off）**：坏种子 off 23.9 / on 28.8；好种子 on 17.3 / off 18.0。同时把 `x_prefix=none` 标为「实验/不推荐」| `context_window_anchor.py`、`md/context_window_anchor_selftest.py` |
| **`x_prefix` 时序 bug（2026-10-11 修复）** | `anchor_memory` 崩溃根因（代码级）：条件流（`c_concat` 等）**先**注入、x 侧（`inject_anchor_x`）**后**注入，两边各自算「记忆空白槽数」⇒ 前者读到尚未写入的 `window._x_prefix_mem_count`=0 ⇒ 只 +K，后者 +K+M ⇒ 差 M=4 ⇒ comfy 崩（`Expected size 25 but got size 21`）。修法：唯一来源 **`_x_memory_slots(window)`**（按 `original_indices` 现算、与调用顺序无关），四处统一调用；日志改为打印实际插入帧数。回归用例23 | `context_window_anchor.py`、`md/context_window_anchor_selftest.py` |
| **顺序规范化回退（2026-10-11）** | 现象（实测，每档 1 段）：物理顺序统一为 `[锚][记忆][本窗]` + `prehistory=K` 后，max 由 12.4~18.1（中位 13.8，旧）变为 **22.4px**；FreeNoise 开时 11.4 → 17.4。**已回退**为 `[记忆][锚][本窗]` + `prehistory=0`。⚠️「相位错开 K 是主因」为**推断，未验证**；若再改该顺序，须同时同步 x 段（见 `anchor_slot`）| `context_window_anchor.py` |
| **`md/frame_metrics.py`**（本地工具，`md/` 已 gitignore）| 帧级客观指标：**瞬移**（max px / pop比 / 落点作者带 / 接地面跨越比）+ **残差**（重影）+ **逐带**（锐度/饱和）；用**内容哈希**识别「同一段被多个 preview 节点重复保存」并自动对齐偏移 ⇒ `--seedtab` 一条命令出同 seed 参数对照表 |

### 勘误与「已撤销的实验」（勿把历史记录当缺陷重新引入）

本节点经历过多次「做出来又撤销」的实验，文档里因此留有互相矛盾的记录。以下几条**都是已撤销的**：

| 历史记录 | 状态 | 为什么撤销 |
|---|---|---|
| `c_concat` 在锚帧上填 **1**（当「干净/已知帧」） | ❌ 已撤销 | 属于「替换式 / 钉住式锚定」时代。锚帧来自**驱动视频**，把它当已知干净帧钉住 ⇒ 驱动者形象被直接带进画面。当前语义是「锚帧当普通待生成帧」（`concat_mask`=1 / `c_concat`=0），且 `c_concat` 本来就是模型从 `1 - mask` **自动派生**的 |
| 替换式锚定（用窗口 0 首帧替换各窗首帧）与 `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug` | ❌ 已移除 | 替换会破坏「逐帧配对」几何、引入合成中间态；接口需保持极简 |
| 尾段只读上下文（`tail_context_frames`） | ❌ 已移除 | 预测被裁掉 + 那些帧归后面的窗口 ⇒ 看不到可见效果 |
| 固定 4 latent 间距的「长程记忆」 | 🔄 改为默认自动间距 | 固定间距在长片段上覆盖不到开头，失去「长程」意义（固定间距仍可用：`memory_stride > 0`） |

### v1.8.0

**新增**
- `WanAnimate2ToVideoCustom`：WanAnimate2（官方二代架构）视频生成节点
- `Context Window Anchor (上下文窗口锚定)`（**实验性 / 测试性——有效性与副作用尚未充分验证**）：**极简接口**——required = `model` / `enabled`；optional = `anchor_frames`(IMAGE) / `anchor_mask_frames`(IMAGE, 仅 SCAIL-2) / `vae`(VAE，接锚帧时必接)。把锚帧当作**已知帧（干净历史）**同时拼进驱动流、驱动遮罩与待生成序列（插在参考前缀之后，采样后裁掉、不参与融合），**给多少用多少**；不接锚帧即等价于不装本节点。必须接在上下文窗口节点之后。
  - **锚帧语义（本轮修正）**：锚帧按「手动把整段视频开头拼到片段开头」的分布注入——驱动流/驱动遮罩前插的同时，**待生成序列前插的锚帧走它自己的生成轨迹**（第 1 步 σ=1 纯噪声；之后用上一步模型对这几帧预测的 x0 重新加噪），并当**普通帧**（输入掩码=1 待生成、history mask=0）。**已移除** `mode`／替换式等分支，节点只有一个行为。
  - **本轮又修**：① 运行期标记 gating（绕过/禁用锚定节点 ⇒ 真正零注入；修掉「上一次运行残留配置污染旁路运行」导致的 x(25) vs ref_mask(27) 崩）；② 新增 `driving_only`（默认开）与 `head_context_frames`（默认 0）。
  - **本轮再修**：① 由此去掉了「把**驱动者画面**钉在窗口头部」导致的**驱动者形象伪影**（此前 σ→0 时锚帧几乎就是一张干净的驱动者画面，模型顺着它续写）；② **掩码编码支持任意帧数**（此前 4/8 帧必崩：`_extract_mask_to_28ch` 的时间分组只对 1+4n 成立，7×7=49 ≠ 28）；③ 新增**采样前编码自检**（x/pose/mask 三池试编）与「SCAIL-2 必须同时接 `anchor_mask_frames`」的明确报错。
  - **~~关键修复：`c_concat` 填 1~~（已作废 —— 见文末「勘误与已撤销的实验」）**：该条属于**已撤销的「替换式 / 钉住式锚定」**语义。当前实现是「锚帧当普通待生成帧」：`concat_mask` / `denoise_mask` = **1**、`c_concat` = **0**；而且模型侧 `c_concat` 本来就由 `comfy/model_base.py::WAN21_SCAIL2.concat_cond` 从 `1 - mask` **自动派生**，插件只需按窗口帧数同步插帧。**移除**：替换式锚定与 `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug`。
  - **配套修复**：`Custom_context.py` 新增 `cond_indices_for_window()`（按 cond 与 x 的长度关系自动对齐坐标系，`cond_len == x_len - 1` 时按参考槽偏移 1 帧，与模型侧 `rope_encode_pose(t_start=1)` 一致），修掉了 animate2 带前缀窗口时 pose 切片可能整体偏 1 帧的问题；`Wan SCAIL Context Windows` 中原本固定取 chunk 前 N 帧的 `pose_video_latent` 切片改为按窗口真实帧索引（含锚定/延伸段）并带越界告警；`combine_context_window_results` 的融合前裁剪扩展为「前缀 + 锚定 + 头段 / 尾段」。
  - **未充分验证**：有效性取决于「对齐基准是否由片段开头锁定」这一假设，副作用（画质 / 人物一致性 / 采样稳定性 / 帧数预算）尚未系统评估；请按上文「实验建议」做开/关对比后再决定是否长期使用，不建议在正式生产工作流中依赖它。
  - **未充分验证**：有效性取决于「对齐基准是否由片段开头锁定」这一假设，副作用（画质 / 人物一致性 / 采样稳定性 / 帧数预算）尚未系统评估；请按上文「实验建议」做开/关对比后再决定是否长期使用，不建议在正式生产工作流中依赖它。

**移除**
- `WanSCAILSparseAttention`：长序列下需要构造 O(S²) 的稠密注意力掩码（2600 帧时可达 TB 级），导致内存直接爆掉，且实测在 SCAIL 上无明显收益，故彻底移除。旧工作流若引用它会提示"缺失节点"，删除该节点即可

**修复**
- `SDPoseEstimateYawSimple` / `SDPoseEstimateYawAdvanced`：短片段、多镜头素材下分组为空会触发 `IndexError` 崩溃（现改为回退到全局基准值）
- `BatchImageReplace`：修正 `repeat_end` 填充逻辑（3 张替换 4 张由 `0,2,2,2` 修正为 `0,1,2,2`）；`images` 未连线时不再抛 `TypeError`
- `ImageSequenceToVideo`：ffconcat 列表与音频临时文件在失败/异常路径也会被清理

**优化**
- `InteractiveBatchCrop`：预览缓存改为"按前端取帧规则缓存帧子集"，2600 帧从约 12.5GB 降到约 48MB，并加了 2GB 总量预算；输出改为预分配
- `SmartVideoCrop`：裁剪改为 8 帧分块批处理（长序列更快，裁剪结果数值不变）
- `MaskCompositeRef` / `BatchImageReplace` / `ImageBatchResize` 等：逐帧拼接改为预分配输出（峰值内存约减半）
- `VideoGridCompose` / `VideoFrameCounter` / `ImageSequenceToVideo`：同一文件的 ffmpeg 探测由 2 次降为 1 次
- `GD_BBoxDetect`：新增 `unload_after_run` 开关（默认关闭＝保持常驻显存），需要时可在检测完把模型移回 CPU
- 文件/文件夹选择弹窗改为工作线程执行，弹窗期间不再阻塞 ComfyUI 事件循环

**⚠️ 不兼容变更（升级前请留意）**

1. `WanUni3CApply`：输入由 `render_latent`(LATENT) 改为 `render_video`(IMAGE) + `vae`；`WanUni3CLoader` 的模型目录由 `models/controlnet` 改为 `models/model_patches`
2. `WanSCAILToVideoMultiRef`：`ref_encoding_mode` 默认值改为 `逐帧编码`（保真度更高；如需旧行为请手动改回 `1+4n批量`）
3. `Custom Context Windows (Manual)` / `Wan SCAIL-2 Context Windows`：已移除 `causal_window_fix` 相关参数与逻辑
4. `WanSCAILSparseAttention` 节点已移除（见上文"移除"）

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
- ComfyUI-Scail2-Sampler-Helper ([checknickname](https://github.com/checknickname)) — SCAIL-2 两阶段采样采样器；借此致谢

同时，本项目中的 WanAnimate 多参考图视频生成逻辑、偏航角估计算法、上下文窗口调度策略、骨骼绘制引擎、SCAIL/SCAIL-2 多参考图支持等核心逻辑均为自主实现。

---

## 许可证

MIT License
