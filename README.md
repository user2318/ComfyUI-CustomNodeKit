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
  - [上下文工具](#上下文工具)
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
| **Load SDPose JSON** | `SDPose` | 加载 JSON 姿态文件，支持按目标帧率自动抽帧/补帧（线性插值或复制） |
| **Slice SDPose Keypoints** | `SDPose` | 对姿态序列进行时间切片（按起始帧与帧数截取） |
| **Concat SDPose Keypoints** | `SDPose` | 将两段姿态序列前后拼接 |
| **Estimate Yaw (Simple)** | `SDPose` | 简化版偏航角估计，从姿态关键点推算人物朝向角度（核心参数：置信度阈值、是否解缠、肩部权重） |
| **Estimate Yaw (Advanced)** | `SDPose` | 完整版偏航角估计，可调节所有底层参数（平滑窗口、EMA alpha、角度限制、侧向校准等），并输出详细调试表格 |
| **Resize SDPose Keypoints** | `SDPose` | 缩放姿态关键点坐标并更新画布尺寸，支持保持宽高比、智能裁剪（基于关键点包围盒） |
| **Resample SDPose Keypoints** | `SDPose` | 对姿态关键点序列进行帧率重采样（抽帧/补帧），独立于 JSON 载入逻辑 |

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
| **Index Selector** | `Utility` | 根据索引从列表中选取元素 |
| **Path Validator** | `Utility` | 验证文件路径是否存在 |
| **Integer Aligner** | `Utility` | 将整数对齐到目标步长的倍数 |

### 上下文工具

| 节点名 | 类别 | 说明 |
|--------|------|------|
| **WAN Context Windows (Manual)** | `context` | 针对 WAN-Animate 视频模型特化的上下文窗口节点。将长视频拆分为滑动窗口逐段采样，支持参考帧前缀（prefix_frames）、FreeNoise 噪声混洗、多种调度策略（均匀/静态/循环/批处理）与融合模式（金字塔/线性重叠/相对加权），解决显存限制下的长视频生成问题 |

---

## 依赖

核心依赖（已在 `requirements.txt` 中声明）：

- `torch` — PyTorch 深度学习框架
- `numpy` — 数值计算
- `colorsys` — 颜色空间转换（Python 标准库）
- `tqdm` — 进度条显示
- `opencv-python` (可选) — 用于面部连线绘制（无 OpenCV 时自动回退到内置绘制器）
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

---

## 许可证

MIT License
