# ComfyUI-CustomNodeKit 节点索引

> 此文件仅供本地参考，不上传 GitHub（已在 .gitignore 中排除）

---

## merged_nodes.py

| 类名 | 显示名 |
|------|--------|
| `PathCollectorNode` | 路径收集器 |
| `IndexSelectorNode` | 索引选择器 |
| `PathValidatorNode` | 路径验证器 |
| `FolderImageLoaderNode` | 文件夹图像载入器 |
| `ImageBatchConcatNode` | 图像批次合并 |
| `ImageBatchResizeNode` | 图像批次缩放 |
| `SingleFrameVAEEncode` | 逐帧独立 VAE 编码 |
| `MaskCompositeBatch` | 遮罩批次合成 |

---

## custom_video_nodes.py

| 类名 | 显示名 |
|------|--------|
| `VideoFrameCounter` | 视频帧数统计 |
| `ImageSequenceToVideo` | 图片序列合成视频 |

---

## custom_sdpose_nodes.py（含 sdpose_yaw.py）

| 类名 | 显示名 | 来源 |
|------|--------|------|
| `SDPoseDrawKeypointsV2` | Draw SDPose Keypoints (V2) | custom_sdpose_nodes.py |
| `SDPoseSaveJson` | Save SDPose Keypoints as JSON | custom_sdpose_nodes.py |
| `SDPoseLoadJson` | Load SDPose JSON | custom_sdpose_nodes.py |
| `SDPoseSliceKeypoints` | Slice SDPose Keypoints | custom_sdpose_nodes.py |
| `SDPoseConcatKeypoints` | Concat SDPose Keypoints | custom_sdpose_nodes.py |
| `SDPoseResizeKeypoints` | Resize SDPose Keypoints | custom_sdpose_nodes.py |
| `SDPoseResampleKeypoints` | Resample SDPose Keypoints | custom_sdpose_nodes.py |
| `SDPoseEstimateYawSimple` | Estimate Yaw (Simple) | sdpose_yaw.py → merged into custom_sdpose_nodes.py |
| `SDPoseEstimateYawAdvanced` | Estimate Yaw (Advanced) | sdpose_yaw.py → merged into custom_sdpose_nodes.py |

---

## WanLoop_FullNodes.py

| 类名 | 显示名 |
|------|--------|
| `WanAnimateToVideoCustom` | WanAnimate To Video (自定义) |

---

## interactive_batch_crop.py

| 类名 | 显示名 |
|------|--------|
| `InteractiveBatchCrop` | 交互式批量裁剪 |

---

## Custom_context.py

| 类名 | 显示名 |
|------|--------|
| `WanContextWindowsManual_Custom` | WAN Context Windows (Manual Custom) |

---

## reference_image_selector.py

| 类名 | 显示名 |
|------|--------|
| `ReferenceImageSelector` | Reference Image Selector (参考图选择器) |

---

## integer_setting_node.py

| 类名 | 显示名 |
|------|--------|
| `IntegerSettingNode` | Integer Setting (整数设置) |

---

## wan_uni3c_camera.py

| 类名 | 显示名 |
|------|--------|
| `WanUni3CLoader` | WanVideo Uni3C ControlNet Loader (Custom) |
| `WanUni3CApply` | WanVideo Uni3C Apply (for KSampler) |

---

## grounding_dino_nodes.py

| 类名 | 显示名 |
|------|--------|
| `GD_ModelLoader` | Load GroundingDINO Model |
| `GD_BBoxDetect` | GD BBox Detect |

---

## wan_scail_multi_ref.py

| 类名 | 显示名 |
|------|--------|
| `WanSCAILToVideoMultiRef` | Wan SCAIL To Video (Multi Ref) |
| `SCAIL2ColoredMaskMultiRef` | Create SCAIL-2 Colored Mask (Multi Ref) |

---

## wan_scail_context.py

| 类名 | 显示名 |
|------|--------|
| `WanSCAILContextWindows` | Wan SCAIL Context Windows |

---

## wan_sparse_attn.py

| 类名 | 显示名 |
|------|--------|
| `WanSCAILSparseAttention` | Wan SCAIL Sparse Attention |