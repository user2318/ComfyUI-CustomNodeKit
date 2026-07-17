# ComfyUI Custom Node Kit

A set of custom nodes for ComfyUI, covering video generation, pose processing, interactive image operations, and common workflow utilities.

Current version: **1.7.1** — Merged batch image operation nodes (BatchImageReplace + BatchFrameReplicate moved into merged_nodes.py), new standalone two-phase sampling node (WanSCAIL2PhaseSampler), IntegerSettingNode frontend refactoring with alignment optimization, SCAIL-2 workflow full rewrite, color drift correction V4.1 fixes and enhancements.

## Table of Contents

- [Installation](#installation)
- [Node List](#node-list)
  - [WanLoop Video Generation](#wanloop-video-generation)
  - [SCAIL/SCAIL-2 Multi-Ref Generation](#scailscail-2-multi-ref-generation)
  - [SDPose Pose System](#sdpose-pose-system)
  - [Video Tools](#video-tools)
  - [Interactive Tools](#interactive-tools)
  - [Utility Tools](#utility-tools)
  - [Image Batch Tools](#image-batch-tools)
  - [Context Tools](#context-tools)
- [Complex Node Details](#complex-node-details)
  - [WanAnimateToVideoCustom](#wananimatetovideocustom)
  - [Custom Context Windows (Manual)](#custom-context-windows-manual)
  - [Reference Image Selector](#reference-image-selector)
  - [SDPose Empty Frame Repair](#sdpose-empty-frame-repair)
  - [SDPoseDrawKeypointsV2 Foot Mode](#sdposedrawkeypointsv2-foot-mode)
  - [Folder Image Loader](#folder-image-loader)
  - [Image Batch Concat](#image-batch-concat)
  - [Image Batch Resize](#image-batch-resize)
  - [WanAnimate Uni3C Camera Control](#wananimate-uni3c-camera-control)
- [Dependencies](#dependencies)
- [Workflows](#workflows)
  - [Long Video Pose Detection Workflow](#long-video-pose-detection-workflow)
  - [WanAnimate Multi-Ref Long Video Workflow](#wananimate-multi-ref-long-video-workflow)
  - [WanSCAIL2 Multi-Ref Long Video Workflow](#wanscail2-multi-ref-long-video-workflow)
- [Usage Examples](#usage-examples)
- [License](#license)

---

## Installation

1. Clone this repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/user2318/ComfyUI-CustomNodeKit.git
```

2. Install Python dependencies:

```bash
cd ComfyUI-CustomNodeKit
pip install -r requirements.txt
```

3. Restart ComfyUI.

> **Note**: Ensure your ComfyUI environment has PyTorch and related core dependencies installed.

---

## Node List

Nodes are organized by functional category. Hover over any node parameter in ComfyUI to see the bilingual (Chinese + English) tooltip for detailed usage instructions.

### WanLoop Video Generation

| Node | Category | Description |
|------|----------|-------------|
| **WanAnimateToVideoCustom** | `WanLoop/Complete` | Core WanAnimate video generation node. Supports pose/face control, tail-frame segmented masking, neutral gray blending, context mode, prev_latent continuation, and full parameter set |
| **Single Frame VAE Encode** | `WanLoop/Tools` | Encodes N images frame-by-frame independently with VAE and concatenates in temporal dimension to produce latent. Provides anchor latents for projects like ComfyUI-EverAnimate |
| **WanUni3CLoader** | `WanVideo/Control` | Loads Uni3C ControlNet model with fp32/bf16/fp16 precision options (optional) |
| **WanUni3CApply** | `WanVideo/Control` | Injects Uni3C camera control into KSampler. Supports strength/start_percent/end_percent parameters (optional) |

### SCAIL/SCAIL-2 Multi-Ref Generation

| Node | Category | Description |
|------|----------|-------------|
| **Wan SCAIL To Video (Multi Ref)** | `model/conditioning/video_models` | SCAIL/SCAIL-2 multi-reference image video generation conditioning node. Supports auto-encoding multiple reference images, SCAIL-2 multi-identity colored mask injection, pose video guidance, motion continuation (prev_latent), etc. |
| **Create SCAIL-2 Colored Mask (Multi Ref)** | `conditioning/video_models/scail` | Renders SAM3 tracking data into SCAIL-2 colored masks. Supports shared palette sorting (left_to_right/area) for reference images and driving video, ensuring consistent identity coloring across multi-person scenes |
| **WanSparseAttention** | `model/conditioning/video_models/scail` | SCAIL sparse attention & causal attention node. Uses attention masks to limit interactions between pose/ref/main tokens, reducing degeneration in long video generation. Supports multiple attention mask strategies: no mask, pose-disabled-from-main, causal-only, pose-disabled-from-main+causal, etc. |
| **Wan SCAIL-2 Phase Sampler** | `sampling` | Standalone two-phase sampling node. Faithfully replicates the two-phase sampling core logic from SCAIL2LoopSampler. Uses pixel frames (IMAGE) as anchor input, internally performs VAE encoding, anchor frame write-back, and noise_mask management |
| **CLIP Vision Multi-Ref Switch** | `conditioning/video_models/scail` | Multi-image CLIP feature concatenation node. Concatenates batch CLIP vision features in token dimension, enabling all reference images to contribute to conditioning |
| **Auto Color Drift Correction V4.1** | `CustomNodes/Video` | Self-aligned color drift correction node V4.1. Three-layer correction: ① intra-segment drift detection (auto mode identifies jumps, bypasses if no drift); ② seam alignment eliminates inter-segment jumps; ③ bump correction for intra-segment oscillation + adaptive EMA template learning. V4.1 fixes auto mode jump detection frame index, seam alignment calculation, and seam_strength parameter ignoring issues |

### SDPose Pose System

| Node | Category | Description |
|------|----------|-------------|
| **Draw SDPose Keypoints (V2)** | `SDPose` | Renders pose keypoint data into visualization images. Supports layered drawing of full body skeleton, hands, face, and feet. Auto-adjusts bone thickness and occlusion order based on yaw angle. New `foot_mode` parameter supports `dots` (circles) and `line` (ankle→big toe) foot rendering |
| **Save SDPose Keypoints as JSON** | `SDPose` | Saves pose keypoint data to JSON files with overwrite mode and auto-increment numbering |
| **Load SDPose JSON** | `SDPose` | Loads JSON pose files with auto frame-rate resampling (linear interpolation or copy) and empty frame auto-repair |
| **Slice SDPose Keypoints** | `SDPose` | Temporal slicing of pose sequences (by start frame and frame count) |
| **Concat SDPose Keypoints** | `SDPose` | Concatenates two pose sequences front-to-back |
| **Estimate Yaw (Simple)** | `SDPose` | Simplified yaw angle estimation from pose keypoints. Core parameters: confidence threshold, disentanglement, shoulder weight |
| **Estimate Yaw (Advanced)** | `SDPose` | Full yaw angle estimation with all underlying parameters adjustable (smoothing window, EMA alpha, angle limits, lateral calibration, etc.) and detailed debug table output |
| **Resize SDPose Keypoints** | `SDPose` | Rescales pose keypoint coordinates and updates canvas size. Supports aspect ratio preservation and smart cropping (based on keypoint bounding box) |
| **Resample SDPose Keypoints** | `SDPose` | Frame-rate resampling (decimation/interpolation) of pose keypoint sequences, independent of JSON loading logic. Supports empty frame auto-repair |
| **Load GroundingDINO Model** | `SDPose/GD` | Loads GroundingDINO model (globally cached, GPU-resident) to provide bbox detection for official SDPose |
| **GD BBox Detect** | `SDPose/GD` | All-in-one GroundingDINO detection: text-based detection → multi-mode filtering → outputs bbox for official SDPoseOODProcessor |
| **Reference Image Selector** | `CustomNodes/SDPose` | Reference image selector: automatically filters and sorts reference image batches based on yaw angle range |

### Video Tools

| Node | Category | Description |
|------|----------|-------------|
| **VideoFrameCounter** | `video` | Gets frame count statistics and frame rate from input video/image sequences |
| **ImageSequenceToVideo** | `video` | Assembles image sequences into video files using ffmpeg. Supports CRF, audio synthesis, frame offset, etc. |

### Interactive Tools

| Node | Category | Description |
|------|----------|-------------|
| **Interactive Batch Crop** | `Interactive` | Interactive batch crop node with frontend JS for graphical region selection and batch processing |

### Utility Tools

| Node | Category | Description |
|------|----------|-------------|
| **Path Collector** | `Utility` | Collects file paths from specified directories |
| **Index Selector** | `Utility` | Selects elements from a list by index, supports `last_folder` output |
| **Path Validator** | `Utility` | Validates file/folder path existence |
| **Integer Setting (整数设置)** | `CustomNodes/Utils` | Rounds integers to target step size (start + step×n), supports negative integers |

### Image Batch Tools

| Node | Category | Description |
|------|----------|-------------|
| **Folder Image Loader** | `image` | Loads image batches from a folder sorted by filename. Supports skip, count limits, and size synchronization |
| **Image Batch Concat** | `image` | Concatenates two image batches. Passes through the non-empty input if only one is connected |
| **Image Batch Resize** | `image` | Resizes image batches to specified dimensions. Optional directional cropping to maintain aspect ratio |
| **Batch Image Replace** | `image` | Batch image replacement node. Replaces images at specified positions in a batch by start index and count. Supports overflow modes and underflow mode, with optional replacement source |
| **Batch Frame Replicate** | `image` | Batch frame replication node. Replicates a specified image and inserts copies after it. Supports negative indexing and dual-channel independent input. Used for extending first/last frame duration, keyframe freeze effects |
| **MaskCompositeRef** | `image` | Mask batch compositing node — reference_image preprocessing for WanSCAILToVideo. Supports replacement mode and animation mode, auto-filters fully black mask frames, background compositing |

### Context Tools

| Node | Category | Description |
|------|----------|-------------|
| **Custom Context Windows (Manual)** | `context` | Universal context window scheduling node. Splits long sequences into sliding windows for piecewise processing. Supports reference frame prefix, noise shuffling, multiple scheduling strategies (uniform/static/cyclic/batched), and fusion modes (pyramid/linear/relative). New `causal_window_fix` feature preserves cross-window interaction details (e.g., footprints) |
| **Wan SCAIL-2 Context Windows** | `context` | SCAIL-2 dedicated context window node. Automatically handles SCAIL-2 specific conditioning fields (ref_mask_28ch, driving_mask_28ch, pose_video_latent). No prefix reference frames or causal_window_fix needed. Works with WanSCAILToVideoMultiRef |

---

## Complex Node Details

### WanAnimateToVideoCustom

The core integration node for WanAnimate video generation. It encodes multiple inputs (reference images, pose video, face video, background video, character mask) into the appropriate conditioning and latent format, ready for direct connection to KSampler.

#### Parameters

**Required:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `positive` | CONDITIONING | Positive prompt conditioning |
| `negative` | CONDITIONING | Negative prompt conditioning |
| `vae` | VAE | VAE model for encoding images to latent |
| `width` / `height` | INT | Output resolution (default 832×480, step 16) |
| `length` | INT | Total output frames in pixels (step 4) |
| `batch_size` | INT | Batch size (usually 1) |
| `continue_motion_max_frames` | INT | Max reference frames for motion continuation (from previous segment tail) |
| `video_frame_offset` | INT | Video frame offset (incremented for multi-segment stitching) |
| `transition_width` | INT | Black frame transition width in fix mode (0-128, step 4) |
| `mode` | enum | `vanilla`=official behavior / `legacy`=old tail-frame segmented mask / `fix`=hard replacement neutral gray + transition diffusion |
| `tail_frame_count` | INT | Legacy mode tail-frame mask coverage |
| `tail_start_strength` | FLOAT | Legacy mode tail-frame mask start strength |
| `tail_end_strength` | FLOAT | Legacy mode tail-frame mask end strength |
| `ref_mode` | enum | `original`=internal 1+4n arrangement + batch encode (connect to selected_images); `compatible`=frame-by-frame independent encode (connect to selected_images, compatible with EverAnimate LoRA) |

**Optional:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `clip_vision_output` | CLIP_VISION_OUTPUT | CLIP Vision output for reference image semantic understanding |
| `reference_image` | IMAGE | Reference image batch (first frame ×1 + remaining ×4 concatenated) |
| `face_video` | IMAGE | Face video input (auto-resized to 512×512, normalized to [-1, 1]) |
| `pose_video` | IMAGE | Pose video input (encoded as pose_video_latent) |
| `continue_motion` | IMAGE | Previous segment's tail frames for motion continuation |
| `background_video` | IMAGE | Background video, overwrites area after motion continuation frames in reference batch |
| `character_mask` | MASK | Character mask, replaces concat_mask area after motion continuation frames |
| `prev_latent` | LATENT | Previous segment's full latent output (concat_latent). When connected, ignores continue_motion and directly replaces neutral gray frame latent with previous segment's latent |
| `face_strength` | FLOAT | Face video strength coefficient (0-1, default 1.0) |
| `mid_frame` | INT | Legacy mode middle anchor frame position (-1=disabled) |
| `mid_strength` | FLOAT | Legacy mode middle anchor frame strength |
| `neutral_mix_min` / `neutral_mix_max` | FLOAT | Legacy mode neutral gray mix range (mask→neutral gray blend ratio) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `positive` | CONDITIONING | Positive conditioning with injected concat_latent_image, concat_mask, etc. |
| `negative` | CONDITIONING | Negative conditioning with corresponding injections |
| `latent` | LATENT | Empty latent placeholder (samples shape matches total frame count) |
| `trim_latent` | INT | Number of leading latent frames to skip (for VAE Decode trimming) |
| `trim_image` | INT | Number of leading pixel frames to skip |
| `video_frame_offset` | INT | Cumulative frame offset (for multi-segment stitching) |
| `concat_latent` | LATENT | Complete concatenated latent image (for decoding output) |

#### Typical Workflow

```
[Prev segment tail] → continue_motion ─┐
[Reference images]  → reference_image ─┤
[Pose video]        → pose_video ──────┤
[Face video]        → face_video ──────┤
[Positive prompt]   → positive ────────┤
[Negative prompt]   → negative ────────┤
                                         ↓
                                WanAnimateToVideoCustom
                                         ↓
                      positive / negative / latent → [KSampler]
                                                         ↓
                                                 [VAE Decode]
                                                         ↓
                                                    Output video
```

---

### Custom Context Windows (Manual)

A context window scheduling node for WAN-type video generation models. When video frames exceed the model's single-processing limit, it splits the long sequence into sliding windows, processes them piecewise, and merges results via fusion strategies. This node wraps the model for use with subsequent KSampler.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | MODEL | Model to wrap |
| `context_length` | INT | Context window length in pixel frames (auto-converted to latent frames, every 4 frames → 1) |
| `context_overlap` | INT | Window overlap length in pixel frames (auto-converted) |
| `context_schedule` | enum | Window scheduling strategy (see below) |
| `context_stride` | INT | Uniform stride exponent upper limit (1-10) |
| `closed_loop` | BOOLEAN | Whether to close the window loop (cyclic schedule only) |
| `fuse_method` | enum | Window fusion method (see below) |
| `freenoise` | BOOLEAN | Enable FreeNoise noise shuffling (improves inter-window continuity) |
| `prefix_latent_num` | INT | Number of prefix reference frame latents. Set to match the count of images from reference image selector's raw_reference_images. These frames are prepended to each window as stable references |
| `split_conds_to_windows` | BOOLEAN | Whether to assign multiple conditionings to windows by region index |
| `causal_window_fix` | BOOLEAN | Causal window fix (default True). When enabled, prepends the previous window's denoised last frame to each window, preserving details like footprints. Sacrifices parallelism (windows execute serially per denoise step) |

#### Scheduling Strategies (context_schedule)

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `looped_uniform` | Cyclic uniform scheduling: windows distributed cyclically on timeline, stride increases exponentially | Scenarios requiring multiple iterative coverage |
| `standard_uniform` | Standard uniform scheduling: similar to looped but without cycles, removing duplicate windows | First choice for most general scenarios |
| `standard_static` | Standard static scheduling: fixed-step sliding windows (step = context_length - context_overlap) | Long videos needing simple segmented processing |
| `batched` | Batch scheduling: direct slicing by context_length, no overlap | Quick batch processing without transitional fusion |

#### Fusion Methods (fuse_method)

| Method | Description | Use Case |
|--------|-------------|----------|
| `pyramid` | Pyramid weights: maximum weight at window center, decreasing toward edges | General purpose, natural transitions |
| `flat` | Flat weights: equal weight for all frames | Simple averaging fusion |
| `overlap-linear` | Linear overlap: linear gradient in overlap regions (first frame 1→0, last frame 0→1) | Scenarios needing clear boundary transitions |
| `relative` | Relative weighted running average: dynamic weighting based on window center distance, smooth batch sampling accumulation | Advanced scenarios requiring fine weight control |

#### Usage Tips

1. **Frame conversion**: `context_length` and `context_overlap` are input in pixel frames; the node auto-converts to latent frames (every 4 pixel frames → 1 latent frame).
2. **Prefix reference frames**: When `prefix_latent_num > 0`, that many reference latents are prepended to each window as stable references, suitable for tasks requiring global context.
3. **freenoise**: Enable this to shuffle noise for improved texture continuity across windows. Recommended for long sequences with small overlaps.
4. **causal_window_fix**: When enabled, prepends the previous window's denoised last frame as an anchor to each window, effectively preserving cross-window continuity details. Note this forces serial execution, reducing parallelism.

#### Typical Workflow

```
[Load model] → model ──┐
                        ↓
             Custom Context Windows (Manual)
                        ↓
                model (wrapped) → [KSampler] → [VAE Decode] → Output
```

---

### Reference Image Selector

Reference image selector that automatically filters and selects the best reference image batch based on the yaw angle range of a video segment. Typical use case: in WanAnimate video generation, select the best-matching reference images from multi-angle reference images based on character orientation (yaw).

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reference_images` | IMAGE | ✅ | Reference image batch (N multi-view reference images) |
| `angle_map` | STRING | ✅ | JSON format angle mapping, e.g. `[-90, -45, 0, 45, 90]`. Array length must match reference image count |
| `yaw_angles` | FLOAT | ❌ | Target video segment's yaw angle sequence (one value per frame). Outputs all reference images when not connected |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `selected_images` | IMAGE | Filtered and sorted reference image batch (main reference ×1 + auxiliary references ×1 each, original images directly output) |
| `raw_reference_images` | IMAGE | Original reference images (unsorted, for connecting to Custom Context Windows' prefix_latent_num) |
| `info` | STRING | Debug info (angle range, candidate indices, sorting results, etc.) |
| `reference_angle_map` | STRING | Reference image angle mapping JSON (for downstream nodes) |

#### How It Works

1. **No yaw_angles or invalid angle_map**: Outputs all reference images (index 0 ×1 + rest ×4 each).
2. **With yaw_angles input**:
   - Computes yaw angle range [min, max]
   - Filters all reference images within this range + the nearest one on each side
   - Selects the reference image covering the most frames as the **main reference** (placed at index 0)
   - Auxiliary references sorted by deviation from the first frame's yaw angle (descending)
   - If the main reference already best matches the first frame's yaw, appends a copy of the main reference at the end
3. **Output batch format**: Index 0 (main reference) ×1, remaining auxiliary references ×4 each (matching WanAnimate's concat format requirement)

#### Typical Workflow

```
[Multi-view references] → reference_images ─┐
[Angle map JSON]       → angle_map ─────────┤
[Yaw sequence]         → yaw_angles ────────┤
                                              ↓
                                   Reference Image Selector
                                              ↓
                                     selected_images → [WanAnimateToVideoCustom] reference_image
```

> **Tip**: The `angle_map` input can be directly connected to the frontend JS node `Angle Map Editor` for visual angle mapping configuration.

---

### SDPose Empty Frame Repair

In the SDPose pose processing pipeline, some frames may have an empty `people` list due to detection failures (producing black frames). The `fix_empty_frames` feature automatically detects and repairs these empty frames.

#### Applicable Nodes

- **Load SDPose JSON**: Repairs when loading JSON (parameter `fix_empty_frames`)
- **Resample SDPose Keypoints**: Repairs after resampling (parameter `fix_empty_frames`)

#### Repair Logic

1. Scans all frames, marking frames with empty `people` or zeroed keypoint coordinates as "invalid"
2. For each invalid frame:
   - **Valid frames on both sides**: Linear interpolation fill (if adjacent frames have sudden changes, copies the nearer side)
   - **Valid frame on one side only**: Copies data from that side
3. The repair process does not modify the original frame list; it returns a new list

#### Usage Tips

- Recommended to enable when pose data sources are unstable
- When used with `Estimate Yaw` nodes, effectively prevents angle jumps caused by empty frames

---

### SDPoseDrawKeypointsV2 Foot Mode

Starting from v1.4.0, SDPoseDrawKeypointsV2 adds the `foot_mode` parameter with two foot rendering styles:

| Mode | Description |
|------|-------------|
| `dots` | Default mode. Draws colored dots at foot keypoint positions (consistent with previous behavior) |
| `line` | Line mode. Draws lines from ankle → big toe for clearer foot orientation and pose indication (no dots) |

Both modes use `bottom_side` / `top_side` parameters for front/back layered rendering, ensuring correct occlusion order.

---

### Folder Image Loader

Reads image batches from a specified folder, sorted by filename. Gracefully handles errors such as invalid paths or empty folders — outputs empty results and prints warnings to console rather than raising exceptions.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `folder_path` | STRING | ✅ | Image folder path. Supports absolute paths or relative paths (relative to ComfyUI root) |
| `size_mode` | enum | ✅ | `resize_to_first`: resize all images to first image's dimensions; `filter_same_size`: only load images matching the first image's dimensions |
| `skip_first_n` | INT | ✅ | Skip the first N images (default 0) |
| `load_count` | INT | ✅ | Maximum number of images to load (0=no limit, load all) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Loaded image batch. Outputs None if path is invalid or no images found |
| `count` | INT | Actual count of loaded images. Outputs 0 for error conditions |

#### Error Handling

- Empty folder path → Outputs `(None, 0)`, console warning
- Folder not found → Outputs `(None, 0)`, console warning
- No images in folder → Outputs `(None, 0)`, console warning
- Skip count exceeds total images → Outputs `(None, 0)`, console warning
- Single image read failure → Skips that image, continues loading others

---

### Image Batch Concat

Concatenates two image batches along the batch dimension (images_b appended after images_a). When one or both inputs are disconnected, passes through the valid input; outputs None when both are disconnected.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `images_a` | IMAGE | ❌ | First image batch |
| `images_b` | IMAGE | ❌ | Second image batch, concatenated after images_a |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Concatenated image batch. Passes through single valid input; outputs None if both inputs are invalid |

#### Typical Workflow

```
[FolderImageLoader A] → images_a ─┐
[FolderImageLoader B] → images_b ─┤
                                    ↓
                           Image Batch Concat
                                    ↓
                               images (merged batch)
```

---

### Image Batch Resize

Resizes an image batch to specified width and height. Supports directional cropping to maintain aspect ratio. Outputs None for invalid inputs.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `width` | INT | ✅ | Target width (default 512) |
| `height` | INT | ✅ | Target height (default 512) |
| `crop_mode` | enum | ✅ | Crop mode. `disabled`=direct stretch; `center`/`top`/`bottom`/`left`/`right`=scale to cover target then crop directionally |
| `images` | IMAGE | ❌ | Input image batch. Outputs None if not connected |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Resized image batch. Outputs None for invalid input |

#### Resize Algorithm

Uses Pillow's LANCZOS algorithm uniformly for high-quality downscaling and upscaling.

#### Crop Mode Illustration

```
Original 1920×1080 → Target 512×512
disabled: Direct stretch, aspect ratio distorted
center:   Scale to 911×512, crop 199px from left and right
top:      Scale to 911×512, crop 399px from bottom
bottom:   Scale to 911×512, crop 399px from top
left:     Scale to 512×288, crop 176px from right (then resize to 512×512)
right:    Scale to 512×288, crop 176px from left (then resize to 512×512)
```

> Crop modes always scale to cover the target dimensions first (using max scale ratio), then crop excess content directionally, ensuring final output dimensions exactly match the target.

---

### WanAnimate Uni3C Camera Control

Uni3C camera control provides optional camera motion control for WanAnimate video generation. It works by injecting ControlNet signals during KSampler sampling, guiding the generation to follow specified camera trajectories. **This is an optional feature** and does not affect normal generation when not in use.

#### Prerequisites

1. Download the Uni3C ControlNet model file (`.safetensors`) and place it in `ComfyUI/models/controlnet`
2. Install `diffusers` and `accelerate` Python packages

#### WanUni3CLoader — Model Loading

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | COMBO | ✅ | Uni3C ControlNet model file, auto-scans controlnet directory |
| `base_precision` | enum | ❌ | Base precision: `fp32` / `bf16` / `fp16` (default `fp16`) |

**Output**: `uni3c_controlnet` (UNI3C_CONTROLNET) — Loaded ControlNet model for WanUni3CApply use.

#### WanUni3CApply — Control Injection

Injects Uni3C camera control signals into KSampler. The node monkey-patches the model's internal `forward_orig` method to compute and inject control signals at each denoising step.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | MODEL | ✅ | Model to inject control into |
| `uni3c_controlnet` | UNI3C_CONTROLNET | ✅ | ControlNet model loaded by WanUni3CLoader |
| `render_latent` | LATENT | ✅ | Pre-rendered reference video latent tensor (B, C, T, H, W) as visual reference for camera control |
| `strength` | FLOAT | ✅ | Control strength (0.0-10.0, default 1.0) |
| `start_percent` | FLOAT | ✅ | Control activation start percentage (0.0-1.0, default 0.0) |
| `end_percent` | FLOAT | ✅ | Control activation end percentage (0.0-1.0, default 1.0) |
| `render_mask` | MASK | ❌ | Optional render mask (experimental) |
| `trim_latent` | INT | ❌ | Number of reference latent frames, connected from WanAnimateToVideoCustom's trim_latent output for temporal alignment |
| `positive` | CONDITIONING | ❌ | Optional: pass conditioning to auto-extract concat_latent_image, replacing WanAnimateChannelPack node |
| `negative` | CONDITIONING | ❌ | Same as above, negative condition |

**Output**: `model` (MODEL) — Control-injected model, ready for direct connection to KSampler.

#### Core Principles

1. **v4 Architecture**: Uses monkey-patch to modify the model's `forward_orig` method, precisely controlling injection timing within the block loop
2. **Temporal Alignment**: Uses the `trim_latent` parameter to auto-align render_latent with the generated video's temporal offset
3. **Async Prefetch**: Control signals use CPU→GPU async prefetch, overlapping with block computation to minimize performance overhead
4. **Step Control**: Uses `start_percent` / `end_percent` to control the activation range of control signals during the denoising process

#### Typical Workflow

```
[WanUni3CLoader] → uni3c_controlnet ─┐
[Rendered latent] → render_latent ────┤
                                        ↓
[Model] → WanUni3CApply → model (injected) → [KSampler] → [VAE Decode] → Output
```

> **Note**: The WanUni3CApply node has built-in WanAnimateChannelPack logic. If `positive`/`negative` conditioning is provided, it automatically extracts `concat_latent_image` from them — no separate WanAnimateChannelPack node needed.

---

## Workflows

The `workflow/` directory contains complete ComfyUI workflow JSON files that can be loaded directly in the ComfyUI interface.

### Long Video Pose Detection Workflow

**Files**: `Long_Pose_detection_SD_offical.json` / `Long_Pose_detection_SDpose.json`

**Purpose**: Performs frame-by-frame body and face keypoint detection on long videos. Supports body ratio alignment (BodyRatioMapper) between detected skeletons and reference character skeletons. Produces skeleton visualization video, face feature JSON, and pose JSON files. Suitable for dance skeleton animation, motion capture data extraction, etc.

- `Long_Pose_detection_SD_offical.json`: Uses official SDPoseOODProcessor for pose recognition
- `Long_Pose_detection_SDpose.json`: Uses SDPose pipeline (with custom nodes) for pose recognition

**Quick Start**:

1. Set resolution (width/height), frame rate (fps), frames-per-segment in the "Parameter Settings" area
2. Configure input video path and segment parameters in the "Video Segment Loop Loading" area
3. Set reference skeleton JSON path and reference image in the "Detect & Save Face Images, Face Keypoints JSON, Pose JSON" area for skeleton ratio alignment
4. Adjust skeleton drawing parameters (line width, face/hand keypoint size, etc.) in the "Skeleton Alignment, Drawing & Video Compilation" area
5. **Clear previous output files from the working directory before running**, especially the face folder, to avoid old data interference
6. Run to process the long video in segments — SDPose recognition, skeleton alignment, visualization drawing, and finally skeleton video compilation (with original audio preserved)

**Video Assembly**: Uses `ImageSequenceToVideo` to compile drawn skeleton frames into video, optionally syncing original video audio (via `VHS_LoadAudio`).

**Third-party node packages used** (excluding `ComfyUI-CustomNodeKit`):

| Package | Main Usage |
|---------|------------|
| `sdpose-ood` | SDPose model loading (`SDPoseOODLoader`) and processing (`SDPoseOODProcessor`) |
| `ComfyUI-WanAnimatePreprocess` | ONNX face detection model loading (`OnnxDetectionModelLoader`) and face image extraction (`PoseAndFaceDetection`) |
| `comfyui-videohelpersuite` | Video loading (`VHS_LoadVideoFFmpegPath`) and audio loading (`VHS_LoadAudio`) |
| `comfyui-kjnodes` | Constant nodes (`INTConstant`, `FloatConstant`, `StringConstant`) |
| `comfyui-easy-use` | While loop control (`easy whileLoopStart`/`easy whileLoopEnd`) and cache clearing (`easy clearCacheAll`) |
| `comfyui-impact-pack` | Loop condition comparison (`ImpactCompare`) |
| `comfyui-custom-scripts` | Math expression evaluation (`MathExpression\|pysssss`) |
| `ComfyUI-BodyRatioMapper` | Skeleton ratio alignment (`BodyRatioMapperProportionTransfer`) |
| `ComfyUI-Addoor` | Temporary file cleanup (`AD_DeleteLocalAny`) |

### WanAnimate Multi-Ref Long Video Workflow

**File**: `WanAnimate_ref_pics_loop+context.json`

**Purpose**: Generates long videos with context window support using multiple reference images based on the WanAnimate model. Includes a **360° image batch preprocessing module** that processes a single reference image into a 360° panoramic view image batch (supporting files stored in the output directory). Users can select suitable angle images as multi-reference inputs from the generated results for better appearance consistency control.

**Quick Start**:

1. Load WanAnimate UNet model, CLIP model, VAE, and LoRAs in the "Model, LoRA & VAE Loading" area
2. Load and resize the main reference image to 512×512 in the "Set Main Reference Image" area
3. Use `PathCollectorNode` to specify the auxiliary reference image directory in the "Set Auxiliary Reference Images" area (can use 360° preprocessing output)
4. Enter positive/negative prompts in the "Prompt Input" area
5. Set per-segment frame count, inter-segment overlap, etc., in the "Video Generation Parameters" area
6. Choose action and face data loading method:
   - **Standard Load**: Loads original video directly with real-time face/pose recognition (no skeleton alignment)
   - **Fast Load**: Loads pre-exported pose JSON and face JSON directly (supports skeleton alignment)
7. Run to generate video frames segment-by-segment via the loop control area, using `Custom Context Windows` for cross-segment context scheduling
8. **After generation**, manually enable the "Save Video" area and run `ImageSequenceToVideo` to compile frame sequences into video (optionally with original audio)

**Video Assembly**: Frames generated by KSampler (via `VAE Decode`) → cross-segment fade-in/out fusion (`CrossFadeImages`) → `ImageSequenceToVideo` compiles complete sequence into video.

**360° Preprocessing Module**:

- Located in the lower half of the workflow (usually muted), includes `InteractiveBatchCrop` interactive cropping, `SDPoseLoadJson` loading 360° keypoint JSON, `SDPoseResizeKeypoints` keypoint resizing, `SDPoseDrawKeypointsV2` pose video drawing
- Supporting files (360° keypoint JSON, etc.) should be placed in the `output/video/360°` directory
- Users can run the preprocessing module to generate 360° panoramic pose video batches and select suitable angle frames as auxiliary references
- For complex preprocessing, use the main workflows (`Long_Pose_detection_SD_offical.json` or `Long_Pose_detection_SDpose.json`)

**Third-party node packages used** (excluding `ComfyUI-CustomNodeKit`):

| Package | Main Usage |
|---------|------------|
| `sdpose-ood` | SDPose model loading and processing |
| `ComfyUI-WanAnimatePreprocess` | ONNX face detection model loading and face image extraction |
| `comfyui-videohelpersuite` | Video loading (`VHS_LoadVideoFFmpegPath`), image splitting (`VHS_SplitImages`), audio loading |
| `comfyui-kjnodes` | Constant nodes, `CrossFadeImages`, `ColorMatch`, `PatchSageAttentionKJ`, etc. |
| `comfyui-easy-use` | While loop control, IfElse branching, cache clearing, `easy imageSwitch` |
| `comfyui-impact-pack` | Loop condition comparison (`ImpactCompare`), null detection (`ImpactIfNone`) |
| `comfyui-custom-scripts` | Math expression evaluation (`MathExpression\|pysssss`) |
| `ComfyUI-BodyRatioMapper` | Skeleton ratio alignment (`BodyRatioMapperProportionTransfer`) |
| `rgthree-comfy` | `Any Switch` multi-image switching, `Fast Muter` |
| `pr-was-node-suite-comfyui-47064894` | `Image Filter Adjustments`, `Random Number` |
| `comfyui_memory_cleanup` | `RAMCleanup` / `VRAMCleanup` |
| `ComfyUI-EasyColorCorrector` | `BatchColorCorrection` |

### WanSCAIL2 Multi-Ref Long Video Workflow

**File**: `WanSCAIL2_ref_pics_loop+context.json`

**Purpose**: Generates long videos with context window support using multiple reference images based on the Wan SCAIL/SCAIL-2 model. Supports multi-identity recognition (via SAM3 tracking + colored mask) with both Animation Mode and Replacement Mode generation strategies.

**Quick Start**:

1. Load SCAIL model and related components in the "Model, LoRA & VAE Loading" area
2. Load reference images and generate SAM3 tracking data
3. Load the driving video and generate SAM3 tracking data
4. Use `Create SCAIL-2 Colored Mask (Multi Ref)` to generate colored masks. Choose appropriate sorting method (left_to_right/area) for consistent multi-identity coloring
5. Connect colored masks and driving video to `Wan SCAIL To Video (Multi Ref)`
6. Run to generate video frames segment-by-segment with context window support
7. Compile frame sequences into video after generation

---

## Dependencies

### Python Dependencies

Core dependencies (declared in `requirements.txt`):

- `torch` — PyTorch deep learning framework (usually provided by ComfyUI environment)
- `numpy` — Numerical computation
- `Pillow` — Image processing
- `aiohttp` — Async HTTP communication
- `colorsys` — Color space conversion (Python standard library)
- `tqdm` — Progress bar display
- `imageio-ffmpeg` — Video compilation (auto-downloads ffmpeg, no system installation needed)

---

## Usage Examples

### 1. WanAnimate Video Generation

```
[Reference Image] → WanAnimateToVideoCustom
[Pose Video]       →  (pose_video input)
[Face Video]       →  (face_video input)
                        ↓
                positive / negative / latent → [KSampler] → [VAE Decode] → Output video
```

### 2. SDPose Pose Visualization Pipeline

```
[SDPose JSON file] → Load SDPose JSON
                       ↓
               [Estimate Yaw] → yaw_array
                       ↓
               [Resize Keypoints] → unified size
                       ↓
               [Draw Keypoints V2] → visualization (supports foot_mode=line/dots)
```

### 3. Context Window Video Generation

```
[Model] → Custom Context Windows (Manual) → model (wrapped)
                                               ↓
[prompt] → CLIP Text Encode → positive ──────┤
[WanAnimateToVideoCustom] → positive/negative/latent → [KSampler] → [VAE Decode] → video
```

### 4. Reference Image Auto-Selection + Video Generation

```
[Multi-angle refs] → Reference Image Selector ← yaw_angles ← [Estimate Yaw]
                        ↓
                selected_images → [WanAnimateToVideoCustom] reference_image
```

### 5. SCAIL/SCAIL-2 Multi-Ref Video Generation

```
[SAM3 ref tracking] → ref_track_data ──┐
[SAM3 driving]      → driving_data ────┤
                                        ↓
                       Create SCAIL-2 Colored Mask (Multi Ref)
                                        ↓
                   pose_video_mask / reference_image_mask
                                        ↓
[Multi ref images] → reference_image ──┤
[Driving video]    → pose_video ───────┤
[Conditioning]     → positive/negative ┤
                                        ↓
                         Wan SCAIL To Video (Multi Ref)
                                        ↓
                        positive / negative / latent → [KSampler]
```

---

## Acknowledgments

Some nodes in this project provide auxiliary services for third-party nodes (e.g., bbox detection, pose data preprocessing). Thanks to these excellent projects for their inspiration:

- sdpose-ood — SDPose pose system
- ComfyUI-WanAnimatePreprocess & comfyui-kjnodes (Kijai) — WanAnimate preprocessing and toolkit
- GroundingDINO (ShilongLiu) — GD detection model interface
- ComfyUI-BodyRatioMapper — Skeleton ratio alignment
- comfyui-videohelpersuite — Video tools
- comfyui-custom-scripts — Utility nodes
- ComfyUI-Scail2-Sampler-Helper ([checknickname](https://github.com/checknickname)) — SCAIL-2 two-phase sampling core logic

Meanwhile, the WanAnimate multi-reference video generation logic, yaw angle estimation algorithm, context window scheduling strategy, skeleton rendering engine, and SCAIL/SCAIL-2 multi-reference support in this project are original implementations.

---

## License

MIT License