# ComfyUI Custom Node Kit

A set of custom nodes for ComfyUI, covering video generation, pose processing, interactive image operations, and common workflow utilities.

Current version: **1.9.0** (2026-10-16) — adds the one-switch **Anchor + In-step Rolling** node, the **In-step Rolling** node and **Detect Edges (Canny) Chunked** (chunked Canny edge detection), which fixes the official Detect Edges (Canny) freezing (silently spilling VRAM) on 200+ frame batches; 200 frames @1024² drops from ~29.7GB extrapolated peak VRAM to 1.31GB (see [Changelog](#changelog)); prior 1.8.0 introduction below: **1.8.0** — Added the WanAnimate2ToVideoCustom node (Wan2.1-I2V base + second-gen pose branch K/V injection, multi-reference image support); reworked WanUni3CApply to accept render_video with built-in VAE encoding and reference-frame offset (fixing the render_latent normalization bias); removed the retired WanSCAILSparseAttention node (its dense O(S²) attention masks blew up memory on long sequences and gave no measurable benefit on SCAIL); plus one **experimental / test-only node, Context Window Anchor (unified-basis anchoring)** — its effectiveness and side effects have not been fully validated, so it is kept for testing reference only; without `anchor_frames` connected it is exactly equivalent to not installing it (see [Context Window Anchor](#context-window-anchor-unified-basis-anchoring)).

This release also fixes a crash in the SDPose yaw nodes on short segments, the BatchImageReplace `repeat_end` padding logic (3 replacement images for 4 slots: `0,2,2,2` → `0,1,2,2`), and brings memory/performance improvements to SmartVideoCrop and InteractiveBatchCrop (the latter's preview cache drops from ~12.5GB to ~48MB for 2600 frames). **Please read the [Changelog](#changelog) at the end before upgrading** — it lists the breaking changes.

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
  - [WanAnimate2ToVideoCustom](#wananimate2tovideocustom)
  - [WanAnimateToVideoCustom](#wananimatetovideocustom)
  - [Custom Context Windows (Manual)](#custom-context-windows-manual)
  - [Context Window Anchor](#context-window-anchor-unified-basis-anchoring)
  - [Reference Image Selector](#reference-image-selector)
  - [SDPose Empty Frame Repair](#sdpose-empty-frame-repair)
  - [SDPoseDrawKeypointsV2 Foot Mode](#sdposedrawkeypointsv2-foot-mode)
  - [Folder Image Loader](#folder-image-loader)
  - [Image Batch Concat](#image-batch-concat)
  - [Image Batch Resize](#image-batch-resize)
  - [WanAnimate Uni3C Camera Control](#wananimate-uni3c-camera-control)
  - [SmartVideoCrop](#smartvideocrop)
  - [VideoGridCompose](#videogridcompose)
  - [BatchImageReplace](#batchimagereplace)
  - [ImageListSplitOutputs](#imagelistsplitoutputs)
  - [ImageListReplaceAt](#imagelistreplaceat)
  - [ImageListToBatch](#imagelisttobatch)
- [Workflows](#workflows)
  - [Long Video Pose Detection Workflow](#long-video-pose-detection-workflow)
  - [WanAnimate Multi-Ref Long Video Workflow](#wananimate-multi-ref-long-video-workflow)
  - [WanSCAIL2 Multi-Ref Long Video Workflow](#wanscail2-multi-ref-long-video-workflow)
- [Changelog](#changelog)
- [Dependencies](#dependencies)
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
| **WanAnimate2ToVideoCustom** | `WanLoop/Complete` | WanAnimate2 (official second-gen architecture) video generation core node. Retains first-gen multi-reference images (frame-by-frame independent encoding) with fix/legacy/vanilla mask handling, neutral gray blending and yaw angle output; adds official second-gen pose_strength / pose_start_percent / pose_end_percent / reference_image_strength / positive_pose / clip_vision_output_pose. Adapted for second-gen pose branch frame-count validation (reference latent frames padded in front for multi-ref scenarios) |
| **Single Frame VAE Encode** | `WanLoop/Tools` | Encodes N images frame-by-frame independently with VAE and concatenates in temporal dimension to produce latent. Provides anchor latents for projects like ComfyUI-EverAnimate |
| **WanUni3CLoader** | `WanVideo/Control` | Loads Uni3C ControlNet model with fp32/bf16/fp16 precision options (optional) |
| **WanUni3CApply** | `WanVideo/Control` | Injects Uni3C camera control into KSampler. Supports strength/start_percent/end_percent parameters (optional) |

### SCAIL/SCAIL-2 Multi-Ref Generation

| Node | Category | Description |
|------|----------|-------------|
| **Wan SCAIL To Video (Multi Ref)** | `model/conditioning/video_models` | SCAIL/SCAIL-2 multi-reference image video generation conditioning node. Supports auto-encoding multiple reference images, SCAIL-2 multi-identity colored mask injection, pose video guidance, motion continuation (prev_latent), etc. |
| **Create SCAIL-2 Colored Mask (Multi Ref)** | `conditioning/video_models/scail` | Renders SAM3 tracking data into SCAIL-2 colored masks. Supports shared palette sorting (left_to_right/area) for reference images and driving video, ensuring consistent identity coloring across multi-person scenes |
| **Wan SCAIL-2 Phase Sampler** | `sampling` | Standalone two-phase sampling node. Faithfully replicates the two-phase sampling core logic from SCAIL2LoopSampler. Uses pixel frames (IMAGE) as anchor input, internally performs VAE encoding, anchor frame write-back, and noise_mask management |
| **CLIP Vision Multi-Ref Switch** | `conditioning/video_models/scail` | Multi-image CLIP feature concatenation node. Concatenates batch CLIP vision features in token dimension, enabling all reference images to contribute to conditioning |
| **Auto Color Drift Correction V4.3** | `CustomNodes/Video` | Self-aligned color drift correction node V4.3. Two-pass correction: ① global offset — multi-frame mean detects segment jumps and corrects all new frames uniformly; ② residual fine-tuning — each frame moves toward the reference within a clamped limit. Supports standard/auto/off modes; `max_offset` caps the per-channel offset |

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
| **SmartVideoCrop** | `video` | Smart video crop driven by SAM3 tracking data: keeps the full height, computes each frame's person centroid, and pans horizontally with EMA smoothing + scene-cut detection before scaling to the target aspect. Supports multiple tracked IDs, empty-mask fallback (hold last position / go to center) and optional synchronized cropping of a second image channel |
| **VideoGridCompose** | `video` | Composes 1–4 videos/images into a grid (horizontal / vertical / 2×2 quad / one-left-N-right / one-top-N-bottom). Supports stretch-or-fit scaling, duration alignment (trim longer clips, freeze-pad shorter ones) and audio source selection. Fully streamed through ffmpeg, so no frames are loaded into memory |

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
| **Folder Image Loader** | `image` | Loads images from a folder sorted by filename; outputs a same-size batch `images`, a `count`, and a per-image list `images_list` (list semantics; with `size_mode=no_resize` the list keeps original sizes). Supports skip, count limits, and size handling |
| **Detect Edges (Canny) Chunked** | `image` | Chunked Canny edge detection that fixes the official node’s "no error, no progress, hang" behaviour on 200+ frame batches: same operator chain, same results (0 pixel difference on real images), peak VRAM ~25GB to ~1GB; progress bar, cancellable, `auto_chunk` sizing from free VRAM. |
| **Image Batch Concat** | `image` | Concatenates two image batches. Passes through the non-empty input if only one is connected |
| **Image Batch Resize** | `image` | Resizes image batches to specified dimensions. Optional directional cropping to maintain aspect ratio |
| **Batch Image Replace** | `image` | Batch image replacement node. Replaces images at specified positions in a batch by start index and count. Supports overflow modes and underflow mode, with optional replacement source |
| **Batch Frame Replicate** | `image` | Batch frame replication node. Replicates a specified image and inserts copies after it. Supports negative indexing and dual-channel independent input. Used for extending first/last frame duration, keyframe freeze effects |
| **Image List Split** | `image` | Splits an **image list** into N independent outputs (1-16; the visible output slots are added/removed live with the widget). Outputs the items from `start_index` onwards; channels beyond the available items are empty (`None`, i.e. **exactly like an unconnected wire**: optional downstream inputs are treated as "not provided"). An unconnected input produces no output and no error. List items may have different sizes; a plain IMAGE batch is split into single-frame items automatically |
| **Image List Replace** | `image` | Overwrites a run of a target image list starting at `start_index` with the replacement list; the length is decided by the replacement list and **never exceeds the original list length** (extra replacement items are dropped). Whole items are swapped, **no resizing**, list length unchanged; the output is a list (downstream runs per element) |
| **Image List To Batch** | `image` | Converts an image list with **mixed sizes** into a single-size batch (**never crops or stretches**): the reference is the item with the longest short side (= largest area × squareness), **its own size becomes the output size** (zero resampling for it), every other item is padded on **one axis only** (symmetric) and scaled proportionally. `output_mode` defaults to `RGB` (3ch, black padding, most compatible with VL/VAE) and can be switched to `RGBA` (4ch, alpha=0 padding, for compositing) |
| **MaskCompositeRef** | `image` | Mask batch compositing node — reference_image preprocessing for WanSCAILToVideo. Supports replacement mode and animation mode, auto-filters fully black mask frames, background compositing |


### Context Tools

| Node | Category | Description |
|------|----------|-------------|
| **Custom Context Windows (Manual)** | `context` | Universal context window scheduling node. Splits long sequences into sliding windows for piecewise processing. Supports reference frame prefix, noise shuffling, multiple scheduling strategies (uniform/static/cyclic/batched), and fusion modes (pyramid/linear/relative) |
| **Wan SCAIL-2 Context Windows** | `context` | SCAIL-2 dedicated context window node. Automatically handles SCAIL-2 specific conditioning fields (ref_mask_28ch, driving_mask_28ch, pose_video_latent). No prefix reference frames needed. Works with WanSCAILToVideoMultiRef |
| **Context Window Anchor (unified-basis anchoring)** | `context` | [Experimental / test-only] Prepends your anchor frames to every window's **driving video (plus SCAIL-2's driving mask)** and puts the same frames — **noised to the current σ**, as **ordinary frames** — at the front of the generation sequence, then trims them after sampling (equivalent to doing "prepend the video opening to a segment, generate, then crop" by hand). Minimal interface: `anchor_frames`(IMAGE) + `vae` (same Wan VAE) + SCAIL-2's `anchor_mask_frames`; **how many you feed is how many are used**. Without `anchor_frames` it is exactly equivalent to not installing the node. Must be placed after a context windows node |

---

### Supplementary list (synced with code, 2026-10-15)

> Nodes that had no row in the original tables.

| Node | Category | Description |
|------|----------|-------------|
| **锚定+步内滚动（极简）** | `context` | 「窗口锚定 + 步内滚动」的一键整合节点（面板只有 enabled 一个设置项） |
| **索引选择器** | `Custom Nodes` |  |
| **路径验证器** | `Custom Nodes` |  |
| **文件夹图像载入器** | `image` | 从文件夹按文件名升序载入图片：images 为同尺寸批次；images_list 为逐张列表（元素 [1,H,W,C]，尺寸可互不相同，下游按元素逐个执行） |
| **图像批次合并** | `image` |  |
| **图像批次缩放** | `image` |  |
| **逐帧独立 VAE 编码** | `WanLoop/工具节点` |  |
| **图像批次指定帧复制 (Frame Replicate)** | `image` |  |
| **步内滚动** | `context` | SCAIL-2 滚动窗口：在同一步内让后面的窗口看到前面窗口【本步】刚产出的 x0 |

## Complex Node Details

### WanAnimate2ToVideoCustom

A second-gen animate node that fuses the official `WanAnimate2ToVideo` (Wan2.1-I2V base + pose branch K/V injection) with the first-gen `WanAnimateToVideoCustom`. It retains first-gen multi-reference image support (frame-by-frame independent encoding, compatible with EverAnimate LoRA; the original 1+4n batch encoding mode was removed because it performed poorly on animate2), fix/legacy/vanilla mask handling, black-frame detection + transition, tail-frame segmented masking, neutral gray blending, and yaw_angles→latent_yaw_angles output. On top of that it adds the official second-gen `pose_strength` / `pose_start_percent` / `pose_end_percent` (pose time window; skipping the pose branch outside the window speeds up sampling) / `reference_image_strength` / `positive_pose` / `clip_vision_output_pose`. The pose branch follows the second-gen model's frame-count validation requirement (`pose_latents.shape[2] == f_gen - 1`) by padding latent frames in front, so multi-reference scenarios can be used directly. face/background/character_mask/prev_latent are removed since the second-gen architecture does not support them.

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
| `fuse_sharpen` | FLOAT | **Fusion weight sharpening (power)**, default 1.0 = off (historical behaviour). 3~6 narrows the crossfade band inside the overlap so one window dominates at any instant ⇒ **stays continuous (still a gradual blend) yet no more two characters ghosted on top of each other**. Most useful with large overlap / large disagreement between windows |

#### Scheduling Strategies (context_schedule)

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `looped_uniform` | Cyclic uniform scheduling: windows distributed cyclically on timeline, stride increases exponentially | Scenarios requiring multiple iterative coverage |
| `standard_uniform` | Standard uniform scheduling: similar to looped but without cycles, removing duplicate windows | First choice for most general scenarios |
| `standard_static` | Standard static scheduling: fixed-step sliding windows (step = context_length - context_overlap). ⚠️ When the last window overflows it **backs off to end-align**, making the last seam overlap much larger than requested (a warning is logged) | Long videos needing simple segmented processing |
| `standard_even` | **Equal-length windows, last window does not back off**: starts are spread evenly over `[0, total - context_length]`, so the overlap is evened out | Fixing the oversized last seam without changing window length |
| `batched` | Batch scheduling: direct slicing by context_length, **no overlap** (last window is shorter) | Pair with `Context Window Anchor`'s head/tail read-only context for "exactly one author per frame" |

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
4. **Overlap & ghosting**: the overlap region is *weighted-averaged* (`fuse_method`); if two windows disagree about the character's scale/position you get "two characters on top of each other". Three remedies: ① `fuse_sharpen=3~6` (narrows the crossfade band, still continuous); ② switch to `standard_even` (removes the oversized last seam); ③ use `batched` + `Context Window Anchor`'s `head/tail_context_frames` so **each frame has exactly one author**.
5. **The total overlap is locked by geometry**: `total overlap = #windows × window length − total length`. To make the overlap *small* you must use **shorter and more numerous** windows (e.g. 11 latent × 4 windows for 43 latent); no schedule can dodge that.

#### Typical Workflow

```
[Load model] → model ──┐
                        ↓
             Custom Context Windows (Manual)
                        ↓
                model (wrapped) → [KSampler] → [VAE Decode] → Output
```

---

### Context Window Anchor (unified-basis anchoring / read-only context / long-range motion memory)
> ⚠️ **Experimental / test-only node (not fully validated)**: effectiveness and side effects (quality, identity consistency, sampling stability) have not been systematically evaluated. A/B it with the **same seed** (bypassed vs active) before relying on it. **Merely connecting this node with no inputs and no parameters is exactly equivalent to not installing it.**

**Problem it addresses**: animate2 / SCAIL-2 have no explicit spatial anchor (no skeleton); under context windows every window's RoPE restarts at 0 (no absolute time), while SCAIL-2 aligns driving↔generated frames **rigidly 1:1 via identical RoPE positions**. Each window therefore infers its own basis from its own driving slice ⇒ two overlapping windows produce **two different motion predictions**, and fusion is a **weighted average** ⇒ ghosting / softened motion / one or two windows off. In addition, the `standard_static` schedule "backs off" the last window when it would overflow, making the **last seam overlap** much larger than requested (e.g. 11 latent instead of 6 with length/overlap = 20/6) — the most ghost-prone stretch.

**Three content-level remedies (position encoding cannot fix any of these):**

1. **Anchor frames** (`anchor_frames`, optional): one shared "preceding motion / basis" for every window.
   - `driving_only = on` (default, SCAIL-2): anchors are **appended to the driving stream as extra tokens** and the RoPE patch places them **before this window** (`t = N_ref-K .. N_ref-1`) — they pair with no video frame and occupy no video position ⇒ **existing geometry is untouched and the animation is no longer shifted by K frames** (the old implementation prepended them, delaying the whole animation by K latent = 4K pixel frames).
   - `driving_only = off`: anchors are prepended to the driving stream **and** the generation sequence (rigid 1:1 pairing, trimmed after sampling) — closest to the training semantics (each trained chunk starts with the previous chunk's output).
2. **Read-only context** (`head_context_frames`, **unit = latent frames**): each window may attend to a few **real preceding frames** (attended but **not fused** — trimmed before fusion). With `overlap = 0` **every frame has exactly one author** (no averaging of two predictions).
   - Content = those frames of the shared latent `x`, i.e. **the values from the previous iteration** (not this step's fresh output): early in sampling (σ≈1) they are basically noise and only become useful later.
   - ⚠️ **The tail (`tail_context_frames`) has been removed**: it cannot produce a visible effect — its prediction is trimmed before fusion, and those frames are "authored" by the *following* window (you only peek, you cannot change the outcome).
3. **Long-range motion memory** (`memory_frames`, **unit = latent frames**): history driving frames **before** this window, appended as extra driving tokens; their RoPE positions **encode the stride and ordering faithfully** (they are not mistaken for adjacent frames or for "future driving").
   - **The stride is set by `memory_stride`**: `0` (default) = auto — spread the `memory_frames` points so they cover the **whole history** from the clip start to this window (so long clips / late windows still reach the beginning; the stride grows and time resolution gets coarser); `>0` = fixed stride (`4` ≈ one sample per 16 pixel frames).

> ⚠️ **Frame units differ**: `anchor_frames` (IMAGE) is in **pixel frames** (as many as you feed; converted internally via `((T-1)//4)+1`); `head_context_frames` / `memory_frames` / `memory_stride` are all in **latent frames**.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Model that already has a context windows node applied |
> Since 2026-10-15 the controls `driving_only` / `anchor_fixed` / `anchor_image_blank` / `anchor_scope` / `shared_head` / `overlap_scope` / `memory_position` were removed from the panel (values are fixed to the measured profile).

| `enabled` | BOOLEAN | true | Master switch; when off the model passes through |
| `head_context_frames` | INT | 0 | Read-only preceding frames per window (attended, not fused); use with `overlap=0` |
| `memory_stride` | INT | 0 | **Sampling stride for the long-range memory group** (latent frames): `0` = auto (recommended) — spread the `memory_frames` points over the whole history from the clip start to this window; `>0` = fixed stride (`4` ≈ one sample per 16 pixel frames). ⚠️ This parameter **takes over the old `tail_context_frames` slot**, so legacy workflows (whose value there was almost always 0) map onto `0` = auto — semantically compatible |
| `memory_frames` | INT | 0 | Long-range memory: take **N history latent frames** before this window as extra driving tokens (0=off, SCAIL only); stride via `memory_stride`. In auto mode `N` is the **budget** (N=4 = summarise the whole preceding history with 4 points) |
| `anchor_frames` | IMAGE (optional) | unconnected = no anchoring | **Basis frames**: driving/pose frames you want as the shared basis (e.g. the first frames of the video). Feed as many as you want |
| `anchor_mask_frames` | IMAGE (optional, SCAIL-2 only) | unconnected | Colored mask frames with the **same frame numbers** as `anchor_frames` (converted to 28 channels internally, no VAE) |
| `vae` | VAE (required when anchoring) | unconnected | The **same** Wan VAE as the conditioning node; connecting `anchor_frames` without `vae` raises an explicit error |

> The read-only context (`head_context_frames`) and the memory group (`memory_frames` / `memory_stride`) need **no connections at all** — parameters only; the memory group does not even need a VAE.
>
> ⚠️ `anchor_frames` comes from the **driving video's picture**. Inside the generation sequence it is only a placeholder (the model grows the content itself); pinning it down as a "known clean frame" would leak the **driver's identity** into the output (the historical "driver-identity artifact").

#### Wiring

```
[driving/pose video] --ImageFromBatch(start=0, length=9)--> anchor_frames
[same Wan VAE] ------------------------------------------> vae
[driving mask video] --ImageFromBatch(start=0, length=9)--> anchor_mask_frames   (SCAIL-2 only)

[Load Model] → Custom Context Windows / Wan SCAIL Context Windows → Context Window Anchor → KSampler
```

> If it is not placed after a context windows node, the node raises an explicit error instead of silently producing wrong results.

#### Usage tips

1. **Read-only context first**: `overlap = 0` (schedule → `batched`, or `standard_even` to spread the overlap) + `head_context_frames = 2` (latent frames; keep "own frames + head" within 21~23 latent).
2. **Anchor frames**: 5~9 pixel frames from the segment opening (plus matching `anchor_mask_frames` for SCAIL-2); run `driving_only` on and off.
3. **Long-range memory**: `memory_frames = 4 → 8` (leave `memory_stride = 0` = auto).

> **Multi-seed paired A/B**: the only random source inside this node is **fixed-seed noise**, so a given seed + config is **deterministic**. "Some seeds work, some don't" is therefore a **real difference**, not noise ⇒ run several seeds, each with the feature on and off, and compare the success rate instead of judging from a single run.
4. Change one variable at a time (`FreeNoise`, `overlap`, basis source separately).

#### Known limits

- Anchor frames must come from the **same driving/pose video** with matching frame numbers (plus matching masks for SCAIL-2).
- Memory tokens carry **no identity mask** (their 28ch mask is zeros): they provide motion context only, not identity.
- "Anchors before the window" and the memory group rely on the SCAIL RoPE position patch (`_apply_scail_anchor_rope_patch`); in replacement mode or when there are no extra tokens the patch falls back and leaves the original geometry unchanged.
- ⚠️ **The old `driving_only` test was inverted** (`and not _needs_driving_mask(model)`): on SCAIL-2 `driving_only=true` actually ran the **prepend (aligned)** path (anchors prepended to both the driving stream and the generation sequence, 1:1 pairing), while `driving_only=false` went driving-only; on animate2 it was the other way round (hitting the `pose = f_gen − 1` assert). This is now fixed to match the docs. ⇒ **A baseline you captured with `driving_only=true` corresponds to the "prepend" path; after the fix `driving_only=true` is the "tail + placed before the window" path. Both are alignment-safe — just A/B them (turn `driving_only` off to reproduce the old behaviour).**
- The old replacement-style anchoring and `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug` were removed.
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
| `size_mode` | enum | ✅ | `resize_to_first`: resize all images to the first image's dimensions; `filter_same_size`: only load images matching the first image's dimensions; `no_resize`: **only affects the `images_list` output** — the list keeps original sizes (no resize, no size filter), while the batch output cannot hold mixed sizes, so it falls back to `resize_to_first` and prints a notice |
| `skip_first_n` | INT | ✅ | Skip the first N images (default 0) |
| `load_count` | INT | ✅ | Maximum number of images to load (0=no limit, load all) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Same-size image batch `[B,H,W,C]`. Outputs None if path is invalid or no images found |
| `count` | INT | Actual count of loaded images (= batch size). Outputs 0 for error conditions |
| `images_list` | IMAGE (list semantics) | Per-image list output, elements `[1,H,W,C]`; **downstream nodes run once per element** and sizes may differ |

#### Relationship between the batch and the list output

- List semantics follow the existing convention of this repo (same as `image_list_nodes.py`): empty output is always `[None]`
  (a single None element) — it must **not** be `None` (ComfyUI's `execution.py::merge_result_data` calls `extend` on list
  outputs; `extend(None)` raises TypeError) and must **not** be `[]` (downstream `slice_dict` takes `[-1]` on an empty list →
  IndexError). `[None]` behaves downstream exactly like "this wire is not connected".
- `resize_to_first` / `filter_same_size`: the list is a per-image **view** of the batch (zero-copy, no extra pixel memory),
  and its content is identical to `images`.
- `no_resize`: the list keeps original sizes and applies **no size filtering** (images with different sizes appear only in the
  list, never in the batch); the batch falls back to `resize_to_first`. Memory-wise only the images whose size differs from the
  first one keep an extra original copy — same-size originals reuse the batch row (same memory), so the common
  "all images in the folder share one size" case costs **nothing extra**.
- Connecting `images_list` to a plain IMAGE input makes ComfyUI run that downstream node **once per element**
  (list propagation) — this is expected behavior.

#### Error Handling

- Empty folder path / folder not found / no images in folder / skip count exceeds total images → outputs `(None, 0, [None])`, console warning
- Single image read failure → skips that image, continues loading others
- First image read failure → empty outputs as above (existing behavior, unchanged)

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
| `render_video` | IMAGE | ✅ | Guidance video rendered from the camera trajectory (pixel frames); encoded internally by the node's VAE as visual reference for camera control |
| `vae` | VAE | ✅ | VAE used to encode render_video |
| `strength` | FLOAT | ✅ | Control strength (0.0-10.0, default 1.0) |
| `start_percent` | FLOAT | ✅ | Control activation start percentage (0.0-1.0, default 0.0) |
| `end_percent` | FLOAT | ✅ | Control activation end percentage (0.0-1.0, default 1.0) |
| `trim_latent` | INT | ❌ | Number of reference latent frames, connected from WanAnimateToVideoCustom's trim_latent output for temporal alignment |

**Output**: `model` (MODEL) — Control-injected model, ready for direct connection to KSampler.

#### Core Principles

1. **v4 Architecture**: Uses monkey-patch to modify the model's `forward_orig` method, precisely controlling injection timing within the block loop
2. **Reference Offset**: Injection skips reference-frame tokens (same `img_offset` semantics as the official node), preventing reference-frame pollution and control-signal misalignment; supports WanAnimate concat / ref_conv / SCAIL / context-window scenarios
3. **Latent Normalization**: render_video is VAE-encoded internally then normalized via `latent_format.process_in`, matching the already-normalized x scale during sampling, avoiding the control signal biasing toward render content
4. **Step Control**: Uses `start_percent` / `end_percent` to control the activation range of control signals during the denoising process

#### Typical Workflow

```
[WanUni3CLoader] → uni3c_controlnet ─┐
[Guidance video] → render_video ────────┤
[VAE] → vae ────────────────────────┤
                                        ↓
[Model] → WanUni3CApply → model (injected) → [KSampler] → [VAE Decode] → Output
```

> **Note**: Older versions exposed `positive`/`negative` conditioning inputs and `render_mask`; they were no-ops (keys written but never read) and have been removed in v4.

---

### SmartVideoCrop

Smart video cropping driven by SAM3 tracking data: the node keeps the **full frame height**, computes each frame's person centroid, pans the crop window horizontally (EMA smoothing + scene-cut detection), then scales the result to the requested output size.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `frames` | IMAGE | Source frames of the current chunk (B, H, W, C) |
| `track_data` | SAM3_TRACK_DATA | Tracking data from SAM3_VideoTrack (connect to `driving_track_data`) |
| `output_width` / `output_height` | INT | Output size; the crop width is derived as `H × output_width / output_height` |
| `object_ids` | STRING | Comma-separated person IDs to track (e.g. `1` or `1,2`); empty = default ID 1 |
| `smoothing` | FLOAT | Inter-frame EMA smoothing (0 = follow exactly, jittery; 1 = almost static) |
| `mask_fallback` | enum | Behaviour when the mask is empty: `hold_last` keeps the last valid position / `center` falls back to the frame center |
| `scene_cut_threshold` | FLOAT | Scene-cut detection threshold (normalized): a centroid jump larger than this between consecutive masked frames resets the smoothing; 0 = disabled |
| `fallback_center` | FLOAT | Horizontal fallback ratio used by `mask_fallback=center` (0=left, 0.5=center, 1=right) |
| `mask_frames` | IMAGE | Optional second image channel (e.g. a mask video) cropped with the very same window and output in sync |

#### Outputs

- `cropped_frames`: cropped and rescaled frame batch
- `cropped_mask`: synchronized crop of the second channel (empty batch when not connected)
- `debug_info`: JSON with per-frame crop positions

> When the input is narrower than the target it automatically switches to "vertical center crop" (full width, height derived from the aspect ratio), so no content is cut off.
> Cropping runs in **8-frame batches** internally: faster on long sequences while producing bit-identical results to the per-frame implementation.

---

### VideoGridCompose

Composes 1–4 videos/images into a single grid video. Everything is **streamed through ffmpeg**, so no frames are ever loaded into memory — safe for long videos.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_paths` | STRING | Multi-line list, one path per line (up to 4, duplicates/empty lines removed). If a line points to an **image**, it is turned into a static video stream with `-loop 1 -t <target duration>` |
| `layout` | enum | Horizontal / Vertical / 2×2 quad (needs 4 inputs) / One-left-N-right / One-top-N-bottom (both need ≥3 inputs); falls back to horizontal when there are not enough inputs |
| `audio_source` | enum | None, or the audio track of input 1–4 |
| `match_method` | enum | Stretch to fill / Keep aspect ratio (equal scale + black bars) / Uniform to max / Uniform to min |
| `fps` | FLOAT | Output frame rate; 0 = use the highest input frame rate |
| `crf` | INT | x264 quality (lower is better, 18–28 recommended) |
| `duration_mode` | enum | Duration reference: auto (shortest) / auto (longest) / video 1–4. Longer clips are trimmed and shorter ones are freeze-padded; audio is aligned with atrim/apad |
| `output_path_prefix` | STRING | Output path prefix, relative to ComfyUI's output directory |

Returns `video_path` (STRING); the file is written to `ComfyUI/output/<output_path_prefix>_0001.mp4` (auto-incremented to avoid overwriting).

---

### BatchImageReplace

Replaces a contiguous run of frames in the main batch, starting at `start_index` for `replace_count` frames. The replacement source (`replace_images`) may differ in length from the requirement; two modes decide what happens.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_index` | INT | Position in the main batch where replacement starts (0-based) |
| `replace_count` | INT | How many frames to replace; `-1` = replace from start_index to the end; `0` = no replacement (pass through) |
| `overflow_mode` | enum | When the source has **more** frames than needed: `take_front` keeps the first N / `take_back` keeps the last N |
| `underflow_mode` | enum | When the source has **fewer** frames than needed: `as_is` replaces only as many as the source actually has / `repeat_begin` pads the front with the first image / `repeat_end` keeps all in original order and pads the end with the last image |
| `images` | IMAGE | Main image batch (empty output when not connected) |
| `replace_images` | IMAGE | Replacement source (main batch is passed through when not connected) |

#### underflow semantics (3 source images, 4 frames to replace)

| Mode | Replacement result |
|------|--------------------|
| `as_is` | Only 3 frames are replaced (the 4th keeps its original content) |
| `repeat_begin` | `0,0,1,2` (first image repeated at the front) |
| `repeat_end` | `0,1,2,2` (all kept in order, last image repeated at the end) |

> `repeat_end` was fixed in v1.8.0: the previous implementation dropped the middle frames and produced `0,2,2,2`.

#### Notes

- Replacement images are automatically resized to the main batch's resolution when they differ
- `start_index` beyond the batch length prints a warning and returns the main batch unchanged

---

### ImageListSplitOutputs

Splits an **image list** into N independent outputs: the item at `start_index` goes to output 1, `start_index + 1` to output 2, and so on.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_count` | INT | Number of outputs (1-16, default 2). **The output slots on the canvas are added/removed live with this value** — only the configured number is shown |
| `start_index` | INT | Index of the first item to output (0 = first item) |
| `images` | IMAGE | Image list input (optional). When not connected: no error, no output |

#### Output rules

| Situation | Result |
|-----------|--------|
| 5 items, `start_index=1`, `output_count=8` | Outputs 1-4 carry items 2-5; outputs 5-8 are empty (`None` = same as an unconnected wire) |
| `start_index` >= list length | All outputs are empty (a warning is printed) |
| `images` not connected / empty list | All outputs are empty, **no error** |
| `output_count` out of 1-16 | Clamped to 1 / 16 |

#### Empty outputs and list semantics (important)

- **An empty channel returns `None`, which behaves exactly like "this wire is not connected"**:
  - if the downstream declares that input as **optional**, it falls back to its Python default and the node
    **runs normally**, treating it as "no such input" (e.g. `ImageBatchConcat` passes the other channel through,
    `BatchImageReplace` / `MaskCompositeRef` just ignore it) — **other inputs are not affected**;
  - if the downstream declares it as **required**, it errors inside that node with a `NoneType`-style message
    (the runtime equivalent of the submit-time `Required input is missing`, just reported at execution time).
- Do **not** use an empty list `[]` (ComfyUI's `slice_dict` indexes `[-1]` on an empty list → IndexError), and do
  **not** use `ExecutionBlocker(None)`: it makes the downstream node skip execution **entirely** and silently drags
  every other input of that node (and the whole branch below it) down with it, with no UI hint at all.
- The node uses **list semantics** (`INPUT_IS_LIST`): it receives the whole list in one execution and
  **items may have different sizes** (a plain batch requires uniform sizes). A plain IMAGE batch fed into it is
  split into single-frame items automatically (batch → list); 3D `[H,W,C]` items are promoted to `[1,H,W,C]`.
- Each output carries a **single image** (`[1,H,W,C]`), so downstream nodes run exactly once.
- The backend declares 16 outputs only to satisfy ComfyUI's prompt type validation; **the visible slot count is
  really added/removed by the frontend (`js/image_list_nodes.js`, `addOutput`/`removeOutput`)**, so the node never
  turns into a long 16-slot strip.

---

### ImageListReplaceAt

Overwrites a run of the target image list starting at `start_index`: item `start_index + i` is replaced wholesale
by item `i` of the replacement list.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_index` | INT | Index where replacement starts (0 = first item) |
| `images` | IMAGE | Target image list (optional). When not connected: no error, no output |
| `replace_images` | IMAGE | Replacement list (optional). When not connected the target list is passed through |

#### Replacement rules

- Replacement count = `min(len(replace_images), len(images) - start_index)` ⇒ **never exceeds the original list length**
  - e.g. 5 target items, `start_index=2`, 4 replacement items → only items 3-5 are replaced (the 4th replacement is dropped)
  - e.g. 5 target items, `start_index=2`, 2 replacement items → items 3-4 are replaced, item 5 stays
- `start_index` >= target list length → prints a warning and returns the target list unchanged
- **No resizing / padding / extension**: list items may have different sizes (this is the biggest difference
  from batch replacement)

#### Difference from [BatchImageReplace](#batchimagereplace)

| | ImageListReplaceAt | BatchImageReplace |
|---|---|---|
| Data unit | List items (sizes may differ) | Same-size images inside one batch |
| Replacement length | Decided by the replacement list, **never exceeds the original length** | Set by `replace_count` (-1 = to the end) |
| Count mismatch | Extra replacement items are dropped | `overflow_mode` / `underflow_mode` strategies |
| Size mismatch | **No resizing**, whole items are swapped | Auto-resized to the main batch size |
| Output | A list (downstream runs per element) | A single batch |

---

### ImageListToBatch

Converts an image list with **mixed sizes** into a single-size batch (for nodes that only accept batches).
It **never crops and never stretches**: the missing area is padded on a single axis (pure black by default,
or **alpha=0** in `RGBA` mode), so every image keeps all of its content.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_mode` | enum | `RGB` (**default**) = 3 channels, black padding, **most compatible** (VL/vision encoders, VAE, save/preview all work); `RGBA` = 4 channels, `alpha=0` padding (real transparency, for compositing — but **3-channel-only consumers will fail**, see Notes) |
| `images` | IMAGE | Image list input (optional, list semantics). When not connected: no error, no output |

Output `images` (batch `[B, Rh, Rw, C]`, C = 3 or 4).

#### Rules

1. **Reference item** = the item with the largest `min(w, h)` (= largest "area × squareness", i.e. the
   "biggest and squarest" one); ties go to the squarest (smaller long side).
   That item's ratio becomes the output ratio and **its own size becomes the output size**.
   - e.g. `[1024×1024, 1920×1080, 800×1200]` → short sides 1024 / **1080** / 800 → reference = `1920×1080`
   - e.g. `[4000×1000 (extreme strip), 1024×1024]` → short sides 1000 / **1024** → reference = `1024×1024`
     (an extreme strip does **not** blow the canvas up to 4000 wide, keeping batch memory/time sane)
2. The reference item gets **zero resampling** (pixel-for-pixel identical; in `RGBA` mode only an alpha channel is added).
3. Every other item: **wider** than the reference → pad **top/bottom only**; **taller** → pad **left/right only**
   (centered, split evenly). Same ratio → no padding. The limiting axis uses its exact size, so an item is
   **never** padded on all four sides.
4. Padding follows `output_mode`: `RGB` → pure black (`0,0,0`); `RGBA` → `alpha=0` outside and `alpha=1` inside
   (a source alpha channel is preserved); 3-channel input gets an alpha channel in `RGBA` mode.
5. Items already at the output size **skip resampling entirely** (no numpy/PIL round-trip).

#### Example (reference = 1920×1080, 16:9)

| Input | Content size in the output | Padding (black by default / alpha=0 in RGBA mode) |
|-------|----------------------------|---------------------------------------------------|
| 1920×1080 (reference) | 1920×1080 | none (zero resampling) |
| 1024×1024 (taller) | 1080×1080 | 420 left + 420 right |
| 2000×1000 (wider) | 1920×960 | 60 top + 60 bottom |

#### Log

Every run prints the reference selection, the output size and the channel mode:

```
[ImageListToBatch] 基准候选: #0 1920×1080 min=1080 | #1 1024×1024 min=1024 | #2 800×1200 min=800  -> 选定 #0
[ImageListToBatch] 输出尺寸 = 1920×1080（基准 #0，比例 16:9），共 3 张，通道 = RGB(3)
```

When the output size is not a multiple of 8 an extra hint is printed (VAE downsampling is 8/16, so a size
alignment is required before VAE encoding).

#### Notes

- **Why the default is RGB**: 4-channel (RGBA) images hit consumers that hard-code 3 channels. A typical example is
  the Qwen VL vision tower: `Qwen35VisionPatchEmbed.forward` → `x.view(-1, 3, 2, 16, 16)` (`in_channels=3`).
  With 4 channels the patch count grows by 1/3 (`8·H·W/1536 = 4608` for 1024×864, while the position embeddings
  are `6·H·W/1536 = 3456`), failing with
  `RuntimeError: The size of tensor a (4608) must match the size of tensor b (3456)`.
  So keep the default `RGB` when feeding VL text encoders / VAE / most nodes.
- Switch to `RGBA` only when you really need transparent padding (compositing, saving PNG with transparency).
  If you then need to feed a 3-channel-only node, insert the core node **`Split Image with Alpha`**:
  its IMAGE output is `i[:,:,:3]` and its MASK output is `1 - alpha` (i.e. the padded area = 1).
- The output size comes from the **reference item's own size**, so it may not be a multiple of 8/16
  (e.g. `876×1024`). Align it before VAE encoding — the node prints a hint in that case.
- Difference from [ImageBatchResize](#image-batch-resize): that one scales to **a size you specify**
  (optionally cropping/stretching); this one **picks the reference from the list itself** and pads on a single axis
  without ever losing content.
- Internally it does **not** "pad a big canvas first and then scale everything"; it scales the content to its exact
  size inside the final canvas and pastes it centered. Mathematically equivalent, but the padding never bleeds into
  the content edges through LANCZOS (no dark halo).

---

### In-step Rolling
> Used together with the context windows node: later windows see what the earlier windows produced **in the same denoise step**, which reduces ghosting / soft motion at the seams.

**Wiring**: `context windows node → In-step Rolling → sampler`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `enabled` | on | off = exactly like not installing this node |
| `roll_beta` | 0.45 | Commit strength (main knob): 0 = passthrough, 1 = fully adopt the previous window’s current-step result; 0.45~0.50 measured best |
| `roll_beta_inherited` | 0.6 | Separate strength for read-only preceding frames; -1 = follow `roll_beta` |

> Removed controls (fixed): `roll_scope=both`, `roll_beta_curve=late`. One log line per denoise step; use `--verbose` for details.

### Anchor + In-step Rolling (minimal)
> A **single-switch** node that chains “Context Window Anchor” and “In-step Rolling” with the measured baseline profile — for people who just want it to work.

**Wiring**: `context windows node → Anchor + In-step Rolling → sampler`; optional inputs `anchor_frames` / `anchor_mask_frames` / `vae` (same as the anchor node).

| Setting | Value | Notes |
|---------|-------|-------|
| `enabled` | on | The only setting; off = both features off (and the anchor pool is cleared) |

The fixed profile lives in `FIXED_ANCHOR` / `FIXED_ROLL` at the top of `context_anchor_rolling.py`. To tune those values, use the two dedicated nodes instead (they keep all controls).



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

## Changelog

### v1.9.0 (2026-10-16)

**Added**
- **Detect Edges (Canny) Chunked** (`CannyChunked`, category `image`): chunked Canny edge detection that fixes the official `Detect Edges (Canny)` hanging (with no error and no progress) on 200+ frame batches.

  Root cause (measured): the official node hands the **whole batch** to `kornia.filters.canny`, whose hysteresis stage is a Python-level `while` loop that grows strong edges by one pixel per iteration (~96 iterations at 512², ~190 at 1024²). Each iteration's `F.conv2d(edges, hysteresis_kernels)` returns `(B, 8, H, W)` — 8× the input volume — and the loop condition is **shared by the whole batch** (one unconverged frame forces all 200 frames to pay another iteration). That makes the peak for 200 frames @1024² roughly 29.7GB; on a 24GB card Windows WDDM silently spills to system RAM, which looks like a freeze — and a node in execution cannot be cancelled.

  This node calls **the exact same line** per chunk (only "whole batch" → "chunk by chunk"):

  | Scenario (RTX 4090, real frames) | Official Canny | CannyChunked |
  |---|---|---|
  | 200 frames @1024² peak VRAM | ~29.7GB (over 24GB → spill/freeze) | **1.31GB** |
  | 200 frames @1024² runtime | frozen (no response for tens of minutes) | **~5.7s** |
  | Interrupt | impossible inside the node | checked per chunk, cancellable anytime |
  | Progress | none | front-end progress bar + console tqdm |

  Parameters and equivalence: `low_threshold` / `high_threshold` keep the official names and semantics (drop-in replacement); a single chunk (`chunk_size` ≥ frame count) is **bit-exact** vs official; real-image chunking measured **0 differing pixels** (256²/512²/1024²) and the CPU path is bit-exact for any chunk size. `auto_chunk` shrinks the chunk size according to `model_management.get_free_memory()` (never grows it), and `use_gpu=off` runs entirely on CPU (zero VRAM, for when a big model occupies the card).
  Equivalence note: CUDA conv picks a cuDNN algorithm by batch shape, so on worst-case input (uniform random noise) chunking can differ from the whole batch by up to ~0.003% of pixels — **the official node deviates in exactly the same way when you split the batch yourself**, so it is not introduced by this node.
  The self-test `md/canny_chunked_selftest.py` (40 assertions) runs without starting ComfyUI.

#### Also added in this release (since 2026-10-15)

**Added**
- **Anchor + In-step Rolling (minimal)**: a one-switch node that chains “Context Window Anchor” and “In-step Rolling” with the measured baseline profile (read-only head 2 / memory 4 / commit strength 0.45 / inherited 0.6 / `curve=late` …). All fixed values live in `FIXED_ANCHOR` / `FIXED_ROLL` at the top of `context_anchor_rolling.py`.
- **In-step Rolling** (`Wan SCAIL-2 Rolling Windows`): later windows see what the earlier windows produced **in the same denoise step**.

**Slimmer panels**
- Anchor **8 → 5 controls**: removed `anchor_scope` / `drive_memory` / `anchor_slot` (fixed to `fresh_only` / `off` / `aligned`).
  ⚠️ **Compatibility note**: the default of `anchor_slot` changed from `context` to **`aligned`** — re-save that node in old workflows (or check the values).
- Rolling **5 → 3 controls**: removed `roll_scope` / `roll_beta_curve` (fixed to `both` / `late`).

**Other**
- The rolling commit formula is back to the re-noise form (`content_delta` measured worse and was removed):
  `x = (1-β)·x + β·[(1-σ)·prev-window x0 + σ·ε]` (ε is seed-deterministic ⇒ both windows share the same noise).
- **Quieter logs**: per-step / per-window messages (`SCAILContext sam_latents`, window bounds, FreeNoise, …) moved to `debug` (~12% fewer log lines in a real run). Start ComfyUI with `--verbose` to see them all again.
- **GBK-safe log/print text** (avoids `UnicodeEncodeError` on cp936 consoles); all `print` calls replaced with `logging.*`; `FolderImageLoader` per-file warnings are aggregated into one line.
- Fixed the misleading “last-seam overlap” warning (it no longer suggests `standard_even` when you already use it).

#### v1.8.1 content (never released separately; included here)

**Fixed**
- **`anchor_fixed` never worked (real bug)**: the "previous-step x0" cache was stored on the **window object**, and window objects are **rebuilt every sampling step** (`Custom_context.get_context_windows` allocates new ones each step) ⇒ the cache was lost every step and the code always fell back to the clean anchor frame. Two consequences: (1) `anchor_fixed` on/off produced **bit-identical** results; (2) late in sampling (σ→0) the anchor became almost a **clean driver frame** that the model continued from ⇒ the **"driver-identity artifact"** (the one v1.8.0 claimed to have fixed).
  The cache now lives on the **handler** (shared for the whole sampling run, surviving step rebuilds) and prefers the CFG-positive output. New self-test covers "a rebuilt window still reads the previous step's cache".
- **The RoPE position patch could silently die**: it is a process-level monkey patch (`SCAILWanModel.rope_encode` class attribute). If comfy core hot-reloads and restores the original, the `_SCAIL_ANCHOR_ROPE_PATCH_APPLIED` flag stays `True` ⇒ the patch is **never re-installed** and **nothing is reported**, so the extra tokens are treated as "future driving" (wrong positions, worse results). The patch is now **self-healing**: before every use it checks whether the current function is ours, reinstalls if not, and logs "applied / already in place / was lost, reinstalled".
- **GBK safety**: two **log strings** in `context_window_anchor.py` contained `⇒`(U+21D2) / `−`(U+2212), which raise `UnicodeEncodeError` on a cp936 console (a known trap in this repo). Replaced with ASCII equivalents; the self-test script was made GBK-safe as well.

**Changed**
- **Removed `tail_context_frames`** (read-only tail context): it cannot produce a visible effect — its prediction is trimmed before fusion and those frames are authored by the following window. The internals (`halo_tail` / `tail_extra_indices` / pre-fusion trimming) are kept but fixed at 0.
- **Added `memory_stride` (`0` = auto)**, **taking over the old `tail_context_frames` parameter slot** (so legacy workflow widget values do not shift; that slot was almost always 0 and 0 happens to mean auto — semantically compatible). `0` = spread the `memory_frames` points over the whole history from the clip start to this window (fixes "long clips / late windows cannot reach the beginning"); `>0` = fixed stride.
- **Decoupled `memory_frames` from `head_context_frames`**: the memory group's sampling baseline is now the window's own start, so enabling head context no longer shifts its sampling positions.
- **Docs now follow the code**: corrected stale entries such as "`c_concat` = 1" (belonging to withdrawn experiments) and documented the frame units (`anchor_frames` = pixel frames; `head_context_frames` / `memory_frames` / `memory_stride` = latent frames).

**Self-test**
- `md/context_window_anchor_selftest.py`: **19/19 pass** (new/updated: cross-step x0 cache, auto & fixed memory stride, memory/head decoupling, RoPE patch hot-reload self-healing, interface contains `memory_stride` and no `tail_context_frames`, **`memory_position` (tail/head + driving-stream/mask sync)**, **RoPE patch ownership & per-step self-heal + per-step global refresh from cfg (immune to node caching)**, **driving-stream vs driving-mask token-count consistency self-check**); the script is now GBK-safe and runs on a cp936 console.

### 2026-10-10 fixes & additions (silent failures first)

| Item | What |
|---|---|
| **RoPE patch contention** | Both the anchor node and `WanSCAIL_MultiRef` assign to `SCAILWanModel.rope_encode` (plain assignment ⇒ last writer wins). If our patch gets displaced, extra tokens are read as "future driving" ⇒ **wrong positions, zero errors** (and whether it happens depends on node execution order / cache hits ⇒ **intermittent**). Now: ① MultiRef skips the SCAIL-level overwrite when the anchor patch is present; ② the anchor side **checks & reinstalls every sampling step**; ③ the warning prints **who owned the displaced patch** |
| **Module globals + node caching** | Four switches (`anchor_scope` / `shared_head` / `overlap_scope` / `memory_position`) previously lived **only in module globals**; when ComfyUI caches the node, `reset_pools` never runs ⇒ sampling reads the **previous run's leftovers**. Now they are part of `AnchorConfig` and **refreshed from cfg at the start of every sampling step** (pool cache is cleared when the source object changes) |
| **`memory_position` default → `head`** | `tail` is only correct while the RoPE patch is alive; `head` is **immune to patch loss**. With a healthy patch the two are **bit-identical** |
| **Driving-stream/mask consistency self-check** | SCAIL adds the driving mask token-by-token ⇒ token counts must match; a mismatch now **warns** (it used to be a silent misalignment) |
| **FreeNoise stride assumption** | Overwrite blocks now use the **real window starts** (the old code assumed a uniform stride ⇒ blocks did not coincide with the true window boundaries under `standard_even`) |
| **Mask log de-dup key** | `sam_latents` / `ref_mask_latents` (both `t_dim=2`) used to **swallow each other's logs**; the key now names the stream |
| **`roll_beta_inherited`** (rolling node) | Splits the commit strength by frame ownership: **only "read-only preceding frames" are made single-author; the window's own frames are untouched** (default -1 = follow `roll_beta` = bit-identical regression) |
| ~~**`roll_w_content` / `roll_w_noise`**~~ **removed (2026-10-11 cleanup)** | Measured: `w_noise<1` degrades the seam hop from 11.4px to 31~40px with ground loss (ratio 0.47); `w_content` showed no monotone effect => both removed, the commit formula is fixed to the legacy re-noise form, widgets 11 -> 5 |
| ~~**`roll_noise_seed`**~~ **hidden (2026-10-11)** | epsilon (commit noise seed) is now **fixed to 0 internally**: any value is statistically equivalent, and changing it was measured to swing the seam hop between 11.4 and 23.3 px (a lottery) => no longer exposed. **Anchor node slimmed too**: removed `anchor_fixed` (node comment: "on/off are fully equivalent, user confirmed") and `shared_head` (documented as a small lever); anchor placeholder noise seed default 0x5EED -> 0 |
| **Anchor node cleanup (2026-10-11)** | 4 widgets removed and their values frozen to the measured-best / your habit: `driving_only` -> **False** (you measured True as clearly worse; anchors are prepended and frame-paired), `anchor_image_blank` -> **True** (anchor slots in the to-be-generated sequence hold noise only, avoiding source-person leakage; same-seed A/B: 19.9 -> 15.7px), `anchor_fixed` (node comment: "on/off fully equivalent, user-confirmed"), `shared_head` (documented as a small lever) | anchor widgets 12 -> **8** |
| **〔rolled back — see "Ordering normalization rolled back" above〕 Anchor ordering normalized (2026-10-11)** | Unified to **`[anchor][memory][window]`** with the **stream / mask / RoPE blocks all in the same order** (guarded by a selftest assertion): anchor = the contiguous `pre` block just before the window; memory = the sparser `mem` block further back. `memory_position` widget removed (frozen to head); `prehistory` is now K (the anchor count) - previously 0, which let the RoPE treat the anchor as the *start of the sequence* (a silent-misalignment risk) | `context_window_anchor.py` |
| **New-slot safety gate + observability (2026-10-11)** | (1) the x-segment split of `shifted`/`aligned` now **passes `steps_h`/`steps_w` through** (previously only the window part got them, so the anchor/memory parts used default grids); (2) when a slot falls back to `context` (because the caller passed `steps_t`, or the window has no anchor) a **explicit log line** is emitted (it used to be silent, so an experiment could silently be a no-op) | `context_window_anchor.py` |
| **Crop offset missed the x-side memory blanks (2026-10-11 fix)** | With `x_prefix=anchor_memory` the x side carries M extra memory-blank slots, but the pre-fusion crop only accounted for prefix + anchor + head ⇒ every window output was **shifted by M frames** (symptom: scrambled, repeating motion; log `窗口N 裁剪后长度 21 != 窗口帧数 17`). Fix: new single source `Custom_context._x_prefix_mem_slots(window)` (reusing `_x_memory_slots`) added to `front_trim`, and the divergence-audit position mapping corrected (`base = anchor + M`) so the ghosting audit works again. Regression case 24 | `Custom_context.py`、`context_window_anchor.py`、`md/context_window_anchor_selftest.py` |
| **FreeNoise implementation (2026-10-11 audit: code-level facts)** | Same semantics as comfy's official `apply_freenoise` (noise pool / randperm / overwrite positions), plus two official fixes: (1) it uses the **real window starts** (upstream assumes a uniform step ⇒ overwrite blocks misalign under non-uniform schedules such as `even_standard`; guarded by selftest case 7); (2) `dynamic_seed` is pinned to `seed` (upstream's leftover `seed + _step*997` is a no-op there). **Scope: each `sample()` shuffles the initial noise once only** (the wrapper is on `SAMPLER_SAMPLE`, not per denoising step) ⇒ inherently limited effect on single-frame seam jumps. Known limits: **no multi-modality noise support** (upstream scales C/OV per modality); the wrapper key is identical to upstream's (when both are present only the last one applies, silently) | `Custom_context.py`, `md/context_windows_schedule_selftest.py` |
| **`anchor_slot` (2026-10-11, new widget)** | Two anchor time-code semantics: `context` (default) = physical `[memory][anchor][window]` + `prehistory=0`; `leading` = physical `[anchor][memory][window]` + `prehistory=K`, and the x segment is split **inside the same RoPE function** into `[x-anchor][x-memory][x-window]` (anchor = `ref−K..ref−1`, memory by stride, window = `ref..`). Never touches the `WanModel` base class ⇒ no impact on other models. Switch + re-run, no restart. Tests: case 3 (widget order/defaults), case 13b (`prehistory=K`, total token count unchanged, base calls ≥ 5). ❓**Not yet measured**: `leading` vs `context` same-seed comparison (2 seeds each) | `context_window_anchor.py`, `md/context_window_anchor_selftest.py` |
| **`drive_memory` widget (2026-10-11)** | Whether the 16ch driving stream (`pose_video_latent`) also receives memory frames: default `off` = old behaviour (no 20ch→16ch trim ⇒ that stream gets no memory frames) / `on` = fill them in. Switch + re-run, no restart; env var `CWA_DRIVE_MEMORY=0` as a fallback. **Measured (1 sample each — direction only, magnitude not reliable)**: bad seed off 23.9 / on 28.8; good seed on 17.3 / off 18.0. `x_prefix=none` is now marked "experimental / not recommended" | `context_window_anchor.py`, `md/context_window_anchor_selftest.py` |
| **Ordering normalization rolled back (2026-10-11)** | Observed (1 sample per config): after unifying the physical order to `[anchor][memory][window]` + `prehistory=K`, max jumped from 12.4–18.1 (median 13.8, old) to **22.4px**; with FreeNoise on, 11.4 → 17.4. **Rolled back** to `[memory][anchor][window]` + `prehistory=0`. ⚠️ The "K out of phase is the main cause" explanation is an **inference, not verified**; if this order is revisited, the x segment must be synchronised at the same time (see `anchor_slot`) | `context_window_anchor.py` |
| **`x_prefix` ordering bug (2026-10-11 fix)** | Root cause of the `anchor_memory` crash (code-level): condition streams (`c_concat` …) are injected **first**, the x side (`inject_anchor_x`) **later**, and both computed the "memory blank slot" count themselves ⇒ the streams read the not-yet-written `window._x_prefix_mem_count` (=0) ⇒ only +K, while the x side added +K+M ⇒ off by M=4 ⇒ comfy crashed (`Expected size 25 but got size 21`). Fix: single source of truth **`_x_memory_slots(window)`** (derived from `original_indices`, order-independent) used in all four places; logs now print the actual inserted count. Regression test 23 | `context_window_anchor.py`, `md/context_window_anchor_selftest.py` |
| **B + `x_prefix` (2026-10-11)** | (1) **B**: `_match_channels` now supports the **20->16 down-conversion** => `pose_video_latent` (16ch) previously could **never receive the memory frames** (the 20ch memory was dropped when matching 16ch); the two pose streams now have identical structure. (2) **`x_prefix`** (new widget, default `anchor_only` = zero regression): `anchor_memory` => x mirrors the driving stream `[anchor][memory-blank][window]` (mask zero-padded in sync); `none` => no prefix on x. Known limitation: x-side extra tokens still get the base-RoPE sequential numbering, so an x-side time-code patch would be the next step if this proves useful | `context_window_anchor.py` |
| **`md/frame_metrics.py`** (local tool, `md/` is gitignored) | Frame-level metrics: **teleport** (max px / pop ratio / landing author-band / ground-crossing ratio) + **residual** (ghosting) + **per-band** (sharpness/saturation); uses **content hashes** to detect the same segment saved by several preview nodes and align the offset => `--seedtab` prints a same-seed parameter table |

### Errata & withdrawn experiments (do not re-introduce historical records as "bugs")

| Historical record | Status | Why it was withdrawn |
|---|---|---|
| `c_concat` filled with **1** on anchor frames ("clean/known") | ❌ withdrawn | Belongs to the "replacement / pinned anchoring" era. Anchors come from the **driving video**; pinning them as known clean frames leaks the driver's identity into the picture. Current semantics: anchors are ordinary frames (`concat_mask`=1 / `c_concat`=0), and `c_concat` is derived by the model from `1 - mask` anyway |
| Replacement-style anchoring and `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug` | ❌ removed | Replacement breaks the rigid 1:1 pairing and feeds synthetic intermediate states; the interface must stay minimal |
| Read-only tail context (`tail_context_frames`) | ❌ removed | Its prediction is trimmed and those frames belong to the following window ⇒ no visible effect |
| Fixed 4-latent-stride "long-range memory" | 🔄 now auto by default | A fixed stride cannot reach the beginning of long clips, defeating "long-range" (a fixed stride is still available with `memory_stride > 0`) |

### v1.8.0

**Added**
- `WanAnimate2ToVideoCustom`: WanAnimate2 (official second-gen architecture) video generation node
- `Context Window Anchor (unified-basis anchoring)` (**experimental / test-only — effectiveness and side effects not fully validated**): **minimal interface** — required = `model` / `enabled`; optional = `anchor_frames`(IMAGE) / `anchor_mask_frames`(IMAGE, SCAIL-2 only) / `vae`(VAE, required when anchoring). Anchor frames are treated as **known (clean-history) frames** and spliced into the driving stream, the driving mask and the generation sequence at once (after the reference prefix; trimmed and never fused); **how many you feed is how many are used**; without `anchor_frames` it is exactly equivalent to not installing the node.
  - **Anchor semantics (this round)**: anchors are injected the way your manual "prepend the video opening to a segment" does — the driving stream and driving mask get them prepended, and the copies injected into the generation sequence **follow their own generation trajectory** (step 1: σ=1 pure noise; afterwards re-noise the x0 the model predicted for those frames) and are treated as **ordinary frames** (input mask=1, history mask=0). `mode` / replacement branches were **removed**; the node now has a single behaviour.
  - **Also fixed (run scoping + new options)**: runtime gating on every injection path (bypassing/disabling the anchor node is now truly zero-effect; fixes the x(25) vs ref_mask(27) crash caused by stale module state leaking into a bypassed run); new `driving_only` (default true) and `head_context_frames` (default 0).
  - **Also fixed this round**: (1) this removes the **driver-identity artifact** caused by pinning the **driver's footage** at the window head (at σ→0 the anchor was almost a clean driver frame, so the model continued from it); (2) the **mask encoder now supports any frame count** (4/8 frames used to crash: `_extract_mask_to_28ch`'s temporal grouping only holds for 1+4n, 7×7=49 ≠ 28); (3) added a **pre-sampling encode self-check** (x/pose/mask) and an explicit "SCAIL-2 requires `anchor_mask_frames`" error.
  - **~~Key fix: `c_concat` = 1~~ (void — see "Errata & withdrawn experiments" at the end of the changelog)**: that entry belongs to the **withdrawn "replacement / pinned anchoring"** semantics. The current implementation treats anchors as **ordinary frames to be generated**: `concat_mask` / `denoise_mask` = **1**, `c_concat` = **0**. Note that the model derives `c_concat` itself from `1 - mask` (`comfy/model_base.py::WAN21_SCAIL2.concat_cond`), so the node only needs to insert the matching number of frames. **Removed**: replacement-style anchoring and `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug`.
  - **Companion fixes**: `Custom_context.py` gained `cond_indices_for_window()`, which aligns the conditioning streams' coordinate system from the cond-vs-x length (`cond_len == x_len - 1` → one-frame reference-slot offset, matching the model's `rope_encode_pose(t_start=1)`), fixing a possible one-frame pose offset for prefixed animate2 windows; the `pose_video_latent` slice inside `Wan SCAIL Context Windows`, which always took the first N frames of the chunk, now follows the window's real frame indices (including anchor/halo frames) with an out-of-range warning.
  - **Not fully validated**: the mechanism rests on the assumption that the alignment basis is locked by the clip head; side effects (image quality / identity consistency / sampling stability / frame budget) have not been systematically evaluated. Always A/B it (see Experiment tips) before relying on it, and do not depend on it in production workflows.

**Removed**
- `WanSCAILSparseAttention`: it required dense O(S²) attention masks (TB-scale at 2600 frames) which blew up memory, and showed no measurable benefit on SCAIL. Workflows still referencing it will report a missing node — just delete that node

**Fixed**
- `SDPoseEstimateYawSimple` / `SDPoseEstimateYawAdvanced`: crashed with `IndexError` on short segments / multi-shot footage when the grouping was empty (now falls back to global reference values)
- `BatchImageReplace`: corrected the `repeat_end` padding logic (3 images for 4 slots: `0,2,2,2` → `0,1,2,2`); no more `TypeError` when `images` is not connected
- `ImageSequenceToVideo`: the ffconcat list and audio temp files are now cleaned up on failure/exception paths as well

**Improvements**
- `InteractiveBatchCrop`: the preview cache now only stores the frames the front end actually requests — ~12.5GB → ~48MB for 2600 frames — with a 2GB total budget; outputs are pre-allocated
- `SmartVideoCrop`: cropping now runs in 8-frame batches (faster on long sequences, identical values)
- `MaskCompositeRef` / `BatchImageReplace` / `ImageBatchResize` and others: per-frame append+concat replaced with pre-allocated outputs (peak memory roughly halved)
- `VideoGridCompose` / `VideoFrameCounter` / `ImageSequenceToVideo`: ffmpeg probing per file reduced from 2 calls to 1
- `GD_BBoxDetect`: new `unload_after_run` switch (off by default = stay resident) that moves the model back to CPU after detection when enabled
- File/folder picker dialogs now run in a worker thread, so they no longer block ComfyUI's event loop

**⚠️ Breaking changes**

1. `WanUni3CApply`: input changed from `render_latent`(LATENT) to `render_video`(IMAGE) + `vae`; `WanUni3CLoader` now reads models from `models/model_patches` instead of `models/controlnet`
2. `WanSCAILToVideoMultiRef`: `ref_encoding_mode` now defaults to `逐帧编码` (per-frame encoding, higher fidelity); switch back to `1+4n批量` manually if you need the old behaviour
3. `Custom Context Windows (Manual)` / `Wan SCAIL-2 Context Windows`: the `causal_window_fix` parameters and logic have been removed
4. The `WanSCAILSparseAttention` node has been removed

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
