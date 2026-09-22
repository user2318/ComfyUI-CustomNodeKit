# ComfyUI Custom Node Kit

A set of custom nodes for ComfyUI, covering video generation, pose processing, interactive image operations, and common workflow utilities.

Current version: **1.8.0** — Added the WanAnimate2ToVideoCustom node (Wan2.1-I2V base + second-gen pose branch K/V injection, multi-reference image support); reworked WanUni3CApply to accept render_video with built-in VAE encoding and reference-frame offset (fixing the render_latent normalization bias); removed the retired WanSCAILSparseAttention node (its dense O(S²) attention masks blew up memory on long sequences and gave no measurable benefit on SCAIL); plus one **experimental / test-only node, Context Window Anchor (unified-basis anchoring)** — its effectiveness and side effects have not been fully validated, so it is kept for testing reference only; without `anchor_frames` connected it is exactly equivalent to not installing it (see [Context Window Anchor](#context-window-anchor-unified-basis-anchoring)).

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
| **Folder Image Loader** | `image` | Loads image batches from a folder sorted by filename. Supports skip, count limits, and size synchronization |
| **Image Batch Concat** | `image` | Concatenates two image batches. Passes through the non-empty input if only one is connected |
| **Image Batch Resize** | `image` | Resizes image batches to specified dimensions. Optional directional cropping to maintain aspect ratio |
| **Batch Image Replace** | `image` | Batch image replacement node. Replaces images at specified positions in a batch by start index and count. Supports overflow modes and underflow mode, with optional replacement source |
| **Batch Frame Replicate** | `image` | Batch frame replication node. Replicates a specified image and inserts copies after it. Supports negative indexing and dual-channel independent input. Used for extending first/last frame duration, keyframe freeze effects |
| **MaskCompositeRef** | `image` | Mask batch compositing node — reference_image preprocessing for WanSCAILToVideo. Supports replacement mode and animation mode, auto-filters fully black mask frames, background compositing |

### Context Tools

| Node | Category | Description |
|------|----------|-------------|
| **Custom Context Windows (Manual)** | `context` | Universal context window scheduling node. Splits long sequences into sliding windows for piecewise processing. Supports reference frame prefix, noise shuffling, multiple scheduling strategies (uniform/static/cyclic/batched), and fusion modes (pyramid/linear/relative) |
| **Wan SCAIL-2 Context Windows** | `context` | SCAIL-2 dedicated context window node. Automatically handles SCAIL-2 specific conditioning fields (ref_mask_28ch, driving_mask_28ch, pose_video_latent). No prefix reference frames needed. Works with WanSCAILToVideoMultiRef |
| **Context Window Anchor (unified-basis anchoring)** | `context` | [Experimental / test-only] Prepends your anchor frames to every window's **driving video (plus SCAIL-2's driving mask)** and puts the same frames — **noised to the current σ**, as **ordinary frames** — at the front of the generation sequence, then trims them after sampling (equivalent to doing "prepend the video opening to a segment, generate, then crop" by hand). Minimal interface: `anchor_frames`(IMAGE) + `vae` (same Wan VAE) + SCAIL-2's `anchor_mask_frames`; **how many you feed is how many are used**. Without `anchor_frames` it is exactly equivalent to not installing the node. Must be placed after a context windows node |

---

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

#### Typical Workflow

```
[Load model] → model ──┐
                        ↓
             Custom Context Windows (Manual)
                        ↓
                model (wrapped) → [KSampler] → [VAE Decode] → Output
```

---

### Context Window Anchor (unified-basis anchoring)
> ⚠️ **Experimental / test-only node (not fully validated)**: effectiveness and side effects (quality, identity consistency, sampling stability) have not been systematically evaluated. A/B it with the **same seed** ("unconnected" vs "connected") before relying on it. **Without `anchor_frames` connected this node is exactly equivalent to not installing it.**

**Problem it addresses**: animate2 / SCAIL-2 have no explicit spatial anchor (no skeleton), and under context windows RoPE is always `0..L-1` per window (no absolute time). Wan-Animate-2 §3.3/§3.4 show the driving↔generated alignment is **rigid and frame-wise** — and nothing guarantees that two windows infer the *same* reference↔driving mapping. So every window re-infers its basis from its own head: a different head → a different result for the same global frame (ghosting / motion mismatch); a hard head → the whole segment breaks structurally.

**What it does**: prepends your anchor frames to every window's **driving video (and, for SCAIL-2, its driving mask)**, and puts the same frames — **noised to the current σ** and treated as **ordinary frames** — at the front of the generation sequence, then trims them (with their predictions) after sampling. This mirrors doing it by hand: "prepend the video opening to a segment, generate, then crop".

- **Why the generation sequence too**: SCAIL-2's driving mask is added **token-wise into the driving stream**, so it must have the **same frame count** as the driving video; and driving↔generated frames are **rigidly 1:1 aligned** (animate2 even asserts `pose = f_gen − 1`) → prepending on the driving side only would misalign them.
- **Why the anchors are noised**: in your manual experiment those frames are "ordinary frames to be denoised"; injecting a *clean* latent gives the model an out-of-distribution "clean frame among noisy frames", which it likely ignores (probably why the earlier version had no effect). So they are injected as `x_t = (1−σ)·anchor + σ·noise` and marked as ordinary frames (input mask = 1 / history mask = 0).
- **How many you feed is how many are used** (no count parameter; 1~9 pixel frames ≈ 1~3 latent frames).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Model that already has a context windows node applied |
| `enabled` | BOOLEAN | true | Master switch; when off the model passes through |
| `anchor_frames` | IMAGE (optional) | unconnected = off | **Basis frames**: driving/pose frames you want as the shared basis (e.g. the first frames of the video). Feed as many as you want |
| `anchor_mask_frames` | IMAGE (optional, SCAIL-2 only) | unconnected | Colored mask frames with the **same frame numbers** as `anchor_frames` (converted to 28 channels internally, no VAE). Recommended, otherwise the driving mask and driving video can drift apart |
| `vae` | VAE (required when anchoring) | unconnected | The **same** Wan VAE as the conditioning node; connecting `anchor_frames` without `vae` raises an explicit error |

#### Wiring

```
[driving/pose video] --ImageFromBatch(start=0, length=9)--> anchor_frames
[same Wan VAE] ------------------------------------------> vae
[driving mask video] --ImageFromBatch(start=0, length=9)--> anchor_mask_frames   (SCAIL-2 only)

[Load Model] → Custom Context Windows / Wan SCAIL Context Windows → Context Window Anchor → KSampler
```

> If it is not placed after a context windows node, the node raises an explicit error instead of silently producing wrong results.

#### Usage tips

1. **A/B**: same seed, `anchor_frames` unconnected vs connected (5~9 pixel frames); compare the same frame's pose/scale and the window seams. The main criterion stays the **side-by-side video** (many differences are temporal and invisible in stills).
2. **Which frames to use as the basis**: start with the first frames of the whole video; if some windows are still unstable, try the frames whose pose is closest to the reference image, or the frames immediately before that chunk (`ImageFromBatch` is enough).
3. Change one variable at a time (`FreeNoise`, `overlap`, basis source separately).

#### Known limits

- Anchor frames must come from the **same driving/pose video** with matching frame numbers (plus matching masks for SCAIL-2), otherwise driving and masks drift apart.
- The "two-sided halo" implementation (letting a window see neighbouring frames without fusing them) is still kept internally but **not exposed** (default off): it did not help and made results worse in testing. It can be re-enabled on request.
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

## Changelog

### v1.8.0

**Added**
- `WanAnimate2ToVideoCustom`: WanAnimate2 (official second-gen architecture) video generation node
- `Context Window Anchor (unified-basis anchoring)` (**experimental / test-only — effectiveness and side effects not fully validated**): **minimal interface** — required = `model` / `enabled`; optional = `anchor_frames`(IMAGE) / `anchor_mask_frames`(IMAGE, SCAIL-2 only) / `vae`(VAE, required when anchoring). Anchor frames are treated as **known (clean-history) frames** and spliced into the driving stream, the driving mask and the generation sequence at once (after the reference prefix; trimmed and never fused); **how many you feed is how many are used**; without `anchor_frames` it is exactly equivalent to not installing the node.
  - **Anchor semantics (this round)**: anchors are now injected the way your manual "prepend the video opening to a segment" does — the driving stream and driving mask get them prepended, and the copies injected into the generation sequence are **noised to the current σ** (`x_t=(1−σ)·anchor+σ·noise`) and treated as **ordinary frames** (input mask=1 to generate, history mask=0) instead of "clean latent marked as known". `mode` / replacement branches were **removed**; the node now has a single behaviour.
  - **Key fix**: SCAIL-2's `c_concat` (4-channel history mask) is filled with **1** on anchor frames (= clean/known anchor) while input masks (`concat_mask` / `denoise_mask`) get 0 — the previous version had this reversed, which is why anchoring had no effect. **Removed**: replacement-style anchoring and `anchor_targets / anchor_offset / blend / anchor_steps / apply_to_negative / global_pose_frames / global_mask_frames / anchor_source / halo_frames_* / log_debug`.
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
