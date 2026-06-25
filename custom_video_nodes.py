import os
import re
import math
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import folder_paths
import comfy.utils
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image

FFMPEG_PATH = get_ffmpeg_exe()
BASE_DIR = folder_paths.base_path


def resolve_path(path: str) -> str:
    """将输入路径转换为绝对路径（相对于 ComfyUI 根目录）
    
    安全措施：
    - 对空路径进行检测
    - 对路径遍历攻击（如 ../../etc/passwd）进行检测和拦截
    """
    if not path or not path.strip():
        raise ValueError("路径不能为空")
    
    norm_path = os.path.normpath(path.strip())
    
    # 路径遍历攻击检测：检查规范化后的路径是否试图逃逸预期范围
    # 绝对路径：检查是否存在 ./.. 或 /.. 等明显遍历模式
    if os.path.isabs(norm_path):
        # 对绝对路径，检查是否有非法字符（如空字节）
        if '\0' in norm_path:
            raise ValueError(f"路径包含非法字符: {path!r}")
        return norm_path
    else:
        resolved = os.path.normpath(os.path.join(BASE_DIR, norm_path))
        # 检查相对路径是否通过 .. 逃逸到了 BASE_DIR 之外
        # 注意：这只适用于相对路径场景；绝对路径场景由用户自己负责
        common_prefix = os.path.commonpath([resolved, BASE_DIR])
        if common_prefix != BASE_DIR:
            logging.warning(f"[Security] 路径遍历攻击检测已拒绝: {path!r} -> {resolved}")
            raise PermissionError(f"路径越权访问: {path}")
        if '\0' in resolved:
            raise ValueError(f"路径包含非法字符: {path!r}")
        return resolved


def get_video_info(file_path: str) -> Tuple[float, float]:
    """使用 ffmpeg 获取视频时长（秒）和帧率（fps），一次调用完成"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    cmd = [FFMPEG_PATH, "-i", file_path]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        output = result.stderr
    except Exception as e:
        raise RuntimeError(f"无法执行 ffmpeg: {e}")

    # 解析 Duration
    pattern = r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)"
    match = re.search(pattern, output)
    if not match:
        raise ValueError(f"无法从 ffmpeg 输出中解析时长: {output[:200]}")
    h, m, s = match.groups()
    duration = int(h) * 3600 + int(m) * 60 + float(s)

    # 解析视频流帧率（Stream #0:0: Video: ..., 1920x1080, 30 fps, ...）
    fps = 0.0
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", output)
    if fps_match:
        fps = float(fps_match.group(1))

    return duration, fps


def get_video_resolution(file_path: str) -> Tuple[int, int]:
    """使用 ffmpeg 获取视频分辨率 (width, height)"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    cmd = [FFMPEG_PATH, "-i", file_path]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        output = result.stderr
    except Exception as e:
        raise RuntimeError(f"无法执行 ffmpeg: {e}")
    
    # 从视频流解析分辨率
    # 注意：必须避免匹配 Stream #[0:0x1] 中的十六进制 0x1
    # 要找的是类似 ", 1080x1440" 的真实分辨率
    for line in output.split('\n'):
        if 'Video:' in line and 'Stream' in line:
            match = re.search(r',\s*(\d+)x(\d+)', line)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    
    raise ValueError(f"无法从 ffmpeg 输出中解析分辨率: {output[:300]}")


def get_media_duration(file_path: str) -> float:
    """兼容封装：仅获取视频时长（秒），内部调用 get_video_info"""
    duration, _ = get_video_info(file_path)
    return duration


class VideoFrameCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "tooltip": "视频文件的路径。Path to the video file."}),
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 120.0, "step": 0.01, "tooltip": "目标帧率，设为0则自动使用视频自身的帧率。Target frame rate, set to 0 to use the video's original frame rate."}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "count_frames"
    CATEGORY = "video"
    DESCRIPTION = "获取视频的帧数和帧率信息。Get the frame count and frame rate information of a video."

    def count_frames(self, video_path: str, fps: float) -> Tuple[int, float]:
        video_path = resolve_path(video_path)
        duration, video_fps = get_video_info(video_path)

        # fps=0 时使用视频自身帧率，否则透传用户设定值
        fps_effective = video_fps if fps == 0 else fps
        frames = int(round(duration * fps_effective))
        return (frames, fps_effective)


class ImageSequenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False, "tooltip": "包含图片序列的文件夹路径。Folder path containing the image sequence."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 120.0, "step": 0.01, "tooltip": "输出视频帧率，设为0则根据音频时长自动推导。Output video frame rate, set to 0 to auto-calculate based on audio duration."}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "使用的总帧数，设为0则使用文件夹中所有图片。Total frames to use, set to 0 to use all images in the folder."}),
                "audio_source": ("STRING", {"default": "", "multiline": False, "tooltip": "音频文件路径，用于为视频添加背景音。Audio file path to add as background to the video."}),
                "output_path_prefix": ("STRING", {"default": "video/temp", "multiline": False, "tooltip": "输出文件路径前缀，相对于 ComfyUI 输出目录。Output file path prefix, relative to the ComfyUI output directory."}),
                "start_frame_offset": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "从音频的起始偏移帧数，用于跳过音频开头。Frame offset from the start of the audio, used to skip the beginning of the audio."}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1, "tooltip": "视频编码质量（CRF值），越低质量越高，建议18-28。Video encoding quality (CRF value), lower is better quality, recommended 18-28."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "video"
    OUTPUT_NODE = True
    DESCRIPTION = "将图片序列合成为视频，支持音频同步和帧率控制。Combine image sequences into a video, with audio synchronization and frame rate control."


    def generate_video(self, folder_path: str,
                       fps: Optional[float] = 0,
                       total_frames: Optional[int] = 0,
                       audio_source: str = "",
                       output_path_prefix: str = "video/temp",
                       start_frame_offset: int = 0,
                       crf: int = 23) -> Tuple[str]:
        # ==============================
        # 1. 路径解析
        # ==============================
        folder_path = resolve_path(folder_path)
        if audio_source:
            audio_source = resolve_path(audio_source)

        logging.info(f"图片文件夹: {folder_path}")
        if audio_source:
            logging.info(f"音频源: {audio_source}")
        logging.info(f"起始帧偏移: {start_frame_offset} 帧")
        logging.info(f"CRF 值: {crf}")

        # ==============================
        # 2. 检查输入文件夹
        # ==============================
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"文件夹不存在: {folder_path}")

        # ==============================
        # 3. 获取图片列表
        # ==============================
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        images = [f for f in os.listdir(folder_path) if Path(f).suffix.lower() in image_extensions]
        if not images:
            raise ValueError(f"文件夹中没有图片文件: {folder_path}")
        images.sort()
        logging.info(f"图片总数: {len(images)}")

        # ==============================
        # 4. 确定帧数上限
        # ==============================
        requested_frames = len(images)
        if total_frames and total_frames > 0:
            requested_frames = total_frames
        num_frames = min(requested_frames, len(images))
        logging.info(f"用户设置最大帧数: {total_frames if total_frames else '无限制'}, 实际可用帧数: {num_frames}")

        # ==============================
        # 5. 判断音频有效性
        # ==============================
        has_audio = bool(audio_source) and os.path.isfile(audio_source)
        if has_audio:
            audio_total_duration = get_media_duration(audio_source)
            logging.info(f"音频总时长: {audio_total_duration:.2f} 秒")
            audio_valid = audio_total_duration > 0
        else:
            audio_total_duration = 0.0
            audio_valid = False

        # ==============================
        # 6. fps 自动推导（处理 fps=0 或 None）
        # ==============================
        if fps is None or fps == 0:
            if audio_valid:
                # Step A：用音频总时长粗算 fps
                fps_rough = num_frames / audio_total_duration
                # Step B：用粗算 fps 计算偏移秒数
                offset_sec = start_frame_offset / fps_rough
                # 检查偏移是否超限
                if offset_sec >= audio_total_duration:
                    raise ValueError(
                        f"起始偏移 {start_frame_offset} 帧（约 {offset_sec:.2f} 秒）"
                        f"已超过或等于音频总时长 {audio_total_duration:.2f} 秒"
                    )
                audio_remaining = audio_total_duration - offset_sec

                # Step C：精算 fps
                fps_calc = num_frames / audio_remaining
                if fps_calc > 120.0:
                    fps = 120.0
                    # 按上限抽取帧，丢多余帧
                    actual_frames = max(1, math.floor(audio_remaining * fps))
                    logging.info(
                        f"计算 fps={fps_calc:.2f} 超过上限 120，"
                        f"已限制为 120。原帧数 {num_frames}，"
                        f"按 120fps 抽帧后实际帧数 {actual_frames}"
                    )
                else:
                    fps = fps_calc
                    actual_frames = num_frames
                    logging.info(f"fps=0 自动推导: fps={fps:.4f}")

                # 统一音频相关变量
                audio_start_time = offset_sec
                final_duration = audio_total_duration - audio_start_time
                # 确保 actual_frames 不超上限
                actual_frames = min(actual_frames, num_frames)
            else:
                # 无音频或音频时长为0
                fps = 25.0
                audio_start_time = 0.0
                final_duration = num_frames / fps
                actual_frames = num_frames
                logging.info(f"fps=0 且无有效音频，使用默认 fps=25")
        else:
            # fps 已由用户显式设定
            if audio_valid:
                audio_start_time = start_frame_offset / fps
                if audio_start_time >= audio_total_duration:
                    raise ValueError(
                        f"起始偏移 {audio_start_time:.2f} 秒"
                        f"超过音频总时长 {audio_total_duration:.2f} 秒"
                    )
                audio_remaining = audio_total_duration - audio_start_time
                video_duration = num_frames / fps
                if video_duration > audio_remaining:
                    # 视频长于音频，截断帧数以匹配音频
                    actual_frames = max(1, math.floor(audio_remaining * fps))
                    final_duration = audio_remaining
                    logging.info("视频长于音频剩余，截断至音频长度")
                else:
                    actual_frames = num_frames
                    final_duration = video_duration
                    logging.info("视频短于或等于音频剩余，以视频长度为准")
            else:
                audio_start_time = 0.0
                final_duration = num_frames / fps
                actual_frames = num_frames

        logging.info(f"最终 fps: {fps:.4f}, 实际使用帧数: {actual_frames}, 视频时长: {final_duration:.2f} 秒")
        used_images = images[:actual_frames]

        # ==============================
        # 7. 音频截取（如果有音频）
        # ==============================
        temp_audio_file = None
        if audio_valid:
            with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp:
                temp_audio_file = tmp.name
            logging.info(f"创建临时音频文件: {temp_audio_file}")

            cut_cmd = [
                FFMPEG_PATH, "-y",
                "-ss", str(audio_start_time),
                "-t", str(final_duration),
                "-i", audio_source,
                "-vn",
                "-c:a", "aac",
                "-b:a", "192k",
                temp_audio_file
            ]
            logging.info(f"截取音频命令: {' '.join(cut_cmd)}")
            cut_process = subprocess.run(cut_cmd, capture_output=True, text=True)
            if cut_process.returncode != 0:
                raise RuntimeError(f"音频截取失败: {cut_process.stderr}")
            logging.info("音频截取完成")
        else:
            logging.info("未提供音频源或音频无效，将生成纯视频")

        # ==============================
        # 8. 构建输出路径
        # ==============================
        output_dir = folder_paths.get_output_directory()
        rel_path = Path(output_path_prefix)
        save_dir = output_dir / rel_path.parent
        filename_prefix = rel_path.name
        save_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        while True:
            final_path = save_dir / f"{filename_prefix}_{counter:04d}.mp4"
            if not final_path.exists():
                break
            counter += 1
        logging.info(f"输出视频路径: {final_path}")

        # ==============================
        # 9. 创建图片列表文件（ffconcat格式）
        # ==============================
        duration_per_frame = 1.0 / fps
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            list_file = f.name
            f.write("ffconcat version 1.0\n")
            for img in used_images:
                abs_path = os.path.join(folder_path, img).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {duration_per_frame}\n")
        logging.info(f"图片列表文件 (ffconcat): {list_file}")

        # ==============================
        # 10. 构建合成命令
        # ==============================
        cmd = [
            FFMPEG_PATH, "-y", "-loglevel", "info",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
        ]

        if temp_audio_file:
            cmd.extend(["-i", temp_audio_file])
            map_opts = ["-map", "0:v:0", "-map", "1:a:0?"]
            audio_enc = ["-c:a", "copy"]
        else:
            map_opts = ["-map", "0:v:0"]
            audio_enc = []

        cmd.extend(map_opts)
        cmd.extend([
            "-r", str(fps),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ])
        if temp_audio_file:
            cmd.extend(audio_enc)
            cmd.append("-shortest")
        cmd.append(str(final_path))

        logging.info(f"合成命令: {' '.join(cmd)}")

        # ==============================
        # 11. 执行并显示进度
        # ==============================
        total_seconds = final_duration
        pbar = comfy.utils.ProgressBar(int(total_seconds * 10))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )

        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
        last_progress = 0
        stderr_lines = []

        for line in process.stderr:
            stderr_lines.append(line)
            match = time_pattern.search(line)
            if match:
                h, m, s = match.groups()
                current_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                progress = int(current_seconds * 10)
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress

        process.wait()

        full_stderr = ''.join(stderr_lines)
        logging.debug(f"ffmpeg 输出:\n{full_stderr}")

        if process.returncode != 0:
            error_msg = f"ffmpeg 合成失败，返回码 {process.returncode}\n完整 stderr:\n{full_stderr}"
            raise RuntimeError(error_msg)

        # ==============================
        # 12. 清理临时文件
        # ==============================
        os.unlink(list_file)
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)

        return (str(final_path),)


class VideoGridCompose:
    """视频网格拼接：最多支持4路视频，多种布局模式，流式处理无需解帧"""
    
    LAYOUT_OPTIONS = ["水平排列", "垂直排列", "田字格2×2", "左一右N", "上一+下N"]
    AUDIO_OPTIONS = ["无音频", "视频1", "视频2", "视频3", "视频4"]
    MATCH_OPTIONS = ["自适应(填满)", "自适应(保持比例)", "统一到最大", "统一到最小"]
    DURATION_OPTIONS = ["自动(最短)", "自动(最长)", "视频1", "视频2", "视频3", "视频4"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_paths": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "多行视频路径，每行一个，最多支持4路（自动去重去空行）"
                }),
                "layout": (cls.LAYOUT_OPTIONS, {
                    "default": "水平排列",
                    "tooltip": "拼接布局模式"
                }),
                "audio_source": (cls.AUDIO_OPTIONS, {
                    "default": "无音频",
                    "tooltip": "选择哪一路视频的音频作为输出音频"
                }),
                "match_method": (cls.MATCH_OPTIONS, {
                    "default": "自适应(填满)",
                    "tooltip": "各视频尺寸匹配方式"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0,
                    "max": 120.0,
                    "step": 0.01,
                    "tooltip": "输出帧率，设为0自动取最高输入帧率"
                }),
                "crf": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "编码质量，越低质量越高（建议18-28）"
                }),
                "duration_mode": (cls.DURATION_OPTIONS, {
                    "default": "自动(最短)",
                    "tooltip": "时长同步模式：以哪一路输入为基准对齐时长。自动(最短)=以最短时长为准；自动(最长)=以最长时长为准；视频N=以第N路视频的时长为准。若选中的是图片则跳过执行。Duration sync mode: which input to use as duration reference. Auto(shortest)=use shortest duration; Auto(longest)=use longest duration; VideoN=use the Nth video's duration. Skips execution if the selected source is an image."
                }),
                "output_path_prefix": ("STRING", {
                    "default": "video/grid_output",
                    "multiline": False,
                    "tooltip": "输出路径前缀（相对于ComfyUI输出目录）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "compose"
    CATEGORY = "video"
    OUTPUT_NODE = True
    DESCRIPTION = "视频网格拼接：支持水平/垂直/田字格/左一右N/上一+下N布局，4路视频输入，流式ffmpeg处理。"
    
    def _parse_paths(self, video_paths: str) -> List[str]:
        """解析多行字符串路径列表，去重去空行，最多取4个"""
        lines = video_paths.strip().split('\n')
        seen = set()
        paths = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line in seen:
                continue
            seen.add(line)
            paths.append(line)
            if len(paths) >= 4:
                break
        return paths
    
    def _determine_target_size(self, resolutions: List[Tuple[int, int]], match_method: str, layout: str, num_videos: int) -> Tuple[int, int]:
        """
        根据匹配策略和布局确定目标画布尺寸
        返回 (canvas_width, canvas_height) 以及每个视频对应的瓦片尺寸
        """
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        
        if match_method == "统一到最小":
            target_w = min(widths)
            target_h = min(heights)
        elif match_method == "统一到最大":
            target_w = max(widths)
            target_h = max(heights)
        elif match_method in ("自适应(填满)", "自适应(保持比例)"):
            # 取最大宽高作为基准
            target_w = max(widths)
            target_h = max(heights)
        else:
            target_w = max(widths)
            target_h = max(heights)
        
        return target_w, target_h
    
    def _build_scale_filter(self, idx: int, src_w: int, src_h: int, tile_w: int, tile_h: int, match_method: str) -> str:
        """生成单个视频的缩放滤镜标签"""
        label = f"s{idx}"
        if match_method == "自适应(填满)":
            # 直接拉伸填满瓦片
            return f"[{idx}:v]scale={tile_w}:{tile_h}:flags=lanczos[{label}]"
        elif match_method == "自适应(保持比例)":
            # 保持比例缩放 + pad黑边填充至瓦片尺寸
            return (
                f"[{idx}:v]scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black[{label}]"
            )
        elif match_method == "统一到最大":
            return f"[{idx}:v]scale={tile_w}:{tile_h}:flags=lanczos[{label}]"
        elif match_method == "统一到最小":
            return f"[{idx}:v]scale={tile_w}:{tile_h}:flags=lanczos[{label}]"
        else:
            return f"[{idx}:v]scale={tile_w}:{tile_h}:flags=lanczos[{label}]"
    
    def _build_hstack_filter(self, num_videos: int, labels: List[str]) -> str:
        """生成水平堆叠滤镜"""
        inputs = ''.join(f'[{l}]' for l in labels)
        return f"{inputs}hstack=inputs={num_videos}[out]"
    
    def _build_vstack_filter(self, num_videos: int, labels: List[str]) -> str:
        """生成垂直堆叠滤镜"""
        inputs = ''.join(f'[{l}]' for l in labels)
        return f"{inputs}vstack=inputs={num_videos}[out]"
    
    def _build_grid2x2_filter(self, labels: List[str]) -> str:
        """生成田字格滤镜"""
        return (
            f"[{labels[0]}][{labels[1]}]hstack=inputs=2[row0];"
            f"[{labels[2]}][{labels[3]}]hstack=inputs=2[row1];"
            f"[row0][row1]vstack=inputs=2[out]"
        )
    
    def _build_left_right_filter(self, labels: List[str]) -> str:
        """生成左一右N滤镜"""
        n = len(labels)
        # 右边 n-1 个垂直排列
        right_labels = labels[1:]
        right_vstack = ''.join(f'[{l}]' for l in right_labels)
        # 左边 + 右边水平排列
        return (
            f"{right_vstack}vstack=inputs={n-1}[right];"
            f"[{labels[0]}][right]hstack=inputs=2[out]"
        )
    
    def _build_top_bottom_filter(self, labels: List[str]) -> str:
        """生成上一+下N滤镜"""
        n = len(labels)
        # 下面 n-1 个水平排列
        bottom_labels = labels[1:]
        bottom_hstack = ''.join(f'[{l}]' for l in bottom_labels)
        return (
            f"{bottom_hstack}hstack=inputs={n-1}[bottom];"
            f"[{labels[0]}][bottom]vstack=inputs=2[out]"
        )
    
    def _is_image_file(self, file_path: str) -> bool:
        """使用 PIL 自动检测文件是否为图片。尝试打开，能打开就是图片。"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def _get_resolution_from_image(self, file_path: str) -> Tuple[int, int]:
        """获取图片分辨率"""
        with Image.open(file_path) as img:
            return img.size  # (width, height)
    
    def _get_duration_for_input(self, file_path: str, is_image: bool) -> Optional[float]:
        """获取输入时长。图片返回 None，视频返回秒数。"""
        if is_image:
            return None
        try:
            duration, _ = get_video_info(file_path)
            return duration
        except Exception:
            return None
    
    def _determine_target_duration(self, is_images: List[bool], durations: List[Optional[float]], 
                                    duration_mode: str) -> Tuple[Optional[float], int]:
        """
        根据 duration_mode 确定基准时长。
        返回 (target_duration, source_index)，source_index 仅对"视频N"模式有意义。
        如果选中的主时长源是图片，返回 (None, -1)。
        """
        if duration_mode.startswith("视频"):
            idx = int(duration_mode.replace("视频", "")) - 1  # "视频1" -> 0
            if idx < len(is_images):
                if is_images[idx]:
                    # 选中的是图片
                    return None, idx
                dur = durations[idx]
                if dur is not None and dur > 0:
                    return dur, idx
            return None, idx
        
        # 自动模式：收集所有视频的时长
        video_durations = [d for d in durations if d is not None and d > 0]
        if not video_durations:
            return None, -1
        
        if duration_mode == "自动(最长)":
            return max(video_durations), -1
        else:  # "自动(最短)"
            return min(video_durations), -1
    
    def _check_layout_availability(self, layout: str, num_videos: int) -> str:
        """检查布局是否可用，不可用时降级"""
        if layout == "田字格2×2" and num_videos != 4:
            logging.warning(f"田字格布局需要4路视频，当前仅{num_videos}路，自动降级为水平排列")
            return "水平排列"
        if layout in ("左一右N", "上一+下N") and num_videos < 3:
            logging.warning(f"{layout}布局需要至少3路视频，当前仅{num_videos}路，自动降级为水平排列")
            return "水平排列"
        return layout
    
    def _calculate_tile_sizes(self, layout: str, resolutions: List[Tuple[int, int]], 
                              base_w: int, base_h: int) -> List[Tuple[int, int]]:
        """根据布局计算每个视频的目标瓦片尺寸"""
        n = len(resolutions)
        
        if layout == "水平排列":
            # 所有视频统一高度为 base_h，宽度按比例缩放
            tiles = []
            for w, h in resolutions:
                tile_h = base_h
                tile_w = int(w * base_h / h)
                # 确保偶数
                if tile_w % 2 != 0:
                    tile_w += 1
                tiles.append((tile_w, tile_h))
            return tiles
        
        elif layout == "垂直排列":
            # 所有视频统一宽度为 base_w，高度按比例缩放
            tiles = []
            for w, h in resolutions:
                tile_w = base_w
                tile_h = int(h * base_w / w)
                if tile_h % 2 != 0:
                    tile_h += 1
                tiles.append((tile_w, tile_h))
            return tiles
        
        elif layout == "田字格2×2":
            # 所有视频缩放到相同尺寸 (base_w//2, base_h//2)
            tw = base_w // 2
            th = base_h // 2
            if tw % 2 != 0:
                tw += 1
            if th % 2 != 0:
                th += 1
            return [(tw, th)] * n
        
        elif layout == "左一右N":
            # 左边1个视频：高度 = base_h，宽度按比例
            # 右边 n-1 个：高度 = base_h // (n-1)，宽度按各自比例
            tiles = []
            # 左
            left_w = int(resolutions[0][0] * base_h / resolutions[0][1])
            if left_w % 2 != 0:
                left_w += 1
            tiles.append((left_w, base_h))
            # 右
            right_h = base_h // (n - 1)
            if right_h % 2 != 0:
                right_h += 1
            for i in range(1, n):
                w, h = resolutions[i]
                right_w = int(w * right_h / h)
                if right_w % 2 != 0:
                    right_w += 1
                tiles.append((right_w, right_h))
            return tiles
        
        elif layout == "上一+下N":
            # 上面1个视频：宽度 = base_w，高度按比例
            # 下面 n-1 个：宽度 = base_w // (n-1)，高度按各自比例
            tiles = []
            # 上
            top_h = int(resolutions[0][1] * base_w / resolutions[0][0])
            if top_h % 2 != 0:
                top_h += 1
            tiles.append((base_w, top_h))
            # 下
            bottom_w = base_w // (n - 1)
            if bottom_w % 2 != 0:
                bottom_w += 1
            for i in range(1, n):
                w, h = resolutions[i]
                bottom_h = int(h * bottom_w / w)
                if bottom_h % 2 != 0:
                    bottom_h += 1
                tiles.append((bottom_w, bottom_h))
            return tiles
        
        return [(base_w, base_h)] * n
    
    def compose(self, video_paths: str,
                layout: str = "水平排列",
                audio_source: str = "无音频",
                match_method: str = "自适应(填满)",
                fps: float = 25.0,
                crf: int = 23,
                duration_mode: str = "自动(最短)",
                output_path_prefix: str = "video/grid_output") -> Tuple[str]:
        
        # ==============================
        # 1. 解析路径列表
        # ==============================
        raw_paths = self._parse_paths(video_paths)
        if not raw_paths:
            raise ValueError("视频路径列表为空，请至少输入1个视频路径")
        
        # 解析为绝对路径
        resolved_paths = []
        for p in raw_paths:
            try:
                resolved = resolve_path(p)
                resolved_paths.append(resolved)
            except (ValueError, PermissionError, FileNotFoundError) as e:
                logging.warning(f"跳过无效路径 '{p}': {e}")
                continue
        
        # 检查文件是否存在
        valid_paths = [p for p in resolved_paths if os.path.isfile(p)]
        if len(valid_paths) != len(resolved_paths):
            missing = [p for p in resolved_paths if not os.path.isfile(p)]
            for p in missing:
                logging.warning(f"文件不存在，已跳过: {p}")
        
        num_inputs = len(valid_paths)
        if num_inputs == 0:
            raise ValueError("没有有效的文件可供处理")
        
        logging.info(f"有效输入数量: {num_inputs}")
        for i, p in enumerate(valid_paths):
            logging.info(f"  输入{i+1}: {p}")
        
        # ==============================
        # 2. 检测每个输入的类型（图片/视频）并获取信息
        # ==============================
        is_images = [self._is_image_file(p) for p in valid_paths]
        resolutions = []
        durations = []  # 视频时长；图片为 None
        max_fps = 0.0
        
        for i, (p, is_img) in enumerate(zip(valid_paths, is_images)):
            if is_img:
                w, h = self._get_resolution_from_image(p)
                resolutions.append((w, h))
                durations.append(None)
                logging.info(f"  输入{i+1} (图片): {w}x{h}")
            else:
                w, h = get_video_resolution(p)
                resolutions.append((w, h))
                dur, vfps = get_video_info(p)
                durations.append(dur)
                max_fps = max(max_fps, vfps)
                logging.info(f"  输入{i+1} (视频): {w}x{h}, {dur:.2f}s, {vfps:.2f}fps")
        
        # ==============================
        # 3. 单输入透传
        # ==============================
        if num_inputs == 1:
            logging.info("仅1路输入，直接透传原路径")
            return (valid_paths[0],)
        
        # ==============================
        # 4. 布局降级检查
        # ==============================
        effective_layout = self._check_layout_availability(layout, num_inputs)
        logging.info(f"布局: {layout} -> 生效布局: {effective_layout}")
        
        # ==============================
        # 5. 确定帧率
        # ==============================
        effective_fps = max_fps if fps == 0 else fps
        logging.info(f"输出帧率: {effective_fps}")
        
        # ==============================
        # 6. 计算目标尺寸和瓦片尺寸
        # ==============================
        base_w, base_h = self._determine_target_size(resolutions, match_method, effective_layout, num_inputs)
        logging.info(f"基准尺寸: {base_w}x{base_h}")
        
        tile_sizes = self._calculate_tile_sizes(effective_layout, resolutions, base_w, base_h)
        logging.info(f"瓦片尺寸: {tile_sizes}")
        
        # ==============================
        # 7. 处理时长同步
        # ==============================
        target_duration, source_idx = self._determine_target_duration(is_images, durations, duration_mode)
        
        # 检查选中主时长源是否为图片
        if target_duration is None and not duration_mode.startswith("自动"):
            # 只有指定"视频N"且选中图片时才跳过
            logging.warning(f"主时长源 ({duration_mode}) 是图片，没有时长信息，跳过执行")
            return ("",)
        
        if target_duration is None:
            # 自动模式下没有有效视频时长（全是图片），取最长图片静止时长
            # 用 5 秒作为默认时长
            target_duration = 5.0
            logging.info(f"所有输入均为图片，使用默认时长 {target_duration}s")
        
        logging.info(f"目标时长: {target_duration:.2f}s (模式: {duration_mode})")
        
        # ==============================
        # 8. 构建 ffmpeg 输入命令与 filter_complex
        # ==============================
        ffmpeg_inputs = []
        filter_parts = []
        map_labels = []
        input_index = 0  # ffmpeg 输入流索引
        
        for i in range(num_inputs):
            is_img = is_images[i]
            path = valid_paths[i]
            tw, th = tile_sizes[i]
            sw, sh = resolutions[i]
            dur = durations[i]
            
            if is_img:
                # 图片输入：用 -loop 1 -t 将其转为指定时长的静止视频流
                ffmpeg_inputs.extend([
                    "-loop", "1",
                    "-t", str(target_duration),
                    "-i", path
                ])
                # 图片直接从输入流缩放
                label = f"s{i}"
                scale_filter = self._build_scale_filter(input_index, sw, sh, tw, th, match_method)
                filter_parts.append(scale_filter)
                map_labels.append(label)
                input_index += 1
            else:
                # 视频输入
                ffmpeg_inputs.extend(["-i", path])
                
                # 先缩放，再时长对齐
                scale_part = self._build_scale_filter(input_index, sw, sh, tw, th, match_method)
                filter_parts.append(scale_part)
                
                # _build_scale_filter 输出的标签格式为 s{input_index}
                current_label = f"s{input_index}"
                
                # 时长对齐处理
                if dur is not None and abs(dur - target_duration) > 0.05:
                    if dur > target_duration:
                        # 比基准长 → 截断
                        trim_label = f"tr{i}"
                        filter_parts.append(
                            f"[{current_label}]trim=end={target_duration},setpts=PTS[{trim_label}]"
                        )
                        current_label = trim_label
                    else:
                        # 比基准短 → 冻结最后一帧补齐
                        tpad_label = f"tp{i}"
                        pad_duration = target_duration - dur
                        filter_parts.append(
                            f"[{current_label}]tpad=stop_mode=clone:stop_duration={pad_duration}[{tpad_label}]"
                        )
                        current_label = tpad_label
                
                map_labels.append(current_label)
                input_index += 1
        
        # 布局拼接滤镜
        if effective_layout == "水平排列":
            compose_filter = self._build_hstack_filter(num_inputs, map_labels)
        elif effective_layout == "垂直排列":
            compose_filter = self._build_vstack_filter(num_inputs, map_labels)
        elif effective_layout == "田字格2×2":
            compose_filter = self._build_grid2x2_filter(map_labels)
        elif effective_layout == "左一右N":
            compose_filter = self._build_left_right_filter(map_labels)
        elif effective_layout == "上一+下N":
            compose_filter = self._build_top_bottom_filter(map_labels)
        else:
            compose_filter = self._build_hstack_filter(num_inputs, map_labels)
        
        filter_parts.append(compose_filter)
        
        # ==============================
        # 9. 处理音频（含时长对齐）
        # ==============================
        audio_label = audio_source.strip()
        has_audio = audio_label != "无音频" and audio_label in self.AUDIO_OPTIONS
        
        audio_input_idx = -1
        audio_map_label = None
        if has_audio:
            # "视频N" 直接对应第 N 行输入（1-based），不跳过图片
            audio_idx = self.AUDIO_OPTIONS.index(audio_label) - 1  # "视频1"->0, "视频2"->1, ...
            if audio_idx < num_inputs:
                audio_input_idx = audio_idx
                # 检查该行是否是图片（图片没有音频流）
                if is_images[audio_idx]:
                    logging.warning(f"音频来源 {audio_label} 是图片，没有音频轨道，将不使用音频")
                    has_audio = False
                else:
                    # 对音频做时长对齐（在 filter_complex 中处理）
                    audio_dur = durations[audio_input_idx]  # 原始音频时长
                    if audio_dur is not None and abs(audio_dur - target_duration) > 0.05:
                        audio_label_name = f"audio_out"
                        if audio_dur > target_duration:
                            # 比基准长 → 截断
                            filter_parts.append(
                                f"[{audio_input_idx}:a:0]atrim=end={target_duration}[{audio_label_name}]"
                            )
                        else:
                            # 比基准短 → 补静音
                            pad_dur = target_duration - audio_dur
                            filter_parts.append(
                                f"[{audio_input_idx}:a:0]apad=pad_dur={pad_dur}[{audio_label_name}]"
                            )
                        audio_map_label = audio_label_name
                    # else: 时长一致，直接映射原始流（audio_map_label=None）
            else:
                logging.warning(f"音频来源 {audio_label} 超出输入数量 {num_inputs}，将不使用音频")
                has_audio = False
        
        # 在所有拼接后添加 pad 确保宽高为偶数（libx264 要求）
        filter_parts.append("[out]pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black[final_out]")
        
        filter_complex = ';'.join(filter_parts)
        logging.info(f"filter_complex: {filter_complex}")
        
        # ==============================
        # 10. 构建输出路径
        # ==============================
        output_dir = folder_paths.get_output_directory()
        rel_path = Path(output_path_prefix)
        save_dir = output_dir / rel_path.parent
        filename_prefix = rel_path.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        counter = 1
        while True:
            final_path = save_dir / f"{filename_prefix}_{counter:04d}.mp4"
            if not final_path.exists():
                break
            counter += 1
        logging.info(f"输出视频路径: {final_path}")
        
        # ==============================
        # 11. 构建 ffmpeg 命令
        # ==============================
        cmd = [FFMPEG_PATH, "-y", "-loglevel", "info"]
        
        # 输入文件
        cmd.extend(ffmpeg_inputs)
        
        # filter_complex
        cmd.extend(["-filter_complex", filter_complex])
        
        # 视频映射（从 pad 后的 final_out 取）
        cmd.extend(["-map", "[final_out]"])
        
        # 音频映射
        if has_audio and audio_input_idx >= 0:
            if audio_map_label:
                # 经过 atrim/apad 处理后的音频流
                cmd.extend(["-map", f"[{audio_map_label}]"])
            else:
                # 原始音频流（时长与目标一致）
                cmd.extend(["-map", f"{audio_input_idx}:a:0?"])
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        
        # 编码参数
        cmd.extend([
            "-c:v", "libx264",
            "-crf", str(crf),
            "-r", str(effective_fps),
            "-pix_fmt", "yuv420p",
            str(final_path)
        ])
        
        logging.info(f"ffmpeg 命令: {' '.join(cmd)}")
        
        # ==============================
        # 12. 执行
        # ==============================
        pbar = comfy.utils.ProgressBar(int(target_duration * 10))
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
        last_progress = 0
        stderr_lines = []
        
        for line in process.stderr:
            stderr_lines.append(line)
            match = time_pattern.search(line)
            if match:
                h, m, s = match.groups()
                current_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                progress = int(current_seconds * 10)
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
        
        process.wait()
        
        full_stderr = ''.join(stderr_lines)
        logging.debug(f"ffmpeg 输出:\n{full_stderr}")
        
        if process.returncode != 0:
            error_msg = f"ffmpeg 拼接失败，返回码 {process.returncode}\n完整 stderr:\n{full_stderr}"
            raise RuntimeError(error_msg)
        
        logging.info(f"视频拼接完成: {final_path}")
        return (str(final_path),)


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "VideoFrameCounter": VideoFrameCounter,
    "ImageSequenceToVideo": ImageSequenceToVideo,
    "VideoGridCompose": VideoGridCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCounter": "视频帧数统计",
    "ImageSequenceToVideo": "图片序列合成视频",
    "VideoGridCompose": "视频网格拼接",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]