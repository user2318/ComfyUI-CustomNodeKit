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

FFMPEG_PATH = get_ffmpeg_exe()
BASE_DIR = folder_paths.base_path


def resolve_path(path: str) -> str:
    """将输入路径转换为绝对路径（相对于 ComfyUI 根目录）"""
    if os.path.isabs(path):
        return os.path.normpath(path)
    else:
        return os.path.normpath(os.path.join(BASE_DIR, path))


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


def get_media_duration(file_path: str) -> float:
    """兼容封装：仅获取视频时长（秒），内部调用 get_video_info"""
    duration, _ = get_video_info(file_path)
    return duration


class VideoFrameCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 120.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "count_frames"
    CATEGORY = "video"

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
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 120.0, "step": 0.01}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "audio_source": ("STRING", {"default": "", "multiline": False}),
                "output_path_prefix": ("STRING", {"default": "video/temp", "multiline": False}),
                "start_frame_offset": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1}),   # 新增 CRF 参数
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "video"
    OUTPUT_NODE = True


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


# ==============================
# 节点注册
# ==============================
NODE_CLASS_MAPPINGS = {
    "VideoFrameCounter": VideoFrameCounter,
    "ImageSequenceToVideo": ImageSequenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCounter": "视频帧数统计",
    "ImageSequenceToVideo": "图片序列合成视频",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
