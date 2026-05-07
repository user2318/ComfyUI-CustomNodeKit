# WanContextWindowsNode.py
# 完全自包含的 WAN Context Windows 节点，支持前缀拼接（真正的参考帧拼接）
# 版本：2.0 - 支持前缀窗口

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import torch
import numpy as np
import collections
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
import comfy.model_management
import comfy.patcher_extension
import nodes

if TYPE_CHECKING:
    from comfy.model_base import BaseModel
    from comfy.model_patcher import ModelPatcher
    from comfy.controlnet import ControlBase


class ContextWindowABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get_tensor(self, full: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def add_window(self, full: torch.Tensor, to_add: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")


class ContextHandlerABC(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: ContextWindowABC, device=None) -> list:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        raise NotImplementedError("Not implemented.")


class IndexListContextWindow(ContextWindowABC):
    def __init__(self, index_list: list[int], dim: int = 0, total_frames: int = 0):
        self.index_list = index_list          # 实际的索引列表（可能包含前缀）
        self.context_length = len(index_list)
        self.dim = dim
        self.total_frames = total_frames
        # 记录原始窗口帧索引（不含前缀），由外部设置
        self.original_indices: list[int] = []
        self.prefix_length: int = 0

    def get_tensor(self, full: torch.Tensor, device=None, dim=None, retain_index_list=[]) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        if dim == 0 and full.shape[dim] == 1:
            return full
        idx = tuple([slice(None)] * dim + [self.index_list])
        window = full[idx]
        # 如果指定了保留索引，则用原始 full 中的对应位置覆盖窗口中的这些位置
        if retain_index_list:
            # 注意：retain_index_list 是全局索引，需要映射到窗口内的位置
            for global_idx in retain_index_list:
                if global_idx in self.index_list:
                    local_pos = self.index_list.index(global_idx)
                    # 构造切片：取该局部位置
                    idx_local = tuple([slice(None)] * dim + [local_pos])
                    idx_global = tuple([slice(None)] * dim + [global_idx])
                    window[idx_local] = full[idx_global]
        return window.to(device)

    def add_window(self, full: torch.Tensor, to_add: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        idx = tuple([slice(None)] * dim + [self.index_list])
        full[idx] += to_add
        return full

    def get_region_index(self, num_regions: int) -> int:
        # 使用原始窗口的中心位置来确定区域（忽略前缀）
        if self.original_indices:
            center = (min(self.original_indices) + max(self.original_indices)) / 2
            center_ratio = center / self.total_frames
        else:
            center_ratio = (min(self.index_list) + max(self.index_list)) / (2 * self.total_frames)
        region_idx = int(center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)


class IndexListCallbacks:
    EVALUATE_CONTEXT_WINDOWS = "evaluate_context_windows"
    COMBINE_CONTEXT_WINDOW_RESULTS = "combine_context_window_results"
    EXECUTE_START = "execute_start"
    EXECUTE_CLEANUP = "execute_cleanup"
    RESIZE_COND_ITEM = "resize_cond_item"

    def init_callbacks(self):
        return {}


@dataclass
class ContextSchedule:
    name: str
    func: Callable


@dataclass
class ContextFuseMethod:
    name: str
    func: Callable


ContextResults = collections.namedtuple("ContextResults", ['window_idx', 'sub_conds_out', 'sub_conds', 'window'])


class IndexListContextHandler(ContextHandlerABC):
    def __init__(self, context_schedule: ContextSchedule, fuse_method: ContextFuseMethod,
                 context_length: int = 1, context_overlap: int = 0, context_stride: int = 1,
                 closed_loop: bool = False, dim: int = 0, freenoise: bool = False,
                 prefix_latent_len: int = 0, split_conds_to_windows: bool = False):
        self.context_schedule = context_schedule
        self.fuse_method = fuse_method
        self.context_length = context_length          # 原始窗口长度（不含前缀）
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.dim = dim
        self._step = 0
        self.freenoise = freenoise
        # 前缀索引（前 N 个 latent 帧作为参考帧）
        self.prefix_latent_len = prefix_latent_len
        self.prefix_indices = list(range(prefix_latent_len))
        self.split_conds_to_windows = split_conds_to_windows
        self.callbacks = {}

    def should_use_context(self, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]) -> bool:
        if x_in.size(self.dim) > self.context_length:
            logging.info(f"Using context windows {self.context_length} with overlap {self.context_overlap} for {x_in.size(self.dim)} frames.")
            if self.prefix_indices:
                logging.info(f"Prefixing frames: {self.prefix_indices} to each window.")
            return True
        return False

    def prepare_control_objects(self, control: ControlBase, device=None) -> ControlBase:
        if control.previous_controlnet is not None:
            self.prepare_control_objects(control.previous_controlnet, device)
        return control

    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None
        resized_cond = []
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            logging.info(f"Splitting conds to windows; using region {region} for window (original {window.original_indices[0]}-{window.original_indices[-1]})")
            cond_in = [cond_in[region]]
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if isinstance(cond_item, torch.Tensor):
                        if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                            actual_cond_item = window.get_tensor(cond_item, retain_index_list=self.prefix_indices)
                            resized_actual_cond[key] = actual_cond_item.to(device)
                        else:
                            resized_actual_cond[key] = cond_item.to(device)
                    elif key == "control":
                        resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        for cond_key, cond_value in new_cond_item.items():
                            logging.debug(f"cond_key: {cond_key}, type: {type(cond_value)}")
                            # 特殊处理 pose_video_latent：使用原始窗口索引（不含前缀）进行切片
                            if cond_key == "pose_video_latent" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                if hasattr(window, "original_indices") and window.original_indices:
                                    # 假设 cond 形状为 [B, C, T, H, W] 或 [B, T, ...]，时间维度通常在索引2
                                    # 使用原始索引切片
                                    indices = window.original_indices
                                    # 根据张量维度动态切片
                                    if cond_value.cond.ndim == 5:
                                        # [B, C, T, H, W] -> 时间维度索引2
                                        idx = (slice(None), slice(None), indices, slice(None), slice(None))
                                    elif cond_value.cond.ndim == 4:
                                        # [B, T, C, H]? 需要根据实际情况，但常见是5D
                                        idx = (slice(None), indices, slice(None), slice(None))
                                    else:
                                        # 降级：直接使用原始索引（假设第一维是batch，第二维是时间）
                                        idx = (slice(None), indices)
                                    sliced_cond = cond_value.cond[idx]
                                    new_cond_item[cond_key] = cond_value._copy_with(sliced_cond)
                                else:
                                    new_cond_item[cond_key] = cond_value
                                continue  # 跳过后续处理
                            handled = False
                            for callback in comfy.patcher_extension.get_all_callbacks(
                                IndexListCallbacks.RESIZE_COND_ITEM, self.callbacks
                            ):
                                result = callback(cond_key, cond_value, window, x_in, device, new_cond_item)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                                    break
                            if handled:
                                continue

                            # 特殊处理 face_pixel_values：根据窗口是否为第一个决定切片范围
                            if cond_key == "face_pixel_values" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                pixel_tensor = cond_value.cond  # [B, C, T, H, W]
                                if hasattr(window, "original_indices") and window.original_indices:
                                    original_indices = window.original_indices
                                else:
                                    original_indices = window.index_list  # fallback

                                prefix_len = window.prefix_length  # 前缀 latent 帧数（连续前 N 帧）

                                # 计算像素帧范围：
                                # 起始位置 = 窗口第一帧对应的像素起始，减去前缀占用的像素帧
                                # 由于前缀总是连续的 [0,1,...,prefix_len-1]，pixel_start 精确为 0
                                pixel_start = max((original_indices[0] - prefix_len) * 4, 0)

                                # 精确切片：窗口 body 部分对应的像素帧范围
                                pixel_end = (original_indices[-1] + 1) * 4
                                pixel_end = min(pixel_end, pixel_tensor.shape[2])   # 边界保护

                                if pixel_start >= pixel_tensor.shape[2]:
                                    # 若起始超出，使用空张量（但这种情况通常不应发生）
                                    sliced_pixels = torch.zeros_like(pixel_tensor[:, :, :0, :, :])
                                else:
                                    sliced_pixels = pixel_tensor[:, :, pixel_start:pixel_end, :, :]
                                new_cond_item[cond_key] = cond_value._copy_with(sliced_pixels)
                                continue

                            # 模型级回调：处理硬编码和回调未覆盖的 cond 类型
                            if not handled and self._model is not None:
                                result = self._model.resize_cond_for_context_window(
                                    cond_key, cond_value, window, x_in, device,
                                    retain_index_list=self.prefix_indices)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                            if handled:
                                continue

                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim)) or \
                                   (cond_value.ndim < self.dim and cond_value.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device, retain_index_list=self.prefix_indices)
                            elif cond_key == "audio_embed" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                audio_cond = cond_value.cond
                                if audio_cond.ndim > 1 and audio_cond.size(1) == x_in.size(self.dim):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(audio_cond, device, dim=1, retain_index_list=self.prefix_indices))
                            elif cond_key == "vace_context" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                vace_cond = cond_value.cond
                                if vace_cond.ndim >= 4 and vace_cond.size(3) == x_in.size(self.dim):
                                    sliced_vace = window.get_tensor(vace_cond, device, dim=3, retain_index_list=self.prefix_indices)
                                    new_cond_item[cond_key] = cond_value._copy_with(sliced_vace)
                            elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                if (self.dim < cond_value.cond.ndim and cond_value.cond.size(self.dim) == x_in.size(self.dim)) or \
                                   (cond_value.cond.ndim < self.dim and cond_value.cond.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(cond_value.cond, device, retain_index_list=self.prefix_indices))
                            elif cond_key == "num_video_frames":
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                                new_cond_item[cond_key].cond = window.context_length
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item
            resized_cond.append(resized_actual_cond)
        return resized_cond

    def set_step(self, timestep: torch.Tensor, model_options: dict[str]):
        mask = torch.isclose(model_options["transformer_options"]["sample_sigmas"], timestep[0], rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) == 0:
            return
        self._step = int(matches[0].item())

    def get_context_windows(self, model: BaseModel, x_in: torch.Tensor, model_options: dict[str]) -> list[IndexListContextWindow]:
        full_length = x_in.size(self.dim)
        # 获取原始窗口（不含前缀）的索引列表
        raw_windows = self.context_schedule.func(full_length, self, model_options)
        context_windows = []
        for raw_window in raw_windows:
            # 构建带前缀的窗口索引列表（去重）
            prefix = [idx for idx in self.prefix_indices if idx not in raw_window]  # 避免重复
            combined_indices = prefix + raw_window
            window = IndexListContextWindow(combined_indices, dim=self.dim, total_frames=full_length)
            window.original_indices = raw_window.copy()
            window.prefix_length = len(prefix)
            context_windows.append(window)
        return context_windows

    def execute(self, calc_cond_batch: Callable, model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        self._model = model
        self.set_step(timestep, model_options)
        context_windows = self.get_context_windows(model, x_in, model_options)
        enumerated_context_windows = list(enumerate(context_windows))

        # 计算窗口像素参数
        window_pixel_len = 4 * self.context_length - 3      # 每个窗口对应的像素帧数
        overlap_pixel_len = 4 * self.context_overlap - 3 if self.context_overlap > 0 else 0
        pixel_start = 0
        for idx, window in enumerate(context_windows):
            window.pixel_start = pixel_start
            # 下一个窗口的起始 = 当前起始 + 窗口像素长度 - 重叠像素长度
            pixel_start += window_pixel_len - overlap_pixel_len
            # 可选调试打印
            logging.debug(f"窗口 {idx}: pixel_start={window.pixel_start}, 原始索引 {window.original_indices[0]}-{window.original_indices[-1]}")


        conds_final = [torch.zeros_like(x_in) for _ in conds]
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            counts_final = [torch.ones(get_shape_for_dim(x_in, self.dim), device=x_in.device) for _ in conds]
        else:
            counts_final = [torch.zeros(get_shape_for_dim(x_in, self.dim), device=x_in.device) for _ in conds]
        biases_final = [([0.0] * x_in.shape[self.dim]) for _ in conds]

        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        for enum_window in enumerated_context_windows:
            results = self.evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep, [enum_window], model_options)
            for result in results:
                self.combine_context_window_results(x_in, result.sub_conds_out, result.sub_conds, result.window, result.window_idx,
                                                    len(enumerated_context_windows), timestep, conds_final, counts_final, biases_final)
        try:
            if self.fuse_method.name == ContextFuseMethods.RELATIVE:
                del counts_final
                return conds_final
            else:
                for i in range(len(conds_final)):
                    conds_final[i] /= counts_final[i]
                del counts_final
                return conds_final
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def evaluate_context_windows(self, calc_cond_batch: Callable, model: BaseModel, x_in: torch.Tensor, conds, timestep: torch.Tensor,
                                 enumerated_context_windows: list[tuple[int, IndexListContextWindow]],
                                 model_options, device=None, first_device=None):
        results: list[ContextResults] = []
        for window_idx, window in enumerated_context_windows:
            comfy.model_management.throw_exception_if_processing_interrupted()
            for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options, window_idx, window, model_options, device, first_device)
            model_options["transformer_options"]["context_window"] = window
            sub_x = window.get_tensor(x_in, device)
            sub_timestep = window.get_tensor(timestep, device, dim=0)
            sub_conds = [self.get_resized_cond(cond, x_in, window, device) for cond in conds]
            sub_conds_out = calc_cond_batch(model, sub_conds, sub_x, sub_timestep, model_options)
            if device is not None:
                for i in range(len(sub_conds_out)):
                    sub_conds_out[i] = sub_conds_out[i].to(x_in.device)
            results.append(ContextResults(window_idx, sub_conds_out, sub_conds, window))
        return results

    def combine_context_window_results(self, x_in: torch.Tensor, sub_conds_out, sub_conds, window: IndexListContextWindow,
                                       window_idx: int, total_windows: int, timestep: torch.Tensor,
                                       conds_final: list[torch.Tensor], counts_final: list[torch.Tensor],
                                       biases_final: list[torch.Tensor]):
        # 根据 self.dim 切割输出，丢弃前缀部分
        prefix_len = window.prefix_length
        if prefix_len > 0:
            # 对于每个 cond 输出，在时间维度上切掉前 prefix_len 帧
            body_out = []
            for tensor in sub_conds_out:
                # tensor 的形状: (..., time_len, ...)，其中 time_len = prefix_len + original_len
                # 我们需要在 self.dim 维度上切片
                slices = [slice(None)] * tensor.ndim
                slices[self.dim] = slice(prefix_len, None)
                body_out.append(tensor[tuple(slices)])
        else:
            body_out = sub_conds_out

        # body_out 中的每个 tensor 现在在时间维度上的长度应为 len(window.original_indices)
        # 融合时使用原始窗口索引（不含前缀）
        if self.fuse_method.name == ContextFuseMethods.RELATIVE:
            # 移植官方加权运行平均逻辑：用窗口中心距离计算 bias，累加并平滑过渡
            original_list = window.original_indices
            center_min = min(original_list)
            center_max = max(original_list)
            for i in range(len(body_out)):
                for pos, idx in enumerate(original_list):
                    # bias = 1 - distance / max_distance，窗口中心权重最大
                    bias = 1 - abs(idx - (center_min + center_max) / 2) / ((center_max - center_min + 1e-2) / 2)
                    bias = max(1e-2, bias)
                    bias_total = biases_final[i][idx]
                    prev_weight = bias_total / (bias_total + bias)
                    new_weight = bias / (bias_total + bias)
                    # 构造索引切片
                    idx_window = tuple([slice(None)] * self.dim + [idx])
                    pos_window = tuple([slice(None)] * self.dim + [pos])
                    # 加权运行平均
                    conds_final[i][idx_window] = conds_final[i][idx_window] * prev_weight + body_out[i][pos_window] * new_weight
                    biases_final[i][idx] = bias_total + bias
        else:
            # 为每个主体部分生成权重（长度 = len(window.original_indices)）
            weights = get_context_weights(len(window.original_indices), x_in.shape[self.dim],
                                          window.original_indices, self, sigma=timestep)
            weights_tensor = match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)

            # 创建一个临时窗口，只包含原始索引，用于 add_window
            temp_window = IndexListContextWindow(window.original_indices, dim=self.dim, total_frames=window.total_frames)

            for i in range(len(body_out)):
                # 确保 body_out[i] 在时间维度上的长度与 weights_tensor 一致
                # 它们应该已经都是 (..., original_len, ...)
                temp_window.add_window(conds_final[i], body_out[i] * weights_tensor)
                temp_window.add_window(counts_final[i], weights_tensor)

        # 回调（可选）
        for callback in comfy.patcher_extension.get_all_callbacks(IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks):
            callback(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep, conds_final, counts_final, biases_final)

def _prepare_sampling_wrapper(executor, model, noise_shape: torch.Tensor, *args, **kwargs):
    model_options = kwargs.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in prepare_sampling_wrapper")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is not None:
        noise_shape = list(noise_shape)
        # 如果使用了前缀，实际输入到模型的长度是 prefix_len + context_length
        prefix_len = len(handler.prefix_indices) if handler.prefix_indices else 0
        total_len = prefix_len + handler.context_length
        noise_shape[handler.dim] = min(noise_shape[handler.dim], total_len)
    return executor(model, noise_shape, *args, **kwargs)


def create_prepare_sampling_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        "ContextWindows_prepare_sampling",
        _prepare_sampling_wrapper
    )


def _sampler_sample_wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
    model_options = extra_args.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in sampler_sample_wrapper")
    handler: IndexListContextHandler = model_options.get("context_handler", None)
    if handler is None:
        raise Exception("context_handler not found")
    if not handler.freenoise:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)

    # 对噪声应用 FreeNoise，前缀帧保持原样，只 shuffle body 部分
    seed = 0
    if "seed" in extra_args:
        seed = extra_args["seed"]
    elif hasattr(guider, "patcher") and hasattr(guider.patcher, "model_options"):
        seed_from_opts = guider.patcher.model_options.get("seed", None)
        if seed_from_opts is not None:
            seed = seed_from_opts

    # 动态种子：不同 timestep 使用不同混洗模式，避免系统性偏差累积
    dynamic_seed = seed + handler._step * 997

    noise = apply_freenoise(
        noise,
        dim=handler.dim,
        context_length=handler.context_length,
        context_overlap=handler.context_overlap,
        seed=dynamic_seed,
        prefix_len=handler.prefix_latent_len
    )
    return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)


def create_sampler_sample_wrapper(model: ModelPatcher):
    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        "ContextWindows_sampler_sample",
        _sampler_sample_wrapper
    )


def match_weights_to_dim(weights: list[float], x_in: torch.Tensor, dim: int, device=None) -> torch.Tensor:
    total_dims = len(x_in.shape)
    weights_tensor = torch.Tensor(weights).to(device=device)
    for _ in range(dim):
        weights_tensor = weights_tensor.unsqueeze(0)
    for _ in range(total_dims - dim - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    return weights_tensor


def get_shape_for_dim(x_in: torch.Tensor, dim: int) -> list[int]:
    total_dims = len(x_in.shape)
    shape = []
    for _ in range(dim):
        shape.append(1)
    shape.append(x_in.shape[dim])
    for _ in range(total_dims - dim - 1):
        shape.append(1)
    return shape


class ContextSchedules:
    UNIFORM_LOOPED = "looped_uniform"
    UNIFORM_STANDARD = "standard_uniform"
    STATIC_STANDARD = "standard_static"
    BATCHED = "batched"


def create_windows_uniform_looped(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames < handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (0 if handler.closed_loop else -handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])
    return windows


def create_windows_uniform_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    context_stride = min(handler.context_stride, int(np.ceil(np.log2(num_frames / handler.context_length))) + 1)
    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(handler._step)))
        for j in range(
            int(ordered_halving(handler._step) * context_step) + pad,
            num_frames + pad + (-handler.context_overlap),
            (handler.context_length * context_step - handler.context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + handler.context_length * context_step, context_step)])
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            if roll_val not in windows[(win_i + 1) % len(windows)]:
                windows.insert(win_i + 1, list(range(roll_val, roll_val + handler.context_length)))
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)
    return windows


def create_windows_static_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    delta = handler.context_length - handler.context_overlap
    for start_idx in range(0, num_frames, delta):
        ending = start_idx + handler.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + handler.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + handler.context_length)))
    return windows


def create_windows_batched(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []
    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows
    for start_idx in range(0, num_frames, handler.context_length):
        windows.append(list(range(start_idx, min(start_idx + handler.context_length, num_frames))))
    return windows


def create_windows_default(num_frames: int, handler: IndexListContextHandler):
    return [list(range(num_frames))]


CONTEXT_MAPPING = {
    ContextSchedules.UNIFORM_LOOPED: create_windows_uniform_looped,
    ContextSchedules.UNIFORM_STANDARD: create_windows_uniform_standard,
    ContextSchedules.STATIC_STANDARD: create_windows_static_standard,
    ContextSchedules.BATCHED: create_windows_batched,
}


def get_matching_context_schedule(context_schedule: str) -> ContextSchedule:
    func = CONTEXT_MAPPING.get(context_schedule, None)
    if func is None:
        raise ValueError(f"Unknown context_schedule '{context_schedule}'.")
    return ContextSchedule(context_schedule, func)


def get_context_weights(length: int, full_length: int, idxs: list[int], handler: IndexListContextHandler, sigma: torch.Tensor = None):
    return handler.fuse_method.func(length, sigma=sigma, handler=handler, full_length=full_length, idxs=idxs)


def create_weights_flat(length: int, **kwargs) -> list[float]:
    return [1.0] * length


def create_weights_pyramid(length: int, **kwargs) -> list[float]:
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


def create_weights_overlap_linear(length: int, full_length: int, idxs: list[int], handler: IndexListContextHandler, **kwargs):
    weights_torch = torch.ones((length))
    if min(idxs) > 0:
        ramp_up = torch.linspace(1e-37, 1, handler.context_overlap)
        weights_torch[:handler.context_overlap] = ramp_up
    if max(idxs) < full_length - 1:
        ramp_down = torch.linspace(1, 1e-37, handler.context_overlap)
        weights_torch[-handler.context_overlap:] = ramp_down
    return weights_torch


class ContextFuseMethods:
    FLAT = "flat"
    PYRAMID = "pyramid"
    RELATIVE = "relative"
    OVERLAP_LINEAR = "overlap-linear"

    LIST = [PYRAMID, FLAT, OVERLAP_LINEAR]
    LIST_STATIC = [PYRAMID, RELATIVE, FLAT, OVERLAP_LINEAR]


FUSE_MAPPING = {
    ContextFuseMethods.FLAT: create_weights_flat,
    ContextFuseMethods.PYRAMID: create_weights_pyramid,
    ContextFuseMethods.RELATIVE: create_weights_pyramid,
    ContextFuseMethods.OVERLAP_LINEAR: create_weights_overlap_linear,
}


def get_matching_fuse_method(fuse_method: str) -> ContextFuseMethod:
    func = FUSE_MAPPING.get(fuse_method, None)
    if func is None:
        raise ValueError(f"Unknown fuse_method '{fuse_method}'.")
    return ContextFuseMethod(fuse_method, func)


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)
    return as_int / (1 << 64)


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        window[i] = window[i] + end_delta


def apply_freenoise(noise: torch.Tensor, dim: int, context_length: int, context_overlap: int, seed: int, prefix_len: int = 0):
    logging.info("Context windows: Applying FreeNoise" + (f" (prefix_len={prefix_len}, preserved)" if prefix_len > 0 else ""))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latent_video_length = noise.shape[dim]
    delta = context_length - context_overlap

    for start_idx in range(prefix_len, latent_video_length - context_length, delta):
        place_idx = start_idx + context_length
        actual_delta = min(delta, latent_video_length - place_idx)
        if actual_delta <= 0:
            break

        list_idx = torch.randperm(actual_delta, generator=generator, device='cpu') + start_idx

        source_slice = [slice(None)] * noise.ndim
        source_slice[dim] = list_idx
        target_slice = [slice(None)] * noise.ndim
        target_slice[dim] = slice(place_idx, place_idx + actual_delta)

        noise[tuple(target_slice)] = noise[tuple(source_slice)]

    return noise


# ==================== 节点定义 ====================
# 标准 ComfyUI 节点：使用 INPUT_TYPES / RETURN_TYPES / FUNCTION 等类属性（非 comfy_api）
class CustomWanContextWindowsManualNode:
    """手动设置 WAN 类模型的上下文窗口（dim=2），支持前缀参考帧。"""

    @classmethod
    def INPUT_TYPES(s):
        schedule_options = [
            ContextSchedules.UNIFORM_LOOPED,
            ContextSchedules.UNIFORM_STANDARD,
            ContextSchedules.STATIC_STANDARD,
            ContextSchedules.BATCHED,
        ]
        fuse_options = ContextFuseMethods.LIST_STATIC
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply context windows to during sampling."}),
                "context_length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                                           "tooltip": "The length of the context window (without prefix)."}),
                "context_overlap": ("INT", {"default": 30, "min": 0, "max": nodes.MAX_RESOLUTION,
                                            "tooltip": "The overlap of the context window."}),
                "context_schedule": (schedule_options, {"tooltip": "The schedule for generating windows."}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 10,
                                           "tooltip": "The stride of the context window; only applicable to uniform schedules."}),
                "closed_loop": ("BOOLEAN", {"default": False,
                                            "tooltip": "Whether to close the context window loop; only applicable to looped schedules."}),
                "fuse_method": (fuse_options, {"default": ContextFuseMethods.PYRAMID,
                                               "tooltip": "The method to use to fuse the context windows."}),
                "freenoise": ("BOOLEAN", {"default": False,
                                          "tooltip": "Whether to apply FreeNoise noise shuffling, improves window blending."}),
                "prefix_frames": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1,
                                          "tooltip": "Number of prefix reference frames (original frame units, auto-converted to latent). These frames are prepended to each window as stable reference."}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Whether to split multiple conditionings to each window based on region index."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_context_windows"
    CATEGORY = "context"
    EXPERIMENTAL = True
    DESCRIPTION = "Manually set context windows for WAN-like models (dim=2). Supports prefixing reference frames."

    def apply_context_windows(self, model, context_length, context_overlap, context_schedule,
                              context_stride, closed_loop, fuse_method, freenoise,
                              prefix_frames=0, split_conds_to_windows=False):
        # 转换前缀帧数：原始帧 → latent 帧（每 4 个原始帧对应 1 个 latent 帧）
        prefix_latent_len = max((prefix_frames - 1) // 4 + 1, 0) if prefix_frames > 0 else 0
        # 转换 WAN 模型的帧步长
        context_length = max(((context_length - 1) // 4) + 1, 1)
        context_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        model = model.clone()
        model.model_options["context_handler"] = IndexListContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=2,
            freenoise=freenoise,
            prefix_latent_len=prefix_latent_len,
            split_conds_to_windows=split_conds_to_windows
        )
        create_prepare_sampling_wrapper(model)
        if freenoise:
            create_sampler_sample_wrapper(model)
        return (model,)


# ==================== 注册 ====================
NODE_CLASS_MAPPINGS = {
    "WanContextWindowsManual_Custom": CustomWanContextWindowsManualNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanContextWindowsManual_Custom": "WAN Context Windows (Manual Custom)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
