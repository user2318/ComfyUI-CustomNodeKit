# ==============================
# WanUni3CCameraControl - 在官方 KSampler 中实现 Uni3C 运镜控制
# 
# v4 架构：monkey-patch 模型 forward_orig
# 直接在模型内部注入 control_states（与 KJ 完全相同的方式）
# 彻㐏解决了 temb 不一致问题（直接用模型内部的 e）和注入顺序问题
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
import logging
import sys
import os
import types
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from tqdm import tqdm
from einops import rearrange
from diffusers.models import ModelMixin
from diffusers.models.attention_processor import Attention
from typing import Optional, Tuple, Union
import numpy as np
from comfy.ldm.wan.model import sinusoidal_embedding_1d


# ============================================================
# 内联的 WanControlNet 相关类（从 uni3c/controlnet.py 移植）
# ============================================================

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )
    freqs = torch.outer(pos, freqs)
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )
        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class SimpleAttnProcessor2_0:
    def __init__(self, attention_mode="sdpa"):
        self.attention_mode = attention_mode

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states, freqs):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float32).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SimpleCogVideoXLayerNormZero(nn.Module):
    def __init__(self, conditioning_dim: int, embedding_dim: int, elementwise_affine: bool = True, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]


class SingleAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, time_embed_dim=512, qk_norm="rms_norm_across_heads", eps=1e-6, attention_mode="sdpa"):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.norm1 = SimpleCogVideoXLayerNormZero(time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True)
        self.self_attn = Attention(query_dim=dim, heads=num_heads, kv_heads=num_heads, dim_head=dim // num_heads, qk_norm=qk_norm, eps=eps, bias=True, cross_attention_dim=None, out_bias=True, processor=SimpleAttnProcessor2_0(attention_mode))
        self.norm2 = SimpleCogVideoXLayerNormZero(time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))

    def forward(self, hidden_states, temb, rotary_emb):
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)
        attn_hidden_states = self.self_attn(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + gate_ff * ff_output
        return hidden_states


class MaskCamEmbed(nn.Module):
    def __init__(self, controlnet_cfg) -> None:
        super().__init__()
        if controlnet_cfg.get("interp", False):
            self.mask_padding = [0, 0, 0, 0, 3, 3]
        else:
            self.mask_padding = [0, 0, 0, 0, 3, 0]
        add_channels = controlnet_cfg.get("add_channels", 1)
        mid_channels = controlnet_cfg.get("mid_channels", 64)
        self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8)), nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = nn.Conv3d(mid_channels, controlnet_cfg["conv_out_dim"], kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, add_inputs: torch.Tensor):
        warp_add_pad = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(warp_add_pad)
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = rearrange(add_embeds, "b c f h w -> b (f h w) c")
        return add_embeds


class WanControlNet(ModelMixin):
    def __init__(self, controlnet_cfg):
        super().__init__()
        self.rope_max_seq_len = 1024
        self.patch_size = (1, 2, 2)
        self.in_channels = controlnet_cfg["in_channels"]
        self.dim = controlnet_cfg["dim"]
        self.num_heads = controlnet_cfg["num_heads"]
        self.quantized = controlnet_cfg.get("quantized", False)
        self.base_dtype = controlnet_cfg.get("base_dtype", torch.float16)
        if controlnet_cfg["conv_out_dim"] != controlnet_cfg["dim"]:
            self.proj_in = nn.Linear(controlnet_cfg["conv_out_dim"], controlnet_cfg["dim"])
        else:
            self.proj_in = nn.Identity()
        self.controlnet_blocks = nn.ModuleList([
            SingleAttentionBlock(dim=self.dim, ffn_dim=controlnet_cfg["ffn_dim"], num_heads=self.num_heads, time_embed_dim=controlnet_cfg["time_embed_dim"], qk_norm="rms_norm_across_heads", attention_mode=controlnet_cfg.get("attention_mode", "sdpa"))
            for _ in range(controlnet_cfg["num_layers"])
        ])
        self.proj_out = nn.ModuleList([
            nn.Linear(self.dim, 5120) for _ in range(controlnet_cfg["num_layers"])
        ])
        self.gradient_checkpointing = False
        self.controlnet_rope = WanRotaryPosEmbed(self.dim // self.num_heads, self.patch_size, self.rope_max_seq_len)
        self.controlnet_patch_embedding = nn.Conv3d(self.in_channels, controlnet_cfg["conv_out_dim"], kernel_size=self.patch_size, stride=self.patch_size, dtype=torch.float32)
        self.controlnet_mask_embedding = MaskCamEmbed(controlnet_cfg)

    def forward(self, render_latent, render_mask, camera_embedding, temb, out_device):
        controlnet_rotary_emb = self.controlnet_rope(render_latent)
        controlnet_inputs = self.controlnet_patch_embedding(render_latent.to(torch.float32))
        if not self.quantized:
            controlnet_inputs = controlnet_inputs.to(render_latent.dtype)
        else:
            controlnet_inputs = controlnet_inputs.to(self.base_dtype)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)
        add_inputs = None
        if camera_embedding is not None and render_mask is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        elif render_mask is not None:
            add_inputs = render_mask
        if add_inputs is not None:
            add_inputs = self.controlnet_mask_embedding(add_inputs)
            controlnet_inputs = controlnet_inputs + add_inputs
        hidden_states = self.proj_in(controlnet_inputs)
        controlnet_states = []
        for i, block in enumerate(self.controlnet_blocks):
            hidden_states = block(hidden_states=hidden_states, temb=temb, rotary_emb=controlnet_rotary_emb)
            controlnet_states.append(self.proj_out[i](hidden_states).to("cpu"))
        return controlnet_states


# ============================================================
# 自定义节点
# ============================================================

class WanUni3CLoader:
    """加载 Uni3C ControlNet 模型"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("controlnet"), {
                    "tooltip": "Uni3C ControlNet 模型文件，放在 ComfyUI/models/controlnet 文件夹"
                }),
            },
            "optional": {
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            }
        }
    RETURN_TYPES = ("UNI3C_CONTROLNET",)
    RETURN_NAMES = ("uni3c_controlnet",)
    FUNCTION = "load"
    CATEGORY = "WanVideo/Control"

    def load(self, model_name, base_precision="fp16"):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[base_precision]
        model_path = folder_paths.get_full_path_or_raise("controlnet", model_name)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        if "controlnet_patch_embedding.weight" not in sd:
            raise ValueError("无效的 Uni3C ControlNet 模型文件")
        in_channels = sd["controlnet_patch_embedding.weight"].shape[1]
        ffn_dim = sd["controlnet_blocks.0.ffn.0.bias"].shape[0]
        controlnet_cfg = {
            "in_channels": in_channels, "conv_out_dim": 5120, "time_embed_dim": 5120, "dim": 1024,
            "ffn_dim": ffn_dim, "num_heads": 16, "num_layers": 20, "add_channels": 7,
            "mid_channels": 256, "attention_mode": "sdpa", "quantized": False, "base_dtype": base_dtype
        }
        if "controlnet_mask_embedding.mask_proj.0.weight" in sd:
            mask_weight = sd["controlnet_mask_embedding.mask_proj.0.weight"]
            controlnet_cfg["add_channels"] = mask_weight.shape[1]
            logging.info(f"[Uni3C] 检测到额外输入通道: {controlnet_cfg['add_channels']}")
        with init_empty_weights():
            controlnet = WanControlNet(controlnet_cfg)
        controlnet.eval()
        dtype = base_dtype
        params_to_keep = {"norm", "head", "time_in", "vector_in", "controlnet_patch_embedding",
                          "time_", "img_emb", "modulation", "text_embedding", "adapter", "proj_in"}
        logging.info("[Uni3C] 加载权重到设备...")
        param_count = sum(1 for _ in controlnet.named_parameters())
        for name, param in tqdm(controlnet.named_parameters(), desc="Loading Uni3C ControlNet", total=param_count, leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            if "controlnet_patch_embedding" in name:
                dtype_to_use = torch.float32
            set_module_tensor_to_device(controlnet, name, device=offload_device, dtype=dtype_to_use, value=sd[name])
        del sd
        gc.collect()
        mm.soft_empty_cache()
        return ({"controlnet": controlnet, "config": controlnet_cfg, "dtype": base_dtype},)


class WanUni3CApply:
    """
    v4 架构：monkey-patch 模型 forward_orig，在模型内部注入 control_states。
    
    原理：
    与 KJ 完全一致——在模型 forward_orig 的 block 循环中，每个 block 输出后
    立即注入 control_states，然后由 face_adapter 处理。
    
    关键优势：
    1. temb 直接用模型内部的 `e`，与 KJ 一字不差
    2. 注入顺序 block→注入→face_adapter，与 KJ 完全相同
    3. render_latent 用初始噪声 `x[0]` 拼接并缓存复用
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "uni3c_controlnet": ("UNI3C_CONTROLNET",),
                "render_latent": ("LATENT", {"tooltip": "预渲染的参考视频潜空间张量 (B, C, T, H, W)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "render_mask": ("MASK", {"tooltip": "可选的渲染遮罩（实验性）"}),
                "trim_latent": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "参考图占用的 latent 帧数（从 WanAnimateToVideoCustom 的 trim_latent 接入），用于时序对齐"}),
                "positive": ("CONDITIONING", {"tooltip": "可选：传入 conditioning 自动提取 concat_latent_image，替代 WanAnimateChannelPack 节点"}),
                "negative": ("CONDITIONING", {"tooltip": "可选：传入 conditioning 自动提取 concat_latent_image，替代 WanAnimateChannelPack 节点"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "WanVideo/Control"

    def apply(self, model, uni3c_controlnet, render_latent, strength,
              start_percent=0.0, end_percent=1.0, render_mask=None, trim_latent=0,
              positive=None, negative=None):
        
        # ===== 内联 WanAnimateChannelPack 逻辑：从 conditioning 提取 concat 数据 =====
        if positive is not None and negative is not None:
            concat_data = {"concat_latent_image": None, "concat_mask": None}
            for cond_list, name in [(positive, "positive"), (negative, "negative")]:
                for cond in cond_list:
                    cond_dict = cond[1] if isinstance(cond, (list, tuple)) and len(cond) > 1 else {}
                    if not isinstance(cond_dict, dict):
                        continue
                    for key in ("concat_latent_image", "concat_mask"):
                        if key in cond_dict and concat_data.get(key) is None:
                            concat_data[key] = cond_dict[key]
                            logging.info(f"[Uni3C] 从 {name} 提取 {key}: shape={cond_dict[key].shape}")
            if concat_data["concat_latent_image"] is not None:
                to = model.model_options.setdefault("transformer_options", {})
                to["wananim_concat_latent_image"] = concat_data["concat_latent_image"]
                to["wananim_concat_mask"] = concat_data.get("concat_mask")
                model.model_options["transformer_options"] = to
                logging.info(f"[Uni3C] 已从 conditioning 提取 concat_latent_image: "
                            f"{concat_data['concat_latent_image'].shape}")
            else:
                logging.warning("[Uni3C] 提供了 positive/negative 但未找到 concat_latent_image，"
                              "Animate 模型的通道拼接将不会执行")
        # ===== 内联结束 =====
        
        render_latent_tensor = render_latent["samples"].clone()
        is_valid_render = not torch.allclose(render_latent_tensor, torch.zeros_like(render_latent_tensor), atol=1e-3)
        if not is_valid_render:
            logging.warning(f"[Uni3C] render_latent 近似全零，跳过 ControlNet")
        
        offload_device = mm.unet_offload_device()
        controlnet = uni3c_controlnet["controlnet"]
        controlnet.to(offload_device)
        
        model = model.clone()
        
        if trim_latent > 0:
            logging.info(f"[Uni3C] 用户指定 trim_latent={trim_latent}，将自动对齐 render_latent 时序")
        
        # 获取 transformer
        inner = model.model
        if hasattr(inner, 'diffusion_model'):
            transformer = inner.diffusion_model
        else:
            transformer = inner
        
        # 保存原始 forward_orig 引用
        self._orig_forward_orig = transformer.forward_orig
        
        # 控制数据
        cn_data = {
            "controlnet": controlnet,
            "config": uni3c_controlnet["config"],
            "dtype": uni3c_controlnet["dtype"],
            "render_latent": render_latent_tensor,
            "strength": strength,
            "start_pct": start_percent,
            "end_pct": end_percent,
            "skip_cn": not is_valid_render,
            "offload_device": offload_device,
            "total_steps": None,
            "trim_latent": trim_latent,
        }
        
        # monkey-patch forward_orig
        def patched_forward_orig(self_model, x, t, context, clip_fea=None,
                                  pose_latents=None, face_pixel_values=None,
                                  freqs=None, transformer_options={}, **kwargs):
            
            nonlocal cn_data
            
            # embeddings（与原始 forward_orig 一致）
            x_p = self_model.patch_embedding(x.float()).to(x.dtype)
            x_p, motion_vec = self_model.after_patch_embedding(x_p, pose_latents, face_pixel_values)
            grid_sizes = x_p.shape[2:]
            x_p = x_p.flatten(2).transpose(1, 2)
            original_seq_len = x_p.shape[1]
            
            # time embeddings（关键：e 就是 KJ 传入 ControlNet 的 temb）
            e = self_model.time_embedding(
                sinusoidal_embedding_1d(self_model.freq_dim, t.flatten()).to(dtype=x_p[0].dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self_model.time_projection(e).unflatten(2, (6, self_model.dim))
            
            # full ref
            full_ref = None
            if self_model.ref_conv is not None:
                full_ref = kwargs.get("reference_latent", None)
                if full_ref is not None:
                    full_ref = self_model.ref_conv(full_ref).flatten(2).transpose(1, 2)
                    x_p = torch.concat((full_ref, x_p), dim=1)
            
            # context
            context = self_model.text_embedding(context)
            context_img_len = None
            if clip_fea is not None:
                if self_model.img_emb is not None:
                    context_clip = self_model.img_emb(clip_fea)
                    context = torch.concat([context_clip, context], dim=1)
                context_img_len = clip_fea.shape[-2]
            
            # ===== step_percentage 计算 =====
            sigmas = transformer_options.get("sigmas", None)
            if sigmas is not None:
                if cn_data["total_steps"] is None:
                    cn_data["total_steps"] = len(sigmas)
            if sigmas is not None and cn_data["total_steps"] is not None:
                current_step_index = cn_data["total_steps"] - len(sigmas)
                step_pct = current_step_index / cn_data["total_steps"] if cn_data["total_steps"] > 0 else 0.0
            else:
                step_pct = 0.5
            
            in_range = cn_data["start_pct"] <= step_pct <= cn_data["end_pct"]
            skip = cn_data["skip_cn"] or not in_range
            
            # ===== ControlNet 推理（每 step 一次，基于完整视频） =====
            uni3c_controlnet_states = None
            if not skip:
                try:
                    device = x_p.device
                    controlnet = cn_data["controlnet"]
                    controlnet.to(device)
                    
                    # 获取上下文窗口信息
                    window = transformer_options.get("context_window", None)
                    
                    # 构建 render_latent_input（基于完整视频，不按窗口切片）
                    # 用 x（patch 之前的原始输入）的初始噪声拼接 render_latent
                    noise_part = x[0:1].to(device=device, dtype=cn_data["dtype"])
                    # 始终只取前 16 通道
                    noise_part = noise_part[:, :16]
                    # 补零到 20 通道
                    pad = torch.zeros(1, 20 - noise_part.shape[1],
                                    noise_part.shape[2], noise_part.shape[3], noise_part.shape[4],
                                    device=noise_part.device, dtype=noise_part.dtype)
                    noise_part = torch.cat([noise_part, pad], dim=1)
                    rl = cn_data["render_latent"].to(device, cn_data["dtype"])
                    
                    # 取完整 render_latent（前 16 通道）
                    rl_slice = rl[0:1, :16].to(device=device, dtype=cn_data["dtype"])
                    
                    # 前缀补齐：补 trim_latent 帧重复（与 pose_video 的 delta 对齐）
                    # 补齐后总帧数 = render_latent_motion_frames + trim_latent，与 concat 总帧数一致
                    trim = cn_data.get("trim_latent", 0)
                    if trim > 0 and rl_slice.shape[2] >= 1:
                        first_frame = rl_slice[:, :, :1, :, :]
                        padding_frames = first_frame.repeat(1, 1, trim, 1, 1)
                        rl_slice = torch.cat([padding_frames, rl_slice], dim=2)
                    
                    if rl_slice.shape[2] != noise_part.shape[2]:
                        rl_slice = F.interpolate(rl_slice, size=(noise_part.shape[2], noise_part.shape[3], noise_part.shape[4]), mode='trilinear', align_corners=False)
                    render_latent_input = torch.cat([noise_part, rl_slice], dim=1)
                    del rl, noise_part, rl_slice
                    
                    # 直接用模型内部的 e 作为 temb
                    temb = e.squeeze(1).to(cn_data["dtype"])
                    
                    uni3c_controlnet_states = controlnet(
                        render_latent_input, None, None, temb, device
                    )
                    
                    del render_latent_input, temb
                    gc.collect()
                    mm.soft_empty_cache()
                    
                    if uni3c_controlnet_states is not None:
                        cs_means = [cs.mean().item() for cs in uni3c_controlnet_states if cs is not None]
                        cs_stds = [cs.std().item() for cs in uni3c_controlnet_states if cs is not None]
                        logging.info(f"[Uni3C] 步骤 {step_pct:.2f}: "
                                    f"已计算 {len(uni3c_controlnet_states)} 个控制信号, "
                                    f"mean={sum(cs_means)/len(cs_means):.6f}, "
                                    f"std={sum(cs_stds)/len(cs_stds):.6f}")
                    
                except Exception as ex:
                    logging.error(f"[Uni3C] ControlNet 推理失败: {ex}")
                    import traceback
                    traceback.print_exc()
                    uni3c_controlnet_states = None
            
            # ===== Block 循环 + 注入（与 KJ 完全一致，异步预取 CPU→GPU） =====
            # 预取第一块控制信号（与第一个 block 计算重叠）
            cs_prefetched = None
            if uni3c_controlnet_states is not None and len(uni3c_controlnet_states) > 0:
                cs_prefetched = uni3c_controlnet_states[0].to(x_p.device, x_p.dtype, non_blocking=True)
            
            for i, block in enumerate(self_model.blocks):
                # 使用已预取的控制信号
                cs = cs_prefetched
                # 预取下一块
                cs_prefetched = None
                if uni3c_controlnet_states is not None and i + 1 < len(uni3c_controlnet_states):
                    cs_prefetched = uni3c_controlnet_states[i + 1].to(x_p.device, x_p.dtype, non_blocking=True)
                
                x_p = block(x_p, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)
                
                # 注入 control_states（block 后，face_adapter 前，与 KJ 完全相同）
                if cs is not None:
                    cs_dim = cs.shape[-1]
                    x_dim = x_p.shape[-1]
                    if cs_dim != x_dim:
                        if cs_dim > x_dim:
                            cs = cs[..., :x_dim]
                        else:
                            pad_cs = torch.zeros(*cs.shape[:-1], x_dim - cs_dim, device=cs.device, dtype=cs.dtype)
                            cs = torch.cat([cs, pad_cs], dim=-1)
                    x_p[:, :original_seq_len] += cs[:, :original_seq_len] * cn_data["strength"]
                
                # face_adapter
                if i % 5 == 0 and motion_vec is not None:
                    x_p = x_p + self_model.face_adapter.fuser_blocks[i // 5](x_p, motion_vec)
            
            # head
            x_p = self_model.head(x_p, e)
            if full_ref is not None:
                x_p = x_p[:, full_ref.shape[1]:]
            x_p = self_model.unpatchify(x_p, grid_sizes)
            return x_p
        
        # 绑定补丁函数到实例方法
        transformer.forward_orig = types.MethodType(patched_forward_orig, transformer)
        
        logging.info(f"[Uni3C] monkey-patch forward_orig 已应用: "
                    f"strength={strength}, start={start_percent}, end={end_percent}, "
                    f"render_latent_shape={render_latent_tensor.shape}")
        
        return (model,)


NODE_CLASS_MAPPINGS = {
    "WanUni3CLoader": WanUni3CLoader,
    "WanUni3CApply": WanUni3CApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanUni3CLoader": "WanVideo Uni3C ControlNet Loader (Custom)",
    "WanUni3CApply": "WanVideo Uni3C Apply (for KSampler)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]