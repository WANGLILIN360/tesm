"""
位置编码对比实验：显式位置编码 vs 卷积隐式位置编码

实验设计：
1. 方案 A：Position Embedding + RoPE（当前方案）
2. 方案 B：Conv1d + RoPE（卷积隐式方案）

测试指标：
- 位置敏感性：相同 token 在不同位置的输出差异
- 长度泛化：训练长度 vs 推理长度的表现
- 因果性：是否正确处理因果依赖
- 训练收敛：loss 下降速度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional
from functools import partial

# 导入原始模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.modules.tesm import BitLinear, TESM


# ==================== 自定义 RMSNorm (支持 device) ====================

class RMSNorm(nn.Module):
    """支持 device 参数的 RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


# ==================== 卷积版本的 TESM ====================

class TESMConv(TESM):
    """使用卷积隐式位置编码的 TESM"""
    
    def __init__(
        self,
        d_model,
        d_state=256,
        expand=2,
        ent_rank=64,
        entanglement_scale=0.2,
        entanglement_threshold=0.1,
        max_seq_len=2048,
        dropout=0.0,
        bit_eps=1e-5,
        bit_threshold=0.5,
        conv_kernel_size=3,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        # 调用父类初始化
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            ent_rank=ent_rank,
            entanglement_scale=entanglement_scale,
            entanglement_threshold=entanglement_threshold,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bit_eps=bit_eps,
            bit_threshold=bit_threshold,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        
        # 添加因果卷积层（隐式位置编码）
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel_size,
            groups=d_model,
            padding=conv_kernel_size - 1,
            device=device,
            dtype=dtype
        )
        self.conv_kernel_size = conv_kernel_size
    
    def forward(self, u, inference_params=None, cross_layer_state=None):
        batch, seqlen, _ = u.shape
        
        # 增量推理：如果是单 token 且有缓存，使用增量计算
        if inference_params is not None and seqlen == 1:
            layer_cache = inference_params.get('state_cache')
            if layer_cache is not None and 'state' in layer_cache:
                return self._forward_incremental(u, inference_params, cross_layer_state)
        
        # ===== 卷积隐式位置编码 =====
        u_conv = u.transpose(1, 2)  # (B, D, L)
        u_conv = self.conv1d(u_conv)[:, :, :seqlen]  # causal: 移除 padding
        u = u_conv.transpose(1, 2)  # (B, L, D)
        
        # 后续与父类相同
        proj = self.in_proj(u)
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(
            proj,
            [self.d_model, self.d_state, self.d_state, self.d_state, self.d_model, self.ent_rank, self.ent_rank],
            dim=-1,
        )
        
        state_value = torch.tanh(state_value)
        decay = torch.sigmoid(decay + self.decay_bias)
        write = torch.sigmoid(write)
        out_gate = torch.sigmoid(out_gate)
        
        # 位置纠缠: RoPE 使纠缠模式随位置变化
        ent_q = self._apply_rope(ent_q)
        ent_k = self._apply_rope(ent_k)
        
        # 跨层纠缠: 上层逐位置状态偏置当前层的 Q
        if cross_layer_state is not None:
            cross_q_bias = self.cross_layer_q_proj(cross_layer_state)
            if cross_q_bias.dim() == 2:
                cross_q_bias = cross_q_bias.unsqueeze(1)
            ent_q = ent_q + cross_q_bias
        
        # Phase 1: 纯状态积累
        update = write * state_value
        states = self._parallel_state_scan(decay, update)
        
        # Phase 2: 三值纠缠真实状态
        signed_avg = self._compute_entanglement(ent_q, ent_k, states)
        entangled_states = states + self.entanglement_scale * (signed_avg - states)
        ent_change = entangled_states - states
        
        # prefill 时写入推理缓存
        if inference_params is not None:
            cache = inference_params.get('state_cache')
            if cache is not None and 'state' in cache:
                last_scan_state_f64 = self._last_scan_state_f64
                if last_scan_state_f64 is None:
                    last_scan_state_f64 = states[:, -1, :].detach().to(torch.float64)
                cache['state'] = last_scan_state_f64.clone()
                cache['seq_pos'] = seqlen
                window = cache['ent_k_cache'].shape[1]
                if seqlen >= window:
                    cache['ent_k_cache'] = ent_k[:, -window:, :].detach().float()
                    cache['ent_v_cache'] = states[:, -window:, :].detach().float()
                else:
                    cache['ent_k_cache'][:, -seqlen:, :] = ent_k.detach().float()
                    cache['ent_v_cache'][:, -seqlen:, :] = states.detach().float()
        
        # Phase 3: 输出
        state_projected = self.state_proj(torch.tanh(entangled_states))
        ent_projected = self.ent_proj(ent_change)
        state_mixed = out_gate * state_projected
        ent_mixed = ent_projected
        y = local + state_mixed + ent_mixed
        y = self.out_proj(self.dropout(y))
        
        self._last_cross_layer_state = states.detach()
        return y
    
    def _forward_incremental(self, u, inference_params, cross_layer_state=None):
        """增量推理：卷积版本"""
        batch, seqlen, _ = u.shape
        cache = inference_params['state_cache']
        
        # 卷积需要缓存历史
        if 'conv_cache' not in cache:
            cache['conv_cache'] = torch.zeros(
                batch, self.d_model, self.conv_kernel_size - 1,
                device=u.device, dtype=u.dtype
            )
        
        # 拼接历史
        conv_input = torch.cat([cache['conv_cache'], u.transpose(1, 2)], dim=2)
        u_conv = self.conv1d(conv_input)[:, :, -1:]  # 只取最后一个
        u = u_conv.transpose(1, 2)
        
        # 更新卷积缓存
        cache['conv_cache'] = conv_input[:, :, -(self.conv_kernel_size - 1):]
        
        # 后续与父类相同
        proj = self.in_proj(u)
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(
            proj,
            [self.d_model, self.d_state, self.d_state, self.d_state, self.d_model, self.ent_rank, self.ent_rank],
            dim=-1,
        )
        
        state_value = torch.tanh(state_value)
        decay = torch.sigmoid(decay + self.decay_bias)
        write = torch.sigmoid(write)
        out_gate = torch.sigmoid(out_gate)
        
        cur_pos = cache['seq_pos']
        ent_q_rope = self._apply_rope(ent_q, pos_offset=cur_pos)
        ent_k_rope = self._apply_rope(ent_k, pos_offset=cur_pos)
        q_vec = ent_q_rope[:, 0, :]
        k_vec = ent_k_rope[:, 0, :]
        v_vec = state_value[:, 0, :]
        
        if cross_layer_state is not None:
            cross_q_bias = self.cross_layer_q_proj(cross_layer_state)
            q_vec = q_vec + cross_q_bias
        
        prev_state = cache['state']
        new_state_f64 = decay[:, 0, :].double() * prev_state + write[:, 0, :].double() * v_vec.double()
        new_state = new_state_f64.to(decay.dtype)
        
        cache['ent_k_cache'] = torch.cat([cache['ent_k_cache'][:, 1:, :], k_vec.unsqueeze(1)], dim=1)
        cache['ent_v_cache'] = torch.cat([cache['ent_v_cache'][:, 1:, :], new_state.unsqueeze(1)], dim=1)
        
        window = cache['ent_k_cache'].shape[1]
        seq_pos = cache['seq_pos']
        valid_len = min(seq_pos + 1, window)
        
        scores = torch.einsum('br,bwr->bw', q_vec, cache['ent_k_cache']) / (self.ent_rank ** 0.5)
        if self.local_entanglement_bias is not None:
            bias = self.local_entanglement_bias[-window:].to(dtype=scores.dtype, device=scores.device)
            scores = scores + bias
        if valid_len < window:
            scores[:, :window - valid_len] = 0.0
        ternary = self.ternary_entanglement(scores)
        if valid_len < window:
            ternary[:, :window - valid_len] = 0.0
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        signed_avg = torch.einsum('bw,bwd->bd', ternary / norm, cache['ent_v_cache'])
        entangled_state = new_state + self.entanglement_scale * (signed_avg - new_state)
        ent_change = entangled_state - new_state
        
        cache['state'] = new_state_f64.detach()
        cache['seq_pos'] += 1
        
        state_projected = self.state_proj(torch.tanh(entangled_state))
        ent_projected = self.ent_proj(ent_change)
        
        state_mixed = out_gate[:, 0, :] * state_projected
        ent_mixed = ent_projected
        y = local[:, 0, :] + state_mixed + ent_mixed
        y = self.out_proj(y.unsqueeze(1))
        
        self._last_cross_layer_state = new_state.detach()
        return y
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """分配增量推理的状态缓存（包含卷积缓存）"""
        cache = super().allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
        cache['conv_cache'] = torch.zeros(
            batch_size, self.d_model, self.conv_kernel_size - 1,
            device=self.out_proj.weight.device, dtype=dtype or torch.float32
        )
        return cache


# ==================== 卷积版本的 MixerModel ====================

class BlockConv(nn.Module):
    """使用卷积版本的 Block"""
    
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.norm2 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
    
    def forward(self, hidden_states, residual=None, inference_params=None, **mixer_kwargs):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self.mixer, "allocate_inference_cache"):
            return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None


class FeedForward(nn.Module):
    """FFN 层"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_backend = config.ssm_cfg.get("kernel_backend", "auto")
        kernel_mode = config.ssm_cfg.get("kernel_mode", "fast")
        self.gate_proj = BitLinear(config.d_model, config.d_intermediate, bias=False, 
                                   bit_eps=config.bit_eps, bit_threshold=config.bit_threshold, 
                                   kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.up_proj = BitLinear(config.d_model, config.d_intermediate, bias=False,
                                  bit_eps=config.bit_eps, bit_threshold=config.bit_threshold,
                                  kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.down_proj = BitLinear(config.d_intermediate, config.d_model, bias=False,
                                    bit_eps=config.bit_eps, bit_threshold=config.bit_threshold,
                                    kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        gate_x = self.gate_proj(x)
        up_x = self.up_proj(x)
        x = F.silu(gate_x) * up_x
        x = self.down_proj(x)
        return self.dropout(x)


class MixerModelConv(nn.Module):
    """使用卷积隐式位置编码的 MixerModel"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        # 无 Position Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        # 使用卷积版本的 TESM
        mixer_cls = lambda layer_idx: (
            lambda dim: TESMConv(
                d_model=dim,
                layer_idx=layer_idx,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bit_eps=config.bit_eps,
                bit_threshold=config.bit_threshold,
                **config.ssm_cfg,
                **factory_kwargs,
            )
        )
        mlp_cls = lambda dim: FeedForward(config, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            BlockConv(
                config.d_model,
                mixer_cls(layer_idx),
                mlp_cls,
                norm_cls=norm_cls,
                residual_in_fp32=config.residual_in_fp32,
            )
            for layer_idx in range(config.n_layer)
        ])
        
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
        
        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(config.d_model, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=config.rescale_prenorm_residual,
                n_residuals_per_layer=1,
            )
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
                for i, layer in enumerate(self.layers)}
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")
        
        # 无 Position Embedding
        hidden_states = self.embedding(input_ids)
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
            if self.gradient_checkpointing and self.training and inference_params is None:
                from torch.utils.checkpoint import checkpoint
                if residual is None:
                    hidden_states, residual = checkpoint(
                        lambda hs, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, None, inference_params=None, **_kw),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, residual = checkpoint(
                        lambda hs, res, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, res, inference_params=None, **_kw),
                        hidden_states,
                        residual,
                        use_reentrant=False,
                    )
            else:
                layer_inference_params = None
                if inference_params is not None and 'state_cache' in inference_params:
                    layer_cache = inference_params['state_cache'].get(i)
                    if layer_cache is not None:
                        layer_inference_params = {'state_cache': layer_cache}
                hidden_states, residual = layer(hidden_states, residual, inference_params=layer_inference_params, **layer_mixer_kwargs)
            
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats)


# ==================== 方案 C：只保留 RoPE ====================

class MixerModelRoPE(nn.Module):
    """只保留 RoPE，移除 Position Embedding 的 MixerModel"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        # 无 Position Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        # 使用原始 TESM（有 RoPE）
        mixer_cls = lambda layer_idx: (
            lambda dim: TESM(
                d_model=dim,
                layer_idx=layer_idx,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bit_eps=config.bit_eps,
                bit_threshold=config.bit_threshold,
                **config.ssm_cfg,
                **factory_kwargs,
            )
        )
        mlp_cls = lambda dim: FeedForward(config, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            BlockConv(
                config.d_model,
                mixer_cls(layer_idx),
                mlp_cls,
                norm_cls=norm_cls,
                residual_in_fp32=config.residual_in_fp32,
            )
            for layer_idx in range(config.n_layer)
        ])
        
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
        
        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(config.d_model, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=config.rescale_prenorm_residual,
                n_residuals_per_layer=1,
            )
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
                for i, layer in enumerate(self.layers)}
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")
        
        # 无 Position Embedding，只有 Token Embedding
        hidden_states = self.embedding(input_ids)
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
            if self.gradient_checkpointing and self.training and inference_params is None:
                from torch.utils.checkpoint import checkpoint
                if residual is None:
                    hidden_states, residual = checkpoint(
                        lambda hs, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, None, inference_params=None, **_kw),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, residual = checkpoint(
                        lambda hs, res, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, res, inference_params=None, **_kw),
                        hidden_states,
                        residual,
                        use_reentrant=False,
                    )
            else:
                layer_inference_params = None
                if inference_params is not None and 'state_cache' in inference_params:
                    layer_cache = inference_params['state_cache'].get(i)
                    if layer_cache is not None:
                        layer_inference_params = {'state_cache': layer_cache}
                hidden_states, residual = layer(hidden_states, residual, inference_params=layer_inference_params, **layer_mixer_kwargs)
            
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats)


class TESMLMHeadModelRoPE(nn.Module):
    """只保留 RoPE 的 LM 模型"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelRoPE(self.tesm_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(self.tesm_config.d_model, self.tesm_config.vocab_size, bias=False, device=device, dtype=dtype)
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def forward(self, input_ids, labels=None, inference_params=None, logits_to_keep=0, **kwargs):
        for k in ["attention_mask", "past_key_values", "use_cache"]:
            kwargs.pop(k, None)
        hidden_states, entanglement_maps, entanglement_stats = self.backbone(input_ids=input_ids, inference_params=inference_params, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tesm_config.label_ignore_index,
            )
        return TESMCausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states, 
                                   entanglement_maps=entanglement_maps, entanglement_stats=entanglement_stats)


# ==================== 方案 D：低维 Position Embedding ====================

class MixerModelLowDim(nn.Module):
    """低维 Position Embedding + 投影的 MixerModel"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None, pos_dim_ratio=4):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        self.pos_dim_ratio = pos_dim_ratio
        
        # Token Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        
        # 低维 Position Embedding + 投影
        pos_dim = config.d_model // pos_dim_ratio
        self.position_embedding_low = nn.Embedding(config.max_seq_len, pos_dim, **factory_kwargs)
        self.position_proj = nn.Linear(pos_dim, config.d_model, bias=False, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        # 使用原始 TESM（有 RoPE）
        mixer_cls = lambda layer_idx: (
            lambda dim: TESM(
                d_model=dim,
                layer_idx=layer_idx,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bit_eps=config.bit_eps,
                bit_threshold=config.bit_threshold,
                **config.ssm_cfg,
                **factory_kwargs,
            )
        )
        mlp_cls = lambda dim: FeedForward(config, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            BlockConv(
                config.d_model,
                mixer_cls(layer_idx),
                mlp_cls,
                norm_cls=norm_cls,
                residual_in_fp32=config.residual_in_fp32,
            )
            for layer_idx in range(config.n_layer)
        ])
        
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
        
        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(config.d_model, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=config.rescale_prenorm_residual,
                n_residuals_per_layer=1,
            )
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
                for i, layer in enumerate(self.layers)}
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")
        
        # 增量推理时用正确的位置偏移
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token Embedding + 低维 Position Embedding 投影
        hidden_states = self.embedding(input_ids)
        pos_embed_low = self.position_embedding_low(positions)  # (B, L, d_model // 4)
        pos_embed = self.position_proj(pos_embed_low)  # (B, L, d_model)
        hidden_states = hidden_states + pos_embed
        
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
            if self.gradient_checkpointing and self.training and inference_params is None:
                from torch.utils.checkpoint import checkpoint
                if residual is None:
                    hidden_states, residual = checkpoint(
                        lambda hs, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, None, inference_params=None, **_kw),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, residual = checkpoint(
                        lambda hs, res, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, res, inference_params=None, **_kw),
                        hidden_states,
                        residual,
                        use_reentrant=False,
                    )
            else:
                layer_inference_params = None
                if inference_params is not None and 'state_cache' in inference_params:
                    layer_cache = inference_params['state_cache'].get(i)
                    if layer_cache is not None:
                        layer_inference_params = {'state_cache': layer_cache}
                hidden_states, residual = layer(hidden_states, residual, inference_params=layer_inference_params, **layer_mixer_kwargs)
            
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats)


class TESMLMHeadModelLowDim(nn.Module):
    """低维 Position Embedding 的 LM 模型"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None, pos_dim_ratio=4):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelLowDim(self.tesm_config, device=device, dtype=dtype, pos_dim_ratio=pos_dim_ratio)
        self.lm_head = nn.Linear(self.tesm_config.d_model, self.tesm_config.vocab_size, bias=False, device=device, dtype=dtype)
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def forward(self, input_ids, labels=None, inference_params=None, logits_to_keep=0, **kwargs):
        for k in ["attention_mask", "past_key_values", "use_cache"]:
            kwargs.pop(k, None)
        hidden_states, entanglement_maps, entanglement_stats = self.backbone(input_ids=input_ids, inference_params=inference_params, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tesm_config.label_ignore_index,
            )
        return TESMCausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states, 
                                   entanglement_maps=entanglement_maps, entanglement_stats=entanglement_stats)


# ==================== 方案 E：Transformer Sinusoidal Position Encoding ====================

class SinusoidalPositionEncoding(nn.Module):
    """Transformer 原始的正弦位置编码"""
    
    def __init__(self, d_model, max_seq_len=2048, device=None, dtype=None):
        super().__init__()
        # 预计算正弦位置编码
        position = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype or torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, positions):
        # positions: (batch, seq_len)
        return self.pe[positions]  # (batch, seq_len, d_model)


class MixerModelSinusoidal(nn.Module):
    """Transformer 风格的 Sinusoidal Position Encoding"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        # Token Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        
        # Sinusoidal Position Encoding (固定，不可学习)
        self.position_encoding = SinusoidalPositionEncoding(
            config.d_model, config.max_seq_len, device=device, dtype=dtype
        )
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        # 使用原始 TESM（有 RoPE）
        mixer_cls = lambda layer_idx: (
            lambda dim: TESM(
                d_model=dim,
                layer_idx=layer_idx,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bit_eps=config.bit_eps,
                bit_threshold=config.bit_threshold,
                **config.ssm_cfg,
                **factory_kwargs,
            )
        )
        mlp_cls = lambda dim: FeedForward(config, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            BlockConv(
                config.d_model,
                mixer_cls(layer_idx),
                mlp_cls,
                norm_cls=norm_cls,
                residual_in_fp32=config.residual_in_fp32,
            )
            for layer_idx in range(config.n_layer)
        ])
        
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
        
        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(config.d_model, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=config.rescale_prenorm_residual,
                n_residuals_per_layer=1,
            )
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
                for i, layer in enumerate(self.layers)}
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")
        
        # 增量推理时用正确的位置偏移
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token Embedding + Sinusoidal Position Encoding
        hidden_states = self.embedding(input_ids)
        pos_encoding = self.position_encoding(positions)
        hidden_states = hidden_states + pos_encoding
        
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
            if self.gradient_checkpointing and self.training and inference_params is None:
                from torch.utils.checkpoint import checkpoint
                if residual is None:
                    hidden_states, residual = checkpoint(
                        lambda hs, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, None, inference_params=None, **_kw),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, residual = checkpoint(
                        lambda hs, res, _layer=layer, _kw=layer_mixer_kwargs: _layer(hs, res, inference_params=None, **_kw),
                        hidden_states,
                        residual,
                        use_reentrant=False,
                    )
            else:
                layer_inference_params = None
                if inference_params is not None and 'state_cache' in inference_params:
                    layer_cache = inference_params['state_cache'].get(i)
                    if layer_cache is not None:
                        layer_inference_params = {'state_cache': layer_cache}
                hidden_states, residual = layer(hidden_states, residual, inference_params=layer_inference_params, **layer_mixer_kwargs)
            
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats)


class TESMLMHeadModelSinusoidal(nn.Module):
    """Transformer 风格 Sinusoidal Position Encoding 的 LM 模型"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelSinusoidal(self.tesm_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(self.tesm_config.d_model, self.tesm_config.vocab_size, bias=False, device=device, dtype=dtype)
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def forward(self, input_ids, labels=None, inference_params=None, logits_to_keep=0, **kwargs):
        for k in ["attention_mask", "past_key_values", "use_cache"]:
            kwargs.pop(k, None)
        hidden_states, entanglement_maps, entanglement_stats = self.backbone(input_ids=input_ids, inference_params=inference_params, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tesm_config.label_ignore_index,
            )
        return TESMCausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states, 
                                   entanglement_maps=entanglement_maps, entanglement_stats=entanglement_stats)


@dataclass
class TESMCausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    entanglement_maps: Optional[List[torch.Tensor]] = None
    entanglement_stats: Optional[Dict[str, float]] = None


class TESMLMHeadModelConv(nn.Module):
    """使用卷积隐式位置编码的 LM 模型"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelConv(self.tesm_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(self.tesm_config.d_model, self.tesm_config.vocab_size, bias=False, device=device, dtype=dtype)
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def forward(self, input_ids, labels=None, inference_params=None, logits_to_keep=0, **kwargs):
        for k in ["attention_mask", "past_key_values", "use_cache"]:
            kwargs.pop(k, None)
        hidden_states, entanglement_maps, entanglement_stats = self.backbone(input_ids=input_ids, inference_params=inference_params, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tesm_config.label_ignore_index,
            )
        return TESMCausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states, 
                                   entanglement_maps=entanglement_maps, entanglement_stats=entanglement_stats)


def _merge_stats(stats_list: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not stats_list:
        return None
    merged = {}
    for key in stats_list[0].keys():
        merged[key] = sum(float(stats[key]) for stats in stats_list) / len(stats_list)
    return merged


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        nn.init.normal_(module.weight, std=initializer_range)
    elif isinstance(module, BitLinear):
        nn.init.normal_(module.weight, std=initializer_range)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, param in module.named_parameters(recurse=False):
            if name in ["out_proj.weight", "down_proj.weight"]:
                with torch.no_grad():
                    param /= (n_residuals_per_layer * n_layer) ** 0.5


# ==================== 实验函数 ====================

def test_position_sensitivity(model, model_name="Model"):
    """测试位置敏感性：相同 token 在不同位置的输出差异"""
    model.eval()
    with torch.no_grad():
        # 创建相同 token 的序列
        seq_len = 32
        token_id = 100
        input_ids = torch.full((1, seq_len), token_id, dtype=torch.long, device=next(model.parameters()).device)
        
        # 获取 hidden states
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states  # (1, L, D)
        
        # 计算相邻位置的余弦相似度
        similarities = []
        for i in range(seq_len - 1):
            h1 = hidden_states[0, i, :]
            h2 = hidden_states[0, i + 1, :]
            cos_sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            similarities.append(cos_sim)
        
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        print(f"\n{model_name} 位置敏感性测试:")
        print(f"  平均相邻位置相似度: {avg_sim:.4f}")
        print(f"  最小相似度: {min_sim:.4f}")
        print(f"  最大相似度: {max_sim:.4f}")
        print(f"  位置区分度: {1 - avg_sim:.4f} (越大越好)")
        
        return {"avg_sim": avg_sim, "min_sim": min_sim, "max_sim": max_sim, "position_distinction": 1 - avg_sim}


def test_length_generalization(model, model_name="Model", train_len=64, test_lens=[32, 64, 128, 256]):
    """测试长度泛化能力"""
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    
    with torch.no_grad():
        for test_len in test_lens:
            # 创建随机输入
            input_ids = torch.randint(0, 1000, (1, test_len), device=device)
            
            try:
                outputs = model(input_ids)
                loss = outputs.loss.item() if outputs.loss is not None else 0.0
                
                # 计算输出范数（检查数值稳定性）
                hidden_norm = outputs.hidden_states.norm().item()
                
                results[test_len] = {
                    "success": True,
                    "loss": loss,
                    "hidden_norm": hidden_norm,
                }
                print(f"{model_name} 长度 {test_len}: 成功, hidden_norm={hidden_norm:.4f}")
            except Exception as e:
                results[test_len] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"{model_name} 长度 {test_len}: 失败 - {e}")
    
    return results


def test_causality(model, model_name="Model"):
    """测试因果性：位置依赖是否正确"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 创建两个序列：前半相同，后半不同
        seq_len = 16
        common_part = torch.randint(0, 1000, (1, seq_len // 2), device=device)
        
        diff_part1 = torch.randint(0, 1000, (1, seq_len // 2), device=device)
        diff_part2 = torch.randint(0, 1000, (1, seq_len // 2), device=device)
        
        seq1 = torch.cat([common_part, diff_part1], dim=1)
        seq2 = torch.cat([common_part, diff_part2], dim=1)
        
        # 获取前半部分的 hidden states
        out1 = model(seq1)
        out2 = model(seq2)
        
        # 前半部分的输出应该相同（因果性）
        h1_first_half = out1.hidden_states[0, :seq_len // 2, :]
        h2_first_half = out2.hidden_states[0, :seq_len // 2, :]
        
        first_half_diff = (h1_first_half - h2_first_half).abs().mean().item()
        
        # 后半部分的输出应该不同
        h1_second_half = out1.hidden_states[0, seq_len // 2:, :]
        h2_second_half = out2.hidden_states[0, seq_len // 2:, :]
        
        second_half_diff = (h1_second_half - h2_second_half).abs().mean().item()
        
        print(f"\n{model_name} 因果性测试:")
        print(f"  前半部分差异: {first_half_diff:.6f} (应接近 0)")
        print(f"  后半部分差异: {second_half_diff:.6f} (应大于 0)")
        print(f"  因果性得分: {second_half_diff / (first_half_diff + 1e-8):.2f} (越大越好)")
        
        return {
            "first_half_diff": first_half_diff,
            "second_half_diff": second_half_diff,
            "causality_score": second_half_diff / (first_half_diff + 1e-8),
        }


def test_training_convergence(model, model_name="Model", num_steps=100, seq_len=32, lr=1e-3):
    """测试训练收敛速度"""
    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    
    for step in range(num_steps):
        # 随机生成输入和标签
        input_ids = torch.randint(0, 1000, (2, seq_len), device=device)
        labels = torch.randint(0, 1000, (2, seq_len), device=device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"{model_name} Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
    
    # 计算收敛指标
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss
    
    print(f"\n{model_name} 训练收敛测试:")
    print(f"  初始 Loss: {initial_loss:.4f}")
    print(f"  最终 Loss: {final_loss:.4f}")
    print(f"  Loss 下降比例: {loss_reduction:.2%}")
    
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "losses": losses,
    }


def run_experiment():
    """运行完整实验：五个方案对比"""
    print("=" * 90)
    print("位置编码对比实验：五个方案对比")
    print("=" * 90)
    print("\n方案说明:")
    print("  A. 显式位置编码: Position Embedding + RoPE (当前方案)")
    print("  B. 卷积隐式位置编码: Conv1d + RoPE")
    print("  C. 只保留 RoPE: 移除 Position Embedding")
    print("  D. 低维 Position Embedding: d_model//4 维度 + 投影")
    print("  E. Transformer Sinusoidal: 正弦位置编码 + RoPE")
    
    # 配置
    config = TESMConfig.small()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 导入原始模型
    from tesm_ssm import TESMLMHeadModel
    
    # 创建五个模型
    print("\n创建模型...")
    model_a = TESMLMHeadModel(config, device=device)              # A: 显式位置编码
    model_b = TESMLMHeadModelConv(config, device=device)          # B: 卷积隐式位置编码
    model_c = TESMLMHeadModelRoPE(config, device=device)          # C: 只保留 RoPE
    model_d = TESMLMHeadModelLowDim(config, device=device)        # D: 低维 Position Embedding
    model_e = TESMLMHeadModelSinusoidal(config, device=device)    # E: Transformer Sinusoidal
    
    # 参数量对比
    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    params_c = sum(p.numel() for p in model_c.parameters())
    params_d = sum(p.numel() for p in model_d.parameters())
    params_e = sum(p.numel() for p in model_e.parameters())
    
    # 计算 Position Embedding 参数量
    pos_embed_params_a = config.max_seq_len * config.d_model
    pos_embed_params_d = config.max_seq_len * (config.d_model // 4) + (config.d_model // 4) * config.d_model
    pos_embed_params_e = 0  # Sinusoidal 无参数
    
    print(f"\n参数量对比:")
    print(f"  A. 显式位置编码: {params_a:,}")
    print(f"  B. 卷积隐式位置编码: {params_b:,}")
    print(f"  C. 只保留 RoPE: {params_c:,}")
    print(f"  D. 低维 Position Embedding: {params_d:,}")
    print(f"  E. Transformer Sinusoidal: {params_e:,}")
    print(f"\n  参数差异 vs A:")
    print(f"  B: {params_b - params_a:,} (卷积层)")
    print(f"  C: {params_c - params_a:,} (移除 Position Embedding)")
    print(f"  D: {params_d - params_a:,} (低维 Position Embedding)")
    print(f"  E: {params_e - params_a:,} (Sinusoidal 无参数)")
    print(f"\n  Position Embedding 参数量:")
    print(f"  A: {pos_embed_params_a:,} (可学习)")
    print(f"  D: {pos_embed_params_d:,} (可学习，低维)")
    print(f"  E: {pos_embed_params_e:,} (固定，无参数)")
    
    # 测试 1: 位置敏感性
    print("\n" + "=" * 90)
    print("测试 1: 位置敏感性")
    print("=" * 90)
    sens_a = test_position_sensitivity(model_a, "A. 显式位置编码")
    sens_b = test_position_sensitivity(model_b, "B. 卷积隐式位置编码")
    sens_c = test_position_sensitivity(model_c, "C. 只保留 RoPE")
    sens_d = test_position_sensitivity(model_d, "D. 低维 Position Embedding")
    sens_e = test_position_sensitivity(model_e, "E. Transformer Sinusoidal")
    
    # 测试 2: 长度泛化
    print("\n" + "=" * 90)
    print("测试 2: 长度泛化")
    print("=" * 90)
    len_a = test_length_generalization(model_a, "A. 显式位置编码")
    len_b = test_length_generalization(model_b, "B. 卷积隐式位置编码")
    len_c = test_length_generalization(model_c, "C. 只保留 RoPE")
    len_d = test_length_generalization(model_d, "D. 低维 Position Embedding")
    len_e = test_length_generalization(model_e, "E. Transformer Sinusoidal")
    
    # 测试 3: 因果性
    print("\n" + "=" * 90)
    print("测试 3: 因果性")
    print("=" * 90)
    caus_a = test_causality(model_a, "A. 显式位置编码")
    caus_b = test_causality(model_b, "B. 卷积隐式位置编码")
    caus_c = test_causality(model_c, "C. 只保留 RoPE")
    caus_d = test_causality(model_d, "D. 低维 Position Embedding")
    caus_e = test_causality(model_e, "E. Transformer Sinusoidal")
    
    # 测试 4: 训练收敛
    print("\n" + "=" * 90)
    print("测试 4: 训练收敛")
    print("=" * 90)
    train_a = test_training_convergence(model_a, "A. 显式位置编码", num_steps=50)
    train_b = test_training_convergence(model_b, "B. 卷积隐式位置编码", num_steps=50)
    train_c = test_training_convergence(model_c, "C. 只保留 RoPE", num_steps=50)
    train_d = test_training_convergence(model_d, "D. 低维 Position Embedding", num_steps=50)
    train_e = test_training_convergence(model_e, "E. Transformer Sinusoidal", num_steps=50)
    
    # 总结
    print("\n" + "=" * 90)
    print("实验总结")
    print("=" * 90)
    
    print("\n| 指标 | A.显式位置编码 | B.卷积隐式 | C.只保留RoPE | D.低维Position | E.Sinusoidal |")
    print("|------|---------------|-----------|-------------|----------------|-------------|")
    print(f"| 参数量 | {params_a:,} | {params_b:,} | {params_c:,} | {params_d:,} | {params_e:,} |")
    print(f"| 位置区分度 | {sens_a['position_distinction']:.4f} | {sens_b['position_distinction']:.4f} | {sens_c['position_distinction']:.4f} | {sens_d['position_distinction']:.4f} | {sens_e['position_distinction']:.4f} |")
    print(f"| 因果性得分 | {caus_a['causality_score']:.2f} | {caus_b['causality_score']:.2f} | {caus_c['causality_score']:.2f} | {caus_d['causality_score']:.2f} | {caus_e['causality_score']:.2f} |")
    print(f"| 最终 Loss | {train_a['final_loss']:.4f} | {train_b['final_loss']:.4f} | {train_c['final_loss']:.4f} | {train_d['final_loss']:.4f} | {train_e['final_loss']:.4f} |")
    print(f"| Loss下降比例 | {train_a['loss_reduction']:.2%} | {train_b['loss_reduction']:.2%} | {train_c['loss_reduction']:.2%} | {train_d['loss_reduction']:.2%} | {train_e['loss_reduction']:.2%} |")
    
    # 长度泛化结果
    print("\n长度泛化结果:")
    print("| 长度 | A | B | C | D | E |")
    print("|------|---|---|---|---|---|")
    for length in [32, 64, 128, 256]:
        a_ok = len_a.get(length, {}).get('success', False)
        b_ok = len_b.get(length, {}).get('success', False)
        c_ok = len_c.get(length, {}).get('success', False)
        d_ok = len_d.get(length, {}).get('success', False)
        e_ok = len_e.get(length, {}).get('success', False)
        print(f"| {length} | {'✅' if a_ok else '❌'} | {'✅' if b_ok else '❌'} | {'✅' if c_ok else '❌'} | {'✅' if d_ok else '❌'} | {'✅' if e_ok else '❌'} |")
    
    # 分析结论
    print("\n" + "=" * 90)
    print("分析结论")
    print("=" * 90)
    
    print("\n1. 位置区分度排名:")
    rankings = sorted([
        ("A. 显式位置编码", sens_a['position_distinction']),
        ("B. 卷积隐式位置编码", sens_b['position_distinction']),
        ("C. 只保留 RoPE", sens_c['position_distinction']),
        ("D. 低维 Position Embedding", sens_d['position_distinction']),
        ("E. Transformer Sinusoidal", sens_e['position_distinction']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(rankings, 1):
        print(f"   {i}. {name}: {score:.4f}")
    
    print("\n2. 训练收敛排名 (Loss 下降比例):")
    train_rankings = sorted([
        ("A. 显式位置编码", train_a['loss_reduction']),
        ("B. 卷积隐式位置编码", train_b['loss_reduction']),
        ("C. 只保留 RoPE", train_c['loss_reduction']),
        ("D. 低维 Position Embedding", train_d['loss_reduction']),
        ("E. Transformer Sinusoidal", train_e['loss_reduction']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(train_rankings, 1):
        print(f"   {i}. {name}: {score:.2%}")
    
    print("\n3. 参数效率 (位置区分度/参数量):")
    eff_a = sens_a['position_distinction'] / params_a * 1e6
    eff_b = sens_b['position_distinction'] / params_b * 1e6
    eff_c = sens_c['position_distinction'] / params_c * 1e6
    eff_d = sens_d['position_distinction'] / params_d * 1e6
    eff_e = sens_e['position_distinction'] / params_e * 1e6 if params_e > 0 else float('inf')
    eff_rankings = sorted([
        ("A. 显式位置编码", eff_a),
        ("B. 卷积隐式位置编码", eff_b),
        ("C. 只保留 RoPE", eff_c),
        ("D. 低维 Position Embedding", eff_d),
        ("E. Transformer Sinusoidal", eff_e),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(eff_rankings, 1):
        if score == float('inf'):
            print(f"   {i}. {name}: ∞ (无参数)")
        else:
            print(f"   {i}. {name}: {score:.6f}")
    
    print("\n4. 方案 A vs 方案 E 对比:")
    print(f"   参数差异: {params_a - params_e:,} (A 有可学习位置编码，E 无参数)")
    print(f"   位置区分度: A={sens_a['position_distinction']:.4f}, E={sens_e['position_distinction']:.4f}")
    print(f"   位置区分度比例: E/A = {sens_e['position_distinction'] / sens_a['position_distinction']:.1%}")
    print(f"   训练收敛差异: A={train_a['loss_reduction']:.2%}, E={train_e['loss_reduction']:.2%}")
    
    return {
        "A_显式位置编码": {"params": params_a, "sensitivity": sens_a, "length_gen": len_a, "causality": caus_a, "training": train_a},
        "B_卷积隐式位置编码": {"params": params_b, "sensitivity": sens_b, "length_gen": len_b, "causality": caus_b, "training": train_b},
        "C_只保留RoPE": {"params": params_c, "sensitivity": sens_c, "length_gen": len_c, "causality": caus_c, "training": train_c},
        "D_低维PositionEmbedding": {"params": params_d, "sensitivity": sens_d, "length_gen": len_d, "causality": caus_d, "training": train_d},
        "E_TransformerSinusoidal": {"params": params_e, "sensitivity": sens_e, "length_gen": len_e, "causality": caus_e, "training": train_e},
    }


if __name__ == "__main__":
    results = run_experiment()
