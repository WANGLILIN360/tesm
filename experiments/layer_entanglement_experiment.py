"""
层纠缠对比实验：比较三种层纠缠机制
1. TESM 层纠缠：跨层状态偏置纠缠 Q
2. Mamba3 风格：跨层状态融合
3. Transformer 风格：跨层 Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict
from functools import partial

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

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


# ==================== 自定义 FeedForward ====================

class FeedForward(nn.Module):
    """简单的 FeedForward 网络"""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_dim = int(config.d_model * 4)
        self.fc1 = BitLinear(config.d_model, hidden_dim, bias=False, 
                             bit_eps=config.bit_eps, bit_threshold=config.bit_threshold,
                             **factory_kwargs)
        self.fc2 = BitLinear(hidden_dim, config.d_model, bias=False,
                             bit_eps=config.bit_eps, bit_threshold=config.bit_threshold,
                             **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


# ==================== 自定义 Block ====================

class Block(nn.Module):
    """带有 Pre-Norm 的 Block"""
    def __init__(self, dim, mixer, mlp_cls, norm_cls, residual_in_fp32=False):
        super().__init__()
        self.mixer = mixer(dim) if callable(mixer) else mixer
        self.norm = norm_cls(dim)
        self.mlp = mlp_cls(dim) if callable(mlp_cls) else mlp_cls
        self.norm2 = norm_cls(dim) if self.mlp is not None else None
        self.residual_in_fp32 = residual_in_fp32
        self.layer_idx = getattr(self.mixer, "layer_idx", None)
    
    def forward(self, hidden_states, residual=None, inference_params=None, **mixer_kwargs):
        residual = hidden_states if residual is None else residual
        hidden_states = self.norm(hidden_states.to(self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states, residual = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs), residual + hidden_states
        
        if self.mlp is not None:
            hidden_states = self.norm2(hidden_states.to(self.norm2.weight.dtype))
            hidden_states, residual = self.mlp(hidden_states), residual + hidden_states
        
        return hidden_states, residual


def _init_weights(module, n_layer, initializer_range, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if not getattr(module, "_no_weight_init", False):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)


def _merge_stats(stats_list):
    if not stats_list:
        return {"positive": 0.0, "negative": 0.0, "zero": 1.0}
    total_pos = sum(s.get("positive", 0.0) for s in stats_list)
    total_neg = sum(s.get("negative", 0.0) for s in stats_list)
    total_zero = sum(s.get("zero", 0.0) for s in stats_list)
    n = len(stats_list)
    return {"positive": total_pos / n, "negative": total_neg / n, "zero": total_zero / n}


@dataclass
class TESMCausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    entanglement_maps: Optional[List[torch.Tensor]] = None
    entanglement_stats: Optional[Dict[str, float]] = None


# ==================== 方案 A：TESM 原始层纠缠 ====================
# 跨层状态偏置纠缠 Q：上层状态摘要偏置当前层的纠缠 Q

class TESMWithLayerEntanglement(TESM):
    """TESM 原始层纠缠：跨层状态偏置纠缠 Q"""
    
    def forward(self, u, inference_params=None, cross_layer_state=None):
        """forward 已在 TESM 中实现跨层纠缠"""
        return super().forward(u, inference_params, cross_layer_state)


class MixerModelTESM(nn.Module):
    """TESM 原始层纠缠方案"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        # Token Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        
        # Position Embedding
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        # 使用带层纠缠的 TESM
        mixer_cls = lambda layer_idx: (
            lambda dim: TESMWithLayerEntanglement(
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
            Block(
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
        
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token + Position Embedding
        hidden_states = self.embedding(input_ids) + self.position_embedding(positions)
        
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
            
            # 获取跨层状态
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats)


class TESMLMHeadModelA(nn.Module):
    """方案 A：TESM 原始层纠缠"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelTESM(self.tesm_config, device=device, dtype=dtype)
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


# ==================== 方案 B：Mamba3 风格层纠缠 ====================
# 跨层状态融合：上层状态直接融合到当前层状态

class TESMMamba3Style(nn.Module):
    """Mamba3 风格层纠缠：跨层状态直接融合"""
    
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
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.ent_rank = ent_rank
        self.entanglement_scale = entanglement_scale
        self.entanglement_threshold = entanglement_threshold
        self.entanglement_init = kwargs.get('entanglement_init', 0.3)
        self.entanglement_window = int(kwargs.get("entanglement_window", 0) or 0)
        self.entanglement_block_size = int(kwargs.get("entanglement_block_size", 256) or 256)
        self.state_scan_chunk_size = int(kwargs.get("state_scan_chunk_size", 16) or 16)
        self.use_triton_kernels = bool(kwargs.get("use_triton_kernels", True))
        self.kernel_backend = str(kwargs.get("kernel_backend", "auto"))
        self.kernel_mode = "precise" if str(kwargs.get("kernel_mode", "fast")) == "precise" else "fast"
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        
        total_proj = (2 * d_model) + (3 * d_state) + (2 * ent_rank)
        self.in_proj = BitLinear(d_model, total_proj, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, 
                                  kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.state_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                     kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.ent_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                   kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.out_proj = BitLinear(d_model, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                   kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
        decay_init = float(kwargs.get('decay_init_bias', 3.0))
        self.decay_bias = nn.Parameter(torch.full((d_state,), decay_init, device=device, dtype=torch.float32))
        
        if self.entanglement_window > 0:
            self.register_parameter("entanglement", None)
            self.local_entanglement_bias = nn.Parameter(
                torch.randn(self.entanglement_window, device=device, dtype=torch.float32) * self.entanglement_init
            )
        else:
            self.register_parameter("local_entanglement_bias", None)
            self.entanglement = nn.Parameter(
                torch.randn(max_seq_len, max_seq_len, device=device, dtype=torch.float32) * self.entanglement_init
            )
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)), persistent=False)
        self.rope_base = float(kwargs.get('rope_base', 10000.0))
        
        # Mamba3 风格：跨层状态融合投影
        self.cross_layer_state_proj = BitLinear(d_state, d_state, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                                 kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.cross_layer_gate = nn.Parameter(torch.ones(d_state, device=device, dtype=torch.float32) * 0.5)
        
        self.last_entanglement_map = None
        self.last_entanglement_stats = None
        self._last_cross_layer_state = None
        self._last_scan_state_f64 = None
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        dev = self.out_proj.weight.device
        _dtype = dtype or torch.float32
        window = max(self.entanglement_window, 1)
        return {
            'state': torch.zeros(batch_size, self.d_state, device=dev, dtype=torch.float64),
            'seq_pos': 0,
            'ent_k_cache': torch.zeros(batch_size, window, self.ent_rank, device=dev, dtype=_dtype),
            'ent_v_cache': torch.zeros(batch_size, window, self.d_state, device=dev, dtype=_dtype),
        }
    
    def ternary_entanglement(self, scores):
        hard = torch.where(
            scores > self.entanglement_threshold,
            torch.ones_like(scores),
            torch.where(scores < -self.entanglement_threshold, -torch.ones_like(scores), torch.zeros_like(scores)),
        )
        return scores + (hard - scores).detach()
    
    def _update_entanglement_stats(self, ternary):
        if self.training:
            self.last_entanglement_map = None
            ternary_detached = ternary.detach()
            total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
            self.last_entanglement_stats = {
                "positive": float((ternary_detached > 0).sum().item()) / total,
                "negative": float((ternary_detached < 0).sum().item()) / total,
                "zero": float((ternary_detached == 0).sum().item()) / total,
            }
            return
        ternary_detached = ternary.detach()
        total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
        self.last_entanglement_map = ternary_detached
        self.last_entanglement_stats = {
            "positive": float((ternary_detached > 0).sum().item()) / total,
            "negative": float((ternary_detached < 0).sum().item()) / total,
            "zero": float((ternary_detached == 0).sum().item()) / total,
        }
    
    def _apply_rope(self, x, pos_offset=0):
        B, L, D = x.shape
        half = D // 2
        pos = torch.arange(pos_offset, pos_offset + L, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(half, device=x.device, dtype=torch.float32)
        theta = pos.unsqueeze(1) * (1.0 / (self.rope_base ** (2.0 * dim_idx / D)))
        cos_t = theta.cos().unsqueeze(0).to(x.dtype)
        sin_t = theta.sin().unsqueeze(0).to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)
    
    def _parallel_state_scan(self, decay, update):
        batch, seqlen, _ = decay.shape
        orig_dtype = decay.dtype
        chunk_size = min(max(self.state_scan_chunk_size, 1), seqlen)
        states = torch.empty_like(update)
        prev_state = torch.zeros(batch, self.d_state, device=update.device, dtype=torch.float64)
        for start in range(0, seqlen, chunk_size):
            end = min(start + chunk_size, seqlen)
            decay_chunk = decay[:, start:end, :].to(torch.float64).clamp_min(1e-12)
            update_chunk = update[:, start:end, :].to(torch.float64)
            prefix = torch.cumprod(decay_chunk, dim=1)
            normalized_update = update_chunk / prefix.clamp_min(1e-12)
            local_states = prefix * torch.cumsum(normalized_update, dim=1)
            local_states = local_states + prefix * prev_state.unsqueeze(1)
            states[:, start:end, :] = local_states.to(dtype=update.dtype)
            prev_state = local_states[:, -1, :]
        self._last_scan_state_f64 = prev_state.detach()
        return states
    
    def _compute_entanglement(self, q, k, values):
        seq_len = q.size(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.ent_rank)
        
        # 处理 entanglement_window > 0 的情况
        if self.entanglement_window > 0 and self.entanglement is None:
            # 使用滑动窗口，简化处理：不添加位置偏置
            pass
        elif self.entanglement is not None:
            scores = scores + self.entanglement[:seq_len, :seq_len].to(device=scores.device, dtype=scores.dtype)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), 0.0)
        ternary = self.ternary_entanglement(scores)
        ternary = ternary * causal_mask.unsqueeze(0).to(ternary.dtype)
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        entangled = torch.matmul(ternary / norm, values)
        self._update_entanglement_stats(ternary)
        return entangled
    
    def forward(self, u, inference_params=None, cross_layer_state=None):
        batch, seqlen, _ = u.shape
        
        if inference_params is not None and seqlen == 1:
            layer_cache = inference_params.get('state_cache')
            if layer_cache is not None and 'state' in layer_cache:
                # 增量推理（简化版）
                return self._forward_incremental(u, inference_params, cross_layer_state)
        
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
        
        # RoPE
        ent_q = self._apply_rope(ent_q)
        ent_k = self._apply_rope(ent_k)
        
        # 状态积累
        update = write * state_value
        states = self._parallel_state_scan(decay, update)
        
        # ===== Mamba3 风格：跨层状态直接融合 =====
        if cross_layer_state is not None:
            # 投影上层状态
            cross_state_proj = self.cross_layer_state_proj(cross_layer_state)
            if cross_state_proj.dim() == 2:
                cross_state_proj = cross_state_proj.unsqueeze(1)
            # 门控融合
            gate = torch.sigmoid(self.cross_layer_gate)
            states = states + gate * cross_state_proj
        
        # 三值纠缠
        signed_avg = self._compute_entanglement(ent_q, ent_k, states)
        entangled_states = states + self.entanglement_scale * (signed_avg - states)
        ent_change = entangled_states - states
        
        # 输出
        state_projected = self.state_proj(torch.tanh(entangled_states))
        ent_projected = self.ent_proj(ent_change)
        state_mixed = out_gate * state_projected
        ent_mixed = ent_projected
        y = local + state_mixed + ent_mixed
        y = self.out_proj(self.dropout(y))
        
        # 传递状态给下一层
        self._last_cross_layer_state = states.detach()
        return y
    
    def _forward_incremental(self, u, inference_params, cross_layer_state):
        """简化的增量推理"""
        cache = inference_params.get('state_cache')
        if cache is None:
            return self.forward(u, inference_params=None, cross_layer_state=cross_layer_state)
        
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
        
        # 增量状态更新
        prev_state = cache['state'].to(torch.float64)
        new_state = decay.squeeze(1).to(torch.float64) * prev_state + write.squeeze(1).to(torch.float64) * state_value.squeeze(1).to(torch.float64)
        cache['state'] = new_state.clone()
        cache['seq_pos'] = cache.get('seq_pos', 0) + 1
        
        states = new_state.unsqueeze(1).to(u.dtype)
        
        # Mamba3 跨层融合
        if cross_layer_state is not None:
            cross_state_proj = self.cross_layer_state_proj(cross_layer_state)
            gate = torch.sigmoid(self.cross_layer_gate)
            states = states + gate * cross_state_proj.unsqueeze(1)
        
        state_projected = self.state_proj(torch.tanh(states))
        y = local + out_gate * state_projected
        y = self.out_proj(self.dropout(y))
        
        self._last_cross_layer_state = states.detach()
        return y


class MixerModelMamba3(nn.Module):
    """Mamba3 风格层纠缠方案"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        mixer_cls = lambda layer_idx: (
            lambda dim: TESMMamba3Style(
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
            Block(
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
        
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embedding(input_ids) + self.position_embedding(positions)
        
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
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


class TESMLMHeadModelB(nn.Module):
    """方案 B：Mamba3 风格层纠缠"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelMamba3(self.tesm_config, device=device, dtype=dtype)
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


# ==================== 方案 C：Transformer 风格层纠缠 ====================
# 跨层 Attention：类似 Transformer 的 Cross-Layer Attention

class TESMTransformerStyle(nn.Module):
    """Transformer 风格层纠缠：跨层 Attention"""
    
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
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.ent_rank = ent_rank
        self.entanglement_scale = entanglement_scale
        self.entanglement_threshold = entanglement_threshold
        self.entanglement_init = kwargs.get('entanglement_init', 0.3)
        self.entanglement_window = int(kwargs.get("entanglement_window", 0) or 0)
        self.entanglement_block_size = int(kwargs.get("entanglement_block_size", 256) or 256)
        self.state_scan_chunk_size = int(kwargs.get("state_scan_chunk_size", 16) or 16)
        self.use_triton_kernels = bool(kwargs.get("use_triton_kernels", True))
        self.kernel_backend = str(kwargs.get("kernel_backend", "auto"))
        self.kernel_mode = "precise" if str(kwargs.get("kernel_mode", "fast")) == "precise" else "fast"
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        
        total_proj = (2 * d_model) + (3 * d_state) + (2 * ent_rank)
        self.in_proj = BitLinear(d_model, total_proj, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                  kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.state_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                     kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.ent_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                   kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.out_proj = BitLinear(d_model, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                   kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
        decay_init = float(kwargs.get('decay_init_bias', 3.0))
        self.decay_bias = nn.Parameter(torch.full((d_state,), decay_init, device=device, dtype=torch.float32))
        
        if self.entanglement_window > 0:
            self.register_parameter("entanglement", None)
            self.local_entanglement_bias = nn.Parameter(
                torch.randn(self.entanglement_window, device=device, dtype=torch.float32) * self.entanglement_init
            )
        else:
            self.register_parameter("local_entanglement_bias", None)
            self.entanglement = nn.Parameter(
                torch.randn(max_seq_len, max_seq_len, device=device, dtype=torch.float32) * self.entanglement_init
            )
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)), persistent=False)
        self.rope_base = float(kwargs.get('rope_base', 10000.0))
        
        # Transformer 风格：跨层 Attention
        self.cross_layer_q = BitLinear(d_state, ent_rank, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                        kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.cross_layer_k = BitLinear(d_state, ent_rank, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                        kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.cross_layer_v = BitLinear(d_state, d_state, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                        kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.cross_layer_out = BitLinear(d_state, d_state, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold,
                                          kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        
        self.last_entanglement_map = None
        self.last_entanglement_stats = None
        self._last_cross_layer_state = None
        self._last_scan_state_f64 = None
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        dev = self.out_proj.weight.device
        _dtype = dtype or torch.float32
        window = max(self.entanglement_window, 1)
        return {
            'state': torch.zeros(batch_size, self.d_state, device=dev, dtype=torch.float64),
            'seq_pos': 0,
            'ent_k_cache': torch.zeros(batch_size, window, self.ent_rank, device=dev, dtype=_dtype),
            'ent_v_cache': torch.zeros(batch_size, window, self.d_state, device=dev, dtype=_dtype),
        }
    
    def ternary_entanglement(self, scores):
        hard = torch.where(
            scores > self.entanglement_threshold,
            torch.ones_like(scores),
            torch.where(scores < -self.entanglement_threshold, -torch.ones_like(scores), torch.zeros_like(scores)),
        )
        return scores + (hard - scores).detach()
    
    def _update_entanglement_stats(self, ternary):
        if self.training:
            self.last_entanglement_map = None
            ternary_detached = ternary.detach()
            total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
            self.last_entanglement_stats = {
                "positive": float((ternary_detached > 0).sum().item()) / total,
                "negative": float((ternary_detached < 0).sum().item()) / total,
                "zero": float((ternary_detached == 0).sum().item()) / total,
            }
            return
        ternary_detached = ternary.detach()
        total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
        self.last_entanglement_map = ternary_detached
        self.last_entanglement_stats = {
            "positive": float((ternary_detached > 0).sum().item()) / total,
            "negative": float((ternary_detached < 0).sum().item()) / total,
            "zero": float((ternary_detached == 0).sum().item()) / total,
        }
    
    def _apply_rope(self, x, pos_offset=0):
        B, L, D = x.shape
        half = D // 2
        pos = torch.arange(pos_offset, pos_offset + L, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(half, device=x.device, dtype=torch.float32)
        theta = pos.unsqueeze(1) * (1.0 / (self.rope_base ** (2.0 * dim_idx / D)))
        cos_t = theta.cos().unsqueeze(0).to(x.dtype)
        sin_t = theta.sin().unsqueeze(0).to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)
    
    def _parallel_state_scan(self, decay, update):
        batch, seqlen, _ = decay.shape
        orig_dtype = decay.dtype
        chunk_size = min(max(self.state_scan_chunk_size, 1), seqlen)
        states = torch.empty_like(update)
        prev_state = torch.zeros(batch, self.d_state, device=update.device, dtype=torch.float64)
        for start in range(0, seqlen, chunk_size):
            end = min(start + chunk_size, seqlen)
            decay_chunk = decay[:, start:end, :].to(torch.float64).clamp_min(1e-12)
            update_chunk = update[:, start:end, :].to(torch.float64)
            prefix = torch.cumprod(decay_chunk, dim=1)
            normalized_update = update_chunk / prefix.clamp_min(1e-12)
            local_states = prefix * torch.cumsum(normalized_update, dim=1)
            local_states = local_states + prefix * prev_state.unsqueeze(1)
            states[:, start:end, :] = local_states.to(dtype=update.dtype)
            prev_state = local_states[:, -1, :]
        self._last_scan_state_f64 = prev_state.detach()
        return states
    
    def _compute_entanglement(self, q, k, values):
        seq_len = q.size(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.ent_rank)
        
        # 处理 entanglement_window > 0 的情况
        if self.entanglement_window > 0 and self.entanglement is None:
            # 使用滑动窗口，简化处理：不添加位置偏置
            pass
        elif self.entanglement is not None:
            scores = scores + self.entanglement[:seq_len, :seq_len].to(device=scores.device, dtype=scores.dtype)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), 0.0)
        ternary = self.ternary_entanglement(scores)
        ternary = ternary * causal_mask.unsqueeze(0).to(ternary.dtype)
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        entangled = torch.matmul(ternary / norm, values)
        self._update_entanglement_stats(ternary)
        return entangled
    
    def forward(self, u, inference_params=None, cross_layer_state=None):
        batch, seqlen, _ = u.shape
        
        if inference_params is not None and seqlen == 1:
            layer_cache = inference_params.get('state_cache')
            if layer_cache is not None and 'state' in layer_cache:
                return self._forward_incremental(u, inference_params, cross_layer_state)
        
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
        
        # RoPE
        ent_q = self._apply_rope(ent_q)
        ent_k = self._apply_rope(ent_k)
        
        # 状态积累
        update = write * state_value
        states = self._parallel_state_scan(decay, update)
        
        # ===== Transformer 风格：跨层 Attention =====
        if cross_layer_state is not None:
            # 当前层状态作为 Query
            cross_q = self.cross_layer_q(states)  # (B, L, ent_rank)
            # 上层状态作为 Key 和 Value
            cross_k = self.cross_layer_k(cross_layer_state)  # (B, L, ent_rank)
            cross_v = self.cross_layer_v(cross_layer_state)  # (B, L, d_state)
            
            # Attention 计算
            attn_scores = torch.matmul(cross_q, cross_k.transpose(-2, -1)) / math.sqrt(self.ent_rank)
            seq_len_q = cross_q.size(1)
            seq_len_k = cross_k.size(1)
            # 因果 mask：当前层只能看到上层相同位置及之前
            causal_mask = self.causal_mask[:seq_len_q, :seq_len_k]
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # 加权求和
            cross_attn_out = torch.matmul(attn_weights, cross_v)  # (B, L, d_state)
            cross_attn_out = self.cross_layer_out(cross_attn_out)
            
            # 残差连接
            states = states + cross_attn_out
        
        # 三值纠缠
        signed_avg = self._compute_entanglement(ent_q, ent_k, states)
        entangled_states = states + self.entanglement_scale * (signed_avg - states)
        ent_change = entangled_states - states
        
        # 输出
        state_projected = self.state_proj(torch.tanh(entangled_states))
        ent_projected = self.ent_proj(ent_change)
        state_mixed = out_gate * state_projected
        ent_mixed = ent_projected
        y = local + state_mixed + ent_mixed
        y = self.out_proj(self.dropout(y))
        
        self._last_cross_layer_state = states.detach()
        return y
    
    def _forward_incremental(self, u, inference_params, cross_layer_state):
        """简化的增量推理"""
        cache = inference_params.get('state_cache')
        if cache is None:
            return self.forward(u, inference_params=None, cross_layer_state=cross_layer_state)
        
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
        
        prev_state = cache['state'].to(torch.float64)
        new_state = decay.squeeze(1).to(torch.float64) * prev_state + write.squeeze(1).to(torch.float64) * state_value.squeeze(1).to(torch.float64)
        cache['state'] = new_state.clone()
        cache['seq_pos'] = cache.get('seq_pos', 0) + 1
        
        states = new_state.unsqueeze(1).to(u.dtype)
        
        # Transformer 风格跨层 Attention（简化）
        if cross_layer_state is not None:
            cross_q = self.cross_layer_q(states)
            cross_k = self.cross_layer_k(cross_layer_state)
            cross_v = self.cross_layer_v(cross_layer_state)
            attn_scores = torch.matmul(cross_q, cross_k.transpose(-2, -1)) / math.sqrt(self.ent_rank)
            attn_weights = F.softmax(attn_scores, dim=-1)
            cross_attn_out = torch.matmul(attn_weights, cross_v)
            cross_attn_out = self.cross_layer_out(cross_attn_out)
            states = states + cross_attn_out
        
        state_projected = self.state_proj(torch.tanh(states))
        y = local + out_gate * state_projected
        y = self.out_proj(self.dropout(y))
        
        self._last_cross_layer_state = states.detach()
        return y


class MixerModelTransformer(nn.Module):
    """Transformer 风格层纠缠方案"""
    
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model, **factory_kwargs)
        
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon, device=device, dtype=dtype)
        
        mixer_cls = lambda layer_idx: (
            lambda dim: TESMTransformerStyle(
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
            Block(
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
        
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embedding(input_ids) + self.position_embedding(positions)
        
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        
        for i, layer in enumerate(self.layers):
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            
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


class TESMLMHeadModelC(nn.Module):
    """方案 C：Transformer 风格层纠缠"""
    
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModelTransformer(self.tesm_config, device=device, dtype=dtype)
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


# ==================== 测试函数 ====================

def test_layer_interaction(model, model_name="Model"):
    """测试层间交互强度"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 创建输入
        seq_len = 32
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        
        # 前向传播，收集每层的 hidden states
        hidden_states_list = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states_list.append(output[0].detach())
        
        hooks = []
        for layer in model.backbone.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        _ = model(input_ids)
        
        for h in hooks:
            h.remove()
        
        # 计算相邻层之间的变化
        layer_changes = []
        for i in range(1, len(hidden_states_list)):
            change = (hidden_states_list[i] - hidden_states_list[i-1]).abs().mean().item()
            layer_changes.append(change)
        
        avg_change = sum(layer_changes) / len(layer_changes) if layer_changes else 0.0
        
        print(f"\n{model_name} 层间交互测试:")
        print(f"  层数: {len(hidden_states_list)}")
        print(f"  平均层间变化: {avg_change:.6f}")
        for i, change in enumerate(layer_changes):
            print(f"  Layer {i} -> {i+1}: {change:.6f}")
        
        return {
            "num_layers": len(hidden_states_list),
            "avg_layer_change": avg_change,
            "layer_changes": layer_changes,
        }


def test_cross_layer_dependency(model, model_name="Model"):
    """测试跨层依赖性"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        seq_len = 16
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        
        # 正常前向传播
        output_normal = model(input_ids)
        hidden_normal = output_normal.hidden_states[0, -1, :]
        
        # 测试：修改第一层的输入，观察最后一层的变化
        # 这需要更细粒度的控制，我们用简化方法：
        # 比较相同输入两次前向传播的稳定性
        
        output1 = model(input_ids)
        output2 = model(input_ids)
        
        hidden1 = output1.hidden_states[0, -1, :]
        hidden2 = output2.hidden_states[0, -1, :]
        
        consistency = (hidden1 - hidden2).abs().mean().item()
        
        # 测试位置敏感的跨层依赖
        input_ids_shifted = torch.cat([input_ids[:, 8:], input_ids[:, :8]], dim=1)
        output_shifted = model(input_ids_shifted)
        hidden_shifted = output_shifted.hidden_states[0, -1, :]
        
        shift_diff = (hidden_normal - hidden_shifted).abs().mean().item()
        
        print(f"\n{model_name} 跨层依赖测试:")
        print(f"  输出一致性: {consistency:.8f} (应接近 0)")
        print(f"  位置变化响应: {shift_diff:.6f} (应大于 0)")
        
        return {
            "output_consistency": consistency,
            "position_response": shift_diff,
        }


def test_training_convergence(model, model_name="Model", num_steps=50, seq_len=32, lr=1e-3):
    """测试训练收敛速度"""
    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    
    for step in range(num_steps):
        input_ids = torch.randint(0, 1000, (2, seq_len), device=device)
        labels = torch.randint(0, 1000, (2, seq_len), device=device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"{model_name} Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    
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


def test_memory_efficiency(model, model_name="Model"):
    """测试内存效率"""
    model.eval()
    device = next(model.parameters()).device
    
    if device.type != 'cuda':
        print(f"\n{model_name} 内存效率测试: 需要 CUDA 设备")
        return {"peak_memory_mb": 0}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        seq_len = 128
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        _ = model(input_ids)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"\n{model_name} 内存效率测试:")
    print(f"  峰值内存: {peak_memory:.2f} MB")
    
    return {"peak_memory_mb": peak_memory}


def run_experiment():
    """运行完整实验：三个方案对比"""
    print("=" * 90)
    print("层纠缠对比实验：三个方案对比")
    print("=" * 90)
    print("\n方案说明:")
    print("  A. TESM 原始层纠缠: 跨层状态偏置纠缠 Q (上层状态摘要偏置当前层纠缠 Q)")
    print("  B. Mamba3 风格层纠缠: 跨层状态直接融合 (门控融合上层状态)")
    print("  C. Transformer 风格层纠缠: 跨层 Attention (当前层 Query 对上层 Key/Value)")
    
    # 配置
    config = TESMConfig.small()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 创建三个模型
    print("\n创建模型...")
    model_a = TESMLMHeadModelA(config, device=device)  # A: TESM 原始
    model_b = TESMLMHeadModelB(config, device=device)  # B: Mamba3 风格
    model_c = TESMLMHeadModelC(config, device=device)  # C: Transformer 风格
    
    # 参数量对比
    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    params_c = sum(p.numel() for p in model_c.parameters())
    
    print(f"\n参数量对比:")
    print(f"  A. TESM 原始层纠缠: {params_a:,}")
    print(f"  B. Mamba3 风格层纠缠: {params_b:,}")
    print(f"  C. Transformer 风格层纠缠: {params_c:,}")
    print(f"\n  参数差异 vs A:")
    print(f"  B: {params_b - params_a:,}")
    print(f"  C: {params_c - params_a:,}")
    
    # 测试 1: 层间交互
    print("\n" + "=" * 90)
    print("测试 1: 层间交互强度")
    print("=" * 90)
    layer_a = test_layer_interaction(model_a, "A. TESM 原始层纠缠")
    layer_b = test_layer_interaction(model_b, "B. Mamba3 风格层纠缠")
    layer_c = test_layer_interaction(model_c, "C. Transformer 风格层纠缠")
    
    # 测试 2: 跨层依赖
    print("\n" + "=" * 90)
    print("测试 2: 跨层依赖性")
    print("=" * 90)
    cross_a = test_cross_layer_dependency(model_a, "A. TESM 原始层纠缠")
    cross_b = test_cross_layer_dependency(model_b, "B. Mamba3 风格层纠缠")
    cross_c = test_cross_layer_dependency(model_c, "C. Transformer 风格层纠缠")
    
    # 测试 3: 训练收敛
    print("\n" + "=" * 90)
    print("测试 3: 训练收敛")
    print("=" * 90)
    train_a = test_training_convergence(model_a, "A. TESM 原始层纠缠", num_steps=50)
    train_b = test_training_convergence(model_b, "B. Mamba3 风格层纠缠", num_steps=50)
    train_c = test_training_convergence(model_c, "C. Transformer 风格层纠缠", num_steps=50)
    
    # 测试 4: 内存效率
    print("\n" + "=" * 90)
    print("测试 4: 内存效率")
    print("=" * 90)
    mem_a = test_memory_efficiency(model_a, "A. TESM 原始层纠缠")
    mem_b = test_memory_efficiency(model_b, "B. Mamba3 风格层纠缠")
    mem_c = test_memory_efficiency(model_c, "C. Transformer 风格层纠缠")
    
    # 总结
    print("\n" + "=" * 90)
    print("实验总结")
    print("=" * 90)
    
    print("\n| 指标 | A.TESM原始 | B.Mamba3风格 | C.Transformer风格 |")
    print("|------|-----------|-------------|-------------------|")
    print(f"| 参数量 | {params_a:,} | {params_b:,} | {params_c:,} |")
    print(f"| 平均层间变化 | {layer_a['avg_layer_change']:.6f} | {layer_b['avg_layer_change']:.6f} | {layer_c['avg_layer_change']:.6f} |")
    print(f"| 位置响应 | {cross_a['position_response']:.6f} | {cross_b['position_response']:.6f} | {cross_c['position_response']:.6f} |")
    print(f"| 最终 Loss | {train_a['final_loss']:.4f} | {train_b['final_loss']:.4f} | {train_c['final_loss']:.4f} |")
    print(f"| Loss下降比例 | {train_a['loss_reduction']:.2%} | {train_b['loss_reduction']:.2%} | {train_c['loss_reduction']:.2%} |")
    print(f"| 峰值内存(MB) | {mem_a['peak_memory_mb']:.2f} | {mem_b['peak_memory_mb']:.2f} | {mem_c['peak_memory_mb']:.2f} |")
    
    # 分析结论
    print("\n" + "=" * 90)
    print("分析结论")
    print("=" * 90)
    
    print("\n1. 层间交互强度排名:")
    layer_rankings = sorted([
        ("A. TESM 原始层纠缠", layer_a['avg_layer_change']),
        ("B. Mamba3 风格层纠缠", layer_b['avg_layer_change']),
        ("C. Transformer 风格层纠缠", layer_c['avg_layer_change']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(layer_rankings, 1):
        print(f"   {i}. {name}: {score:.6f}")
    
    print("\n2. 训练收敛排名 (Loss 下降比例):")
    train_rankings = sorted([
        ("A. TESM 原始层纠缠", train_a['loss_reduction']),
        ("B. Mamba3 风格层纠缠", train_b['loss_reduction']),
        ("C. Transformer 风格层纠缠", train_c['loss_reduction']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(train_rankings, 1):
        print(f"   {i}. {name}: {score:.2%}")
    
    print("\n3. 内存效率排名 (峰值内存越低越好):")
    mem_rankings = sorted([
        ("A. TESM 原始层纠缠", mem_a['peak_memory_mb']),
        ("B. Mamba3 风格层纠缠", mem_b['peak_memory_mb']),
        ("C. Transformer 风格层纠缠", mem_c['peak_memory_mb']),
    ], key=lambda x: x[1])
    for i, (name, score) in enumerate(mem_rankings, 1):
        print(f"   {i}. {name}: {score:.2f} MB")
    
    return {
        "A_TESM原始层纠缠": {"params": params_a, "layer": layer_a, "cross": cross_a, "training": train_a, "memory": mem_a},
        "B_Mamba3风格层纠缠": {"params": params_b, "layer": layer_b, "cross": cross_b, "training": train_b, "memory": mem_b},
        "C_Transformer风格层纠缠": {"params": params_c, "layer": layer_c, "cross": cross_c, "training": train_c, "memory": mem_c},
    }


if __name__ == "__main__":
    results = run_experiment()
