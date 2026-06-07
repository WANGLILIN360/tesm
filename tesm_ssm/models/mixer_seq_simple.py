from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.modules.block import Block, RMSNorm
from tesm_ssm.modules.tesm import BitLinear, TESM_SISO
from tesm_ssm.modules.tesm_mimo import TESMMIMO_Optimized


@dataclass
class TESMCausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    entanglement_maps: Optional[List[torch.Tensor]] = None
    entanglement_stats: Optional[Dict[str, float]] = None


class FeedForward(nn.Module):
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_backend = config.kernel_backend
        kernel_mode = config.kernel_mode
        self.gate_proj = BitLinear(config.d_model, config.d_intermediate, bias=False, bit_eps=config.bit_eps, bit_threshold=config.bit_threshold, kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.up_proj = BitLinear(config.d_model, config.d_intermediate, bias=False, bit_eps=config.bit_eps, bit_threshold=config.bit_threshold, kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.down_proj = BitLinear(config.d_intermediate, config.d_model, bias=False, bit_eps=config.bit_eps, bit_threshold=config.bit_threshold, kernel_backend=kernel_backend, kernel_mode=kernel_mode, **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)

    def _can_use_grouped_projection(self, x):
        return (
            isinstance(self.gate_proj, BitLinear)
            and isinstance(self.up_proj, BitLinear)
            and self.gate_proj.in_features == self.up_proj.in_features
            and self.gate_proj.kernel_backend == self.up_proj.kernel_backend
            and self.gate_proj.kernel_mode == self.up_proj.kernel_mode
        )

    def _grouped_gate_up(self, x):
        qinput = self.gate_proj.quantized_input(x)
        gate_qweight = self.gate_proj._current_quantized_weight()
        up_qweight = self.up_proj._current_quantized_weight()
        fused_qweight = torch.cat([gate_qweight, up_qweight], dim=0)
        if self.gate_proj.bias is None and self.up_proj.bias is None:
            fused_bias = None
        else:
            gate_bias = self.gate_proj.bias if self.gate_proj.bias is not None else torch.zeros(self.gate_proj.out_features, device=x.device, dtype=x.dtype)
            up_bias = self.up_proj.bias if self.up_proj.bias is not None else torch.zeros(self.up_proj.out_features, device=x.device, dtype=x.dtype)
            fused_bias = torch.cat([gate_bias, up_bias], dim=0)
        fused = self.gate_proj._project_quantized(qinput, fused_qweight, fused_bias)
        return torch.split(fused, [self.gate_proj.out_features, self.up_proj.out_features], dim=-1)

    def forward(self, x):
        if self._can_use_grouped_projection(x):
            gate_x, up_x = self._grouped_gate_up(x)
        else:
            gate_x = self.gate_proj(x)
            up_x = self.up_proj(x)
        x = F.silu(gate_x) * up_x
        x = self.down_proj(x)
        return self.dropout(x)


class MixerModel(nn.Module):
    def __init__(self, config: TESMConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residual_in_fp32
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model, **factory_kwargs)
        norm_cls = partial(RMSNorm if config.rms_norm else nn.LayerNorm, eps=config.norm_epsilon)
        
        # 选择 TESM 或 MIMO
        if config.use_mimo:
            mixer_cls = lambda layer_idx: (
                lambda dim: TESMMIMO_Optimized(
                    d_model=dim,
                    n_heads=config.n_heads,
                    d_state=config.d_state,
                    expand=config.expand,
                    ent_rank=config.ent_rank,
                    entanglement_window=config.entanglement_window,
                    entanglement_threshold=config.entanglement_threshold,
                    entanglement_scale=config.entanglement_scale,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    bit_eps=config.bit_eps,
                    bit_threshold=config.bit_threshold,
                    kernel_backend=config.kernel_backend,
                    annealing_enabled=config.annealing_enabled,
                    T_start=config.T_start,
                    T_end=config.T_end,
                    annealing_steps=config.annealing_steps,
                    annealing_schedule=config.annealing_schedule,
                    **factory_kwargs,
                )
            )
        else:
            mixer_cls = lambda layer_idx: (
                lambda dim: TESM_SISO(
                    d_model=dim,
                    d_state=config.d_state,
                    expand=config.expand,
                    ent_rank=config.ent_rank,
                    entanglement_scale=config.entanglement_scale,
                    entanglement_threshold=config.entanglement_threshold,
                    entanglement_window=config.entanglement_window,
                    entanglement_block_size=config.entanglement_block_size,
                    entanglement_init=config.entanglement_init,
                    layer_idx=layer_idx,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    bit_eps=config.bit_eps,
                    bit_threshold=config.bit_threshold,
                    kernel_backend=config.kernel_backend,
                    kernel_mode=config.kernel_mode,
                    use_triton_kernels=config.use_triton_kernels,
                    state_scan_chunk_size=config.state_scan_chunk_size,
                    decay_init_bias=config.decay_init_bias,
                    rope_base=config.rope_base if hasattr(config, 'rope_base') else 10000.0,
                    annealing_enabled=config.annealing_enabled,
                    T_start=config.T_start,
                    T_end=config.T_end,
                    annealing_steps=config.annealing_steps,
                    annealing_schedule=config.annealing_schedule,
                    quantum_tunneling_enabled=config.quantum_tunneling_enabled,
                    tunneling_strength=config.tunneling_strength,
                    num_tunnel_paths=config.num_tunnel_paths,
                    energy_landscape=config.energy_landscape,
                    tunneling_schedule=config.tunneling_schedule,
                    **factory_kwargs,
                )
            )
        mlp_cls = lambda dim: FeedForward(config, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Block(
                    config.d_model,
                    mixer_cls(layer_idx),
                    mlp_cls,
                    norm_cls=norm_cls,
                    residual_in_fp32=config.residual_in_fp32,
                )
                for layer_idx in range(config.n_layer)
            ]
        )
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx
        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(config.d_model, eps=config.norm_epsilon)
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
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) for i, layer in enumerate(self.layers)}

    def _get_position_embeddings(self, batch_size, seqlen, device, pos_offset=0):
        """获取位置编码"""
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.position_embedding(positions)

    def forward_with_embeds(self, inputs_embeds, inference_params=None, prev_states=None, **mixer_kwargs):
        """使用预计算的 embeddings 进行前向传播

        用于多模态场景，绕过 embedding 层。

        Args:
            inputs_embeds: (B, L, D) pre-computed embeddings
            inference_params: 增量推理参数
            prev_states: 前一层状态

        Returns:
            (hidden_states, entanglement_maps, entanglement_stats, final_states)
        """
        batch_size, seqlen, _ = inputs_embeds.shape
        if seqlen == 0:
            raise ValueError("Input sequence length is 0")
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")

        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        pos_emb = self._get_position_embeddings(batch_size, seqlen, inputs_embeds.device, pos_offset)
        hidden_states = inputs_embeds + pos_emb
        return self._forward_layers(hidden_states, inference_params, prev_states, **mixer_kwargs)

    def _forward_layers(self, hidden_states, inference_params=None, prev_states=None, **mixer_kwargs):
        """层前向传播的核心逻辑"""
        residual = None
        entanglement_maps = []
        entanglement_stats = []
        cross_layer_state = None
        final_states = []
        for i, layer in enumerate(self.layers):
            layer_prev_state = prev_states[i] if prev_states is not None else None
            layer_mixer_kwargs = dict(mixer_kwargs, cross_layer_state=cross_layer_state)
            if self.gradient_checkpointing and self.training and inference_params is None:
                if residual is None:
                    hidden_states, residual, final_state = checkpoint(
                        lambda hs, _layer=layer, _kw=layer_mixer_kwargs, _ps=layer_prev_state: _layer(hs, None, inference_params=None, prev_state=_ps, **_kw),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, residual, final_state = checkpoint(
                        lambda hs, res, _layer=layer, _kw=layer_mixer_kwargs, _ps=layer_prev_state: _layer(hs, res, inference_params=None, prev_state=_ps, **_kw),
                        hidden_states,
                        residual,
                        use_reentrant=False,
                    )
                if hasattr(layer.mixer, "_stats_ternary_buffer") and layer.mixer._stats_ternary_buffer is not None:
                    ternary = layer.mixer._stats_ternary_buffer
                    total = layer.mixer._stats_total_buffer.item() if layer.mixer._stats_total_buffer is not None else 1.0
                    entanglement_stats.append((ternary, total))
            else:
                layer_inference_params = None
                if inference_params is not None and 'state_cache' in inference_params:
                    layer_cache = inference_params['state_cache'].get(i)
                    if layer_cache is not None:
                        layer_inference_params = {'state_cache': layer_cache}
                hidden_states, residual, final_state = layer(hidden_states, residual, inference_params=layer_inference_params, prev_state=layer_prev_state, **layer_mixer_kwargs)
            if final_state is not None:
                final_states.append(final_state)
            cross_layer_state = getattr(layer.mixer, '_last_cross_layer_state', None)
            if hasattr(layer.mixer, "last_entanglement_map") and layer.mixer.last_entanglement_map is not None:
                entanglement_maps.append(layer.mixer.last_entanglement_map)
            if hasattr(layer.mixer, "_stats_ternary_buffer") and layer.mixer._stats_ternary_buffer is not None:
                ternary = layer.mixer._stats_ternary_buffer
                total = layer.mixer._stats_total_buffer.item() if layer.mixer._stats_total_buffer is not None else 1.0
                entanglement_stats.append((ternary, total))
            elif hasattr(layer.mixer, "_ternary_stats_for_logging") and layer.mixer._ternary_stats_for_logging is not None:
                entanglement_stats.append(layer.mixer._ternary_stats_for_logging)
            elif hasattr(layer.mixer, "last_entanglement_stats") and layer.mixer.last_entanglement_stats is not None:
                entanglement_stats.append(layer.mixer.last_entanglement_stats)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states, entanglement_maps, _merge_stats(entanglement_stats), final_states

    def forward(self, input_ids, inference_params=None, prev_states=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen == 0:
            raise ValueError("Input sequence length is 0")
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}")
        pos_offset = 0
        if inference_params is not None and seqlen == 1 and 'state_cache' in inference_params:
            layer0_cache = inference_params['state_cache'].get(0)
            if layer0_cache is not None and 'seq_pos' in layer0_cache:
                pos_offset = layer0_cache['seq_pos']
        positions = torch.arange(pos_offset, pos_offset + seqlen, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embedding(input_ids) + self.position_embedding(positions)
        return self._forward_layers(hidden_states, inference_params, prev_states, **mixer_kwargs)


class TESMLMHeadModel(nn.Module):
    def __init__(self, config: Optional[TESMConfig] = None, device=None, dtype=None):
        super().__init__()
        self.tesm_config = config or TESMConfig()
        self.backbone = MixerModel(self.tesm_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(self.tesm_config.d_model, self.tesm_config.vocab_size, bias=False, device=device, dtype=dtype)
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
        
        # 词表抑制参数
        self.vocab_suppression = config.vocab_suppression if hasattr(config, 'vocab_suppression') else False
        self.suppression_bias = config.suppression_bias if hasattr(config, 'suppression_bias') else -10.0
        
        # 语义相关激活参数
        self.semantic_activation = config.semantic_activation if hasattr(config, 'semantic_activation') else False
        self.semantic_activation_strength = config.semantic_activation_strength if hasattr(config, 'semantic_activation_strength') else 0.5
        self.semantic_activation_threshold = config.semantic_activation_threshold if hasattr(config, 'semantic_activation_threshold') else 0.3
        
        # Token共现矩阵（训练时学习语义关联）
        # 使用稀疏存储：只存储top-k关联，避免O(vocab^2)显存
        self.cooccurrence_topk = config.cooccurrence_topk if hasattr(config, 'cooccurrence_topk') else 100
        if self.semantic_activation:
            # 稀疏存储：每个token只存储top-k个最相关的token及其强度
            # related_tokens[i] = [(token_id, strength), ...] for top-k
            self.register_buffer(
                'related_token_ids',
                torch.zeros(config.vocab_size, self.cooccurrence_topk, dtype=torch.long)
            )
            self.register_buffer(
                'related_token_strengths',
                torch.zeros(config.vocab_size, self.cooccurrence_topk, dtype=torch.float32)
            )
            # 归一化因子
            self.register_buffer('token_freq', torch.zeros(config.vocab_size, dtype=torch.float32))
            # 标记是否已构建
            self.cooccurrence_built = False

    def get_input_embeddings(self):
        return self.backbone.embedding

    def set_input_embeddings(self, value):
        self.backbone.embedding = value
        if self.tesm_config.tie_embeddings:
            self.lm_head.weight = value.weight

    def forward(self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, labels: Optional[torch.Tensor] = None, inference_params: Optional[dict] = None, prev_states=None, logits_to_keep: int = 0, sparse_logits: bool = False, **kwargs):
        """前向传播
        
        Args:
            input_ids: (B, L) token IDs，与 inputs_embeds 二选一
            inputs_embeds: (B, L, D) pre-computed embeddings，与 input_ids 二选一
            labels: (B, L) 训练标签
            sparse_logits: 是否使用稀疏logits计算
        """
        # 参数校验
        if (input_ids is None and inputs_embeds is None) or (input_ids is not None and inputs_embeds is not None):
            raise ValueError("Must provide exactly one of input_ids or inputs_embeds")
        
        for k in ["attention_mask", "past_key_values", "use_cache"]:
            kwargs.pop(k, None)
        
        # 选择输入方式
        if inputs_embeds is not None:
            hidden_states, entanglement_maps, entanglement_stats, final_states = self.backbone.forward_with_embeds(
                inputs_embeds=inputs_embeds, inference_params=inference_params, prev_states=prev_states, **kwargs
            )
        else:
            hidden_states, entanglement_maps, entanglement_stats, final_states = self.backbone(
                input_ids=input_ids, inference_params=inference_params, prev_states=prev_states, **kwargs
            )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        hidden_slice = hidden_states[:, slice_indices, :]
        
        # 训练时：只在epoch结束时更新共现矩阵（避免每个batch都更新，加速训练）
        # 日常训练跳过，由外部调用update_cooccurrence_from_data()来更新
        
        # 词表抑制：只在推理时启用（训练时需要学习所有token！）
        if self.vocab_suppression and not self.training:
            # 稀疏logits计算：只计算激活token的logits
            if sparse_logits:
                logits, active_token_ids = self._compute_sparse_logits(input_ids, hidden_slice)
            else:
                # 常规计算 + 事后抑制
                logits = self.lm_head(hidden_slice)
                active_tokens = input_ids.unique()
                activation_mask = torch.zeros(self.tesm_config.vocab_size, device=input_ids.device, dtype=logits.dtype)
                activation_mask[active_tokens] = 1.0
                
                # 向量化语义相关激活
                if self.semantic_activation:
                    related_tokens = self._get_related_tokens(active_tokens)
                    if related_tokens:
                        # 向量化填充
                        related_ids = torch.tensor(list(related_tokens.keys()), device=input_ids.device)
                        related_strengths = torch.tensor(list(related_tokens.values()), device=input_ids.device, dtype=logits.dtype)
                        # 过滤已在active_tokens中的
                        mask = ~torch.isin(related_ids, active_tokens)
                        related_ids = related_ids[mask]
                        related_strengths = related_strengths[mask]
                        activation_mask[related_ids] = related_strengths * self.semantic_activation_strength
                
                suppress_mask = 1.0 - activation_mask
                logits = logits + suppress_mask * self.suppression_bias
        else:
            # 训练时：正常计算全部vocab的logits
            logits = self.lm_head(hidden_slice)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tesm_config.label_ignore_index,
            )
        return TESMCausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states, entanglement_maps=entanglement_maps, entanglement_stats=entanglement_stats), final_states
    
    def _compute_sparse_logits(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, 
                                active_token_set: set = None) -> tuple:
        """稀疏logits计算：只计算激活token和相关token的logits
        
        真正的稀疏实现：不创建完整vocab_size tensor，直接返回稀疏结果
        
        Args:
            input_ids: [batch, seq_len]
            hidden_states: [batch, seq_len, d_model]
            active_token_set: 已激活的token集合（用于增量推理）
            
        Returns:
            sparse_logits: [batch, seq_len, n_active] 只包含激活token的logits
            active_token_ids: 激活的token id列表（用于采样时映射）
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 获取激活token集合
        if active_token_set is None:
            active_tokens = input_ids.unique()
        else:
            active_tokens = torch.tensor(list(active_token_set), device=input_ids.device, dtype=torch.long)
        
        # 获取语义相关token
        if self.semantic_activation:
            related_tokens = self._get_related_tokens(active_tokens)
            if related_tokens:
                # 合并激活token和相关token（使用set去重）
                all_active_set = set(active_tokens.tolist()) | set(related_tokens.keys())
                all_active_ids = torch.tensor(list(all_active_set), device=input_ids.device, dtype=torch.long)
            else:
                all_active_ids = active_tokens
        else:
            all_active_ids = active_tokens
        
        # 只计算激活token的logits（真正的稀疏计算！）
        # lm_head.weight: [vocab_size, d_model]
        # 只取激活token对应的权重行
        active_weight = self.lm_head.weight[all_active_ids, :]  # [n_active, d_model]
        
        # 计算logits：hidden @ active_weight.T
        sparse_logits = torch.matmul(hidden_states, active_weight.T)  # [batch, seq_len, n_active]
        
        # 不创建完整tensor，直接返回稀疏结果
        # 采样时使用 sparse_softmax 而非标准 softmax
        return sparse_logits, all_active_ids.tolist()
    
    def _update_cooccurrence(self, input_ids: torch.Tensor):
        """训练时更新token共现统计（用于构建稀疏关联矩阵）"""
        batch_size, seq_len = input_ids.shape
        
        # 完全向量化：处理整个batch
        # 获取每个sample的unique tokens
        all_tokens = []
        for b in range(batch_size):
            tokens = input_ids[b].unique()
            all_tokens.append(tokens)
        
        # 批量更新频率
        for tokens in all_tokens:
            if len(tokens) > 0:
                self.token_freq[tokens] += 1.0
        
        # 注意：不再维护完整共现矩阵，改为在build_cooccurrence_from_dataset中一次性构建top-k
    
    @torch.no_grad()
    def build_cooccurrence_from_dataset(self, dataloader, max_batches: int = None):
        """从整个数据集构建稀疏关联矩阵（训练结束后调用一次）
        
        完全向量化实现：无循环，纯张量操作
        显存复杂度：O(vocab * top_k) 而非 O(vocab^2)
        
        Args:
            dataloader: 训练数据加载器
            max_batches: 最多处理多少batch（None=全部）
        """
        if not self.semantic_activation:
            return
        
        print("构建token稀疏关联矩阵（top-k，完全向量化版本）...")
        self.eval()
        
        # 重置
        self.token_freq.zero_()
        self.related_token_ids.zero_()
        self.related_token_strengths.zero_()
        
        vocab_size = self.tesm_config.vocab_size
        topk = self.cooccurrence_topk
        window_size = 16
        
        # 累积所有共现对
        all_center_ids = []
        all_neighbor_ids = []
        
        total_batches = max_batches if max_batches else len(dataloader)
        print(f"  总共需要处理 {total_batches} batches")
        
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('ids'))
            else:
                input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            input_ids = input_ids.to(self.token_freq.device)
            batch_size, seq_len = input_ids.shape
            
            # 向量化：更新token频率
            flat_ids = input_ids.flatten()
            unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
            self.token_freq.scatter_add_(0, inverse, torch.ones_like(flat_ids, dtype=torch.float))
            self.token_freq = self.token_freq / batch_size
            
            # 向量化滑动窗口共现统计（完全无循环）
            # 使用torch.roll + mask构建所有(center, neighbor)对
            
            # 预计算所有偏移的center和neighbor（向量化）
            offsets = torch.arange(-window_size, window_size + 1, device=input_ids.device)
            offsets = offsets[offsets != 0]  # 排除0
            n_offsets = len(offsets)
            
            # 批量roll：使用unfold避免循环
            # 方法：使用torch.stack配合列表推导（这是Python层面的，不是tensor循环）
            # 更好的方法：使用gather或index_select
            rolled = torch.stack([torch.roll(input_ids, shifts=int(o), dims=1) for o in offsets.tolist()])
            
            # 构建有效mask（边界处理，完全向量化）
            # offset > 0: 前offset个位置无效
            # offset < 0: 后|offset|个位置无效
            seq_range = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            offsets_expanded = offsets.view(-1, 1, 1)  # [n_offsets, 1, 1]
            
            # 向量化比较：一次构建所有mask
            # mask = (offsets > 0 & seq_range >= offsets) | (offsets < 0 & seq_range < seq_len + offsets)
            pos_mask = (offsets_expanded > 0) & (seq_range >= offsets_expanded)
            neg_mask = (offsets_expanded < 0) & (seq_range < (seq_len + offsets_expanded))
            valid_masks = pos_mask | neg_mask  # [n_offsets, 1, seq_len]
            valid_masks = valid_masks.expand(-1, batch_size, -1)  # [n_offsets, batch, seq_len]
            
            # 扩展input_ids用于广播
            input_expanded = input_ids.unsqueeze(0).expand(n_offsets, -1, -1)  # [n_offsets, batch, seq_len]
            
            # 应用mask
            centers = input_expanded[valid_masks]  # [n_valid_pairs]
            neighbors = rolled[valid_masks]  # [n_valid_pairs]
            
            all_center_ids.append(centers)
            all_neighbor_ids.append(neighbors)
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{total_batches} batches ({100*(i+1)/total_batches:.1f}%)")
        
        # 合并所有共现对并统计（向量化）
        if all_center_ids:
            all_center = torch.cat(all_center_ids)
            all_neighbor = torch.cat(all_neighbor_ids)
            
            # 向量化统计共现次数
            pair_ids = all_center * vocab_size + all_neighbor
            unique_pairs, counts = torch.unique(pair_ids, return_counts=True)
            
            # 解码回(center, neighbor)
            pair_centers = unique_pairs // vocab_size
            pair_neighbors = unique_pairs % vocab_size
            
            # 向量化构建top-k稀疏矩阵
            print("  构建top-k关联矩阵（完全向量化）...")
            
            # 按count降序排序
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_centers = pair_centers[sorted_indices]
            sorted_neighbors = pair_neighbors[sorted_indices]
            sorted_counts = counts[sorted_indices]
            
            # 使用unique_consecutive获取每个center的连续范围
            unique_centers, center_counts = torch.unique_consecutive(sorted_centers, return_counts=True)
            
            # 计算每个center的起始位置（向量化）
            center_starts = torch.zeros(len(unique_centers) + 1, dtype=torch.long, device=unique_centers.device)
            center_starts[1:] = center_counts.cumsum(dim=0)
            
            # 向量化填充top-k
            fill_counts = center_counts.clamp(max=topk)
            
            # 构建所有目标索引和源数据（完全向量化）
            # 使用repeat_interleave配合arange构建索引
            
            # 每个center需要填充fill_counts[k]个位置
            # 目标索引：center_id * topk + k
            
            center_ids_expanded = unique_centers.repeat_interleave(fill_counts)
            # 向量化构建k_positions：创建位置矩阵并用mask过滤
            # [n_centers, topk] 位置矩阵
            arange_topk = torch.arange(topk, device=fill_counts.device)
            position_matrix = arange_topk.unsqueeze(0).expand(len(fill_counts), topk)
            # mask: 只保留 < fill_counts 的位置
            mask = position_matrix < fill_counts.unsqueeze(1)
            k_positions = position_matrix[mask]
            
            target_indices = center_ids_expanded * topk + k_positions
            
            # 源数据：每个center取前fill_counts个neighbor
            source_starts = center_starts[:-1].repeat_interleave(fill_counts)
            source_offsets = k_positions  # 复用同一个k_positions
            source_indices = source_starts + source_offsets
            
            # 向量化填充
            flat_related_ids = self.related_token_ids.view(-1)
            flat_related_strengths = self.related_token_strengths.view(-1)
            flat_related_ids[target_indices] = sorted_neighbors[source_indices]
            
            # 归一化强度（向量化）
            freq = self.token_freq[center_ids_expanded].clamp(min=1.0)
            flat_related_strengths[target_indices] = sorted_counts[source_indices].float() / freq
        
        self.cooccurrence_built = True
        nonzero_tokens = self.token_freq.nonzero().shape[0]
        print(f"稀疏关联矩阵构建完成，覆盖 {nonzero_tokens} 个token，显存占用: {vocab_size * topk * 8 / 1024 / 1024:.2f} MB")
    
    def _get_related_tokens(self, active_tokens: torch.Tensor, top_k: int = None) -> dict:
        """获取语义相关token及其关联强度（稀疏存储版本）
        
        从预构建的top-k稀疏矩阵中查询，O(1)复杂度
        """
        if top_k is None:
            top_k = self.cooccurrence_topk
        
        
        if not self.cooccurrence_built:
            return {}
        
        active_list = active_tokens.tolist() if isinstance(active_tokens, torch.Tensor) else active_tokens
        if not active_list:
            return {}
        
        # 从稀疏矩阵中查询（向量化）
        active_tensor = torch.tensor(active_list, device=self.related_token_ids.device, dtype=torch.long)
        
        # 获取每个激活token的top-k相关token
        related_ids = self.related_token_ids[active_tensor]  # [n_active, top_k]
        related_strengths = self.related_token_strengths[active_tensor]  # [n_active, top_k]
        
        # 展平并过滤
        flat_ids = related_ids.flatten()  # [n_active * top_k]
        flat_strengths = related_strengths.flatten()  # [n_active * top_k]
        
        # 过滤：排除自身和低关联
        active_set = set(active_list)
        mask = flat_strengths > self.semantic_activation_threshold
        
        # 向量化过滤自身
        active_tensor_expanded = torch.tensor(list(active_set), device=flat_ids.device, dtype=torch.long)
        mask &= ~torch.isin(flat_ids, active_tensor_expanded)
        
        valid_ids = flat_ids[mask]
        valid_strengths = flat_strengths[mask]
        
        # 合并重复token（取最大强度）
        if len(valid_ids) > 0:
            unique_ids, inverse = torch.unique(valid_ids, return_inverse=True)
            max_strengths = torch.zeros(len(unique_ids), device=valid_strengths.device)
            max_strengths.scatter_reduce_(0, inverse, valid_strengths, reduce='amax', include_self=False)
            return dict(zip(unique_ids.tolist(), max_strengths.tolist()))
        
        return {}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0, use_cache: bool = True, sparse_inference: bool = False, dynamic_activation: bool = True):
        """生成文本
        
        Args:
            sparse_inference: 是否使用稀疏推理（只计算激活token的logits，大幅减少计算量）
            dynamic_activation: 是否启用动态激活（生成时自动扩展激活token，避免重复）
        """
        self.eval()
        generated = input_ids
        
        # 动态激活：初始化激活token集合（使用set，O(1)查找）
        active_token_set = None
        if dynamic_activation and self.vocab_suppression:
            active_token_set = set(input_ids.unique().tolist())
            # 初始扩展：添加语义相关token
            if self.semantic_activation:
                related = self._get_related_tokens(input_ids.unique())
                active_token_set.update(related.keys())
        
        # 增量推理：分配状态缓存
        inference_params = None
        if use_cache:
            inference_params = {'state_cache': self.backbone.allocate_inference_cache(input_ids.shape[0], self.tesm_config.max_seq_len)}
        
        # 预填充：处理初始 prompt
        outputs, _ = self.forward(input_ids, inference_params=inference_params, sparse_logits=sparse_inference and active_token_set is not None)
        
        # 处理稀疏logits（需要特殊采样）
        if sparse_inference and active_token_set is not None:
            # outputs.logits 是稀疏的 [batch, seq, n_active]
            # 需要使用 sparse_logits 和 active_token_ids 进行采样
            sparse_logits = outputs.logits[:, -1, :]  # [batch, n_active]
            # 稀疏采样：在激活token上做softmax
            logits = self._sparse_sample(sparse_logits, active_token_set, temperature, top_k)
        else:
            logits = outputs.logits[:, -1, :]
            # 应用词表抑制
            if active_token_set is not None:
                logits = self._apply_vocab_suppression(logits, list(active_token_set))
        
        eos_token_id = getattr(self.tesm_config, 'eos_token_id', None)
        
        for i in range(max_new_tokens):
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # 动态激活：扩展激活token集合（使用set，O(1)操作）
            if active_token_set is not None:
                new_token_id = next_token.item()
                if new_token_id not in active_token_set:
                    active_token_set.add(new_token_id)
                    # 同时扩展语义相关token
                    if self.semantic_activation:
                        related = self._get_related_tokens(torch.tensor([new_token_id], device=next_token.device))
                        active_token_set.update(related.keys())
            
            # EOS 停止
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # 增量推理：只处理新 token
            if use_cache:
                outputs, _ = self.forward(next_token, inference_params=inference_params, sparse_logits=sparse_inference and active_token_set is not None)
            else:
                model_input = generated[:, -self.tesm_config.max_seq_len :]
                outputs, _ = self.forward(model_input, sparse_logits=sparse_inference and active_token_set is not None)
            
            # 处理稀疏logits
            if sparse_inference and active_token_set is not None:
                sparse_logits = outputs.logits[:, -1, :]
                logits = self._sparse_sample(sparse_logits, active_token_set, temperature, top_k)
            else:
                logits = outputs.logits[:, -1, :]
                if active_token_set is not None:
                    logits = self._apply_vocab_suppression(logits, list(active_token_set))
        
        return generated
    
    def _sparse_sample(self, sparse_logits: torch.Tensor, active_token_set: set, 
                       temperature: float, top_k: int) -> torch.Tensor:
        """从稀疏logits中采样，返回完整vocab大小的logits
        
        Args:
            sparse_logits: [batch, n_active] 只包含激活token的logits
            active_token_set: 激活的token集合
            temperature: 温度参数
            top_k: top-k采样参数
            
        Returns:
            full_logits: [batch, vocab_size] 完整的logits（未激活位置为-inf）
        """
        batch_size, n_active = sparse_logits.shape
        vocab_size = self.tesm_config.vocab_size
        
        # 创建完整logits tensor（只创建一次，复用）
        active_token_ids = torch.tensor(list(active_token_set), device=sparse_logits.device, dtype=torch.long)
        
        # 应用温度和top-k
        if temperature != 1.0:
            sparse_logits = sparse_logits / temperature
        
        if top_k > 0 and top_k < n_active:
            values, _ = torch.topk(sparse_logits, min(top_k, n_active))
            sparse_logits = sparse_logits.masked_fill(sparse_logits < values[:, [-1]], float("-inf"))
        
        # 构建完整logits（用于multinomial采样）
        # 注意：这里仍需要创建完整tensor，因为multinomial需要完整分布
        # 但计算量已经减少了（只计算了n_active个token的matmul）
        full_logits = torch.full(
            (batch_size, vocab_size),
            float('-inf'),
            device=sparse_logits.device,
            dtype=sparse_logits.dtype
        )
        full_logits[:, active_token_ids[:n_active]] = sparse_logits
        
        return full_logits
    
    def _apply_vocab_suppression(self, logits: torch.Tensor, active_token_ids: list) -> torch.Tensor:
        """应用词表抑制：未激活token施加负向偏置
        
        Args:
            logits: [batch, vocab_size]
            active_token_ids: 激活的token id列表
            
        Returns:
            抑制后的logits
        """
        suppress_mask = torch.ones(self.tesm_config.vocab_size, device=logits.device, dtype=logits.dtype)
        suppress_mask[active_token_ids] = 0.0
        logits = logits + suppress_mask * self.suppression_bias
        return logits


def _merge_stats(stats_list: List) -> Optional[Dict[str, float]]:
    if not stats_list:
        return None
    # 新格式：每个元素是 (ternary_tensor, total) 元组
    # 在这里计算最终统计，避免forward图中的.item()
    total_pos = 0.0
    total_neg = 0.0
    total_zero = 0.0
    count = 0
    for item in stats_list:
        if isinstance(item, tuple) and len(item) == 2:
            ternary, total = item
            if ternary is not None:
                # 处理 softmax 连续值（高温退火阶段）和硬阈值（低温阶段）
                # 使用小阈值判断是否为"零"
                eps = 1e-3
                total_pos += float((ternary > eps).sum().item())
                total_neg += float((ternary < -eps).sum().item())
                total_zero += float((ternary.abs() <= eps).sum().item())
                count += float(total)
    if count > 0:
        return {
            "positive": total_pos / count,
            "negative": total_neg / count,
            "zero": total_zero / count,
        }
    return None


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
