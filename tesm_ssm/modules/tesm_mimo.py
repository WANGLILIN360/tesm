"""
TESM_SISO-MIMO 优化版

MIMO 是 TESM_SISO 的多头扩展模式，继承 TESM_SISO 获得:
- 温度退火纠缠调度
- BitLinear 量化
- 状态扫描
- 位置编码 (RoPE)

新增:
- 多头并行 (n_heads)
- TileLang kernel 加速
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tesm_ssm.modules.tesm import TESM_SISO, BitLinear
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from tesm import TESM_SISO, BitLinear

# 导入优化 kernel
try:
    from tesm_ssm.ops.tilelang.tesm_mimo_tilelang import (
        tesm_chunked_scan_tilelang_fwd,
        tesm_local_entanglement_tilelang_fwd,
        tesm_chunked_scan_tilelang_autograd,
        tesm_local_entanglement_tilelang_autograd,
    )
    TILELANG_AVAILABLE = True
except ImportError:
    TILELANG_AVAILABLE = False
    tesm_chunked_scan_tilelang_autograd = None
    tesm_local_entanglement_tilelang_autograd = None

try:
    from tesm_ssm.ops.triton.tesm_mimo_kernel import (
        tesm_state_scan_triton,
        tesm_state_scan_triton_autograd,
        tesm_local_entanglement_triton,
        tesm_local_entanglement_triton_autograd,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    tesm_state_scan_triton_autograd = None
    tesm_local_entanglement_triton = None
    tesm_local_entanglement_triton_autograd = None

try:
    from tesm_ssm.ops.cuda import (
        tesm_cuda_is_available,
        cuda_chunk_state_scan_mimo,
        cuda_chunk_state_scan_mimo_autograd,
        cuda_local_entanglement_mimo,
        cuda_local_entanglement_mimo_autograd,
        cuda_global_entanglement_mimo,
        cuda_global_entanglement_mimo_autograd,
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    tesm_cuda_is_available = lambda: False
    cuda_chunk_state_scan_mimo_autograd = None
    cuda_local_entanglement_mimo_autograd = None
    cuda_global_entanglement_mimo = None
    cuda_global_entanglement_mimo_autograd = None


class TESMMIMO_Optimized(TESM_SISO):
    """TESM-MIMO 优化版 - 继承 TESM_SISO
    
    MIMO 是 TESM_SISO 的多头扩展模式:
    - 继承 TESM_SISO 的温度退火纠缠调度
    - 继承 BitLinear 量化
    - 新增多头并行 (n_heads)
    - 新增 TileLang kernel 加速
    
    使用方式:
        # 启用 MIMO 模式
        model = TESMMIMO_Optimized(
            d_model=512,
            n_heads=4,           # 多头数量
            annealing_enabled=True,  # 温度退火
            ...
        )
    """
    
    def __init__(
        self,
        d_model,
        d_state=256,
        n_heads=4,           # MIMO: 多头数量
        mimo_rank=4,         # MIMO: rank 扩展维度 (参考 Mamba-3)
        expand=2,
        ent_rank=48,
        entanglement_window=16,
        max_seq_len=2048,
        entanglement_threshold=0.5,
        dropout=0.0,
        bit_eps=1e-5,
        bit_threshold=0.5,
        kernel_backend="auto",
        # 温度退火参数 (继承自 TESM_SISO)
        annealing_enabled=True,
        T_start=10.0,
        T_end=0.1,
        annealing_steps=1000,
        annealing_schedule='cosine',
        **kwargs
    ):
        # 调用父类 TESM_SISO 初始化
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            ent_rank=ent_rank,
            entanglement_window=entanglement_window,
            max_seq_len=max_seq_len,
            entanglement_threshold=entanglement_threshold,
            dropout=dropout,
            bit_eps=bit_eps,
            bit_threshold=bit_threshold,
            annealing_enabled=annealing_enabled,
            T_start=T_start,
            T_end=T_end,
            annealing_steps=annealing_steps,
            annealing_schedule=annealing_schedule,
            **kwargs
        )
        
        # MIMO 特有参数
        self.n_heads = n_heads
        self.mimo_rank = mimo_rank  # MIMO rank 扩展维度
        
        # 每个头的维度 (与 Mamba-3 一致: headdim = d_inner // nheads)
        self.d_inner = int(expand * d_model)
        self.d_head = self.d_inner // n_heads  # 每头维度
        self.d_state_total = d_state * n_heads  # 总状态维度
        
        self.kernel_backend = kernel_backend
        
        # 重写衰减偏置为多头格式 (每个头独立的 d_state 维度偏置)
        decay_init = float(kwargs.get('decay_init_bias', 3.0))
        self.decay_bias = nn.Parameter(torch.full((n_heads, d_state), decay_init,
                                                   device=kwargs.get('device'),
                                                   dtype=torch.float32))
        
        # 重写局部纠缠偏置为多头格式
        if entanglement_window > 0:
            self.local_entanglement_bias = nn.Parameter(
                torch.randn(n_heads, entanglement_window, 
                           device=kwargs.get('device'), dtype=torch.float32) * 0.02
            )
        else:
            self.register_parameter("local_entanglement_bias", None)
        
        
        # RoPE
        self.rope_base = float(kwargs.get('rope_base', 10000.0))
        
        # MIMO 投影参数 (参考 Mamba-3)
        # mimo_x: V 投影 (d_state -> mimo_rank * d_state)
        # mimo_z: Z (gate) 投影
        # mimo_o: 输出合并投影 (mimo_rank * d_state -> d_state)
        # 注意: TESM_SISO 的状态维度是 d_state，不是 headdim
        mimo_x_init = torch.ones(n_heads, mimo_rank, d_state, 
                                 device=kwargs.get('device'), dtype=torch.float32) / mimo_rank
        mimo_z_init = torch.ones(n_heads, mimo_rank, d_state,
                                 device=kwargs.get('device'), dtype=torch.float32)
        mimo_o_init = torch.ones(n_heads, mimo_rank, d_state,
                                 device=kwargs.get('device'), dtype=torch.float32) / mimo_rank
        
        self.mimo_x = nn.Parameter(mimo_x_init, requires_grad=True)
        self.mimo_z = nn.Parameter(mimo_z_init, requires_grad=True)
        self.mimo_o = nn.Parameter(mimo_o_init, requires_grad=True)
        
        # 重写输入投影 (MIMO 需要不同的维度)
        # SISO: (2 * d_model) + (3 * d_state) + (2 * ent_rank)
        # MIMO: local + out_gate + decay + write + state_value + ent_q + ent_k
        # decay/write/state_value: d_state * n_heads (每个头独立的 d_state)
        # ent_q/ent_k: ent_rank * n_heads
        d_in_proj_mimo = (2 * d_model) + (3 * d_state * n_heads) + (2 * ent_rank * n_heads)
        self.in_proj = BitLinear(d_model, d_in_proj_mimo, bias=False,
                                  bit_eps=bit_eps, bit_threshold=bit_threshold,
                                  kernel_backend=kernel_backend,
                                  device=kwargs.get('device'), dtype=kwargs.get('dtype'))
        
        # 重写输出投影 (MIMO: 从 d_state_total 投影回 d_model)
        self.state_proj = BitLinear(self.d_state_total, d_model, bias=False,
                                    bit_eps=bit_eps, bit_threshold=bit_threshold,
                                    kernel_backend=kernel_backend,
                                    device=kwargs.get('device'), dtype=kwargs.get('dtype'))
        self.ent_proj = BitLinear(self.d_state_total, d_model, bias=False,
                                  bit_eps=bit_eps, bit_threshold=bit_threshold,
                                  kernel_backend=kernel_backend,
                                  device=kwargs.get('device'), dtype=kwargs.get('dtype'))
    
    def _apply_rope(self, x, pos_offset=0):
        """位置编码 - MIMO 多头版本 (4D 张量)"""
        B, L, H, D = x.shape
        half = D // 2
        pos = torch.arange(pos_offset, pos_offset + L, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(half, device=x.device, dtype=torch.float32)
        theta = pos.unsqueeze(1).unsqueeze(1) * (1.0 / (self.rope_base ** (2.0 * dim_idx / D)))
        cos_t = theta.cos().unsqueeze(0).to(x.dtype)  # (1, L, 1, half)
        sin_t = theta.sin().unsqueeze(0).to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)
    
    # ========================================================================
    # 后端选择辅助方法 (参考 SISO)
    # ========================================================================
    
    def _can_use_cuda_ext_training_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and CUDA_AVAILABLE
            and tesm_cuda_is_available()
            and tensor.is_cuda
            and torch.is_grad_enabled()
        )
    
    def _can_use_cuda_ext_fast_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and CUDA_AVAILABLE
            and tesm_cuda_is_available()
            and tensor.is_cuda
            and not torch.is_grad_enabled()
        )
    
    def _can_use_triton_training_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "triton"}
            and TRITON_AVAILABLE
            and tensor.is_cuda
            and torch.is_grad_enabled()
        )
    
    def _can_use_triton_fast_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "triton"}
            and TRITON_AVAILABLE
            and tensor.is_cuda
            and not torch.is_grad_enabled()
        )
    
    def _can_use_tilelang_training_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tesm_chunked_scan_tilelang_autograd is not None
            and tensor.is_cuda
            and torch.is_grad_enabled()
        )
    
    def _can_use_tilelang_fast_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tensor.is_cuda
            and not torch.is_grad_enabled()
        )
    
    # ========================================================================
    # MIMO 状态扫描 (统一后端选择)
    # ========================================================================
    
    def _parallel_state_scan_mimo(self, decay, update):
        """MIMO 优化状态扫描 - 统一后端选择 (CUDA/Triton/TileLang/PyTorch)
        
        后端优先级 (参考 SISO):
        1. CUDA training (cuda_chunk_state_scan_mimo_autograd)
        2. CUDA inference (cuda_chunk_state_scan_mimo)
        3. TileLang training (tesm_chunked_scan_tilelang_autograd)
        4. TileLang inference (tesm_chunked_scan_tilelang_fwd)
        5. Triton inference (tesm_state_scan_triton) - 无 autograd
        6. PyTorch fallback
        
        Args:
            decay: (B, L, H, D)
            update: (B, L, H, D)
        
        Returns:
            states: (B, L, H, D)
        """
        B, L, H, D = decay.shape
        orig_dtype = decay.dtype
        chunk_size = min(max(self.state_scan_chunk_size, 1), L)
        
        # 1. CUDA training path
        if self._can_use_cuda_ext_training_path(decay) and cuda_chunk_state_scan_mimo_autograd is not None:
            return cuda_chunk_state_scan_mimo_autograd(decay.float(), update.float(), chunk_size).to(orig_dtype)
        
        # 2. CUDA inference path
        if self._can_use_cuda_ext_fast_path(decay) and cuda_chunk_state_scan_mimo_autograd is not None:
            return cuda_chunk_state_scan_mimo(decay.float(), update.float(), chunk_size).to(orig_dtype)
        
        # 3. TileLang training path
        if self._can_use_tilelang_training_path(decay) and tesm_chunked_scan_tilelang_autograd is not None:
            return tesm_chunked_scan_tilelang_autograd(decay.float(), update.float(), chunk_size).to(orig_dtype)
        
        # 4. TileLang inference path
        if self._can_use_tilelang_fast_path(decay) and tesm_chunked_scan_tilelang_fwd is not None:
            return tesm_chunked_scan_tilelang_fwd(decay.float(), update.float()).to(orig_dtype)
        
        # 5. Triton training path
        if self._can_use_triton_training_path(decay) and tesm_state_scan_triton_autograd is not None:
            return tesm_state_scan_triton_autograd(decay.float(), update.float()).to(orig_dtype)
        
        # 6. Triton inference path
        if self._can_use_triton_fast_path(decay) and tesm_state_scan_triton is not None:
            return tesm_state_scan_triton(decay.float(), update.float()).to(orig_dtype)
        
        # 7. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            return self._parallel_state_scan_pytorch_mimo(decay, update)
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but MIMO state scan kernel not available. "
            f"Available backends: cuda={cuda_chunk_state_scan_mimo_autograd is not None}, "
            f"triton={tesm_state_scan_triton_autograd is not None}, "
            f"tilelang={tesm_chunked_scan_tilelang_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )
    
    def _parallel_state_scan_mimo_stable(self, decay, update):
        """MIMO 稳定状态扫描 - 使用 float64 防止下溢 (与 SISO 一致)
        
        Args:
            decay: (B, L, H, D)
            update: (B, L, H, D)
        
        Returns:
            states: (B, L, H, D)
        """
        B, L, H, D = decay.shape
        orig_dtype = decay.dtype
        
        # 使用 float64 保证数值稳定性 (与 SISO 一致)
        decay_f64 = decay.to(torch.float64).clamp_min(1e-12)
        update_f64 = update.to(torch.float64)
        
        # 展平多头: (B, L, H, D) -> (B*H, L, D)
        decay_flat = decay_f64.permute(0, 2, 1, 3).reshape(B * H, L, D)
        update_flat = update_f64.permute(0, 2, 1, 3).reshape(B * H, L, D)
        
        # 顺序扫描 (float64)
        states_list = []
        h = torch.zeros(B * H, D, device=decay.device, dtype=torch.float64)
        
        for t in range(L):
            h = decay_flat[:, t] * h + update_flat[:, t]
            states_list.append(h.unsqueeze(1))
        
        states = torch.cat(states_list, dim=1)
        # 重塑回多头: (B*H, L, D) -> (B, L, H, D)
        states = states.reshape(B, H, L, D).permute(0, 2, 1, 3)
        
        return states.to(dtype=orig_dtype)
    
    def _parallel_state_scan_pytorch_mimo(self, decay, update):
        """PyTorch 回退实现 (MIMO 多头版本)"""
        B, L, H, D = decay.shape
        
        decay_flat = decay.permute(0, 2, 1, 3).reshape(B * H, L, D)
        update_flat = update.permute(0, 2, 1, 3).reshape(B * H, L, D)
        
        states_list = []
        h = torch.zeros(B * H, D, device=decay.device, dtype=decay.dtype)
        
        for t in range(L):
            h = decay_flat[:, t] * h + update_flat[:, t]
            states_list.append(h.unsqueeze(1))
        
        states = torch.cat(states_list, dim=1)
        states = states.reshape(B, H, L, D).permute(0, 2, 1, 3)
        
        return states
    
    def _compute_local_entanglement_mimo(self, q, k, v, bias):
        """MIMO 局部纠缠 - 统一后端选择 (CUDA/Triton/TileLang/PyTorch)
        
        后端优先级 (参考 SISO):
        1. CUDA training (cuda_local_entanglement_mimo_autograd)
        2. CUDA inference (cuda_local_entanglement_mimo)
        3. TileLang training (tesm_local_entanglement_tilelang_autograd)
        4. TileLang inference (tesm_local_entanglement_tilelang_fwd)
        5. PyTorch fallback
        
        继承 TESM_SISO 的温度退火调度:
        - 高温 (T > 1.0): 密集矩阵纠缠 (softmax)
        - 低温 (T <= 1.0): 硬化阈值纠缠
        """
        B, L, H, R = q.shape
        D = v.shape[-1]
        T = self.get_temperature()
        threshold = self.entanglement_threshold
        
        # 1. CUDA training path
        if self._can_use_cuda_ext_training_path(q) and cuda_local_entanglement_mimo_autograd is not None:
            return cuda_local_entanglement_mimo_autograd(q.float(), k.float(), v.float(), bias.float(), threshold)
        
        # 2. CUDA inference path
        if self._can_use_cuda_ext_fast_path(q) and cuda_local_entanglement_mimo is not None:
            return cuda_local_entanglement_mimo(q.float(), k.float(), v.float(), bias.float(), threshold)
        
        # 3. TileLang training path
        if self._can_use_tilelang_training_path(q) and tesm_local_entanglement_tilelang_autograd is not None:
            return tesm_local_entanglement_tilelang_autograd(
                q.float(), k.float(), v.float(), bias.float(),
                temperature=T,
                threshold=threshold
            )
        
        # 4. TileLang inference path
        if self._can_use_tilelang_fast_path(q) and tesm_local_entanglement_tilelang_fwd is not None:
            return tesm_local_entanglement_tilelang_fwd(
                q.float(), k.float(), v.float(), bias.float(),
                temperature=T,
                threshold=threshold
            )
        
        # 5. Triton training path
        if self._can_use_triton_training_path(q) and tesm_local_entanglement_triton_autograd is not None:
            return tesm_local_entanglement_triton_autograd(q.float(), k.float(), v.float(), bias.float(), threshold)
        
        # 6. Triton inference path
        if self._can_use_triton_fast_path(q) and tesm_local_entanglement_triton is not None:
            return tesm_local_entanglement_triton(q.float(), k.float(), v.float(), bias.float(), threshold)
        
        # 7. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            return self._compute_local_entanglement_pytorch_mimo(q, k, v, bias)
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but MIMO local entanglement kernel not available. "
            f"Available backends: cuda={cuda_local_entanglement_mimo_autograd is not None}, "
            f"triton={tesm_local_entanglement_triton_autograd is not None}, "
            f"tilelang={tesm_local_entanglement_tilelang_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )
    
    def _update_entanglement_stats_mimo(self, q, k):
        """更新 MIMO 纠缠统计 - 完全向量化版本
        
        统计始终使用硬阈值，反映最终低温时的目标纠缠比例
        """
        with torch.no_grad():
            B, L, H, R = q.shape
            W = min(self.entanglement_window, L) if self.entanglement_window > 0 else L
            threshold = self.entanglement_threshold
            
            # 全局纠缠模式：使用完整序列
            if self.entanglement_window == 0:
                # 全局纠缠：采样计算避免 O(L²)
                sample_len = min(L, 32)
                q_sample = q[:, :sample_len, :, :]  # (B, S, H, R)
                k_sample = k[:, :sample_len, :, :]  # (B, S, H, R)
                
                # 计算全局分数 (S x S)
                scores = torch.einsum('bshr,bthr->bst', q_sample, k_sample) / math.sqrt(R)
                
                # 硬阈值判断
                ternary = torch.where(
                    scores > threshold,
                    torch.ones_like(scores),
                    torch.where(scores < -threshold, -torch.ones_like(scores), torch.zeros_like(scores))
                )
            else:
                # 局部窗口纠缠
                # 采样前64个位置
                sample_len = min(L, 64)
                q_sample = q[:, :sample_len, :, :]  # (B, S, H, R)
                k_sample = k[:, :sample_len, :, :]  # (B, S, H, R)
                
                # 使用 unfold 一次性提取所有窗口
                k_windows = k_sample.unfold(1, W, 1)  # (B, S-W+1, H, R, W)
                
                valid_len = k_windows.size(1)  # S-W+1
                q_valid = q_sample[:, W-1:W-1+valid_len, :, :]  # (B, valid_len, H, R)
                
                # 广播计算所有分数
                scores = (q_valid.unsqueeze(-1) * k_windows).sum(dim=-2) / math.sqrt(R)  # (B, V, H, W)
                
                # 统一使用硬阈值判断（与 TESM_SISO 基础版一致）
                ternary = torch.where(
                    scores > threshold,
                    torch.ones_like(scores),
                    torch.where(scores < -threshold, -torch.ones_like(scores), torch.zeros_like(scores))
                )
            
            # 计算统计
            total = float(ternary.numel())
            if total > 0:
                ternary_flat = ternary.detach().flatten()
                self._ternary_stats_for_logging = (ternary_flat, total)
                # 同时设置 buffer（gradient checkpointing 安全）
                self._stats_ternary_buffer = ternary_flat
                self._stats_total_buffer = torch.tensor(total, device=ternary.device)
                self.last_entanglement_stats = {
                    "positive": float((ternary > 0).sum().item()) / total,
                    "negative": float((ternary < 0).sum().item()) / total,
                    "zero": float((ternary == 0).sum().item()) / total,
                }
            else:
                self._ternary_stats_for_logging = None
                self._stats_ternary_buffer = None
                self.last_entanglement_stats = None
    
    def _compute_local_entanglement_pytorch_mimo(self, q, k, v, bias):
        """PyTorch 回退 - 支持量子退火 + 归一化 (与 SISO 一致)"""
        B, L, H, R = q.shape
        D = v.shape[-1]
        W = self.entanglement_window
        
        # 继承自 TESM_SISO 的温度退火调度
        T = self.get_temperature()
        
        entangled = torch.zeros_like(v)
        
        for t in range(L):
            window_len = min(W, t + 1)
            
            if T > 1.0:
                # 高温: 密集矩阵纠缠 (softmax)
                scores_list = []
                v_list = []
                
                for w in range(window_len):
                    hist_t = t - W + 1 + w
                    if hist_t < 0:
                        continue
                    
                    score = (q[:, t] * k[:, hist_t]).sum(dim=-1) / math.sqrt(R)
                    score = score + bias[:, w]
                    scores_list.append(score)
                    v_list.append(v[:, hist_t])
                
                if scores_list:
                    scores = torch.stack(scores_list, dim=-1)
                    weights = F.softmax(scores / T, dim=-1)
                    
                    for w_idx, v_hist in enumerate(v_list):
                        entangled[:, t] += weights[:, :, w_idx:w_idx+1] * v_hist
            else:
                # 低温: 硬化阈值纠缠
                ternary_list = []
                v_list = []
                
                for w in range(window_len):
                    hist_t = t - W + 1 + w
                    if hist_t < 0:
                        continue
                    
                    score = (q[:, t] * k[:, hist_t]).sum(dim=-1) / math.sqrt(R)
                    score = score + bias[:, w]
                    
                    # 使用继承自 TESM_SISO 的 ternary_entanglement
                    ternary = self.ternary_entanglement(score)
                    ternary_list.append(ternary)
                    v_list.append(v[:, hist_t])
                
                if ternary_list:
                    # 堆叠三值: (B, H, W)
                    ternary_stack = torch.stack(ternary_list, dim=-1)
                    # 归一化 (与 SISO 一致): norm = |ternary|.sum().clamp_min(1.0)
                    norm = ternary_stack.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
                    weights = ternary_stack / norm
                    
                    for w_idx, v_hist in enumerate(v_list):
                        entangled[:, t] += weights[:, :, w_idx:w_idx+1] * v_hist
        
        return entangled
    
    def _compute_global_entanglement_mimo(self, q, k, states):
        """MIMO 全局纠缠 - 统一后端选择 (CUDA/Triton/TileLang/PyTorch)
        
        Args:
            q: (B, L, H, R) - 纠缠查询
            k: (B, L, H, R) - 纠缠键  
            states: (B, L, H, D) - 状态值
            
        Returns:
            entangled: (B, L, H, D) - 纠缠后的状态
            ent_change: (B, L, H, D) - 状态变化
        """
        B, L, H, R = q.shape
        D = states.shape[-1]
        
        # 计算相对位置偏置
        positions = torch.arange(L, device=q.device, dtype=q.dtype)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L]
        freq_bands = torch.arange(1, self.global_rel_pos_dim + 1, device=q.device, dtype=q.dtype)
        freq_bands = freq_bands * math.pi / 1000.0
        rel_pos_expanded = rel_pos.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
        sin_features = torch.sin(rel_pos_expanded)
        cos_features = torch.cos(rel_pos_expanded)
        embed = self.global_rel_pos_embed.to(q.dtype)
        rel_bias = (sin_features * embed).sum(dim=-1) + (cos_features * embed).sum(dim=-1)
        rel_bias = rel_bias * self.global_rel_pos_scale.to(q.dtype) + self.global_rel_pos_bias.to(q.dtype)
        
        # 1. CUDA training path
        if self._can_use_cuda_ext_training_path(q) and cuda_global_entanglement_mimo_autograd is not None:
            entangled = cuda_global_entanglement_mimo_autograd(q.float(), k.float(), states.float(), rel_bias.float(), float(self.entanglement_threshold))
            self._update_entanglement_stats_mimo(q, k)
            ent_change = entangled - states
            return entangled.to(q.dtype), ent_change.to(q.dtype)
        
        # 2. CUDA inference path
        if self._can_use_cuda_ext_fast_path(q) and cuda_global_entanglement_mimo is not None:
            entangled = cuda_global_entanglement_mimo(q.float(), k.float(), states.float(), rel_bias.float(), float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            ent_change = entangled - states
            return entangled.to(q.dtype), ent_change.to(q.dtype)
        
        # 3. TileLang training path
        if self._can_use_tilelang_training_path(q) and hasattr(self, '_tilelang_global_entanglement_available') and self._tilelang_global_entanglement_available:
            pass  # TODO: TileLang MIMO global entanglement
        
        # 4. Triton training path
        if self._can_use_triton_training_path(q) and hasattr(self, '_triton_global_entanglement_available') and self._triton_global_entanglement_available:
            pass  # TODO: Triton MIMO global entanglement
        
        # 5. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            return self._compute_global_entanglement_mimo_pytorch(q, k, states, rel_bias)
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but MIMO global entanglement kernel not available. "
            f"Available backends: cuda={cuda_global_entanglement_mimo_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )
    
    def _compute_global_entanglement_mimo_pytorch(self, q, k, states, rel_bias):
        """MIMO 全局纠缠 PyTorch 回退实现"""
        B, L, H, R = q.shape
        D = states.shape[-1]
        
        # 将多头展平为单头计算
        q_flat = q.reshape(B, L, H * R)  # (B, L, H*R)
        k_flat = k.reshape(B, L, H * R)  # (B, L, H*R)
        v_flat = states.reshape(B, L, H * D)  # (B, L, H*D)
        
        # 计算 Q-K 相似度矩阵
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / math.sqrt(H * R)  # [B, L, L]
        scores = scores + rel_bias.unsqueeze(0)
        
        # 应用因果掩码
        causal_mask = self.causal_mask[:L, :L]
        
        # 三值纠缠（在 mask 前计算）
        ternary = self.ternary_entanglement(scores)
        
        # 应用因果掩码
        ternary = ternary * causal_mask.unsqueeze(0).to(ternary.dtype)
        
        # 归一化
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        
        # 加权求和
        entangled_flat = torch.matmul(ternary / norm, v_flat)  # [B, L, H*D]
        
        # 重塑回多头
        entangled = entangled_flat.reshape(B, L, H, D)
        ent_change = entangled - states
        
        # 更新纠缠统计
        self._update_entanglement_stats(ternary)
        
        return entangled, ent_change
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """MIMO 版本推理缓存分配 - 适配多头维度
        
        与 SISO 的区别:
        - state: (batch, n_heads, d_state) 而非 (batch, d_state)
        - ent_k_cache: (batch, window, n_heads, ent_rank) 而非 (batch, window, ent_rank)
        - ent_v_cache: (batch, window, n_heads, d_state) 而非 (batch, window, d_state)
        """
        dev = self.out_proj.weight.device
        _dtype = dtype or torch.float32
        window = max(self.entanglement_window, 1)
        n_heads = self.n_heads
        
        use_paged = kwargs.get('use_paged_cache', False)
        
        if use_paged and max_seqlen > 1024:
            from tesm_ssm.utils.paged_cache import PagedStateCache
            return {
                'use_paged': True,
                'paged_cache': PagedStateCache(
                    batch_size=batch_size,
                    d_state=self.d_state * n_heads,
                    ent_rank=self.ent_rank * n_heads,
                    window=window,
                    page_size=kwargs.get('page_size', 512),
                    max_gpu_pages=kwargs.get('max_gpu_pages', 100),
                    device=dev,
                ),
                'state': torch.zeros(batch_size, n_heads, self.d_state, device=dev, dtype=torch.float64),
                'seq_pos': 0,
                'ent_k_cache': torch.zeros(batch_size, window, n_heads, self.ent_rank, device=dev, dtype=_dtype),
                'ent_v_cache': torch.zeros(batch_size, window, n_heads, self.d_state, device=dev, dtype=_dtype),
            }
        
        return {
            'use_paged': False,
            'state': torch.zeros(batch_size, n_heads, self.d_state, device=dev, dtype=torch.float64),
            'seq_pos': 0,
            'ent_k_cache': torch.zeros(batch_size, window, n_heads, self.ent_rank, device=dev, dtype=_dtype),
            'ent_v_cache': torch.zeros(batch_size, window, n_heads, self.d_state, device=dev, dtype=_dtype),
            'cache_idx': 0,
            'cache_filled': False,
        }
    
    def _compute_entanglement(self, q, k, values):
        """覆盖父类方法 - MIMO 纠缠分发
        
        父类 TESM_SISO._compute_entanglement 只处理 SISO 3D 张量 (B,L,R)，
        MIMO 需要 4D 张量 (B,L,H,R)，因此必须覆盖。
        
        Args:
            q: (B, L, H, R) 纠缠查询
            k: (B, L, H, R) 纠缠键
            values: (B, L, H, D) 状态值
            
        Returns:
            entangled: (B, L, H, D) 纠缠后的状态
        """
        if self.entanglement_window > 0 and self.local_entanglement_bias is not None:
            return self._compute_local_entanglement_mimo(q, k, values, self.local_entanglement_bias)
        else:
            entangled, _ = self._compute_global_entanglement_mimo(q, k, values)
            return entangled
    
    def _forward_incremental(self, u, inference_params, cross_layer_state=None):
        """MIMO 增量推理 - 支持多头维度
        
        与 SISO 的区别:
        - 状态: (batch, n_heads, d_state) 而非 (batch, d_state)
        - 缓存: (batch, window, n_heads, rank/d_state) 而非 (batch, window, rank/d_state)
        - 衰减偏置: (n_heads, d_state) 广播
        """
        batch, seqlen, _ = u.shape
        cache = inference_params['state_cache']
        n_heads = self.n_heads
        
        proj = self.in_proj(u)
        chunks = [
            self.d_model,
            self.d_state * n_heads,
            self.d_state * n_heads,
            self.d_state * n_heads,
            self.d_model,
            self.ent_rank * n_heads,
            self.ent_rank * n_heads,
        ]
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(proj, chunks, dim=-1)
        
        # 重塑为多头 (seqlen=1)
        decay = decay.view(batch, seqlen, n_heads, self.d_state)
        write = write.view(batch, seqlen, n_heads, self.d_state)
        state_value = state_value.view(batch, seqlen, n_heads, self.d_state)
        ent_q = ent_q.view(batch, seqlen, n_heads, self.ent_rank)
        ent_k = ent_k.view(batch, seqlen, n_heads, self.ent_rank)
        
        state_value = torch.tanh(state_value)
        decay = torch.sigmoid(decay + self.decay_bias.unsqueeze(0).unsqueeze(0))
        write = torch.sigmoid(write)
        out_gate = torch.sigmoid(out_gate)
        
        # RoPE at current position
        cur_pos = cache['seq_pos']
        ent_q_rope = self._apply_rope(ent_q, pos_offset=cur_pos)
        ent_k_rope = self._apply_rope(ent_k, pos_offset=cur_pos)
        
        # 跨层纠缠
        # cross_layer_q_proj: BitLinear(d_state, ent_rank) - 逐头处理
        if cross_layer_state is not None:
            # cross_layer_state: (B, n_heads, d_state) 或 (B, L, n_heads, d_state)
            if cross_layer_state.dim() == 3:
                # (B, n_heads, d_state) -> 逐头投影 -> (B, n_heads, ent_rank)
                cs_flat = cross_layer_state.reshape(batch * n_heads, self.d_state)
                cross_q_bias = self.cross_layer_q_proj(cs_flat)
                cross_q_bias = cross_q_bias.view(batch, n_heads, self.ent_rank)
            else:
                # (B, L, n_heads, d_state) -> 逐头投影 -> (B, L, n_heads, ent_rank)
                B2, L2, H2, D2 = cross_layer_state.shape
                cs_flat = cross_layer_state.reshape(B2 * L2 * H2, self.d_state)
                cross_q_bias = self.cross_layer_q_proj(cs_flat)
                cross_q_bias = cross_q_bias.view(B2, L2, H2, self.ent_rank)
            ent_q_rope = ent_q_rope + cross_q_bias.unsqueeze(1)
        
        # 分页缓存支持
        if cache.get('use_paged', False):
            paged_cache = cache['paged_cache']
            if cur_pos > 0 and cur_pos % paged_cache.page_size == 0:
                paged_cache.save_state(cur_pos, {
                    'state': cache['state'],
                    'ent_k_cache': cache['ent_k_cache'],
                    'ent_v_cache': cache['ent_v_cache'],
                })
        
        # Phase 1: 纯状态更新 (float64 精度)
        prev_state = cache['state']  # (batch, n_heads, d_state) float64
        # decay[:, 0]: (batch, n_heads, d_state), write[:, 0]: (batch, n_heads, d_state)
        # state_value[:, 0]: (batch, n_heads, d_state)
        new_state_f64 = decay[:, 0].double() * prev_state + write[:, 0].double() * state_value[:, 0].double()
        new_state = new_state_f64.to(decay.dtype)
        
        # Phase 2: 三值纠缠（使用循环缓冲区）
        window = cache['ent_k_cache'].shape[1]  # (batch, window, n_heads, rank)
        cache_idx = cache.get('cache_idx', 0)
        
        # 写入当前位置: k_vec (batch, n_heads, ent_rank), new_state (batch, n_heads, d_state)
        k_vec = ent_k_rope[:, 0, :, :]  # (batch, n_heads, ent_rank)
        cache['ent_k_cache'][:, cache_idx, :, :] = k_vec
        cache['ent_v_cache'][:, cache_idx, :, :] = new_state
        
        # 更新循环索引
        cache['cache_idx'] = (cache_idx + 1) % window
        
        # 计算有效长度
        seq_pos = cache['seq_pos']
        valid_len = min(seq_pos + 1, window)
        
        # 构建有效缓存索引
        if valid_len >= window:
            indices = torch.arange(cache_idx, cache_idx + window, device=cache['ent_k_cache'].device) % window
            k_eff = cache['ent_k_cache'][:, indices, :, :]  # (batch, window, n_heads, ent_rank)
            v_eff = cache['ent_v_cache'][:, indices, :, :]  # (batch, window, n_heads, d_state)
        else:
            k_eff = cache['ent_k_cache'][:, :valid_len, :, :]
            v_eff = cache['ent_v_cache'][:, :valid_len, :, :]
        
        # MIMO 纠缠计算: 逐头计算 scores
        q_vec = ent_q_rope[:, 0, :, :]  # (batch, n_heads, ent_rank)
        
        # scores: (batch, n_heads, valid_len)
        # q_vec: (batch, n_heads, R), k_eff: (batch, W, n_heads, R)
        # 使用 einsum: 'bhr,bwhr->bhw'
        scores = torch.einsum('bhr,bwhr->bhw', q_vec, k_eff) / (self.ent_rank ** 0.5)
        
        # 添加偏置
        if self.local_entanglement_bias is not None:
            bias = self.local_entanglement_bias[:, -valid_len:].to(dtype=scores.dtype, device=scores.device)
            # bias: (n_heads, valid_len) -> (1, n_heads, valid_len)
            scores = scores + bias.unsqueeze(0)
        
        # 如果缓存未满，补零到 window 大小
        if valid_len < window:
            full_scores = scores.new_zeros(batch, n_heads, window)
            full_scores[:, :, window - valid_len:] = scores
            scores = full_scores
        
        # 三值纠缠
        ternary = self.ternary_entanglement(scores)
        if valid_len < window:
            ternary[:, :, :window - valid_len] = 0.0
        
        # 归一化
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        
        # 加权求和: ternary (batch, n_heads, window), v_eff (batch, window, n_heads, d_state)
        # signed_avg: (batch, n_heads, d_state)
        if valid_len < window:
            full_v = v_eff.new_zeros(batch, window, n_heads, self.d_state)
            full_v[:, window - valid_len:] = v_eff
            # 'bhw,bwhd->bhd'
            signed_avg = torch.einsum('bhw,bwhd->bhd', ternary / norm, full_v)
        else:
            signed_avg = torch.einsum('bhw,bwhd->bhd', ternary / norm, v_eff)
        
        entangled_state = new_state + self.entanglement_scale * (signed_avg - new_state)
        ent_change = entangled_state - new_state
        
        cache['state'] = new_state_f64.detach()
        cache['seq_pos'] += 1
        
        # Phase 3: 输出
        # (batch, n_heads, d_state) -> (batch, n_heads * d_state)
        entangled_flat = entangled_state.reshape(batch, n_heads * self.d_state)
        ent_change_flat = ent_change.reshape(batch, n_heads * self.d_state)
        
        state_projected = self.state_proj(torch.tanh(entangled_flat))
        ent_projected = self.ent_proj(ent_change_flat)
        
        state_mixed = out_gate[:, 0] * state_projected
        ent_mixed = ent_projected
        y = local[:, 0] + state_mixed + ent_mixed
        y = self.out_proj(y.unsqueeze(1))
        
        # 跨层状态
        self._last_cross_layer_state = new_state.detach()
        return y, new_state_f64.detach()
    
    def forward(self, u, inference_params=None, cross_layer_state=None, prev_state=None, **kwargs):
        """MIMO 前向传播 - 多头并行 + MIMO rank 扩展
        
        继承 TESM_SISO 的:
        - 温度退火调度
        - ternary_entanglement
        - _apply_rope
        
        MIMO 核心机制 (参考 Mamba-3):
        1. V 投影到 MIMO 空间: (B,L,H,D) -> (B,L,R,H,D) via mimo_x
        2. Z 投影到 MIMO 空间: (B,L,H,D) -> (B,L,R,H,D) via mimo_z
        3. 在 MIMO 空间进行状态扫描和纠缠
        4. 合并输出: (B,L,R,H,D) -> (B,L,H,D) via mimo_o
        """
        batch, seqlen, dim = u.shape
        
        # 增量推理：如果是单 token 且有缓存，使用增量计算
        if inference_params is not None and seqlen == 1:
            layer_cache = inference_params.get('state_cache')
            if layer_cache is not None and 'state' in layer_cache:
                return self._forward_incremental(u, inference_params, cross_layer_state)
        
        # 输入投影
        proj = self.in_proj(u)
        
        # MIMO 多头解析 (与 SISO 顺序一致: local, state_value, decay, write, out_gate, ent_q, ent_k)
        chunks = [
            self.d_model,  # local
            self.d_state * self.n_heads,  # state_value
            self.d_state * self.n_heads,  # decay
            self.d_state * self.n_heads,  # write
            self.d_model,  # out_gate
            self.ent_rank * self.n_heads,  # ent_q
            self.ent_rank * self.n_heads,  # ent_k
        ]
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(proj, chunks, dim=-1)
        
        # 重塑为多头: (B, L, H, d_state)
        decay = decay.view(batch, seqlen, self.n_heads, self.d_state)
        write = write.view(batch, seqlen, self.n_heads, self.d_state)
        state_value = state_value.view(batch, seqlen, self.n_heads, self.d_state)
        ent_q = ent_q.view(batch, seqlen, self.n_heads, self.ent_rank)
        ent_k = ent_k.view(batch, seqlen, self.n_heads, self.ent_rank)
        
        # 应用 sigmoid 和 tanh
        decay = torch.sigmoid(decay + self.decay_bias.unsqueeze(0).unsqueeze(0))
        write = torch.sigmoid(write)
        state_value = torch.tanh(state_value)
        out_gate = torch.sigmoid(out_gate)
        
        # ===== MIMO 投影: V 投影到 MIMO 空间 =====
        B, L, H, D = state_value.shape
        R = self.mimo_rank
        
        # 投影到 MIMO 空间
        state_value_mimo = torch.einsum('blhd,hrd->blrhd', state_value.float(), self.mimo_x.float())
        
        # write gate 也需要投影到 MIMO 空间
        write_mimo = torch.einsum('blhd,hrd->blrhd', write.float(), self.mimo_x.float())
        
        # 计算 update in MIMO space
        update_mimo = write_mimo * state_value_mimo  # (B, L, R, H, D)
        
        # MIMO 状态扫描 - 逐 rank 扫描
        states_mimo = []
        final_states_mimo = []
        for r in range(R):
            decay_r = decay  # (B, L, H, D) - decay 共享
            update_r = update_mimo[:, :, r, :, :].contiguous()  # (B, L, H, D)
            states_r = self._parallel_state_scan_mimo(decay_r, update_r)
            states_mimo.append(states_r)  # (B, L, H, D)
            # 收集最终状态用于推理缓存
            final_states_mimo.append(states_r[:, -1, :, :])  # (B, H, D)
        
        # 堆叠为 (B, L, R, H, D)
        states_mimo = torch.stack(states_mimo, dim=2)  # (B, L, R, H, D)
        
        # 应用 RoPE
        ent_q = self._apply_rope(ent_q)
        ent_k = self._apply_rope(ent_k)
        
        # MIMO 纠缠 - 在每个 rank 上独立计算
        entangled_mimo = []
        ent_change_mimo = []
        for r in range(R):
            states_r = states_mimo[:, :, r, :, :].contiguous()  # (B, L, H, D)
            if self.entanglement_window > 0 and self.local_entanglement_bias is not None:
                signed_avg_r = self._compute_local_entanglement_mimo(
                    ent_q, ent_k, states_r, self.local_entanglement_bias
                )
                entangled_r = states_r + self.entanglement_scale * (signed_avg_r - states_r)
                ent_change_r = entangled_r - states_r
            else:
                entangled_r, ent_change_r = self._compute_global_entanglement_mimo(ent_q, ent_k, states_r)
            entangled_mimo.append(entangled_r.unsqueeze(2))
            ent_change_mimo.append(ent_change_r.unsqueeze(2))
        
        entangled_mimo = torch.cat(entangled_mimo, dim=2)  # (B, L, R, H, D)
        ent_change_mimo = torch.cat(ent_change_mimo, dim=2)  # (B, L, R, H, D)
        
        # ===== MIMO 输出合并 =====
        # 应用 mimo_o 合并 rank
        entangled = torch.einsum('blrhd,hrd->blhd', entangled_mimo.float(), self.mimo_o.float())
        ent_change = torch.einsum('blrhd,hrd->blhd', ent_change_mimo.float(), self.mimo_o.float())
        
        # 更新纠缠统计
        self._update_entanglement_stats_mimo(ent_q, ent_k)
        
        # prefill 时写入推理缓存
        if inference_params is not None:
            cache = inference_params.get('state_cache')
            if cache is not None and 'state' in cache:
                # 收集所有 rank 的最终状态合并为 (B, H, D)
                # 使用 rank-0 的最终状态作为缓存状态（简化，与 SISO 一致）
                cache['state'] = final_states_mimo[0].detach().to(torch.float64).clone()
                cache['seq_pos'] = seqlen
                window = cache['ent_k_cache'].shape[1]
                if seqlen >= window:
                    cache['ent_k_cache'] = ent_k[:, -window:, :, :].detach().float()
                    cache['ent_v_cache'] = states_mimo[:, -window:, 0, :, :].detach().float()  # rank-0 的状态
                    cache['cache_idx'] = 0
                else:
                    cache['ent_k_cache'][:, -seqlen:, :, :] = ent_k.detach().float()
                    cache['ent_v_cache'][:, -seqlen:, :, :] = states_mimo[:, -seqlen:, 0, :, :].detach().float()
                    cache['cache_idx'] = 0
        
        # 输出投影
        entangled_flat = entangled.reshape(batch, seqlen, self.n_heads * self.d_state).contiguous()
        ent_change_flat = ent_change.reshape(batch, seqlen, self.n_heads * self.d_state).contiguous()
        
        state_projected = self.state_proj(torch.tanh(entangled_flat))
        ent_projected = self.ent_proj(ent_change_flat)
        
        # 三部分输出
        state_mixed = out_gate * state_projected
        ent_mixed = ent_projected
        y = local + state_mixed + ent_mixed
        y = self.out_proj(self.dropout(y))
        
        # 跨层状态
        self._last_cross_layer_state = states_mimo[:, -1, 0, :, :].detach()  # (B, H, D) rank-0
        
        # 温度退火
        if self.training and self.annealing_enabled:
            self.annealing_step.add_(1)
        
        # 最终状态: 使用 rank-0 的最终扫描状态
        final_state = final_states_mimo[0].detach().to(torch.float64) if final_states_mimo else None
        return y, final_state


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("TESM_SISO-MIMO 优化版测试")
    print("="*70)
    
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"TileLang: {'可用' if TILELANG_AVAILABLE else '不可用'}")
    print(f"Triton: {'可用' if TRITON_AVAILABLE else '不可用'}")
    
    # 对比测试
    from tesm_ssm.modules.tesm import TESM_SISO
    
    d_model = 512
    
    # 原始 TESM_SISO
    tesm_base = TESM_SISO(d_model=d_model, d_state=256, expand=2, ent_rank=48).to(device)
    
    # 优化 MIMO
    mimo_opt = TESMMIMO_Optimized(d_model=d_model, d_state=256, n_heads=4, expand=2, ent_rank=48).to(device)
    
    print(f"\n参数量:")
    print(f"  原始 TESM_SISO: {sum(p.numel() for p in tesm_base.parameters())/1e6:.2f}M")
    print(f"  优化 MIMO: {sum(p.numel() for p in mimo_opt.parameters())/1e6:.2f}M")
    
    # 性能测试
    import time
    
    def measure(model, bs, seq, iters=20):
        model.eval()
        x = torch.randn(bs, seq, d_model, device=device)
        
        with torch.no_grad():
            # 预热
            for _ in range(3):
                _ = model(x)
            torch.cuda.synchronize()
            
            # 计时
            start = time.time()
            for _ in range(iters):
                _ = model(x)
            torch.cuda.synchronize()
            
        return (time.time() - start) / iters * 1000
    
    print(f"\n性能测试:")
    for seq in [128, 256, 512]:
        t_tesm = measure(tesm_base, 1, seq)
        t_mimo = measure(mimo_opt, 1, seq)
        print(f"  seq={seq}: TESM_SISO={t_tesm:.2f}ms, MIMO={t_mimo:.2f}ms, 加速={t_tesm/t_mimo:.1f}x")
    
    print("\n" + "="*70)
    print("测试完成")
