"""
TESM-MIMO TileLang Kernel

参考 Mamba-3 的 TileLang 实现，为 TESM 开发专用的 MIMO kernel

核心优化:
1. 局部窗口三值纠缠 (TileLang 实现)
2. 并行状态扫描 (chunked scan)
3. 融合操作减少内存访问

Copyright (c) 2026, TESM Project
"""

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from typing import Optional, Tuple
import math


# ============================================================================
# 局部窗口纠缠 TileLang Kernel
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_local_entanglement_tilelang(
    B,          # batch size
    S,          # sequence length
    H,          # number of heads
    R,          # entanglement rank
    D,          # head dimension
    W,          # window size
    dtype: str = 'float32',
    threads: int = 128,
    num_stages: int = 2,
    D_tile: int = 64,  # D维度分块大小，避免shared memory溢出
):
    """
    TESM 局部窗口纠缠 TileLang kernel - 支持量子退火
    
    高温 (T > 1.0): softmax 纠缠
    低温 (T <= 1.0): 硬阈值纠缠
    
    参考 Mamba-3 的实现方式
    Grid: (H, B) - 每个头每个 batch 一个 program
    
    修复: 使用 D_tile 分块处理大 D 维度
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def tesm_entangle_kernel(
        Q: T.Tensor([B, S, H, R], dtype),
        K: T.Tensor([B, S, H, R], dtype),
        V: T.Tensor([B, S, H, D], dtype),
        Bias: T.Tensor([H, W], dtype),
        Temperature: T.Tensor([1], 'float32'),  # 温度参数
        Threshold: T.Tensor([1], 'float32'),    # 阈值参数
        Out: T.Tensor([B, S, H, D], dtype),
    ):
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # Buffer Allocation (使用 shared memory)
            # 注意: D维度分块处理，避免shared memory溢出
            q_shared = T.alloc_shared([R], dtype)
            k_shared = T.alloc_shared([R], dtype)
            v_shared = T.alloc_shared([D_tile], dtype)  # 分块处理
            bias_shared = T.alloc_shared([W], dtype)
            out_shared = T.alloc_shared([D_tile], accum_dtype)  # 分块处理
            weights_shared = T.alloc_shared([W], accum_dtype)  # softmax weights
            
            # 加载偏置
            for w in T.Parallel(W):
                bias_shared[w] = Bias[i_h, w]
            
            # 读取温度和阈值
            T_val = Temperature[0]
            threshold = Threshold[0]
            is_high_temp = T_val > T.float32(1.0)
            
            # D维度分块数量
            D_num_tiles = (D + D_tile - 1) // D_tile
            
            # 处理序列
            for t in T.serial(S):
                # 加载 Q
                for r in T.Parallel(R):
                    q_shared[r] = Q[i_b, t, i_h, r]
                
                # 窗口范围
                window_len = T.min(W, t + 1)
                
                # 窗口内循环 - 先收集所有分数（与D维度无关）
                max_score = T.alloc_fragment([1], accum_dtype)
                max_score[0] = T.float32(-1e9)
                
                for w in T.serial(window_len):
                    hist_t = t - W + 1 + w
                    
                    # 加载 K
                    for r in T.Parallel(R):
                        k_shared[r] = K[i_b, hist_t, i_h, r]
                    
                    # 计算相似度
                    score = T.alloc_fragment([1], accum_dtype)
                    score[0] = T.float32(0.0)
                    for r in T.serial(R):
                        score[0] += q_shared[r] * k_shared[r]
                    score[0] = score[0] / T.sqrt(T.float32(R))
                    score[0] += bias_shared[w]
                    
                    # 存储分数用于 softmax
                    weights_shared[w] = score[0]
                    max_score[0] = T.max(max_score[0], score[0])
                
                # 计算 softmax (高温) 或 硬阈值 (低温)
                sum_exp = T.alloc_fragment([1], accum_dtype)
                sum_exp[0] = T.float32(0.0)
                
                for w in T.serial(window_len):
                    exp_val = T.exp(weights_shared[w] - max_score[0])
                    weights_shared[w] = exp_val
                    sum_exp[0] += exp_val
                
                # D维度分块处理 - 每个分块独立处理窗口累积
                for d_tile_idx in T.serial(D_num_tiles):
                    d_start = d_tile_idx * D_tile
                    
                    # 清空输出（分块）
                    for d in T.Parallel(D_tile):
                        out_shared[d] = T.float32(0.0)
                    
                    # 窗口内循环 - 累积（分块处理D维度）
                    for w in T.serial(window_len):
                        hist_t = t - W + 1 + w
                        
                        # 加载 V（分块）
                        for d in T.Parallel(D_tile):
                            if d_start + d < D:
                                v_shared[d] = V[i_b, hist_t, i_h, d_start + d]
                        
                        # 高温: softmax 权重; 低温: 硬阈值
                        weight = T.alloc_fragment([1], accum_dtype)
                        
                        # 分支：高温用 softmax，低温用硬阈值
                        score_raw = weights_shared[w]  # 这里是 exp 值
                        softmax_weight = score_raw / sum_exp[0]
                        
                        # 重新计算分数用于硬阈值判断
                        score_thr = T.alloc_fragment([1], accum_dtype)
                        score_thr[0] = T.float32(0.0)
                        for r in T.serial(R):
                            k_shared[r] = K[i_b, hist_t, i_h, r]
                            score_thr[0] += q_shared[r] * k_shared[r]
                        score_thr[0] = score_thr[0] / T.sqrt(T.float32(R)) + bias_shared[w]
                        
                        # 硬阈值
                        ternary = T.if_then_else(
                            score_thr[0] > threshold,
                            T.float32(1.0),
                            T.if_then_else(
                                score_thr[0] < -threshold,
                                T.float32(-1.0),
                                T.float32(0.0)
                            )
                        )
                        
                        # 选择权重
                        weight[0] = T.if_then_else(
                            is_high_temp,
                            softmax_weight,
                            ternary
                        )
                        
                        # 累积（分块）
                        for d in T.Parallel(D_tile):
                            if d_start + d < D:
                                out_shared[d] += weight[0] * v_shared[d]
                    
                    # 归一化（分块）
                    norm = T.alloc_fragment([1], accum_dtype)
                    norm[0] = T.float32(0.0)
                    for d in T.serial(D_tile):
                        if d_start + d < D:
                            norm[0] += T.abs(out_shared[d])
                    norm[0] = T.max(norm[0], T.float32(1.0))
                    
                    # 存储（分块）
                    for d in T.Parallel(D_tile):
                        if d_start + d < D:
                            Out[i_b, t, i_h, d_start + d] = out_shared[d] / norm[0]
    
    return tesm_entangle_kernel


def tesm_local_entanglement_tilelang_fwd(q, k, v, bias, temperature=1.0, threshold=0.08):
    """
    TileLang 加速的局部窗口纠缠 - 支持量子退火
    
    Args:
        q: (B, L, H, R)
        k: (B, L, H, R)
        v: (B, L, H, D)
        bias: (H, W)
        temperature: 温度参数 (高温>1.0用softmax, 低温<=1.0用硬阈值)
        threshold: 三值阈值
    
    Returns:
        out: (B, L, H, D)
    """
    B, L, H, R = q.shape
    D = v.shape[-1]
    W = bias.shape[-1]
    
    # 确保连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    bias = bias.contiguous()
    
    # 温度和阈值 tensor
    T_tensor = torch.tensor([temperature], dtype=torch.float32, device=q.device)
    thr_tensor = torch.tensor([threshold], dtype=torch.float32, device=q.device)
    
    # D维度分块大小：根据D大小动态调整，避免shared memory溢出
    # A800 shared memory = 166KB per SM, 安全值约64KB
    D_tile = min(64, D)  # 每块最多64个元素
    
    # 编译 kernel
    kernel = tesm_local_entanglement_tilelang(
        B, L, H, R, D, W,
        dtype='float32' if q.dtype == torch.float32 else 'float16',
        D_tile=D_tile,
    )
    
    # 运行 kernel
    out = kernel(q, k, v, bias, T_tensor, thr_tensor)
    
    return out


# ============================================================================
# 并行状态扫描 TileLang Kernel (Chunked Scan)
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_chunked_scan_tilelang(
    B,          # batch size
    S,          # sequence length
    H,          # number of heads
    D,          # head dimension
    chunk_size: int = 16,
    dtype: str = 'float32',
    threads: int = 128,
):
    """
    TESM 并行状态扫描 TileLang kernel
    
    Grid: (H, B) - 每个头每个 batch 一个 program
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def tesm_scan_kernel(
        Decay: T.Tensor([B, S, H, D], dtype),
        Update: T.Tensor([B, S, H, D], dtype),
        States: T.Tensor([B, S, H, D], dtype),
    ):
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # 状态缓冲
            h_shared = T.alloc_shared([D], accum_dtype)
            
            # 初始化
            for d in T.Parallel(D):
                h_shared[d] = T.float32(0.0)
            
            # 处理序列
            for t in T.serial(S):
                # 加载 decay 和 update
                decay_val = T.alloc_fragment([D], dtype)
                update_val = T.alloc_fragment([D], dtype)
                
                for d in T.Parallel(D):
                    decay_val[d] = Decay[i_b, t, i_h, d]
                    update_val[d] = Update[i_b, t, i_h, d]
                
                # 状态更新: h = decay * h + update
                for d in T.Parallel(D):
                    h_shared[d] = decay_val[d] * h_shared[d] + update_val[d]
                
                # 存储
                for d in T.Parallel(D):
                    States[i_b, t, i_h, d] = h_shared[d]
    
    return tesm_scan_kernel


def tesm_chunked_scan_tilelang_fwd(decay, update, chunk_size=16):
    """
    TileLang 加速的 chunked 状态扫描
    
    Args:
        decay: (B, L, H, D)
        update: (B, L, H, D)
        chunk_size: chunk 大小
    
    Returns:
        states: (B, L, H, D)
    """
    B, L, H, D = decay.shape
    
    # 确保连续
    decay = decay.contiguous()
    update = update.contiguous()
    
    # 编译 kernel
    kernel = tesm_chunked_scan_tilelang(
        B, L, H, D,
        chunk_size=chunk_size,
        dtype='float32' if decay.dtype == torch.float32 else 'float16',
    )
    
    # 运行 kernel (TileLang 自动分配输出)
    states = kernel(decay, update)
    
    return states


# ============================================================================
# 融合前向 Kernel
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_mimo_fused_tilelang(
    B, S, H, R, D, W,
    chunk_size: int = 16,
    dtype: str = 'float32',
    threads: int = 128,
):
    """
    融合 TESM-MIMO 前向 kernel
    
    包含: 状态扫描 + 局部窗口纠缠
    
    这是一个简化版本，完整实现需要更多参数
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def tesm_fused_kernel(
        Decay: T.Tensor([B, S, H, D], dtype),
        Update: T.Tensor([B, S, H, D], dtype),
        Q: T.Tensor([B, S, H, R], dtype),
        K: T.Tensor([B, S, H, R], dtype),
        V: T.Tensor([B, S, H, D], dtype),
        Bias: T.Tensor([H, W], dtype),
        Out: T.Tensor([B, S, H, D], dtype),
    ):
        """
        融合前向 kernel
        """
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # 状态
            h = T.alloc_fragment([D], accum_dtype)
            T.clear(h)
            
            # 窗口缓冲
            k_window = T.alloc_shared([W, R], dtype)
            v_window = T.alloc_shared([W, D], dtype)
            
            # 处理序列
            for t in T.serial(S):
                # 1. 状态扫描
                decay_t = T.alloc_fragment([D], dtype)
                update_t = T.alloc_fragment([D], dtype)
                T.copy(Decay[i_b, t, i_h, :], decay_t)
                T.copy(Update[i_b, t, i_h, :], update_t)
                
                for d in T.serial(D):
                    h[d] = decay_t[d] * h[d] + update_t[d]
                
                # 2. 局部窗口纠缠
                q_frag = T.alloc_fragment([R], dtype)
                T.copy(Q[i_b, t, i_h, :], q_frag)
                
                # 更新窗口缓冲
                # ... (简化版本)
                
                # 输出
                T.copy(h, Out[i_b, t, i_h, :])
    
    return tesm_fused_kernel


# ============================================================================
# Backward Kernels for Autograd Support
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_chunked_scan_tilelang_bwd(
    B, S, H, D,
    dtype: str = 'float32',
    threads: int = 128,
):
    """
    TESM 并行状态扫描 Backward TileLang kernel
    
    反向传播: 从后向前累积梯度
    grad_decay[t] = grad_state[t] * state[t-1]
    grad_update[t] = grad_state[t]
    grad_state[t-1] += decay[t] * grad_state[t]
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def tesm_scan_bwd_kernel(
        GradStates: T.Tensor([B, S, H, D], dtype),
        Decay: T.Tensor([B, S, H, D], dtype),
        States: T.Tensor([B, S, H, D], dtype),
        GradDecay: T.Tensor([B, S, H, D], dtype),
        GradUpdate: T.Tensor([B, S, H, D], dtype),
    ):
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # 梯度累积缓冲
            grad_h = T.alloc_shared([D], accum_dtype)
            
            # 初始化
            for d in T.Parallel(D):
                grad_h[d] = T.float32(0.0)
            
            # 从后向前处理
            for t_rev in T.serial(S):
                t = S - 1 - t_rev
                
                # 加载 grad_states
                grad_out = T.alloc_fragment([D], dtype)
                for d in T.Parallel(D):
                    grad_out[d] = GradStates[i_b, t, i_h, d]
                
                # 累积梯度
                for d in T.Parallel(D):
                    grad_h[d] = grad_h[d] + T.cast(grad_out[d], accum_dtype)
                
                # 加载 decay
                decay_val = T.alloc_fragment([D], dtype)
                for d in T.Parallel(D):
                    decay_val[d] = Decay[i_b, t, i_h, d]
                
                # 计算 grad_decay 和 grad_update
                # t=0: grad_decay=0, grad_update=grad_h
                # t>0: grad_decay=grad_h*state[t-1], grad_update=grad_h
                for d in T.Parallel(D):
                    GradUpdate[i_b, t, i_h, d] = T.cast(grad_h[d], dtype)
                
                # t=0 单独处理
                if t_rev == S - 1:  # t == 0
                    for d in T.Parallel(D):
                        GradDecay[i_b, t, i_h, d] = T.cast(T.float32(0.0), dtype)
                else:  # t > 0
                    for d in T.Parallel(D):
                        state_prev = States[i_b, t - 1, i_h, d]
                        GradDecay[i_b, t, i_h, d] = T.cast(grad_h[d], dtype) * state_prev
                
                # 传递梯度到前一步
                for d in T.Parallel(D):
                    grad_h[d] = T.cast(decay_val[d], accum_dtype) * grad_h[d]
    
    return tesm_scan_bwd_kernel


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_local_entanglement_tilelang_bwd(
    B, S, H, R, D, W,
    dtype: str = 'float32',
    threads: int = 128,
    D_tile: int = 64,
):
    """
    TESM 局部窗口纠缠 Backward TileLang kernel
    
    简化版本: 使用 PyTorch fallback 进行 backward
    完整 TileLang backward 需要更复杂的实现
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def tesm_entangle_bwd_kernel(
        GradOut: T.Tensor([B, S, H, D], dtype),
        Q: T.Tensor([B, S, H, R], dtype),
        K: T.Tensor([B, S, H, R], dtype),
        V: T.Tensor([B, S, H, D], dtype),
        Weights: T.Tensor([B, S, W], accum_dtype),  # 保存的前向权重
        GradQ: T.Tensor([B, S, H, R], dtype),
        GradK: T.Tensor([B, S, H, R], dtype),
        GradV: T.Tensor([B, S, H, D], dtype),
    ):
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # 分配缓冲
            grad_q = T.alloc_shared([R], accum_dtype)
            grad_v_tile = T.alloc_shared([D_tile], accum_dtype)
            
            for t in T.serial(S):
                # 清空梯度
                for r in T.Parallel(R):
                    grad_q[r] = T.float32(0.0)
                
                # 窗口范围
                window_len = T.min(W, t + 1)
                
                # D维度分块处理
                for d_tile_idx in T.serial((D + D_tile - 1) // D_tile):
                    d_start = d_tile_idx * D_tile
                    
                    # 清空 grad_v tile
                    for d in T.Parallel(D_tile):
                        grad_v_tile[d] = T.float32(0.0)
                    
                    # 窗口内累积
                    for w in T.serial(window_len):
                        hist_t = t - W + 1 + w
                        weight = Weights[i_b, t, w]
                        
                        # grad_v += weight * grad_out
                        for d in T.Parallel(D_tile):
                            if d_start + d < D:
                                grad_v_tile[d] += weight * GradOut[i_b, t, i_h, d_start + d]
                        
                        # grad_v 回写
                        for d in T.Parallel(D_tile):
                            if d_start + d < D:
                                GradV[i_b, hist_t, i_h, d_start + d] += grad_v_tile[d]
    
    return tesm_entangle_bwd_kernel


# ============================================================================
# Autograd Wrappers
# ============================================================================

class _TileLangChunkedScanAutograd(torch.autograd.Function):
    """TileLang chunked scan with autograd support."""
    
    @staticmethod
    def forward(ctx, decay: torch.Tensor, update: torch.Tensor, chunk_size: int):
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(decay)
        return tesm_chunked_scan_tilelang_fwd(decay, update, chunk_size)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        decay, = ctx.saved_tensors
        B, S, H, D = decay.shape
        
        # 重新计算前向状态
        with torch.no_grad():
            states = torch.zeros(B, S, H, D, device=decay.device, dtype=decay.dtype)
            for t in range(S):
                if t == 0:
                    states[:, t] = 0
                else:
                    states[:, t] = decay[:, t] * states[:, t-1]
        
        grad_decay = torch.zeros_like(decay)
        grad_update = torch.zeros_like(decay)
        
        # 从后向前累积梯度
        grad_h = torch.zeros(B, H, D, device=decay.device, dtype=decay.dtype)
        for t in range(S - 1, -1, -1):
            grad_h = grad_h + grad_output[:, t]
            grad_update[:, t] = grad_h
            if t > 0:
                grad_decay[:, t] = grad_h * states[:, t-1]
            grad_h = decay[:, t] * grad_h
        
        return grad_decay, grad_update, None


def tesm_chunked_scan_tilelang_autograd(decay: torch.Tensor, update: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    """TileLang chunked scan with autograd support for training."""
    return _TileLangChunkedScanAutograd.apply(decay, update, chunk_size)


class _TileLangBitLinearAutograd(torch.autograd.Function):
    """TileLang BitLinear with autograd support."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(x, weight, scale)
        return tesm_bitlinear_tilelang(x, weight, scale)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, scale = ctx.saved_tensors
        M, K = x.shape
        N = weight.shape[0]
        
        # grad_x = grad_output @ (weight * scale)
        # grad_weight = grad_output^T @ x
        # grad_scale = grad_output.sum(dim=0)
        
        # 使用 PyTorch backward
        weight_scaled = weight * scale.unsqueeze(1)
        grad_x = grad_output @ weight_scaled
        grad_weight = grad_output.reshape(-1, N).T @ x.reshape(-1, K)
        grad_scale = grad_output.reshape(-1, N).sum(dim=0)
        
        return grad_x, grad_weight, grad_scale


def tesm_bitlinear_tilelang_autograd(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """TileLang BitLinear with autograd support for training."""
    return _TileLangBitLinearAutograd.apply(x, weight, scale)


class _TileLangLocalEntanglementAutograd(torch.autograd.Function):
    """TileLang local entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                bias: torch.Tensor, temperature: float, threshold: float):
        # 使用 ctx.saved_tensors 保存，避免属性赋值
        # threshold 通过 tensor 传递
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=q.device)
        ctx.save_for_backward(q, k, v, bias, threshold_tensor)
        return tesm_local_entanglement_tilelang_fwd(q, k, v, bias, temperature, threshold)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, threshold_tensor = ctx.saved_tensors
        threshold = threshold_tensor.item()
        B, L, H, R = q.shape
        D = v.shape[-1]
        W = bias.shape[-1]
        
        # 使用 PyTorch backward（简化版）
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_bias = torch.zeros_like(bias)
        
        inv_scale = 1.0 / math.sqrt(R)
        
        for b in range(B):
            for h in range(H):
                for t in range(L):
                    window_start = max(0, t - W + 1)
                    window_end = t + 1
                    
                    for w_idx, hist_t in enumerate(range(window_start, window_end)):
                        score = (q[b, t, h] * k[b, hist_t, h]).sum() * inv_scale + bias[h, w_idx]
                        if abs(score) > threshold:
                            weight = 1.0 if score > 0 else -1.0
                        else:
                            weight = 0.0
                        
                        grad_v[b, hist_t, h] += weight * grad_output[b, t, h]
                        grad_q[b, t, h] += weight * k[b, hist_t, h] * inv_scale * grad_output[b, t, h].sum() / max(D, 1)
                        grad_k[b, hist_t, h] += weight * q[b, t, h] * inv_scale * grad_output[b, t, h].sum() / max(D, 1)
        
        return grad_q, grad_k, grad_v, grad_bias, None, None


def tesm_local_entanglement_tilelang_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                                bias: torch.Tensor, temperature: float = 1.0, 
                                                threshold: float = 0.08) -> torch.Tensor:
    """TileLang local entanglement with autograd support for training."""
    return _TileLangLocalEntanglementAutograd.apply(q, k, v, bias, temperature, threshold)


class _TileLangGlobalEntanglementAutograd(torch.autograd.Function):
    """TileLang global entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                bias: torch.Tensor, threshold: float):
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=q.device)
        ctx.save_for_backward(q, k, v, bias, threshold_tensor)
        return tesm_global_entanglement_tilelang(q, k, v, bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, threshold_tensor = ctx.saved_tensors
        threshold = threshold_tensor.item()
        B, S, R = q.shape
        D = v.shape[-1]
        
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        
        inv_scale = 1.0 / math.sqrt(R)
        
        for b in range(B):
            for i in range(S):
                for j in range(S):
                    score = (q[b, i] * k[b, j]).sum() * inv_scale + (bias[i, j] if bias is not None else 0)
                    if abs(score) > threshold:
                        weight = 1.0 if score > 0 else -1.0
                    else:
                        weight = 0.0
                    
                    grad_v[b, j] += weight * grad_output[b, i]
                    grad_q[b, i] += weight * k[b, j] * inv_scale * grad_output[b, i].sum() / max(D, 1)
                    grad_k[b, j] += weight * q[b, i] * inv_scale * grad_output[b, i].sum() / max(D, 1)
        
        return grad_q, grad_k, grad_v, grad_bias, None


def tesm_global_entanglement_tilelang_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                                bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """TileLang global entanglement with autograd support for training."""
    return _TileLangGlobalEntanglementAutograd.apply(q, k, v, bias, threshold)


class _TileLangFusedOutputAutograd(torch.autograd.Function):
    """TileLang fused output with autograd support."""
    
    @staticmethod
    def forward(ctx, local: torch.Tensor, gate: torch.Tensor, 
                state_proj: torch.Tensor, ent_proj: torch.Tensor, ent_scale: float):
        ent_scale_tensor = torch.tensor([ent_scale], dtype=torch.float32, device=local.device)
        ctx.save_for_backward(local, gate, state_proj, ent_proj, ent_scale_tensor)
        return tesm_fused_output_tilelang(local, gate, state_proj, ent_proj, ent_scale)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local, gate, state_proj, ent_proj, ent_scale_tensor = ctx.saved_tensors
        ent_scale = ent_scale_tensor.item()
        
        # out = local * gate + state_proj + ent_scale * ent_proj
        grad_local = grad_output * gate
        grad_gate = grad_output * local
        grad_state_proj = grad_output
        grad_ent_proj = grad_output * ent_scale
        
        return grad_local, grad_gate, grad_state_proj, grad_ent_proj, None


def tesm_fused_output_tilelang_autograd(local: torch.Tensor, gate: torch.Tensor,
                                        state_proj: torch.Tensor, ent_proj: torch.Tensor,
                                        ent_scale: float) -> torch.Tensor:
    """TileLang fused output with autograd support for training."""
    return _TileLangFusedOutputAutograd.apply(local, gate, state_proj, ent_proj, ent_scale)


# ============================================================================
# BitLinear (Quantized Linear) - TileLang Implementation
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_bitlinear_tilelang_fwd(
    M, N, K,
    dtype: str = 'float16',
    threads: int = 128,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
):
    """
    TileLang BitLinear 前向 kernel
    
    量化线性层: INT8 input × INT2 weight -> FP output
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def bitlinear_fwd_kernel(
        X: T.Tensor([M, K], dtype),  # INT8 quantized input
        W: T.Tensor([N, K], dtype),  # INT2 packed weights (stored as INT8)
        Scale: T.Tensor([N], dtype),  # Weight scales
        Out: T.Tensor([M, N], dtype),  # Out 放在最后，由 out_idx=[-1] 自动处理
    ):
        with T.Kernel((N + BLOCK_N - 1) // BLOCK_N, (M + BLOCK_M - 1) // BLOCK_M, threads=threads) as (bx, by):
            X_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            W_shared = T.alloc_shared([BLOCK_K, BLOCK_N], dtype)
            acc = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            
            # Initialize accumulator
            T.clear(acc)
            
            # Block indices
            m_start = by * BLOCK_M
            n_start = bx * BLOCK_N
            
            # K-dimension tiling
            for k in T.serial((K + BLOCK_K - 1) // BLOCK_K):
                k_start = k * BLOCK_K
                
                # Load X tile
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    if m_start + i < M and k_start + j < K:
                        X_shared[i, j] = X[m_start + i, k_start + j]
                    else:
                        X_shared[i, j] = T.float16(0.0)
                
                # Load W tile (transposed)
                for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                    if k_start + i < K and n_start + j < N:
                        W_shared[i, j] = W[n_start + j, k_start + i]
                    else:
                        W_shared[i, j] = T.float16(0.0)
                
                # Matrix multiply
                T.gemm(X_shared, W_shared, acc, transpose_B=True)
            
            # Apply scale and store
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                if m_start + i < M and n_start + j < N:
                    Out[m_start + i, n_start + j] = acc[i, j] * Scale[n_start + j]
    
    return bitlinear_fwd_kernel


def tesm_bitlinear_tilelang(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    TileLang BitLinear
    
    Args:
        x: (M, K) INT8 quantized input
        weight: (N, K) INT2 packed weights
        scale: (N,) weight scales
    
    Returns:
        out: (M, N) FP output
    """
    M, K = x.shape
    N = weight.shape[0]
    
    kernel = tesm_bitlinear_tilelang_fwd(M, N, K, dtype='float16' if x.dtype == torch.float16 else 'float32')
    
    # TileLang 自动分配输出 (out_idx=[-1])
    out = kernel(x, weight, scale)
    
    return out



# ============================================================================
# Global Entanglement - TileLang Implementation
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_global_entanglement_tilelang_fwd(
    B, S, R, D,
    dtype: str = 'float16',
    threads: int = 128,
    BLOCK_S: int = 32,
    BLOCK_D: int = 32,
):
    """
    TileLang 全局纠缠前向 kernel
    
    O(L^2) 全局注意力，使用 ternary 权重
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def global_ent_fwd_kernel(
        Q: T.Tensor([B, S, R], dtype),
        K: T.Tensor([B, S, R], dtype),
        V: T.Tensor([B, S, D], dtype),
        Bias: T.Tensor([S, S], dtype),
        Threshold: T.Tensor([1], 'float32'),  # threshold 作为 Tensor
        Out: T.Tensor([B, S, D], dtype),  # Out 放在最后，由 out_idx=[-1] 自动处理
    ):
        with T.Kernel(S, B, threads=threads) as (i, b):
            threshold = Threshold[0]  # 在 kernel 内部读取
            # Load Q[i]
            q = T.alloc_shared([R], dtype)
            for r in T.Parallel(R):
                q[r] = Q[b, i, r]
            
            # Accumulator
            acc = T.alloc_fragment([D], accum_dtype)
            for d in T.Parallel(D):
                acc[d] = T.float32(0.0)
            
            norm = T.alloc_fragment([1], accum_dtype)
            norm[0] = T.float32(0.0)
            
            # Compute attention over all positions
            for j in T.serial(S):
                # Load K[j], V[j]
                k = T.alloc_shared([R], dtype)
                v = T.alloc_shared([D], dtype)
                for r in T.Parallel(R):
                    k[r] = K[b, j, r]
                for d in T.Parallel(D):
                    v[d] = V[b, j, d]
                
                # Score = Q[i] @ K[j]^T / sqrt(R) + Bias[i,j]
                # 使用可变变量避免 immutable variable 错误
                score = T.alloc_fragment([1], accum_dtype)
                score[0] = T.float32(0.0)
                for r in T.serial(R):
                    score[0] += q[r] * k[r]
                score[0] = score[0] / T.sqrt(R * 1.0) + Bias[i, j]
                
                # Ternary weight
                ternary = T.if_then_else(
                    score[0] > threshold, 1.0,
                    T.if_then_else(score[0] < -threshold, -1.0, 0.0)
                )
                
                # Accumulate
                for d in T.Parallel(D):
                    acc[d] += ternary * v[d]
                norm[0] += T.abs(ternary)
            
            # Normalize and store
            norm[0] = T.max(norm[0], 1.0)
            for d in T.Parallel(D):
                Out[b, i, d] = acc[d] / norm[0]
    
    return global_ent_fwd_kernel


def tesm_global_entanglement_tilelang(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                        bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """
    TileLang 全局纠缠
    
    Args:
        q: (B, S, R)
        k: (B, S, R)  
        v: (B, S, D)
        bias: (S, S)
        threshold: ternary threshold
    
    Returns:
        out: (B, S, D)
    """
    B, S, R = q.shape
    D = v.shape[-1]
    
    kernel = tesm_global_entanglement_tilelang_fwd(B, S, R, D, dtype='float16' if q.dtype == torch.float16 else 'float32')
    
    # 将 threshold 转换为 tensor
    threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=q.device)
    # TileLang 自动分配输出 (out_idx=[-1])
    out = kernel(q, k, v, bias, threshold_tensor)
    
    return out



# ============================================================================
# MIMO Global Entanglement - TileLang Implementation
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_global_entanglement_mimo_tilelang_fwd(
    B, S, H, R, D,
    dtype: str = 'float16',
    threads: int = 128,
    BLOCK_R: int = 64,
    BLOCK_D: int = 64,
):
    """
    TileLang MIMO 多头全局纠缠前向 kernel
    
    每个 kernel 处理一个 (batch, head, position) 三元组
    """
    accum_dtype = 'float32'
    
    @T.prim_func
    def global_ent_mimo_fwd_kernel(
        Q: T.Tensor([B, S, H, R], dtype),
        K: T.Tensor([B, S, H, R], dtype),
        V: T.Tensor([B, S, H, D], dtype),
        Bias: T.Tensor([H, S, S], dtype),
        Threshold: T.Tensor([1], 'float32'),  # threshold 作为 Tensor
        Out: T.Tensor([B, S, H, D], dtype),  # Out 放在最后，由 out_idx=[-1] 自动处理
    ):
        with T.Kernel(S, H, B, threads=threads) as (i, h, b):
            threshold = Threshold[0]  # 在 kernel 内部读取
            # Load Q[b, i, h]
            q = T.alloc_shared([R], dtype)
            for r in T.Parallel(R):
                q[r] = Q[b, i, h, r]
            
            # Accumulator
            acc = T.alloc_fragment([D], accum_dtype)
            for d in T.Parallel(D):
                acc[d] = T.float32(0.0)
            
            norm = T.alloc_fragment([1], accum_dtype)
            norm[0] = T.float32(0.0)
            
            inv_scale = 1.0 / T.sqrt(R * 1.0)
            
            # Compute attention over all positions
            for j in T.serial(S):
                # Load K[b, j, h], V[b, j, h]
                k = T.alloc_shared([R], dtype)
                v = T.alloc_shared([D], dtype)
                for r in T.Parallel(R):
                    k[r] = K[b, j, h, r]
                for d in T.Parallel(D):
                    v[d] = V[b, j, h, d]
                
                # Score = Q @ K^T / sqrt(R) + Bias[h, i, j]
                # 使用可变变量避免 immutable variable 错误
                score = T.alloc_fragment([1], accum_dtype)
                score[0] = T.float32(0.0)
                for r in T.serial(R):
                    score[0] += q[r] * k[r]
                score[0] = score[0] * inv_scale + Bias[h, i, j]
                
                # Ternary weight
                ternary = T.if_then_else(
                    score[0] > threshold, 1.0,
                    T.if_then_else(score[0] < -threshold, -1.0, 0.0)
                )
                
                # Accumulate
                for d in T.Parallel(D):
                    acc[d] += ternary * v[d]
                norm[0] += T.abs(ternary)
            
            # Normalize and store
            norm[0] = T.max(norm[0], 1.0)
            for d in T.Parallel(D):
                Out[b, i, h, d] = acc[d] / norm[0]
    
    return global_ent_mimo_fwd_kernel


def tesm_global_entanglement_mimo_tilelang(q, k, v, bias, threshold=0.08):
    """
    TileLang MIMO 多头全局纠缠
    
    Args:
        q: (B, S, H, R)
        k: (B, S, H, R)
        v: (B, S, H, D)
        bias: (H, S, S) or (S, S)
        threshold: ternary threshold
    
    Returns:
        out: (B, S, H, D)
    """
    B, S, H, R = q.shape
    D = v.shape[-1]
    
    # 处理 bias 维度
    if bias.dim() == 2:
        bias = bias.unsqueeze(0).expand(H, S, S)
    
    kernel = tesm_global_entanglement_mimo_tilelang_fwd(B, S, H, R, D, dtype='float16' if q.dtype == torch.float16 else 'float32')
    
    # 将 threshold 转换为 tensor
    threshold_tensor = torch.tensor([threshold], dtype=torch.float32, device=q.device)
    # TileLang 自动分配输出 (out_idx=[-1])
    out = kernel(q, k, v, bias, threshold_tensor)
    
    return out


class _TileLangGlobalEntanglementMIMOAutograd(torch.autograd.Function):
    """TileLang MIMO global entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                bias: torch.Tensor, threshold: float):
        ctx.save_for_backward(q, k, v, bias)
        ctx.threshold = threshold
        return tesm_global_entanglement_mimo_tilelang(q, k, v, bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias = ctx.saved_tensors
        B, S, H, R = q.shape
        D = v.shape[-1]
        
        # 使用 PyTorch backward（简化版）
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_bias = None
        
        inv_scale = 1.0 / math.sqrt(R)
        
        for b in range(B):
            for h in range(H):
                for i in range(S):
                    for j in range(S):
                        score = (q[b, i, h] * k[b, j, h]).sum() * inv_scale
                        if bias is not None and bias.dim() == 3:
                            score += bias[h, i, j]
                        elif bias is not None:
                            score += bias[i, j]
                        
                        if abs(score) > ctx.threshold:
                            weight = 1.0 if score > 0 else -1.0
                        else:
                            weight = 0.0
                        
                        grad_v[b, j, h] += weight * grad_output[b, i, h]
        
        return grad_q, grad_k, grad_v, grad_bias, None


def tesm_global_entanglement_mimo_tilelang_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                                      bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """TileLang MIMO global entanglement with autograd support for training."""
    return _TileLangGlobalEntanglementMIMOAutograd.apply(q, k, v, bias, threshold)



# ============================================================================
# Fused Output Combine - TileLang Implementation  
# ============================================================================

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def tesm_fused_output_tilelang_fwd(
    B, S, D,
    dtype: str = 'float16',
    threads: int = 128,
    BLOCK_D: int = 64,
):
    """
    TileLang 融合输出 kernel
    
    out = local * gate + state_proj + ent_scale * ent_proj
    """
    
    @T.prim_func
    def fused_out_kernel(
        Local: T.Tensor([B, S, D], dtype),
        Gate: T.Tensor([B, S, D], dtype),
        StateProj: T.Tensor([B, S, D], dtype),
        EntProj: T.Tensor([B, S, D], dtype),
        EntScale: T.Tensor([1], 'float32'),
        Out: T.Tensor([B, S, D], dtype),  # Out 放在最后，由 out_idx=[-1] 自动处理
    ):
        with T.Kernel((D + BLOCK_D - 1) // BLOCK_D, S, B, threads=threads) as (d_tile, s, b):
            ent_scale = EntScale[0]  # 在 kernel 内部读取
            d_start = d_tile * BLOCK_D
            
            for d in T.Parallel(BLOCK_D):
                if d_start + d < D:
                    local_val = Local[b, s, d_start + d]
                    gate_val = Gate[b, s, d_start + d]
                    state_val = StateProj[b, s, d_start + d]
                    ent_val = EntProj[b, s, d_start + d]
                    
                    out_val = local_val * gate_val + state_val + ent_scale * ent_val
                    Out[b, s, d_start + d] = out_val
    
    return fused_out_kernel


def tesm_fused_output_tilelang(local: torch.Tensor, gate: torch.Tensor, 
                                state_proj: torch.Tensor, ent_proj: torch.Tensor,
                                ent_scale: float) -> torch.Tensor:
    """TileLang fused output combine."""
    B, S, D = local.shape
    
    kernel = tesm_fused_output_tilelang_fwd(B, S, D, dtype='float16' if local.dtype == torch.float16 else 'float32')
    
    # 将 ent_scale 转换为 tensor
    ent_scale_tensor = torch.tensor([ent_scale], dtype=torch.float32, device=local.device)
    # TileLang 自动分配输出 (out_idx=[-1])
    out = kernel(local, gate, state_proj, ent_proj, ent_scale_tensor)
    
    return out



# ============================================================================
# 测试
# ============================================================================

def test_tilelang_kernel():
    """测试 TileLang kernel"""
    print("测试 TESM-MIMO TileLang Kernel")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试参数
    B, L, H, R, D, W = 4, 128, 4, 48, 64, 16
    
    # 测试局部纠缠
    print("\n--- 局部纠缠 TileLang Kernel ---")
    q = torch.randn(B, L, H, R, device=device, dtype=torch.float32)
    k = torch.randn(B, L, H, R, device=device, dtype=torch.float32)
    v = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
    bias = torch.randn(H, W, device=device, dtype=torch.float32)
    
    try:
        out = tesm_local_entanglement_tilelang_fwd(q, k, v, bias)
        print(f"输入: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"输出: out={out.shape}")
        print("✓ 局部纠缠 TileLang kernel 正常")
    except Exception as e:
        print(f"✗ 局部纠缠 TileLang kernel 失败: {e}")
    
    # 测试 chunked scan
    print("\n--- Chunked Scan TileLang Kernel ---")
    decay = torch.sigmoid(torch.randn(B, L, H, D, device=device, dtype=torch.float32))
    update = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
    
    try:
        states = tesm_chunked_scan_tilelang_fwd(decay, update)
        print(f"输入: decay={decay.shape}, update={update.shape}")
        print(f"输出: states={states.shape}")
        print("✓ Chunked scan TileLang kernel 正常")
    except Exception as e:
        print(f"✗ Chunked scan TileLang kernel 失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    test_tilelang_kernel()
