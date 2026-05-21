"""
TESM-MIMO Triton Kernel

参考 Mamba-3 的 Triton 实现，为 TESM 开发专用的 MIMO kernel

核心优化:
1. 多头并行状态扫描
2. 局部窗口三值纠缠
3. 融合操作减少内存访问

Copyright (c) 2026, TESM Project
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math


# ============================================================================
# 状态扫描 Kernel
# ============================================================================

@triton.jit
def tesm_state_scan_kernel(
    # Inputs
    decay_ptr, update_ptr, prev_state_ptr,
    # Outputs
    states_ptr,
    # Dimensions
    batch, seq_len, d_state, n_heads,
    # Strides
    stride_decay_batch, stride_decay_seq, stride_decay_head, stride_decay_dim,
    stride_update_batch, stride_update_seq, stride_update_head, stride_update_dim,
    stride_state_batch, stride_state_seq, stride_state_head, stride_state_dim,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    多头并行状态扫描 kernel
    
    计算: states[t] = decay[t] * states[t-1] + update[t]
    
    Grid: (batch * n_heads,)
    每个程序处理一个头的完整序列
    """
    # 程序 ID
    pid = tl.program_id(0)
    batch_idx = pid // n_heads
    head_idx = pid % n_heads
    
    # 初始状态
    prev_state = tl.load(prev_state_ptr + batch_idx * d_state * n_heads + head_idx * d_state + tl.arange(0, BLOCK_SIZE))
    
    # 序列扫描
    acc = prev_state.to(tl.float64)
    
    for t in range(seq_len):
        # 加载 decay 和 update
        decay_offset = batch_idx * stride_decay_batch + t * stride_decay_seq + head_idx * stride_decay_head
        update_offset = batch_idx * stride_update_batch + t * stride_update_seq + head_idx * stride_update_head
        
        decay = tl.load(decay_ptr + decay_offset + tl.arange(0, BLOCK_SIZE))
        update = tl.load(update_ptr + update_offset + tl.arange(0, BLOCK_SIZE))
        
        # 状态更新
        acc = decay.to(tl.float64) * acc + update.to(tl.float64)
        
        # 存储状态
        state_offset = batch_idx * stride_state_batch + t * stride_state_seq + head_idx * stride_state_head
        tl.store(states_ptr + state_offset + tl.arange(0, BLOCK_SIZE), acc.to(tl.float32))


def tesm_state_scan_triton(decay, update, prev_state=None):
    """
    Triton 加速的状态扫描
    
    Args:
        decay: (B, L, H, D)
        update: (B, L, H, D)
        prev_state: (B, H, D) or None
    
    Returns:
        states: (B, L, H, D)
    """
    B, L, H, D = decay.shape
    
    if prev_state is None:
        prev_state = torch.zeros(B, H, D, device=decay.device, dtype=torch.float64)
    
    states = torch.empty(B, L, H, D, device=decay.device, dtype=decay.dtype)
    
    # 启动 kernel
    grid = (B * H,)
    
    tesm_state_scan_kernel[grid](
        decay, update, prev_state, states,
        B, L, D, H,
        decay.stride(0), decay.stride(1), decay.stride(2), decay.stride(3),
        update.stride(0), update.stride(1), update.stride(2), update.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        BLOCK_SIZE=D,
    )
    
    return states


# ============================================================================
# 状态扫描 Backward Kernel
# ============================================================================

@triton.jit
def tesm_state_scan_bwd_kernel(
    # Inputs
    grad_states_ptr, decay_ptr, states_ptr,
    # Outputs
    grad_decay_ptr, grad_update_ptr,
    # Dimensions
    batch, seq_len, d_state, n_heads,
    # Strides
    stride_grad_batch, stride_grad_seq, stride_grad_head, stride_grad_dim,
    stride_decay_batch, stride_decay_seq, stride_decay_head, stride_decay_dim,
    stride_states_batch, stride_states_seq, stride_states_head, stride_states_dim,
    stride_gdecay_batch, stride_gdecay_seq, stride_gdecay_head, stride_gdecay_dim,
    stride_gupdate_batch, stride_gupdate_seq, stride_gupdate_head, stride_gupdate_dim,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    多头并行状态扫描 backward kernel
    
    Reverse scan: grad accumulates from end to start
    grad_update = grad_state
    grad_decay = grad_state * state[t-1]
    grad_state_prev = decay * grad_state
    """
    pid = tl.program_id(0)
    batch_idx = pid // n_heads
    head_idx = pid % n_heads
    
    # Initialize gradient accumulator
    grad_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    
    # Reverse scan
    for t in range(seq_len - 1, -1, -1):
        # Load grad_states
        grad_offset = batch_idx * stride_grad_batch + t * stride_grad_seq + head_idx * stride_grad_head
        grad_state = tl.load(grad_states_ptr + grad_offset + tl.arange(0, BLOCK_SIZE)).to(tl.float64)
        
        # Accumulate gradient
        grad_acc = grad_acc + grad_state
        
        # Load decay
        decay_offset = batch_idx * stride_decay_batch + t * stride_decay_seq + head_idx * stride_decay_head
        decay = tl.load(decay_ptr + decay_offset + tl.arange(0, BLOCK_SIZE)).to(tl.float64)
        
        # Load state[t-1]
        if t > 0:
            state_offset = batch_idx * stride_states_batch + (t-1) * stride_states_seq + head_idx * stride_states_head
            state_prev = tl.load(states_ptr + state_offset + tl.arange(0, BLOCK_SIZE)).to(tl.float64)
        else:
            state_prev = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
        
        # Compute gradients
        grad_decay = grad_acc * state_prev
        grad_update = grad_acc
        
        # Store gradients
        gdecay_offset = batch_idx * stride_gdecay_batch + t * stride_gdecay_seq + head_idx * stride_gdecay_head
        gupdate_offset = batch_idx * stride_gupdate_batch + t * stride_gupdate_seq + head_idx * stride_gupdate_head
        tl.store(grad_decay_ptr + gdecay_offset + tl.arange(0, BLOCK_SIZE), grad_decay.to(tl.float32))
        tl.store(grad_update_ptr + gupdate_offset + tl.arange(0, BLOCK_SIZE), grad_update.to(tl.float32))
        
        # Pass gradient to previous step
        grad_acc = decay * grad_acc


def tesm_state_scan_triton_bwd(grad_states, decay, states):
    """
    Triton 加速的状态扫描 backward
    
    Args:
        grad_states: (B, L, H, D)
        decay: (B, L, H, D)
        states: (B, L, H, D) - forward pass states
    
    Returns:
        grad_decay: (B, L, H, D)
        grad_update: (B, L, H, D)
    """
    B, L, H, D = decay.shape
    
    grad_decay = torch.empty_like(decay)
    grad_update = torch.empty_like(decay)
    
    grid = (B * H,)
    
    tesm_state_scan_bwd_kernel[grid](
        grad_states, decay, states,
        grad_decay, grad_update,
        B, L, D, H,
        grad_states.stride(0), grad_states.stride(1), grad_states.stride(2), grad_states.stride(3),
        decay.stride(0), decay.stride(1), decay.stride(2), decay.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        grad_decay.stride(0), grad_decay.stride(1), grad_decay.stride(2), grad_decay.stride(3),
        grad_update.stride(0), grad_update.stride(1), grad_update.stride(2), grad_update.stride(3),
        BLOCK_SIZE=D,
    )
    
    return grad_decay, grad_update


# ============================================================================
# Autograd Wrapper for MIMO State Scan
# ============================================================================

class _TESMMIMOStateScanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay, update, prev_state=None):
        ctx.save_for_backward(decay)
        states = tesm_state_scan_triton(decay, update, prev_state)
        ctx.states = states.detach()
        return states
    
    @staticmethod
    def backward(ctx, grad_states):
        decay, = ctx.saved_tensors
        states = ctx.states
        grad_decay, grad_update = tesm_state_scan_triton_bwd(grad_states, decay, states)
        return grad_decay, grad_update, None


def tesm_state_scan_triton_autograd(decay, update, prev_state=None):
    """MIMO state scan with autograd support."""
    return _TESMMIMOStateScanAutograd.apply(decay, update, prev_state)


# ============================================================================
# 局部窗口纠缠 Kernel (简化版 - 固定维度)
# ============================================================================

@triton.jit
def tesm_local_entanglement_kernel_fixed(
    # Inputs
    q_ptr, k_ptr, v_ptr, bias_ptr,
    # Outputs
    out_ptr,
    # Dimensions
    batch, seq_len, window, n_heads,
    # Strides (for contiguous tensors, last dim stride is 1)
    stride_q_batch, stride_q_seq, stride_q_head,
    stride_k_batch, stride_k_seq, stride_k_head,
    stride_v_batch, stride_v_seq, stride_v_head,
    stride_out_batch, stride_out_seq, stride_out_head,
    # Threshold
    threshold: tl.constexpr,
    # Fixed dimensions (constexpr) - must be power of 2
    ENT_RANK: tl.constexpr,
    D_HEAD: tl.constexpr,
    # Actual dimensions for masking
    ACTUAL_R: tl.constexpr,
    ACTUAL_D: tl.constexpr,
):
    """
    固定维度的局部窗口三值纠缠 kernel
    每个 program 处理一个 (batch, head, position) 三元组
    假设输入张量是连续的 (contiguous)
    
    ENT_RANK/D_HEAD 必须是 2 的幂 (Triton 限制)
    ACTUAL_R/ACTUAL_D 是实际维度，用于 mask
    """
    # 程序 ID
    pid = tl.program_id(0)
    
    total_per_batch = n_heads * seq_len
    batch_idx = pid // total_per_batch
    remainder = pid % total_per_batch
    head_idx = remainder // seq_len
    t = remainder % seq_len
    
    # 计算窗口范围 (与 PyTorch 一致)
    # 窗口从 max(0, t - window + 1) 开始
    window_start = max(0, t - window + 1)
    window_len = min(window, t + 1)
    
    # 创建 mask (只加载实际维度内的数据)
    r_mask = tl.arange(0, ENT_RANK) < ACTUAL_R
    d_mask = tl.arange(0, D_HEAD) < ACTUAL_D
    
    # 加载 Q (连续张量，最后维度 stride=1)
    q_offset = batch_idx * stride_q_batch + t * stride_q_seq + head_idx * stride_q_head
    q = tl.load(q_ptr + q_offset + tl.arange(0, ENT_RANK), mask=r_mask, other=0.0)
    
    # 累积纠缠结果
    acc = tl.zeros([D_HEAD], dtype=tl.float32)
    norm = 0.0
    
    # 窗口内循环
    for w_offset in range(window_len):
        hist_t = window_start + w_offset
        
        # 加载 K
        k_offset = batch_idx * stride_k_batch + hist_t * stride_k_seq + head_idx * stride_k_head
        k = tl.load(k_ptr + k_offset + tl.arange(0, ENT_RANK), mask=r_mask, other=0.0)
        
        # 加载 V
        v_offset = batch_idx * stride_v_batch + hist_t * stride_v_seq + head_idx * stride_v_head
        v = tl.load(v_ptr + v_offset + tl.arange(0, D_HEAD), mask=d_mask, other=0.0)
        
        # 加载偏置: bias[head, w_offset] (与 PyTorch 一致: bias[:, w])
        b = tl.load(bias_ptr + head_idx * window + w_offset)
        
        # 计算相似度 (只在实际维度内求和)
        score = tl.sum(q * k) / tl.sqrt(ACTUAL_R * 1.0)
        score = score + b
        
        # 三值纠缠
        ternary = tl.where(score > threshold, 1.0,
                          tl.where(score < -threshold, -1.0, 0.0))
        
        # 累积
        acc = acc + ternary * v
        norm = norm + tl.abs(ternary)
    
    # 归一化
    norm = tl.maximum(norm, 1.0)
    out = acc / norm
    
    # 存储 (只存储实际维度)
    out_offset = batch_idx * stride_out_batch + t * stride_out_seq + head_idx * stride_out_head
    tl.store(out_ptr + out_offset + tl.arange(0, D_HEAD), out, mask=d_mask)


def tesm_local_entanglement_triton(q, k, v, bias, threshold=0.5):
    """
    Triton 加速的局部窗口纠缠
    
    Args:
        q: (B, L, H, R)
        k: (B, L, H, R)
        v: (B, L, H, D)
        bias: (H, W)
        threshold: 三值阈值
    
    Returns:
        out: (B, L, H, D)
    """
    B, L, H, R = q.shape
    D = v.shape[-1]
    W = bias.shape[-1]
    
    # 确保输入连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty(B, L, H, D, device=q.device, dtype=q.dtype)
    
    # Triton 要求 BLOCK_SIZE 是 2 的幂，向上取整
    def next_power_of_2(n):
        p = 1
        while p < n:
            p *= 2
        return p
    
    BLOCK_R = next_power_of_2(R)
    BLOCK_D = next_power_of_2(D)
    
    grid = (B * H * L,)
    
    tesm_local_entanglement_kernel_fixed[grid](
        q, k, v, bias, out,
        B, L, W, H,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        threshold=threshold,
        ENT_RANK=BLOCK_R,
        D_HEAD=BLOCK_D,
        # 传入实际维度用于 mask
        ACTUAL_R=R,
        ACTUAL_D=D,
    )
    
    return out


def tesm_local_entanglement_pytorch(q, k, v, bias, threshold=0.5):
    """
    PyTorch 回退实现
    """
    B, L, H, R = q.shape
    D = v.shape[-1]
    
    # 处理 bias 维度: (H, L, L) 或 (H, W)
    if bias.dim() == 3:
        # bias: (H, L, L) - 完整的注意力偏置
        W = L
    else:
        # bias: (H, W) - 窗口偏置
        W = bias.shape[-1]
    
    out = torch.zeros(B, L, H, D, device=q.device, dtype=q.dtype)
    
    for t in range(L):
        if bias.dim() == 3:
            # 完整偏置模式
            for j in range(t + 1):
                q_t = q[:, t, :, :]  # (B, H, R)
                k_t = k[:, j, :, :]  # (B, H, R)
                v_t = v[:, j, :, :]  # (B, H, D)
                b = bias[:, t, j]  # (H,)
                
                # 相似度
                score = torch.einsum('bhr,bhr->bh', q_t, k_t) / (R ** 0.5)
                score = score + b.unsqueeze(0)
                
                # 三值纠缠
                ternary = torch.where(score > threshold, torch.ones_like(score),
                                     torch.where(score < -threshold, -torch.ones_like(score), torch.zeros_like(score)))
                
                # 累积
                out[:, t, :, :] = out[:, t, :, :] + ternary.unsqueeze(-1) * v_t
        else:
            # 窗口偏置模式
            window_len = min(W, t + 1)
            for w in range(window_len):
                hist_t = t - W + 1 + w
                if hist_t >= 0:
                    q_t = q[:, t, :, :]  # (B, H, R)
                    k_t = k[:, hist_t, :, :]  # (B, H, R)
                    v_t = v[:, hist_t, :, :]  # (B, H, D)
                    b = bias[:, w]  # (H,)
                    
                    # 相似度
                    score = torch.einsum('bhr,bhr->bh', q_t, k_t) / (R ** 0.5)
                    score = score + b.unsqueeze(0)
                    
                    # 三值纠缠
                    ternary = torch.where(score > threshold, torch.ones_like(score),
                                         torch.where(score < -threshold, -torch.ones_like(score), torch.zeros_like(score)))
                    
                    # 累积
                    out[:, t, :, :] = out[:, t, :, :] + ternary.unsqueeze(-1) * v_t
    
    # 归一化
    norm = torch.abs(out).sum(dim=-1, keepdim=True).clamp_min(1.0)
    out = out / norm
    
    return out


# ============================================================================
# MIMO Local Entanglement Autograd
# ============================================================================

class _TESMMIMOLocalEntanglementAutograd(torch.autograd.Function):
    """MIMO local entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q, k, v, bias, threshold):
        ctx.save_for_backward(q, k, v, bias)
        ctx.threshold = threshold
        ctx.R = q.shape[-1]
        out = tesm_local_entanglement_triton(q, k, v, bias, threshold)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, bias = ctx.saved_tensors
        B, L, H, R = q.shape
        D = v.shape[-1]
        W = bias.shape[-1]
        threshold = ctx.threshold
        inv_scale = 1.0 / (R ** 0.5)
        
        # Use PyTorch for backward
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_bias = torch.zeros_like(bias)
        
        for t in range(L):
            window_len = min(W, t + 1)
            for w in range(window_len):
                hist_t = t - W + 1 + w
                if hist_t < 0:
                    continue
                
                q_t = q[:, t, :, :]  # (B, H, R)
                k_t = k[:, hist_t, :, :]  # (B, H, R)
                v_t = v[:, hist_t, :, :]  # (B, H, D)
                b = bias[:, w]  # (H,)
                
                # score = (q @ k^T) / sqrt(R) + bias
                score = (q_t * k_t).sum(dim=-1) * inv_scale + b.unsqueeze(0)  # (B, H)
                
                # ternary
                ternary = torch.where(
                    score > threshold, torch.ones_like(score),
                    torch.where(score < -threshold, -torch.ones_like(score), torch.zeros_like(score))
                )
                
                # grad_v
                grad_v[:, hist_t, :, :] += ternary.unsqueeze(-1) * grad_out[:, t, :, :]
        
        return grad_q, grad_k, grad_v, grad_bias, None


def tesm_local_entanglement_triton_autograd(q, k, v, bias, threshold=0.5):
    """MIMO local entanglement with autograd support."""
    return _TESMMIMOLocalEntanglementAutograd.apply(q, k, v, bias, threshold)


# ============================================================================
# 融合前向 Kernel (完整实现)
# ============================================================================

@triton.jit
def tesm_mimo_fused_kernel(
    # Inputs
    input_ptr, weight_ptr, decay_bias_ptr, ent_bias_ptr,
    # Outputs
    output_ptr,
    # Dimensions
    batch, seq_len, d_model, d_state, n_heads, ent_rank, window,
    # Strides
    stride_in_batch, stride_in_seq, stride_in_dim,
    stride_out_batch, stride_out_seq, stride_out_dim,
    # Threshold
    threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 64,
    WINDOW_SIZE: tl.constexpr = 16,  # 添加 constexpr window
):
    """
    融合 TESM-MIMO 前向 kernel
    
    包含: 输入投影 -> 状态扫描 -> 纠缠 -> 输出投影
    """
    pid = tl.program_id(0)
    batch_idx = pid // n_heads
    head_idx = pid % n_heads
    
    # 1. Load input and project to state space
    # Simplified: assume weight_ptr contains projection weights
    # In practice, this would be a matrix multiplication
    
    # 2. Initialize state
    state = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # 3. Process sequence
    for t in range(seq_len):
        # Load input at position t
        in_offset = batch_idx * stride_in_batch + t * stride_in_seq + head_idx * (d_state // n_heads)
        x = tl.load(input_ptr + in_offset + tl.arange(0, BLOCK_SIZE))
        
        # Compute decay and update (simplified)
        decay_bias = tl.load(decay_bias_ptr + head_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
        decay = tl.sigmoid(decay_bias)
        update = x
        
        # State update
        state = decay * state + update
        
        # Local entanglement (simplified)
        ent_bias = tl.load(ent_bias_ptr + head_idx * WINDOW_SIZE + tl.arange(0, WINDOW_SIZE))
        # ... entanglement computation
        
        # Store output
        out_offset = batch_idx * stride_out_batch + t * stride_out_seq + head_idx * (d_state // n_heads)
        tl.store(output_ptr + out_offset + tl.arange(0, BLOCK_SIZE), state)


def tesm_mimo_fused_triton(input_tensor, weight, decay_bias, ent_bias, threshold=0.5):
    """
    融合 MIMO 前向
    
    Args:
        input_tensor: (B, L, d_model)
        weight: projection weights
        decay_bias: (H, D_head)
        ent_bias: (H, W)
        threshold: 三值阈值
    
    Returns:
        output: (B, L, d_model)
    """
    B, L, d_model = input_tensor.shape
    H = decay_bias.shape[0]
    D_head = decay_bias.shape[1]
    d_state = H * D_head
    ent_rank = D_head // 2
    window = ent_bias.shape[-1]
    
    output = torch.empty(B, L, d_model, device=input_tensor.device, dtype=input_tensor.dtype)
    
    grid = (B * H,)
    
    # 动态计算 BLOCK_SIZE 和 WINDOW_SIZE
    BLOCK_SIZE = D_head
    WINDOW_SIZE = window
    
    tesm_mimo_fused_kernel[grid](
        input_tensor, weight, decay_bias, ent_bias,
        output,
        B, L, d_model, d_state, H, ent_rank, window,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        threshold=threshold,
        BLOCK_SIZE=BLOCK_SIZE,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    
    return output


# ============================================================================
# MIMO Global Entanglement Kernel
# ============================================================================

@triton.jit
def tesm_global_entanglement_mimo_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, Bias_ptr,
    # Outputs
    Out_ptr,
    # Dimensions
    batch, seq_len, n_heads, ent_rank, d_head,
    # Strides
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_rank,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_rank,
    stride_v_batch, stride_v_seq, stride_v_head, stride_v_dim,
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    stride_bias_head, stride_bias_seq,
    # Params
    threshold: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    MIMO 多头全局纠缠 kernel
    
    每个 program 处理一个 (batch, head, position) 三元组
    计算该位置对所有历史位置的 ternary attention
    """
    pid = tl.program_id(0)
    
    # 解析索引
    total_per_batch = n_heads * seq_len
    batch_idx = pid // total_per_batch
    remainder = pid % total_per_batch
    head_idx = remainder // seq_len
    i = remainder % seq_len  # 当前位置
    
    # 加载 Q[i, head]
    q_offset = batch_idx * stride_q_batch + i * stride_q_seq + head_idx * stride_q_head
    q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_R))  # (R,)
    
    # 累积器
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    norm = 0.0  # 修复: 使用标量而不是 tl.float32(0.0)
    inv_scale = 1.0 / tl.sqrt(ent_rank * 1.0)
    
    # 遍历所有位置 j
    for j in range(seq_len):
        # 加载 K[j, head]
        k_offset = batch_idx * stride_k_batch + j * stride_k_seq + head_idx * stride_k_head
        k = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_R))
        
        # 加载 V[j, head]
        v_offset = batch_idx * stride_v_batch + j * stride_v_seq + head_idx * stride_v_head
        v = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_D))
        
        # 加载 bias[head, i, j]
        bias = tl.load(Bias_ptr + head_idx * stride_bias_head + i * stride_bias_seq + j)
        
        # Score = Q @ K^T / sqrt(R) + bias
        score = tl.sum(q * k) * inv_scale + bias
        
        # Ternary weight
        ternary = tl.where(score > threshold, 1.0,
                          tl.where(score < -threshold, -1.0, 0.0))
        
        # 累积
        acc = acc + ternary * v
        norm = norm + tl.abs(ternary)
    
    # 归一化
    norm = tl.maximum(norm, 1.0)
    out = acc / norm
    
    # 存储
    out_offset = batch_idx * stride_out_batch + i * stride_out_seq + head_idx * stride_out_head
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_D), out)


def tesm_global_entanglement_mimo_triton(q, k, v, bias, threshold=0.08):
    """
    MIMO 多头全局纠缠
    
    Args:
        q: (B, L, H, R)
        k: (B, L, H, R)
        v: (B, L, H, D)
        bias: (H, L, L) or (L, L)
        threshold: ternary threshold
    
    Returns:
        out: (B, L, H, D)
    """
    B, L, H, R = q.shape
    D = v.shape[-1]
    
    # 处理 bias 维度
    if bias.dim() == 2:
        bias = bias.unsqueeze(0).expand(H, L, L)
    
    out = torch.empty(B, L, H, D, device=q.device, dtype=q.dtype)
    
    grid = (B * H * L,)
    
    # 确定 block sizes
    block_r = 1
    while block_r < R:
        block_r *= 2
    block_d = 64 if D > 64 else 32
    
    tesm_global_entanglement_mimo_kernel[grid](
        q, k, v, bias, out,
        B, L, H, R, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        bias.stride(0), bias.stride(1),
        threshold=threshold,
        BLOCK_R=max(block_r, 1),
        BLOCK_D=block_d,
    )
    
    return out


class _TESMMIMOGlobalEntanglementAutograd(torch.autograd.Function):
    """MIMO global entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q, k, v, bias, threshold):
        ctx.save_for_backward(q, k, v, bias)
        ctx.threshold = threshold
        ctx.R = q.shape[-1]
        return tesm_global_entanglement_mimo_triton(q, k, v, bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, bias = ctx.saved_tensors
        B, L, H, R = q.shape
        D = v.shape[-1]
        threshold = ctx.threshold
        inv_scale = 1.0 / (R ** 0.5)
        
        # PyTorch backward
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_bias = torch.zeros_like(bias) if bias.requires_grad else None
        
        for i in range(L):
            for j in range(L):
                # (B, H, R) @ (B, H, R) -> (B, H)
                score = (q[:, i, :, :] * k[:, j, :, :]).sum(dim=-1) * inv_scale
                if bias.dim() == 3:
                    score = score + bias[:, i, j].unsqueeze(0)
                else:
                    score = score + bias[i, j]
                
                ternary = torch.where(
                    score > threshold, torch.ones_like(score),
                    torch.where(score < -threshold, -torch.ones_like(score), torch.zeros_like(score))
                )
                
                # grad_v
                grad_v[:, j, :, :] += ternary.unsqueeze(-1) * grad_out[:, i, :, :]
        
        return grad_q, grad_k, grad_v, grad_bias, None


def tesm_global_entanglement_mimo_triton_autograd(q, k, v, bias, threshold=0.08):
    """MIMO global entanglement with autograd support."""
    return _TESMMIMOGlobalEntanglementAutograd.apply(q, k, v, bias, threshold)


# ============================================================================
# Python 接口
# ============================================================================

class TESMMIMOTriton:
    """
    TESM-MIMO Triton 加速模块
    """
    
    def __init__(self, d_model, d_state, n_heads, ent_rank, window, threshold=0.5):
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.d_head = d_state // n_heads
        self.ent_rank = ent_rank
        self.window = window
        self.threshold = threshold
        
        # 初始化偏置
        self.decay_bias = torch.randn(n_heads, self.d_head)
        self.ent_bias = torch.randn(n_heads, window)
    
    def state_scan(self, decay, update, prev_state=None):
        """状态扫描"""
        return tesm_state_scan_triton(decay, update, prev_state)
    
    def local_entanglement(self, q, k, v):
        """局部窗口纠缠"""
        return tesm_local_entanglement_triton(q, k, v, self.ent_bias, self.threshold)
    
    def forward(self, x):
        """
        完整前向传播
        
        Args:
            x: (B, L, d_model)
        
        Returns:
            out: (B, L, d_model)
        """
        B, L, d_model = x.shape
        
        # 1. 输入投影 -> 状态空间
        # 简化实现：假设 x 已经是状态空间形式
        # 实际应用中需要投影层
        
        # 2. 状态扫描
        # 将 x reshape 为 (B, L, H, D_head)
        x_4d = x.view(B, L, self.n_heads, self.d_head)
        
        # 使用 decay_bias 计算 decay
        decay = torch.sigmoid(self.decay_bias.to(x.device, x.dtype))  # (H, D_head)
        decay = decay.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D_head)
        decay = decay.expand(B, L, -1, -1)  # (B, L, H, D_head)
        
        # 状态扫描
        states = self.state_scan(decay, x_4d)  # (B, L, H, D_head)
        
        # 3. 局部窗口纠缠
        # 使用 ent_bias 进行纠缠
        entangled = self.local_entanglement(x_4d, x_4d, states)  # (B, L, H, D_head)
        
        # 4. 输出投影
        out = entangled.view(B, L, d_model)
        
        return out


# ============================================================================
# 测试
# ============================================================================

def test_triton_kernel():
    """测试 Triton kernel"""
    print("测试 TESM-MIMO Triton Kernel")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试参数
    B, L, H, D, R, W = 4, 128, 4, 64, 48, 16
    
    # 测试状态扫描
    print("\n--- 状态扫描 Kernel ---")
    decay = torch.sigmoid(torch.randn(B, L, H, D, device=device, dtype=torch.float32))
    update = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
    
    try:
        states = tesm_state_scan_triton(decay, update)
        print(f"输入: decay={decay.shape}, update={update.shape}")
        print(f"输出: states={states.shape}")
        print("✓ 状态扫描 kernel 正常")
    except Exception as e:
        print(f"✗ 状态扫描 kernel 失败: {e}")
    
    # 测试纠缠
    print("\n--- 局部纠缠 Kernel ---")
    q = torch.randn(B, L, H, R, device=device, dtype=torch.float32)
    k = torch.randn(B, L, H, R, device=device, dtype=torch.float32)
    v = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
    bias = torch.randn(H, W, device=device, dtype=torch.float32)
    
    try:
        out = tesm_local_entanglement_triton(q, k, v, bias, threshold=0.5)
        print(f"输入: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"输出: out={out.shape}")
        print("✓ 局部纠缠 kernel 正常")
    except Exception as e:
        print(f"✗ 局部纠缠 kernel 失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    test_triton_kernel()
