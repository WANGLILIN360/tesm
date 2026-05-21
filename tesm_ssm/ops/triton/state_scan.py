"""Triton状态扫描kernel：借鉴Mamba的优化实现

优势：
1. 自动调优block size
2. 更易维护（vs CUDA）
3. 支持批处理不同长度序列
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['d_state'],
)
@triton.jit
def _state_scan_fwd_kernel(
    # 输入指针
    decay_ptr, update_ptr,
    # 输出指针
    states_ptr, final_state_ptr,
    # 维度
    batch, seqlen, d_state,
    # 步长
    stride_decay_batch, stride_decay_seqlen, stride_decay_state,
    stride_update_batch, stride_update_seqlen, stride_update_state,
    stride_states_batch, stride_states_seqlen, stride_states_state,
    # 元参数
    BLOCK_SIZE: tl.constexpr,
):
    """状态扫描前向kernel
    
    状态更新公式: state[t] = decay[t] * state[t-1] + update[t]
    """
    pid_b = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # state dim
    
    # 计算偏移
    offs_state = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_state = offs_state < d_state
    
    # 初始化状态
    state = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    
    # 遍历序列
    for t in range(seqlen):
        # 加载decay和update
        decay_offs = pid_b * stride_decay_batch + t * stride_decay_seqlen + offs_state * stride_decay_state
        update_offs = pid_b * stride_update_batch + t * stride_update_seqlen + offs_state * stride_update_state
        
        decay = tl.load(decay_ptr + decay_offs, mask=mask_state, other=0.0).to(tl.float64)
        update = tl.load(update_ptr + update_offs, mask=mask_state, other=0.0).to(tl.float64)
        
        # 更新状态
        state = decay * state + update
        
        # 存储状态
        state_offs = pid_b * stride_states_batch + t * stride_states_seqlen + offs_state * stride_states_state
        tl.store(states_ptr + state_offs, state, mask=mask_state)
    
    # 存储最终状态
    final_offs = pid_b * d_state + offs_state
    tl.store(final_state_ptr + final_offs, state, mask=mask_state)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['d_state'],
)
@triton.jit
def _state_scan_bwd_kernel(
    # 输入指针
    grad_states_ptr, decay_ptr, states_ptr,
    # 输出指针
    grad_decay_ptr, grad_update_ptr,
    # 维度
    batch, seqlen, d_state,
    # 步长
    stride_grad_states_batch, stride_grad_states_seqlen, stride_grad_states_state,
    stride_decay_batch, stride_decay_seqlen, stride_decay_state,
    stride_states_batch, stride_states_seqlen, stride_states_state,
    stride_grad_decay_batch, stride_grad_decay_seqlen, stride_grad_decay_state,
    stride_grad_update_batch, stride_grad_update_seqlen, stride_grad_update_state,
    # 元参数
    BLOCK_SIZE: tl.constexpr,
):
    """状态扫描反向kernel
    
    反向传播: 从后向前累积梯度
    """
    pid_b = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # state dim
    
    offs_state = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_state = offs_state < d_state
    
    # 初始化梯度累积
    grad_state = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    
    # 从后向前遍历
    for t in range(seqlen - 1, -1, -1):
        # 加载梯度
        grad_offs = pid_b * stride_grad_states_batch + t * stride_grad_states_seqlen + offs_state * stride_grad_states_state
        grad_out = tl.load(grad_states_ptr + grad_offs, mask=mask_state, other=0.0).to(tl.float64)
        
        # 累积梯度
        grad_state = grad_state + grad_out
        
        # 加载decay和state
        decay_offs = pid_b * stride_decay_batch + t * stride_decay_seqlen + offs_state * stride_decay_state
        state_offs = pid_b * stride_states_batch + t * stride_states_seqlen + offs_state * stride_states_state
        
        decay = tl.load(decay_ptr + decay_offs, mask=mask_state, other=0.0).to(tl.float64)
        state = tl.load(states_ptr + state_offs, mask=mask_state, other=0.0).to(tl.float64)
        
        # 计算梯度
        grad_decay = grad_state * state
        grad_update = grad_state
        
        # 存储梯度
        grad_decay_offs = pid_b * stride_grad_decay_batch + t * stride_grad_decay_seqlen + offs_state * stride_grad_decay_state
        grad_update_offs = pid_b * stride_grad_update_batch + t * stride_grad_update_seqlen + offs_state * stride_grad_update_state
        
        tl.store(grad_decay_ptr + grad_decay_offs, grad_decay, mask=mask_state)
        tl.store(grad_update_ptr + grad_update_offs, grad_update, mask=mask_state)
        
        # 传递梯度到前一步
        grad_state = decay * grad_state


def triton_state_scan_forward(decay: torch.Tensor, update: torch.Tensor) -> tuple:
    """Triton状态扫描前向
    
    Args:
        decay: (batch, seqlen, d_state) float32
        update: (batch, seqlen, d_state) float32
        
    Returns:
        states: (batch, seqlen, d_state) float64
        final_state: (batch, d_state) float64
    """
    batch, seqlen, d_state = decay.shape
    
    # 分配输出
    states = torch.empty(batch, seqlen, d_state, dtype=torch.float64, device=decay.device)
    final_state = torch.empty(batch, d_state, dtype=torch.float64, device=decay.device)
    
    # 计算grid
    grid = lambda META: (
        batch,
        triton.cdiv(d_state, META['BLOCK_SIZE']),
    )
    
    # 启动kernel
    _state_scan_fwd_kernel[grid](
        decay, update,
        states, final_state,
        batch, seqlen, d_state,
        decay.stride(0), decay.stride(1), decay.stride(2),
        update.stride(0), update.stride(1), update.stride(2),
        states.stride(0), states.stride(1), states.stride(2),
    )
    
    return states, final_state


class TritonStateScanFunction(torch.autograd.Function):
    """Triton状态扫描自动求导函数"""
    
    @staticmethod
    def forward(ctx, decay, update):
        states, final_state = triton_state_scan_forward(decay, update)
        ctx.save_for_backward(decay, states)
        return states, final_state
    
    @staticmethod
    def backward(ctx, grad_states, grad_final_state):
        decay, states = ctx.saved_tensors
        batch, seqlen, d_state = decay.shape
        
        # 分配梯度输出
        grad_decay = torch.empty_like(decay, dtype=torch.float64)
        grad_update = torch.empty_like(decay, dtype=torch.float64)
        
        # 添加最终状态梯度
        if grad_final_state is not None:
            grad_states = grad_states.clone()
            grad_states[:, -1, :] += grad_final_state
        
        # 计算grid
        grid = lambda META: (
            batch,
            triton.cdiv(d_state, META['BLOCK_SIZE']),
        )
        
        # 启动反向kernel
        _state_scan_bwd_kernel[grid](
            grad_states, decay, states,
            grad_decay, grad_update,
            batch, seqlen, d_state,
            grad_states.stride(0), grad_states.stride(1), grad_states.stride(2),
            decay.stride(0), decay.stride(1), decay.stride(2),
            states.stride(0), states.stride(1), states.stride(2),
            grad_decay.stride(0), grad_decay.stride(1), grad_decay.stride(2),
            grad_update.stride(0), grad_update.stride(1), grad_update.stride(2),
        )
        
        return grad_decay.to(decay.dtype), grad_update.to(decay.dtype)


def triton_state_scan(decay: torch.Tensor, update: torch.Tensor) -> tuple:
    """Triton状态扫描（带梯度）
    
    Args:
        decay: (batch, seqlen, d_state)
        update: (batch, seqlen, d_state)
        
    Returns:
        states: (batch, seqlen, d_state)
        final_state: (batch, d_state)
    """
    return TritonStateScanFunction.apply(decay, update)
