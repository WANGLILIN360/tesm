"""
BitLinear 融合 Kernel

优化: 将量化+矩阵乘法+反量化融合到一个 kernel
"""

import torch
import triton
import triton.language as tl


@triton.jit
def bitlinear_kernel(
    # 指针
    x_ptr, w_ptr, out_ptr,
    # 形状
    M, N, K,
    # Strides
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,
    stride_out_m, stride_out_n,
    # 量化参数
    eps: tl.constexpr,
    # Block size
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """融合 BitLinear kernel
    
    融合操作:
    1. 输入量化: x -> quantize(x)
    2. 权重量化: w -> quantize(w) to {-1, 0, +1}
    3. 矩阵乘法: out = x @ w.T
    4. 反量化: out = out / scale
    """
    
    # Block 索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算 block 起始位置
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 输入 scale (每个 row 一个)
    x_scale = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BLOCK_K):
        # 加载 x block
        x = tl.load(
            x_ptr + (rm + tl.arange(0, BLOCK_M)[:, None]) * stride_x_m +
            (k + tl.arange(0, BLOCK_K)[None, :]) * stride_x_k,
            mask=(rm + tl.arange(0, BLOCK_M)[:, None] < M) &
                 (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0
        )
        
        # 加载 w block (转置)
        w = tl.load(
            w_ptr + (rn + tl.arange(0, BLOCK_N)[:, None]) * stride_w_n +
            (k + tl.arange(0, BLOCK_K)[None, :]) * stride_w_k,
            mask=(rn + tl.arange(0, BLOCK_N)[:, None] < N) &
                 (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0
        )
        
        # 量化权重到 {-1, 0, +1}
        w_scale = 1.0 / (tl.abs(w).max() + eps)
        w_q = tl.where(w * w_scale > 0.5, 1.0,
                      tl.where(w * w_scale < -0.5, -1.0, 0.0))
        
        # 累加矩阵乘法
        acc += tl.dot(x, w_q.T)
        
        # 累加输入 scale
        x_scale = tl.maximum(x_scale, tl.abs(x).max(1, keepdim=True))
    
    # 反量化
    x_scale = 127.0 / (x_scale + eps)
    out = acc / x_scale
    
    # 存储
    tl.store(
        out_ptr + (rm + tl.arange(0, BLOCK_M)[:, None]) * stride_out_m +
        (rn + tl.arange(0, BLOCK_N)[None, :]) * stride_out_n,
        out,
        mask=(rm + tl.arange(0, BLOCK_M)[:, None] < M) &
             (rn + tl.arange(0, BLOCK_N)[None, :] < N)
    )


def bitlinear_fused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """融合 BitLinear 前向传播
    
    Args:
        x: (M, K) 输入
        weight: (N, K) 权重
        eps: 数值稳定性
    
    Returns:
        out: (M, N) 输出
    """
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    # 分配输出
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    # Block size
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # 启动 kernel
    bitlinear_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("BitLinear 融合 Kernel 测试")
    print("="*70)
    
    device = torch.device('cuda')
    
    # 测试尺寸
    M, N, K = 4096, 512, 512  # 典型 TESM 尺寸
    
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)
    
    import time
    
    # 原始实现
    def bitlinear_original(x, w, eps=1e-5):
        # 量化输入
        x_scale = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_min(eps)
        x_q = (x * x_scale).round().clamp(-128, 127) / x_scale
        x_q = x + (x_q - x).detach()
        
        # 量化权重
        w_scale = 1.0 / w.abs().mean().clamp_min(eps)
        w_q = (w * w_scale).round().clamp(-1, 1)
        w_q = w + (w_q - w).detach()
        w_q = w_q / w_scale
        
        # 矩阵乘法
        return torch.nn.functional.linear(x_q, w_q)
    
    # Warmup
    for _ in range(10):
        out_orig = bitlinear_original(x, w)
        out_fused = bitlinear_fused(x, w)
    
    torch.cuda.synchronize()
    
    # 测试原始
    start = time.time()
    for _ in range(100):
        out_orig = bitlinear_original(x, w)
    torch.cuda.synchronize()
    orig_ms = (time.time() - start) / 100 * 1000
    
    # 测试融合
    start = time.time()
    for _ in range(100):
        out_fused = bitlinear_fused(x, w)
    torch.cuda.synchronize()
    fused_ms = (time.time() - start) / 100 * 1000
    
    print(f"原始: {orig_ms:.3f} ms")
    print(f"融合: {fused_ms:.3f} ms")
    print(f"加速: {orig_ms/fused_ms:.2f}x")
    
    # 正确性
    diff = (out_orig - out_fused).abs().max().item()
    print(f"最大差异: {diff:.6f}")
    
    print("\n" + "="*70)
