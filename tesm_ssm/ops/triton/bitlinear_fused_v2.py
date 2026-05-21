"""
BitLinear 融合 Kernel - 简化版

优化: 融合量化+矩阵乘法
"""

import torch
import triton
import triton.language as tl


@triton.jit
def bitlinear_fused_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """融合 BitLinear: 量化 + 矩阵乘法"""
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_max = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    # K 循环
    for k in range(0, K, BLOCK_K):
        # 加载 x
        offs_xm = rm + tl.arange(0, BLOCK_M)
        offs_xk = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_xm[:, None] * stride_xm + offs_xk[None, :] * stride_xk
        x_mask = (offs_xm[:, None] < M) & (offs_xk[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # 加载 w
        offs_wn = rn + tl.arange(0, BLOCK_N)
        w_ptrs = w_ptr + offs_wn[:, None] * stride_wn + offs_xk[None, :] * stride_wk
        w_mask = (offs_wn[:, None] < N) & (offs_xk[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # 量化权重
        w_abs_max = tl.max(tl.abs(w), axis=1, keep_dims=True) + eps
        w_scale = 1.0 / w_abs_max
        w_q = tl.where(w * w_scale > 0.5, 1.0,
                      tl.where(w * w_scale < -0.5, -1.0, 0.0))
        
        # 矩阵乘法
        acc += tl.dot(x, w_q.trans())
        
        # 累加 x max
        x_max = tl.maximum(x_max, tl.max(tl.abs(x), axis=1, keep_dims=True))
    
    # 反量化
    x_scale = 127.0 / (x_max + eps)
    out = acc / x_scale
    
    # 存储
    offs_om = rm + tl.arange(0, BLOCK_M)
    offs_on = rn + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def bitlinear_fused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """融合 BitLinear"""
    M, K = x.shape
    N = weight.shape[0]
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    bitlinear_fused_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out


if __name__ == "__main__":
    print("BitLinear 融合 Kernel 测试")
    print("="*70)
    
    device = torch.device('cuda')
    
    M, N, K = 1024, 512, 512
    x = torch.randn(M, K, device=device)
    w = torch.randn(N, K, device=device)
    
    import time
    
    # 原始实现 (与 BitLinear 一致)
    def bitlinear_orig(x, w, eps=1e-5):
        # 输入量化 (per-token)
        x_scale = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_min(eps)
        x_q = (x * x_scale).round().clamp(-128, 127) / x_scale
        x_q = x + (x_q - x).detach()  # STE
        
        # 权重量化 (per-tensor)
        w_scale = 1.0 / w.abs().mean().clamp_min(eps)
        w_q = (w * w_scale).round().clamp(-1, 1)
        w_q = w + (w_q - w).detach()  # STE
        w_q = w_q / w_scale
        
        return torch.nn.functional.linear(x_q, w_q)
    
    # Warmup
    for _ in range(10):
        out_orig = bitlinear_orig(x, w)
        out_fused = bitlinear_fused(x, w)
    
    torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    for _ in range(100):
        out_orig = bitlinear_orig(x, w)
    torch.cuda.synchronize()
    orig_ms = (time.time() - start) / 100 * 1000
    
    start = time.time()
    for _ in range(100):
        out_fused = bitlinear_fused(x, w)
    torch.cuda.synchronize()
    fused_ms = (time.time() - start) / 100 * 1000
    
    print(f"原始: {orig_ms:.3f} ms")
    print(f"融合: {fused_ms:.3f} ms")
    print(f"加速: {orig_ms/fused_ms:.2f}x")
    
    diff = (out_orig - out_fused).abs().max().item()
    print(f"差异: {diff:.6f}")
    
    print("="*70)
