"""测试 INT2 CUDA kernel（JIT 编译）"""
import sys
sys.path.insert(0, "d:/tesm-main-official-backup/tesm-main-official-backup")

import torch
import time
from pathlib import Path

# JIT 编译 INT2 kernel
def compile_int2_kernel():
    from torch.utils.cpp_extension import load
    
    root = Path(__file__).resolve().parents[1]
    source_root = root / "csrc" / "tesm_ops"
    
    sources = [
        str(source_root / "tesm_ops.cpp"),
        str(source_root / "tesm_int2_linear.cu"),
    ]
    
    print("JIT 编译 INT2 kernel...")
    module = load(
        name="int2_ops",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    print("编译完成!")
    return module

# 尝试编译
try:
    int2_ops = compile_int2_kernel()
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA 编译失败: {e}")
    print("将使用 PyTorch fallback")
    CUDA_AVAILABLE = False

device = torch.device("cuda")

print("=" * 60)
print("INT2 量化性能测试")
print("=" * 60)

# 测试参数
M, K, N = 1, 512, 2048  # 典型推理尺寸

# 创建测试数据
x = torch.randn(M, K, device=device, dtype=torch.float32)
weight_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

# 打包权重
def pack_weight(weight):
    scale = 1.0 / weight.abs().mean().clamp_min(1e-8)
    normalized = weight * scale
    quantized = normalized.round().clamp(-1, 1).to(torch.int8)
    
    # 编码
    encoded = (quantized + 1).to(torch.uint8)
    K = weight.shape[1]
    
    # 打包
    encoded = encoded.reshape(N, K // 4, 4)
    packed = (
        encoded[:, :, 0] |
        (encoded[:, :, 1] << 2) |
        (encoded[:, :, 2] << 4) |
        (encoded[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return packed.contiguous(), scale

packed, scale = pack_weight(weight_fp32)
scale_tensor = torch.tensor([scale], device=device, dtype=torch.float32)

print(f"\n测试尺寸: M={M}, K={K}, N={N}")
print(f"FP32 权重大小: {weight_fp32.numel() * 4 / 1024:.2f} KB")
print(f"INT2 权重大小: {packed.numel() / 1024:.2f} KB")
print(f"压缩比: {weight_fp32.numel() * 4 / packed.numel():.1f}x")

# 测试 FP32 线性层
print("\n--- FP32 线性层 ---")
with torch.no_grad():
    # 预热
    _ = torch.nn.functional.linear(x, weight_fp32)
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        out_fp32 = torch.nn.functional.linear(x, weight_fp32)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_fp32 = sum(times) / len(times)
    print(f"平均时间: {avg_fp32:.3f} ms")

# 测试 INT2 kernel
if CUDA_AVAILABLE:
    print("\n--- INT2 CUDA kernel ---")
    with torch.no_grad():
        # 预热
        _ = int2_ops.int2_linear_fwd(x, packed, scale_tensor)
        torch.cuda.synchronize()
        
        # 测试
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            out_int2 = int2_ops.int2_linear_fwd(x, packed, scale_tensor)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
        
        avg_int2 = sum(times) / len(times)
        print(f"平均时间: {avg_int2:.3f} ms")
        print(f"加速比: {avg_fp32 / avg_int2:.2f}x")
        
        # 验证输出
        diff = (out_fp32 - out_int2).abs().mean().item()
        print(f"输出差异 (MAE): {diff:.6f}")

# 测试 PyTorch fallback
print("\n--- INT2 PyTorch fallback ---")
def int2_fallback(x, packed, scale):
    # 解包
    packed_uint8 = packed.to(torch.uint8)
    v0 = (packed_uint8 & 0x03).to(torch.int8) - 1
    v1 = ((packed_uint8 >> 2) & 0x03).to(torch.int8) - 1
    v2 = ((packed_uint8 >> 4) & 0x03).to(torch.int8) - 1
    v3 = ((packed_uint8 >> 6) & 0x03).to(torch.int8) - 1
    
    weight = torch.stack([v0, v1, v2, v3], dim=2).reshape(N, K).float() / scale
    return torch.nn.functional.linear(x, weight)

with torch.no_grad():
    # 预热
    _ = int2_fallback(x, packed, scale)
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        out_fallback = int2_fallback(x, packed, scale)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_fallback = sum(times) / len(times)
    print(f"平均时间: {avg_fallback:.3f} ms")
    print(f"加速比: {avg_fp32 / avg_fallback:.2f}x")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"模型大小压缩: {weight_fp32.numel() * 4 / packed.numel():.1f}x")
if CUDA_AVAILABLE:
    print(f"CUDA kernel 加速: {avg_fp32 / avg_int2:.2f}x")
print(f"PyTorch fallback 加速: {avg_fp32 / avg_fallback:.2f}x")
