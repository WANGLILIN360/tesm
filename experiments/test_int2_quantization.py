"""测试 INT2 量化推理性能"""
import sys
sys.path.insert(0, "d:/tesm-main-official-backup/tesm-main-official-backup")

import torch
import time

from tesm_ssm.modules.tesm import TESM, BitLinear
from tesm_ssm.utils.int2_quantization import (
    quantize_weight_to_int2,
    Int2Linear,
    export_model_to_int2,
    load_int2_weights_to_model,
)

device = torch.device("cuda")

print("=" * 60)
print("INT2 量化测试")
print("=" * 60)

# 创建模型
d_model = 512
d_state = 256
ent_rank = 48
window = 16

print("\n创建 TESM 模型...")
model = TESM(
    d_model=d_model,
    d_state=d_state,
    expand=2,
    ent_rank=ent_rank,
    entanglement_window=window,
    max_seq_len=2048,
    dropout=0.0,
    device=device,
)
model.eval()

# 计算原始模型大小
original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024*1024)  # FP32
print(f"原始模型大小 (FP32): {original_size:.2f} MB")

# 测试原始推理速度
print("\n--- 原始模型推理速度 ---")
batch_size = 1
seq_len = 128
x = torch.randn(batch_size, seq_len, d_model, device=device)

with torch.no_grad():
    # 预热
    _ = model(x)
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = model(x)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time:.2f} ms")

# 导出 INT2 权重
print("\n--- 导出 INT2 权重 ---")
exported = export_model_to_int2(model)

# 计算打包后大小
packed_size = sum(
    data['packed_weight'].numel() 
    for data in exported.values()
) / (1024*1024)  # uint8 = 1 byte
print(f"打包后模型大小 (INT2): {packed_size:.2f} MB")
print(f"压缩比: {original_size / packed_size:.1f}x")

# 创建 INT2 模型并测试
print("\n--- INT2 模型推理速度 ---")

# 创建新模型并加载 INT2 权重
model_int2 = TESM(
    d_model=d_model,
    d_state=d_state,
    expand=2,
    ent_rank=ent_rank,
    entanglement_window=window,
    max_seq_len=2048,
    dropout=0.0,
    device=device,
)
model_int2.eval()

# 替换 BitLinear 为 Int2Linear
load_int2_weights_to_model(model_int2, exported)

with torch.no_grad():
    # 预热
    _ = model_int2(x)
    torch.cuda.synchronize()
    
    # 测试
    times_int2 = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out_int2 = model_int2(x)
        torch.cuda.synchronize()
        times_int2.append((time.time() - start) * 1000)
    
    avg_time_int2 = sum(times_int2) / len(times_int2)
    print(f"平均推理时间: {avg_time_int2:.2f} ms")

# 对比
print("\n" + "=" * 60)
print("性能对比")
print("=" * 60)
print(f"模型大小: {original_size:.2f} MB → {packed_size:.2f} MB ({original_size/packed_size:.1f}x 压缩)")
print(f"推理速度: {avg_time:.2f} ms → {avg_time_int2:.2f} ms ({avg_time/avg_time_int2:.2f}x)")

# 验证输出一致性
print("\n--- 输出一致性验证 ---")
with torch.no_grad():
    out_orig = model(x)
    out_int2 = model_int2(x)
    if isinstance(out_orig, tuple):
        out_orig = out_orig[0]
    if isinstance(out_int2, tuple):
        out_int2 = out_int2[0]
    
    diff = (out_orig - out_int2).abs().mean().item()
    print(f"输出差异 (MAE): {diff:.6f}")
    
    # 相对误差
    rel_diff = diff / out_orig.abs().mean().item()
    print(f"相对误差: {rel_diff*100:.2f}%")
