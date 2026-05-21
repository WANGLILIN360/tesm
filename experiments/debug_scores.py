"""调试TESM中scores的范围"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tesm_ssm.modules.tesm import TESM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 创建TESM
tesm = TESM(
    d_model=256,
    d_state=256,
    ent_rank=32,
    max_seq_len=128,
    entanglement_threshold=0.1,  # 阈值
    quantum_tunneling_enabled=True,
    tunneling_strength=0.1,
    num_tunnel_paths=4,
    device=device,
).to(device)

# Hook来捕获scores
captured_scores = []

def hook_fn(module, inp, out):
    # 在ternary_entanglement之前捕获scores
    pass

# 修改ternary_entanglement来打印scores范围
original_ternary = tesm.ternary_entanglement

def debug_ternary(scores):
    print(f"\n[ternary_entanglement] scores shape: {scores.shape}")
    print(f"  min: {scores.min().item():.4f}, max: {scores.max().item():.4f}")
    print(f"  mean: {scores.mean().item():.4f}, std: {scores.std().item():.4f}")
    print(f"  threshold: {tesm.entanglement_threshold}")
    
    # 统计边界区域
    boundary_mask = scores.abs() < tesm.entanglement_threshold * 1.5
    boundary_rate = boundary_mask.float().mean().item()
    print(f"  边界率: {boundary_rate*100:.1f}%")
    
    return original_ternary(scores)

tesm.ternary_entanglement = debug_ternary

# 测试前向传播
x = torch.randn(2, 16, 256, device=device)
tesm.train()
y, state = tesm(x)
print(f"\n输出形状: {y.shape}")

# 检查隧穿统计
if tesm.quantum_tunneler is not None:
    print(f"\n隧穿统计:")
    print(f"  隧穿到+1: {tesm.quantum_tunneler.tunnel_to_positive.item()}")
    print(f"  隧穿到-1: {tesm.quantum_tunneler.tunnel_to_negative.item()}")
    print(f"  边界总数: {tesm.quantum_tunneler.total_boundary.item()}")
