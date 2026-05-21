"""调试三值隧穿模块"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tesm_ssm.modules.tesm import TernaryQuantumTunneling, TESM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 测试模块
print("\n=== 测试 TernaryQuantumTunneling 模块 ===")
tunnel_module = TernaryQuantumTunneling(
    threshold=0.1,
    tunneling_strength=0.1,
    num_tunnel_paths=4,
).to(device)

# 创建测试分数
scores = torch.randn(2, 10, 32, device=device) * 0.2  # 小分数，应该在边界区域
print(f"分数范围: [{scores.min().item():.3f}, {scores.max().item():.3f}]")

ternary, info = tunnel_module.apply_tunneling(scores, training=True)
print(f"隧穿率: {info['tunnel_rate']*100:.1f}%")
print(f"边界率: {info['boundary_rate']*100:.1f}%")
print(f"隧穿到+1: {tunnel_module.tunnel_to_positive.item()}")
print(f"隧穿到-1: {tunnel_module.tunnel_to_negative.item()}")

# 测试 TESM 模块
print("\n=== 测试 TESM 模块 ===")
tesm = TESM(
    d_model=256,
    d_state=256,
    ent_rank=32,
    max_seq_len=128,
    quantum_tunneling_enabled=True,
    tunneling_strength=0.1,
    num_tunnel_paths=4,
    device=device,
).to(device)

print(f"quantum_tunneling_enabled: {tesm.quantum_tunneling_enabled}")
print(f"quantum_tunneler: {tesm.quantum_tunneler}")

# 测试前向传播
x = torch.randn(2, 10, 256, device=device)
tesm.train()
y, state = tesm(x)
print(f"输出形状: {y.shape}")

# 检查隧穿统计
if tesm.quantum_tunneler is not None:
    print(f"隧穿到+1: {tesm.quantum_tunneler.tunnel_to_positive.item()}")
    print(f"隧穿到-1: {tesm.quantum_tunneler.tunnel_to_negative.item()}")
    print(f"边界总数: {tesm.quantum_tunneler.total_boundary.item()}")
