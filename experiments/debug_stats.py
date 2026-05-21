"""调试隧穿统计更新"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tesm_ssm.modules.tesm import TernaryQuantumTunneling, TESM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 创建模块
tunnel_module = TernaryQuantumTunneling(
    threshold=0.1,
    tunneling_strength=0.1,
    num_tunnel_paths=4,
).to(device)

# 测试
scores = torch.randn(2, 16, 16, device=device) * 0.3
print(f"scores min: {scores.min().item():.4f}, max: {scores.max().item():.4f}")

# 调用apply_tunneling
print("\n调用 apply_tunneling(training=True):")
ternary, info = tunnel_module.apply_tunneling(scores, training=True)
print(f"  tunnel_rate: {info['tunnel_rate']*100:.1f}%")
print(f"  boundary_rate: {info['boundary_rate']*100:.1f}%")
print(f"  tunnel_to_positive: {tunnel_module.tunnel_to_positive.item()}")
print(f"  tunnel_to_negative: {tunnel_module.tunnel_to_negative.item()}")
print(f"  total_boundary: {tunnel_module.total_boundary.item()}")

# 再次调用
print("\n再次调用:")
ternary, info = tunnel_module.apply_tunneling(scores, training=True)
print(f"  tunnel_to_positive: {tunnel_module.tunnel_to_positive.item()}")
print(f"  tunnel_to_negative: {tunnel_module.tunnel_to_negative.item()}")
print(f"  total_boundary: {tunnel_module.total_boundary.item()}")

# 测试TESM
print("\n=== 测试 TESM ===")
tesm = TESM(
    d_model=256,
    d_state=256,
    ent_rank=32,
    max_seq_len=128,
    entanglement_threshold=0.1,
    quantum_tunneling_enabled=True,
    tunneling_strength=0.1,
    num_tunnel_paths=4,
    device=device,
).to(device)

tesm.train()
print(f"tesm.training: {tesm.training}")
print(f"tesm.quantum_tunneler: {tesm.quantum_tunneler}")

# 检查get_temperature
T = tesm.get_temperature()
print(f"Temperature: {T}")

# 手动调用ternary_entanglement
test_scores = torch.randn(2, 16, 16, device=device) * 0.3
print(f"\n手动调用 ternary_entanglement:")
result = tesm.ternary_entanglement(test_scores)
print(f"  结果形状: {result.shape}")
print(f"  tunnel_to_positive: {tesm.quantum_tunneler.tunnel_to_positive.item()}")
print(f"  tunnel_to_negative: {tesm.quantum_tunneler.tunnel_to_negative.item()}")
print(f"  total_boundary: {tesm.quantum_tunneler.total_boundary.item()}")

# 完整前向传播
x = torch.randn(2, 16, 256, device=device)
y, state = tesm(x)
print(f"\n完整前向传播后:")
print(f"  tunnel_to_positive: {tesm.quantum_tunneler.tunnel_to_positive.item()}")
print(f"  tunnel_to_negative: {tesm.quantum_tunneler.tunnel_to_negative.item()}")
print(f"  total_boundary: {tesm.quantum_tunneler.total_boundary.item()}")
