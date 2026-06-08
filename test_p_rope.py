"""PRoPE (Proportional RoPE) 完整测试

测试覆盖:
1. 基本前向传播
2. 与标准 RoPE 的数值对比
3. 频率剪枝正确性（active_dims < dim）
4. 长序列外推能力
5. positions 参数
6. 不同 prune_ratio
7. 不同输入维度
"""

import sys
import math
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.modules.multimodal.p_rope import PRoPE


class TestPRoPE(unittest.TestCase):
    """PRoPE 测试套件"""

    def test_01_basic(self):
        """测试1: 基本前向"""
        print("\n[Test 1] PRoPE 基本...")
        prope = PRoPE(dim=64, max_seq_len=128, prune_ratio=0.5)

        x = torch.randn(1, 8, 64)
        x_rotated = prope(x)

        self.assertEqual(x_rotated.shape, x.shape)
        self.assertFalse(torch.isnan(x_rotated).any())
        print(f"  ✓ PRoPE: {x.shape} -> {x_rotated.shape}")

    def test_02_vs_standard_rope(self):
        """测试2: 与标准 RoPE 对比 (prune_ratio=0 时应等价)"""
        print("\n[Test 2] vs 标准 RoPE...")
        # prune_ratio=0 时 p-RoPE 应等价于标准 RoPE
        prope_no_prune = PRoPE(dim=64, max_seq_len=128, prune_ratio=0.0)

        # 手动实现标准 RoPE
        def standard_rope(x, base=10000.0):
            B, L, D = x.shape
            half = D // 2
            pos = torch.arange(L, dtype=torch.float32)
            dim_idx = torch.arange(half, dtype=torch.float32)
            theta = pos.unsqueeze(1) * (1.0 / (base ** (2.0 * dim_idx / D)))
            cos_t = theta.cos().unsqueeze(0).to(x.dtype)
            sin_t = theta.sin().unsqueeze(0).to(x.dtype)
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)

        x = torch.randn(1, 8, 64)
        prope_out = prope_no_prune(x)
        std_out = standard_rope(x)

        self.assertTrue(torch.allclose(prope_out, std_out, atol=1e-5),
                       "prune_ratio=0 时应等价于标准 RoPE")
        print("  ✓ prune_ratio=0 等价于标准 RoPE")

    def test_03_prune_effect(self):
        """测试3: 剪枝效果（低频维度不应旋转）"""
        print("\n[Test 3] 剪枝效果...")
        prope = PRoPE(dim=64, max_seq_len=128, prune_ratio=0.5)

        # 创建特殊输入：前半部分=1，后半部分=0
        x = torch.zeros(1, 8, 64)
        x[:, :, :32] = 1.0  # 活跃维度
        x[:, :, 32:] = 0.5  # 剪枝维度

        x_rotated = prope(x)

        # 剪枝维度（后32维）应该保持不变
        pruned_part = x_rotated[:, :, 32:]
        original_pruned = x[:, :, 32:]
        self.assertTrue(torch.allclose(pruned_part, original_pruned, atol=1e-5),
                       "剪枝维度不应被旋转")

        # 活跃维度应该被旋转（不等于原值）
        active_part = x_rotated[:, :, :32]
        original_active = x[:, :, :32]
        self.assertFalse(torch.allclose(active_part, original_active, atol=1e-3),
                        "活跃维度应被旋转")
        print(f"  ✓ 剪枝正确: active_dims={prope.active_dims}, dim={prope.dim}")

    def test_04_active_dims(self):
        """测试4: active_dims 计算"""
        print("\n[Test 4] active_dims...")
        test_cases = [
            (64, 0.0, 64),   # 无剪枝
            (64, 0.5, 32),   # 剪50%
            (64, 0.25, 48),  # 剪25%
            (64, 0.75, 16),  # 剪75%
            (65, 0.5, 32),   # 奇数维度 → 偶数化
        ]

        for dim, prune_ratio, expected_active in test_cases:
            prope = PRoPE(dim=dim, max_seq_len=128, prune_ratio=prune_ratio)
            self.assertEqual(prope.active_dims, expected_active,
                           f"dim={dim}, prune={prune_ratio}: active_dims 应为 {expected_active}")
        print("  ✓ active_dims 计算正确")

    def test_05_long_sequence(self):
        """测试5: 长序列"""
        print("\n[Test 5] 长序列...")
        prope = PRoPE(dim=64, max_seq_len=4096, prune_ratio=0.5)

        x = torch.randn(1, 2048, 64)
        x_rotated = prope(x)

        self.assertEqual(x_rotated.shape, (1, 2048, 64))
        self.assertFalse(torch.isnan(x_rotated).any())
        print("  ✓ 长序列 (2048) 正确")

    def test_06_positions_parameter(self):
        """测试6: positions 参数"""
        print("\n[Test 6] positions 参数...")
        prope = PRoPE(dim=64, max_seq_len=128, prune_ratio=0.5)

        x = torch.randn(1, 4, 64)
        positions = torch.tensor([0, 5, 10, 15])

        x_rotated = prope(x, positions=positions)
        self.assertEqual(x_rotated.shape, x.shape)

        # 与标准位置对比，使用不同 positions 应产生不同结果
        x_rotated_std = prope(x, positions=torch.arange(4))
        self.assertFalse(torch.allclose(x_rotated, x_rotated_std, atol=1e-4))
        print("  ✓ positions 参数正确")

    def test_07_4d_input(self):
        """测试7: 4D 输入 (batch, heads, seq, dim)"""
        print("\n[Test 7] 4D 输入...")
        prope = PRoPE(dim=32, max_seq_len=128, prune_ratio=0.3)

        x = torch.randn(2, 8, 16, 32)  # (B, H, L, D)
        x_rotated = prope(x)

        self.assertEqual(x_rotated.shape, x.shape)
        self.assertFalse(torch.isnan(x_rotated).any())
        print(f"  ✓ 4D 输入: {x.shape} -> {x_rotated.shape}")

    def test_08_different_prune_ratios(self):
        """测试8: 不同剪枝比例"""
        print("\n[Test 8] 不同剪枝比例...")
        x = torch.randn(1, 8, 64)

        for prune_ratio in [0.0, 0.25, 0.5, 0.75]:
            prope = PRoPE(dim=64, max_seq_len=128, prune_ratio=prune_ratio)
            y = prope(x)
            self.assertEqual(y.shape, x.shape)
            self.assertFalse(torch.isnan(y).any())

        print("  ✓ 多种剪枝比例测试通过")

    def test_09_repr(self):
        """测试9: __repr__"""
        print("\n[Test 9] __repr__...")
        prope = PRoPE(dim=64, max_seq_len=2048, prune_ratio=0.5)
        r = repr(prope)
        self.assertIn("PRoPE", r)
        self.assertIn("64", r)
        self.assertIn("0.5", r)
        print(f"  ✓ {r}")

    def test_10_cached_values(self):
        """测试10: 缓存的 cos/sin 值"""
        print("\n[Test 10] 缓存值...")
        prope = PRoPE(dim=64, max_seq_len=128, prune_ratio=0.5)

        self.assertTrue(hasattr(prope, 'cos_cached'))
        self.assertTrue(hasattr(prope, 'sin_cached'))
        self.assertEqual(prope.cos_cached.shape, (128, 32))  # (max_seq_len, active_dims)
        self.assertEqual(prope.sin_cached.shape, (128, 32))
        print(f"  ✓ 缓存: cos={prope.cos_cached.shape}, sin={prope.sin_cached.shape}")


def run_tests():
    print("=" * 60)
    print("PRoPE 完整测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPRoPE)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"总测试数: {result.testsRun}")
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"通过: {passed}, 失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 60)
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
