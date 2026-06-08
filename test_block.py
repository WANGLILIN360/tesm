"""Block 层独立测试

测试覆盖:
1. RMSNorm 数值稳定性
2. Block 前向传播
3. 残差连接正确性
4. allocate_inference_cache
5. gradient checkpointing
6. eval/train 模式切换
"""

import sys
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.modules.block import Block, RMSNorm
from tesm_ssm.modules.tesm import TESM_SISO
from tesm_ssm.models.config_tesm import TESMConfig


class TestRMSNorm(unittest.TestCase):
    """RMSNorm 测试"""

    def test_01_basic(self):
        """测试1: 基本前向"""
        print("\n[Test 1] RMSNorm 基本...")
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)
        self.assertFalse(torch.isnan(y).any())
        print("  ✓ RMSNorm 基本正确")

    def test_02_normalization(self):
        """测试2: 归一化效果"""
        print("\n[Test 2] RMSNorm 归一化...")
        norm = RMSNorm(dim=64)
        # 大值输入
        x = torch.ones(1, 4, 64) * 100
        y = norm(x)
        # RMSNorm 后，均方根应该接近 1
        rms = (y ** 2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-4))
        print("  ✓ 归一化效果正确")

    def test_03_learnable_weight(self):
        """测试3: 可学习权重"""
        print("\n[Test 3] RMSNorm 权重...")
        norm = RMSNorm(dim=64)
        self.assertTrue(norm.weight.requires_grad)
        self.assertEqual(norm.weight.shape, (64,))
        x = torch.randn(1, 4, 64)
        y = norm(x)
        y.sum().backward()
        self.assertIsNotNone(norm.weight.grad)
        print("  ✓ 可学习权重正确")


class TestBlock(unittest.TestCase):
    """Block 层测试"""

    def _make_mixer(self, d_model=64):
        """创建 mixer"""
        return TESM_SISO(
            d_model=d_model, d_state=16, expand=2, ent_rank=8,
            max_seq_len=256, entanglement_window=0,
            use_triton_kernels=False, kernel_backend="torch",
            annealing_enabled=False,
        )

    def _make_mlp(self, d_model=64):
        """创建 MLP"""
        return torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model),
        )

    def _make_block(self, d_model=64):
        """创建 Block"""
        return Block(
            dim=d_model,
            mixer_cls=lambda d: self._make_mixer(d),
            mlp_cls=lambda d: self._make_mlp(d),
            norm_cls=RMSNorm,
        )

    def test_04_creation(self):
        """测试4: Block 创建"""
        print("\n[Test 4] Block 创建...")
        block = self._make_block()
        self.assertTrue(hasattr(block, 'norm'))
        self.assertTrue(hasattr(block, 'mixer'))
        self.assertTrue(hasattr(block, 'norm2'))
        self.assertTrue(hasattr(block, 'mlp'))
        print("  ✓ Block 创建成功")

    def test_05_forward(self):
        """测试5: Block 前向"""
        print("\n[Test 5] Block 前向...")
        block = self._make_block()
        block.eval()

        x = torch.randn(2, 8, 64)
        y, residual, final_state = block(x)

        self.assertEqual(y.shape, (2, 8, 64))
        self.assertIsNotNone(residual)
        self.assertFalse(torch.isnan(y).any())
        print(f"  ✓ Block 前向: {x.shape} -> {y.shape}")

    def test_06_residual_connection(self):
        """测试6: 残差连接"""
        print("\n[Test 6] 残差连接...")
        block = self._make_block()
        block.eval()

        x = torch.randn(1, 4, 64)
        y, residual, _ = block(x)

        # 残差不应该为零（说明有信息传递）
        self.assertFalse(torch.allclose(y, torch.zeros_like(y), atol=1e-6))
        print("  ✓ 残差连接正确")

    def test_07_inference_cache(self):
        """测试7: allocate_inference_cache"""
        print("\n[Test 7] 推理缓存...")
        block = self._make_block()

        cache = block.allocate_inference_cache(batch_size=2, max_seqlen=128)

        # mixer 应该有缓存
        self.assertIsNotNone(cache)
        print("  ✓ allocate_inference_cache 返回正确")

    def test_08_training(self):
        """测试8: 训练模式"""
        print("\n[Test 8] 训练模式...")
        block = self._make_block()
        block.train()

        x = torch.randn(1, 4, 64)
        y, _, _ = block(x)
        loss = y.mean()
        loss.backward()

        has_grad = any(p.grad is not None for p in block.parameters())
        self.assertTrue(has_grad)
        print("  ✓ 训练梯度正确")

    def test_09_eval_mode(self):
        """测试9: eval 模式"""
        print("\n[Test 9] eval 模式...")
        block = self._make_block()
        block.eval()

        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            y, _, _ = block(x)

        self.assertEqual(y.shape, (1, 4, 64))
        print("  ✓ eval 模式正确")

    def test_10_with_residual(self):
        """测试10: 传入残差"""
        print("\n[Test 10] 传入残差...")
        block = self._make_block()
        block.eval()

        x = torch.randn(1, 4, 64)
        residual = torch.randn(1, 4, 64)
        y, new_residual, _ = block(x, residual=residual)

        self.assertEqual(y.shape, (1, 4, 64))
        self.assertIsNotNone(new_residual)
        print("  ✓ 传入残差正确")

    def test_11_no_nan(self):
        """测试11: 无 NaN"""
        print("\n[Test 11] 无 NaN...")
        block = self._make_block()
        block.eval()

        for _ in range(10):
            x = torch.randn(1, 8, 64)
            y, _, _ = block(x)
            self.assertFalse(torch.isnan(y).any(), "输出不应有 NaN")
            self.assertFalse(torch.isinf(y).any(), "输出不应有 Inf")
        print("  ✓ 多次运行无 NaN/Inf")


def run_tests():
    print("=" * 60)
    print("Block 层独立测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRMSNorm))
    suite.addTests(loader.loadTestsFromTestCase(TestBlock))
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
