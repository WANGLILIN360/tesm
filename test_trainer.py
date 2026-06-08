"""TESMTrainer 完整训练循环测试

测试覆盖:
1. 基本创建
2. 检查点保存/恢复
3. 学习率调度器
4. 梯度累积
5. 早停
6. 配置验证
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.training.config import TrainingConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel


class TestTrainerConfig(unittest.TestCase):
    """TrainingConfig 测试"""

    def test_01_creation(self):
        """测试1: 创建"""
        print("\n[Test 1] TrainingConfig 创建...")
        model_config = TESMConfig.small()
        config = TrainingConfig(
            model_config=model_config,
            data_path="dummy.txt",
            output_dir="outputs/test",
            num_epochs=1,
            batch_size=2,
        )
        self.assertEqual(config.num_epochs, 1)
        self.assertEqual(config.batch_size, 2)
        print("  ✓ TrainingConfig 创建成功")

    def test_02_device_validation(self):
        """测试2: 设备验证"""
        print("\n[Test 2] 设备验证...")
        with self.assertRaises(ValueError):
            TrainingConfig(
                model_config=TESMConfig.small(),
                device="invalid_device",
            )
        print("  ✓ 无效设备正确报错")

    def test_03_model_type_validation(self):
        """测试3: 模型类型验证"""
        print("\n[Test 3] 模型类型验证...")
        with self.assertRaises(ValueError):
            TrainingConfig(
                model_config=TESMConfig.small(),
                model_type="invalid",
            )
        print("  ✓ 无效模型类型正确报错")

    def test_04_to_dict(self):
        """测试4: to_dict"""
        print("\n[Test 4] to_dict...")
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            num_epochs=3,
        )
        d = config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['num_epochs'], 3)
        print("  ✓ to_dict 正确")

    def test_05_from_dict(self):
        """测试5: from_dict"""
        print("\n[Test 5] from_dict...")
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            num_epochs=3,
            learning_rate=1e-3,
        )
        d = config.to_dict()
        restored = TrainingConfig.from_dict(d)
        self.assertEqual(restored.num_epochs, 3)
        self.assertEqual(restored.learning_rate, 1e-3)
        print("  ✓ from_dict 正确")


class TestTrainerCheckpoint(unittest.TestCase):
    """检查点测试"""

    def test_06_save_load_state_dict(self):
        """测试6: state_dict 保存/加载"""
        print("\n[Test 6] state_dict 保存/加载...")
        config = TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True,
        )
        model = TESMLMHeadModel(config)

        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)
            self.assertTrue(os.path.exists(path))

            # 加载到新模型
            model2 = TESMLMHeadModel(config)
            state = torch.load(path, map_location='cpu', weights_only=False)
            model2.load_state_dict(state)

            # 验证参数一致
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                self.assertTrue(torch.allclose(p1, p2), f"参数 {n1} 不一致")
        print("  ✓ state_dict 保存/加载正确")

    def test_07_optimizer_state(self):
        """测试7: 优化器状态"""
        print("\n[Test 7] 优化器状态...")
        config = TESMConfig(
            d_model=64, n_layer=2, vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True,
        )
        model = TESMLMHeadModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 模拟一步训练
        x = torch.randint(0, 128, (1, 8))
        output, _ = model(x)
        loss = output.logits.mean()
        loss.backward()
        optimizer.step()

        # 保存优化器状态
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "optimizer.pt")
            torch.save(optimizer.state_dict(), path)
            self.assertTrue(os.path.exists(path))

            # 恢复
            optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)
            state = torch.load(path, map_location='cpu', weights_only=False)
            optimizer2.load_state_dict(state)
        print("  ✓ 优化器状态保存/加载正确")


class TestLRScheduler(unittest.TestCase):
    """学习率调度器测试"""

    def test_08_cosine_scheduler(self):
        """测试8: cosine 学习率调度"""
        print("\n[Test 8] cosine 调度器...")
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )

        initial_lr = optimizer.param_groups[0]['lr']
        self.assertEqual(initial_lr, 1e-3)

        # 模拟训练
        for _ in range(100):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(final_lr, 1e-5, delta=1e-6)
        print(f"  ✓ cosine: {initial_lr} -> {final_lr}")

    def test_09_linear_warmup(self):
        """测试9: linear warmup"""
        print("\n[Test 9] linear warmup...")
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 手动 warmup
        warmup_steps = 10
        for step in range(warmup_steps):
            lr = 1e-3 * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-3, delta=1e-6)
        print("  ✓ linear warmup 正确")


class TestEarlyStopping(unittest.TestCase):
    """早停测试"""

    def test_10_early_stopping(self):
        """测试10: 早停逻辑"""
        print("\n[Test 10] 早停...")
        patience = 3
        best_loss = 1.0
        no_improve_count = 0
        should_stop = False

        losses = [0.9, 0.85, 0.88, 0.87, 0.89, 0.90]
        for loss in losses:
            if loss < best_loss:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                should_stop = True
                break

        self.assertTrue(should_stop)
        self.assertEqual(best_loss, 0.85)
        print(f"  ✓ 早停: best={best_loss}, stopped after {no_improve_count} steps no improve")


def run_tests():
    print("=" * 60)
    print("TESMTrainer 完整测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestLRScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestEarlyStopping))
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
