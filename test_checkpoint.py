"""检查点序列化/反序列化完整测试

测试覆盖:
1. TESMConfig to_dict / from_dict 往返
2. MultimodalConfig to_dict / from_dict 往返
3. TrainingConfig to_dict / from_dict 往返
4. 模型权重保存/加载一致性
5. 优化器状态保存/加载
6. 完整训练状态（模型+优化器+调度器）保存/恢复
7. 多模态模型保存/恢复
"""

import sys
import os
import tempfile
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.training.config import TrainingConfig


class TestConfigSerialization(unittest.TestCase):
    """配置序列化测试"""

    def test_01_tesm_config_roundtrip(self):
        """测试1: TESMConfig 往返"""
        print("\n[Test 1] TESMConfig 往返...")
        config = TESMConfig.small()
        d = config.to_dict()
        restored = TESMConfig.from_dict(d)

        self.assertEqual(config.d_model, restored.d_model)
        self.assertEqual(config.n_layer, restored.n_layer)
        self.assertEqual(config.vocab_size, restored.vocab_size)
        self.assertEqual(config.max_seq_len, restored.max_seq_len)
        print("  ✓ TESMConfig 往返正确")

    def test_02_multimodal_config_roundtrip(self):
        """测试2: MultimodalConfig 往返"""
        print("\n[Test 2] MultimodalConfig 往返...")
        mm_config = MultimodalConfig.from_tesm_config(
            TESMConfig.small(),
            vision_enabled=True,
            audio_enabled=False,
        )
        d = mm_config.to_dict()
        restored = MultimodalConfig.from_dict(d)

        self.assertEqual(mm_config.vision_enabled, restored.vision_enabled)
        self.assertEqual(mm_config.audio_enabled, restored.audio_enabled)
        self.assertEqual(mm_config.tesm.d_model, restored.tesm.d_model)
        print("  ✓ MultimodalConfig 往返正确")

    def test_03_training_config_roundtrip(self):
        """测试3: TrainingConfig 往返"""
        print("\n[Test 3] TrainingConfig 往返...")
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            num_epochs=5,
            learning_rate=2e-4,
            batch_size=8,
        )
        d = config.to_dict()
        restored = TrainingConfig.from_dict(d)

        self.assertEqual(config.num_epochs, restored.num_epochs)
        self.assertEqual(config.learning_rate, restored.learning_rate)
        self.assertEqual(config.batch_size, restored.batch_size)
        self.assertEqual(config.model_config.d_model, restored.model_config.d_model)
        print("  ✓ TrainingConfig 往返正确")

    def test_04_config_with_none(self):
        """测试4: 含 None 字段的配置"""
        print("\n[Test 4] None 字段...")
        config = TESMConfig.small()
        config.eos_token_id = None

        d = config.to_dict()
        restored = TESMConfig.from_dict(d)

        self.assertIsNone(restored.eos_token_id)
        print("  ✓ None 字段正确处理")


class TestModelCheckpoint(unittest.TestCase):
    """模型检查点测试"""

    def _make_tiny_config(self):
        return TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True, dropout=0.0,
        )

    def test_05_save_load_weights(self):
        """测试5: 模型权重保存/加载"""
        print("\n[Test 5] 权重保存/加载...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)

            model2 = TESMLMHeadModel(config)
            state = torch.load(path, map_location='cpu', weights_only=False)
            model2.load_state_dict(state)

            for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                          model2.named_parameters()):
                self.assertTrue(torch.allclose(p1, p2), f"参数 {n1} 不一致")
        print("  ✓ 权重保存/加载正确")

    def test_06_output_consistency_after_load(self):
        """测试6: 加载后输出一致性"""
        print("\n[Test 6] 加载后输出一致性...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        x = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            out1, _ = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)

            model2 = TESMLMHeadModel(config)
            model2.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            model2.eval()

            with torch.no_grad():
                out2, _ = model2(x)

            self.assertTrue(torch.allclose(out1.logits, out2.logits, atol=1e-6),
                          "加载后输出应完全一致")
        print("  ✓ 加载后输出一致")

    def test_07_full_checkpoint(self):
        """测试7: 完整检查点（模型+优化器+步数）"""
        print("\n[Test 7] 完整检查点...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 模拟一步训练
        x = torch.randint(0, 128, (1, 8))
        output, _ = model(x)
        loss = output.logits.mean()
        loss.backward()
        optimizer.step()

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': 42,
            'epoch': 3,
            'best_loss': 1.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(checkpoint, path)

            loaded = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(loaded['step'], 42)
            self.assertEqual(loaded['epoch'], 3)
            self.assertEqual(loaded['best_loss'], 1.5)

            # 恢复模型
            model2 = TESMLMHeadModel(config)
            model2.load_state_dict(loaded['model_state_dict'])

            # 恢复优化器
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            optimizer2.load_state_dict(loaded['optimizer_state_dict'])
        print("  ✓ 完整检查点保存/恢复正确")

    def test_08_multimodal_model_checkpoint(self):
        """测试8: 多模态模型检查点"""
        print("\n[Test 8] 多模态检查点...")
        config = MultimodalConfig.from_tesm_config(
            self._make_tiny_config(),
            vision_enabled=True,
            audio_enabled=True,
        )
        model = TESMMultimodalModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multimodal.pt")
            torch.save(model.state_dict(), path)

            model2 = TESMMultimodalModel(config)
            state = torch.load(path, map_location='cpu', weights_only=False)
            model2.load_state_dict(state)

            # 对比输出（小尺寸输入避免序列过长）
            images = torch.randn(1, 3, 32, 32)
            audio = torch.randn(1, 400)
            text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))

            model.eval()
            model2.eval()

            with torch.no_grad():
                out1, _ = model(images=images, audio=audio, text_ids=text_ids)
                out2, _ = model2(images=images, audio=audio, text_ids=text_ids)

            self.assertTrue(torch.allclose(out1.logits, out2.logits, atol=1e-6))
        print("  ✓ 多模态模型检查点正确")

    def test_09_partial_load(self):
        """测试9: 部分加载（strict=False）"""
        print("\n[Test 9] 部分加载...")
        config1 = self._make_tiny_config()
        model1 = TESMLMHeadModel(config1)

        config2 = TESMConfig(
            d_model=64, n_layer=1, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True,
        )
        model2 = TESMLMHeadModel(config2)

        # 不同层数，strict=False 应该可以加载部分参数
        missing, unexpected = model2.load_state_dict(
            model1.state_dict(), strict=False
        )

        # model1 有更多层 → state_dict 中有 model2 没有的 key → unexpected > 0
        self.assertGreater(len(unexpected), 0)
        print(f"  ✓ 部分加载: missing={len(missing)}, unexpected={len(unexpected)}")


def run_tests():
    print("=" * 60)
    print("检查点序列化/反序列化测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestConfigSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCheckpoint))
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
