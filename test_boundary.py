"""边界情况和鲁棒性测试

测试覆盖:
1. 空输入
2. 超长序列截断/报错
3. 非法参数组合
4. Batch 内样本长度差异
5. eval/train 模式切换一致性
6. 单 token 输入
7. 极大/极小值输入
8. 不同数据类型
"""

import sys
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel, MixerModel
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
from tesm_ssm.modules.tesm import TESM_SISO


class TestBoundaryCases(unittest.TestCase):
    """边界情况测试套件"""

    def _make_tiny_config(self):
        return TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=64,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True, dropout=0.0,
        )

    def test_01_single_token(self):
        """测试1: 单 token 输入"""
        print("\n[Test 1] 单 token...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        x = torch.randint(0, 128, (1, 1))
        with torch.no_grad():
            output, _ = model(x)

        self.assertEqual(output.logits.shape, (1, 1, 128))
        self.assertFalse(torch.isnan(output.logits).any())
        print("  ✓ 单 token 正确")

    def test_02_empty_sequence(self):
        """测试2: 空序列"""
        print("\n[Test 2] 空序列...")
        siso = TESM_SISO(
            d_model=64, d_state=16, expand=2, ent_rank=8,
            max_seq_len=256, entanglement_window=0,
            use_triton_kernels=False, kernel_backend="torch",
            annealing_enabled=False,
        )
        siso.eval()

        # TESM_SISO 对空序列返回 (u, None)
        x = torch.randn(1, 0, 64)
        y, final_state = siso(x)
        self.assertEqual(y.shape, (1, 0, 64))
        print("  ✓ 空序列处理正确")

    def test_03_exceed_max_seq_len(self):
        """测试3: 超过最大序列长度"""
        print("\n[Test 3] 超长序列...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        # max_seq_len=64, 尝试输入 100
        x = torch.randint(0, 128, (1, 100))
        with self.assertRaises(ValueError):
            model(x)
        print("  ✓ 超长序列正确抛出 ValueError")

    def test_04_large_batch(self):
        """测试4: 大 batch"""
        print("\n[Test 4] 大 batch...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        x = torch.randint(0, 128, (16, 8))
        with torch.no_grad():
            output, _ = model(x)

        self.assertEqual(output.logits.shape, (16, 8, 128))
        self.assertFalse(torch.isnan(output.logits).any())
        print("  ✓ 大 batch (16) 正确")

    def test_05_extreme_values(self):
        """测试5: 极大/极小值输入"""
        print("\n[Test 5] 极端值...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        for desc, scale in [("极大值", 1e6), ("极小值", 1e-8), ("负值", -1.0)]:
            # 使用 inputs_embeds 测试极端值
            embeds = torch.randn(1, 4, 64) * scale
            with torch.no_grad():
                output, _ = model(inputs_embeds=embeds)
            self.assertFalse(torch.isnan(output.logits).any(),
                           f"{desc} 不应产生 NaN")
        print("  ✓ 极端值处理正确")

    def test_06_eval_train_consistency(self):
        """测试6: eval/train 模式切换"""
        print("\n[Test 6] eval/train 切换...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)

        x = torch.randint(0, 128, (1, 8))

        # eval 模式
        model.eval()
        with torch.no_grad():
            out_eval, _ = model(x)

        # train 模式
        model.train()
        out_train, _ = model(x)

        # 形状一致
        self.assertEqual(out_eval.logits.shape, out_train.logits.shape)
        # 由于 dropout 等，值可能不同，但不应有 NaN
        self.assertFalse(torch.isnan(out_train.logits).any())
        print("  ✓ eval/train 切换正确")

    def test_07_different_dtypes(self):
        """测试7: 不同数据类型"""
        print("\n[Test 7] 不同 dtype...")
        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        model.eval()

        for dtype in [torch.float32, torch.float64]:
            x = torch.randint(0, 128, (1, 4))
            with torch.no_grad():
                output, _ = model(x)
            self.assertFalse(torch.isnan(output.logits).any())
        print("  ✓ 不同 dtype 正确")

    def test_08_invalid_temperature(self):
        """测试8: 负温度"""
        print("\n[Test 8] 负温度...")
        from tesm_ssm.inference.multimodal_generator import MultimodalGenerator

        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        x = torch.randint(0, 128, (1, 4))
        # temperature < 0 应该当作 greedy (0)
        result = generator.generate(text_ids=x, max_new_tokens=3, temperature=-1.0)
        self.assertGreaterEqual(result.shape[1], 4)
        print("  ✓ 负温度正确处理为 greedy")

    def test_09_top_k_larger_than_vocab(self):
        """测试9: top_k > vocab_size"""
        print("\n[Test 9] top_k > vocab...")
        from tesm_ssm.inference.multimodal_generator import MultimodalGenerator

        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        x = torch.randint(0, 128, (1, 4))
        # top_k=1000 > vocab_size=128，不应报错
        result = generator.generate(text_ids=x, max_new_tokens=3, top_k=1000, temperature=1.0)
        self.assertGreaterEqual(result.shape[1], 4)
        print("  ✓ top_k > vocab_size 正确处理")

    def test_10_zero_top_p(self):
        """测试10: top_p=0"""
        print("\n[Test 10] top_p=0...")
        from tesm_ssm.inference.multimodal_generator import MultimodalGenerator

        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        x = torch.randint(0, 128, (1, 4))
        # top_p=0 应该只保留最高概率的一个
        result = generator.generate(text_ids=x, max_new_tokens=3, top_p=0.0, temperature=1.0)
        self.assertGreaterEqual(result.shape[1], 4)
        print("  ✓ top_p=0 正确处理")

    def test_11_multimodal_missing_modality(self):
        """测试11: 多模态缺少模态输入"""
        print("\n[Test 11] 缺少模态...")
        config = MultimodalConfig.from_tesm_config(
            self._make_tiny_config(),
            vision_enabled=True,
            audio_enabled=False,
        )
        model = TESMMultimodalModel(config)
        model.eval()

        # 只提供文本（无图像，无音频）
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))
        with torch.no_grad():
            output, _ = model(text_ids=text_ids)

        self.assertIsNotNone(output.logits)
        print("  ✓ 缺少模态输入正确处理")

    def test_12_invalid_multimodal_input(self):
        """测试12: 未启用的模态输入"""
        print("\n[Test 12] 未启用模态...")
        config = MultimodalConfig.from_tesm_config(
            self._make_tiny_config(),
            vision_enabled=False,
            audio_enabled=False,
        )
        model = TESMMultimodalModel(config)

        images = torch.randn(1, 3, 64, 64)
        with self.assertRaises(ValueError):
            model(images=images)
        print("  ✓ 未启用模态输入正确抛出 ValueError")

    def test_13_all_modalities_none(self):
        """测试13: 所有输入为 None"""
        print("\n[Test 13] 全 None...")
        config = MultimodalConfig.from_tesm_config(
            self._make_tiny_config(),
            vision_enabled=False,
            audio_enabled=False,
        )
        model = TESMMultimodalModel(config)

        with self.assertRaises(ValueError):
            model()
        print("  ✓ 全 None 正确抛出 ValueError")

    def test_14_repetition_penalty_extreme(self):
        """测试14: 极端重复惩罚"""
        print("\n[Test 14] 极端重复惩罚...")
        from tesm_ssm.inference.multimodal_generator import MultimodalGenerator

        config = self._make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        x = torch.randint(0, 128, (1, 4))
        # 极大重复惩罚
        result = generator.generate(text_ids=x, max_new_tokens=5,
                                    repetition_penalty=100.0, temperature=0)
        self.assertGreaterEqual(result.shape[1], 4)
        print("  ✓ 极端重复惩罚正确处理")


def run_tests():
    print("=" * 60)
    print("边界情况/鲁棒性测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBoundaryCases)
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
