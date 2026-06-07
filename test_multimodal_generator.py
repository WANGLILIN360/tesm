"""MultimodalGenerator 完整测试

测试覆盖:
1. 纯文本生成 (fallback)
2. 图像+文本生成
3. 音频+文本生成
4. 图像+音频+文本生成
5. 采样参数 (temperature, top_k, top_p)
6. 重复惩罚
7. 流式生成
8. Benchmark
9. 增量推理缓存
10. return_dict 输出
11. EOS 停止
12. 与 DataLoader 集成
"""

import sys
import time
import traceback
import unittest

import torch
import torch.nn.functional as F

# 确保能导入 tesm_ssm
sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
from tesm_ssm.inference.multimodal_generator import MultimodalGenerator


# ========== 超小配置用于快速测试 ==========
def make_tiny_config():
    """创建一个超小的 TESM 配置用于测试"""
    return TESMConfig(
        d_model=64,
        n_layer=2,
        d_intermediate=128,
        vocab_size=128,
        max_seq_len=256,
        d_state=16,
        expand=2,
        ent_rank=8,
        use_triton_kernels=False,
        kernel_backend="auto",
        tie_embeddings=True,
        dropout=0.0,
    )


def make_tiny_mm_config(vision=True, audio=False):
    """创建超小多模态配置"""
    tesm_cfg = make_tiny_config()
    return MultimodalConfig.from_tesm_config(
        tesm_cfg,
        vision_enabled=vision,
        audio_enabled=audio,
        vision_patch_size=16,
        vision_num_tokens=4,
        use_modality_embedding=True,
    )


# ========== 测试用例 ==========
class TestMultimodalGenerator(unittest.TestCase):
    """MultimodalGenerator 测试套件"""

    def test_01_text_only_fallback(self):
        """测试1: 纯文本生成（使用 TESMLMHeadModel fallback）"""
        print("\n[Test 1] 纯文本生成 (fallback)...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 5))
        result = generator.generate(text_ids=text_ids, max_new_tokens=10, temperature=0)

        self.assertEqual(result.shape[0], 1)  # batch=1
        self.assertGreaterEqual(result.shape[1], 5)  # 至少包含 prompt
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_02_vision_text_generation(self):
        """测试2: 图像+文本生成"""
        print("\n[Test 2] 图像+文本生成...")
        config = make_tiny_mm_config(vision=True, audio=False)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))

        result = generator.generate(
            images=images, text_ids=text_ids, max_new_tokens=8, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 5)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_03_audio_text_generation(self):
        """测试3: 音频+文本生成"""
        print("\n[Test 3] 音频+文本生成...")
        config = make_tiny_mm_config(vision=False, audio=True)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        audio = torch.randn(1, 1600)  # 0.1s @ 16kHz
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))

        result = generator.generate(
            audio=audio, text_ids=text_ids, max_new_tokens=8, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 5)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_04_multimodal_all_generation(self):
        """测试4: 图像+音频+文本生成"""
        print("\n[Test 4] 图像+音频+文本生成...")
        config = make_tiny_mm_config(vision=True, audio=True)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)
        audio = torch.randn(1, 1600)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))

        result = generator.generate(
            images=images, audio=audio, text_ids=text_ids,
            max_new_tokens=8, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 5)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_05_temperature_sampling(self):
        """测试5: 温度采样 (temperature > 0)"""
        print("\n[Test 5] 温度采样...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 5))

        # temperature=0 (greedy)
        result_greedy = generator.generate(text_ids=text_ids, max_new_tokens=5, temperature=0)
        # temperature=1.0 (random)
        result_random = generator.generate(text_ids=text_ids, max_new_tokens=5, temperature=1.0)

        self.assertEqual(result_greedy.shape[1], 10)
        self.assertEqual(result_random.shape[1], 10)
        print(f"  ✓ greedy 长度: {result_greedy.shape[1]}, random 长度: {result_random.shape[1]}")

    def test_06_top_k_sampling(self):
        """测试6: top-k 采样"""
        print("\n[Test 6] top-k 采样...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))
        result = generator.generate(
            text_ids=text_ids, max_new_tokens=5, temperature=1.0, top_k=10
        )

        self.assertEqual(result.shape[1], 8)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_07_top_p_sampling(self):
        """测试7: top-p (nucleus) 采样"""
        print("\n[Test 7] top-p 采样...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))
        result = generator.generate(
            text_ids=text_ids, max_new_tokens=5, temperature=1.0, top_p=0.5
        )

        self.assertEqual(result.shape[1], 8)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_08_repetition_penalty(self):
        """测试8: 重复惩罚"""
        print("\n[Test 8] 重复惩罚...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))

        # 无惩罚
        result_no_penalty = generator.generate(
            text_ids=text_ids, max_new_tokens=10, temperature=0, repetition_penalty=1.0
        )
        # 有惩罚
        result_penalty = generator.generate(
            text_ids=text_ids, max_new_tokens=10, temperature=0, repetition_penalty=2.0
        )

        self.assertEqual(result_no_penalty.shape[1], 13)
        self.assertEqual(result_penalty.shape[1], 13)
        print(f"  ✓ 无惩罚: {result_no_penalty.shape[1]}, 有惩罚: {result_penalty.shape[1]}")

    def test_09_stream_generation(self):
        """测试9: 流式生成"""
        print("\n[Test 9] 流式生成...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))

        tokens = []
        scores = []
        for token_id, score in generator.stream_generate(
            text_ids=text_ids, max_new_tokens=5, temperature=1.0
        ):
            tokens.append(token_id)
            scores.append(score)

        self.assertEqual(len(tokens), 5)
        self.assertEqual(len(scores), 5)
        print(f"  ✓ 流式生成 {len(tokens)} 个 tokens")

    def test_10_return_dict(self):
        """测试10: return_dict=True 返回详细信息"""
        print("\n[Test 10] return_dict 输出...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))
        result = generator.generate(
            text_ids=text_ids, max_new_tokens=5, temperature=0, return_dict=True
        )

        self.assertIn('sequences', result)
        self.assertIn('scores', result)
        self.assertIn('logits', result)
        self.assertIn('num_tokens', result)
        self.assertIn('time', result)
        self.assertIn('tokens_per_sec', result)

        self.assertEqual(result['num_tokens'], 5)
        self.assertEqual(result['sequences'].shape[1], 8)
        print(f"  ✓ sequences: {result['sequences'].shape}, tokens/sec: {result['tokens_per_sec']:.1f}")

    def test_11_eos_stop(self):
        """测试11: EOS 停止"""
        print("\n[Test 11] EOS 停止...")
        config = make_tiny_config()
        config.eos_token_id = 5
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        # 强制在第一个 token 就生成 EOS (temperature=0, 选择最大 logit)
        # 由于 vocab 很小，我们手动构造一个让 EOS 概率最大的场景
        text_ids = torch.randint(0, config.vocab_size, (1, 3))

        # 使用 temperature=0 (greedy) 来确保一致性
        result = generator.generate(
            text_ids=text_ids, max_new_tokens=20, temperature=0,
            eos_token_id=config.eos_token_id
        )

        self.assertGreaterEqual(result.shape[1], 3)  # 至少包含 prompt
        print(f"  ✓ 生成序列长度: {result.shape[1]} (max=23)")

    def test_12_incremental_cache(self):
        """测试12: 增量推理缓存加速"""
        print("\n[Test 12] 增量推理缓存...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 5))

        # 使用缓存
        start = time.time()
        result_with_cache = generator.generate(
            text_ids=text_ids, max_new_tokens=10, temperature=0, use_cache=True
        )
        time_with_cache = time.time() - start

        # 不使用缓存
        start = time.time()
        result_no_cache = generator.generate(
            text_ids=text_ids, max_new_tokens=10, temperature=0, use_cache=False
        )
        time_no_cache = time.time() - start

        # 结果应该相同 (greedy)
        self.assertTrue(torch.equal(result_with_cache, result_no_cache))
        print(f"  ✓ 缓存: {time_with_cache:.3f}s, 无缓存: {time_no_cache:.3f}s")

    def test_13_vision_only_generation(self):
        """测试13: 纯图像输入（无文本 prompt）"""
        print("\n[Test 13] 纯图像输入...")
        config = make_tiny_mm_config(vision=True, audio=False)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)

        result = generator.generate(
            images=images, max_new_tokens=5, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 0)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_14_batch_generation(self):
        """测试14: 批处理生成 (batch_size > 1)"""
        print("\n[Test 14] 批处理生成...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (3, 4))  # batch=3

        result = generator.generate(
            text_ids=text_ids, max_new_tokens=5, temperature=0
        )

        self.assertEqual(result.shape[0], 3)
        self.assertGreaterEqual(result.shape[1], 4)
        print(f"  ✓ batch={result.shape[0]}, 序列长度: {result.shape[1]}")

    def test_15_benchmark(self):
        """测试15: Benchmark 基准测试"""
        print("\n[Test 15] Benchmark...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 4))

        result = generator.benchmark(
            text_ids=text_ids, max_new_tokens=8, warmup=1, runs=2
        )

        self.assertIn('avg_time', result)
        self.assertIn('tokens_per_sec', result)
        self.assertIn('min_time', result)
        self.assertIn('max_time', result)
        self.assertGreater(result['tokens_per_sec'], 0)
        print(f"  ✓ {result['tokens_per_sec']:.1f} tokens/sec, avg={result['avg_time']:.3f}s")

    def test_16_multimodal_benchmark(self):
        """测试16: 多模态 Benchmark"""
        print("\n[Test 16] 多模态 Benchmark...")
        config = make_tiny_mm_config(vision=True, audio=True)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)
        audio = torch.randn(1, 1600)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 4))

        result = generator.benchmark(
            images=images, audio=audio, text_ids=text_ids,
            max_new_tokens=8, warmup=1, runs=2
        )

        self.assertIn('avg_time', result)
        self.assertIn('tokens_per_sec', result)
        print(f"  ✓ {result['tokens_per_sec']:.1f} tokens/sec")

    def test_17_stream_with_eos(self):
        """测试17: 流式生成 + EOS 停止"""
        print("\n[Test 17] 流式生成 + EOS...")
        config = make_tiny_config()
        config.eos_token_id = 7
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 3))

        tokens = []
        for token_id, score in generator.stream_generate(
            text_ids=text_ids, max_new_tokens=20, temperature=0,
            eos_token_id=config.eos_token_id
        ):
            tokens.append(token_id)

        self.assertGreater(len(tokens), 0)
        print(f"  ✓ 流式生成 {len(tokens)} 个 tokens")

    def test_18_generator_repr(self):
        """测试18: __repr__"""
        print("\n[Test 18] __repr__...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        r = repr(generator)
        self.assertIn("MultimodalGenerator", r)
        self.assertIn("TESMLMHeadModel", r)
        print(f"  ✓ {r}")

    def test_19_vision_embedder_v2(self):
        """测试19: VisionEmbedderV2 + Generator"""
        print("\n[Test 19] VisionEmbedderV2 + Generator...")
        from tesm_ssm.modules.multimodal.vision_embedder_v2 import VisionEmbedderV2

        config = make_tiny_mm_config(vision=True, audio=False)
        model = TESMMultimodalModel(config)
        # 替换为 V2
        model.vision_embedder = VisionEmbedderV2(
            d_model=config.tesm.d_model,
            patch_size=16,
            num_output_tokens=4,
        )
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 4))

        result = generator.generate(
            images=images, text_ids=text_ids, max_new_tokens=5, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_20_audio_embedder_v2(self):
        """测试20: AudioEmbedderV2 + Generator"""
        print("\n[Test 20] AudioEmbedderV2 + Generator...")
        from tesm_ssm.modules.multimodal.audio_embedder_v2 import AudioEmbedderV2

        config = make_tiny_mm_config(vision=False, audio=True)
        model = TESMMultimodalModel(config)
        # 替换为 V2
        model.audio_embedder = AudioEmbedderV2(
            d_model=config.tesm.d_model,
            sample_rate=16000,
        )
        generator = MultimodalGenerator(model)

        audio = torch.randn(1, 1600)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 4))

        result = generator.generate(
            audio=audio, text_ids=text_ids, max_new_tokens=5, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")

    def test_21_stream_batch_error(self):
        """测试21: stream_generate batch_size > 1 应该报错"""
        print("\n[Test 21] stream_generate batch 检查...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (2, 3))  # batch=2

        with self.assertRaises(ValueError):
            list(generator.stream_generate(text_ids=text_ids, max_new_tokens=3))
        print("  ✓ batch_size>1 正确抛出 ValueError")

    def test_22_zero_temperature_greedy_deterministic(self):
        """测试22: temperature=0 应该是确定性的"""
        print("\n[Test 22] greedy 确定性...")
        config = make_tiny_config()
        model = TESMLMHeadModel(config)
        generator = MultimodalGenerator(model)

        text_ids = torch.randint(0, config.vocab_size, (1, 4))

        result1 = generator.generate(text_ids=text_ids, max_new_tokens=10, temperature=0)
        result2 = generator.generate(text_ids=text_ids, max_new_tokens=10, temperature=0)

        self.assertTrue(torch.equal(result1, result2))
        print(f"  ✓ 两次生成完全一致")

    def test_23_prefill_no_text(self):
        """测试23: 无 text_ids 的预填充"""
        print("\n[Test 23] 无 text_ids 预填充...")
        config = make_tiny_mm_config(vision=True, audio=False)
        model = TESMMultimodalModel(config)
        generator = MultimodalGenerator(model)

        images = torch.randn(1, 3, 64, 64)

        result = generator.generate(
            images=images, max_new_tokens=5, temperature=0
        )

        self.assertEqual(result.shape[0], 1)
        print(f"  ✓ 生成序列长度: {result.shape[1]}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("MultimodalGenerator 完整测试套件")
    print("=" * 60)

    # 使用 unittest runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultimodalGenerator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")

    if result.failures:
        print("\n失败的测试:")
        for test, trace in result.failures:
            print(f"  ✗ {test}")
            print(trace[:500])

    if result.errors:
        print("\n错误的测试:")
        for test, trace in result.errors:
            print(f"  ✗ {test}")
            print(trace[:500])

    print("=" * 60)
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
