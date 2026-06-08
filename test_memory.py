"""内存使用/泄漏测试

测试覆盖:
1. 短时间增量推理
2. 循环创建/销毁模型
3. 推理缓存复用
"""

import sys
import gc
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel


class TestMemory(unittest.TestCase):
    """内存测试套件"""

    def _make_config(self):
        return TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True, dropout=0.0,
        )

    def test_01_incremental_memory(self):
        print("\n[Test 1] 增量推理内存...")
        config = self._make_config()
        model = TESMLMHeadModel(config)
        model.eval()

        x_prefill = torch.randint(0, 128, (1, 5))
        cache = model.backbone.allocate_inference_cache(1, 128)
        inference_params = {'state_cache': cache}

        with torch.no_grad():
            _, _ = model.backbone(x_prefill, inference_params=inference_params)

        for i in range(10):
            x_step = torch.randint(0, 128, (1, 1))
            with torch.no_grad():
                _, _ = model.backbone(x_step, inference_params=inference_params)

        print("  ✓ 10步增量推理完成，无内存异常")

    def test_02_model_create_destroy(self):
        print("\n[Test 2] 模型创建/销毁...")
        config = self._make_config()

        for i in range(5):
            model = TESMLMHeadModel(config)
            x = torch.randint(0, 128, (1, 4))
            with torch.no_grad():
                output, _ = model(x)
            self.assertEqual(output.logits.shape, (1, 4, 128))
            del model
            gc.collect()

        print("  ✓ 5次模型创建/销毁完成")

    def test_03_cache_reuse(self):
        print("\n[Test 3] 缓存复用...")
        config = self._make_config()
        model = TESMLMHeadModel(config)
        model.eval()

        cache = model.backbone.allocate_inference_cache(1, 64)

        for round_idx in range(3):
            cache['seq_pos'] = 0
            cache['state'].zero_()

            x = torch.randint(0, 128, (1, 4))
            inference_params = {'state_cache': cache}

            with torch.no_grad():
                _, _ = model.backbone(x, inference_params=inference_params)

            for _ in range(5):
                x_step = torch.randint(0, 128, (1, 1))
                with torch.no_grad():
                    _, _ = model.backbone(x_step, inference_params=inference_params)

        print("  ✓ 推理缓存复用 3 轮完成")


def run_tests():
    print("=" * 60)
    print("内存使用/泄漏测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMemory)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 60)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"总测试数: {result.testsRun}, 通过: {passed}, 失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 60)
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
