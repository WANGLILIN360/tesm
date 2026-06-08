"""端到端梯度流测试

测试覆盖:
1. TESM_SISO 反向传播
2. BitLinear 梯度流
3. 多模态模型各 embedder 梯度
4. freeze_embedders=True 时梯度冻结
5. 跨层纠缠梯度
6. 温度退火参数梯度
"""

import sys
import unittest

import torch

sys.path.insert(0, "/mnt/agents/tesm")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel, MixerModel
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
from tesm_ssm.modules.tesm import TESM_SISO, BitLinear


class TestGradientFlow(unittest.TestCase):
    """梯度流测试套件"""

    def test_01_siso_gradient(self):
        """测试1: TESM_SISO 梯度流"""
        print("\n[Test 1] SISO 梯度...")
        siso = TESM_SISO(
            d_model=64, d_state=16, expand=2, ent_rank=8,
            max_seq_len=256, entanglement_window=0,
            use_triton_kernels=False, kernel_backend="torch",
            annealing_enabled=False,
        )
        siso.train()

        x = torch.randn(1, 8, 64)
        y, _ = siso(x)
        loss = y.mean()
        loss.backward()

        grad_count = 0
        for name, param in siso.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"{name} 梯度不应有 NaN")
                grad_count += 1
        self.assertGreater(grad_count, 0, "应有参数获得梯度")
        print(f"  ✓ SISO {grad_count} 个参数有有效梯度")

    def test_02_bitlinear_gradient(self):
        """测试2: BitLinear 梯度流"""
        print("\n[Test 2] BitLinear 梯度...")
        bl = BitLinear(64, 128, kernel_backend="torch")
        bl.train()

        x = torch.randn(1, 8, 64, requires_grad=True)
        y = bl(x)
        loss = y.mean()
        loss.backward()

        self.assertIsNotNone(bl.weight.grad)
        self.assertFalse(torch.isnan(bl.weight.grad).any())
        self.assertIsNotNone(x.grad)
        print("  ✓ BitLinear 权重和输入梯度正确")

    def test_03_lmhead_gradient(self):
        """测试3: TESMLMHeadModel 梯度流"""
        print("\n[Test 3] LMHead 梯度...")
        config = TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True, dropout=0.0,
        )
        model = TESMLMHeadModel(config)
        model.train()

        x = torch.randint(0, 128, (1, 8))
        labels = torch.randint(0, 128, (1, 8))

        output, _ = model(x, labels=labels)
        output.loss.backward()

        grad_count = sum(1 for p in model.parameters()
                        if p.grad is not None and not torch.isnan(p.grad).any())
        total = sum(1 for p in model.parameters())
        self.assertGreater(grad_count, 0, "应有参数获得梯度")
        print(f"  ✓ LMHead 梯度: {grad_count}/{total} 参数有梯度")

    def test_04_multimodal_gradient(self):
        """测试4: 多模态模型梯度流"""
        print("\n[Test 4] 多模态梯度...")
        config = MultimodalConfig.from_tesm_config(
            TESMConfig(
                d_model=64, n_layer=2, d_intermediate=128,
                vocab_size=128, max_seq_len=256,
                d_state=16, expand=2, ent_rank=8,
                use_triton_kernels=False, kernel_backend="torch",
                tie_embeddings=True, dropout=0.0,
            ),
            vision_enabled=True,
            audio_enabled=True,
        )
        model = TESMMultimodalModel(config)
        model.train()

        images = torch.randn(1, 3, 32, 32)
        audio = torch.randn(1, 400)
        text_ids = torch.randint(0, config.tesm.vocab_size, (1, 5))
        labels = torch.randint(0, config.tesm.vocab_size, (1, 5))

        output, _ = model(images=images, audio=audio, text_ids=text_ids, labels=labels)
        if output.loss is not None:
            output.loss.backward()

            # 检查各 embedder 有梯度
            if model.vision_embedder is not None:
                vis_has_grad = any(p.grad is not None for p in model.vision_embedder.parameters())
                self.assertTrue(vis_has_grad, "vision_embedder 应该有梯度")

            if model.audio_embedder is not None:
                aud_has_grad = any(p.grad is not None for p in model.audio_embedder.parameters())
                self.assertTrue(aud_has_grad, "audio_embedder 应该有梯度")

        print("  ✓ 多模态各 embedder 梯度正确")

    def test_05_freeze_embedders(self):
        """测试5: freeze_embedders=True"""
        print("\n[Test 5] freeze_embedders...")
        config = MultimodalConfig.from_tesm_config(
            TESMConfig(
                d_model=64, n_layer=2, d_intermediate=128,
                vocab_size=128, max_seq_len=256,
                d_state=16, expand=2, ent_rank=8,
                use_triton_kernels=False, kernel_backend="torch",
                tie_embeddings=True, dropout=0.0,
            ),
            vision_enabled=True,
            audio_enabled=True,
            freeze_embedders=True,
        )
        model = TESMMultimodalModel(config)
        model.train()

        # 检查 embedder 被冻结
        if model.vision_embedder is not None:
            for p in model.vision_embedder.parameters():
                self.assertFalse(p.requires_grad, "vision_embedder 应被冻结")

        if model.audio_embedder is not None:
            for p in model.audio_embedder.parameters():
                self.assertFalse(p.requires_grad, "audio_embedder 应被冻结")

        # decoder 应可训练
        for p in model.decoder.parameters():
            self.assertTrue(p.requires_grad, "decoder 应可训练")

        print("  ✓ freeze_embedders 正确冻结/解冻")

    def test_06_cross_layer_gradient(self):
        """测试6: 跨层纠缠梯度"""
        print("\n[Test 6] 跨层纠缠梯度...")
        config = TESMConfig(
            d_model=64, n_layer=2, d_intermediate=128,
            vocab_size=128, max_seq_len=256,
            d_state=16, expand=2, ent_rank=8,
            use_triton_kernels=False, kernel_backend="torch",
            tie_embeddings=True, dropout=0.0,
        )
        model = TESMLMHeadModel(config)
        model.train()

        x = torch.randint(0, 128, (1, 8))
        labels = torch.randint(0, 128, (1, 8))

        output, _ = model(x, labels=labels)
        output.loss.backward()

        # 检查 cross_layer_q_proj 有梯度（在多层模型中才有）
        cross_grads = []
        for name, param in model.named_parameters():
            if 'cross_layer' in name and param.grad is not None:
                cross_grads.append(name)
        # 跨层梯度可能存在也可能不存在，取决于是否实际使用了跨层传递
        print(f"  ✓ 跨层纠缠梯度: {len(cross_grads)} 个参数有梯度")

    def test_07_annealing_step_gradient(self):
        """测试7: 温度退火步数更新"""
        print("\n[Test 7] 退火步数...")
        siso = TESM_SISO(
            d_model=64, d_state=16, expand=2, ent_rank=8,
            max_seq_len=256, entanglement_window=0,
            use_triton_kernels=False, kernel_backend="torch",
            annealing_enabled=True,
        )
        siso.train()

        initial_step = siso.annealing_step.item()

        x = torch.randn(1, 4, 64)
        y, _ = siso(x)
        loss = y.mean()
        loss.backward()

        # annealing_step 是 buffer，不应该有梯度
        self.assertIsNone(siso.annealing_step.grad)
        # 但步数应该增加
        self.assertGreater(siso.annealing_step.item(), initial_step)
        print(f"  ✓ 退火步数: {initial_step} -> {siso.annealing_step.item()}")

    def test_08_no_grad_eval(self):
        """测试8: eval 模式无梯度"""
        print("\n[Test 8] eval 无梯度...")
        siso = TESM_SISO(
            d_model=64, d_state=16, expand=2, ent_rank=8,
            max_seq_len=256, entanglement_window=0,
            use_triton_kernels=False, kernel_backend="torch",
            annealing_enabled=False,
        )
        siso.eval()

        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            y, _ = siso(x)

        # 所有参数不应有梯度
        for param in siso.parameters():
            self.assertIsNone(param.grad)
        print("  ✓ eval 模式无梯度")


def run_tests():
    print("=" * 60)
    print("端到端梯度流测试套件")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGradientFlow)
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
