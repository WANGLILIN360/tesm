"""模型测试"""
import pytest
import torch

from tesm_ssm import TESMConfig, TESMLMHeadModel


class TestTESMLMHeadModel:
    @pytest.fixture
    def tiny_config(self):
        return TESMConfig.tiny()

    def test_model_creation(self, tiny_config):
        """测试模型创建"""
        model = TESMLMHeadModel(tiny_config)
        assert model is not None

    def test_forward_pass(self, tiny_config):
        """测试前向传播"""
        model = TESMLMHeadModel(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs, _ = model(input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_with_labels(self, tiny_config):
        """测试带标签的前向传播"""
        model = TESMLMHeadModel(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        with torch.no_grad():
            outputs, _ = model(input_ids, labels=labels)

        assert outputs.loss is not None

    def test_gradient_flow(self, tiny_config):
        """测试梯度流"""
        model = TESMLMHeadModel(tiny_config)
        model.train()

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        outputs, _ = model(input_ids, labels=labels)
        outputs.loss.backward()

        # 检查关键参数有梯度
        for name, param in model.named_parameters():
            if param.requires_grad and "embedding" not in name:
                assert param.grad is not None, f"No gradient for {name}"
                break

    def test_different_seq_lengths(self, tiny_config):
        """测试不同序列长度"""
        model = TESMLMHeadModel(tiny_config)
        model.eval()

        for seq_len in [16, 32, 64, 128]:
            if seq_len > tiny_config.max_seq_len:
                continue
            input_ids = torch.randint(0, tiny_config.vocab_size, (1, seq_len))
            with torch.no_grad():
                outputs, _ = model(input_ids)
            assert outputs.logits.shape[1] == seq_len


class TestBitLinear:
    def test_bitlinear_creation(self):
        """测试 BitLinear 创建"""
        from tesm_ssm.modules.tesm import BitLinear

        layer = BitLinear(256, 512)
        assert layer.in_features == 256
        assert layer.out_features == 512

    def test_bitlinear_forward(self):
        """测试 BitLinear 前向传播"""
        from tesm_ssm.modules.tesm import BitLinear

        layer = BitLinear(256, 512)
        layer.eval()

        x = torch.randn(2, 32, 256)
        with torch.no_grad():
            out = layer(x)

        assert out.shape == (2, 32, 512)
