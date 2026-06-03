"""配置系统测试"""
import pytest

from tesm_ssm import TESMConfig


class TestTESMConfig:
    def test_default_config(self):
        """测试默认配置"""
        config = TESMConfig()
        assert config.d_model == 768
        assert config.n_layer == 24
        assert config.vocab_size == 151936
        assert config.max_seq_len == 2048

    def test_tiny_preset(self):
        """测试 tiny 预设"""
        config = TESMConfig.tiny()
        assert config.d_model == 768
        assert config.n_layer == 12
        assert config.max_seq_len == 1024
        assert config.decay_init_bias == 1.0

    def test_small_preset(self):
        """测试 small 预设"""
        config = TESMConfig.small()
        assert config.d_model == 512
        assert config.n_layer == 16
        assert config.decay_init_bias == 1.0

    def test_base_preset(self):
        """测试 base 预设"""
        config = TESMConfig.base()
        assert config.d_model == 768
        assert config.n_layer == 24
        assert config.decay_init_bias == 2.0

    def test_long_context_preset(self):
        """测试长上下文预设"""
        config = TESMConfig.long_context()
        assert config.max_seq_len == 16384
        assert config.decay_init_bias == 6.0

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        config = TESMConfig.small()
        d = config.to_dict()
        restored = TESMConfig.from_dict(d)
        assert restored.d_model == config.d_model
        assert restored.n_layer == config.n_layer

    def test_ssm_defaults(self):
        """测试 SSM 配置默认值"""
        config = TESMConfig()
        assert config.d_state == 256
        assert config.entanglement_window == 16
        assert config.entanglement_block_size == 256
        assert config.state_scan_chunk_size == 16
