"""算子测试"""
import pytest
import torch

from tesm_ssm.modules.tesm import BitLinear


class TestQuantization:
    def test_int2_quantization_shape(self):
        """测试 INT2 量化形状"""
        layer = BitLinear(256, 512)
        x = torch.randn(2, 32, 256)

        qweight = layer._current_quantized_weight()
        assert qweight.shape == (512, 256)

    def test_quantized_input(self):
        """测试量化输入"""
        layer = BitLinear(256, 512)
        x = torch.randn(2, 32, 256)

        qinput = layer.quantized_input(x)
        assert qinput.shape == x.shape

    def test_kernel_backend_torch(self):
        """测试 PyTorch 后端"""
        layer = BitLinear(256, 512, kernel_backend="torch")
        x = torch.randn(2, 32, 256)

        out = layer(x)
        assert out.shape == (2, 32, 512)


class TestCUDAOps:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_availability(self):
        """测试 CUDA 可用性"""
        from tesm_ssm.ops.cuda import tesm_cuda_is_available

        # 可能返回 False 如果扩展未编译
        result = tesm_cuda_is_available()
        assert isinstance(result, bool)


class TestTritonOps:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_availability(self):
        """测试 Triton 可用性"""
        from tesm_ssm.ops.triton import tesm_triton_is_available

        result = tesm_triton_is_available()
        assert isinstance(result, bool)
