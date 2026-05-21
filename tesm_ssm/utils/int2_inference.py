"""INT2 量化模型推理工具

提供三种使用方式：
1. 训练时同时保存 FP32 和 INT2
2. 训练后转换 checkpoint
3. 直接加载 INT2 模型推理
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time


class Int2InferenceEngine:
    """INT2 推理引擎
    
    使用方法：
        # 方式1：从训练好的模型创建
        engine = Int2InferenceEngine.from_trained_model(model)
        
        # 方式2：从 checkpoint 加载
        engine = Int2InferenceEngine.from_checkpoint('model_int2.pt', model_class)
        
        # 方式3：从 FP32 checkpoint 转换
        engine = Int2InferenceEngine.from_fp32_checkpoint('model_fp32.pt', model_class)
        
        # 推理
        output = engine.generate(input_ids, max_length=100)
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # 统计信息
        self._inference_count = 0
        self._total_time = 0.0
    
    @classmethod
    def from_trained_model(cls, model: nn.Module, device: str = 'cuda') -> 'Int2InferenceEngine':
        """从训练好的模型创建 INT2 推理引擎
        
        Args:
            model: 训练好的模型（包含 BitLinear 层）
            device: 设备
        
        Returns:
            Int2InferenceEngine 实例
        """
        from .int2_quantization import Int2Model
        
        # 转换为 INT2 模型
        int2_model = Int2Model(original_model=model)
        
        return cls(int2_model, device)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_template: nn.Module, device: str = 'cuda') -> 'Int2InferenceEngine':
        """从 INT2 checkpoint 加载
        
        Args:
            checkpoint_path: INT2 checkpoint 路径
            model_template: 模型模板（用于获取结构）
            device: 设备
        
        Returns:
            Int2InferenceEngine 实例
        """
        from .int2_quantization import Int2Model
        
        # 加载 INT2 模型
        int2_model = Int2Model.load(checkpoint_path, model_template)
        
        return cls(int2_model, device)
    
    @classmethod
    def from_fp32_checkpoint(cls, checkpoint_path: str, model_class: callable, device: str = 'cuda') -> 'Int2InferenceEngine':
        """从 FP32 checkpoint 创建（自动转换）
        
        Args:
            checkpoint_path: FP32 checkpoint 路径
            model_class: 模型类（无参数构造函数）
            device: 设备
        
        Returns:
            Int2InferenceEngine 实例
        """
        # 创建模型
        model = model_class()
        
        # 加载 FP32 权重
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # 转换为 INT2
        return cls.from_trained_model(model, device)
    
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        """前向推理"""
        return self.model(*args, **kwargs)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """生成文本（自回归）
        
        Args:
            input_ids: 输入 token IDs，shape [batch, seq_len]
            max_length: 最大生成长度
            temperature: 温度
            top_k: top-k 采样
            top_p: top-p 采样
            do_sample: 是否采样
        
        Returns:
            生成的 token IDs
        """
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]
        
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # 前向传播
            outputs = self.model(generated, **kwargs)
            
            # 获取 logits（假设最后一个维度是 vocab）
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
            
            # 取最后一个位置
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Top-k 采样
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                
                # Top-p 采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def benchmark(self, input_shape: tuple, num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
        """性能基准测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            warmup: 预热次数
        
        Returns:
            性能统计
        """
        # 创建随机输入
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # 预热
        for _ in range(warmup):
            _ = self.forward(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        times = []
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = self.forward(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            'avg_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'throughput': 1000.0 / avg_time,  # samples/sec
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        from .int2_quantization import Int2Linear, Int2Model
        
        # 统计层类型
        layer_counts = {}
        int2_layers = 0
        total_params = 0
        int2_params = 0
        
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
            
            if isinstance(module, Int2Linear):
                int2_layers += 1
                int2_params += module.packed_weight.numel()
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
        
        for name, buf in self.model.named_buffers():
            total_params += buf.numel()
        
        # 计算大小
        total_size_mb = total_params * 4 / (1024 * 1024)  # 假设 FP32
        int2_size_mb = int2_params / (1024 * 1024)  # INT2 = 1 byte
        
        return {
            'total_params': total_params,
            'int2_params': int2_params,
            'int2_layers': int2_layers,
            'layer_counts': layer_counts,
            'estimated_fp32_size_mb': total_size_mb,
            'int2_weight_size_mb': int2_size_mb,
            'compression_ratio': total_size_mb / int2_size_mb if int2_size_mb > 0 else 1.0,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_int2_engine(model: nn.Module = None, checkpoint_path: str = None, model_class: callable = None, device: str = 'cuda') -> Int2InferenceEngine:
    """创建 INT2 推理引擎（自动选择方式）
    
    Args:
        model: 训练好的模型
        checkpoint_path: checkpoint 路径
        model_class: 模型类
        device: 设备
    
    Returns:
        Int2InferenceEngine 实例
    """
    if model is not None:
        return Int2InferenceEngine.from_trained_model(model, device)
    elif checkpoint_path is not None and model_class is not None:
        # 判断是 INT2 还是 FP32 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'packed_weight' in str(checkpoint):
            return Int2InferenceEngine.from_checkpoint(checkpoint_path, model_class(), device)
        else:
            return Int2InferenceEngine.from_fp32_checkpoint(checkpoint_path, model_class, device)
    else:
        raise ValueError("需要提供 model 或 (checkpoint_path + model_class)")


def benchmark_int2_vs_fp32(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> Dict[str, Any]:
    """对比 INT2 和 FP32 推理性能
    
    Args:
        model: 模型
        input_shape: 输入形状
        device: 设备
    
    Returns:
        对比结果
    """
    # FP32 基准
    model_fp32 = model.to(device)
    model_fp32.eval()
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    # FP32 测试
    with torch.no_grad():
        for _ in range(10):
            _ = model_fp32(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        times_fp32 = []
        for _ in range(100):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model_fp32(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times_fp32.append((time.time() - start) * 1000)
    
    # INT2 测试
    engine_int2 = Int2InferenceEngine.from_trained_model(model, device)
    stats_int2 = engine_int2.benchmark(input_shape)
    
    return {
        'fp32_avg_ms': sum(times_fp32) / len(times_fp32),
        'int2_avg_ms': stats_int2['avg_ms'],
        'speedup': sum(times_fp32) / len(times_fp32) / stats_int2['avg_ms'],
        'model_info': engine_int2.get_model_info(),
    }
