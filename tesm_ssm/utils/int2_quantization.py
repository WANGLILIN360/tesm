"""INT2 权重打包和解包

三值量化：{-1, 0, +1} 需要 2 bit 存储
打包方式：4 个 int2 值打包成 1 个 uint8

编码：
  -1 → 0b00 (0)
   0 → 0b01 (1)
  +1 → 0b10 (2)
  0b11 未使用

打包示例：
  [-1, 0, +1, 0] → 0b_01_10_01_00 = 0x64
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


def pack_int2_to_uint8(int2_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将三值权重打包成 uint8
    
    Args:
        int2_values: int8 tensor，值在 {-1, 0, +1} 范围内，shape [out_features, in_features]
    
    Returns:
        packed: uint8 tensor，shape [out_features, in_features // 4]
        scale: 缩放因子
    """
    # 计算缩放因子
    scale = 1.0 / int2_values.abs().mean().clamp_min(1e-8)
    
    # 量化到 {-1, 0, +1}
    normalized = int2_values * scale
    quantized = normalized.round().clamp(-1, 1).to(torch.int8)
    
    # 编码：-1→0, 0→1, +1→2
    encoded = (quantized + 1).to(torch.uint8)  # 现在 {0, 1, 2}
    
    # 确保 in_features 可以被 4 整除
    out_features, in_features = encoded.shape
    if in_features % 4 != 0:
        pad_size = 4 - (in_features % 4)
        encoded = torch.nn.functional.pad(encoded, (0, pad_size), value=1)  # 用 0（编码为1）填充
        in_features = encoded.shape[1]
    
    # 重塑为 [out_features, in_features // 4, 4]
    encoded = encoded.reshape(out_features, in_features // 4, 4)
    
    # 打包 4 个 int2 到 1 个 uint8
    # 低位在前：[v0, v1, v2, v3] → v0 | (v1<<2) | (v2<<4) | (v3<<6)
    packed = (
        encoded[:, :, 0] |
        (encoded[:, :, 1] << 2) |
        (encoded[:, :, 2] << 4) |
        (encoded[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return packed, scale.detach().clone()


def unpack_uint8_to_int2(packed: torch.Tensor, scale: float) -> torch.Tensor:
    """将 uint8 解包为三值权重
    
    Args:
        packed: uint8 tensor，shape [out_features, in_features_packed]
        scale: 缩放因子
    
    Returns:
        unpacked: float tensor，shape [out_features, in_features_packed * 4]
    """
    out_features, in_features_packed = packed.shape
    in_features = in_features_packed * 4
    
    # 解包每个 uint8 到 4 个 int2
    v0 = (packed & 0b00000011).to(torch.int8)  # bits 0-1
    v1 = ((packed >> 2) & 0b00000011).to(torch.int8)  # bits 2-3
    v2 = ((packed >> 4) & 0b00000011).to(torch.int8)  # bits 4-5
    v3 = ((packed >> 6) & 0b00000011).to(torch.int8)  # bits 6-7
    
    # 解码：0→-1, 1→0, 2→+1
    decoded_v0 = v0 - 1
    decoded_v1 = v1 - 1
    decoded_v2 = v2 - 1
    decoded_v3 = v3 - 1
    
    # 合并
    unpacked = torch.stack([decoded_v0, decoded_v1, decoded_v2, decoded_v3], dim=2)
    unpacked = unpacked.reshape(out_features, in_features)
    
    # 反量化
    return unpacked.float() / scale


def quantize_weight_to_int2(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 FP32 权重量化并打包为 INT2
    
    Args:
        weight: float tensor，shape [out_features, in_features]
    
    Returns:
        packed: uint8 tensor，shape [out_features, in_features_packed]
        scale: 缩放因子
    """
    with torch.no_grad():
        return pack_int2_to_uint8(weight)


class Int2Linear(nn.Module):
    """INT2 量化线性层（用于推理）
    
    权重以打包的 INT2 格式存储，推理时使用 CUDA kernel。
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_cuda_kernel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cuda_kernel = use_cuda_kernel
        
        # 注册打包的权重（uint8）
        self.register_buffer('packed_weight', packed_weight)
        self.register_buffer('weight_scale', weight_scale)
        
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        # 预解包权重（fallback）
        self._pre_unpacked = None
    
    def _can_use_cuda_kernel(self, x: torch.Tensor) -> bool:
        if not self.use_cuda_kernel:
            return False
        try:
            from tesm_ssm.ops.cuda import cuda_int2_linear, tesm_cuda_is_available
            return tesm_cuda_is_available() and x.is_cuda
        except ImportError:
            return False
    
    def _get_unpacked_weight(self) -> torch.Tensor:
        if self._pre_unpacked is None:
            self._pre_unpacked = unpack_uint8_to_int2(
                self.packed_weight, self.weight_scale.item()
            )
        return self._pre_unpacked
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入 tensor，shape [..., in_features]
        
        Returns:
            输出 tensor，shape [..., out_features]
        """
        # 尝试使用 CUDA kernel
        if self._can_use_cuda_kernel(x):
            from tesm_ssm.ops.cuda import cuda_int2_linear
            return cuda_int2_linear(x, self.packed_weight, self.weight_scale, self.bias)
        
        # Fallback：使用预解包的权重
        weight = self._get_unpacked_weight()
        return torch.nn.functional.linear(x, weight, self.bias)
    
    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'Int2Linear':
        """从 FP32 线性层创建 INT2 层
        
        Args:
            linear: FP32 线性层
        
        Returns:
            Int2Linear 实例
        """
        packed, scale = quantize_weight_to_int2(linear.weight.data)
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            packed_weight=packed,
            weight_scale=scale,
            bias=linear.bias.data if linear.bias is not None else None,
        )


def export_model_to_int2(model: nn.Module) -> Dict[str, Dict]:
    """将模型的所有 BitLinear 层导出为 INT2 格式
    
    Args:
        model: 包含 BitLinear 层的模型
    
    Returns:
        字典：{层名: {'packed_weight': tensor, 'scale': tensor, 'bias': tensor或None}}
    """
    from tesm_ssm.modules.tesm import BitLinear
    
    exported = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            packed, scale = quantize_weight_to_int2(module.weight.data)
            exported[name] = {
                'packed_weight': packed.cpu(),
                'weight_scale': scale.cpu(),
                'bias': module.bias.data.cpu() if module.bias is not None else None,
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
    return exported


def load_int2_weights_to_model(model: nn.Module, exported: Dict[str, Dict]):
    """将 INT2 权重加载到模型的 BitLinear 层
    
    Args:
        model: 目标模型
        exported: export_model_to_int2 的输出
    """
    from tesm_ssm.modules.tesm import BitLinear
    
    for name, module in model.named_modules():
        if name in exported and isinstance(module, BitLinear):
            data = exported[name]
            # 创建 Int2Linear 替换
            int2_linear = Int2Linear(
                in_features=data['in_features'],
                out_features=data['out_features'],
                packed_weight=data['packed_weight'].to(module.weight.device),
                weight_scale=data['weight_scale'].to(module.weight.device),
                bias=data['bias'].to(module.weight.device) if data['bias'] is not None else None,
            )
            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            setattr(parent, child_name, int2_linear)


def save_int2_model(model: nn.Module, path: str):
    """保存 INT2 量化模型
    
    Args:
        model: 模型
        path: 保存路径
    """
    exported = export_model_to_int2(model)
    torch.save(exported, path)
    print(f"INT2 模型已保存到 {path}")
    
    # 计算大小
    total_params = sum(
        data['packed_weight'].numel() 
        for data in exported.values()
    )
    print(f"打包后权重大小: {total_params / (1024*1024):.2f} MB (INT2)")


def load_int2_model(model: nn.Module, path: str):
    """加载 INT2 量化模型
    
    Args:
        model: 目标模型（结构需要匹配）
        path: 保存路径
    """
    exported = torch.load(path)
    load_int2_weights_to_model(model, exported)
    print(f"INT2 模型已从 {path} 加载")


# ============================================================================
# 训练保存钩子
# ============================================================================

class Int2SaveHook:
    """训练保存钩子：同时保存 FP32 和 INT2 模型
    
    使用方法：
        hook = Int2SaveHook(model, save_dir='checkpoints')
        
        # 训练循环中
        for epoch in range(epochs):
            train(...)
            hook.save(epoch=epoch, metrics={'loss': loss})
    """
    
    def __init__(
        self, 
        model: nn.Module,
        save_dir: str = 'checkpoints',
        save_fp32: bool = True,
        save_int2: bool = True,
        save_every: int = 1,  # 每 N 个 epoch 保存
    ):
        self.model = model
        self.save_dir = save_dir
        self.save_fp32 = save_fp32
        self.save_int2 = save_int2
        self.save_every = save_every
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self._epoch = 0
    
    def save(self, epoch: int = None, metrics: Dict = None, force: bool = False):
        """保存模型
        
        Args:
            epoch: 当前 epoch
            metrics: 额外的指标
            force: 强制保存（忽略 save_every）
        """
        if epoch is not None:
            self._epoch = epoch
        
        # 检查是否需要保存
        if not force and self._epoch % self.save_every != 0:
            return
        
        import os
        import json
        
        # 保存指标
        if metrics:
            metrics_path = os.path.join(self.save_dir, f'metrics_epoch_{self._epoch}.json')
            with open(metrics_path, 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        
        # 保存 FP32 模型
        if self.save_fp32:
            fp32_path = os.path.join(self.save_dir, f'model_fp32_epoch_{self._epoch}.pt')
            torch.save(self.model.state_dict(), fp32_path)
            print(f"[SaveHook] FP32 模型已保存: {fp32_path}")
        
        # 保存 INT2 模型
        if self.save_int2:
            int2_path = os.path.join(self.save_dir, f'model_int2_epoch_{self._epoch}.pt')
            save_int2_model(self.model, int2_path)
    
    def save_best(self, metrics: Dict, best_metric: str = 'loss', lower_is_better: bool = True):
        """保存最佳模型
        
        Args:
            metrics: 指标字典
            best_metric: 用于判断的指标名
            lower_is_better: 是否越低越好
        """
        import os
        
        # 加载历史最佳
        best_path = os.path.join(self.save_dir, 'best_metrics.json')
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                import json
                best = json.load(f)
        else:
            best = {best_metric: float('inf') if lower_is_better else float('-inf')}
        
        current = metrics.get(best_metric, float('inf') if lower_is_better else float('-inf'))
        
        # 判断是否更好
        is_better = (current < best[best_metric]) if lower_is_better else (current > best[best_metric])
        
        if is_better:
            print(f"[SaveHook] 新最佳模型! {best_metric}: {current:.4f} (之前: {best[best_metric]:.4f})")
            
            # 保存最佳指标
            import json
            with open(best_path, 'w') as f:
                json.dump({**metrics, 'epoch': self._epoch}, f, indent=2)
            
            # 保存模型
            if self.save_fp32:
                fp32_path = os.path.join(self.save_dir, 'model_fp32_best.pt')
                torch.save(self.model.state_dict(), fp32_path)
            
            if self.save_int2:
                int2_path = os.path.join(self.save_dir, 'model_int2_best.pt')
                save_int2_model(self.model, int2_path)


def convert_checkpoint_to_int2(checkpoint_path: str, output_path: str, model_class=None):
    """将训练好的 FP32 checkpoint 转换为 INT2 格式
    
    Args:
        checkpoint_path: FP32 checkpoint 路径
        output_path: INT2 输出路径
        model_class: 模型类（如果需要构建模型结构）
    """
    if model_class is not None:
        # 加载完整模型
        model = model_class()
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        save_int2_model(model, output_path)
    else:
        # 直接转换权重
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        from tesm_ssm.modules.tesm import BitLinear
        
        exported = {}
        for name, tensor in state_dict.items():
            if 'weight' in name and tensor.dim() == 2:
                # 可能是 BitLinear 权重
                packed, scale = quantize_weight_to_int2(tensor)
                base_name = name.replace('.weight', '')
                
                # 查找对应的 bias
                bias_key = name.replace('.weight', '.bias')
                bias = state_dict.get(bias_key, None)
                
                exported[base_name] = {
                    'packed_weight': packed,
                    'weight_scale': scale,
                    'bias': bias,
                    'in_features': tensor.shape[1],
                    'out_features': tensor.shape[0],
                }
        
        torch.save(exported, output_path)
        print(f"INT2 模型已保存到 {output_path}")


# ============================================================================
# INT2 推理模型
# ============================================================================

class Int2Model(nn.Module):
    """INT2 量化推理模型
    
    将普通模型的 BitLinear 层替换为 Int2Linear 进行高效推理。
    """
    
    def __init__(self, original_model: nn.Module = None, exported_weights: Dict = None):
        super().__init__()
        
        if original_model is not None:
            self.model = self._convert_model(original_model)
        elif exported_weights is not None:
            self.model = self._build_from_exported(exported_weights)
        else:
            raise ValueError("需要提供 original_model 或 exported_weights")
    
    def _convert_model(self, model: nn.Module) -> nn.Module:
        """将模型的 BitLinear 替换为 Int2Linear"""
        from tesm_ssm.modules.tesm import BitLinear
        
        # 深拷贝模型
        import copy
        converted = copy.deepcopy(model)
        
        # 替换层
        for name, module in converted.named_modules():
            if isinstance(module, BitLinear):
                int2_linear = Int2Linear.from_float(module)
                # 替换
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = converted
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                setattr(parent, child_name, int2_linear)
        
        return converted
    
    def _build_from_exported(self, exported: Dict) -> nn.Module:
        """从导出的权重构建模型"""
        # 这里需要知道模型结构，简化实现
        raise NotImplementedError("请使用 original_model 参数")
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def save(self, path: str):
        """保存 INT2 模型"""
        save_int2_model(self.model, path)
    
    @classmethod
    def load(cls, path: str, model_template: nn.Module):
        """加载 INT2 模型
        
        Args:
            path: INT2 模型路径
            model_template: 模型模板（用于获取结构）
        """
        exported = torch.load(path, map_location='cpu')
        
        # 创建模板副本
        import copy
        model = copy.deepcopy(model_template)
        load_int2_weights_to_model(model, exported)
        
        return cls(original_model=model)
    
    def get_model_size_mb(self) -> float:
        """获取模型大小（MB）"""
        total_bytes = 0
        for name, param in self.named_parameters():
            total_bytes += param.numel() * param.element_size()
        for name, buf in self.named_buffers():
            total_bytes += buf.numel() * buf.element_size()
        return total_bytes / (1024 * 1024)
