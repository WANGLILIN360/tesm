import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tesm_ssm.ops.cuda import (
        cuda_chunk_state_scan,
        cuda_chunk_state_scan_autograd,
        cuda_local_entanglement,
        cuda_local_entanglement_autograd,
        cuda_quantized_linear,
        cuda_quantized_linear_autograd,
        tesm_cuda_is_available,
        # Global Entanglement
        cuda_global_entanglement,
        cuda_global_entanglement_autograd,
        cuda_global_entanglement_mimo,
        cuda_global_entanglement_mimo_autograd,
        # Fused Output
        cuda_fused_output,
        cuda_fused_output_autograd,
        cuda_fused_output_mimo,
        cuda_fused_output_mimo_autograd,
        # MIMO kernels
        cuda_chunk_state_scan_mimo,
        cuda_chunk_state_scan_mimo_autograd,
        cuda_local_entanglement_mimo,
        cuda_local_entanglement_mimo_autograd,
    )
except Exception:
    cuda_chunk_state_scan = None
    cuda_chunk_state_scan_autograd = None
    cuda_local_entanglement = None
    cuda_local_entanglement_autograd = None
    cuda_quantized_linear = None
    cuda_quantized_linear_autograd = None
    tesm_cuda_is_available = lambda: False
    # Global Entanglement
    cuda_global_entanglement = None
    cuda_global_entanglement_autograd = None
    cuda_global_entanglement_mimo = None
    cuda_global_entanglement_mimo_autograd = None
    # Fused Output
    cuda_fused_output = None
    cuda_fused_output_autograd = None
    cuda_fused_output_mimo = None
    cuda_fused_output_mimo_autograd = None
    # MIMO kernels
    cuda_chunk_state_scan_mimo = None
    cuda_chunk_state_scan_mimo_autograd = None
    cuda_local_entanglement_mimo = None
    cuda_local_entanglement_mimo_autograd = None

try:
    from tesm_ssm.ops.triton import (
        tesm_triton_is_available,
        triton_chunk_state_scan,
        triton_fused_output_combine,
        triton_local_entanglement,
        triton_quantized_linear,
        triton_global_entanglement,
        # Autograd versions for training
        triton_chunk_state_scan_autograd,
        triton_quantized_linear_autograd,
        triton_local_entanglement_autograd,
        triton_fused_output_combine_autograd,
        triton_global_entanglement_autograd,
    )
except Exception:
    tesm_triton_is_available = lambda: False
    triton_chunk_state_scan = None
    triton_fused_output_combine = None
    triton_local_entanglement = None
    triton_quantized_linear = None
    triton_global_entanglement = None
    triton_chunk_state_scan_autograd = None
    triton_quantized_linear_autograd = None
    triton_local_entanglement_autograd = None
    triton_fused_output_combine_autograd = None
    triton_global_entanglement_autograd = None

try:
    from tesm_ssm.ops.tilelang import (
        tesm_chunked_scan_tilelang_fwd,
        tesm_local_entanglement_tilelang_fwd,
        tesm_bitlinear_tilelang,
        # Autograd versions for training
        tesm_chunked_scan_tilelang_autograd,
        tesm_local_entanglement_tilelang_autograd,
        # BitLinear
        tesm_bitlinear_tilelang_autograd,
        # Global Entanglement
        tesm_global_entanglement_tilelang_autograd,
        # Fused Output
        tesm_fused_output_tilelang_autograd,
    )
    TILELANG_AVAILABLE = True
except Exception:
    tesm_chunked_scan_tilelang_fwd = None
    tesm_local_entanglement_tilelang_fwd = None
    tesm_bitlinear_tilelang = None
    tesm_chunked_scan_tilelang_autograd = None
    tesm_local_entanglement_tilelang_autograd = None
    tesm_bitlinear_tilelang_autograd = None
    tesm_global_entanglement_tilelang_autograd = None
    tesm_fused_output_tilelang_autograd = None
    TILELANG_AVAILABLE = False


_OFFICIAL_BITNET_GPU_PATH = Path(__file__).resolve().parents[3] / "BitNet-main" / "gpu"
if _OFFICIAL_BITNET_GPU_PATH.exists() and str(_OFFICIAL_BITNET_GPU_PATH) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_BITNET_GPU_PATH))

try:
    from pack_weight import convert_weight_int8_to_int2 as _official_convert_weight_int8_to_int2
except ImportError:
    _official_convert_weight_int8_to_int2 = None


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, bit_eps=1e-5, bit_threshold=0.5, kernel_backend="auto", kernel_mode="fast", device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.bit_eps = bit_eps
        self.bit_threshold = bit_threshold
        self.kernel_backend = str(kernel_backend)
        self.kernel_mode = "precise" if str(kernel_mode) == "precise" else "fast"
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.bias = None
        self._cached_qweight = None
        self._cached_weight_version = -1
        self.reset_parameters()

    def _can_use_cuda_ext_eval_path(self, x: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and tesm_cuda_is_available()
            and cuda_quantized_linear is not None
            and x.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
        )

    def _can_use_cuda_ext_training_path(self, x: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and tesm_cuda_is_available()
            and cuda_quantized_linear_autograd is not None
            and x.is_cuda
            and torch.is_grad_enabled()
        )

    def _can_use_triton_eval_path(self, x: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "triton"}
            and tesm_triton_is_available()
            and triton_quantized_linear is not None
            and x.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
        )

    def _can_use_triton_training_path(self, x: torch.Tensor) -> bool:
        """Check if Triton autograd kernels can be used during training."""
        return (
            self.kernel_backend in {"auto", "triton"}
            and tesm_triton_is_available()
            and triton_quantized_linear_autograd is not None
            and x.is_cuda
            and torch.is_grad_enabled()
        )

    def _can_use_tilelang_training_path(self, x: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tesm_bitlinear_tilelang_autograd is not None
            and x.is_cuda
            and torch.is_grad_enabled()
        )

    def _can_use_tilelang_fast_path(self, x: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tesm_bitlinear_tilelang is not None
            and x.is_cuda
            and not self.training
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantized_input(self, x):
        scale = 127 / x.detach().abs().max(dim=-1, keepdim=True).values.clamp_min(self.bit_eps)
        quantized = (x * scale).round().clamp(-128, 127) / scale
        return x + (quantized - x).detach()

    def quantized_weight(self):
        scale = 1.0 / self.weight.detach().abs().mean().clamp_min(self.bit_eps)
        normalized = self.weight * scale
        quantized = normalized.round().clamp(-1, 1)
        return (normalized + (quantized - normalized).detach()) / scale

    def export_bitnet_weights(self):
        if _official_convert_weight_int8_to_int2 is None:
            raise ImportError("Official BitNet pack_weight.py is not available")
        scale = 1.0 / self.weight.detach().abs().mean().clamp_min(self.bit_eps)
        quantized = (self.weight.detach() * scale).round().clamp(-1, 1).to(torch.int8).cpu()
        packed = _official_convert_weight_int8_to_int2(quantized)
        weight_scale = (1.0 / scale).to(torch.bfloat16).reshape(1)
        return packed, weight_scale

    def _get_eval_quantized_weight(self):
        weight_version = self.weight._version
        if self._cached_qweight is None or self._cached_weight_version != weight_version:
            self._cached_qweight = self.quantized_weight()
            self._cached_weight_version = weight_version
        return self._cached_qweight

    def _current_quantized_weight(self):
        if not self.training and not torch.is_grad_enabled():
            return self._get_eval_quantized_weight()
        return self.quantized_weight()

    def _project_quantized(self, qinput: torch.Tensor, qweight: torch.Tensor, bias: torch.Tensor | None = None):
        # 1. CUDA training path
        if self._can_use_cuda_ext_training_path(qinput) and cuda_quantized_linear_autograd is not None:
            return cuda_quantized_linear_autograd(qinput, qweight, bias)
        
        # 2. CUDA inference path
        if self._can_use_cuda_ext_eval_path(qinput) and cuda_quantized_linear is not None:
            return cuda_quantized_linear(qinput, qweight, bias)
        
        # 3. Triton training path
        if self._can_use_triton_training_path(qinput) and triton_quantized_linear_autograd is not None:
            return triton_quantized_linear_autograd(qinput, qweight, bias, precision_mode=self.kernel_mode)
        
        # 4. Triton inference path
        if self._can_use_triton_eval_path(qinput) and triton_quantized_linear is not None:
            return triton_quantized_linear(qinput, qweight, bias, precision_mode=self.kernel_mode)
        
        # 5. TileLang training path
        if self._can_use_tilelang_training_path(qinput) and tesm_bitlinear_tilelang_autograd is not None:
            # TileLang BitLinear requires scale parameter
            # Handle 3D input (batch, seq, hidden) -> 2D (batch*seq, hidden)
            original_shape = qinput.shape[:-1]
            qinput_2d = qinput.reshape(-1, qinput.shape[-1])
            scale = qweight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            out = tesm_bitlinear_tilelang_autograd(qinput_2d, qweight, scale.squeeze(1))
            return out.reshape(*original_shape, -1)
        
        # 6. TileLang inference path
        if self._can_use_tilelang_fast_path(qinput) and tesm_bitlinear_tilelang is not None:
            # Handle 3D input (batch, seq, hidden) -> 2D (batch*seq, hidden)
            original_shape = qinput.shape[:-1]
            qinput_2d = qinput.reshape(-1, qinput.shape[-1])
            scale = qweight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            out = tesm_bitlinear_tilelang(qinput_2d, qweight, scale.squeeze(1))
            return out.reshape(*original_shape, -1)
        
        # 7. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            return F.linear(qinput, qweight, bias)
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but BitLinear kernel not available. "
            f"Available backends: cuda={cuda_quantized_linear_autograd is not None}, "
            f"triton={triton_quantized_linear is not None}, "
            f"tilelang={tesm_bitlinear_tilelang_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )

    def forward(self, x):
        qweight = self._current_quantized_weight()
        qinput = self.quantized_input(x)
        return self._project_quantized(qinput, qweight, self.bias)


class TernaryQuantumTunneling(nn.Module):
    """三值量子隧穿模块
    
    作用于三值纠缠决策过程：
    - 在阈值边界附近，允许分数"隧穿"到不同的三值状态
    - 模拟量子隧穿穿过能量势垒的效应
    
    物理原理启发：
    - 经典：分数|score| < threshold → 三值=0
    - 量子隧穿：边界分数有机会"穿越"阈值变成 ±1
    - 势垒高度 = |threshold - |score||
    - 隧穿概率 = exp(-barrier / strength)
    """
    def __init__(
        self,
        threshold: float = 0.1,
        tunneling_strength: float = 0.1,
        num_tunnel_paths: int = 4,
        min_tunnel_prob: float = 0.05,
        max_tunnel_prob: float = 0.5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.threshold = threshold
        self.tunneling_strength = tunneling_strength
        self.num_tunnel_paths = num_tunnel_paths
        self.min_tunnel_prob = min_tunnel_prob
        self.max_tunnel_prob = max_tunnel_prob
        
        # 可学习的隧穿强度调节因子
        self.tunnel_scale = nn.Parameter(
            torch.ones(1, device=device, dtype=torch.float32)
        )
        
        # 隧穿统计
        self.register_buffer('tunnel_step', torch.tensor(0))
        self.register_buffer('tunnel_to_positive', torch.tensor(0))  # 隧穿到+1
        self.register_buffer('tunnel_to_negative', torch.tensor(0))  # 隧穿到-1
        self.register_buffer('tunnel_to_zero', torch.tensor(0))       # 隧穿到0
        self.register_buffer('total_boundary', torch.tensor(0))       # 边界区域总数
    
    def compute_barrier_height(self, scores: torch.Tensor) -> torch.Tensor:
        """计算势垒高度
        
        势垒 = 阈值 - |分数|（边界分数离阈值的距离）
        距离越小（越接近阈值），势垒越低，隧穿概率越高
        """
        distance_to_threshold = (self.threshold - scores.abs()).clamp(min=0)
        return distance_to_threshold
    
    def get_tunneling_probability(self, barrier_height: torch.Tensor) -> torch.Tensor:
        """计算隧穿概率
        
        量子启发：P = exp(-barrier / strength)
        势垒越低（越接近阈值），隧穿概率越高
        """
        strength = self.tunneling_strength * self.tunnel_scale.abs() + 1e-6
        prob = torch.exp(-barrier_height / strength)
        return prob.clamp(self.min_tunnel_prob, self.max_tunnel_prob)
    
    def apply_tunneling(self, scores: torch.Tensor, training: bool = True) -> tuple:
        """对三值纠缠分数应用量子隧穿
        
        Args:
            scores: (B, L, R) 或 (B, L, D) 纠缠分数
            training: 是否训练模式
            
        Returns:
            ternary_values: 隧穿后的三值 {-1, 0, +1}
            tunnel_info: 隧穿统计信息
        """
        original_shape = scores.shape
        scores_flat = scores.reshape(-1)  # 展平
        
        # 计算原始三值决策
        original_ternary = torch.where(
            scores_flat > self.threshold, torch.ones_like(scores_flat),
            torch.where(scores_flat < -self.threshold, -torch.ones_like(scores_flat),
                       torch.zeros_like(scores_flat))
        )
        
        # 识别边界区域（可能隧穿的区域）
        boundary_mask = scores_flat.abs() < self.threshold * 1.5  # 边界扩展
        
        # 计算势垒高度
        barrier_height = self.compute_barrier_height(scores_flat)
        
        # 计算隧穿概率
        tunnel_prob = self.get_tunneling_probability(barrier_height)
        
        # 决定是否隧穿
        if training:
            # 训练时：随机采样
            tunnel_mask = (torch.rand_like(scores_flat) < tunnel_prob) & boundary_mask
        else:
            # 推理时：只对高概率隧穿点执行
            tunnel_mask = (tunnel_prob > 0.3) & boundary_mask
        
        # 隧穿后的三值决策
        # 边界分数可以隧穿到 ±1 或保持 0
        # 使用分数的符号决定隧穿方向
        tunneled_ternary = original_ternary.clone()
        
        # 隧穿决策：根据分数值和随机性决定目标三值
        boundary_indices = tunnel_mask.nonzero().squeeze(-1)
        
        if boundary_indices.numel() > 0:
            boundary_scores = scores_flat[boundary_indices]
            
            # 隧穿目标：根据分数符号和幅度
            # 正分数 → 可能隧穿到 +1
            # 负分数 → 可能隧穿到 -1
            # 接近0 → 保持 0
            
            tunnel_targets = torch.zeros_like(boundary_scores)
            
            # 正分数边界：隧穿到 +1
            pos_mask = boundary_scores > 0
            tunnel_targets[pos_mask] = 1.0
            
            # 负分数边界：隧穿到 -1
            neg_mask = boundary_scores < 0
            tunnel_targets[neg_mask] = -1.0
            
            # 接近0的保持0
            near_zero_mask = boundary_scores.abs() < self.threshold * 0.3
            tunnel_targets[near_zero_mask] = 0.0
            
            tunneled_ternary[boundary_indices] = tunnel_targets
        
        
        # 更新统计
        if training:
            self.tunnel_step.add_(1)
            self.total_boundary.add_(boundary_mask.sum().item())
            
            # 统计隧穿方向
            tunneled_diff = tunneled_ternary - original_ternary
            self.tunnel_to_positive.add_((tunneled_diff == 1).sum().item())
            self.tunnel_to_negative.add_((tunneled_diff == -1).sum().item())
            self.tunnel_to_zero.add_((tunneled_diff == 0).sum().item())
        
        
        # 计算隧穿率
        tunnel_rate = tunnel_mask.float().mean().item()
        boundary_rate = boundary_mask.float().mean().item()
        
        tunnel_info = {
            'tunnel_rate': tunnel_rate,
            'boundary_rate': boundary_rate,
            'avg_tunnel_prob': tunnel_prob[boundary_mask].mean().item() if boundary_mask.any() else 0,
        }
        
        ternary_values = tunneled_ternary.reshape(original_shape)
        
        return ternary_values, tunnel_info


# 兼容旧名称
QuantumTunneling = TernaryQuantumTunneling


class TESM_SISO(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=256,
        expand=2,
        ent_rank=64,
        entanglement_scale=0.2,
        entanglement_threshold=0.1,
        max_seq_len=2048,
        dropout=0.0,
        bit_eps=1e-5,
        bit_threshold=0.5,
        layer_idx=None,
        device=None,
        dtype=None,
        # 温度退火参数（原量子退火，重命名以准确描述）
        annealing_enabled=True,
        T_start=10.0,
        T_end=0.1,
        annealing_steps=1000,
        annealing_schedule='cosine',
        # 量子隧穿启发参数
        quantum_tunneling_enabled=False,
        tunneling_strength=0.1,
        num_tunnel_paths=4,
        energy_landscape="entropy",
        tunneling_schedule="adaptive",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # 参数验证
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if ent_rank <= 0:
            raise ValueError(f"ent_rank must be positive, got {ent_rank}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.ent_rank = ent_rank
        self.entanglement_scale = entanglement_scale
        self.entanglement_threshold = entanglement_threshold
        self.entanglement_init = kwargs.get('entanglement_init', 0.3)
        self.entanglement_window = int(kwargs.get("entanglement_window", 0) or 0)
        self.entanglement_block_size = int(kwargs.get("entanglement_block_size", 256) or 256)
        self.state_scan_chunk_size = int(kwargs.get("state_scan_chunk_size", 16) or 16)
        self.use_triton_kernels = bool(kwargs.get("use_triton_kernels", True))
        self.kernel_backend = str(kwargs.get("kernel_backend", "auto"))
        self.kernel_mode = "precise" if str(kwargs.get("kernel_mode", "fast")) == "precise" else "fast"
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        
        # 温度退火参数（原量子退火）
        self.annealing_enabled = annealing_enabled
        self.T_start = T_start
        self.T_end = T_end
        self.annealing_steps = annealing_steps
        self.annealing_schedule = annealing_schedule
        self.register_buffer('annealing_step', torch.tensor(0))
        
        # 量子隧穿启发模块
        self.quantum_tunneling_enabled = quantum_tunneling_enabled
        if quantum_tunneling_enabled:
            self.quantum_tunneler = TernaryQuantumTunneling(
                threshold=entanglement_threshold,
                tunneling_strength=tunneling_strength,
                num_tunnel_paths=num_tunnel_paths,
                device=device,
                dtype=dtype,
            )
        else:
            self.quantum_tunneler = None
        total_proj = (2 * d_model) + (3 * d_state) + (2 * ent_rank)
        self.in_proj = BitLinear(d_model, total_proj, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.state_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.ent_proj = BitLinear(d_state, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.out_proj = BitLinear(d_model, d_model, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        # 修复1: 衰减偏置 - 初始化为正值使 sigmoid(raw+bias) 接近1，状态持续更久
        decay_init = float(kwargs.get('decay_init_bias', 3.0))
        self.decay_bias = nn.Parameter(torch.full((d_state,), decay_init, device=device, dtype=torch.float32))
        if self.entanglement_window > 0:
            self.register_parameter("entanglement", None)
            self.local_entanglement_bias = nn.Parameter(
                torch.randn(self.entanglement_window, device=device, dtype=torch.float32) * self.entanglement_init
            )
            local_positions = torch.arange(max_seq_len, device=device).unsqueeze(1) - (self.entanglement_window - 1) + torch.arange(self.entanglement_window, device=device).unsqueeze(0)
            self.register_buffer("local_window_valid_mask", local_positions >= 0, persistent=False)
            # 全局纠缠：使用相对位置偏置（线性参数）
            self.register_parameter("global_entanglement_bias", None)
            self.register_buffer("global_relative_positions", None, persistent=False)
        else:
            self.register_parameter("local_entanglement_bias", None)
            # 真全局纠缠：使用可学习的相对位置编码函数，不受 max_seq_len 限制
            # 方案：使用 RoPE 风格的相对位置编码，参数量固定且可泛化到任意长度
            self.global_rel_pos_scale = nn.Parameter(
                torch.ones(1, device=device, dtype=torch.float32)  # 可学习的缩放因子
            )
            self.global_rel_pos_bias = nn.Parameter(
                torch.zeros(1, device=device, dtype=torch.float32)  # 可学习的偏置
            )
            # 相对位置编码维度（可配置）
            self.global_rel_pos_dim = int(kwargs.get('global_rel_pos_dim', 64))
            # 可学习的相对位置嵌入（用于生成连续偏置函数）
            self.global_rel_pos_embed = nn.Parameter(
                torch.randn(self.global_rel_pos_dim, device=device, dtype=torch.float32) * 0.02
            )
            # 不再预计算固定大小的索引矩阵
            self.register_parameter("global_entanglement_bias", None)  # 弃用固定参数表
            self.register_buffer("global_relative_positions", None, persistent=False)
            self.register_parameter("entanglement", None)
            self.register_buffer("local_window_valid_mask", None, persistent=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)), persistent=False)
        # 位置纠缠: RoPE 使纠缠模式随位置变化（无额外参数，运行时计算）
        self.rope_base = float(kwargs.get('rope_base', 10000.0))
        # 跨层纠缠: 上层状态摘要偏置当前层的纠缠 Q
        self.cross_layer_q_proj = BitLinear(d_state, ent_rank, bias=False, bit_eps=bit_eps, bit_threshold=bit_threshold, kernel_backend=self.kernel_backend, kernel_mode=self.kernel_mode, device=device, dtype=dtype)
        self.last_entanglement_map = None
        self.last_entanglement_stats = None
        self._last_cross_layer_state = None  # 供 MixerModel 读取
        self._last_scan_state_f64 = None
        # 使用 register_buffer 持久化统计，避免被 gradient checkpointing 重置
        self.register_buffer("_stats_ternary_buffer", None, persistent=False)
        self.register_buffer("_stats_total_buffer", torch.tensor(1.0), persistent=False)
        self._debug_printed = False  # 调试标志

    def get_temperature(self):
        """获取当前温度 (温度退火调度)
        
        训练模式: 按调度从高温→低温
        推理模式: 固定低温 (确定性三值纠缠)
        """
        # 推理模式: 固定低温，使用确定性三值纠缠
        if not self.training:
            return self.T_end
        
        if not self.annealing_enabled:
            return self.T_end
        
        progress = min(self.annealing_step.item() / self.annealing_steps, 1.0)
        
        if self.annealing_schedule == 'linear':
            T = self.T_start * (1 - progress) + self.T_end * progress
        elif self.annealing_schedule == 'exponential':
            import math
            decay_rate = math.log(self.T_start / max(self.T_end, 1e-6))
            T = self.T_start * math.exp(-progress * decay_rate)
        elif self.annealing_schedule == 'cosine':
            import math
            T = self.T_end + 0.5 * (self.T_start - self.T_end) * (1 + math.cos(math.pi * progress))
        else:
            import warnings
            warnings.warn(f"Unknown annealing_schedule '{self.annealing_schedule}', "
                         f"falling back to constant T_start={self.T_start}. "
                         f"Valid options: 'linear', 'exponential', 'cosine'")
            T = self.T_start
        
        return max(T, self.T_end)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """分配增量推理的状态缓存
        
        Args:
            batch_size: 批次大小
            max_seqlen: 最大序列长度
            dtype: 数据类型
            use_paged_cache: 是否使用分页缓存（支持超长上下文）
            page_size: 分页缓存页大小
            max_gpu_pages: GPU最多存储的页数
        """
        dev = self.out_proj.weight.device
        _dtype = dtype or torch.float32
        window = max(self.entanglement_window, 1)
        
        # 检查是否使用分页缓存
        use_paged = kwargs.get('use_paged_cache', False)
        
        if use_paged and max_seqlen > 1024:
            # 使用分页缓存支持超长上下文
            from tesm_ssm.utils.paged_cache import PagedStateCache
            return {
                'use_paged': True,
                'paged_cache': PagedStateCache(
                    batch_size=batch_size,
                    d_state=self.d_state,
                    ent_rank=self.ent_rank,
                    window=window,
                    page_size=kwargs.get('page_size', 512),
                    max_gpu_pages=kwargs.get('max_gpu_pages', 100),
                    device=dev,
                ),
                # 当前活跃状态（用于增量推理）
                'state': torch.zeros(batch_size, self.d_state, device=dev, dtype=torch.float64),
                'seq_pos': 0,
                'ent_k_cache': torch.zeros(batch_size, window, self.ent_rank, device=dev, dtype=_dtype),
                'ent_v_cache': torch.zeros(batch_size, window, self.d_state, device=dev, dtype=_dtype),
            }
        
        # 传统缓存（添加循环缓冲区索引）
        return {
            'use_paged': False,
            'state': torch.zeros(batch_size, self.d_state, device=dev, dtype=torch.float64),
            'seq_pos': 0,
            # 纠缠滑动窗口缓存
            'ent_k_cache': torch.zeros(batch_size, window, self.ent_rank, device=dev, dtype=_dtype),
            'ent_v_cache': torch.zeros(batch_size, window, self.d_state, device=dev, dtype=_dtype),
            # 循环缓冲区索引
            'cache_idx': 0,  # 当前写入位置
            'cache_filled': False,  # 缓存是否已填满
        }

    def ternary_entanglement(self, scores):
        """三值纠缠 - 支持温度退火调度 + 量子隧穿
        
        高温 (T > 1.0): 密集矩阵纠缠 (softmax)
        低温 (T <= 1.0): 硬化阈值纠缠 {-1, 0, +1} + 量子隧穿
        """
        T = self.get_temperature()
        
        if T > 1.0:
            # 高温: 密集矩阵纠缠 (softmax 平滑)
            import torch.nn.functional as F
            weights = F.softmax(scores / T, dim=-1)
            # 高温阶段：返回softmax权重用于计算，但统计用硬阈值
            # 存储硬阈值统计到buffer（供日志使用）
            with torch.no_grad():
                hard = torch.where(
                    scores > self.entanglement_threshold,
                    torch.ones_like(scores),
                    torch.where(scores < -self.entanglement_threshold, -torch.ones_like(scores), torch.zeros_like(scores))
                )
                # 临时存储硬阈值统计
                self._high_temp_ternary_stats = hard.detach()
            return weights
        else:
            # 低温: 硬化阈值纠缠
            hard = torch.where(
                scores > self.entanglement_threshold,
                torch.ones_like(scores),
                torch.where(scores < -self.entanglement_threshold, -torch.ones_like(scores), torch.zeros_like(scores)),
            )
            
            # 量子隧穿：边界分数可隧穿到不同三值
            if self.quantum_tunneling_enabled and self.quantum_tunneler is not None:
                tunneled_ternary, tunnel_info = self.quantum_tunneler.apply_tunneling(
                    scores, training=self.training
                )
                # 混合原始硬阈值和隧穿结果
                # 隧穿只影响边界区域，非边界保持原值
                boundary_mask = scores.abs() < self.entanglement_threshold * 1.5
                result = torch.where(boundary_mask, tunneled_ternary, hard)
                return scores + (result - scores).detach()
            
            return scores + (hard - scores).detach()

    def _update_entanglement_stats(self, ternary):
        # 使用 register_buffer 持久化统计，避免被 gradient checkpointing 重置
        if self.training:
            self.last_entanglement_map = None
            ternary_detached = ternary.detach()
            
            # 高温阶段：使用硬阈值统计而非softmax值
            if hasattr(self, '_high_temp_ternary_stats') and self._high_temp_ternary_stats is not None:
                ternary_detached = self._high_temp_ternary_stats
                self._high_temp_ternary_stats = None  # 清空临时统计
            
            total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
            # 存储到 buffer
            self._stats_ternary_buffer = ternary_detached
            self._stats_total_buffer = torch.tensor(total, device=ternary.device)
            self._ternary_stats_for_logging = (ternary_detached, total)
            return
        ternary_detached = ternary.detach()
        
        # 高温阶段：使用硬阈值统计
        if hasattr(self, '_high_temp_ternary_stats') and self._high_temp_ternary_stats is not None:
            ternary_detached = self._high_temp_ternary_stats
            self._high_temp_ternary_stats = None
            
        total = float(ternary_detached.numel()) if ternary_detached.numel() > 0 else 1.0
        self.last_entanglement_map = ternary_detached
        self._stats_ternary_buffer = ternary_detached
        self._stats_total_buffer = torch.tensor(total, device=ternary.device)
        self._ternary_stats_for_logging = (ternary_detached, total)

    def _update_local_entanglement_stats_sample(self, q, k):
        with torch.no_grad():
            _, seq_len, _ = q.shape
            window = min(self.entanglement_window, seq_len)
            if window <= 0:
                self._update_entanglement_stats(q.new_zeros((1, 1, 1)))
                return
            q_sample = q[:1, : min(seq_len, 64), :].detach()
            sample_len = q_sample.size(1)
            k_sample = k[:1, :sample_len, :].detach()
            relative_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype).view(1, 1, window)
            window_offsets = torch.arange(window, device=q.device)
            positions = torch.arange(sample_len, device=q.device)
            window_positions = positions.unsqueeze(1) - (window - 1) + window_offsets.unsqueeze(0)
            gather_positions = window_positions.clamp(min=0, max=sample_len - 1)
            valid_mask = window_positions >= 0
            k_block = k_sample[:, gather_positions, :]
            scores = torch.einsum("bqr,bqwr->bqw", q_sample, k_block) / math.sqrt(self.ent_rank)
            local_bias = relative_bias[:, :, -window:].masked_fill(~valid_mask.unsqueeze(0), 0.0)
            scores = scores + local_bias
            
            # 调试：打印scores分布
            if self.training and hasattr(self, '_debug_printed') and not self._debug_printed:
                self._debug_printed = True
                print(f"[DEBUG] scores: min={scores.min().item():.4f} max={scores.max().item():.4f} mean={scores.mean().item():.4f}")
                print(f"[DEBUG] local_bias: min={local_bias.min().item():.4f} max={local_bias.max().item():.4f} mean={local_bias.mean().item():.4f}")
                print(f"[DEBUG] threshold={self.entanglement_threshold}")
            
            # 统计始终用硬阈值（反映最终低温目标），不受温度退火影响
            ternary = torch.where(
                scores > self.entanglement_threshold,
                torch.ones_like(scores),
                torch.where(scores < -self.entanglement_threshold, -torch.ones_like(scores), torch.zeros_like(scores))
            )
            ternary = ternary.masked_fill(~valid_mask.unsqueeze(0), 0.0)
            self._update_entanglement_stats(ternary)

    def _apply_rope(self, x, pos_offset=0):
        """位置纠缠: 对 ent_q/ent_k 施加 RoPE，使纠缠模式随位置变化"""
        B, L, D = x.shape
        half = D // 2
        pos = torch.arange(pos_offset, pos_offset + L, device=x.device, dtype=torch.float32)
        dim_idx = torch.arange(half, device=x.device, dtype=torch.float32)
        theta = pos.unsqueeze(1) * (1.0 / (self.rope_base ** (2.0 * dim_idx / D)))  # (L, half)
        cos_t = theta.cos().unsqueeze(0).to(x.dtype)  # (1, L, half)
        sin_t = theta.sin().unsqueeze(0).to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)

    def _can_use_triton_fast_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend in {"auto", "triton"}
            and self.use_triton_kernels
            and tesm_triton_is_available()
            and tensor.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
        )

    def _can_use_cuda_ext_fast_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and tesm_cuda_is_available()
            and tensor.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
        )

    def _can_use_cuda_ext_training_path(self, tensor: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "cuda"
            and tesm_cuda_is_available()
            and tensor.is_cuda
            and torch.is_grad_enabled()
        )

    def _can_use_triton_training_path(self, tensor: torch.Tensor) -> bool:
        """Check if Triton autograd kernels can be used during training."""
        return (
            self.kernel_backend in {"auto", "triton"}
            and self.use_triton_kernels
            and tesm_triton_is_available()
            and tensor.is_cuda
            and torch.is_grad_enabled()
            and triton_chunk_state_scan_autograd is not None
        )

    def _can_use_tilelang_training_path(self, tensor: torch.Tensor) -> bool:
        """Check if TileLang autograd kernels can be used during training."""
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tesm_chunked_scan_tilelang_autograd is not None
            and tensor.is_cuda
            and torch.is_grad_enabled()
        )

    def _can_use_tilelang_fast_path(self, tensor: torch.Tensor) -> bool:
        """Check if TileLang inference kernels can be used."""
        return (
            self.kernel_backend in {"auto", "tilelang"}
            and TILELANG_AVAILABLE
            and tesm_chunked_scan_tilelang_fwd is not None
            and tensor.is_cuda
            and not torch.is_grad_enabled()
        )

    def _compute_local_entanglement(self, q, k, values):
        _, seq_len, _ = q.shape
        window = min(self.entanglement_window, seq_len)
        
        # 1. TileLang training path
        if self._can_use_tilelang_training_path(q) and tesm_local_entanglement_tilelang_autograd is not None:
            B, L, R = q.shape
            D = values.shape[-1]
            q_4d = q.unsqueeze(2).contiguous()  # (B, L, 1, R)
            k_4d = k.unsqueeze(2).contiguous()  # (B, L, 1, R)
            v_4d = values.unsqueeze(2).contiguous()  # (B, L, 1, D)
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype)
            local_bias_2d = local_bias.unsqueeze(0).contiguous()  # (1, W)
            entangled_4d = tesm_local_entanglement_tilelang_autograd(
                q_4d.float(), k_4d.float(), v_4d.float(), local_bias_2d.float(),
                temperature=1.0,  # 局部纠缠不使用温度退火
                threshold=float(self.entanglement_threshold)
            )
            return entangled_4d.squeeze(2).to(q.dtype)  # (B, L, D)
        
        # 2. TileLang inference path
        if self._can_use_tilelang_fast_path(q) and tesm_local_entanglement_tilelang_fwd is not None:
            B, L, R = q.shape
            D = values.shape[-1]
            q_4d = q.unsqueeze(2).contiguous()  # (B, L, 1, R)
            k_4d = k.unsqueeze(2).contiguous()  # (B, L, 1, R)
            v_4d = values.unsqueeze(2).contiguous()  # (B, L, 1, D)
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype)
            local_bias_2d = local_bias.unsqueeze(0).contiguous()  # (1, W)
            entangled_4d = tesm_local_entanglement_tilelang_fwd(q_4d.float(), k_4d.float(), v_4d.float(), local_bias_2d.float(), threshold=float(self.entanglement_threshold))
            return entangled_4d.squeeze(2).to(q.dtype)  # (B, L, D)
        
        # 3. CUDA training path
        if self._can_use_cuda_ext_training_path(q) and cuda_local_entanglement_autograd is not None:
            q_cuda = q.contiguous()
            k_cuda = k.to(dtype=q.dtype).contiguous()
            values_cuda = values.to(dtype=q.dtype).contiguous()
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype).contiguous()
            entangled = cuda_local_entanglement_autograd(q_cuda, k_cuda, values_cuda, local_bias, float(self.entanglement_threshold))
            self._update_local_entanglement_stats_sample(q_cuda, k_cuda)
            return entangled
        
        # 4. CUDA inference path
        if self._can_use_cuda_ext_fast_path(q) and cuda_local_entanglement is not None:
            q_cuda = q.contiguous()
            k_cuda = k.to(dtype=q.dtype).contiguous()
            values_cuda = values.to(dtype=q.dtype).contiguous()
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype).contiguous()
            entangled = cuda_local_entanglement(q_cuda, k_cuda, values_cuda, local_bias, float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            return entangled
        
        # 5. Triton training path
        if self._can_use_triton_training_path(q) and triton_local_entanglement_autograd is not None:
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype).contiguous()
            entangled = triton_local_entanglement_autograd(q, k, values, local_bias, float(self.entanglement_threshold))
            self._update_local_entanglement_stats_sample(q, k)
            return entangled
        
        # 6. Triton inference path
        if self._can_use_triton_fast_path(q) and triton_local_entanglement is not None:
            local_bias = self.local_entanglement_bias[-window:].to(device=q.device, dtype=q.dtype).contiguous()
            entangled = triton_local_entanglement(q, k, values, local_bias, float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            return entangled
        
        # 7. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            block_size = min(max(self.entanglement_block_size, 1), seq_len)
            entangled = torch.empty_like(values)
            ternary_blocks = []
            relative_bias = self.local_entanglement_bias[-window:].to(dtype=q.dtype).view(1, 1, window)
            window_offsets = torch.arange(window, device=q.device)
            for start in range(0, seq_len, block_size):
                end = min(start + block_size, seq_len)
                block_len = end - start
                q_block = q[:, start:end, :]
                positions = torch.arange(start, end, device=q.device)
                window_positions = positions.unsqueeze(1) - (window - 1) + window_offsets.unsqueeze(0)
                gather_positions = window_positions.clamp(min=0, max=seq_len - 1)
                valid_mask = window_positions >= 0
                k_block = k[:, gather_positions, :]
                value_block = values[:, gather_positions, :]
                scores = torch.einsum("bqr,bqwr->bqw", q_block, k_block) / math.sqrt(self.ent_rank)
                local_bias = relative_bias[:, :, -window:].masked_fill(~valid_mask.unsqueeze(0), 0.0)
                scores = scores + local_bias
                ternary = self.ternary_entanglement(scores)
                ternary = ternary.masked_fill(~valid_mask.unsqueeze(0), 0.0)
                norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
                entangled[:, start:end, :] = torch.einsum("bqw,bqwd->bqd", ternary / norm, value_block)
                if ternary_blocks is not None:
                    ternary_blocks.append(ternary.detach())
            if len(ternary_blocks) > 0:
                all_ternary = torch.cat(ternary_blocks, dim=1)
                self._update_entanglement_stats(all_ternary)
            else:
                self._update_entanglement_stats(entangled.new_zeros((1, 1, 1)))
            return entangled
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but local entanglement kernel not available. "
            f"Available backends: cuda={cuda_local_entanglement_autograd is not None}, "
            f"triton={triton_local_entanglement is not None}, "
            f"tilelang={tesm_local_entanglement_tilelang_fwd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )

    def _compute_entanglement(self, q, k, values):
        seq_len = q.size(1)
        if self.entanglement_window > 0:
            return self._compute_local_entanglement(q, k, values)
        
        # 全局纠缠
        # 计算相对位置偏置
        positions = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L]
        
        freq_bands = torch.arange(1, self.global_rel_pos_dim + 1, device=q.device, dtype=q.dtype)
        freq_bands = freq_bands * math.pi / 1000.0
        
        rel_pos_expanded = rel_pos.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
        sin_features = torch.sin(rel_pos_expanded)
        cos_features = torch.cos(rel_pos_expanded)
        
        embed = self.global_rel_pos_embed.to(q.dtype)
        rel_bias = (sin_features * embed).sum(dim=-1) + (cos_features * embed).sum(dim=-1)
        rel_bias = rel_bias * self.global_rel_pos_scale.to(q.dtype) + self.global_rel_pos_bias.to(q.dtype)
        
        # 1. CUDA training path
        if self._can_use_cuda_ext_training_path(q) and cuda_global_entanglement_autograd is not None:
            entangled = cuda_global_entanglement_autograd(q, k, values, rel_bias, float(self.entanglement_threshold))
            self._update_global_entanglement_stats_sample(q, k, rel_bias)
            return entangled
        
        # 2. CUDA inference path
        if self._can_use_cuda_ext_fast_path(q) and cuda_global_entanglement is not None:
            entangled = cuda_global_entanglement(q, k, values, rel_bias, float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            return entangled
        
        # 3. TileLang training path
        if self._can_use_tilelang_training_path(q) and tesm_global_entanglement_tilelang_autograd is not None:
            # TileLang 期望 4D 输入
            q_4d = q.unsqueeze(2)  # (B, L, 1, R)
            k_4d = k.unsqueeze(2)  # (B, L, 1, R)
            v_4d = values.unsqueeze(2)  # (B, L, 1, D)
            entangled_4d = tesm_global_entanglement_tilelang_autograd(q_4d, k_4d, v_4d, rel_bias, float(self.entanglement_threshold))
            self._update_global_entanglement_stats_sample(q, k, rel_bias)
            return entangled_4d.squeeze(2)
        
        # 4. TileLang inference path
        if self._can_use_tilelang_fast_path(q) and tesm_global_entanglement_tilelang_autograd is not None:
            q_4d = q.unsqueeze(2)
            k_4d = k.unsqueeze(2)
            v_4d = values.unsqueeze(2)
            entangled_4d = tesm_global_entanglement_tilelang_autograd(q_4d, k_4d, v_4d, rel_bias, float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            return entangled_4d.squeeze(2)
        
        # 5. Triton training path (if autograd available)
        if self._can_use_triton_training_path(q) and triton_global_entanglement_autograd is not None:
            entangled = triton_global_entanglement_autograd(q, k, values, rel_bias, float(self.entanglement_threshold))
            self._update_global_entanglement_stats_sample(q, k, rel_bias)
            return entangled
        
        # 6. Triton inference path
        if self._can_use_triton_fast_path(q) and triton_global_entanglement is not None:
            entangled = triton_global_entanglement(q, k, values, rel_bias, float(self.entanglement_threshold))
            self.last_entanglement_map = None
            self.last_entanglement_stats = None
            return entangled
        
        # 7. PyTorch fallback - only for auto or torch backend
        if self.kernel_backend in {"auto", "torch"}:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.ent_rank)  # [B, L, L]
            scores = scores + rel_bias.unsqueeze(0)  # [B, L, L]
            
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            ternary = self.ternary_entanglement(scores)  # [B, L, L]
            ternary = ternary * causal_mask.unsqueeze(0).to(ternary.dtype)
            
            norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
            entangled = torch.matmul(ternary / norm, values)  # [B, L, d_state]
            
            self._update_entanglement_stats(ternary)
            return entangled
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but global entanglement kernel not available. "
            f"Available backends: cuda={cuda_global_entanglement_autograd is not None}, "
            f"triton={triton_global_entanglement is not None}, "
            f"tilelang={tesm_global_entanglement_tilelang_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )
    
    def _update_global_entanglement_stats_sample(self, q, k, rel_bias):
        """采样更新全局纠缠统计"""
        seq_len = q.size(1)
        with torch.no_grad():
            sample_len = min(seq_len, 32)
            scores = torch.matmul(q[:, :sample_len], k[:, :sample_len].transpose(-2, -1)) / math.sqrt(self.ent_rank)
            scores = scores + rel_bias[:sample_len, :sample_len].unsqueeze(0)
            causal_mask = self.causal_mask[:sample_len, :sample_len]
            ternary = self.ternary_entanglement(scores) * causal_mask.unsqueeze(0)
            self._update_entanglement_stats(ternary)

    def _parallel_prefix_scan(self, decay, update, h0):
        """并行前缀和 - 纯 PyTorch tensor 操作，GPU 高效
        
        计算: states[t] = decay[t] * states[t-1] + update[t], states[0] 的前驱 = h0
        
        数学推导:
            令 A[t] = prod_{i=0}^{t} decay[i]  (inclusive cumprod)
            h[t] = A[t] * h0 + A[t] * cumsum(update / A)[t]
                 = A[t] * (h0 + cumsum(update / A)[t])
        
        所有操作 (cumprod, cumsum, 除法, 乘法) 都是 PyTorch 内建的并行操作。
        
        Args:
            decay: (B, L, D) - 衰减因子
            update: (B, L, D) - 更新值
            h0: (B, D) - 初始状态
            
        Returns:
            states: (B, L, D) - 每个时间步的状态
        """
        A = torch.cumprod(decay, dim=1)  # (B, L, D)
        weighted_update = update / A.clamp_min(1e-12)  # (B, L, D)
        cum_weighted = torch.cumsum(weighted_update, dim=1)  # (B, L, D)
        states = A * (h0.unsqueeze(1) + cum_weighted)  # (B, L, D)
        return states

    def _parallel_state_scan(self, decay, update, prev_state=None):
        batch, seqlen, _ = decay.shape
        orig_dtype = decay.dtype
        chunk_size = min(max(self.state_scan_chunk_size, 1), seqlen)
        # 保存原始标志，用于后续路径选择
        prev_state_is_none = prev_state is None
        # 支持传入初始状态（分块训练用）
        if prev_state is None:
            prev_state = torch.zeros(batch, self.d_state, device=update.device, dtype=torch.float64)
        else:
            prev_state = prev_state.to(torch.float64)
        
        # 1. CUDA training path
        if prev_state_is_none and self._can_use_cuda_ext_training_path(decay) and cuda_chunk_state_scan_autograd is not None:
            return cuda_chunk_state_scan_autograd(decay.float(), update.float(), chunk_size).to(orig_dtype)
        
        # 2. CUDA inference path
        if prev_state_is_none and self._can_use_cuda_ext_fast_path(decay) and cuda_chunk_state_scan is not None:
            return cuda_chunk_state_scan(decay.float(), update.float(), chunk_size).to(orig_dtype)
        
        # 3. Triton training path
        if prev_state_is_none and self._can_use_triton_training_path(decay) and triton_chunk_state_scan_autograd is not None:
            return triton_chunk_state_scan_autograd(decay, update, chunk_size)
        
        # 4. Triton inference path
        if prev_state_is_none and self._can_use_triton_fast_path(decay) and triton_chunk_state_scan is not None:
            return triton_chunk_state_scan(decay, update, chunk_size)
        
        # 5. TileLang training path
        if prev_state_is_none and self._can_use_tilelang_training_path(decay) and tesm_chunked_scan_tilelang_autograd is not None:
            decay_4d = decay.unsqueeze(2)  # (B, L, 1, D)
            update_4d = update.unsqueeze(2)  # (B, L, 1, D)
            states_4d = tesm_chunked_scan_tilelang_autograd(decay_4d, update_4d, chunk_size)
            return states_4d.squeeze(2).to(orig_dtype)  # (B, L, D)
        
        # 6. TileLang inference path
        if prev_state_is_none and self._can_use_tilelang_fast_path(decay) and tesm_chunked_scan_tilelang_fwd is not None:
            decay_4d = decay.unsqueeze(2)  # (B, L, 1, D)
            update_4d = update.unsqueeze(2)  # (B, L, 1, D)
            states_4d = tesm_chunked_scan_tilelang_fwd(decay_4d.float(), update_4d.float())
            return states_4d.squeeze(2).to(orig_dtype)  # (B, L, D)
        
        # 7. PyTorch fallback - only for auto or torch backend, or when prev_state is provided
        if self.kernel_backend in {"auto", "torch"} or not prev_state_is_none:
            # ===== 并行前缀和 (associative scan) =====
            # 纯 PyTorch tensor 操作，GPU 高效，无 Python for 循环
            decay_f64 = decay.to(torch.float64).clamp_min(1e-12)
            update_f64 = update.to(torch.float64)
            
            # 初始状态
            if not prev_state_is_none:
                h0 = prev_state
            else:
                h0 = torch.zeros(batch, self.d_state, device=decay.device, dtype=torch.float64)
            
            # 分块并行 scan
            states_chunks = []
            h = h0
            
            for c in range(0, seqlen, chunk_size):
                end = min(c + chunk_size, seqlen)
                decay_chunk = decay_f64[:, c:end]   # (B, chunk_len, D)
                update_chunk = update_f64[:, c:end]  # (B, chunk_len, D)
                
                # 并行前缀和: states[t] = decay[t]*states[t-1] + update[t]
                # 使用 Blelloch 风格的向上/向下扫描
                chunk_len = end - c
                chunk_states = self._parallel_prefix_scan(decay_chunk, update_chunk, h)
                h = chunk_states[:, -1, :].detach()
                states_chunks.append(chunk_states)
            
            states = torch.cat(states_chunks, dim=1)
            self._last_scan_state_f64 = h
            return states.to(dtype=orig_dtype)
        
        # 指定的后端不可用，报错
        raise RuntimeError(
            f"kernel_backend='{self.kernel_backend}' specified but state scan kernel not available. "
            f"Available backends: cuda={cuda_chunk_state_scan_autograd is not None}, "
            f"triton={triton_chunk_state_scan_autograd is not None}, "
            f"tilelang={tesm_chunked_scan_tilelang_autograd is not None}. "
            f"Use kernel_backend='auto' or 'torch' for PyTorch fallback."
        )

    def forward(self, u, inference_params=None, cross_layer_state=None, prev_state=None, **kwargs):
        # 处理可能的额外参数（如从上层传递的labels等）
        batch, seqlen, _ = u.shape
        
        # 序列长度检查
        if seqlen > self.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")
        
        # 空序列检查
        if seqlen == 0:
            return u, None
        
        # 增量推理：如果是单 token 且有缓存，使用增量计算
        if inference_params is not None and seqlen == 1:
            layer_cache = inference_params.get('state_cache')
            if layer_cache is not None and 'state' in layer_cache:
                return self._forward_incremental(u, inference_params, cross_layer_state)
        
        _debug = hasattr(self, '_debug_fwd') and self._debug_fwd
        if _debug: print(f"  [FWD] in_proj...", flush=True)
        proj = self.in_proj(u)
        if _debug: print(f"  [FWD] split...", flush=True)
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(
            proj,
            [self.d_model, self.d_state, self.d_state, self.d_state, self.d_model, self.ent_rank, self.ent_rank],
            dim=-1,
        )
        state_value = torch.tanh(state_value)
        decay = torch.sigmoid(decay + self.decay_bias)
        write = torch.sigmoid(write)
        out_gate = torch.sigmoid(out_gate)
        
        # ===== 位置纠缠: RoPE 使纠缠模式随位置变化 =====
        if _debug: print(f"  [FWD] rope...", flush=True)
        ent_q = self._apply_rope(ent_q)
        ent_k = self._apply_rope(ent_k)
        
        # ===== 跨层纠缠: 上层逐位置状态偏置当前层的 Q =====
        if cross_layer_state is not None:
            # parallel: (B, L, d_state) → (B, L, ent_rank), incremental: (B, d_state) → (B, ent_rank)
            cross_q_bias = self.cross_layer_q_proj(cross_layer_state)
            if cross_q_bias.dim() == 2:  # incremental
                cross_q_bias = cross_q_bias.unsqueeze(1)
            ent_q = ent_q + cross_q_bias
        
        # ===== Phase 1: 纯状态积累 =====
        if _debug: print(f"  [FWD] state_scan...", flush=True)
        update = write * state_value
        states = self._parallel_state_scan(decay, update, prev_state=prev_state)
        if _debug: print(f"  [FWD] state_scan done", flush=True)
        
        # ===== Phase 2: 三值纠缠真实状态 (Bell态 lerp) =====
        # 注意：量子隧穿现在集成在 ternary_entanglement 中，作用于三值决策
        if _debug: print(f"  [FWD] entanglement...", flush=True)
        signed_avg = self._compute_entanglement(ent_q, ent_k, states)
        if _debug: print(f"  [FWD] entanglement done", flush=True)
        entangled_states = states + self.entanglement_scale * (signed_avg - states)
        ent_change = entangled_states - states
        
        # prefill 时写入推理缓存（ent_k 带 RoPE）
        if inference_params is not None:
            cache = inference_params.get('state_cache')
            if cache is not None and 'state' in cache:
                last_scan_state_f64 = self._last_scan_state_f64
                if last_scan_state_f64 is None:
                    last_scan_state_f64 = states[:, -1, :].detach().to(torch.float64)
                cache['state'] = last_scan_state_f64.clone()  # float64 精度状态
                cache['seq_pos'] = seqlen
                window = cache['ent_k_cache'].shape[1]
                if seqlen >= window:
                    cache['ent_k_cache'] = ent_k[:, -window:, :].detach().float()
                    cache['ent_v_cache'] = states[:, -window:, :].detach().float()
                    # 设置循环索引：下一个写入位置
                    cache['cache_idx'] = 0  # 缓存已满，从头开始覆盖
                else:
                    cache['ent_k_cache'][:, -seqlen:, :] = ent_k.detach().float()
                    cache['ent_v_cache'][:, -seqlen:, :] = states.detach().float()
                    # 设置循环索引：下一个写入位置
                    cache['cache_idx'] = 0  # 从位置 0 开始（因为数据在末尾）
        
        # ===== Phase 3: 输出 =====
        state_projected = self.state_proj(torch.tanh(entangled_states))
        ent_projected = self.ent_proj(ent_change)
        state_mixed = out_gate * state_projected
        ent_mixed = ent_projected
        y = local + state_mixed + ent_mixed
        y = self.out_proj(self.dropout(y))
        
        # 跨层状态: 逐位置状态传给下一层 (B, L, d_state)
        self._last_cross_layer_state = states.detach()
        # 返回最终状态用于分块训练
        final_state = self._last_scan_state_f64  # float64
        
        # 温度退火: 更新步数
        if self.training and self.annealing_enabled:
            self.annealing_step.add_(1)
        
        return y, final_state
    
    def _forward_incremental(self, u, inference_params, cross_layer_state=None):
        """增量推理：位置纠缠(RoPE) + 跨层纠缠 + 状态纠缠
        
        支持分页缓存：当use_paged=True时，定期保存状态到分页缓存
        """
        batch, seqlen, _ = u.shape
        cache = inference_params['state_cache']
        
        proj = self.in_proj(u)
        local, state_value, decay, write, out_gate, ent_q, ent_k = torch.split(
            proj,
            [self.d_model, self.d_state, self.d_state, self.d_state, self.d_model, self.ent_rank, self.ent_rank],
            dim=-1,
        )
        
        state_value = torch.tanh(state_value)
        decay = torch.sigmoid(decay + self.decay_bias)
        write = torch.sigmoid(write)
        out_gate = torch.sigmoid(out_gate)
        
        # 位置纠缠: RoPE at current position
        cur_pos = cache['seq_pos']
        ent_q_rope = self._apply_rope(ent_q, pos_offset=cur_pos)
        ent_k_rope = self._apply_rope(ent_k, pos_offset=cur_pos)
        q_vec = ent_q_rope[:, 0, :]
        k_vec = ent_k_rope[:, 0, :]
        v_vec = state_value[:, 0, :]
        
        # 跨层纠缠: 上层状态摘要偏置 Q
        if cross_layer_state is not None:
            cross_q_bias = self.cross_layer_q_proj(cross_layer_state)
            q_vec = q_vec + cross_q_bias
        
        # 分页缓存支持：在需要时加载状态
        if cache.get('use_paged', False):
            paged_cache = cache['paged_cache']
            # 每512个token保存一次状态到分页缓存
            if cur_pos > 0 and cur_pos % paged_cache.page_size == 0:
                paged_cache.save_state(cur_pos, {
                    'state': cache['state'],
                    'ent_k_cache': cache['ent_k_cache'],
                    'ent_v_cache': cache['ent_v_cache'],
                })
        
        # Phase 1: 纯状态更新 (float64 精度, 与 parallel scan 一致)
        prev_state = cache['state']  # float64
        new_state_f64 = decay[:, 0, :].double() * prev_state + write[:, 0, :].double() * v_vec.double()
        new_state = new_state_f64.to(decay.dtype)  # 下游计算用模型精度
        
        # Phase 2: 三值纠缠真实状态（使用循环缓冲区）
        window = cache['ent_k_cache'].shape[1]
        cache_idx = cache.get('cache_idx', 0)
        
        # 直接写入当前位置（避免 torch.cat）
        cache['ent_k_cache'][:, cache_idx, :] = k_vec
        cache['ent_v_cache'][:, cache_idx, :] = new_state
        
        # 更新循环索引
        cache['cache_idx'] = (cache_idx + 1) % window
        
        # 计算有效长度
        seq_pos = cache['seq_pos']
        valid_len = min(seq_pos + 1, window)
        
        # 高效计算：使用索引而不是 roll
        # 当缓存已满时，按时间顺序构建索引
        if valid_len >= window:
            # 缓存已满：构建索引 [cache_idx, cache_idx+1, ..., cache_idx+window-1] % window
            # 这等价于从最旧到最新的顺序
            indices = torch.arange(cache_idx, cache_idx + window, device=cache['ent_k_cache'].device) % window
            k_eff = cache['ent_k_cache'][:, indices, :]  # (1, window, ent_rank)
            v_eff = cache['ent_v_cache'][:, indices, :]  # (1, window, d_state)
        else:
            # 缓存未满：有效数据在 [:valid_len]，需要补零
            k_eff = cache['ent_k_cache'][:, :valid_len, :]
            v_eff = cache['ent_v_cache'][:, :valid_len, :]
        
        # 计算 scores
        scores = torch.einsum('br,bwr->bw', q_vec, k_eff) / (self.ent_rank ** 0.5)
        
        # 添加偏置
        if self.local_entanglement_bias is not None:
            bias = self.local_entanglement_bias[-valid_len:].to(dtype=scores.dtype, device=scores.device)
            scores = scores + bias
        
        # 如果缓存未满，需要补零到 window 大小
        if valid_len < window:
            # 使用实际batch size，而非硬编码1
            full_scores = cache['ent_k_cache'].new_zeros(batch, window)
            full_scores[:, window - valid_len:] = scores
            scores = full_scores
        
        # 三值纠缠
        ternary = self.ternary_entanglement(scores)
        if valid_len < window:
            ternary[:, :window - valid_len] = 0.0
        
        # 归一化
        norm = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        
        # 计算加权平均
        if valid_len < window:
            # 需要补零，使用实际batch size
            full_v = cache['ent_v_cache'].new_zeros(batch, window, self.d_state)
            full_v[:, window - valid_len:] = v_eff
            signed_avg = torch.einsum('bw,bwd->bd', ternary / norm, full_v)
        else:
            signed_avg = torch.einsum('bw,bwd->bd', ternary / norm, v_eff)
        
        entangled_state = new_state + self.entanglement_scale * (signed_avg - new_state)
        ent_change = entangled_state - new_state
        
        cache['state'] = new_state_f64.detach()  # 保持 float64 精度
        cache['seq_pos'] += 1
        
        # Phase 3: 输出
        state_projected = self.state_proj(torch.tanh(entangled_state))
        ent_projected = self.ent_proj(ent_change)
        
        state_mixed = out_gate[:, 0, :] * state_projected
        ent_mixed = ent_projected
        y = local[:, 0, :] + state_mixed + ent_mixed
        y = self.out_proj(y.unsqueeze(1))
        
        # 跨层状态
        self._last_cross_layer_state = new_state.detach()
        return y, new_state_f64.detach()
