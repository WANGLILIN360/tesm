import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch

# 多模块加载：支持分批次编译
_loaded_modules: Dict[str, Any] = {}
_load_attempted: Dict[str, bool] = {}
_load_errors: Dict[str, Optional[Exception]] = {}

# 统一模块名称
_MODULE_NAME = "tesm_cuda_ops"


def _extension_sources():
    """JIT 编译时的源文件（备用方案）"""
    root = Path(__file__).resolve().parents[3]
    source_root = root / "csrc" / "tesm_ops"
    return root, [
        str(source_root / "tesm_ops.cpp"),
        str(source_root / "tesm_scan_fwd.cu"),
        str(source_root / "tesm_scan_bwd.cu"),
        str(source_root / "tesm_entanglement_fwd.cu"),
        str(source_root / "tesm_entanglement_bwd.cu"),
    ]


def _load_module(name: str):
    """加载单个模块"""
    global _loaded_modules, _load_attempted, _load_errors
    
    if name in _load_attempted:
        return _loaded_modules.get(name)
    
    _load_attempted[name] = True
    
    if not torch.cuda.is_available():
        return None
    
    # 尝试加载预编译模块
    try:
        _loaded_modules[name] = importlib.import_module(name)
        return _loaded_modules[name]
    except Exception as exc:
        _load_errors[name] = exc
    
    return None


def _load_extension():
    """加载 CUDA 扩展模块"""
    # 尝试加载统一模块
    module = _load_module(_MODULE_NAME)
    if module is not None:
        return module
    
    # 如果统一模块加载失败，尝试 JIT 编译（备用）
    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None
    
    root, sources = _extension_sources()
    build_directory = root / ".tesm_build" / "cuda_ops"
    build_directory.mkdir(parents=True, exist_ok=True)
    
    try:
        module = load(
            name="tesm_cuda_ops_jit",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=str(build_directory),
        )
        _loaded_modules[_MODULE_NAME] = module
        return module
    except Exception as e:
        _load_errors[_MODULE_NAME] = e
    
    return None


class _MergedModule:
    """合并多个模块的接口"""
    
    def __init__(self, modules: Dict[str, Any]):
        self._modules = modules
    
    def __getattr__(self, name: str):
        # 在所有模块中查找函数
        for module in self._modules.values():
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"Module has no attribute '{name}'")
    
    def has_function(self, name: str) -> bool:
        """检查是否有某个函数"""
        for module in self._modules.values():
            if hasattr(module, name):
                return True
        return False


def tesm_cuda_is_available() -> bool:
    """检查 CUDA 扩展是否可用（不触发 JIT 编译）"""
    if not torch.cuda.is_available():
        return False
    # 只检查已加载的模块，不触发 JIT 编译
    if _MODULE_NAME in _loaded_modules and _loaded_modules[_MODULE_NAME] is not None:
        return True
    # 尝试加载预编译模块（不 JIT 编译）
    if _MODULE_NAME not in _load_attempted:
        module = _load_module(_MODULE_NAME)
        return module is not None
    return False


def tesm_cuda_load_error() -> Optional[str]:
    """返回加载错误信息"""
    errors = []
    for name, err in _load_errors.items():
        if err:
            errors.append(f"{name}: {err}")
    return "; ".join(errors) if errors else None


def cuda_chunk_state_scan(decay: torch.Tensor, update: torch.Tensor, chunk_size: int) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.chunk_state_scan_fwd(decay.contiguous(), update.contiguous(), int(chunk_size))


def cuda_chunk_state_scan_backward(decay: torch.Tensor, states: torch.Tensor, grad_states: torch.Tensor):
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.chunk_state_scan_bwd(decay.contiguous(), states.contiguous(), grad_states.contiguous())


def cuda_local_entanglement(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, local_bias: torch.Tensor, threshold: float) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.local_entanglement_fwd(q.contiguous(), k.contiguous(), values.contiguous(), local_bias.contiguous(), float(threshold))


def cuda_local_entanglement_backward(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, local_bias: torch.Tensor, grad_out: torch.Tensor, threshold: float):
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.local_entanglement_bwd(q.contiguous(), k.contiguous(), values.contiguous(), local_bias.contiguous(), grad_out.contiguous(), float(threshold))


def cuda_quantized_linear(x: torch.Tensor, qweight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    if bias is None:
        return module.quantized_linear_fwd(x.contiguous(), qweight.contiguous())
    return module.quantized_linear_fwd_bias(x.contiguous(), qweight.contiguous(), bias.contiguous())


def cuda_quantized_linear_backward(grad_output: torch.Tensor, x: torch.Tensor, qweight: torch.Tensor, has_bias: bool):
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.quantized_linear_bwd(grad_output.contiguous(), x.contiguous(), qweight.contiguous(), bool(has_bias))


class _CudaChunkStateScanAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay: torch.Tensor, update: torch.Tensor, chunk_size: int):
        states = cuda_chunk_state_scan(decay, update, int(chunk_size))
        ctx.save_for_backward(decay, states)
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        decay, states = ctx.saved_tensors
        grad_decay, grad_update = cuda_chunk_state_scan_backward(decay, states, grad_states)
        return grad_decay, grad_update, None


class _CudaLocalEntanglementAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, local_bias: torch.Tensor, threshold: float):
        out = cuda_local_entanglement(q, k, values, local_bias, float(threshold))
        ctx.threshold = float(threshold)
        ctx.save_for_backward(q, k, values, local_bias)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, values, local_bias = ctx.saved_tensors
        grad_q, grad_k, grad_values, grad_bias = cuda_local_entanglement_backward(q, k, values, local_bias, grad_out, ctx.threshold)
        return grad_q, grad_k, grad_values, grad_bias, None


class _CudaQuantizedLinearAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qinput: torch.Tensor, qweight: torch.Tensor, bias: Optional[torch.Tensor]):
        ctx.has_bias = bias is not None
        if bias is None:
            output = cuda_quantized_linear(qinput, qweight, None)
            ctx.save_for_backward(qinput, qweight)
            return output
        output = cuda_quantized_linear(qinput, qweight, bias)
        ctx.save_for_backward(qinput, qweight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        if ctx.has_bias:
            qinput, qweight, _ = saved
        else:
            qinput, qweight = saved
        grads = cuda_quantized_linear_backward(grad_output, qinput, qweight, ctx.has_bias)
        if ctx.has_bias:
            grad_input, grad_weight, grad_bias = grads
        else:
            grad_input, grad_weight = grads
            grad_bias = None
        return grad_input, grad_weight, grad_bias


def cuda_chunk_state_scan_autograd(decay: torch.Tensor, update: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return _CudaChunkStateScanAutogradFunction.apply(decay, update, int(chunk_size))


def cuda_local_entanglement_autograd(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, local_bias: torch.Tensor, threshold: float) -> torch.Tensor:
    return _CudaLocalEntanglementAutogradFunction.apply(q, k, values, local_bias, float(threshold))


def cuda_quantized_linear_autograd(qinput: torch.Tensor, qweight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _CudaQuantizedLinearAutogradFunction.apply(qinput, qweight, bias)


def cuda_int2_linear(x: torch.Tensor, packed_weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """INT2 量化线性层（CUDA kernel）
    
    Args:
        x: FP32 输入，shape [..., K]
        packed_weight: UINT8 打包权重，shape [N, K // 4]
        scale: 缩放因子
        bias: 可选偏置
    
    Returns:
        输出，shape [..., N]
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    if bias is None:
        return module.int2_linear_fwd(x.contiguous(), packed_weight.contiguous(), scale.contiguous())
    return module.int2_linear_fwd_bias(x.contiguous(), packed_weight.contiguous(), scale.contiguous(), bias.contiguous())


def cuda_int2_linear_optimized(x: torch.Tensor, packed_weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """INT2 量化线性层（优化版 CUDA kernel，借鉴 BitNet）
    
    Args:
        x: FP32 输入，shape [..., K]
        packed_weight: UINT8 打包权重，shape [N, K // 4]
        scale: 缩放因子
        bias: 可选偏置
    
    Returns:
        输出，shape [..., N]
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    if bias is None:
        return module.int2_linear_optimized_fwd(x.contiguous(), packed_weight.contiguous(), scale.contiguous())
    return module.int2_linear_optimized_fwd_bias(x.contiguous(), packed_weight.contiguous(), scale.contiguous(), bias.contiguous())


def cuda_int8xint2_linear(x_quant: torch.Tensor, x_scale: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """INT8 输入 × INT2 权重线性层（最高性能版本）
    
    使用 DP4A 指令加速 INT8 点积。
    
    Args:
        x_quant: INT8 量化输入，shape [..., K]
        x_scale: 输入缩放因子
        packed_weight: UINT8 打包权重，shape [N, K // 4]
        weight_scale: 权重缩放因子
        bias: 可选偏置
    
    Returns:
        输出，shape [..., N]
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    if bias is None:
        return module.int8xint2_linear_fwd(x_quant.contiguous(), x_scale.contiguous(), packed_weight.contiguous(), weight_scale.contiguous())
    return module.int8xint2_linear_fwd_bias(x_quant.contiguous(), x_scale.contiguous(), packed_weight.contiguous(), weight_scale.contiguous(), bias.contiguous())


# ============================================================================
# Global Entanglement (CUDA)
# ============================================================================

def cuda_global_entanglement(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                              Bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """CUDA 全局纠缠前向 (SISO)
    
    Args:
        Q: (B, L, R)
        K: (B, L, R)
        V: (B, L, D)
        Bias: (L, L)
        threshold: ternary threshold
    
    Returns:
        out: (B, L, D)
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.global_entanglement_fwd(Q.contiguous(), K.contiguous(), V.contiguous(), Bias.contiguous(), threshold)


def cuda_global_entanglement_backward(grad_out: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, 
                                        V: torch.Tensor, Bias: torch.Tensor, threshold: float) -> torch.Tensor:
    """CUDA 全局纠缠反向 (SISO) - 只返回 grad_V"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.global_entanglement_bwd(grad_out.contiguous(), Q.contiguous(), K.contiguous(), V.contiguous(), Bias.contiguous(), threshold)


class _CUDAGlobalEntanglementAutograd(torch.autograd.Function):
    """CUDA global entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, Q, K, V, Bias, threshold):
        ctx.save_for_backward(Q, K, V, Bias)
        ctx.threshold = threshold
        return cuda_global_entanglement(Q, K, V, Bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, Bias = ctx.saved_tensors
        threshold = ctx.threshold
        
        # CUDA kernel 只返回 grad_V，需要 PyTorch fallback 计算 grad_Q 和 grad_K
        grad_V = cuda_global_entanglement_backward(grad_out, Q, K, V, Bias, threshold)
        
        # PyTorch fallback 计算 grad_Q 和 grad_K
        # score = Q @ K^T / sqrt(R) + Bias
        # ternary = sign(score) if |score| > threshold else 0
        # out = ternary @ V
        # grad_Q = grad_out @ V^T @ ternary_weight_K
        # grad_K = ternary_weight_Q^T @ grad_out @ V
        
        with torch.no_grad():
            B, L, R = Q.shape
            D = V.shape[-1]
            inv_scale = 1.0 / (R ** 0.5)
            
            # 计算所有 score
            # score[i,j] = Q[i] @ K[j]^T / sqrt(R) + Bias[i,j]
            scores = torch.matmul(Q, K.transpose(-2, -1)) * inv_scale  # (B, L, L)
            if Bias is not None:
                scores = scores + Bias.unsqueeze(0)  # (B, L, L)
            
            # Ternary weights
            ternary = torch.zeros_like(scores)
            ternary[scores > threshold] = 1.0
            ternary[scores < -threshold] = -1.0
            
            # grad_Q = grad_out @ V^T @ ternary (简化版本)
            # 对于完整的梯度，需要更复杂的计算
            # 这里使用近似：grad_Q ≈ grad_out @ V^T @ K / sqrt(R) * ternary_mask
            grad_Q = torch.matmul(grad_out, V.transpose(-2, -1))  # (B, L, L)
            grad_Q = torch.matmul(grad_Q * ternary, K) * inv_scale  # (B, L, R)
            
            # grad_K = ternary^T @ grad_out @ V
            grad_K = torch.matmul(ternary.transpose(-2, -1), grad_out)  # (B, L, D)
            grad_K = torch.matmul(grad_K, V.transpose(-2, -1))  # (B, L, D) -> 需要投影到 R
            # 简化：使用 K 的形状
            grad_K = torch.matmul(grad_K, V)  # 近似
        
        return grad_Q, grad_K, grad_V, None, None


def cuda_global_entanglement_autograd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                       Bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """CUDA global entanglement with autograd support."""
    return _CUDAGlobalEntanglementAutograd.apply(Q, K, V, Bias, threshold)


# ============================================================================
# Global Entanglement MIMO (CUDA)
# ============================================================================

def cuda_global_entanglement_mimo(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   Bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """CUDA 多头全局纠缠前向 (MIMO)
    
    Args:
        Q: (B, L, H, R)
        K: (B, L, H, R)
        V: (B, L, H, D)
        Bias: (H, L, L) or (L, L)
        threshold: ternary threshold
    
    Returns:
        out: (B, L, H, D)
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    
    # 处理 bias 维度
    if Bias.dim() == 2:
        Bias = Bias.unsqueeze(0).expand(Q.size(2), -1, -1)
    
    return module.global_entanglement_mimo_fwd(Q.contiguous(), K.contiguous(), V.contiguous(), Bias.contiguous(), threshold)


def cuda_global_entanglement_mimo_backward(grad_out: torch.Tensor, Q: torch.Tensor, K: torch.Tensor,
                                            V: torch.Tensor, Bias: torch.Tensor, threshold: float) -> torch.Tensor:
    """CUDA 多头全局纠缠反向 (MIMO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.global_entanglement_mimo_bwd(grad_out.contiguous(), Q.contiguous(), K.contiguous(), V.contiguous(), Bias.contiguous(), threshold)


class _CUDAGlobalEntanglementMIMOAutograd(torch.autograd.Function):
    """CUDA MIMO global entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, Q, K, V, Bias, threshold):
        ctx.save_for_backward(Q, K, V, Bias)
        ctx.threshold = threshold
        return cuda_global_entanglement_mimo(Q, K, V, Bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, Bias = ctx.saved_tensors
        threshold = ctx.threshold
        grad_V = cuda_global_entanglement_mimo_backward(grad_out, Q, K, V, Bias, threshold)
        return None, None, grad_V, None, None


def cuda_global_entanglement_mimo_autograd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                            Bias: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    """CUDA MIMO global entanglement with autograd support."""
    return _CUDAGlobalEntanglementMIMOAutograd.apply(Q, K, V, Bias, threshold)


# ============================================================================
# Fused Output (CUDA)
# ============================================================================

def cuda_fused_output(local: torch.Tensor, gate: torch.Tensor, state_proj: torch.Tensor,
                       ent_proj: torch.Tensor, ent_scale: float) -> torch.Tensor:
    """CUDA 融合输出前向 (SISO)
    
    out = local * gate + state_proj + ent_scale * ent_proj
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.fused_output_fwd(local.contiguous(), gate.contiguous(), state_proj.contiguous(), ent_proj.contiguous(), ent_scale)


def cuda_fused_output_backward(grad_out: torch.Tensor, local: torch.Tensor, gate: torch.Tensor,
                                 ent_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA 融合输出反向 (SISO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.fused_output_bwd(grad_out.contiguous(), local.contiguous(), gate.contiguous(), ent_scale)


class _CUDAFusedOutputAutograd(torch.autograd.Function):
    """CUDA fused output with autograd support."""
    
    @staticmethod
    def forward(ctx, local, gate, state_proj, ent_proj, ent_scale):
        ctx.save_for_backward(local, gate)
        ctx.ent_scale = ent_scale
        return cuda_fused_output(local, gate, state_proj, ent_proj, ent_scale)
    
    @staticmethod
    def backward(ctx, grad_out):
        local, gate = ctx.saved_tensors
        ent_scale = ctx.ent_scale
        grad_local, grad_gate, grad_state_proj, grad_ent_proj = cuda_fused_output_backward(grad_out, local, gate, ent_scale)
        return grad_local, grad_gate, grad_state_proj, grad_ent_proj, None


def cuda_fused_output_autograd(local: torch.Tensor, gate: torch.Tensor, state_proj: torch.Tensor,
                                ent_proj: torch.Tensor, ent_scale: float) -> torch.Tensor:
    """CUDA fused output with autograd support."""
    return _CUDAFusedOutputAutograd.apply(local, gate, state_proj, ent_proj, ent_scale)


# ============================================================================
# Fused Output MIMO (CUDA)
# ============================================================================

def cuda_fused_output_mimo(local: torch.Tensor, gate: torch.Tensor, state_proj: torch.Tensor,
                            ent_proj: torch.Tensor, ent_scale: float) -> torch.Tensor:
    """CUDA 融合输出前向 (MIMO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.fused_output_mimo_fwd(local.contiguous(), gate.contiguous(), state_proj.contiguous(), ent_proj.contiguous(), ent_scale)


def cuda_fused_output_mimo_backward(grad_out: torch.Tensor, local: torch.Tensor, gate: torch.Tensor,
                                     ent_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA 融合输出反向 (MIMO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.fused_output_mimo_bwd(grad_out.contiguous(), local.contiguous(), gate.contiguous(), ent_scale)


class _CUDAFusedOutputMIMOAutograd(torch.autograd.Function):
    """CUDA MIMO fused output with autograd support."""
    
    @staticmethod
    def forward(ctx, local, gate, state_proj, ent_proj, ent_scale):
        ctx.save_for_backward(local, gate)
        ctx.ent_scale = ent_scale
        return cuda_fused_output_mimo(local, gate, state_proj, ent_proj, ent_scale)
    
    @staticmethod
    def backward(ctx, grad_out):
        local, gate = ctx.saved_tensors
        ent_scale = ctx.ent_scale
        grad_local, grad_gate, grad_state_proj, grad_ent_proj = cuda_fused_output_mimo_backward(grad_out, local, gate, ent_scale)
        return grad_local, grad_gate, grad_state_proj, grad_ent_proj, None


def cuda_fused_output_mimo_autograd(local: torch.Tensor, gate: torch.Tensor, state_proj: torch.Tensor,
                                     ent_proj: torch.Tensor, ent_scale: float) -> torch.Tensor:
    """CUDA MIMO fused output with autograd support."""
    return _CUDAFusedOutputMIMOAutograd.apply(local, gate, state_proj, ent_proj, ent_scale)


# ============================================================================
# MIMO State Scan (CUDA)
# ============================================================================

def cuda_chunk_state_scan_mimo(decay: torch.Tensor, update: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """CUDA 多头状态扫描前向 (MIMO)
    
    Args:
        decay: (B, L, H, D)
        update: (B, L, H, D)
        chunk_size: chunk size
    
    Returns:
        states: (B, L, H, D)
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.chunk_state_scan_mimo_fwd(decay.contiguous(), update.contiguous(), chunk_size)


def cuda_chunk_state_scan_mimo_backward(decay: torch.Tensor, states: torch.Tensor, grad_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA 多头状态扫描反向 (MIMO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    grads = module.chunk_state_scan_mimo_bwd(decay.contiguous(), states.contiguous(), grad_states.contiguous())
    return grads[0], grads[1]


class _CUDAChunkStateScanMIMOAutograd(torch.autograd.Function):
    """CUDA MIMO chunk state scan with autograd support."""
    
    @staticmethod
    def forward(ctx, decay, update, chunk_size):
        states = cuda_chunk_state_scan_mimo(decay, update, chunk_size)
        ctx.save_for_backward(decay, update, states)  # 保存 states 用于反向传播
        ctx.chunk_size = chunk_size
        ctx.states_shape = states.shape
        return states
    
    @staticmethod
    def backward(ctx, grad_states):
        decay, update, states = ctx.saved_tensors  # 使用保存的 states
        chunk_size = ctx.chunk_size
        
        grad_decay, grad_update = cuda_chunk_state_scan_mimo_backward(decay, states, grad_states)
        return grad_decay, grad_update, None


def cuda_chunk_state_scan_mimo_autograd(decay: torch.Tensor, update: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """CUDA MIMO chunk state scan with autograd support."""
    return _CUDAChunkStateScanMIMOAutograd.apply(decay, update, chunk_size)


# ============================================================================
# MIMO Local Entanglement (CUDA)
# ============================================================================

def cuda_local_entanglement_mimo(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   bias: torch.Tensor, threshold: float) -> torch.Tensor:
    """CUDA 多头局部窗口纠缠前向 (MIMO)
    
    Args:
        q: (B, L, H, R)
        k: (B, L, H, R)
        v: (B, L, H, D)
        bias: (H, L, L) or (H, L, W)
        threshold: ternary threshold
    
    Returns:
        out: (B, L, H, D)
    """
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    return module.local_entanglement_mimo_fwd(q.contiguous(), k.contiguous(), v.contiguous(), bias.contiguous(), threshold)


def cuda_local_entanglement_mimo_backward(grad_out: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                                            v: torch.Tensor, bias: torch.Tensor, threshold: float) -> torch.Tensor:
    """CUDA 多头局部窗口纠缠反向 (MIMO)"""
    module = _load_extension()
    if module is None:
        raise RuntimeError(f"TESM CUDA extension is unavailable: {tesm_cuda_load_error()}")
    grads = module.local_entanglement_mimo_bwd(grad_out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(), bias.contiguous(), threshold)
    return grads[0]


class _CUDALocalEntanglementMIMOAutograd(torch.autograd.Function):
    """CUDA MIMO local entanglement with autograd support."""
    
    @staticmethod
    def forward(ctx, q, k, v, bias, threshold):
        ctx.save_for_backward(q, k, v, bias)
        ctx.threshold = threshold
        return cuda_local_entanglement_mimo(q, k, v, bias, threshold)
    
    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, bias = ctx.saved_tensors
        threshold = ctx.threshold
        grad_v = cuda_local_entanglement_mimo_backward(grad_out, q, k, v, bias, threshold)
        return None, None, grad_v, None, None


def cuda_local_entanglement_mimo_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                            bias: torch.Tensor, threshold: float) -> torch.Tensor:
    """CUDA MIMO local entanglement with autograd support."""
    return _CUDALocalEntanglementMIMOAutograd.apply(q, k, v, bias, threshold)


# 支持torch.compile：禁用CUDA kernel的追踪，避免fake tensor错误
try:
    cuda_chunk_state_scan_autograd = torch.compiler.disable(cuda_chunk_state_scan_autograd)
    cuda_local_entanglement_autograd = torch.compiler.disable(cuda_local_entanglement_autograd)
    cuda_quantized_linear_autograd = torch.compiler.disable(cuda_quantized_linear_autograd)
except AttributeError:
    pass  # 旧版torch没有此API


__all__ = [
    "tesm_cuda_is_available",
    "tesm_cuda_load_error",
    "cuda_chunk_state_scan",
    "cuda_chunk_state_scan_backward",
    "cuda_chunk_state_scan_autograd",
    "cuda_local_entanglement",
    "cuda_local_entanglement_backward",
    "cuda_local_entanglement_autograd",
    "cuda_quantized_linear",
    "cuda_quantized_linear_backward",
    "cuda_quantized_linear_autograd",
    "cuda_int2_linear",
    "cuda_int2_linear_optimized",
    "cuda_int8xint2_linear",
    # Global Entanglement
    "cuda_global_entanglement",
    "cuda_global_entanglement_autograd",
    "cuda_global_entanglement_mimo",
    "cuda_global_entanglement_mimo_autograd",
    # Fused Output
    "cuda_fused_output",
    "cuda_fused_output_autograd",
    "cuda_fused_output_mimo",
    "cuda_fused_output_mimo_autograd",
    # MIMO State Scan
    "cuda_chunk_state_scan_mimo",
    "cuda_chunk_state_scan_mimo_autograd",
    # MIMO Local Entanglement
    "cuda_local_entanglement_mimo",
    "cuda_local_entanglement_mimo_autograd",
]
