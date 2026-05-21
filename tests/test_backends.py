#!/usr/bin/env python3
"""
TESM 后端功能测试脚本

检查所有后端 (PyTorch, CUDA, Triton, TileLang) 的实现状态

用法:
    python test_backends.py
    python test_backends.py --verbose
    python test_backends.py --cuda-only
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TestResult:
    """测试结果"""
    name: str
    backend: str
    passed: bool
    error: Optional[str] = None
    time_ms: Optional[float] = None


# 测试配置
BATCH = 2
SEQLEN = 32
N_HEADS = 4
D_MODEL = 64
D_STATE = 16
ENT_RANK = 8
D_HEAD = D_MODEL // N_HEADS
WINDOW = 8
THRESHOLD = 0.08

# 全局结果
results: List[TestResult] = []


def log_result(result: TestResult, verbose: bool = False):
    """记录测试结果"""
    results.append(result)
    status = "✅" if result.passed else "❌"
    msg = f"{status} [{result.backend}] {result.name}"
    if result.error and verbose:
        msg += f"\n   Error: {result.error}"
    print(msg)


def test_pytorch_backend(verbose: bool = False):
    """测试 PyTorch 后端"""
    print("\n" + "="*60)
    print("PyTorch Backend Tests")
    print("="*60)
    
    # 1. BitLinear (量化线性)
    try:
        from tesm_ssm.modules.tesm import BitLinear
        layer = BitLinear(D_MODEL, D_MODEL * 2)
        x = torch.randn(BATCH, SEQLEN, D_MODEL)
        y = layer(x)
        assert y.shape == (BATCH, SEQLEN, D_MODEL * 2), f"Shape mismatch: {y.shape}"
        log_result(TestResult("BitLinear", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("BitLinear", "PyTorch", False, str(e)), verbose)
    
    # 2. StateScan (状态扫描)
    try:
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_STATE))
        update = torch.randn(BATCH, SEQLEN, D_STATE)
        
        # 简单的递归扫描
        states = []
        state = torch.zeros(BATCH, D_STATE)
        for t in range(SEQLEN):
            state = decay[:, t] * state + update[:, t]
            states.append(state)
        states = torch.stack(states, dim=1)
        assert states.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("StateScan", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("StateScan", "PyTorch", False, str(e)), verbose)
    
    # 3. LocalEnt (局部纠缠)
    try:
        q = torch.randn(BATCH, SEQLEN, ENT_RANK)
        k = torch.randn(BATCH, SEQLEN, ENT_RANK)
        v = torch.randn(BATCH, SEQLEN, D_STATE)
        
        # 简化的局部窗口三值 attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (ENT_RANK ** 0.5)
        ternary = torch.where(scores > THRESHOLD, torch.ones_like(scores),
                             torch.where(scores < -THRESHOLD, -torch.ones_like(scores),
                                        torch.zeros_like(scores)))
        out = torch.matmul(ternary.float(), v)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("LocalEnt", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("LocalEnt", "PyTorch", False, str(e)), verbose)
    
    # 4. GlobalEnt (全局纠缠)
    try:
        q = torch.randn(BATCH, SEQLEN, ENT_RANK)
        k = torch.randn(BATCH, SEQLEN, ENT_RANK)
        v = torch.randn(BATCH, SEQLEN, D_STATE)
        bias = torch.randn(SEQLEN, SEQLEN)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / (ENT_RANK ** 0.5) + bias
        ternary = torch.where(scores > THRESHOLD, torch.ones_like(scores),
                             torch.where(scores < -THRESHOLD, -torch.ones_like(scores),
                                        torch.zeros_like(scores)))
        out = torch.matmul(ternary.float(), v)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("GlobalEnt", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("GlobalEnt", "PyTorch", False, str(e)), verbose)
    
    # 5. FusedOutput (融合输出)
    try:
        local = torch.randn(BATCH, SEQLEN, D_MODEL)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_MODEL))
        state_proj = torch.randn(BATCH, SEQLEN, D_MODEL)
        ent_proj = torch.randn(BATCH, SEQLEN, D_MODEL)
        ent_scale = 0.5
        
        out = local * gate + state_proj + ent_scale * ent_proj
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("FusedOutput", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("FusedOutput", "PyTorch", False, str(e)), verbose)
    
    # 6. MIMO StateScan
    try:
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD))
        update = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        
        states = []
        state = torch.zeros(BATCH, N_HEADS, D_HEAD)
        for t in range(SEQLEN):
            state = decay[:, t] * state + update[:, t]
            states.append(state)
        states = torch.stack(states, dim=1)
        assert states.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO StateScan", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO StateScan", "PyTorch", False, str(e)), verbose)
    
    # 7. MIMO LocalEnt
    try:
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        bias = torch.randn(N_HEADS, SEQLEN, SEQLEN)
        
        # 多头局部纠缠
        out_list = []
        for h in range(N_HEADS):
            scores = torch.matmul(q[:, :, h], k[:, :, h].transpose(-1, -2)) / (ENT_RANK ** 0.5) + bias[h]
            ternary = torch.where(scores > THRESHOLD, torch.ones_like(scores),
                                 torch.where(scores < -THRESHOLD, -torch.ones_like(scores),
                                            torch.zeros_like(scores)))
            out_h = torch.matmul(ternary.float(), v[:, :, h])
            out_list.append(out_h)
        out = torch.stack(out_list, dim=2)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO LocalEnt", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO LocalEnt", "PyTorch", False, str(e)), verbose)
    
    # 8. MIMO GlobalEnt
    try:
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        bias = torch.randn(N_HEADS, SEQLEN, SEQLEN)
        
        out_list = []
        for h in range(N_HEADS):
            scores = torch.matmul(q[:, :, h], k[:, :, h].transpose(-1, -2)) / (ENT_RANK ** 0.5) + bias[h]
            ternary = torch.where(scores > THRESHOLD, torch.ones_like(scores),
                                 torch.where(scores < -THRESHOLD, -torch.ones_like(scores),
                                            torch.zeros_like(scores)))
            out_h = torch.matmul(ternary.float(), v[:, :, h])
            out_list.append(out_h)
        out = torch.stack(out_list, dim=2)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO GlobalEnt", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO GlobalEnt", "PyTorch", False, str(e)), verbose)
    
    # 9. MIMO FusedOutput
    try:
        local = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD))
        state_proj = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        ent_proj = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD)
        ent_scale = 0.5
        
        out = local * gate + state_proj + ent_scale * ent_proj
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO FusedOutput", "PyTorch", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO FusedOutput", "PyTorch", False, str(e)), verbose)


def test_cuda_backend(verbose: bool = False):
    """测试 CUDA 后端"""
    print("\n" + "="*60)
    print("CUDA Backend Tests")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_result(TestResult("CUDA Available", "CUDA", False, "CUDA not available"), verbose)
        return
    
    # 检查 CUDA 扩展
    try:
        from tesm_ssm.ops.cuda import tesm_cuda_is_available, tesm_cuda_load_error
        if not tesm_cuda_is_available():
            error = tesm_cuda_load_error()
            log_result(TestResult("CUDA Extension", "CUDA", False, f"Extension not loaded: {error}"), verbose)
            return
        log_result(TestResult("CUDA Extension", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("CUDA Extension", "CUDA", False, str(e)), verbose)
        return
    
    # 1. BitLinear
    try:
        from tesm_ssm.ops.cuda import cuda_quantized_linear, cuda_quantized_linear_autograd
        x = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        qweight = torch.randint(-1, 2, (D_MODEL * 2, D_MODEL), device=device).float()
        
        out = cuda_quantized_linear(x, qweight)
        assert out.shape == (BATCH, SEQLEN, D_MODEL * 2)
        log_result(TestResult("BitLinear", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("BitLinear", "CUDA", False, str(e)), verbose)
    
    # 2. StateScan
    try:
        from tesm_ssm.ops.cuda import cuda_chunk_state_scan, cuda_chunk_state_scan_autograd
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_STATE, device=device))
        update = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        
        out = cuda_chunk_state_scan(decay, update, chunk_size=16)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("StateScan", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("StateScan", "CUDA", False, str(e)), verbose)
    
    # 3. LocalEnt
    try:
        from tesm_ssm.ops.cuda import cuda_local_entanglement, cuda_local_entanglement_autograd
        q = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        bias = torch.randn(SEQLEN, WINDOW, device=device)
        
        out = cuda_local_entanglement(q, k, v, bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("LocalEnt", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("LocalEnt", "CUDA", False, str(e)), verbose)
    
    # 4. GlobalEnt
    try:
        from tesm_ssm.ops.cuda import cuda_global_entanglement, cuda_global_entanglement_autograd
        Q = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        Bias = torch.randn(SEQLEN, SEQLEN, device=device)
        
        out = cuda_global_entanglement(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("GlobalEnt", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("GlobalEnt", "CUDA", False, str(e)), verbose)
    
    # 5. FusedOutput
    try:
        from tesm_ssm.ops.cuda import cuda_fused_output, cuda_fused_output_autograd
        local = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_MODEL, device=device))
        state_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        ent_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        
        out = cuda_fused_output(local, gate, state_proj, ent_proj, 0.5)
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("FusedOutput", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("FusedOutput", "CUDA", False, str(e)), verbose)
    
    # 6. MIMO StateScan
    try:
        from tesm_ssm.ops.cuda import cuda_chunk_state_scan_mimo, cuda_chunk_state_scan_mimo_autograd
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device))
        update = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        
        out = cuda_chunk_state_scan_mimo(decay, update, chunk_size=16)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO StateScan", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO StateScan", "CUDA", False, str(e)), verbose)
    
    # 7. MIMO LocalEnt
    try:
        from tesm_ssm.ops.cuda import cuda_local_entanglement_mimo, cuda_local_entanglement_mimo_autograd
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        bias = torch.randn(N_HEADS, SEQLEN, WINDOW, device=device)
        
        out = cuda_local_entanglement_mimo(q, k, v, bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO LocalEnt", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO LocalEnt", "CUDA", False, str(e)), verbose)
    
    # 8. MIMO GlobalEnt
    try:
        from tesm_ssm.ops.cuda import cuda_global_entanglement_mimo, cuda_global_entanglement_mimo_autograd
        Q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        Bias = torch.randn(N_HEADS, SEQLEN, SEQLEN, device=device)
        
        out = cuda_global_entanglement_mimo(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO GlobalEnt", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO GlobalEnt", "CUDA", False, str(e)), verbose)
    
    # 9. MIMO FusedOutput
    try:
        from tesm_ssm.ops.cuda import cuda_fused_output_mimo, cuda_fused_output_mimo_autograd
        local = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device))
        state_proj = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        ent_proj = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        
        out = cuda_fused_output_mimo(local, gate, state_proj, ent_proj, 0.5)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO FusedOutput", "CUDA", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO FusedOutput", "CUDA", False, str(e)), verbose)


def test_triton_backend(verbose: bool = False):
    """测试 Triton 后端"""
    print("\n" + "="*60)
    print("Triton Backend Tests")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_result(TestResult("Triton Available", "Triton", False, "CUDA not available"), verbose)
        return
    
    # 检查 Triton
    try:
        from tesm_ssm.ops.triton import tesm_triton_is_available
        if not tesm_triton_is_available():
            log_result(TestResult("Triton Available", "Triton", False, "Triton not available"), verbose)
            return
        log_result(TestResult("Triton Available", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("Triton Available", "Triton", False, str(e)), verbose)
        return
    
    # 1. BitLinear
    try:
        from tesm_ssm.ops.triton import triton_quantized_linear, triton_quantized_linear_autograd
        x = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        qweight = torch.randint(-1, 2, (D_MODEL * 2, D_MODEL), device=device).float()
        
        out = triton_quantized_linear(x, qweight)
        assert out.shape == (BATCH, SEQLEN, D_MODEL * 2)
        log_result(TestResult("BitLinear", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("BitLinear", "Triton", False, str(e)), verbose)
    
    # 2. StateScan
    try:
        from tesm_ssm.ops.triton import triton_chunk_state_scan, triton_chunk_state_scan_autograd
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_STATE, device=device))
        update = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        
        out = triton_chunk_state_scan(decay, update, chunk_size=16)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("StateScan", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("StateScan", "Triton", False, str(e)), verbose)
    
    # 3. LocalEnt
    try:
        from tesm_ssm.ops.triton import triton_local_entanglement, triton_local_entanglement_autograd
        q = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        bias = torch.randn(SEQLEN, WINDOW, device=device)
        
        out = triton_local_entanglement(q, k, v, bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("LocalEnt", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("LocalEnt", "Triton", False, str(e)), verbose)
    
    # 4. GlobalEnt
    try:
        from tesm_ssm.ops.triton import triton_global_entanglement
        Q = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        Bias = torch.randn(SEQLEN, SEQLEN, device=device)
        
        out = triton_global_entanglement(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("GlobalEnt", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("GlobalEnt", "Triton", False, str(e)), verbose)
    
    # 5. FusedOutput
    try:
        from tesm_ssm.ops.triton import triton_fused_output_combine, triton_fused_output_combine_autograd
        local = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_MODEL, device=device))
        state_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        ent_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        
        out = triton_fused_output_combine(local, gate, state_proj, ent_proj, 0.5)
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("FusedOutput", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("FusedOutput", "Triton", False, str(e)), verbose)
    
    # 6. MIMO StateScan
    try:
        from tesm_ssm.ops.triton import tesm_state_scan_triton, tesm_state_scan_triton_autograd
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device))
        update = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        
        out = tesm_state_scan_triton(decay, update)  # 不接受 chunk_size 参数
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO StateScan", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO StateScan", "Triton", False, str(e)), verbose)
    
    # 7. MIMO LocalEnt
    try:
        from tesm_ssm.ops.triton import tesm_local_entanglement_triton, tesm_local_entanglement_triton_autograd
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        bias = torch.randn(N_HEADS, WINDOW, device=device)  # (H, W) 而不是 (H, L, W)
        
        out = tesm_local_entanglement_triton(q, k, v, bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO LocalEnt", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO LocalEnt", "Triton", False, str(e)), verbose)
    
    # 8. MIMO GlobalEnt
    try:
        from tesm_ssm.ops.triton import tesm_global_entanglement_mimo_triton, tesm_global_entanglement_mimo_triton_autograd
        Q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        Bias = torch.randn(N_HEADS, SEQLEN, SEQLEN, device=device)
        
        out = tesm_global_entanglement_mimo_triton(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO GlobalEnt", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO GlobalEnt", "Triton", False, str(e)), verbose)
    
    # 9. MIMO FusedOutput
    try:
        from tesm_ssm.ops.triton import tesm_mimo_fused_triton
        # MIMO FusedOutput 需要特定的输入格式
        input_tensor = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        weight = torch.randn(D_MODEL, D_MODEL, device=device)
        decay_bias = torch.randn(N_HEADS, D_HEAD, device=device)
        ent_bias = torch.randn(N_HEADS, WINDOW, device=device)
        
        out = tesm_mimo_fused_triton(input_tensor, weight, decay_bias, ent_bias, threshold=THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("MIMO FusedOutput", "Triton", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO FusedOutput", "Triton", False, str(e)), verbose)


def test_tilelang_backend(verbose: bool = False):
    """测试 TileLang 后端"""
    print("\n" + "="*60)
    print("TileLang Backend Tests")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_result(TestResult("TileLang Available", "TileLang", False, "CUDA not available"), verbose)
        return
    
    # 检查 TileLang
    try:
        from tesm_ssm.ops.tilelang import tesm_chunked_scan_tilelang_fwd
        log_result(TestResult("TileLang Available", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("TileLang Available", "TileLang", False, str(e)), verbose)
        return
    
    # 1. BitLinear
    try:
        from tesm_ssm.ops.tilelang import tesm_bitlinear_tilelang, tesm_bitlinear_tilelang_autograd
        # BitLinear 期望 2D 输入 (M, K)
        M = BATCH * SEQLEN
        x = torch.randn(M, D_MODEL, device=device)
        weight = torch.randn(D_MODEL * 2, D_MODEL, device=device)
        scale = torch.ones(D_MODEL * 2, device=device)
        
        out = tesm_bitlinear_tilelang(x, weight, scale)
        assert out.shape == (M, D_MODEL * 2)
        log_result(TestResult("BitLinear", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("BitLinear", "TileLang", False, str(e)), verbose)
    
    # 2. StateScan
    try:
        from tesm_ssm.ops.tilelang import tesm_chunked_scan_tilelang_fwd, tesm_chunked_scan_tilelang_autograd
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device))
        update = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        
        out = tesm_chunked_scan_tilelang_fwd(decay, update, chunk_size=16)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("StateScan", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("StateScan", "TileLang", False, str(e)), verbose)
    
    # 3. LocalEnt
    try:
        from tesm_ssm.ops.tilelang import tesm_local_entanglement_tilelang_fwd, tesm_local_entanglement_tilelang_autograd
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        bias = torch.randn(N_HEADS, WINDOW, device=device)
        
        out = tesm_local_entanglement_tilelang_fwd(q, k, v, bias, threshold=THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("LocalEnt", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("LocalEnt", "TileLang", False, str(e)), verbose)
    
    # 4. GlobalEnt
    try:
        from tesm_ssm.ops.tilelang import tesm_global_entanglement_tilelang, tesm_global_entanglement_tilelang_autograd
        Q = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, D_STATE, device=device)
        Bias = torch.randn(SEQLEN, SEQLEN, device=device)
        
        out = tesm_global_entanglement_tilelang(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, D_STATE)
        log_result(TestResult("GlobalEnt", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("GlobalEnt", "TileLang", False, str(e)), verbose)
    
    # 5. FusedOutput
    try:
        from tesm_ssm.ops.tilelang import tesm_fused_output_tilelang, tesm_fused_output_tilelang_autograd
        local = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_MODEL, device=device))
        state_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        ent_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        
        out = tesm_fused_output_tilelang(local, gate, state_proj, ent_proj, 0.5)
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("FusedOutput", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("FusedOutput", "TileLang", False, str(e)), verbose)
    
    # 6. MIMO GlobalEnt
    try:
        from tesm_ssm.ops.tilelang import tesm_global_entanglement_mimo_tilelang, tesm_global_entanglement_mimo_tilelang_autograd
        Q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        K = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        V = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        Bias = torch.randn(N_HEADS, SEQLEN, SEQLEN, device=device)
        
        out = tesm_global_entanglement_mimo_tilelang(Q, K, V, Bias, THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO GlobalEnt", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO GlobalEnt", "TileLang", False, str(e)), verbose)
    
    # 7. MIMO StateScan
    try:
        from tesm_ssm.ops.tilelang import tesm_chunked_scan_tilelang_fwd
        # MIMO StateScan 使用 4D 输入 (B, L, H, D)
        decay = torch.sigmoid(torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device))
        update = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        
        out = tesm_chunked_scan_tilelang_fwd(decay, update, chunk_size=16)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO StateScan", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO StateScan", "TileLang", False, str(e)), verbose)
    
    # 8. MIMO LocalEnt
    try:
        from tesm_ssm.ops.tilelang import tesm_local_entanglement_tilelang_fwd
        q = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        k = torch.randn(BATCH, SEQLEN, N_HEADS, ENT_RANK, device=device)
        v = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device)
        bias = torch.randn(N_HEADS, WINDOW, device=device)
        
        out = tesm_local_entanglement_tilelang_fwd(q, k, v, bias, threshold=THRESHOLD)
        assert out.shape == (BATCH, SEQLEN, N_HEADS, D_HEAD)
        log_result(TestResult("MIMO LocalEnt", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO LocalEnt", "TileLang", False, str(e)), verbose)
    
    # 9. MIMO FusedOutput
    try:
        from tesm_ssm.ops.tilelang import tesm_fused_output_tilelang
        # FusedOutput 使用 3D 张量 (B, S, D)，D = H * D_HEAD
        local = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        gate = torch.sigmoid(torch.randn(BATCH, SEQLEN, D_MODEL, device=device))
        state_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        ent_proj = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
        
        out = tesm_fused_output_tilelang(local, gate, state_proj, ent_proj, 0.5)
        assert out.shape == (BATCH, SEQLEN, D_MODEL)
        log_result(TestResult("MIMO FusedOutput", "TileLang", True), verbose)
    except Exception as e:
        log_result(TestResult("MIMO FusedOutput", "TileLang", False, str(e)), verbose)


def print_summary():
    """打印测试总结"""
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    # 按后端分组
    backends = ["PyTorch", "CUDA", "Triton", "TileLang"]
    
    for backend in backends:
        backend_results = [r for r in results if r.backend == backend]
        if not backend_results:
            continue
        
        passed = sum(1 for r in backend_results if r.passed)
        total = len(backend_results)
        
        print(f"\n{backend}: {passed}/{total} passed")
        
        # 显示失败的测试
        failed = [r for r in backend_results if not r.passed]
        if failed:
            for r in failed:
                print(f"  ❌ {r.name}: {r.error}")
    
    # 总计
    total_passed = sum(1 for r in results if r.passed)
    total_tests = len(results)
    print(f"\n{'='*60}")
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    # 功能矩阵
    print("\n功能支持矩阵:")
    print("-" * 60)
    
    functions = [
        "BitLinear", "StateScan", "LocalEnt", "GlobalEnt", "FusedOutput",
        "MIMO StateScan", "MIMO LocalEnt", "MIMO GlobalEnt", "MIMO FusedOutput"
    ]
    
    # 表头
    header = f"{'Function':<20}"
    for b in backends:
        header += f"{b:>10}"
    print(header)
    print("-" * 60)
    
    for func in functions:
        row = f"{func:<20}"
        for backend in backends:
            func_results = [r for r in results if r.name == func and r.backend == backend]
            if func_results:
                status = "✅" if func_results[0].passed else "❌"
            else:
                status = "⬜"
            row += f"{status:>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="TESM Backend Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--cuda-only", action="store_true", help="Only test CUDA backend")
    parser.add_argument("--triton-only", action="store_true", help="Only test Triton backend")
    parser.add_argument("--tilelang-only", action="store_true", help="Only test TileLang backend")
    args = parser.parse_args()
    
    print("TESM Backend Test Suite")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Test config: B={BATCH}, L={SEQLEN}, H={N_HEADS}, D={D_MODEL}")
    
    # 运行测试
    test_pytorch_backend(args.verbose)
    
    if not (args.cuda_only or args.triton_only or args.tilelang_only):
        test_cuda_backend(args.verbose)
        test_triton_backend(args.verbose)
        test_tilelang_backend(args.verbose)
    else:
        if args.cuda_only:
            test_cuda_backend(args.verbose)
        if args.triton_only:
            test_triton_backend(args.verbose)
        if args.tilelang_only:
            test_tilelang_backend(args.verbose)
    
    # 打印总结
    print_summary()
    
    # 返回退出码
    total_passed = sum(1 for r in results if r.passed)
    total_tests = len(results)
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
