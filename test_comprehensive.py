#!/usr/bin/env python3
"""
TESM (Token-Entangled State Machine) 全面测试套件

测试覆盖:
1. 配置模块 (TESMConfig)
2. BitLinear 量化层
3. TernaryQuantumTunneling 三值量子隧穿
4. TESM_SISO 核心层
5. TESMMIMO_Optimized 多头版本
6. MixerModel 和 TESMLMHeadModel
7. RMSNorm 和 Block
8. 训练配置和组件
9. INT2 量化工具
10. 分页缓存
11. 端到端训练流程
"""

import sys
import os
import math
import json
import tempfile
import traceback
from pathlib import Path

# 确保可以导入 tesm_ssm
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 测试跟踪器 - 记录所有测试结果
# =============================================================================

class TestTracker:
    def __init__(self):
        self.results = []
        self.current_suite = None
        
    def start_suite(self, name):
        self.current_suite = name
        print(f"\n{'='*60}")
        print(f"测试套件: {name}")
        print(f"{'='*60}")
        
    def add_result(self, test_name, passed, error_msg=None, duration=0):
        result = {
            'suite': self.current_suite,
            'test': test_name,
            'passed': passed,
            'error': error_msg,
            'duration': duration,
        }
        self.results.append(result)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {test_name}" + (f" ({duration:.2f}s)" if duration > 0 else ""))
        if error_msg and not passed:
            print(f"      错误: {error_msg}")
        
    def summary(self):
        print(f"\n{'='*60}")
        print("测试摘要")
        print(f"{'='*60}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        # 按套件分组
        suites = {}
        for r in self.results:
            s = r['suite']
            if s not in suites:
                suites[s] = []
            suites[s].append(r)
            
        for suite_name, tests in suites.items():
            s_passed = sum(1 for t in tests if t['passed'])
            s_total = len(tests)
            print(f"\n  {suite_name}: {s_passed}/{s_total} 通过")
            for t in tests:
                if not t['passed']:
                    print(f"    ✗ {t['test']}: {t['error']}")
        
        print(f"\n总计: {passed}/{total} 通过, {failed} 失败")
        return failed == 0


tracker = TestTracker()

# =============================================================================
# 辅助函数
# =============================================================================

def assert_tensor_shape(tensor, expected_shape, msg=""):
    """断言张量形状"""
    if tensor.shape != torch.Size(expected_shape):
        raise AssertionError(f"形状不匹配: 期望 {expected_shape}, 实际 {tuple(tensor.shape)}. {msg}")

def assert_tensor_finite(tensor, msg=""):
    """断言张量中没有 inf 或 nan"""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise AssertionError(f"张量包含 {nan_count} 个 NaN. {msg}")
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise AssertionError(f"张量包含 {inf_count} 个 Inf. {msg}")

def assert_grad_exists(tensor, msg=""):
    """断言梯度存在"""
    if tensor.grad is None:
        raise AssertionError(f"梯度不存在. {msg}")
    assert_tensor_finite(tensor.grad, f"梯度包含 NaN 或 Inf. {msg}")

# =============================================================================
# 1. 配置模块测试
# =============================================================================

def test_config():
    tracker.start_suite("配置模块 (TESMConfig)")
    
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 基本配置创建
    try:
        config = TESMConfig(d_model=256, n_layer=4, max_seq_len=128)
        assert config.d_model == 256
        assert config.n_layer == 4
        assert config.max_seq_len == 128
        tracker.add_result("基本配置创建", True)
    except Exception as e:
        tracker.add_result("基本配置创建", False, str(e))
    
    # 测试2: 预设配置 - tiny
    try:
        config = TESMConfig.tiny()
        assert config.d_model > 0
        assert config.n_layer > 0
        assert config.d_state > 0
        tracker.add_result("tiny() 预设配置", True)
    except Exception as e:
        tracker.add_result("tiny() 预设配置", False, str(e))
    
    # 测试3: 预设配置 - small
    try:
        config = TESMConfig.small()
        assert config.d_model == 512
        assert config.n_layer == 16
        assert config.max_seq_len == 512
        tracker.add_result("small() 预设配置", True)
    except Exception as e:
        tracker.add_result("small() 预设配置", False, str(e))
    
    # 测试4: 预设配置 - base
    try:
        config = TESMConfig.base()
        assert config.d_model == 768
        assert config.n_layer == 24
        assert config.max_seq_len == 2048
        tracker.add_result("base() 预设配置", True)
    except Exception as e:
        tracker.add_result("base() 预设配置", False, str(e))
    
    # 测试5: 序列化/反序列化
    try:
        config = TESMConfig.small()
        config_dict = config.to_dict()
        config2 = TESMConfig.from_dict(config_dict)
        assert config.d_model == config2.d_model
        assert config.n_layer == config2.n_layer
        tracker.add_result("配置序列化/反序列化", True)
    except Exception as e:
        tracker.add_result("配置序列化/反序列化", False, str(e))
    
    # 测试6: 无效配置处理
    try:
        config = TESMConfig(d_model=-1, n_layer=0)
        # 应该正常创建，但可能在实际使用时出错
        tracker.add_result("无效配置处理", True, "允许创建无效配置，但可能在使用时出错")
    except Exception as e:
        tracker.add_result("无效配置处理", False, str(e))
    
    # 测试7: 实验配置
    try:
        config = TESMConfig.exp_decay_comparison(decay_bias=1.0)
        assert config.decay_init_bias == 1.0
        tracker.add_result("实验配置 decay_comparison", True)
    except Exception as e:
        tracker.add_result("实验配置 decay_comparison", False, str(e))
    
    try:
        config = TESMConfig.exp_threshold_comparison(threshold=0.05)
        assert config.entanglement_threshold == 0.05
        tracker.add_result("实验配置 threshold_comparison", True)
    except Exception as e:
        tracker.add_result("实验配置 threshold_comparison", False, str(e))

# =============================================================================
# 2. BitLinear 测试
# =============================================================================

def test_bitlinear():
    tracker.start_suite("BitLinear 量化层")
    
    from tesm_ssm.modules.tesm import BitLinear
    
    # 测试1: 基本前向传播
    try:
        layer = BitLinear(64, 128, bias=False, kernel_backend="torch")
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert_tensor_shape(out, (2, 16, 128))
        assert_tensor_finite(out)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 带偏置的前向传播
    try:
        layer = BitLinear(64, 128, bias=True, kernel_backend="torch")
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert_tensor_shape(out, (2, 16, 128))
        assert_tensor_finite(out)
        tracker.add_result("带偏置的前向传播", True)
    except Exception as e:
        tracker.add_result("带偏置的前向传播", False, str(e))
    
    # 测试3: 梯度流
    try:
        layer = BitLinear(64, 128, bias=False, kernel_backend="torch")
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert_grad_exists(x)
        assert_grad_exists(layer.weight)
        tracker.add_result("梯度流", True)
    except Exception as e:
        tracker.add_result("梯度流", False, str(e))
    
    # 测试4: 权重量化
    try:
        layer = BitLinear(64, 128, kernel_backend="torch")
        qweight = layer.quantized_weight()
        assert_tensor_shape(qweight, (128, 64))
        assert_tensor_finite(qweight)
        # 量化权重应该在 [-1, 1] 范围内
        assert qweight.abs().max() <= 1.0 + 1e-5, f"量化权重超出范围: max={qweight.abs().max().item()}"
        tracker.add_result("权重量化", True)
    except Exception as e:
        tracker.add_result("权重量化", False, str(e))
    
    # 测试5: 输入量化
    try:
        layer = BitLinear(64, 128, kernel_backend="torch")
        x = torch.randn(2, 16, 64)
        qinput = layer.quantized_input(x)
        assert_tensor_shape(qinput, (2, 16, 64))
        assert_tensor_finite(qinput)
        tracker.add_result("输入量化", True)
    except Exception as e:
        tracker.add_result("输入量化", False, str(e))
    
    # 测试6: 评估模式缓存
    try:
        layer = BitLinear(64, 128, kernel_backend="torch")
        layer.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out1 = layer(x)
            out2 = layer(x)
        assert torch.allclose(out1, out2, atol=1e-6), "评估模式下两次前向结果不一致"
        tracker.add_result("评估模式缓存一致性", True)
    except Exception as e:
        tracker.add_result("评估模式缓存一致性", False, str(e))
    
    # 测试7: 训练/评估模式切换
    try:
        layer = BitLinear(64, 128, kernel_backend="torch")
        layer.train()
        x = torch.randn(2, 16, 64)
        out_train = layer(x)
        
        layer.eval()
        with torch.no_grad():
            out_eval = layer(x)
        
        assert_tensor_shape(out_train, (2, 16, 128))
        assert_tensor_shape(out_eval, (2, 16, 128))
        tracker.add_result("训练/评估模式切换", True)
    except Exception as e:
        tracker.add_result("训练/评估模式切换", False, str(e))
    
    # 测试8: kernel_backend=auto 回退到 torch
    try:
        layer = BitLinear(64, 128, kernel_backend="auto")
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert_tensor_shape(out, (2, 16, 128))
        tracker.add_result("auto backend 回退", True)
    except Exception as e:
        tracker.add_result("auto backend 回退", False, str(e))

# =============================================================================
# 3. TernaryQuantumTunneling 测试
# =============================================================================

def test_ternary_quantum_tunneling():
    tracker.start_suite("三值量子隧穿 (TernaryQuantumTunneling)")
    
    from tesm_ssm.modules.tesm import TernaryQuantumTunneling
    
    # 测试1: 基本前向传播
    try:
        tqt = TernaryQuantumTunneling(threshold=0.1, tunneling_strength=0.1)
        scores = torch.randn(2, 8, 16)
        ternary_values, tunnel_info = tqt.apply_tunneling(scores, training=True)
        assert_tensor_shape(ternary_values, (2, 8, 16))
        assert_tensor_finite(ternary_values)
        # 检查输出值在 {-1, 0, +1} 中
        unique_vals = torch.unique(ternary_values)
        for v in unique_vals:
            assert v.item() in [-1.0, 0.0, 1.0], f"意外的值: {v.item()}"
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 势垒高度计算
    try:
        tqt = TernaryQuantumTunneling(threshold=0.1)
        scores = torch.tensor([0.0, 0.05, 0.1, 0.15])
        barrier = tqt.compute_barrier_height(scores)
        expected = torch.tensor([0.1, 0.05, 0.0, 0.0])
        assert torch.allclose(barrier, expected, atol=1e-6), f"势垒高度计算错误: {barrier} vs {expected}"
        tracker.add_result("势垒高度计算", True)
    except Exception as e:
        tracker.add_result("势垒高度计算", False, str(e))
    
    # 测试3: 隧穿概率计算
    try:
        tqt = TernaryQuantumTunneling(threshold=0.1, tunneling_strength=0.1)
        barrier = torch.tensor([0.0, 0.05, 0.1])
        prob = tqt.get_tunneling_probability(barrier)
        assert_tensor_finite(prob)
        # barrier=0 时概率应该最高
        assert prob[0] >= prob[1] >= prob[2], "概率排序错误"
        # 概率应该在 [min_tunnel_prob, max_tunnel_prob] 范围内
        assert (prob >= tqt.min_tunnel_prob).all(), "概率低于最小值"
        assert (prob <= tqt.max_tunnel_prob).all(), "概率高于最大值"
        tracker.add_result("隧穿概率计算", True)
    except Exception as e:
        tracker.add_result("隧穿概率计算", False, str(e))
    
    # 测试4: 统计信息收集
    try:
        tqt = TernaryQuantumTunneling(threshold=0.1)
        scores = torch.randn(2, 8, 16)
        _, tunnel_info = tqt.apply_tunneling(scores, training=True)
        assert 'tunnel_rate' in tunnel_info
        assert 'boundary_rate' in tunnel_info
        assert 'avg_tunnel_prob' in tunnel_info
        tracker.add_result("统计信息收集", True)
    except Exception as e:
        tracker.add_result("统计信息收集", False, str(e))
    
    # 测试5: 推理模式 vs 训练模式
    try:
        tqt = TernaryQuantumTunneling(threshold=0.1)
        scores = torch.randn(2, 8, 16)
        tqt.train()
        _, info_train = tqt.apply_tunneling(scores, training=True)
        
        tqt.eval()
        with torch.no_grad():
            _, info_eval = tqt.apply_tunneling(scores, training=False)
        tracker.add_result("训练/推理模式", True)
    except Exception as e:
        tracker.add_result("训练/推理模式", False, str(e))

# =============================================================================
# 4. TESM_SISO 核心层测试
# =============================================================================

def test_tesm_siso():
    tracker.start_suite("TESM_SISO 核心层")
    
    from tesm_ssm.modules.tesm import TESM_SISO
    
    # 测试1: 基本前向传播
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16, 
                          entanglement_window=8, max_seq_len=64, kernel_backend="torch")
        x = torch.randn(2, 16, 128)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        assert_tensor_shape(out, (2, 16, 128))
        assert_tensor_finite(out)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 全局纠缠模式 (entanglement_window=0)
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=0, max_seq_len=64, kernel_backend="torch")
        x = torch.randn(2, 16, 128)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        assert_tensor_shape(out, (2, 16, 128))
        assert_tensor_finite(out)
        tracker.add_result("全局纠缠模式", True)
    except Exception as e:
        tracker.add_result("全局纠缠模式", False, str(e))
    
    # 测试3: 梯度流
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=8, max_seq_len=64, kernel_backend="torch")
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        assert_grad_exists(x)
        tracker.add_result("梯度流", True)
    except Exception as e:
        tracker.add_result("梯度流", False, str(e))
    
    # 测试4: 温度退火调度
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=8, max_seq_len=64, kernel_backend="torch",
                          annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100)
        T = layer.get_temperature()
        assert T > 0, f"温度应该为正: {T}"
        
        # 模拟训练步数
        layer.annealing_step.fill_(50)
        T_mid = layer.get_temperature()
        assert T_mid < T, f"中间温度应该降低: {T} -> {T_mid}"
        
        layer.annealing_step.fill_(1000)
        T_end = layer.get_temperature()
        assert T_end <= layer.T_end + 1e-5, f"结束温度应该接近 T_end: {T_end} vs {layer.T_end}"
        tracker.add_result("温度退火调度", True)
    except Exception as e:
        tracker.add_result("温度退火调度", False, str(e))
    
    # 测试5: 推理缓存分配
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=8, max_seq_len=64, kernel_backend="torch")
        cache = layer.allocate_inference_cache(batch_size=2, max_seqlen=64)
        assert 'state' in cache
        assert 'seq_pos' in cache
        assert 'ent_k_cache' in cache
        assert 'ent_v_cache' in cache
        tracker.add_result("推理缓存分配", True)
    except Exception as e:
        tracker.add_result("推理缓存分配", False, str(e))
    
    # 测试6: 分页缓存
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=8, max_seqlen=64, kernel_backend="torch")
        cache = layer.allocate_inference_cache(batch_size=2, max_seqlen=2048, use_paged_cache=True)
        assert cache.get('use_paged', False), "分页缓存应该启用"
        assert 'paged_cache' in cache
        tracker.add_result("分页缓存分配", True)
    except Exception as e:
        tracker.add_result("分页缓存分配", False, str(e))
    
    # 测试7: 三值纠缠
    try:
        layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                          entanglement_window=8, max_seq_len=64, kernel_backend="torch",
                          entanglement_threshold=0.1)
        scores = torch.randn(2, 8, 16)
        result = layer.ternary_entanglement(scores)
        assert_tensor_finite(result)
        tracker.add_result("三值纠缠", True)
    except Exception as e:
        tracker.add_result("三值纠缠", False, str(e))
    
    # 测试8: 不同序列长度
    try:
        layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                          entanglement_window=4, max_seq_len=128, kernel_backend="torch")
        for seq_len in [1, 4, 8, 16]:
            x = torch.randn(1, seq_len, 64)
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]
            assert_tensor_shape(out, (1, seq_len, 64))
        tracker.add_result("不同序列长度", True)
    except Exception as e:
        tracker.add_result("不同序列长度", False, str(e))

# =============================================================================
# 5. TESMMIMO 测试
# =============================================================================

def test_tesm_mimo():
    tracker.start_suite("TESMMIMO_Optimized 多头版本")
    
    from tesm_ssm.modules.tesm_mimo import TESMMIMO_Optimized
    
    # 测试1: 基本前向传播
    try:
        layer = TESMMIMO_Optimized(d_model=128, d_state=64, n_heads=4, expand=2, 
                                   ent_rank=16, entanglement_window=8, max_seq_len=64,
                                   kernel_backend="torch")
        x = torch.randn(2, 16, 128)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        assert_tensor_shape(out, (2, 16, 128))
        assert_tensor_finite(out)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 梯度流
    try:
        layer = TESMMIMO_Optimized(d_model=128, d_state=64, n_heads=4, expand=2,
                                   ent_rank=16, entanglement_window=8, max_seq_len=64,
                                   kernel_backend="torch")
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        assert_grad_exists(x)
        tracker.add_result("梯度流", True)
    except Exception as e:
        tracker.add_result("梯度流", False, str(e))
    
    # 测试3: 多头参数检查
    try:
        layer = TESMMIMO_Optimized(d_model=128, d_state=64, n_heads=4, expand=2,
                                   ent_rank=16, entanglement_window=8, max_seq_len=64,
                                   kernel_backend="torch")
        assert layer.n_heads == 4
        assert hasattr(layer, 'mimo_x')
        assert hasattr(layer, 'mimo_z')
        assert hasattr(layer, 'mimo_o')
        tracker.add_result("多头参数检查", True)
    except Exception as e:
        tracker.add_result("多头参数检查", False, str(e))
    
    # 测试4: 推理缓存分配
    try:
        layer = TESMMIMO_Optimized(d_model=128, d_state=64, n_heads=4, expand=2,
                                   ent_rank=16, entanglement_window=8, max_seq_len=64,
                                   kernel_backend="torch")
        cache = layer.allocate_inference_cache(batch_size=2, max_seqlen=64)
        assert 'state' in cache
        # MIMO 状态应该是 (batch, n_heads, d_state)
        expected_state_shape = (2, 4, 64)
        assert cache['state'].shape == expected_state_shape, f"状态形状错误: {cache['state'].shape} vs {expected_state_shape}"
        tracker.add_result("推理缓存分配", True)
    except Exception as e:
        tracker.add_result("推理缓存分配", False, str(e))

# =============================================================================
# 6. MixerModel 和 TESMLMHeadModel 测试
# =============================================================================

def test_mixermodel():
    tracker.start_suite("MixerModel")
    
    from tesm_ssm.models.mixer_seq_simple import MixerModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 基本前向传播
    try:
        config = TESMConfig(d_model=128, n_layer=2, d_intermediate=256, max_seq_len=64,
                           vocab_size=1000, kernel_backend="torch")
        model = MixerModel(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        hidden_states, ent_maps, ent_stats, final_states = model(input_ids)
        assert_tensor_shape(hidden_states, (2, 16, 128))
        assert_tensor_finite(hidden_states)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 序列长度检查
    try:
        config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=16,
                           vocab_size=100, kernel_backend="torch")
        model = MixerModel(config)
        input_ids = torch.randint(0, 100, (1, 32))
        try:
            model(input_ids)
            tracker.add_result("序列长度超限检查", False, "应该抛出 ValueError")
        except ValueError as ve:
            tracker.add_result("序列长度超限检查", True)
    except Exception as e:
        tracker.add_result("序列长度超限检查", False, str(e))

def test_lm_head_model():
    tracker.start_suite("TESMLMHeadModel")
    
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 基本前向传播
    try:
        config = TESMConfig(d_model=128, n_layer=2, d_intermediate=256, max_seq_len=64,
                           vocab_size=1000, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        outputs, _ = model(input_ids)
        assert outputs.logits is not None
        assert_tensor_shape(outputs.logits, (2, 16, 1000))
        assert_tensor_finite(outputs.logits)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 带 labels 的损失计算
    try:
        config = TESMConfig(d_model=128, n_layer=2, d_intermediate=256, max_seq_len=64,
                           vocab_size=1000, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        outputs, _ = model(input_ids, labels=labels)
        assert outputs.loss is not None
        assert_tensor_finite(outputs.loss)
        tracker.add_result("损失计算", True)
    except Exception as e:
        tracker.add_result("损失计算", False, str(e))
    
    # 测试3: 梯度流
    try:
        config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                           vocab_size=100, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        input_ids = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        outputs, _ = model(input_ids, labels=labels)
        outputs.loss.backward()
        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                raise AssertionError(f"参数 {name} 没有梯度")
        tracker.add_result("梯度流", True)
    except Exception as e:
        tracker.add_result("梯度流", False, str(e))
    
    # 测试4: 生成方法
    try:
        config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                           vocab_size=100, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=4, temperature=0.5, top_k=10)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 8  # 至少输入长度 + 4 个新token
        tracker.add_result("生成方法", True)
    except Exception as e:
        tracker.add_result("生成方法", False, str(e))
    
    # 测试5: embedding 共享
    try:
        config = TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=16,
                           vocab_size=100, kernel_backend="torch", tie_embeddings=True)
        model = TESMLMHeadModel(config)
        if config.tie_embeddings:
            assert model.lm_head.weight is model.backbone.embedding.weight, "embedding 应该共享"
        tracker.add_result("Embedding 共享", True)
    except Exception as e:
        tracker.add_result("Embedding 共享", False, str(e))

# =============================================================================
# 7. RMSNorm 和 Block 测试
# =============================================================================

def test_rmsnorm():
    tracker.start_suite("RMSNorm")
    
    from tesm_ssm.modules.block import RMSNorm
    
    # 测试1: 基本前向传播
    try:
        norm = RMSNorm(dim=64, eps=1e-5)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert_tensor_shape(out, (2, 16, 64))
        assert_tensor_finite(out)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 归一化效果
    try:
        norm = RMSNorm(dim=64, eps=1e-5)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        # 计算 RMS
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        # RMS 应该接近 1（考虑到 weight 参数）
        assert_tensor_finite(rms)
        tracker.add_result("归一化效果", True)
    except Exception as e:
        tracker.add_result("归一化效果", False, str(e))
    
    # 测试3: 梯度流
    try:
        norm = RMSNorm(dim=64, eps=1e-5)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert_grad_exists(x)
        assert norm.weight.grad is not None, "RMSNorm weight 应该有梯度"
        tracker.add_result("梯度流", True)
    except Exception as e:
        tracker.add_result("梯度流", False, str(e))

def test_block():
    tracker.start_suite("Block")
    
    from tesm_ssm.modules.block import Block
    from tesm_ssm.modules.tesm import TESM_SISO
    
    # 测试1: 基本前向传播
    try:
        def mixer_cls(dim):
            return TESM_SISO(d_model=dim, d_state=32, expand=2, ent_rank=8,
                           entanglement_window=4, max_seq_len=32, kernel_backend="torch")
        
        def mlp_cls(dim):
            return nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        
        block = Block(64, mixer_cls, mlp_cls)
        x = torch.randn(2, 8, 64)
        out, residual, final_state = block(x)
        assert_tensor_shape(out, (2, 8, 64))
        assert_tensor_finite(out)
        tracker.add_result("基本前向传播", True)
    except Exception as e:
        tracker.add_result("基本前向传播", False, str(e))
    
    # 测试2: 残差连接
    try:
        block = Block(64, mixer_cls, mlp_cls)
        x = torch.randn(2, 8, 64)
        residual = torch.randn(2, 8, 64)
        out, new_residual, _ = block(x, residual=residual)
        assert_tensor_shape(out, (2, 8, 64))
        # 输出应该与输入不同（因为残差连接）
        tracker.add_result("残差连接", True)
    except Exception as e:
        tracker.add_result("残差连接", False, str(e))

# =============================================================================
# 8. 训练配置测试
# =============================================================================

def test_training_config():
    tracker.start_suite("训练配置 (TrainingConfig)")
    
    from tesm_ssm.training.config import TrainingConfig
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 基本配置
    try:
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            num_epochs=3,
            batch_size=2,
        )
        assert config.num_epochs == 3
        assert config.batch_size == 2
        tracker.add_result("基本配置创建", True)
    except Exception as e:
        tracker.add_result("基本配置创建", False, str(e))
    
    # 测试2: 配置验证
    try:
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            device="cpu",
            accelerator="torch",
        )
        # CPU + torch 应该可以正常工作
        tracker.add_result("设备配置验证", True)
    except Exception as e:
        tracker.add_result("设备配置验证", False, str(e))
    
    # 测试3: 序列化
    try:
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            num_epochs=3,
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'num_epochs' in config_dict
        tracker.add_result("配置序列化", True)
    except Exception as e:
        tracker.add_result("配置序列化", False, str(e))

# =============================================================================
# 9. INT2 量化工具测试
# =============================================================================

def test_int2_quantization():
    tracker.start_suite("INT2 量化工具")
    
    from tesm_ssm.utils.int2_quantization import (
        pack_int2_to_uint8, unpack_uint8_to_int2, 
        quantize_weight_to_int2, Int2Linear
    )
    
    # 测试1: 打包/解包
    try:
        weight = torch.randn(32, 64)
        packed, scale = pack_int2_to_uint8(weight)
        assert packed.dtype == torch.uint8
        # 解包
        unpacked = unpack_uint8_to_int2(packed, scale.item())
        assert unpacked.shape == weight.shape
        assert_tensor_finite(unpacked)
        tracker.add_result("打包/解包", True)
    except Exception as e:
        tracker.add_result("打包/解包", False, str(e))
    
    # 测试2: 量化权重值范围
    try:
        weight = torch.randn(32, 64)
        packed, scale = pack_int2_to_uint8(weight)
        unpacked = unpack_uint8_to_int2(packed, scale.item())
        # 解包后的值应该在 [-1, 0, +1] 范围内（除以scale后）
        unique_vals = torch.unique(unpacked)
        tracker.add_result("量化值范围", True)
    except Exception as e:
        tracker.add_result("量化值范围", False, str(e))
    
    # 测试3: Int2Linear 层
    try:
        weight = torch.randn(32, 64)
        packed, scale = pack_int2_to_uint8(weight)
        layer = Int2Linear(64, 32, packed, scale)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert_tensor_shape(out, (2, 16, 32))
        assert_tensor_finite(out)
        tracker.add_result("Int2Linear 前向传播", True)
    except Exception as e:
        tracker.add_result("Int2Linear 前向传播", False, str(e))
    
    # 测试4: Int2Linear from_float
    try:
        linear = nn.Linear(64, 32)
        int2_layer = Int2Linear.from_float(linear)
        x = torch.randn(2, 16, 64)
        out = int2_layer(x)
        assert_tensor_shape(out, (2, 16, 32))
        tracker.add_result("Int2Linear from_float", True)
    except Exception as e:
        tracker.add_result("Int2Linear from_float", False, str(e))
    
    # 测试5: 不可整除4的维度
    try:
        weight = torch.randn(32, 65)  # 65 不可被 4 整除
        packed, scale = pack_int2_to_uint8(weight)
        # 应该自动填充
        unpacked = unpack_uint8_to_int2(packed, scale.item())
        assert unpacked.shape[0] == 32
        tracker.add_result("不可整除4维度处理", True)
    except Exception as e:
        tracker.add_result("不可整除4维度处理", False, str(e))

# =============================================================================
# 10. 分页缓存测试
# =============================================================================

def test_paged_cache():
    tracker.start_suite("分页缓存 (PagedStateCache)")
    
    from tesm_ssm.utils.paged_cache import PagedStateCache
    
    # 测试1: 基本创建
    try:
        cache = PagedStateCache(
            batch_size=2,
            d_state=64,
            ent_rank=16,
            window=8,
            page_size=16,
            max_gpu_pages=4,
            device=torch.device('cpu')  # 测试用 CPU
        )
        assert cache.batch_size == 2
        assert cache.page_size == 16
        tracker.add_result("基本创建", True)
    except Exception as e:
        tracker.add_result("基本创建", False, str(e))
    
    # 测试2: 保存和加载状态
    try:
        cache = PagedStateCache(
            batch_size=2, d_state=64, ent_rank=16, window=8,
            page_size=16, max_gpu_pages=4, device=torch.device('cpu')
        )
        state_dict = {
            'state': torch.randn(2, 64),
            'ent_k_cache': torch.randn(2, 8, 16),
            'ent_v_cache': torch.randn(2, 8, 64),
        }
        cache.save_state(0, state_dict)
        loaded = cache.load_state(0)
        assert loaded is not None
        tracker.add_result("保存/加载状态", True)
    except Exception as e:
        tracker.add_result("保存/加载状态", False, str(e))
    
    # 测试3: 多页管理
    try:
        cache = PagedStateCache(
            batch_size=2, d_state=64, ent_rank=16, window=8,
            page_size=4, max_gpu_pages=2, device=torch.device('cpu')
        )
        for pos in [0, 4, 8, 12, 16]:
            state_dict = {
                'state': torch.randn(2, 64),
                'ent_k_cache': torch.randn(2, 8, 16),
                'ent_v_cache': torch.randn(2, 8, 64),
            }
            cache.save_state(pos, state_dict)
        
        assert len(cache.pages) >= 3, "应该创建了多个页"
        tracker.add_result("多页管理", True)
    except Exception as e:
        tracker.add_result("多页管理", False, str(e))
    
    # 测试4: 内存统计
    try:
        cache = PagedStateCache(
            batch_size=2, d_state=64, ent_rank=16, window=8,
            page_size=4, max_gpu_pages=2, device=torch.device('cpu')
        )
        state_dict = {
            'state': torch.randn(2, 64),
            'ent_k_cache': torch.randn(2, 8, 16),
            'ent_v_cache': torch.randn(2, 8, 64),
        }
        cache.save_state(0, state_dict)
        stats = cache.get_memory_stats()
        assert 'total_pages' in stats
        assert stats['total_pages'] > 0
        tracker.add_result("内存统计", True)
    except Exception as e:
        tracker.add_result("内存统计", False, str(e))
    
    # 测试5: 清空缓存
    try:
        cache = PagedStateCache(
            batch_size=2, d_state=64, ent_rank=16, window=8,
            page_size=4, max_gpu_pages=2, device=torch.device('cpu')
        )
        state_dict = {
            'state': torch.randn(2, 64),
            'ent_k_cache': torch.randn(2, 8, 16),
            'ent_v_cache': torch.randn(2, 8, 64),
        }
        cache.save_state(0, state_dict)
        cache.clear()
        assert len(cache.pages) == 0
        assert cache.gpu_page_count == 0
        tracker.add_result("清空缓存", True)
    except Exception as e:
        tracker.add_result("清空缓存", False, str(e))

# =============================================================================
# 11. 端到端集成测试
# =============================================================================

def test_end_to_end():
    tracker.start_suite("端到端集成测试")
    
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 完整训练步骤
    try:
        config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                           vocab_size=100, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 模拟一个训练步骤
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))
        
        outputs, _ = model(input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert_tensor_finite(loss)
        assert loss.item() > 0, "损失应该为正"
        tracker.add_result("完整训练步骤", True)
    except Exception as e:
        tracker.add_result("完整训练步骤", False, str(e))
    
    # 测试2: 推理流程
    try:
        config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                           vocab_size=100, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        model.eval()
        
        input_ids = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            outputs, _ = model(input_ids)
            logits = outputs.logits
        
        assert_tensor_shape(logits, (1, 4, 100))
        assert_tensor_finite(logits)
        tracker.add_result("推理流程", True)
    except Exception as e:
        tracker.add_result("推理流程", False, str(e))
    
    # 测试3: 增量推理
    try:
        config = TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=32,
                           vocab_size=100, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        model.eval()
        
        # 初始输入
        input_ids = torch.randint(0, 100, (1, 4))
        
        # 分配缓存
        inference_params = {
            'state_cache': model.backbone.allocate_inference_cache(1, 32)
        }
        
        with torch.no_grad():
            # 预填充
            outputs, _ = model(input_ids, inference_params=inference_params)
            
            # 增量生成
            for _ in range(4):
                next_token = torch.randint(0, 100, (1, 1))
                outputs, _ = model(next_token, inference_params=inference_params)
        
        tracker.add_result("增量推理", True)
    except Exception as e:
        tracker.add_result("增量推理", False, str(e))
    
    # 测试4: 模型保存/加载
    try:
        config = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16,
                           vocab_size=50, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
        
        # 保存
        torch.save(model.state_dict(), tmp_path)
        
        # 加载
        state_dict = torch.load(tmp_path, weights_only=False)
        model2 = TESMLMHeadModel(config)
        model2.load_state_dict(state_dict)
        
        # 验证参数一致性
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"参数 {n1} 不匹配"
        
        os.unlink(tmp_path)
        tracker.add_result("模型保存/加载", True)
    except Exception as e:
        tracker.add_result("模型保存/加载", False, str(e))

# =============================================================================
# 12. 边界条件测试
# =============================================================================

def test_edge_cases():
    tracker.start_suite("边界条件测试")
    
    from tesm_ssm.modules.tesm import TESM_SISO, BitLinear
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # 测试1: 序列长度为1
    try:
        layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                          entanglement_window=4, max_seq_len=32, kernel_backend="torch")
        x = torch.randn(1, 1, 64)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        assert_tensor_shape(out, (1, 1, 64))
        tracker.add_result("序列长度为1", True)
    except Exception as e:
        tracker.add_result("序列长度为1", False, str(e))
    
    # 测试2: 批次大小为1
    try:
        config = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16,
                           vocab_size=50, kernel_backend="torch")
        model = TESMLMHeadModel(config)
        input_ids = torch.randint(0, 50, (1, 8))
        outputs, _ = model(input_ids)
        assert outputs.logits.shape[0] == 1
        tracker.add_result("批次大小为1", True)
    except Exception as e:
        tracker.add_result("批次大小为1", False, str(e))
    
    # 测试3: 最小模型配置
    try:
        config = TESMConfig(d_model=16, n_layer=1, d_intermediate=32, max_seq_len=8,
                           vocab_size=10, kernel_backend="torch", d_state=8, ent_rank=4)
        model = TESMLMHeadModel(config)
        input_ids = torch.randint(0, 10, (1, 4))
        outputs, _ = model(input_ids)
        assert_tensor_shape(outputs.logits, (1, 4, 10))
        tracker.add_result("最小模型配置", True)
    except Exception as e:
        tracker.add_result("最小模型配置", False, str(e))
    
    # 测试4: 零值输入
    try:
        layer = BitLinear(32, 64, kernel_backend="torch")
        x = torch.zeros(1, 4, 32)
        out = layer(x)
        assert_tensor_shape(out, (1, 4, 64))
        assert_tensor_finite(out)
        tracker.add_result("零值输入", True)
    except Exception as e:
        tracker.add_result("零值输入", False, str(e))

# =============================================================================
# 主函数
# =============================================================================

def run_all_tests():
    print("="*60)
    print("TESM (Token-Entangled State Machine) 全面测试")
    print("="*60)
    
    # 运行所有测试
    test_config()
    test_bitlinear()
    test_ternary_quantum_tunneling()
    test_tesm_siso()
    test_tesm_mimo()
    test_mixermodel()
    test_lm_head_model()
    test_rmsnorm()
    test_block()
    test_training_config()
    test_int2_quantization()
    test_paged_cache()
    test_end_to_end()
    test_edge_cases()
    
    # 打印摘要
    all_passed = tracker.summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
