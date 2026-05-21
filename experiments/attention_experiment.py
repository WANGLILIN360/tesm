"""
注意力机制对比实验：比较三种注意力机制
1. 标准 Attention：O(n²) 复杂度的 Self-Attention
2. Linear Attention：O(n) 复杂度的线性注意力
3. TESM 纠缠机制：基于状态扫描的注意力替代方案
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict
from functools import partial
import time

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.modules.tesm import BitLinear, TESM


# ==================== 自定义 RMSNorm ====================

class RMSNorm(nn.Module):
    """支持 device 参数的 RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


# ==================== 方案 A：标准 Attention ====================

class StandardAttention(nn.Module):
    """标准 Self-Attention：O(n²) 复杂度"""
    
    def __init__(
        self,
        d_model,
        n_heads=8,
        dropout=0.0,
        max_seq_len=2048,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)),
            persistent=False
        )
    
    def forward(self, x, inference_params=None, cross_layer_state=None):
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


# ==================== 方案 B：Linear Attention ====================

class LinearAttention(nn.Module):
    """Linear Attention：O(n) 复杂度"""
    
    def __init__(
        self,
        d_model,
        n_heads=8,
        dropout=0.0,
        max_seq_len=2048,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5  # 添加 scale 属性
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        
        # Feature map for linear attention (ELU + 1)
        self.feature_map = lambda x: F.elu(x) + 1
    
    def forward(self, x, inference_params=None, cross_layer_state=None):
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # ===== 因果线性注意力实现 =====
        # 使用累积求和实现因果性
        
        # Step 1: 计算 KV 累积矩阵
        # k: (B, H, N, D), v: (B, H, N, D)
        # kv: (B, H, N, D, D) - 每个位置的 KV 外积
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)
        
        # Step 2: 累积求和（实现因果）
        # kv_cumsum: (B, H, N, D, D) - 位置 i 包含 0~i 的累积 KV
        kv_cumsum = torch.cumsum(kv, dim=2)
        
        # Step 3: 计算输出
        # q: (B, H, N, D), kv_cumsum: (B, H, N, D, D)
        # output: (B, H, N, D)
        output = torch.einsum('bhnd,bhnde->bhne', q, kv_cumsum)
        
        # Step 4: 计算归一化因子
        # k_cumsum: (B, H, N, D) - 位置 i 包含 0~i 的累积 K
        k_cumsum = torch.cumsum(k, dim=2)
        
        # normalizer: (B, H, N)
        normalizer = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum) + 1e-6
        
        # Step 5: 归一化
        attn_output = output / normalizer.unsqueeze(-1)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(self.dropout(attn_output))
        
        return output


# ==================== 方案 C：TESM 纠缠机制（窗口滑动） ====================

class TESMAttention(TESM):
    """TESM 纠缠机制作为注意力替代（窗口滑动版本）"""
    
    def forward(self, x, inference_params=None, cross_layer_state=None):
        # TESM 的 forward 返回的是输出，不需要额外处理
        return super().forward(x, inference_params, cross_layer_state)


# ==================== 方案 D：TESM 纠缠机制（全局注意力） ====================

class TESMAttentionGlobal(TESM):
    """TESM 纠缠机制 - 全局注意力版本（entanglement_window=0）"""
    
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
        **kwargs,
    ):
        # 强制设置 entanglement_window=0 使用全局注意力
        kwargs['entanglement_window'] = 0
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            ent_rank=ent_rank,
            entanglement_scale=entanglement_scale,
            entanglement_threshold=entanglement_threshold,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bit_eps=bit_eps,
            bit_threshold=bit_threshold,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    
    def forward(self, x, inference_params=None, cross_layer_state=None):
        return super().forward(x, inference_params, cross_layer_state)


# ==================== 测试函数 ====================

def test_attention_pattern(model, model_name="Model", seq_len=64):
    """测试注意力模式"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 创建输入
        batch_size = 1
        x = torch.randn(batch_size, seq_len, model.d_model if hasattr(model, 'd_model') else 512, device=device)
        
        # 获取注意力权重或类似信息
        if hasattr(model, 'n_heads'):
            # 标准 Attention 或 Linear Attention
            q = model.q_proj(x)
            k = model.k_proj(x)
            
            q = q.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * model.scale
            
            # 分析注意力分布
            attn_entropy = []
            for head in range(model.n_heads):
                head_scores = scores[0, head, :, :]
                # 因果 mask
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
                head_scores = head_scores.masked_fill(~causal_mask, float('-inf'))
                head_attn = F.softmax(head_scores, dim=-1)
                
                # 计算熵
                entropy = -(head_attn * torch.log(head_attn + 1e-9)).sum(dim=-1).mean().item()
                attn_entropy.append(entropy)
            
            avg_entropy = sum(attn_entropy) / len(attn_entropy)
            
            print(f"\n{model_name} 注意力模式测试:")
            print(f"  序列长度: {seq_len}")
            print(f"  注意力头数: {model.n_heads}")
            print(f"  平均注意力熵: {avg_entropy:.4f} (越高越分散)")
            print(f"  各头熵值: {[f'{e:.3f}' for e in attn_entropy]}")
            
            return {
                "seq_len": seq_len,
                "n_heads": model.n_heads,
                "avg_entropy": avg_entropy,
                "head_entropies": attn_entropy,
            }
        else:
            # TESM 纠缠机制
            output = model(x)
            
            # 分析纠缠统计
            if hasattr(model, 'last_entanglement_stats') and model.last_entanglement_stats is not None:
                stats = model.last_entanglement_stats
                print(f"\n{model_name} 纠缠模式测试:")
                print(f"  序列长度: {seq_len}")
                print(f"  正纠缠比例: {stats['positive']:.2%}")
                print(f"  负纠缠比例: {stats['negative']:.2%}")
                print(f"  零纠缠比例: {stats['zero']:.2%}")
                
                return {
                    "seq_len": seq_len,
                    "entanglement_stats": stats,
                }
            else:
                print(f"\n{model_name} 纠缠模式测试:")
                print(f"  序列长度: {seq_len}")
                print(f"  无纠缠统计信息")
                
                return {"seq_len": seq_len}


def test_memory_and_speed(model, model_name="Model", seq_len=128, num_runs=10):
    """测试内存占用和速度"""
    model.eval()
    device = next(model.parameters()).device
    
    d_model = model.d_model if hasattr(model, 'd_model') else 512
    
    # 预热
    with torch.no_grad():
        x = torch.randn(1, seq_len, d_model, device=device)
        _ = model(x)
    
    # 测试速度
    times = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(num_runs):
            x = torch.randn(1, seq_len, d_model, device=device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = time.time()
                _ = model(x)
                torch.cuda.synchronize()
                end = time.time()
            else:
                start = time.time()
                _ = model(x)
                end = time.time()
            
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    
    # 测试内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            x = torch.randn(1, seq_len, d_model, device=device)
            _ = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory = 0
    
    print(f"\n{model_name} 性能测试:")
    print(f"  序列长度: {seq_len}")
    print(f"  平均推理时间: {avg_time * 1000:.2f} ms")
    if device.type == 'cuda':
        print(f"  峰值内存: {peak_memory:.2f} MB")
    
    return {
        "seq_len": seq_len,
        "avg_time_ms": avg_time * 1000,
        "peak_memory_mb": peak_memory,
    }


def test_length_scaling(model, model_name="Model", lengths=None):
    """测试长度扩展性"""
    if lengths is None:
        lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    model.eval()
    device = next(model.parameters()).device
    d_model = model.d_model if hasattr(model, 'd_model') else 512
    
    results = {}
    
    for length in lengths:
        try:
            x = torch.randn(1, length, d_model, device=device)
            
            with torch.no_grad():
                # 预热
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 计时
                start = time.time()
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
            
            results[length] = {
                'success': True,
                'time_ms': (end - start) * 1000,
            }
            print(f"  长度 {length}: ✅ 成功, 耗时 {results[length]['time_ms']:.2f} ms")
        except Exception as e:
            results[length] = {
                'success': False,
                'error': str(e),
                'time_ms': 0,
            }
            print(f"  长度 {length}: ❌ 失败 - {str(e)[:50]}")
    
    return results


def test_causality(model, model_name="Model"):
    """测试因果性"""
    model.eval()
    device = next(model.parameters()).device
    d_model = model.d_model if hasattr(model, 'd_model') else 512
    
    with torch.no_grad():
        seq_len = 16
        common_part = torch.randn(1, seq_len // 2, d_model, device=device)
        
        diff_part1 = torch.randn(1, seq_len // 2, d_model, device=device)
        diff_part2 = torch.randn(1, seq_len // 2, d_model, device=device)
        
        seq1 = torch.cat([common_part, diff_part1], dim=1)
        seq2 = torch.cat([common_part, diff_part2], dim=1)
        
        out1 = model(seq1)
        out2 = model(seq2)
        
        if isinstance(out1, tuple):
            out1 = out1[0] if len(out1) > 0 else out1
        if isinstance(out2, tuple):
            out2 = out2[0] if len(out2) > 0 else out2
        
        # 前半部分的输出应该相同
        h1_first_half = out1[0, :seq_len // 2, :]
        h2_first_half = out2[0, :seq_len // 2, :]
        
        first_half_diff = (h1_first_half - h2_first_half).abs().mean().item()
        
        # 后半部分的输出应该不同
        h1_second_half = out1[0, seq_len // 2:, :]
        h2_second_half = out2[0, seq_len // 2:, :]
        
        second_half_diff = (h1_second_half - h2_second_half).abs().mean().item()
        
        print(f"\n{model_name} 因果性测试:")
        print(f"  前半部分差异: {first_half_diff:.6f} (应接近 0)")
        print(f"  后半部分差异: {second_half_diff:.6f} (应大于 0)")
        print(f"  因果性得分: {second_half_diff / (first_half_diff + 1e-8):.2f} (越大越好)")
        
        return {
            "first_half_diff": first_half_diff,
            "second_half_diff": second_half_diff,
            "causality_score": second_half_diff / (first_half_diff + 1e-8),
        }


def run_experiment():
    """运行完整实验：四个方案对比"""
    print("=" * 90)
    print("注意力机制对比实验：四个方案对比")
    print("=" * 90)
    print("\n方案说明:")
    print("  A. 标准 Attention: O(n²) 复杂度的 Self-Attention")
    print("  B. Linear Attention: O(n) 复杂度的线性注意力")
    print("  C. TESM 纠缠机制(窗口滑动): entanglement_window=16")
    print("  D. TESM 纠缠机制(全局注意力): entanglement_window=0")
    
    # 配置
    config = TESMConfig.small()
    config.max_seq_len = 16384  # 增大以支持超长文本
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    print(f"max_seq_len: {config.max_seq_len}")
    
    d_model = config.d_model
    n_heads = 8
    
    # 创建四个模型
    print("\n创建模型...")
    model_a = StandardAttention(d_model, n_heads=n_heads, dropout=config.dropout, 
                                  max_seq_len=config.max_seq_len, device=device)
    model_b = LinearAttention(d_model, n_heads=n_heads, dropout=config.dropout,
                               max_seq_len=config.max_seq_len, device=device)
    
    # TESM 窗口滑动版本
    model_c = TESM(
        d_model=d_model,
        d_state=256,
        expand=2,
        ent_rank=48,
        entanglement_window=16,  # 窗口滑动
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        bit_eps=config.bit_eps,
        bit_threshold=config.bit_threshold,
        device=device,
    )
    
    # TESM 全局版本（使用相对位置偏置，参数量接近窗口版本）
    model_d = TESM(
        d_model=d_model,
        d_state=256,
        expand=2,
        ent_rank=48,
        entanglement_window=0,  # 全局注意力（使用相对位置偏置）
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        bit_eps=config.bit_eps,
        bit_threshold=config.bit_threshold,
        device=device,
    )
    
    # 参数量对比
    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    params_c = sum(p.numel() for p in model_c.parameters())
    params_d = sum(p.numel() for p in model_d.parameters())
    
    print(f"\n参数量对比:")
    print(f"  A. 标准 Attention: {params_a:,}")
    print(f"  B. Linear Attention: {params_b:,}")
    print(f"  C. TESM 纠缠(窗口滑动): {params_c:,}")
    print(f"  D. TESM 纠缠(全局注意力): {params_d:,}")
    print(f"\n  参数差异 vs A:")
    print(f"  B: {params_b - params_a:,}")
    print(f"  C: {params_c - params_a:,}")
    print(f"  D: {params_d - params_a:,}")
    
    # 测试 1: 注意力模式
    print("\n" + "=" * 90)
    print("测试 1: 注意力模式")
    print("=" * 90)
    pattern_a = test_attention_pattern(model_a, "A. 标准 Attention")
    pattern_b = test_attention_pattern(model_b, "B. Linear Attention")
    pattern_c = test_attention_pattern(model_c, "C. TESM 纠缠(窗口滑动)")
    pattern_d = test_attention_pattern(model_d, "D. TESM 纠缠(全局注意力)")
    
    # 测试 2: 内存和速度
    print("\n" + "=" * 90)
    print("测试 2: 内存和速度")
    print("=" * 90)
    perf_a = test_memory_and_speed(model_a, "A. 标准 Attention")
    perf_b = test_memory_and_speed(model_b, "B. Linear Attention")
    perf_c = test_memory_and_speed(model_c, "C. TESM 纠缠(窗口滑动)")
    perf_d = test_memory_and_speed(model_d, "D. TESM 纠缠(全局注意力)")
    
    # 测试 3: 长度扩展性
    print("\n" + "=" * 90)
    print("测试 3: 长度扩展性")
    print("=" * 90)
    len_a = test_length_scaling(model_a, "A. 标准 Attention")
    len_b = test_length_scaling(model_b, "B. Linear Attention")
    len_c = test_length_scaling(model_c, "C. TESM 纠缠(窗口滑动)")
    len_d = test_length_scaling(model_d, "D. TESM 纠缠(全局注意力)")
    
    # 测试 4: 因果性
    print("\n" + "=" * 90)
    print("测试 4: 因果性")
    print("=" * 90)
    caus_a = test_causality(model_a, "A. 标准 Attention")
    caus_b = test_causality(model_b, "B. Linear Attention")
    caus_c = test_causality(model_c, "C. TESM 纠缠(窗口滑动)")
    caus_d = test_causality(model_d, "D. TESM 纠缠(全局注意力)")
    
    # 总结
    print("\n" + "=" * 90)
    print("实验总结")
    print("=" * 90)
    
    print("\n| 指标 | A.标准Attention | B.Linear | C.TESM窗口 | D.TESM全局 |")
    print("|------|-----------------|----------|------------|------------|")
    print(f"| 参数量 | {params_a:,} | {params_b:,} | {params_c:,} | {params_d:,} |")
    print(f"| 推理时间(ms) | {perf_a['avg_time_ms']:.2f} | {perf_b['avg_time_ms']:.2f} | {perf_c['avg_time_ms']:.2f} | {perf_d['avg_time_ms']:.2f} |")
    print(f"| 峰值内存(MB) | {perf_a['peak_memory_mb']:.2f} | {perf_b['peak_memory_mb']:.2f} | {perf_c['peak_memory_mb']:.2f} | {perf_d['peak_memory_mb']:.2f} |")
    print(f"| 因果性得分 | {caus_a['causality_score']:.0f} | {caus_b['causality_score']:.0f} | {caus_c['causality_score']:.0f} | {caus_d['causality_score']:.0f} |")
    
    # 长度扩展性结果
    print("\n长度扩展性结果:")
    print("| 长度 | A | B | C | D |")
    print("|------|---|---|---|---|")
    for length in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        a_ok = len_a.get(length, {}).get('success', False)
        b_ok = len_b.get(length, {}).get('success', False)
        c_ok = len_c.get(length, {}).get('success', False)
        d_ok = len_d.get(length, {}).get('success', False)
        a_time = len_a.get(length, {}).get('time_ms', 0)
        b_time = len_b.get(length, {}).get('time_ms', 0)
        c_time = len_c.get(length, {}).get('time_ms', 0)
        d_time = len_d.get(length, {}).get('time_ms', 0)
        print(f"| {length} | {'✅' if a_ok else '❌'} {a_time:.1f}ms | {'✅' if b_ok else '❌'} {b_time:.1f}ms | {'✅' if c_ok else '❌'} {c_time:.1f}ms | {'✅' if d_ok else '❌'} {d_time:.1f}ms |")
    
    # 分析结论
    print("\n" + "=" * 90)
    print("分析结论")
    print("=" * 90)
    
    print("\n1. 推理速度排名 (越快越好):")
    speed_rankings = sorted([
        ("A. 标准 Attention", perf_a['avg_time_ms']),
        ("B. Linear Attention", perf_b['avg_time_ms']),
        ("C. TESM 纠缠(窗口滑动)", perf_c['avg_time_ms']),
        ("D. TESM 纠缠(全局注意力)", perf_d['avg_time_ms']),
    ], key=lambda x: x[1])
    for i, (name, score) in enumerate(speed_rankings, 1):
        print(f"   {i}. {name}: {score:.2f} ms")
    
    print("\n2. 内存效率排名 (峰值内存越低越好):")
    mem_rankings = sorted([
        ("A. 标准 Attention", perf_a['peak_memory_mb']),
        ("B. Linear Attention", perf_b['peak_memory_mb']),
        ("C. TESM 纠缠(窗口滑动)", perf_c['peak_memory_mb']),
        ("D. TESM 纠缠(全局注意力)", perf_d['peak_memory_mb']),
    ], key=lambda x: x[1])
    for i, (name, score) in enumerate(mem_rankings, 1):
        print(f"   {i}. {name}: {score:.2f} MB")
    
    print("\n3. 因果性排名 (得分越高越好):")
    caus_rankings = sorted([
        ("A. 标准 Attention", caus_a['causality_score']),
        ("B. Linear Attention", caus_b['causality_score']),
        ("C. TESM 纠缠(窗口滑动)", caus_c['causality_score']),
        ("D. TESM 纠缠(全局注意力)", caus_d['causality_score']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(caus_rankings, 1):
        print(f"   {i}. {name}: {score:.0f}")
    
    print("\n4. 复杂度对比:")
    print("   A. 标准 Attention: O(n²) 时间, O(n²) 空间")
    print("   B. Linear Attention: O(n) 时间, O(n) 空间")
    print("   C. TESM 纠缠(窗口滑动): O(n*w) 时间, O(1) 状态空间, w=窗口大小")
    print("   D. TESM 纠缠(全局注意力): O(n²) 时间, O(n²) 纠缠矩阵")
    
    print("\n5. 窗口滑动 vs 全局注意力对比:")
    print(f"   参数量差异: {params_d - params_c:,} (全局多 {params_d - params_c:,} 个纠缠矩阵参数)")
    print(f"   速度差异: {perf_d['avg_time_ms'] - perf_c['avg_time_ms']:.2f} ms")
    print(f"   因果性差异: {caus_d['causality_score'] - caus_c['causality_score']:.0f}")
    
    return {
        "A_标准Attention": {"params": params_a, "pattern": pattern_a, "perf": perf_a, "length": len_a, "causality": caus_a},
        "B_LinearAttention": {"params": params_b, "pattern": pattern_b, "perf": perf_b, "length": len_b, "causality": caus_b},
        "C_TESM窗口滑动": {"params": params_c, "pattern": pattern_c, "perf": perf_c, "length": len_c, "causality": caus_c},
        "D_TESM全局注意力": {"params": params_d, "pattern": pattern_d, "perf": perf_d, "length": len_d, "causality": caus_d},
    }


if __name__ == "__main__":
    results = run_experiment()
