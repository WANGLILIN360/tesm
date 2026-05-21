"""
架构对比测试：TESM (Mamba风格) vs Transformer

测试项目：
1. 训练速度（前向+反向传播）
2. 增量推理速度（单token生成）
3. 收敛性（训练损失曲线）
4. 批处理效率
5. 混合精度性能
"""

import math
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.modules.tesm import TESM


# ==================== 方案 A：标准 Transformer ====================

class TransformerBlock(nn.Module):
    """标准 Transformer Block"""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = None, dropout: float = 0.0, 
                 max_seq_len: int = 2048, device=None):
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # FFN
        self.ff_up = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Self-Attention
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.o_proj(out)
        x = residual + self.dropout(out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ff_up(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ff_down(x)
        x = residual + self.dropout(x)
        
        return x


class TransformerModel(nn.Module):
    """标准 Transformer 模型"""
    
    def __init__(self, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, 
                 d_ff: int = None, dropout: float = 0.0, max_seq_len: int = 2048, 
                 device=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len, device)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ==================== 方案 B：Mamba 风格（简化版） ====================

class MambaBlock(nn.Module):
    """简化的 Mamba 风格 Block（状态空间模型）"""
    
    def __init__(self, d_model: int, d_state: int = 256, dropout: float = 0.0, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 状态空间投影
        self.in_proj = nn.Linear(d_model, d_model + 2 * d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 状态参数
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        residual = x
        x = self.norm(x)
        
        proj = self.in_proj(x)
        u, delta, B = torch.split(proj, [self.d_model, self.d_state, self.d_state], dim=-1)
        
        # 简化的状态扫描
        delta = F.softplus(delta)
        
        # 并行扫描（简化版）
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = h * torch.sigmoid(delta[:, t, :]) + B[:, t, :]
            outputs.append(h)
        
        h_seq = torch.stack(outputs, dim=1)
        out = self.out_proj(u + h_seq.sum(dim=-1, keepdim=True).expand(-1, -1, self.d_model))
        
        return residual + self.dropout(out)


class MambaModel(nn.Module):
    """Mamba 风格模型"""
    
    def __init__(self, d_model: int = 512, n_layers: int = 6, d_state: int = 256,
                 dropout: float = 0.0, device=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout, device)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ==================== 方案 C：TESM（纠缠状态空间模型） ====================

class TESMBlock(nn.Module):
    """TESM Block 包装"""
    
    def __init__(self, d_model: int, d_state: int = 256, ent_rank: int = 48,
                 entanglement_window: int = 16, dropout: float = 0.0,
                 max_seq_len: int = 2048, device=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tesm = TESM(
            d_model=d_model,
            d_state=d_state,
            ent_rank=ent_rank,
            entanglement_window=entanglement_window,
            max_seq_len=max_seq_len,
            dropout=dropout,
            device=device,
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, inference_params=None, cross_layer_state=None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        out = self.tesm(x, inference_params=inference_params, cross_layer_state=cross_layer_state)
        # TESM 可能返回 tuple (output, state) 或单个 output
        if isinstance(out, tuple):
            out = out[0]
        return residual + out


class TESMModel(nn.Module):
    """TESM 模型"""
    
    def __init__(self, d_model: int = 512, n_layers: int = 6, d_state: int = 256,
                 ent_rank: int = 48, entanglement_window: int = 16, dropout: float = 0.0,
                 max_seq_len: int = 2048, device=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            TESMBlock(d_model, d_state, ent_rank, entanglement_window, dropout, max_seq_len, device)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        """为所有层分配推理缓存"""
        # 每层需要独立的缓存
        caches = []
        for layer in self.layers:
            caches.append(layer.tesm.allocate_inference_cache(batch_size, max_seqlen, dtype))
        return {'state_cache': caches}
    
    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        cross_layer_state = None
        # 为每层准备独立的 inference_params
        layer_caches = None
        if inference_params is not None and 'state_cache' in inference_params:
            layer_caches = inference_params['state_cache']
            # 确保 layer_caches 是列表
            if not isinstance(layer_caches, list):
                layer_caches = None
        
        for i, layer in enumerate(self.layers):
            # 为当前层准备 inference_params
            layer_inference_params = None
            if layer_caches is not None and i < len(layer_caches):
                layer_inference_params = {'state_cache': layer_caches[i]}
            
            out = layer(x, inference_params=layer_inference_params, cross_layer_state=cross_layer_state)
            # TESMBlock 返回 tensor，TESM 可能返回 tuple
            if isinstance(out, tuple):
                x, _ = out
            else:
                x = out
            cross_layer_state = getattr(layer.tesm, '_last_cross_layer_state', None)
        return self.norm(x)


# ==================== 测试函数 ====================

def test_training_speed(model, model_name: str, seq_len: int = 128, batch_size: int = 4,
                        n_steps: int = 50, device=None) -> Dict:
    """测试训练速度（前向+反向传播）"""
    model.train()
    d_model = model.d_model
    
    # 预热
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    target = torch.randn(batch_size, seq_len, d_model, device=device)
    
    for _ in range(5):
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
    
    # 测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for _ in range(n_steps):
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        target = torch.randn(batch_size, seq_len, d_model, device=device)
        
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / n_steps * 1000
    
    print(f"\n{model_name} 训练速度测试:")
    print(f"  序列长度: {seq_len}, 批次大小: {batch_size}")
    print(f"  平均每步时间: {avg_time:.2f} ms")
    print(f"  吞吐量: {batch_size * seq_len / (avg_time / 1000):.0f} tokens/s")
    
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "avg_time_ms": avg_time,
        "tokens_per_sec": batch_size * seq_len / (avg_time / 1000),
    }


def test_incremental_inference(model, model_name: str, prompt_len: int = 256, 
                               gen_len: int = 64, device=None) -> Dict:
    """测试增量推理速度（单token生成）"""
    model.eval()
    d_model = model.d_model
    
    # 预填充
    prompt = torch.randn(1, prompt_len, d_model, device=device)
    
    # 初始化推理缓存（如果模型支持）
    inference_params = None
    if hasattr(model, 'allocate_inference_cache'):
        inference_params = {
            'state_cache': model.allocate_inference_cache(batch_size=1, max_seqlen=prompt_len + gen_len)
        }
    
    with torch.no_grad():
        # 预填充
        if inference_params:
            _ = model(prompt, inference_params=inference_params)
        else:
            _ = model(prompt)
        
        # 增量生成
        times = []
        for i in range(gen_len):
            token = torch.randn(1, 1, d_model, device=device)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            if inference_params:
                _ = model(token, inference_params=inference_params)
            else:
                _ = model(token)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    
    cache_status = "使用缓存" if inference_params else "无缓存"
    print(f"\n{model_name} 增量推理测试 ({cache_status}):")
    print(f"  预填充长度: {prompt_len}, 生成长度: {gen_len}")
    print(f"  平均单token时间: {avg_time:.2f} ms")
    print(f"  总生成时间: {total_time:.2f} ms")
    print(f"  生成速度: {1000 / avg_time:.1f} tokens/s")
    
    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "avg_token_time_ms": avg_time,
        "total_time_ms": total_time,
        "tokens_per_sec": 1000 / avg_time,
        "used_cache": inference_params is not None,
    }


def test_convergence(model, model_name: str, seq_len: int = 128, batch_size: int = 8,
                     n_steps: int = 100, lr: float = 1e-3, device=None) -> Dict:
    """测试收敛性"""
    model.train()
    d_model = model.d_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    
    print(f"\n{model_name} 收敛性测试:")
    print(f"  序列长度: {seq_len}, 批次大小: {batch_size}, 步数: {n_steps}")
    
    for step in range(n_steps):
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        target = torch.randn(batch_size, seq_len, d_model, device=device)
        
        out = model(x)
        loss = F.mse_loss(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: loss = {loss.item():.4f}")
    
    return {
        "losses": losses,
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1],
    }


def test_batch_efficiency(model, model_name: str, seq_len: int = 128, 
                          batch_sizes: List[int] = None, device=None) -> Dict:
    """测试批处理效率"""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    
    model.eval()
    d_model = model.d_model
    results = {}
    
    print(f"\n{model_name} 批处理效率测试:")
    print(f"  序列长度: {seq_len}")
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, seq_len, d_model, device=device)
            
            # 预热
            with torch.no_grad():
                _ = model(x)
            
            # 测试
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            avg_time = (end - start) / 10 * 1000
            throughput = bs * seq_len / (avg_time / 1000)
            
            results[bs] = {
                "time_ms": avg_time,
                "throughput": throughput,
            }
            print(f"  批次 {bs:2d}: {avg_time:6.2f} ms, {throughput:8.0f} tokens/s")
        except Exception as e:
            print(f"  批次 {bs:2d}: ❌ OOM")
            results[bs] = {"error": str(e)}
    
    return results


def test_mixed_precision(model, model_name: str, seq_len: int = 128, 
                          batch_size: int = 4, device=None) -> Dict:
    """测试混合精度性能"""
    model.train()
    d_model = model.d_model
    results = {}
    
    print(f"\n{model_name} 混合精度测试:")
    print(f"  序列长度: {seq_len}, 批次大小: {batch_size}")
    
    # FP32
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    target = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 预热
    for _ in range(5):
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(20):
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    fp32_time = (time.time() - start) / 20 * 1000
    
    results["fp32"] = {"time_ms": fp32_time}
    print(f"  FP32: {fp32_time:.2f} ms")
    
    # FP16 (如果支持)
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        model.half()
        x = x.half()
        target = target.half()
        
        # 预热
        for _ in range(5):
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / 20 * 1000
        
        results["fp16"] = {"time_ms": fp16_time}
        results["speedup"] = fp32_time / fp16_time
        print(f"  FP16: {fp16_time:.2f} ms (加速 {fp32_time/fp16_time:.2f}x)")
        
        model.float()
    
    return results


# ==================== 主测试函数 ====================

def run_comparison():
    """运行完整对比测试"""
    print("=" * 90)
    print("架构对比测试：Transformer vs Mamba vs TESM")
    print("=" * 90)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 配置
    config = {
        "d_model": 512,
        "n_layers": 6,
        "dropout": 0.0,
        "max_seq_len": 2048,
        "device": device,
    }
    
    # 创建模型
    print("\n创建模型...")
    
    model_a = TransformerModel(**config).to(device)
    model_b = MambaModel(**config, d_state=256).to(device)
    model_c = TESMModel(**config, d_state=256, ent_rank=48, entanglement_window=16).to(device)
    
    models = {
        "A. Transformer": model_a,
        "B. Mamba": model_b,
        "C. TESM": model_c,
    }
    
    # 参数量对比
    print("\n" + "=" * 90)
    print("参数量对比")
    print("=" * 90)
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,}")
    
    # 测试 1: 训练速度
    print("\n" + "=" * 90)
    print("测试 1: 训练速度")
    print("=" * 90)
    train_results = {}
    for name, model in models.items():
        train_results[name] = test_training_speed(model, name, device=device)
    
    # 测试 2: 增量推理
    print("\n" + "=" * 90)
    print("测试 2: 增量推理")
    print("=" * 90)
    infer_results = {}
    for name, model in models.items():
        infer_results[name] = test_incremental_inference(model, name, device=device)
    
    # 测试 3: 收敛性
    print("\n" + "=" * 90)
    print("测试 3: 收敛性")
    print("=" * 90)
    conv_results = {}
    for name, model in models.items():
        # 重置模型权重
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        conv_results[name] = test_convergence(model, name, device=device)
    
    # 测试 4: 批处理效率
    print("\n" + "=" * 90)
    print("测试 4: 批处理效率")
    print("=" * 90)
    batch_results = {}
    for name, model in models.items():
        batch_results[name] = test_batch_efficiency(model, name, device=device)
    
    # 测试 5: 混合精度
    print("\n" + "=" * 90)
    print("测试 5: 混合精度")
    print("=" * 90)
    mp_results = {}
    for name, model in models.items():
        mp_results[name] = test_mixed_precision(model, name, device=device)
    
    # 总结
    print("\n" + "=" * 90)
    print("实验总结")
    print("=" * 90)
    
    print("\n| 指标 | Transformer | Mamba | TESM |")
    print("|------|-------------|-------|------|")
    print(f"| 参数量 | {sum(p.numel() for p in model_a.parameters()):,} | {sum(p.numel() for p in model_b.parameters()):,} | {sum(p.numel() for p in model_c.parameters()):,} |")
    print(f"| 训练速度(ms) | {train_results['A. Transformer']['avg_time_ms']:.1f} | {train_results['B. Mamba']['avg_time_ms']:.1f} | {train_results['C. TESM']['avg_time_ms']:.1f} |")
    print(f"| 增量推理(ms) | {infer_results['A. Transformer']['avg_token_time_ms']:.1f} | {infer_results['B. Mamba']['avg_token_time_ms']:.1f} | {infer_results['C. TESM']['avg_token_time_ms']:.1f} |")
    print(f"| 收敛最终loss | {conv_results['A. Transformer']['final_loss']:.4f} | {conv_results['B. Mamba']['final_loss']:.4f} | {conv_results['C. TESM']['final_loss']:.4f} |")
    
    print("\n" + "=" * 90)
    print("分析结论")
    print("=" * 90)
    
    # 训练速度排名
    train_speeds = [(name, r['avg_time_ms']) for name, r in train_results.items()]
    train_speeds.sort(key=lambda x: x[1])
    print("\n1. 训练速度排名 (越快越好):")
    for i, (name, t) in enumerate(train_speeds, 1):
        print(f"   {i}. {name}: {t:.2f} ms")
    
    # 增量推理排名
    infer_speeds = [(name, r['avg_token_time_ms']) for name, r in infer_results.items()]
    infer_speeds.sort(key=lambda x: x[1])
    print("\n2. 增量推理排名 (越快越好):")
    for i, (name, t) in enumerate(infer_speeds, 1):
        print(f"   {i}. {name}: {t:.2f} ms/token")
    
    # 收敛性排名
    conv_final = [(name, r['final_loss']) for name, r in conv_results.items()]
    conv_final.sort(key=lambda x: x[1])
    print("\n3. 收敛性排名 (loss越低越好):")
    for i, (name, l) in enumerate(conv_final, 1):
        print(f"   {i}. {name}: {l:.4f}")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_comparison()
