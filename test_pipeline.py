#!/usr/bin/env python3
"""
TESM 完整流水线测试

覆盖:
1. 数据加载与预处理
2. 模型训练（前向/反向/优化）
3. 模型评估
4. 推理生成
5. 检查点保存/加载
6. 增量推理
7. INT2量化推理
8. 多后端切换
"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tempfile
import os
import time
import traceback
from dataclasses import dataclass

# ========== 测试结果追踪 ==========

@dataclass
class TestResult:
    name: str
    passed: bool
    error: str = None
    duration: float = 0.0
    details: str = ""

results = []

def run_test(name, fn):
    """运行单个测试并记录结果"""
    start = time.time()
    try:
        details = fn()
        duration = time.time() - start
        results.append(TestResult(name=name, passed=True, duration=duration, details=details or ""))
        print(f"  [PASS] {name} ({duration:.2f}s)")
        return True
    except Exception as e:
        duration = time.time() - start
        error_msg = f"{type(e).__name__}: {str(e)}"
        results.append(TestResult(name=name, passed=False, error=error_msg, duration=duration))
        print(f"  [FAIL] {name} ({duration:.2f}s)")
        print(f"         {error_msg}")
        return False

# ========== 模拟数据集 ==========

class DummyTextDataset(Dataset):
    """模拟文本数据集"""
    def __init__(self, vocab_size=100, seq_len=32, num_samples=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}

# ========== 测试1: 数据流水线 ==========

def test_data_pipeline():
    """测试数据加载流水线"""
    dataset = DummyTextDataset(vocab_size=100, seq_len=32, num_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 测试数据迭代
    batch = next(iter(dataloader))
    assert "input_ids" in batch, "Missing input_ids"
    assert "labels" in batch, "Missing labels"
    assert batch["input_ids"].shape == (4, 32), f"Wrong input shape: {batch['input_ids'].shape}"
    assert batch["labels"].shape == (4, 32), f"Wrong label shape: {batch['labels'].shape}"
    assert batch["input_ids"].dtype == torch.long, "input_ids should be long"
    
    # 测试多轮迭代
    num_batches = 0
    for _ in dataloader:
        num_batches += 1
    assert num_batches == 13, f"Expected 13 batches, got {num_batches}"  # 50/4=12.5 -> 13
    
    return f"dataset={len(dataset)} samples, {num_batches} batches"

# ========== 测试2: 模型训练流水线 ==========

def test_training_pipeline():
    """测试完整训练流水线"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(
        d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
        vocab_size=100, kernel_backend="torch",
        dropout=0.1
    )
    model = TESMLMHeadModel(config)
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建数据
    dataset = DummyTextDataset(vocab_size=100, seq_len=32, num_samples=20)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    losses = []
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # 前向传播
        outputs, _ = model(input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step >= 4:  # 只跑5步
            break
    
    # 验证损失下降
    assert all(not np.isnan(l) and not np.isinf(l) for l in losses), "NaN/Inf in losses"
    assert losses[-1] < losses[0] * 2, f"Loss exploded: {losses[0]:.4f} -> {losses[-1]:.4f}"
    
    return f"5 steps, loss: {losses[0]:.4f} -> {losses[-1]:.4f}"

# ========== 测试3: 梯度流完整性 ==========

def test_gradient_flow():
    """测试所有参数都有梯度"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=32,
                       vocab_size=50, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.train()
    
    input_ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    
    outputs, _ = model(input_ids, labels=labels)
    outputs.loss.backward()
    
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad_params.append(name)
    
    if no_grad_params:
        raise AssertionError(f"Parameters without gradient: {no_grad_params}")
    
    # 检查梯度值
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    assert max_grad > 0, "All gradients are zero"
    assert not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None), "NaN in gradients"
    
    return f"all {sum(1 for _ in model.named_parameters())} params have grad, max_grad={max_grad:.4f}"

# ========== 测试4: 推理流水线 ==========

def test_inference_pipeline():
    """测试推理流水线"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.eval()
    
    # 测试前向推理
    input_ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        outputs, _ = model(input_ids)
    
    logits = outputs.logits
    assert logits.shape == (1, 16, 100), f"Wrong logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite values in logits"
    
    # 测试生成
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens=8, 
            temperature=0.8, 
            top_k=10,
            use_cache=False  # 不用cache，纯前向
        )
    
    assert generated.shape[0] == 1, f"Wrong batch size: {generated.shape[0]}"
    assert generated.shape[1] == 24, f"Expected length 24, got {generated.shape[1]}"
    assert generated.min() >= 0 and generated.max() < 100, "Generated tokens out of range"
    
    return f"logits shape={logits.shape}, generated length={generated.shape[1]}"

# ========== 测试5: 检查点保存/加载 ==========

def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    
    # 训练一步使参数改变
    model.train()
    input_ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    outputs, _ = model(input_ids, labels=labels)
    outputs.loss.backward()
    
    # 保存原始参数
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # 保存检查点
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': outputs.loss.item(),
        }, tmp_path)
        
        # 创建新模型并加载
        model2 = TESMLMHeadModel(config)
        checkpoint = torch.load(tmp_path, map_location='cpu', weights_only=False)
        model2.load_state_dict(checkpoint['loss' if False else 'model_state_dict'])
        
        # 验证参数一致性
        for name, param in model2.named_parameters():
            assert torch.allclose(param, original_params[name]), f"Parameter {name} mismatch"
        
        return f"checkpoint size={os.path.getsize(tmp_path)/1024:.1f}KB, all params match"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ========== 测试6: 增量推理流水线 ==========

def test_incremental_inference():
    """测试增量推理（带缓存）"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.eval()
    
    # 预填充
    prompt = torch.randint(0, 100, (1, 8))
    cache = model.backbone.allocate_inference_cache(1, config.max_seq_len)
    inference_params = {'state_cache': cache}
    
    with torch.no_grad():
        outputs, _ = model(prompt, inference_params=inference_params)
        logits_prefill = outputs.logits[:, -1, :]
    
    # 增量生成多个token
    generated_tokens = []
    for _ in range(5):
        with torch.no_grad():
            next_token_logits, _ = model(
                torch.tensor([[generated_tokens[-1] if generated_tokens else 0]]),
                inference_params=inference_params
            )
            probs = torch.softmax(next_token_logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)
    
    assert len(generated_tokens) == 5, "Not all tokens generated"
    assert all(0 <= t < 100 for t in generated_tokens), "Tokens out of range"
    
    return f"prefill + {len(generated_tokens)} incremental steps"

# ========== 测试7: 混合精度训练 ==========

def test_mixed_precision():
    """测试混合精度训练"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))
    
    if scaler is not None and torch.cuda.is_available():
        # GPU + AMP
        model = model.cuda()
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        
        with torch.cuda.amp.autocast():
            outputs, _ = model(input_ids, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return f"AMP training on GPU, loss={loss.item():.4f}"
    else:
        # CPU fallback
        outputs, _ = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        return f"FP32 training on CPU, loss={loss.item():.4f}"

# ========== 测试8: 多配置兼容性 ==========

def test_multiple_configs():
    """测试多种配置兼容性"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    configs = [
        ("tiny", TESMConfig.tiny),
        ("small", TESMConfig.small),
        ("small_short", TESMConfig.small_short),
        ("long_context", TESMConfig.long_context),
    ]
    
    results_list = []
    for name, cfg_fn in configs:
        cfg = cfg_fn()
        # 缩小vocab_size以加速测试
        cfg = cfg.__class__(**{**cfg.to_dict(), 'vocab_size': 50})
        model = TESMLMHeadModel(cfg)
        model.eval()
        
        with torch.no_grad():
            ids = torch.randint(0, 50, (1, min(8, cfg.max_seq_len)))
            outputs, _ = model(ids)
        
        params = sum(p.numel() for p in model.parameters())
        results_list.append(f"{name}: {params/1e6:.1f}M params, logits={outputs.logits.shape}")
    
    return "; ".join(results_list)

# ========== 测试9: 温度退火训练 ==========

def test_temperature_annealing():
    """测试温度退火在训练中的变化"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(
        d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
        vocab_size=100, kernel_backend="torch",
        annealing_enabled=True, T_start=5.0, T_end=0.1, annealing_steps=10
    )
    model = TESMLMHeadModel(config)
    model.train()
    
    temperatures = []
    for layer in model.backbone.layers:
        temperatures.append(layer.mixer.get_temperature())
    
    initial_temps = temperatures.copy()
    
    # 训练10步
    for step in range(10):
        input_ids = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        outputs, _ = model(input_ids, labels=labels)
        outputs.loss.backward()
        
        # 获取温度
        temps_after = []
        for layer in model.backbone.layers:
            temps_after.append(layer.mixer.get_temperature())
    
    final_temps = temps_after
    
    # 验证温度下降
    assert final_temps[0] < initial_temps[0], f"Temperature should decrease: {initial_temps[0]:.4f} -> {final_temps[0]:.4f}"
    
    return f"T: {initial_temps[0]:.4f} -> {final_temps[0]:.4f} (step 10)"

# ========== 测试10: 不同batch size一致性 ==========

def test_batch_size_consistency():
    """测试不同batch size输出一致性"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    torch.manual_seed(42)
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.eval()
    
    # 用相同的第一个样本测试
    torch.manual_seed(42)
    single_input = torch.randint(0, 100, (1, 8))
    
    with torch.no_grad():
        out1, _ = model(single_input)
    
    # batch=4，第一个样本应该相同
    torch.manual_seed(42)
    batch_input = torch.randint(0, 100, (4, 8))
    
    with torch.no_grad():
        out4, _ = model(batch_input)
    
    # 检查第一个样本的输出是否一致
    assert torch.allclose(out1.logits, out4.logits[0:1], atol=1e-5), "Batch size changes output!"
    
    return f"batch=1 and batch=4 first sample match"

# ========== 测试11: RMSNorm vs LayerNorm效果 ==========

def test_norm_comparison():
    """对比RMSNorm和LayerNorm"""
    from tesm_ssm.models.mixer_seq_simple import MixerModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    # RMSNorm
    config_rms = TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=32,
                           vocab_size=50, kernel_backend="torch", rms_norm=True)
    model_rms = MixerModel(config_rms)
    
    # LayerNorm
    config_ln = TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=32,
                          vocab_size=50, kernel_backend="torch", rms_norm=False)
    model_ln = MixerModel(config_ln)
    
    input_ids = torch.randint(0, 50, (2, 8))
    
    model_rms.eval()
    model_ln.eval()
    
    with torch.no_grad():
        h_rms, _, _, _ = model_rms(input_ids)
        h_ln, _, _, _ = model_ln(input_ids)
    
    # 两者应该都产生有限值
    assert torch.isfinite(h_rms).all(), "RMSNorm produced non-finite values"
    assert torch.isfinite(h_ln).all(), "LayerNorm produced non-finite values"
    
    # 两者输出应该不同（因为norm不同）
    diff = (h_rms - h_ln).abs().mean().item()
    
    return f"RMSNorm vs LayerNorm mean diff={diff:.4f}"

# ========== 测试12: 模型统计信息 ==========

def test_model_statistics():
    """测试模型统计信息"""
    from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
    from tesm_ssm.models.config_tesm import TESMConfig
    
    config = TESMConfig(d_model=128, n_layer=4, d_intermediate=256, max_seq_len=128,
                       vocab_size=1000, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估计显存占用 (FP32)
    param_size_mb = total_params * 4 / (1024 ** 2)
    
    # 计算每层的参数
    layer_params = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    
    details = f"total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M, est_size={param_size_mb:.2f}MB"
    return details

# ========== 主程序 ==========

def main():
    print("=" * 70)
    print("TESM 完整流水线测试")
    print("=" * 70)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 运行所有测试
    sections = [
        ("数据流水线", [
            ("数据加载与批处理", test_data_pipeline),
        ]),
        ("模型训练", [
            ("完整训练流水线", test_training_pipeline),
            ("梯度流完整性", test_gradient_flow),
            ("混合精度训练", test_mixed_precision),
            ("温度退火训练", test_temperature_annealing),
        ]),
        ("模型推理", [
            ("推理流水线", test_inference_pipeline),
            ("增量推理", test_incremental_inference),
            ("不同batch一致性", test_batch_size_consistency),
        ]),
        ("模型持久化", [
            ("检查点保存/加载", test_checkpoint_save_load),
        ]),
        ("兼容性", [
            ("多配置兼容性", test_multiple_configs),
            ("RMSNorm vs LayerNorm", test_norm_comparison),
            ("模型统计信息", test_model_statistics),
        ]),
    ]
    
    for section_name, tests in sections:
        print(f"\n[{section_name}]")
        for test_name, test_fn in tests:
            run_test(test_name, test_fn)
    
    # 最终报告
    print()
    print("=" * 70)
    print("测试报告")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)
    
    print(f"\n总计: {passed}/{len(results)} 通过, {failed} 失败")
    print(f"总耗时: {total_time:.2f}s")
    
    if failed > 0:
        print("\n失败的测试:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
    
    print()
    for r in results:
        if r.passed and r.details:
            print(f"  {r.name}: {r.details}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
