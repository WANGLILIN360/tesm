#!/usr/bin/env python3
"""TESM 精简流水线测试 - 减少内存占用"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import traceback
from torch.utils.data import Dataset, DataLoader

results = []

def test(name, fn):
    start = time.time()
    try:
        result = fn()
        dur = time.time() - start
        results.append((name, True, dur, result or ""))
        print(f"  [PASS] {name} ({dur:.1f}s)")
        return True
    except Exception as e:
        dur = time.time() - start
        err = f"{type(e).__name__}: {str(e)[:100]}"
        results.append((name, False, dur, err))
        print(f"  [FAIL] {name} ({dur:.1f}s)")
        print(f"         {err}")
        return False

class DummyDataset(Dataset):
    def __init__(self, vocab=50, seqlen=16, n=20):
        self.data = torch.randint(0, vocab, (n, seqlen))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return {"input_ids": self.data[i], "labels": self.data[i]}

from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.models.config_tesm import TESMConfig

# 超小配置
CFG = dict(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=32,
           vocab_size=50, kernel_backend="torch")

def make_model():
    return TESMLMHeadModel(TESMConfig(**CFG))

print("=" * 60)
print("TESM 精简流水线测试")
print("=" * 60)
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ===== 1. 数据流水线 =====
print("\n[1. 数据流水线]")

def t_data():
    ds = DummyDataset(n=16)
    dl = DataLoader(ds, batch_size=4, collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]), "labels": torch.stack([x["labels"] for x in b])})
    batch = next(iter(dl))
    assert batch["input_ids"].shape == (4, 16)
    return f"shape={batch['input_ids'].shape}"

test("数据加载", t_data)

# ===== 2. 训练流水线 =====
print("\n[2. 训练流水线]")

def t_train():
    model = make_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = DummyDataset(n=12)
    dl = DataLoader(ds, batch_size=4)
    losses = []
    for i, b in enumerate(dl):
        out, _ = model(b["input_ids"], labels=b["labels"])
        losses.append(out.loss.item())
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if i >= 2: break
    assert losses[-1] < losses[0] * 3
    return f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}"

test("训练步骤", t_train)

def t_grad_flow():
    model = make_model()
    model.train()
    ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    out, _ = model(ids, labels=labels)
    out.loss.backward()
    # cross_layer_q_proj 只在 cross_layer_state 不为 None 时才有梯度
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # 这些参数在 cross_layer_state=None 时不参与前向传播，没有梯度是正常的
    expected_no_grad = ['cross_layer_q_proj']
    unexpected = [n for n in no_grad if not any(e in n for e in expected_no_grad)]
    assert not unexpected, f"Unexpected no-grad params: {unexpected}"
    max_g = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    assert max_g > 0
    return f"max_grad={max_g:.4f}, {len(no_grad)} params skipped (no cross_layer_state)"

test("梯度流", t_grad_flow)

# ===== 3. 推理流水线 =====
print("\n[3. 推理流水线]")

def t_inference():
    model = make_model()
    model.eval()
    ids = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        out, _ = model(ids)
    assert out.logits.shape == (1, 8, 50)
    assert torch.isfinite(out.logits).all()
    return f"logits={out.logits.shape}"

test("前向推理", t_inference)

def t_generate():
    model = make_model()
    model.eval()
    ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=4, temperature=0.8, top_k=10, use_cache=False)
    assert gen.shape[1] == 8
    assert gen.min() >= 0 and gen.max() < 50
    return f"generated {gen.shape[1]} tokens"

test("生成推理", t_generate)

def t_incremental():
    model = make_model()
    model.eval()
    prompt = torch.randint(0, 50, (1, 4))
    cache = model.backbone.allocate_inference_cache(1, 32)
    inf = {'state_cache': cache}
    with torch.no_grad():
        model(prompt, inference_params=inf)
        for _ in range(3):
            tok = torch.tensor([[torch.randint(0, 50, (1,)).item()]])
            out, _ = model(tok, inference_params=inf)
    return f"prefill + 3 incremental"

test("增量推理", t_incremental)

# ===== 4. 保存/加载 =====
print("\n[4. 持久化]")

def t_checkpoint():
    model = make_model()
    ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    out, _ = model(ids, labels=labels)
    out.loss.backward()
    orig = {n: p.clone() for n, p in model.named_parameters()}
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    torch.save({'state': model.state_dict()}, tmp)
    model2 = make_model()
    ckpt = torch.load(tmp, weights_only=False)
    model2.load_state_dict(ckpt['state'])
    os.unlink(tmp)
    for n, p in model2.named_parameters():
        assert torch.allclose(p, orig[n]), f"Mismatch: {n}"
    return "all params match"

test("检查点保存/加载", t_checkpoint)

# ===== 5. 兼容性 =====
print("\n[5. 兼容性]")

def t_multi_config():
    from tesm_ssm.models.config_tesm import TESMConfig
    presets = [("tiny", TESMConfig.tiny), ("small", TESMConfig.small), ("small_short", TESMConfig.small_short)]
    infos = []
    for name, fn in presets:
        cfg = fn()
        cfg.vocab_size = 50
        model = TESMLMHeadModel(cfg)
        p = sum(x.numel() for x in model.parameters())
        with torch.no_grad():
            out, _ = model(torch.randint(0, 50, (1, min(8, cfg.max_seq_len))))
        infos.append(f"{name}={p/1e6:.1f}M")
    return ", ".join(infos)

test("多配置兼容", t_multi_config)

def t_batch_consistency():
    torch.manual_seed(42)
    model = make_model()
    model.eval()
    torch.manual_seed(42)
    x1 = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        o1, _ = model(x1)
    torch.manual_seed(42)
    x4 = torch.randint(0, 50, (4, 8))
    with torch.no_grad():
        o4, _ = model(x4)
    assert torch.allclose(o1.logits, o4.logits[0:1], atol=1e-5)
    return "batch=1 vs batch=4 match"

test("Batch一致性", t_batch_consistency)

def t_rmsnorm_vs_ln():
    from tesm_ssm.models.mixer_seq_simple import MixerModel
    cfg_rms = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16, vocab_size=50, kernel_backend="torch", rms_norm=True)
    cfg_ln = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16, vocab_size=50, kernel_backend="torch", rms_norm=False)
    m_rms = MixerModel(cfg_rms)
    m_ln = MixerModel(cfg_ln)
    ids = torch.randint(0, 50, (2, 4))
    with torch.no_grad():
        h1, _, _, _ = m_rms(ids)
        h2, _, _, _ = m_ln(ids)
    assert torch.isfinite(h1).all() and torch.isfinite(h2).all()
    diff = (h1 - h2).abs().mean().item()
    return f"diff={diff:.4f}"

test("RMSNorm vs LayerNorm", t_rmsnorm_vs_ln)

# ===== 6. 数值稳定性 =====
print("\n[6. 数值稳定性]")

def t_numerical_stability():
    model = make_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(10):
        ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        out, _ = model(ids, labels=labels)
        out.loss.backward()
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        assert not has_nan, f"NaN in grad at step {i}"
        assert not has_inf, f"Inf in grad at step {i}"
        opt.step()
        opt.zero_grad()
    return "10 steps, no NaN/Inf"

test("数值稳定性(10步)", t_numerical_stability)

def t_temperature():
    model = make_model()
    temps = []
    for layer in model.backbone.layers:
        temps.append(layer.mixer.get_temperature())
    model.train()
    for _ in range(5):
        ids = torch.randint(0, 50, (1, 4))
        labels = torch.randint(0, 50, (1, 4))
        out, _ = model(ids, labels=labels)
        out.loss.backward()
    final = [layer.mixer.get_temperature() for layer in model.backbone.layers]
    assert final[0] < temps[0], f"T should decrease: {temps[0]:.4f} -> {final[0]:.4f}"
    return f"T: {temps[0]:.4f} -> {final[0]:.4f}"

test("温度退火", t_temperature)

# ===== 7. 模型统计 =====
print("\n[7. 模型统计]")

def t_model_stats():
    from tesm_ssm.models.config_tesm import TESMConfig
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64, vocab_size=100)
    model = TESMLMHeadModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    n_layers = len(model.backbone.layers)
    return f"{total/1e6:.2f}M params, {n_layers} layers, ~{size_mb:.1f}MB (FP32)"

test("模型统计", t_model_stats)

# ===== 报告 =====
print()
print("=" * 60)
print("测试报告")
print("=" * 60)
passed = sum(1 for _, p, _, _ in results if p)
failed = len(results) - passed
total_t = sum(t for _, _, t, _ in results)
print(f"\n总计: {passed}/{len(results)} 通过, {failed} 失败")
print(f"总耗时: {total_t:.2f}s")
if failed:
    print("\n失败项:")
    for n, p, _, d in results:
        if not p: print(f"  - {n}: {d}")
print()
for n, p, _, d in results:
    if p and d: print(f"  {n}: {d}")
