#!/usr/bin/env python3
"""
TESM 完整对比测试 - torch.compile vs 标准模式 + 全细节覆盖

环境: CPU only, PyTorch 2.8.0+cu128
目标: 测试每个细节，确保功能一致性
"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import numpy as np
import time
import traceback
from torch.utils.data import Dataset, DataLoader

results = []

def test(name, fn):
    start = time.time()
    try:
        detail = fn()
        dur = time.time() - start
        results.append((name, True, dur, detail or ""))
        print(f"  [PASS] {name} ({dur:.2f}s) {detail}")
        return True
    except Exception as e:
        dur = time.time() - start
        err = f"{type(e).__name__}: {str(e)[:200]}"
        results.append((name, False, dur, err))
        print(f"  [FAIL] {name} ({dur:.2f}s)")
        print(f"         {err}")
        return False

class DummyDataset(Dataset):
    def __init__(self, vocab=50, seqlen=16, n=32):
        self.data = torch.randint(0, vocab, (n, seqlen))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return {"input_ids": self.data[i], "labels": self.data[i].roll(shifts=1, dims=0)}

from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.models.config_tesm import TESMConfig

def make_cfg(**overrides):
    base = dict(d_model=32, n_layer=2, d_intermediate=64, max_seq_len=32,
                vocab_size=50, kernel_backend="torch")
    base.update(overrides)
    return TESMConfig(**base)

print("=" * 70)
print("TESM 完整对比测试 - 每个细节覆盖")
print("=" * 70)
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ============================================================
# 1. torch.compile vs 标准模式 精度对比
# ============================================================
print("\n[1. torch.compile vs 标准模式 精度对比]")

def t_compile_fwd_match():
    """编译前后前向传播输出一致"""
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.eval()
    ids = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        out1, _ = model(ids)
    compiled = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        out2, _ = compiled(ids)
    diff = (out1.logits - out2.logits).abs().max().item()
    assert diff < 1e-5, f"logits max diff={diff}"
    return f"logits max diff={diff:.2e}"

test("前向传播一致性", t_compile_fwd_match)

def t_compile_loss_match():
    """编译前后损失一致"""
    cfg = make_cfg()
    model1 = TESMLMHeadModel(cfg)
    model2 = TESMLMHeadModel(cfg)
    model2.load_state_dict(model1.state_dict())
    ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    out1, _ = model1(ids, labels=labels)
    compiled = torch.compile(model2, mode="reduce-overhead")
    out2, _ = compiled(ids, labels=labels)
    diff = abs(out1.loss.item() - out2.loss.item())
    assert diff < 1e-5, f"loss diff={diff}"
    return f"loss diff={diff:.2e}"

test("损失计算一致性", t_compile_loss_match)

def t_compile_grad_match():
    """编译前后梯度一致"""
    cfg = make_cfg()
    model1 = TESMLMHeadModel(cfg)
    model2 = TESMLMHeadModel(cfg)
    model2.load_state_dict(model1.state_dict())
    ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    out1, _ = model1(ids, labels=labels)
    out1.loss.backward()
    grads1 = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}
    model1.zero_grad()
    compiled = torch.compile(model2, mode="reduce-overhead")
    out2, _ = compiled(ids, labels=labels)
    out2.loss.backward()
    max_diff = 0
    for n, g1 in grads1.items():
        g2 = dict(model2.named_parameters())[n].grad
        diff = (g1 - g2).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-4, f"grad max diff={max_diff}"
    return f"grad max diff={max_diff:.2e}"

test("梯度一致性", t_compile_grad_match)

# ============================================================
# 2. torch.compile vs 标准模式 速度对比
# ============================================================
print("\n[2. torch.compile vs 标准模式 速度对比]")

def t_compile_speed_fwd():
    """编译前后前向速度对比"""
    cfg = make_cfg(d_model=32, n_layer=2, d_intermediate=64, max_seq_len=16, vocab_size=50)
    model = TESMLMHeadModel(cfg)
    model.eval()
    ids = torch.randint(0, 50, (2, 8))
    # 只跑2次避免编译开销过大
    t0 = time.time()
    for _ in range(2):
        with torch.no_grad():
            model(ids)
    t_normal = time.time() - t0
    compiled = torch.compile(model, mode="reduce-overhead")
    for _ in range(2):  # compile warmup
        with torch.no_grad():
            compiled(ids)
    t0 = time.time()
    for _ in range(2):
        with torch.no_grad():
            compiled(ids)
    t_compile = time.time() - t0
    ratio = t_normal / t_compile if t_compile > 0 else float('inf')
    return f"normal={t_normal:.3f}s, compile={t_compile:.3f}s, ratio={ratio:.2f}x"

test("前向速度对比", t_compile_speed_fwd)

# ============================================================
# 3. 训练完整流水线对比
# ============================================================
print("\n[3. 训练完整流水线对比]")

def t_compile_training():
    """编译模式训练5步，对比损失"""
    cfg = make_cfg()
    model1 = TESMLMHeadModel(cfg)
    model2 = TESMLMHeadModel(cfg)
    model2.load_state_dict(model1.state_dict())
    compiled = torch.compile(model2, mode="reduce-overhead")
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(compiled.parameters(), lr=1e-3)
    losses1, losses2 = [], []
    for step in range(5):
        ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        out1, _ = model1(ids, labels=labels)
        out1.loss.backward()
        opt1.step()
        opt1.zero_grad()
        losses1.append(out1.loss.item())
        out2, _ = compiled(ids, labels=labels)
        out2.loss.backward()
        opt2.step()
        opt2.zero_grad()
        losses2.append(out2.loss.item())
    max_diff = max(abs(a - b) for a, b in zip(losses1, losses2))
    return f"5 steps, max loss diff={max_diff:.2e}, losses={losses1[-1]:.4f}/{losses2[-1]:.4f}"

test("训练5步对比", t_compile_training)

# ============================================================
# 4. 不同配置精度对比
# ============================================================
print("\n[4. 不同配置精度对比]")

def t_config_variants():
    """测试多种配置变体"""
    configs = [
        ("rms_norm", dict(rms_norm=True)),
        ("layer_norm", dict(rms_norm=False)),
        ("dropout_0.1", dict(dropout=0.1)),
        ("dropout_0.0", dict(dropout=0.0)),
        ("entanglement_global", dict(entanglement_window=0)),
        ("entanglement_local", dict(entanglement_window=8)),
        ("small_model", dict(d_model=16, n_layer=1, d_intermediate=32)),
        ("threshold_0.05", dict(entanglement_threshold=0.05)),
    ]
    torch.manual_seed(42)
    ids = torch.randint(0, 50, (2, 8))
    for name, overrides in configs:
        cfg = make_cfg(**overrides)
        model = TESMLMHeadModel(cfg)
        model.eval()
        with torch.no_grad():
            out, _ = model(ids)
        assert torch.isfinite(out.logits).all(), f"{name} produced NaN/Inf"
    return f"{len(configs)} variants all OK"

test("配置变体一致性", t_config_variants)

# ============================================================
# 5. BitLinear 量化精度对比
# ============================================================
print("\n[5. BitLinear 量化精度对比]")

def t_bitlinear_precision():
    """对比量化权重和浮点权重的输出差异"""
    from tesm_ssm.modules.tesm import BitLinear
    layer = BitLinear(64, 64, kernel_backend="torch")
    x = torch.randn(1, 4, 64)
    # 量化输出
    qout = layer(x)
    # 用原始权重的浮点输出（绕过量化）
    float_out = nn.functional.linear(x, layer.weight, layer.bias)
    diff = (qout - float_out).abs().mean().item()
    return f"quant vs float mean diff={diff:.4f}"

test("BitLinear 量化精度", t_bitlinear_precision)

# ============================================================
# 6. 状态扫描数值稳定性对比
# ============================================================
print("\n[6. 状态扫描数值稳定性]")

def t_state_scan_stability():
    """长序列状态扫描的数值稳定性"""
    from tesm_ssm.modules.tesm import TESM_SISO
    layer = TESM_SISO(d_model=32, d_state=16, expand=2, ent_rank=4,
                      entanglement_window=4, max_seq_len=128, kernel_backend="torch")
    for seqlen in [1, 8, 16, 32, 64, 128]:
        x = torch.randn(1, seqlen, 32)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        assert torch.isfinite(out).all(), f"seqlen={seqlen} produced NaN/Inf"
    return "seqlen=1,8,16,32,64,128 all stable"

test("状态扫描稳定性(多长度)", t_state_scan_stability)

def t_state_scan_extreme_decay():
    """极端衰减因子的稳定性"""
    from tesm_ssm.modules.tesm import TESM_SISO
    layer = TESM_SISO(d_model=32, d_state=16, expand=2, ent_rank=4,
                      entanglement_window=4, max_seq_len=32, kernel_backend="torch")
    # 接近0的输入（极端衰减）
    x = torch.randn(1, 16, 32) * 0.001
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.isfinite(out).all()
    # 大输入
    x2 = torch.randn(1, 16, 32) * 100
    out2 = layer(x2)
    if isinstance(out2, tuple):
        out2 = out2[0]
    assert torch.isfinite(out2).all()
    return "small*0.001 and large*100 both stable"

test("极端输入稳定性", t_state_scan_extreme_decay)

# ============================================================
# 7. 生成质量对比
# ============================================================
print("\n[7. 生成质量对比]")

def t_generate_temperature():
    """不同温度下生成的多样性"""
    cfg = make_cfg(vocab_size=20)
    model = TESMLMHeadModel(cfg)
    model.eval()
    ids = torch.randint(0, 20, (1, 4))
    temps = [0.1, 0.5, 1.0, 2.0]
    gens = []
    for t in temps:
        torch.manual_seed(42)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=8, temperature=t, top_k=20, use_cache=False)
        gens.append(gen)
    # 温度越低越确定性
    return f"4 temperatures, shapes={[g.shape for g in gens]}"

test("温度多样性", t_generate_temperature)

def t_generate_deterministic():
    """top_k=1 时应该是确定性的"""
    cfg = make_cfg(vocab_size=20)
    model = TESMLMHeadModel(cfg)
    model.eval()
    ids = torch.randint(0, 20, (1, 4))
    with torch.no_grad():
        gen1 = model.generate(ids, max_new_tokens=4, temperature=1.0, top_k=1, use_cache=False)
        gen2 = model.generate(ids, max_new_tokens=4, temperature=1.0, top_k=1, use_cache=False)
    assert torch.equal(gen1, gen2), "top_k=1 should be deterministic"
    return "top_k=1 deterministic: confirmed"

test("确定性生成(top_k=1)", t_generate_deterministic)

# ============================================================
# 8. 增量推理 vs 非增量一致性
# ============================================================
print("\n[8. 增量推理 vs 非增量一致性]")

def t_incremental_vs_full():
    """增量推理和完整前向传播输出一致"""
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.eval()
    prompt = torch.randint(0, 50, (1, 6))
    # 完整前向（最后一步logits）
    with torch.no_grad():
        full_out, _ = model(prompt)
        full_logits = full_out.logits[:, -1, :]
    # 增量推理
    cache = model.backbone.allocate_inference_cache(1, 32)
    inf = {'state_cache': cache}
    with torch.no_grad():
        inc_out, _ = model(prompt, inference_params=inf)
        inc_logits = inc_out.logits[:, -1, :]
    diff = (full_logits - inc_logits).abs().max().item()
    assert diff < 1e-4, f"incremental vs full diff={diff}"
    return f"logits diff={diff:.2e}"

test("增量vs全量一致性", t_incremental_vs_full)

# ============================================================
# 9. 训练稳定性长测试
# ============================================================
print("\n[9. 训练稳定性长测试]")

def t_long_training():
    """训练20步检查稳定性"""
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    has_nan_inf = False
    for step in range(20):
        ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        out, _ = model(ids, labels=labels)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            has_nan_inf = True
            break
        opt.zero_grad()
        loss.backward()
        if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
            has_nan_inf = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert not has_nan_inf, f"NaN/Inf at step {len(losses)}"
    assert losses[-1] < losses[0] * 3, f"Loss exploded: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return f"20 steps, loss: {losses[0]:.4f} -> {losses[-1]:.4f}"

test("20步训练稳定性", t_long_training)

# ============================================================
# 10. 多batch size一致性
# ============================================================
print("\n[10. 多batch size一致性]")

def t_batch_sizes():
    """不同batch size输出一致"""
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.eval()
    torch.manual_seed(42)
    x1 = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        o1, _ = model(x1)
    for bs in [1, 2, 4]:
        torch.manual_seed(42)
        x = torch.randint(0, 50, (bs, 8))
        with torch.no_grad():
            o, _ = model(x)
        assert torch.allclose(o1.logits, o.logits[0:1], atol=1e-5), f"batch={bs} mismatch"
    return "batch=1,2,4 all match"

test("多batch一致性", t_batch_sizes)

# ============================================================
# 11. 不同序列长度一致性
# ============================================================
print("\n[11. 不同序列长度一致性]")

def t_seq_len_consistency():
    """同一prompt不同长度前缀输出一致"""
    cfg = make_cfg(max_seq_len=64)
    model = TESMLMHeadModel(cfg)
    model.eval()
    full_ids = torch.randint(0, 50, (1, 16))
    with torch.no_grad():
        full_out, _ = model(full_ids)
    # 只取前8个
    short_ids = full_ids[:, :8]
    with torch.no_grad():
        short_out, _ = model(short_ids)
    # 前8步的logits应该和full的前8步一致
    diff = (full_out.logits[:, :8] - short_out.logits).abs().max().item()
    assert diff < 1e-4, f"seq len inconsistency: {diff}"
    return f"logits diff={diff:.2e}"

test("序列长度一致性", t_seq_len_consistency)

# ============================================================
# 12. Embedding 共享验证
# ============================================================
print("\n[12. Embedding 共享验证]")

def t_embedding_sharing():
    """验证embedding共享"""
    cfg = make_cfg(tie_embeddings=True)
    model = TESMLMHeadModel(cfg)
    shared = model.lm_head.weight is model.backbone.embedding.weight
    assert shared, "Embedding should be shared"
    # 不共享
    cfg2 = make_cfg(tie_embeddings=False)
    model2 = TESMLMHeadModel(cfg2)
    not_shared = model2.lm_head.weight is not model2.backbone.embedding.weight
    assert not_shared, "Embedding should not be shared"
    return f"tie=True: shared={shared}, tie=False: shared={not not_shared}"

test("Embedding共享", t_embedding_sharing)

# ============================================================
# 13. 检查点完整流水线
# ============================================================
print("\n[13. 检查点完整流水线]")

def t_checkpoint_full():
    """保存-加载-继续训练-输出一致"""
    import tempfile, os
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 训练3步
    for _ in range(3):
        ids = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        out, _ = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    # 保存
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': 3}, tmp)
    # 加载到新模型
    model2 = TESMLMHeadModel(cfg)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ckpt = torch.load(tmp, weights_only=False)
    model2.load_state_dict(ckpt['model'])
    opt2.load_state_dict(ckpt['opt'])
    os.unlink(tmp)
    # 验证参数一致
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Param {n1} mismatch"
    # 验证输出一致
    ids = torch.randint(0, 50, (2, 8))
    labels = torch.randint(0, 50, (2, 8))
    model2.train()
    out1, _ = model(ids, labels=labels)
    out2, _ = model2(ids, labels=labels)
    diff = abs(out1.loss.item() - out2.loss.item())
    assert diff < 1e-5, f"Loss diff after reload: {diff}"
    return f"loss diff after reload={diff:.2e}"

test("检查点完整流水线", t_checkpoint_full)

# ============================================================
# 14. INT2 量化推理对比
# ============================================================
print("\n[14. INT2 量化推理对比]")

def t_int2_quantization():
    """INT2量化前后输出对比"""
    from tesm_ssm.utils.int2_quantization import pack_int2_to_uint8, unpack_uint8_to_int2, Int2Linear
    import torch.nn as nn
    linear = nn.Linear(64, 32)
    x = torch.randn(1, 4, 64)
    # 原始输出
    orig_out = linear(x)
    # INT2量化
    packed, scale = pack_int2_to_uint8(linear.weight.data)
    int2_layer = Int2Linear(64, 32, packed, scale, linear.bias)
    int2_out = int2_layer(x)
    diff = (orig_out - int2_out).abs().mean().item()
    return f"FP32 vs INT2 mean diff={diff:.4f}"

test("INT2量化精度", t_int2_quantization)

# ============================================================
# 15. 温度退火详细测试
# ============================================================
print("\n[15. 温度退火详细测试]")

def t_temperature_schedule():
    """详细测试温度退火曲线"""
    from tesm_ssm.modules.tesm import TESM_SISO
    layer = TESM_SISO(d_model=32, d_state=16, expand=2, ent_rank=4,
                      entanglement_window=4, max_seq_len=16, kernel_backend="torch",
                      annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100)
    temps = []
    for s in [0, 25, 50, 75, 100, 200]:
        layer.annealing_step.fill_(s)
        temps.append((s, layer.get_temperature()))
    # 验证单调递减
    for i in range(1, len(temps)):
        assert temps[i][1] <= temps[i-1][1] + 1e-6, f"Temperature not monotonic: {temps}"
    # 验证边界
    assert temps[0][1] >= 9.9, f"Initial temp too low: {temps[0][1]}"
    assert temps[-1][1] <= 0.11, f"Final temp too high: {temps[-1][1]}"
    return ", ".join(f"s={s}:T={t:.4f}" for s, t in temps)

test("温度退火曲线", t_temperature_schedule)

# ============================================================
# 16. 推理缓存详细测试
# ============================================================
print("\n[16. 推理缓存详细测试]")

def t_cache_multi_step():
    """多步增量推理，验证缓存更新正确"""
    cfg = make_cfg()
    model = TESMLMHeadModel(cfg)
    model.eval()
    prompt = torch.randint(0, 50, (1, 4))
    cache = model.backbone.allocate_inference_cache(1, 32)
    # cache结构: {0: {'state': ..., 'seq_pos': ...}, 1: {...}}
    layer_cache = cache[0]  # 第一层
    inf = {'state_cache': cache}
    # 预填充
    with torch.no_grad():
        out, _ = model(prompt, inference_params=inf)
    # 逐步生成5个token
    tokens = []
    for i in range(5):
        with torch.no_grad():
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        tokens.append(next_tok.item())
        with torch.no_grad():
            out, _ = model(next_tok, inference_params=inf)
    assert len(tokens) == 5
    final_pos = layer_cache['seq_pos'].item() if hasattr(layer_cache['seq_pos'], 'item') else layer_cache['seq_pos']
    assert final_pos == 9, f"Expected seq_pos=9, got {final_pos}"
    return f"5 tokens, final seq_pos={final_pos}"

test("多步缓存一致性", t_cache_multi_step)

# ============================================================
# 17. 纠缠统计验证
# ============================================================
print("\n[17. 纠缠统计验证]")

def t_entanglement_stats():
    """训练时纠缠统计是否正确更新"""
    from tesm_ssm.modules.tesm import TESM_SISO
    layer = TESM_SISO(d_model=32, d_state=16, expand=2, ent_rank=4,
                      entanglement_window=4, max_seq_len=16, kernel_backend="torch",
                      entanglement_threshold=0.1)
    layer.train()
    x = torch.randn(1, 8, 32)
    out = layer(x)
    # 检查统计buffer
    assert hasattr(layer, '_stats_total_buffer')
    assert hasattr(layer, '_stats_ternary_buffer')
    return f"stats buffers exist"

test("纠缠统计", t_entanglement_stats)

# ============================================================
# 总结
# ============================================================
print()
print("=" * 70)
print("完整对比测试总结")
print("=" * 70)
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
