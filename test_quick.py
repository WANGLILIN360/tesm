#!/usr/bin/env python3
"""TESM 快速测试套件"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn

# 追踪结果
results = []

def test(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  [FAIL] {name}: {e}")

# ===========================
# 测试配置
# ===========================
print("\n[配置模块]")
from tesm_ssm.models.config_tesm import TESMConfig

def test_config_basic():
    c = TESMConfig(d_model=256, n_layer=4)
    assert c.d_model == 256 and c.n_layer == 4

def test_config_presets():
    for p in ['tiny', 'small', 'base']:
        c = getattr(TESMConfig, p)()
        assert c.d_model > 0 and c.n_layer > 0

def test_config_serde():
    c1 = TESMConfig.small()
    d = c1.to_dict()
    c2 = TESMConfig.from_dict(d)
    assert c1.d_model == c2.d_model

test("基本配置", test_config_basic)
test("预设配置", test_config_presets)
test("序列化", test_config_serde)

# ===========================
# 测试 BitLinear
# ===========================
print("\n[BitLinear]")
from tesm_ssm.modules.tesm import BitLinear

def test_bitlinear_fwd():
    layer = BitLinear(64, 128, kernel_backend="torch")
    x = torch.randn(2, 16, 64)
    out = layer(x)
    assert out.shape == (2, 16, 128)

def test_bitlinear_quant():
    layer = BitLinear(64, 128, kernel_backend="torch")
    qw = layer.quantized_weight()
    assert qw.abs().max() <= 1.0 + 1e-5

def test_bitlinear_grad():
    layer = BitLinear(64, 128, kernel_backend="torch")
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None and layer.weight.grad is not None

def test_bitlinear_eval_consistency():
    layer = BitLinear(64, 128, kernel_backend="torch")
    layer.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        o1 = layer(x)
        o2 = layer(x)
    assert torch.allclose(o1, o2, atol=1e-6)

test("前向传播", test_bitlinear_fwd)
test("权重量化", test_bitlinear_quant)
test("梯度流", test_bitlinear_grad)
test("评估一致性", test_bitlinear_eval_consistency)

# ===========================
# 测试 TernaryQuantumTunneling
# ===========================
print("\n[TernaryQuantumTunneling]")
from tesm_ssm.modules.tesm import TernaryQuantumTunneling

def test_tunnel_fwd():
    t = TernaryQuantumTunneling(threshold=0.1)
    scores = torch.randn(2, 8, 16)
    tv, info = t.apply_tunneling(scores, training=True)
    assert tv.shape == scores.shape
    for v in torch.unique(tv):
        assert v.item() in [-1.0, 0.0, 1.0]

def test_tunnel_barrier():
    t = TernaryQuantumTunneling(threshold=0.1)
    b = t.compute_barrier_height(torch.tensor([0.0, 0.05, 0.1]))
    assert torch.allclose(b, torch.tensor([0.1, 0.05, 0.0]), atol=1e-6)

def test_tunnel_prob():
    t = TernaryQuantumTunneling(threshold=0.1, tunneling_strength=0.1)
    p = t.get_tunneling_probability(torch.tensor([0.0, 0.05, 0.1]))
    assert (p >= t.min_tunnel_prob).all() and (p <= t.max_tunnel_prob).all()

test("前向传播", test_tunnel_fwd)
test("势垒高度", test_tunnel_barrier)
test("概率范围", test_tunnel_prob)

# ===========================
# 测试 TESM_SISO
# ===========================
print("\n[TESM_SISO]")
from tesm_ssm.modules.tesm import TESM_SISO

def test_siso_fwd():
    layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                      entanglement_window=8, max_seq_len=64, kernel_backend="torch")
    x = torch.randn(2, 16, 128)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 16, 128)

def test_siso_global():
    layer = TESM_SISO(d_model=128, d_state=64, expand=2, ent_rank=16,
                      entanglement_window=0, max_seq_len=64, kernel_backend="torch")
    x = torch.randn(2, 16, 128)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 16, 128)

def test_siso_grad():
    layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                      entanglement_window=4, max_seq_len=32, kernel_backend="torch")
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    out.sum().backward()
    assert x.grad is not None

def test_siso_cache():
    layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                      entanglement_window=4, max_seq_len=32, kernel_backend="torch")
    cache = layer.allocate_inference_cache(batch_size=2, max_seqlen=32)
    assert 'state' in cache and 'seq_pos' in cache

def test_siso_temp_schedule():
    layer = TESM_SISO(d_model=32, d_state=16, expand=2, ent_rank=4,
                      entanglement_window=4, max_seq_len=32, kernel_backend="torch",
                      annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100)
    T1 = layer.get_temperature()
    layer.annealing_step.fill_(50)
    T2 = layer.get_temperature()
    layer.annealing_step.fill_(1000)
    T3 = layer.get_temperature()
    assert T1 >= T2 >= T3

def test_siso_seq_len_1():
    layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                      entanglement_window=4, max_seq_len=32, kernel_backend="torch")
    x = torch.randn(1, 1, 64)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (1, 1, 64)

test("前向传播(local)", test_siso_fwd)
test("前向传播(global)", test_siso_global)
test("梯度流", test_siso_grad)
test("推理缓存", test_siso_cache)
test("温度调度", test_siso_temp_schedule)
test("序列长度1", test_siso_seq_len_1)

# ===========================
# 测试 TESMMIMO
# ===========================
print("\n[TESMMIMO_Optimized]")
from tesm_ssm.modules.tesm_mimo import TESMMIMO_Optimized

def test_mimo_fwd():
    layer = TESMMIMO_Optimized(d_model=128, d_state=64, n_heads=4, expand=2,
                               ent_rank=16, entanglement_window=8, max_seq_len=64,
                               kernel_backend="torch")
    x = torch.randn(2, 16, 128)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 16, 128)

def test_mimo_grad():
    layer = TESMMIMO_Optimized(d_model=64, d_state=32, n_heads=4, expand=2,
                               ent_rank=8, entanglement_window=4, max_seq_len=32,
                               kernel_backend="torch")
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    out.sum().backward()
    assert x.grad is not None

def test_mimo_cache():
    layer = TESMMIMO_Optimized(d_model=64, d_state=32, n_heads=4, expand=2,
                               ent_rank=8, entanglement_window=4, max_seq_len=32,
                               kernel_backend="torch")
    cache = layer.allocate_inference_cache(batch_size=2, max_seqlen=32)
    assert cache['state'].shape == (2, 4, 32)  # (batch, n_heads, d_state)

test("前向传播", test_mimo_fwd)
test("梯度流", test_mimo_grad)
test("推理缓存", test_mimo_cache)

# ===========================
# 测试 MixerModel 和 LMHeadModel
# ===========================
print("\n[MixerModel / TESMLMHeadModel]")
from tesm_ssm.models.mixer_seq_simple import MixerModel, TESMLMHeadModel

def test_mixermodel_fwd():
    config = TESMConfig(d_model=128, n_layer=2, d_intermediate=256, max_seq_len=64,
                       vocab_size=1000, kernel_backend="torch")
    model = MixerModel(config)
    ids = torch.randint(0, 1000, (2, 16))
    h, _, _, _ = model(ids)
    assert h.shape == (2, 16, 128)

def test_mixermodel_len_check():
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=16,
                       vocab_size=100, kernel_backend="torch")
    model = MixerModel(config)
    try:
        model(torch.randint(0, 100, (1, 32)))
        raise AssertionError("应该抛出ValueError")
    except ValueError:
        pass

def test_lmhead_fwd():
    config = TESMConfig(d_model=128, n_layer=2, d_intermediate=256, max_seq_len=64,
                       vocab_size=1000, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    ids = torch.randint(0, 1000, (2, 16))
    out, _ = model(ids)
    assert out.logits.shape == (2, 16, 1000)

def test_lmhead_loss():
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    out, _ = model(ids, labels=labels)
    assert out.loss is not None and out.loss.item() > 0

def test_lmhead_generate():
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.eval()
    ids = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=4, temperature=0.5, top_k=10)
    assert gen.shape[1] >= 8

def test_lmhead_training_step():
    config = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=32,
                       vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    out, _ = model(ids, labels=labels)
    loss = out.loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert loss.item() > 0

test("MixerModel前向", test_mixermodel_fwd)
test("序列长度检查", test_mixermodel_len_check)
test("LMHead前向", test_lmhead_fwd)
test("损失计算", test_lmhead_loss)
test("生成方法", test_lmhead_generate)
test("训练步骤", test_lmhead_training_step)

# ===========================
# 测试 Block
# ===========================
print("\n[Block / RMSNorm]")
from tesm_ssm.modules.block import RMSNorm, Block

def test_rmsnorm():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == (2, 16, 64)

def test_rmsnorm_grad():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = norm(x)
    out.sum().backward()
    assert x.grad is not None and norm.weight.grad is not None

def test_block_fwd():
    def m_c(dim):
        return TESM_SISO(d_model=dim, d_state=16, expand=2, ent_rank=4,
                         entanglement_window=4, max_seq_len=16, kernel_backend="torch")
    def f_c(dim):
        return nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    block = Block(32, m_c, f_c)
    x = torch.randn(2, 8, 32)
    out, _, _ = block(x)
    assert out.shape == (2, 8, 32)

test("RMSNorm前向", test_rmsnorm)
test("RMSNorm梯度", test_rmsnorm_grad)
test("Block前向", test_block_fwd)

# ===========================
# 测试 TrainingConfig
# ===========================
print("\n[TrainingConfig]")
from tesm_ssm.training.config import TrainingConfig

def test_train_config():
    tc = TrainingConfig(model_config=TESMConfig.small(), num_epochs=3, batch_size=2)
    assert tc.num_epochs == 3 and tc.batch_size == 2

def test_train_config_serde():
    tc = TrainingConfig(model_config=TESMConfig.small(), num_epochs=3)
    d = tc.to_dict()
    assert isinstance(d, dict) and 'num_epochs' in d

test("基本配置", test_train_config)
test("序列化", test_train_config_serde)

# ===========================
# 测试 INT2 量化
# ===========================
print("\n[Int2 Quantization]")
from tesm_ssm.utils.int2_quantization import pack_int2_to_uint8, unpack_uint8_to_int2, Int2Linear

def test_int2_pack():
    w = torch.randn(32, 64)
    packed, scale = pack_int2_to_uint8(w)
    assert packed.dtype == torch.uint8
    unpacked = unpack_uint8_to_int2(packed, scale.item())
    assert unpacked.shape[0] == 32

def test_int2_odd_dim():
    w = torch.randn(32, 65)
    packed, scale = pack_int2_to_uint8(w)
    unpacked = unpack_uint8_to_int2(packed, scale.item())
    assert unpacked.shape[0] == 32

def test_int2_linear():
    w = torch.randn(32, 64)
    packed, scale = pack_int2_to_uint8(w)
    layer = Int2Linear(64, 32, packed, scale)
    x = torch.randn(2, 16, 64)
    out = layer(x)
    assert out.shape == (2, 16, 32)

def test_int2_from_float():
    linear = nn.Linear(64, 32)
    i2 = Int2Linear.from_float(linear)
    x = torch.randn(2, 16, 64)
    out = i2(x)
    assert out.shape == (2, 16, 32)

test("打包/解包", test_int2_pack)
test("不可整除4维度", test_int2_odd_dim)
test("Int2Linear", test_int2_linear)
test("from_float", test_int2_from_float)

# ===========================
# 测试 分页缓存
# ===========================
print("\n[PagedStateCache]")
from tesm_ssm.utils.paged_cache import PagedStateCache

def test_paged_create():
    c = PagedStateCache(batch_size=2, d_state=64, ent_rank=16, window=8,
                        page_size=16, max_gpu_pages=4, device=torch.device('cpu'))
    assert c.batch_size == 2

def test_paged_save_load():
    c = PagedStateCache(batch_size=2, d_state=64, ent_rank=16, window=8,
                        page_size=16, max_gpu_pages=4, device=torch.device('cpu'))
    s = {'state': torch.randn(2, 64), 'ent_k_cache': torch.randn(2, 8, 16),
         'ent_v_cache': torch.randn(2, 8, 64)}
    c.save_state(0, s)
    loaded = c.load_state(0)
    assert loaded is not None and loaded['state'].shape == (2, 64)

def test_paged_stats():
    c = PagedStateCache(batch_size=2, d_state=64, ent_rank=16, window=8,
                        page_size=16, max_gpu_pages=4, device=torch.device('cpu'))
    s = {'state': torch.randn(2, 64), 'ent_k_cache': torch.randn(2, 8, 16),
         'ent_v_cache': torch.randn(2, 8, 64)}
    c.save_state(0, s)
    stats = c.get_memory_stats()
    assert stats['total_pages'] > 0

def test_paged_clear():
    c = PagedStateCache(batch_size=2, d_state=64, ent_rank=16, window=8,
                        page_size=16, max_gpu_pages=4, device=torch.device('cpu'))
    s = {'state': torch.randn(2, 64), 'ent_k_cache': torch.randn(2, 8, 16),
         'ent_v_cache': torch.randn(2, 8, 64)}
    c.save_state(0, s)
    c.clear()
    assert len(c.pages) == 0

test("基本创建", test_paged_create)
test("保存/加载", test_paged_save_load)
test("内存统计", test_paged_stats)
test("清空缓存", test_paged_clear)

# ===========================
# 测试 模型保存/加载
# ===========================
print("\n[模型保存/加载]")
import tempfile, os

def test_model_save_load():
    config = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16,
                       vocab_size=50, kernel_backend="torch")
    model = TESMLMHeadModel(config)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    torch.save(model.state_dict(), tmp)
    state = torch.load(tmp, weights_only=False)
    model2 = TESMLMHeadModel(config)
    model2.load_state_dict(state)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2)
    os.unlink(tmp)

test("保存/加载一致性", test_model_save_load)

# ===========================
# 总结
# ===========================
print("\n" + "="*60)
print("测试摘要")
print("="*60)
passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
print(f"总计: {passed}/{len(results)} 通过, {failed} 失败")

if failed > 0:
    print("\n失败项:")
    for name, p, e in results:
        if not p:
            print(f"  - {name}: {e}")

sys.exit(0 if failed == 0 else 1)
