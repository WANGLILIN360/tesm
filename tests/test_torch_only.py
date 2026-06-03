"""TESM 全功能测试 - 仅依赖 PyTorch (kernel_backend=torch, CPU)

覆盖: 配置、量化、核心模块(SISO/MIMO)、模型、推理、工具函数
用法: pytest tests/test_torch_only.py -v
"""

import math
import pytest
import torch
import torch.nn as nn
from tesm_ssm import TESMConfig, TESMLMHeadModel
from tesm_ssm.modules.tesm import BitLinear, TESM_SISO, TernaryQuantumTunneling
from tesm_ssm.modules.tesm_mimo import TESMMIMO_Optimized
from tesm_ssm.modules.block import Block, RMSNorm
from tesm_ssm.utils.int2_quantization import (
    pack_int2_to_uint8, unpack_uint8_to_int2, quantize_weight_to_int2,
    Int2Linear, export_model_to_int2
)
from tesm_ssm.utils.paged_cache import PagedStateCache


# === 1. TESMConfig ===
class TestTESMConfig:
    def test_default(self):
        c = TESMConfig()
        assert c.d_model == 768 and c.n_layer == 24 and c.decay_init_bias == 3.0

    def test_presets(self):
        for name, dm, nl in [("tiny",768,12),("small",512,16),("base",768,24),("medium",1024,32)]:
            c = getattr(TESMConfig, name)()
            assert c.d_model == dm and c.n_layer == nl

    def test_long_context(self):
        c = TESMConfig.long_context()
        assert c.max_seq_len == 16384 and c.decay_init_bias == 6.0

    def test_large_presets(self):
        for n in ["large_40b","large_70b","large_100b","large_200b","large_400b"]:
            c = getattr(TESMConfig, n)()
            assert c.d_model >= 4096

    def test_exp_configs(self):
        c = TESMConfig.exp_decay_comparison(2.0)
        assert c.decay_init_bias == 2.0
        c = TESMConfig.exp_threshold_comparison(0.05)
        assert c.entanglement_threshold == 0.05
        c = TESMConfig.exp_scale_comparison(0.3)
        assert c.entanglement_scale == 0.3

    def test_roundtrip(self):
        c = TESMConfig.small()
        d = c.to_dict()
        r = TESMConfig.from_dict(d)
        assert r.d_model == c.d_model and r.decay_init_bias == c.decay_init_bias

    def test_mimo_config(self):
        c = TESMConfig(use_mimo=True, n_heads=4)
        assert c.use_mimo and c.n_heads == 4

    def test_tunneling_config(self):
        c = TESMConfig(quantum_tunneling_enabled=True, tunneling_strength=0.2)
        assert c.quantum_tunneling_enabled and c.tunneling_strength == 0.2


# === 2. BitLinear ===
class TestBitLinear:
    def test_creation(self):
        l = BitLinear(64, 128, kernel_backend="torch")
        assert l.in_features == 64 and l.out_features == 128 and l.bias is None

    def test_forward_eval(self):
        l = BitLinear(64, 128, kernel_backend="torch"); l.eval()
        with torch.no_grad():
            assert l(torch.randn(2,16,64)).shape == (2,16,128)

    def test_forward_train_grad(self):
        l = BitLinear(64, 128, kernel_backend="torch"); l.train()
        l(torch.randn(2,16,64)).sum().backward()
        assert l.weight.grad is not None

    def test_quantized_weight(self):
        l = BitLinear(64, 128, kernel_backend="torch")
        qw = l.quantized_weight()
        assert qw.shape == (128,64) and torch.isfinite(qw).all()

    def test_quantized_input(self):
        l = BitLinear(64, 128, kernel_backend="torch")
        qi = l.quantized_input(torch.randn(2,16,64))
        assert qi.shape == (2,16,64)

    def test_cached_weight_eval(self):
        l = BitLinear(64, 128, kernel_backend="torch"); l.eval()
        with torch.no_grad():
            assert l._get_eval_quantized_weight() is l._get_eval_quantized_weight()

    def test_2d_input(self):
        l = BitLinear(64, 128, kernel_backend="torch"); l.eval()
        with torch.no_grad():
            assert l(torch.randn(10,64)).shape == (10,128)

    def test_unavailable_backend_raises(self):
        l = BitLinear(64, 128, kernel_backend="cuda")
        with pytest.raises(RuntimeError, match="kernel_backend"):
            l(torch.randn(2,16,64))


# === 3. RMSNorm ===
class TestRMSNorm:
    def test_forward(self):
        n = RMSNorm(64)
        assert n(torch.randn(2,16,64)).shape == (2,16,64)

    def test_grad(self):
        n = RMSNorm(64)
        x = torch.randn(2,16,64, requires_grad=True)
        n(x).sum().backward()
        assert x.grad is not None and n.weight.grad is not None


# === 4. TernaryQuantumTunneling ===
class TestTunneling:
    def test_barrier_height(self):
        t = TernaryQuantumTunneling(threshold=0.1)
        b = t.compute_barrier_height(torch.tensor([0.05, 0.08, 0.12]))
        assert b[0] > 0 and b[1] > 0 and b[2] == 0

    def test_tunnel_prob_monotonic(self):
        t = TernaryQuantumTunneling(threshold=0.1)
        p = t.get_tunneling_probability(torch.tensor([0.0, 0.05, 0.1]))
        assert p[0] >= p[1] >= p[2]

    def test_apply_training(self):
        t = TernaryQuantumTunneling(threshold=0.1)
        v, info = t.apply_tunneling(torch.randn(2,16,8)*0.2, training=True)
        assert v.shape == (2,16,8) and "tunnel_rate" in info

    def test_learnable_scale(self):
        assert TernaryQuantumTunneling(threshold=0.1).tunnel_scale.requires_grad


# === 5. TESM_SISO ===
class TestSISO:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, expand=2, ent_rank=8,
                 entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESM_SISO(**d)

    def test_forward_eval(self):
        m = self._m(); m.eval()
        with torch.no_grad():
            y, fs = m(torch.randn(2,16,64))
        assert y.shape == (2,16,64) and fs is not None

    def test_forward_train(self):
        m = self._m(); m.train()
        m(torch.randn(2,16,64))[0].sum().backward()

    def test_local_ent(self):
        m = self._m(entanglement_window=8); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,32,64))[0].shape == (2,32,64)

    def test_global_ent(self):
        m = self._m(entanglement_window=0); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,8,64))[0].shape == (2,8,64)

    def test_rope(self):
        m = self._m()
        x = torch.randn(2,16,8)
        assert m._apply_rope(x).shape == x.shape
        assert not torch.allclose(m._apply_rope(x), m._apply_rope(x, pos_offset=10), atol=1e-6)

    def test_annealing_cosine(self):
        m = self._m(annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100, annealing_schedule="cosine")
        m.train()
        assert m.get_temperature() > 1.0
        m.annealing_step.fill_(100)
        assert abs(m.get_temperature() - 0.1) < 0.2

    def test_annealing_linear(self):
        m = self._m(annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100, annealing_schedule="linear")
        m.train(); m.annealing_step.fill_(50)
        assert 0.1 < m.get_temperature() < 10.0

    def test_annealing_eval_fixed(self):
        m = self._m(annealing_enabled=True, T_start=10.0, T_end=0.1); m.eval()
        assert m.get_temperature() == 0.1

    def test_state_scan(self):
        m = self._m()
        s = m._parallel_state_scan(torch.sigmoid(torch.randn(2,16,16)), torch.randn(2,16,16))
        assert s.shape == (2,16,16)

    def test_state_scan_prev(self):
        m = self._m()
        s = m._parallel_state_scan(torch.sigmoid(torch.randn(2,8,16)), torch.randn(2,8,16),
                                    prev_state=torch.randn(2,16,dtype=torch.float64))
        assert s.shape == (2,8,16) and not torch.allclose(s[:,0,:], torch.zeros_like(s[:,0,:]))

    def test_cross_layer(self):
        m = self._m(); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,16,64), cross_layer_state=torch.randn(2,16,16))[0].shape == (2,16,64)

    def test_inference_cache(self):
        m = self._m()
        c = m.allocate_inference_cache(2,64)
        assert "state" in c and c["state"].dtype == torch.float64

    def test_paged_cache(self):
        m = self._m()
        c = m.allocate_inference_cache(2,4096, use_paged_cache=True, page_size=512)
        assert c["use_paged"] and "paged_cache" in c

    def test_incremental(self):
        m = self._m(); m.eval()
        c = m.allocate_inference_cache(1,64)
        with torch.no_grad():
            m(torch.randn(1,8,64), inference_params={"state_cache": c})
            y, _ = m(torch.randn(1,1,64), inference_params={"state_cache": c})
        assert y.shape == (1,1,64)

    def test_tunneling_forward(self):
        m = self._m(quantum_tunneling_enabled=True); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,16,64))[0].shape == (2,16,64)


# === 6. TESMMIMO ===
class TestMIMO:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, n_heads=2, mimo_rank=2, expand=2,
                 ent_rank=8, entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESMMIMO_Optimized(**d)

    def test_forward_eval(self):
        m = self._m(); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,16,64))[0].shape == (2,16,64)

    def test_forward_train(self):
        m = self._m(); m.train()
        m(torch.randn(2,16,64))[0].sum().backward()

    def test_global_ent(self):
        m = self._m(entanglement_window=0); m.eval()
        with torch.no_grad(): assert m(torch.randn(2,8,64))[0].shape == (2,8,64)

    def test_rope_4d(self):
        m = self._m()
        x = torch.randn(2,16,2,8)
        assert m._apply_rope(x).shape == x.shape

    def test_scan_pytorch_mimo(self):
        m = self._m()
        s = m._parallel_state_scan_pytorch_mimo(torch.sigmoid(torch.randn(2,16,2,16)), torch.randn(2,16,2,16))
        assert s.shape == (2,16,2,16)

    def test_scan_stable(self):
        m = self._m()
        s = m._parallel_state_scan_mimo_stable(torch.sigmoid(torch.randn(2,16,2,16)), torch.randn(2,16,2,16))
        assert s.shape == (2,16,2,16) and torch.isfinite(s).all()

    def test_mimo_params(self):
        m = self._m()
        assert m.mimo_x.shape == (2,2,16) and m.mimo_z is not None and m.mimo_o is not None

    def test_more_params_than_siso(self):
        s = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64, kernel_backend="torch")
        sp = sum(p.numel() for p in s.parameters())
        mp = sum(p.numel() for p in self._m().parameters())
        assert mp > sp


# === 7. Block ===
class TestBlock:
    def test_forward(self):
        def mk_mixer(dim):
            return TESM_SISO(d_model=dim, d_state=8, ent_rank=4, entanglement_window=4,
                             max_seq_len=32, kernel_backend="torch", decay_init_bias=0.0)
        b = Block(64, mk_mixer, lambda d: nn.Sequential(BitLinear(d,d*2,kernel_backend="torch"),nn.ReLU(),BitLinear(d*2,d,kernel_backend="torch")), norm_cls=nn.LayerNorm)
        b.eval()
        with torch.no_grad():
            out, res, fs = b(torch.randn(2,16,64))
        assert out.shape == (2,16,64)

    def test_allocate_cache(self):
        def mk_mixer(dim):
            return TESM_SISO(d_model=dim, d_state=8, ent_rank=4, entanglement_window=4,
                             max_seq_len=32, kernel_backend="torch", decay_init_bias=0.0)
        b = Block(64, mk_mixer, lambda d: nn.Identity(), norm_cls=nn.LayerNorm)
        c = b.allocate_inference_cache(2, 32)
        assert c is not None


# === 8. TESMLMHeadModel ===
class TestLMModel:
    def _tiny(self, **kw):
        d = dict(d_model=64, n_layer=2, d_intermediate=128, vocab_size=1000,
                 max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                 kernel_backend="torch", decay_init_bias=0.0, tie_embeddings=False)
        d.update(kw); return TESMLMHeadModel(TESMConfig(**d))

    def test_creation(self):
        assert self._tiny() is not None

    def test_forward_eval(self):
        m = self._tiny(); m.eval()
        with torch.no_grad():
            out, _ = m(torch.randint(0,1000,(2,16)))
        assert out.logits.shape == (2,16,1000)

    def test_forward_with_labels(self):
        m = self._tiny(); m.eval()
        ids = torch.randint(0,1000,(2,16))
        with torch.no_grad():
            out, _ = m(ids, labels=ids)
        assert out.loss is not None

    def test_gradient_flow(self):
        m = self._tiny(); m.train()
        ids = torch.randint(0,1000,(2,16))
        out, _ = m(ids, labels=ids)
        out.loss.backward()
        has_grad = any(p.grad is not None for p in m.parameters() if p.requires_grad)
        assert has_grad

    def test_mimo_model(self):
        m = self._tiny(use_mimo=True, n_heads=2); m.eval()
        with torch.no_grad():
            out, _ = m(torch.randint(0,1000,(2,16)))
        assert out.logits.shape == (2,16,1000)

    def test_generate(self):
        m = self._tiny(); m.eval()
        ids = torch.randint(0,1000,(1,4))
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=4, use_cache=False)
        assert gen.shape[0] == 1 and gen.shape[1] >= 4

    def test_generate_with_cache(self):
        m = self._tiny(); m.eval()
        ids = torch.randint(0,1000,(1,4))
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=4, use_cache=True)
        assert gen.shape[0] == 1

    def test_vocab_suppression_generate(self):
        m = self._tiny(vocab_suppression=True, suppression_bias=-10.0); m.eval()
        ids = torch.randint(0,1000,(1,4))
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=4, use_cache=False)
        assert gen.shape[0] == 1

    def test_different_seq_lens(self):
        m = self._tiny(); m.eval()
        for sl in [8, 16, 32]:
            with torch.no_grad():
                out, _ = m(torch.randint(0,1000,(1,sl)))
            assert out.logits.shape[1] == sl

    def test_tie_embeddings(self):
        m = self._tiny(tie_embeddings=True)
        assert m.lm_head.weight is m.backbone.embedding.weight


# === 9. INT2 量化工具 ===
class TestInt2Quantization:
    def test_pack_unpack_roundtrip(self):
        w = torch.randn(32, 64)
        packed, scale = pack_int2_to_uint8(w)
        unpacked = unpack_uint8_to_int2(packed, scale.item())
        # 三值量化有损，但形状应一致
        assert unpacked.shape[0] == 32

    def test_int2_linear_forward(self):
        w = torch.randn(32, 64)
        packed, scale = pack_int2_to_uint8(w)
        l = Int2Linear(64, 32, packed, scale)
        l.eval()
        with torch.no_grad():
            out = l(torch.randn(2, 64))
        assert out.shape == (2, 32)

    def test_int2_linear_from_float(self):
        fl = nn.Linear(64, 32)
        il = Int2Linear.from_float(fl)
        assert il.in_features == 64 and il.out_features == 32

    def test_export_model(self):
        m = TESMLMHeadModel(TESMConfig(d_model=64, n_layer=1, d_intermediate=128,
                                        vocab_size=100, max_seq_len=32, d_state=8,
                                        ent_rank=4, entanglement_window=4,
                                        kernel_backend="torch", decay_init_bias=0.0,
                                        tie_embeddings=False))
        exported = export_model_to_int2(m)
        assert len(exported) > 0

    def test_pack_shape(self):
        w = torch.randn(16, 64)
        packed, _ = pack_int2_to_uint8(w)
        # 4个int2打包成1个uint8，所以packed的最后一维 = 64//4 = 16
        assert packed.shape == (16, 16)
        assert packed.dtype == torch.uint8


# === 10. PagedStateCache ===
class TestPagedCache:
    def _cpu_cache(self, **kw):
        d = dict(batch_size=1, d_state=16, ent_rank=8, window=4, page_size=512,
                 device=torch.device('cpu'))
        d.update(kw); return PagedStateCache(**d)

    def test_creation(self):
        assert self._cpu_cache().page_size == 512

    def test_save_load(self):
        c = self._cpu_cache()
        state = torch.zeros(1, 16, dtype=torch.float64)
        ek = torch.zeros(1, 4, 8)
        ev = torch.zeros(1, 4, 16)
        c.save_state(0, {"state": state, "ent_k_cache": ek, "ent_v_cache": ev})
        loaded = c.load_state(0)
        assert loaded is not None and "state" in loaded

    def test_memory_stats(self):
        stats = self._cpu_cache().get_memory_stats()
        assert "gpu_pages" in stats and "total_pages" in stats

    def test_clear(self):
        c = self._cpu_cache()
        c.save_state(0, {"state": torch.zeros(1,16,dtype=torch.float64),
                         "ent_k_cache": torch.zeros(1,4,8), "ent_v_cache": torch.zeros(1,4,16)})
        c.clear()
        assert c.gpu_page_count == 0


# === 11. FeedForward (grouped projection) ===
class TestFeedForward:
    def test_forward(self):
        from tesm_ssm.models.mixer_seq_simple import FeedForward
        c = TESMConfig(d_model=64, d_intermediate=128, kernel_backend="torch",
                        bit_eps=1e-5, bit_threshold=0.5)
        ff = FeedForward(c); ff.eval()
        with torch.no_grad():
            assert ff(torch.randn(2,16,64)).shape == (2,16,64)

    def test_grouped_gate_up(self):
        from tesm_ssm.models.mixer_seq_simple import FeedForward
        c = TESMConfig(d_model=64, d_intermediate=128, kernel_backend="torch",
                        bit_eps=1e-5, bit_threshold=0.5)
        ff = FeedForward(c); ff.eval()
        x = torch.randn(2,16,64)
        with torch.no_grad():
            g, u = ff._grouped_gate_up(x)
        assert g.shape == (2,16,128) and u.shape == (2,16,128)


# === 12. 数值稳定性 ===
class TestNumericalStability:
    def test_state_scan_float64(self):
        """状态扫描使用 float64 防止下溢"""
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.eval()
        # 极小 decay 值
        x = torch.randn(1, 64, 64) * 0.01
        with torch.no_grad():
            y, _ = m(x)
        assert torch.isfinite(y).all()

    def test_mimo_stable_scan(self):
        m = TESMMIMO_Optimized(d_model=64, d_state=16, n_heads=2, mimo_rank=2,
                                expand=2, ent_rank=8, entanglement_window=8,
                                max_seq_len=64, kernel_backend="torch", decay_init_bias=0.0)
        decay = torch.sigmoid(torch.randn(1,64,2,16)) * 0.01
        update = torch.randn(1,64,2,16) * 0.01
        s = m._parallel_state_scan_mimo_stable(decay, update)
        assert torch.isfinite(s).all()

    def test_long_sequence(self):
        m = TESM_SISO(d_model=32, d_state=8, expand=2, ent_rank=4,
                       entanglement_window=4, max_seq_len=128,
                       kernel_backend="torch", decay_init_bias=2.0)
        m.eval()
        with torch.no_grad():
            y, _ = m(torch.randn(1, 128, 32))
        assert torch.isfinite(y).all()


# === 13. 纠缠统计 ===
class TestEntanglementStats:
    def test_stats_buffer_training(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.train(); m(torch.randn(2,16,64))
        assert m._stats_ternary_buffer is not None

    def test_stats_map_eval(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.eval()
        with torch.no_grad(): m(torch.randn(2,16,64))
        assert m._stats_ternary_buffer is not None or m.last_entanglement_map is not None

    def test_mimo_stats(self):
        m = TESMMIMO_Optimized(d_model=64, d_state=16, n_heads=2, mimo_rank=2,
                                expand=2, ent_rank=8, entanglement_window=8,
                                max_seq_len=64, kernel_backend="torch", decay_init_bias=0.0)
        m.train(); m(torch.randn(2,16,64))
        assert m._stats_ternary_buffer is not None or m._ternary_stats_for_logging is not None


# === 14. MIMO PyTorch 回退 ===
class TestMIMOPyTorchFallback:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, n_heads=2, mimo_rank=2, expand=2,
                 ent_rank=8, entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESMMIMO_Optimized(**d)

    def test_local_ent_pytorch(self):
        m = self._m(); m.eval()
        q,k,v = torch.randn(2,16,2,8), torch.randn(2,16,2,8), torch.randn(2,16,2,16)
        bias = torch.randn(2,8)*0.02
        with torch.no_grad():
            out = m._compute_local_entanglement_pytorch_mimo(q,k,v,bias)
        assert out.shape == v.shape

    def test_global_ent_mimo(self):
        m = self._m(entanglement_window=0); m.eval()
        q,k,s = torch.randn(2,8,2,8), torch.randn(2,8,2,8), torch.randn(2,8,2,16)
        with torch.no_grad():
            ent, ec = m._compute_global_entanglement_mimo(q,k,s)
        assert ent.shape == s.shape and ec.shape == s.shape

    def test_scan_fallback(self):
        m = self._m()
        s = m._parallel_state_scan_mimo(torch.sigmoid(torch.randn(2,16,2,16)), torch.randn(2,16,2,16))
        assert s.shape == (2,16,2,16) and torch.isfinite(s).all()


# === 15. 语义激活 ===
class TestSemanticActivation:
    def _model(self, **kw):
        d = dict(d_model=64, n_layer=1, d_intermediate=128, vocab_size=200,
                 max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                 kernel_backend="torch", decay_init_bias=0.0, tie_embeddings=False,
                 vocab_suppression=True, suppression_bias=-10.0,
                 semantic_activation=True, semantic_activation_strength=0.5,
                 semantic_activation_threshold=0.3)
        d.update(kw); return TESMLMHeadModel(TESMConfig(**d))

    def test_semantic_buffers(self):
        m = self._model()
        assert m.related_token_ids.shape == (200,100)
        assert m.token_freq.shape == (200,)

    def test_get_related_empty(self):
        m = self._model(); m.eval()
        assert m._get_related_tokens(torch.tensor([1,2,3])) == {}

    def test_sparse_logits(self):
        m = self._model(); m.eval()
        with torch.no_grad():
            out, _ = m(torch.randint(0,200,(1,8)), sparse_logits=True)
        assert out.logits is not None


# === 16. Int2InferenceEngine ===
class TestInt2InferenceEngine:
    def _tiny(self):
        return TESMLMHeadModel(TESMConfig(d_model=64, n_layer=1, d_intermediate=128,
                                           vocab_size=100, max_seq_len=32, d_state=8,
                                           ent_rank=4, entanglement_window=4,
                                           kernel_backend="torch", decay_init_bias=0.0,
                                           tie_embeddings=False))

    def test_from_trained_model(self):
        from tesm_ssm.utils.int2_inference import Int2InferenceEngine
        engine = Int2InferenceEngine.from_trained_model(self._tiny(), device='cpu')
        assert engine is not None

    def test_forward(self):
        from tesm_ssm.utils.int2_inference import Int2InferenceEngine
        engine = Int2InferenceEngine.from_trained_model(self._tiny(), device='cpu')
        with torch.no_grad():
            out = engine.forward(torch.randint(0,100,(1,8)))
        assert out is not None

    def test_model_info(self):
        from tesm_ssm.utils.int2_inference import Int2InferenceEngine
        engine = Int2InferenceEngine.from_trained_model(self._tiny(), device='cpu')
        info = engine.get_model_info()
        assert 'total_params' in info and 'int2_params' in info


# === 17. 工具函数 ===
class TestUtilities:
    def test_merge_stats_empty(self):
        from tesm_ssm.models.mixer_seq_simple import _merge_stats
        assert _merge_stats([]) is None

    def test_merge_stats_with_data(self):
        from tesm_ssm.models.mixer_seq_simple import _merge_stats
        t = torch.tensor([1.0, -1.0, 0.0, 0.5, -0.3])
        r = _merge_stats([(t, 5.0)])
        assert r is not None and "positive" in r and "negative" in r

    def test_merge_stats_multiple(self):
        from tesm_ssm.models.mixer_seq_simple import _merge_stats
        r = _merge_stats([(torch.tensor([1.0,0.0,-1.0]),3.0),(torch.tensor([0.0,0.0,1.0]),3.0)])
        assert r is not None

    def test_init_weights(self):
        from tesm_ssm.models.mixer_seq_simple import _init_weights
        l = nn.Linear(64,64); _init_weights(l, n_layer=6)
        assert l.weight.std() < 1.0


# === 18. 梯度 checkpointing ===
class TestGradientCheckpointing:
    def test_checkpoint_enabled(self):
        c = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, vocab_size=100,
                        max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                        kernel_backend="torch", decay_init_bias=0.0,
                        gradient_checkpointing=True, tie_embeddings=False)
        m = TESMLMHeadModel(c); m.train()
        out, _ = m(torch.randint(0,100,(2,16)), labels=torch.randint(0,100,(2,16)))
        out.loss.backward()
        assert any(p.grad is not None for p in m.parameters() if p.requires_grad)

    def test_checkpoint_disabled(self):
        c = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, vocab_size=100,
                        max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                        kernel_backend="torch", decay_init_bias=0.0,
                        gradient_checkpointing=False, tie_embeddings=False)
        m = TESMLMHeadModel(c); m.train()
        out, _ = m(torch.randint(0,100,(2,16)), labels=torch.randint(0,100,(2,16)))
        out.loss.backward()
        assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


# === 19. TESMCausalLMOutput ===
class TestCausalLMOutput:
    def test_output_fields(self):
        from tesm_ssm.models.mixer_seq_simple import TESMCausalLMOutput
        out = TESMCausalLMOutput(loss=torch.tensor(1.0), logits=torch.randn(2,16,100))
        assert out.loss is not None and out.logits is not None
        assert out.hidden_states is None and out.entanglement_maps is None


# === 20. 退火步数递增 ===
class TestAnnealingStepIncrement:
    def test_step_increments(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0,
                       annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100)
        m.train()
        s0 = m.annealing_step.item()
        m(torch.randn(2,16,64))
        assert m.annealing_step.item() == s0 + 1

    def test_no_increment_disabled(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0, annealing_enabled=False)
        m.train()
        s0 = m.annealing_step.item()
        m(torch.randn(2,16,64))
        assert m.annealing_step.item() == s0

    def test_no_increment_eval(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0,
                       annealing_enabled=True, T_start=10.0, T_end=0.1, annealing_steps=100)
        m.eval()
        s0 = m.annealing_step.item()
        with torch.no_grad(): m(torch.randn(2,16,64))
        assert m.annealing_step.item() == s0


# === 21. BitLinear 量化精度 ===
class TestBitLinearQuantizationPrecision:
    def test_weight_ternary(self):
        l = BitLinear(64, 128, kernel_backend="torch")
        qw = l.quantized_weight()
        scale = 1.0 / l.weight.detach().abs().mean().clamp_min(l.bit_eps)
        norm = qw * scale
        close = ((norm.abs()-1).abs()<0.1) | (norm.abs()<0.1)
        assert close.float().mean() > 0.8

    def test_ste_gradient(self):
        l = BitLinear(64, 128, kernel_backend="torch")
        x = torch.randn(2,16,64, requires_grad=True)
        qi = l.quantized_input(x)
        qi.sum().backward()
        assert x.grad is not None and x.grad.abs().mean() < 2.0


# === 22. 模型参数统计 ===
class TestModelParamCount:
    def test_siso_param_count(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64, kernel_backend="torch")
        n = sum(p.numel() for p in m.parameters())
        assert n > 0

    def test_mimo_param_count(self):
        m = TESMMIMO_Optimized(d_model=64, d_state=16, n_heads=2, mimo_rank=2,
                                expand=2, ent_rank=8, entanglement_window=8,
                                max_seq_len=64, kernel_backend="torch")
        n = sum(p.numel() for p in m.parameters())
        assert n > 0

    def test_lm_param_count(self):
        m = TESMLMHeadModel(TESMConfig(d_model=64, n_layer=2, d_intermediate=128,
                                        vocab_size=100, max_seq_len=64, d_state=16,
                                        ent_rank=8, entanglement_window=8,
                                        kernel_backend="torch", decay_init_bias=0.0,
                                        tie_embeddings=False))
        n = sum(p.numel() for p in m.parameters())
        assert n > 0


# === 23. 增量推理细节 ===
class TestIncrementalDetails:
    def test_state_accumulates(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=3.0)
        m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1,8,64), inference_params={"state_cache": cache})
        s1 = cache['state'].clone()
        with torch.no_grad():
            m(torch.randn(1,1,64), inference_params={"state_cache": cache})
        assert not torch.allclose(cache['state'], s1, atol=1e-6)
        assert cache['seq_pos'] == 9

    def test_cache_idx_rotation(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=4, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1,4,64), inference_params={"state_cache": cache})
        for _ in range(10):
            with torch.no_grad():
                m(torch.randn(1,1,64), inference_params={"state_cache": cache})
        assert 0 <= cache['cache_idx'] < 4


# === 24. _compute_entanglement 直接 ===
class TestComputeEntanglementDirect:
    def test_local_direct(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.eval()
        with torch.no_grad():
            out = m._compute_entanglement(torch.randn(2,16,8), torch.randn(2,16,8), torch.randn(2,16,16))
        assert out.shape == (2,16,16) and torch.isfinite(out).all()

    def test_global_direct(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=0, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0)
        m.eval()
        with torch.no_grad():
            out = m._compute_entanglement(torch.randn(2,8,8), torch.randn(2,8,8), torch.randn(2,8,16))
        assert out.shape == (2,8,16) and torch.isfinite(out).all()


# === 25. 词表抑制/稀疏采样 ===
class TestVocabSuppressionSparse:
    def _m(self, **kw):
        d = dict(d_model=64, n_layer=1, d_intermediate=128, vocab_size=200,
                 max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                 kernel_backend="torch", decay_init_bias=0.0, tie_embeddings=False,
                 vocab_suppression=True, suppression_bias=-10.0)
        d.update(kw); return TESMLMHeadModel(TESMConfig(**d))

    def test_apply_suppression(self):
        m = self._m(); m.eval()
        logits = torch.randn(1, 200); active = [1, 5, 10]
        out = m._apply_vocab_suppression(logits, active)
        for i in range(200):
            if i not in active:
                assert abs(out[0,i] - (logits[0,i] - 10.0)) < 1e-5

    def test_sparse_sample(self):
        m = self._m(); m.eval()
        sparse = torch.randn(1, 5); active = {1, 5, 10, 50, 100}
        full = m._sparse_sample(sparse, active, temperature=1.0, top_k=0)
        assert full.shape == (1, 200)
        inactive = torch.ones(200, dtype=torch.bool); inactive[list(active)] = False
        assert (full[:, inactive] == float('-inf')).all()

    def test_sparse_logits(self):
        m = self._m(); m.eval()
        with torch.no_grad():
            sl, ids = m._compute_sparse_logits(torch.randint(0,200,(1,8)), torch.randn(1,8,64))
        assert sl.shape[0] == 1 and len(ids) > 0

    def test_generate_sparse(self):
        m = self._m(); m.eval()
        with torch.no_grad():
            gen = m.generate(torch.randint(0,200,(1,4)), max_new_tokens=3,
                             use_cache=False, sparse_inference=True, dynamic_activation=True)
        assert gen.shape[0] == 1


# === 26. Int2 工具链 ===
class TestInt2Utils:
    def _tiny(self):
        return TESMLMHeadModel(TESMConfig(d_model=64, n_layer=1, d_intermediate=128,
                                           vocab_size=100, max_seq_len=32, d_state=8,
                                           ent_rank=4, entanglement_window=4,
                                           kernel_backend="torch", decay_init_bias=0.0,
                                           tie_embeddings=False))

    def test_load_weights(self):
        from tesm_ssm.utils.int2_quantization import export_model_to_int2, load_int2_weights_to_model, Int2Linear
        m = self._tiny()
        exported = export_model_to_int2(m)
        load_int2_weights_to_model(m, exported)
        assert any(isinstance(mod, Int2Linear) for mod in m.modules())

    def test_int2_model(self):
        from tesm_ssm.utils.int2_quantization import Int2Model
        im = Int2Model(original_model=self._tiny())
        assert im.get_model_size_mb() > 0

    def test_int2_save_hook(self):
        from tesm_ssm.utils.int2_quantization import Int2SaveHook
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            hook = Int2SaveHook(self._tiny(), save_dir=td, save_fp32=True, save_int2=True)
            hook.save(epoch=0, force=True)
            assert os.path.exists(os.path.join(td, "model_fp32_epoch_0.pt"))


# === 27. get/set embeddings ===
class TestEmbeddings:
    def test_get_set(self):
        m = TESMLMHeadModel(TESMConfig(d_model=64, n_layer=1, d_intermediate=128,
                                        vocab_size=100, max_seq_len=32, d_state=8,
                                        ent_rank=4, entanglement_window=4,
                                        kernel_backend="torch", decay_init_bias=0.0,
                                        tie_embeddings=False))
        assert m.get_input_embeddings() is m.backbone.embedding
        m.set_input_embeddings(nn.Embedding(100, 64))
        assert m.backbone.embedding is not None


# === 28. MixerModel allocate_inference_cache ===
class TestMixerAllocateCache:
    def test_allocate(self):
        c = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, vocab_size=100,
                        max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                        kernel_backend="torch", decay_init_bias=0.0, tie_embeddings=False)
        cache = TESMLMHeadModel(c).backbone.allocate_inference_cache(1, 64)
        assert 0 in cache and 1 in cache and 'state' in cache[0]


# === 29. residual_in_fp32 ===
class TestResidualFP32:
    def test_residual_fp32(self):
        c = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, vocab_size=100,
                        max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                        kernel_backend="torch", decay_init_bias=0.0,
                        residual_in_fp32=True, tie_embeddings=False)
        m = TESMLMHeadModel(c); m.train()
        out, _ = m(torch.randint(0,100,(2,16)), labels=torch.randint(0,100,(2,16)))
        out.loss.backward()
        assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


# === 30. Dropout ===
class TestDropout:
    def test_dropout_enabled(self):
        m = TESM_SISO(d_model=64, d_state=16, expand=2, ent_rank=8,
                       entanglement_window=8, max_seq_len=64,
                       kernel_backend="torch", decay_init_bias=0.0, dropout=0.5)
        assert m.dropout.p == 0.5


# === 31. _merge_stats ===
class TestMergeStats:
    def test_empty(self):
        from tesm_ssm.models.mixer_seq_simple import _merge_stats
        assert _merge_stats([]) is None

    def test_with_ternary(self):
        from tesm_ssm.models.mixer_seq_simple import _merge_stats
        ternary = torch.tensor([1.0, -1.0, 0.0, 0.5])
        result = _merge_stats([(ternary, 4.0)])
        assert result is not None
        assert 'positive' in result and 'negative' in result and 'zero' in result


# === 32. MIMO allocate_inference_cache ===
class TestMIMOAllocateCache:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, n_heads=2, mimo_rank=2, expand=2,
                 ent_rank=8, entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESMMIMO_Optimized(**d)

    def test_cache_shape(self):
        m = self._m()
        cache = m.allocate_inference_cache(2, 64)
        # state: (batch, n_heads, d_state)
        assert cache['state'].shape == (2, 2, 16)
        assert cache['state'].dtype == torch.float64
        # ent_k_cache: (batch, window, n_heads, ent_rank)
        assert cache['ent_k_cache'].shape == (2, 8, 2, 8)
        # ent_v_cache: (batch, window, n_heads, d_state)
        assert cache['ent_v_cache'].shape == (2, 8, 2, 16)

    def test_paged_cache_shape(self):
        m = self._m()
        cache = m.allocate_inference_cache(2, 4096, use_paged_cache=True, page_size=512)
        assert cache['use_paged']
        assert cache['state'].shape == (2, 2, 16)


# === 33. MIMO 增量推理 ===
class TestMIMOIncremental:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, n_heads=2, mimo_rank=2, expand=2,
                 ent_rank=8, entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESMMIMO_Optimized(**d)

    def test_incremental_local(self):
        m = self._m(); m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1, 8, 64), inference_params={"state_cache": cache})
            y, _ = m(torch.randn(1, 1, 64), inference_params={"state_cache": cache})
        assert y.shape == (1, 1, 64)
        assert cache['seq_pos'] == 9

    def test_incremental_global(self):
        m = self._m(entanglement_window=0); m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1, 8, 64), inference_params={"state_cache": cache})
            y, _ = m(torch.randn(1, 1, 64), inference_params={"state_cache": cache})
        assert y.shape == (1, 1, 64)

    def test_state_accumulates(self):
        m = self._m(decay_init_bias=3.0); m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1, 8, 64), inference_params={"state_cache": cache})
        s1 = cache['state'].clone()
        with torch.no_grad():
            m(torch.randn(1, 1, 64), inference_params={"state_cache": cache})
        assert not torch.allclose(cache['state'], s1, atol=1e-6)

    def test_cache_idx_rotation(self):
        m = self._m(entanglement_window=4); m.eval()
        cache = m.allocate_inference_cache(1, 64)
        with torch.no_grad():
            m(torch.randn(1, 4, 64), inference_params={"state_cache": cache})
        for _ in range(10):
            with torch.no_grad():
                m(torch.randn(1, 1, 64), inference_params={"state_cache": cache})
        assert 0 <= cache['cache_idx'] < 4


# === 34. MIMO _compute_entanglement 覆盖 ===
class TestMIMOComputeEntanglement:
    def _m(self, **kw):
        d = dict(d_model=64, d_state=16, n_heads=2, mimo_rank=2, expand=2,
                 ent_rank=8, entanglement_window=8, max_seq_len=64,
                 kernel_backend="torch", decay_init_bias=0.0)
        d.update(kw); return TESMMIMO_Optimized(**d)

    def test_local_entanglement_4d(self):
        m = self._m(); m.eval()
        q = torch.randn(2, 16, 2, 8)
        k = torch.randn(2, 16, 2, 8)
        v = torch.randn(2, 16, 2, 16)
        with torch.no_grad():
            out = m._compute_entanglement(q, k, v)
        assert out.shape == (2, 16, 2, 16)

    def test_global_entanglement_4d(self):
        m = self._m(entanglement_window=0); m.eval()
        q = torch.randn(2, 8, 2, 8)
        k = torch.randn(2, 8, 2, 8)
        v = torch.randn(2, 8, 2, 16)
        with torch.no_grad():
            out = m._compute_entanglement(q, k, v)
        assert out.shape == (2, 8, 2, 16)
        assert torch.isfinite(out).all()


# === 35. MIMO LM模型增量推理 ===
class TestMIMOLMIncremental:
    def _tiny(self, **kw):
        d = dict(d_model=64, n_layer=2, d_intermediate=128, vocab_size=1000,
                 max_seq_len=64, d_state=16, ent_rank=8, entanglement_window=8,
                 kernel_backend="torch", decay_init_bias=0.0, tie_embeddings=False,
                 use_mimo=True, n_heads=2)
        d.update(kw); return TESMLMHeadModel(TESMConfig(**d))

    def test_mimo_generate_with_cache(self):
        m = self._tiny(); m.eval()
        ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=4, use_cache=True)
        assert gen.shape[0] == 1 and gen.shape[1] >= 4

    def test_mimo_generate_no_cache(self):
        m = self._tiny(); m.eval()
        ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=4, use_cache=False)
        assert gen.shape[0] == 1

    def test_mimo_forward_backward(self):
        m = self._tiny(); m.train()
        ids = torch.randint(0, 1000, (2, 16))
        out, _ = m(ids, labels=ids)
        out.loss.backward()
        has_grad = any(p.grad is not None for p in m.parameters() if p.requires_grad)
        assert has_grad
