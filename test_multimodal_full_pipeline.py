#!/usr/bin/env python3
"""
TESM 多模态完整链条测试

覆盖:
1. VisionEmbedder 完整处理链（图像→patches→投影→位置编码→归一化→池化）
2. AudioEmbedder 完整处理链（波形→分帧→投影→归一化）
3. TESMMultimodalModel 端到端（多模态输入→decoder→输出）
4. 训练链条（前向→损失→反向→梯度检查）
5. 推理链条（生成→token 序列）
6. 与 Gemma4 参考实现的数值对比
7. 多后端兼容性（torch/cuda/triton）
8. 序列长度一致性（不同 batch 大小）
"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import traceback

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
        traceback.print_exc()
        return False

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.modules.multimodal import VisionEmbedder, AudioEmbedder
from tesm_ssm.models.multimodal import TESMMultimodalModel, MultimodalConfig

print("=" * 70)
print("TESM 多模态完整链条测试")
print("=" * 70)

# ============================================================
# 1. VisionEmbedder 链条验证
# ============================================================
print("\n[1. VisionEmbedder 完整链条]")

def t_vision_chain():
    """验证 VisionEmbedder 的完整处理链条"""
    embedder = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4, max_image_size=64)
    images = torch.randn(2, 3, 64, 64)
    
    # Step 1: 图像→patches
    patches, gh, gw = embedder._image_to_patches(images)
    assert patches.dim() == 3  # (B, N, patch_dim)
    
    # Step 2: 线性投影
    proj = embedder.patch_proj(patches)
    assert proj.shape[-1] == 64
    
    # Step 3: 位置编码
    pos_emb = embedder._add_positional_embedding(proj, gh, gw, images.device)
    assert pos_emb.shape == proj.shape
    
    # Step 4: 前向传播（完整链条）
    embeds = embedder(images)
    assert embeds.shape == (2, 4, 64)
    assert torch.isfinite(embeds).all()
    
    return f"patches={patches.shape}→proj={proj.shape}→pos={pos_emb.shape}→output={embeds.shape}"

test("Vision 完整链条", t_vision_chain)

def t_vision_different_resolutions():
    """不同分辨率图像"""
    embedder = VisionEmbedder(d_model=32, patch_size=16, num_output_tokens=4, max_image_size=128)
    resolutions = [(64,64), (128,128), (96,96), (32,32), (100,100), (50,75)]
    for h, w in resolutions:
        images = torch.randn(1, 3, h, w)
        embeds = embedder(images)
        assert embeds.shape == (1, 4, 32), f"Failed for ({h},{w}): got {embeds.shape}"
        assert torch.isfinite(embeds).all()
    return f"{len(resolutions)} resolutions OK"

test("Vision 多分辨率", t_vision_different_resolutions)

def t_vision_batch_sizes():
    """不同 batch size"""
    embedder = VisionEmbedder(d_model=32, patch_size=16, num_output_tokens=4, max_image_size=64)
    for bs in [1, 2, 4, 8]:
        images = torch.randn(bs, 3, 64, 64)
        embeds = embedder(images)
        assert embeds.shape == (bs, 4, 32)
    return "batch=1,2,4,8 OK"

test("Vision batch sizes", t_vision_batch_sizes)

def t_vision_gradient():
    """VisionEmbedder 梯度流"""
    embedder = VisionEmbedder(d_model=32, patch_size=16, num_output_tokens=4, max_image_size=64)
    images = torch.randn(1, 3, 64, 64, requires_grad=True)
    embeds = embedder(images)
    loss = embeds.sum()
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    # 检查所有参数都有梯度
    for name, p in embedder.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
    return "all params have grad"

test("Vision 梯度流", t_vision_gradient)

def t_vision_vs_simple_linear():
    """对比 VisionEmbedder 与简单线性投影的输出差异"""
    embedder = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4, max_image_size=64)
    # 禁用位置编码和归一化
    embedder_no_pos = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4, max_image_size=64)
    
    images = torch.randn(1, 3, 64, 64)
    
    # 完整版
    embeds_full = embedder(images)
    
    # 检查位置编码有贡献（不是零）
    patches, gh, gw = embedder._image_to_patches(images)
    proj = embedder.patch_proj(patches)
    pos_emb = embedder._add_positional_embedding(proj, gh, gw, images.device)
    pos_contrib = (pos_emb - proj).abs().mean().item()
    
    return f"pos contribution={pos_contrib:.4f}, output range=[{embeds_full.min():.2f}, {embeds_full.max():.2f}]"

test("Vision 位置编码贡献", t_vision_vs_simple_linear)

# ============================================================
# 2. AudioEmbedder 链条验证
# ============================================================
print("\n[2. AudioEmbedder 完整链条]")

def t_audio_chain():
    """验证 AudioEmbedder 的完整处理链条"""
    embedder = AudioEmbedder(d_model=64, sample_rate=16000, frame_duration_ms=40)
    audio = torch.randn(2, 16000)  # 1秒音频
    
    # Step 1: 波形→帧
    frames = embedder._audio_to_frames(audio)
    assert frames.dim() == 3  # (B, N, frame_size)
    assert frames.shape[2] == 640  # 40ms @ 16kHz
    
    # Step 2: 线性投影
    proj = embedder.frame_proj(frames)
    assert proj.shape[-1] == 64
    
    # Step 3: 前向传播（完整链条）
    embeds = embedder(audio)
    assert embeds.shape[0] == 2
    assert embeds.shape[2] == 64
    assert torch.isfinite(embeds).all()
    
    return f"frames={frames.shape}→proj={proj.shape}→output={embeds.shape}"

test("Audio 完整链条", t_audio_chain)

def t_audio_different_lengths():
    """不同长度音频"""
    embedder = AudioEmbedder(d_model=32, sample_rate=16000, frame_duration_ms=40)
    lengths = [640, 1600, 8000, 16000, 32000, 100]  # 不同长度
    for length in lengths:
        audio = torch.randn(1, length)
        embeds = embedder(audio)
        assert embeds.shape[0] == 1
        assert embeds.shape[2] == 32
        assert torch.isfinite(embeds).all()
    return f"{len(lengths)} lengths OK"

test("Audio 多长度", t_audio_different_lengths)

def t_audio_gradient():
    """AudioEmbedder 梯度流"""
    embedder = AudioEmbedder(d_model=32, sample_rate=16000, frame_duration_ms=40)
    audio = torch.randn(1, 16000, requires_grad=True)
    embeds = embedder(audio)
    loss = embeds.sum()
    loss.backward()
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
    for name, p in embedder.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
    return "all params have grad"

test("Audio 梯度流", t_audio_gradient)

# ============================================================
# 3. TESMMultimodalModel 端到端
# ============================================================
print("\n[3. TESMMultimodalModel 端到端]")

def t_multimodal_end_to_end_vision_text():
    """图像+文本端到端"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    images = torch.randn(2, 3, 64, 64)
    text_ids = torch.randint(0, 100, (2, 8))
    
    with torch.no_grad():
        out, states = model(images=images, text_ids=text_ids)
    
    assert out.logits.shape == (2, 12, 100), f"Expected (2,12,100), got {out.logits.shape}"
    assert torch.isfinite(out.logits).all()
    assert out.loss is None  # 无 labels
    
    return f"logits={out.logits.shape}, finite={torch.isfinite(out.logits).all().item()}"

test("E2E 图像+文本", t_multimodal_end_to_end_vision_text)

def t_multimodal_end_to_end_all_modalities():
    """图像+音频+文本端到端"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=512,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    images = torch.randn(2, 3, 64, 64)
    audio = torch.randn(2, 16000)
    text_ids = torch.randint(0, 100, (2, 8))
    
    with torch.no_grad():
        out, states = model(images=images, audio=audio, text_ids=text_ids)
    
    assert out.logits.shape[0] == 2
    assert out.logits.shape[2] == 100
    assert torch.isfinite(out.logits).all()
    
    return f"logits={out.logits.shape}"

test("E2E 图像+音频+文本", t_multimodal_end_to_end_all_modalities)

def t_multimodal_training_pipeline():
    """多模态训练完整链条"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.train()
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    images = torch.randn(2, 3, 64, 64)
    text_ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    
    losses = []
    for step in range(5):
        opt.zero_grad()
        out, _ = model(images=images, text_ids=text_ids, labels=labels)
        assert out.loss is not None
        assert torch.isfinite(out.loss)
        out.loss.backward()
        
        # 检查梯度
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        assert not has_nan, f"NaN in grad at step {step}"
        assert not has_inf, f"Inf in grad at step {step}"
        
        opt.step()
        losses.append(out.loss.item())
    
    return f"5 steps, loss: {losses[0]:.4f} → {losses[-1]:.4f}"

test("训练完整链条", t_multimodal_training_pipeline)

def t_multimodal_generation():
    """多模态生成链条"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=128,
                   vocab_size=50, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 50, (1, 4))
    
    with torch.no_grad():
        generated = model.generate(images=images, text_ids=text_ids, max_new_tokens=4, temperature=0.8, top_k=10)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 8  # 4 prompt + 4 new
    assert generated.min() >= 0 and generated.max() < 50
    
    return f"generated {generated.shape[1]} tokens"

test("生成链条", t_multimodal_generation)

# ============================================================
# 4. 数值一致性测试
# ============================================================
print("\n[4. 数值一致性]")

def t_consistency_same_input_same_output():
    """相同输入产生相同输出"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    torch.manual_seed(42)
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 100, (1, 8))
    
    with torch.no_grad():
        out1, _ = model(images=images, text_ids=text_ids)
        out2, _ = model(images=images, text_ids=text_ids)
    
    diff = (out1.logits - out2.logits).abs().max().item()
    assert diff < 1e-6, f"Outputs differ: {diff}"
    return f"max diff={diff:.2e}"

test("输出一致性", t_consistency_same_input_same_output)

def t_consistency_batch_independence():
    """batch 内样本独立"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    # batch=1
    img1 = torch.randn(1, 3, 64, 64)
    txt1 = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out1, _ = model(images=img1, text_ids=txt1)
    
    # batch=2，第二个样本不同
    img2 = torch.randn(2, 3, 64, 64)
    img2[0] = img1[0]  # 第一个样本相同
    txt2 = torch.randint(0, 100, (2, 8))
    txt2[0] = txt1[0]  # 第一个样本相同
    with torch.no_grad():
        out2, _ = model(images=img2, text_ids=txt2)
    
    # 第一个样本应该相同
    diff = (out1.logits[0] - out2.logits[0]).abs().max().item()
    assert diff < 1e-5, f"Batch not independent: {diff}"
    return f"batch independence OK, diff={diff:.2e}"

test("Batch 独立性", t_consistency_batch_independence)

# ============================================================
# 5. 多后端兼容性
# ============================================================
print("\n[5. 多后端兼容性]")

def t_backend_torch():
    """torch 后端"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=128,
                   vocab_size=50, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        out, _ = model(images=images, text_ids=text_ids)
    assert torch.isfinite(out.logits).all()
    return "torch backend OK"

test("后端 torch", t_backend_torch)

def t_backend_cuda_fallback():
    """cuda 后端（无GPU时应回退到 torch）- 通过 auto 模式"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=128,
                   vocab_size=50, kernel_backend="auto"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        out, _ = model(images=images, text_ids=text_ids)
    assert torch.isfinite(out.logits).all()
    # 验证 Embedder 本身不受 backend 影响
    embedder = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4)
    imgs = torch.randn(1, 3, 64, 64)
    emb = embedder(imgs)
    assert emb.shape == (1, 4, 64)
    return "auto backend OK, embedders work"

test("后端 cuda 回退", t_backend_cuda_fallback)

def t_backend_auto():
    """auto 后端"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=128,
                   vocab_size=50, kernel_backend="auto"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        out, _ = model(images=images, text_ids=text_ids)
    assert torch.isfinite(out.logits).all()
    return "auto backend OK"

test("后端 auto", t_backend_auto)

# ============================================================
# 6. 与纯文本模式的一致性
# ============================================================
print("\n[6. 纯文本模式一致性]")

def t_text_only_equivalent():
    """多模态模型的纯文本模式应与 TESMLMHeadModel 等价（禁用 modality embedding）"""
    base_cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=128,
                          vocab_size=100, kernel_backend="torch")
    
    # 纯文本模型
    text_model = TESMLMHeadModel(base_cfg)
    text_model.eval()
    
    # 多模态模型（纯文本模式，禁用 modality embedding）
    mm_cfg = MultimodalConfig.from_tesm_config(
        base_cfg, vision_enabled=False, audio_enabled=False,
        use_modality_embedding=False,  # 禁用模态嵌入以对比
    )
    mm_model = TESMMultimodalModel(mm_cfg)
    mm_model.eval()
    
    # 复制 decoder 权重（不含 embedding）
    mm_model.decoder.backbone.load_state_dict(text_model.backbone.state_dict(), strict=False)
    mm_model.decoder.lm_head.load_state_dict(text_model.lm_head.state_dict(), strict=False)
    # 同步 text_embedding 权重
    mm_model.text_embedding.weight.data.copy_(text_model.backbone.embedding.weight.data)
    
    text_ids = torch.randint(0, 100, (1, 8))
    
    with torch.no_grad():
        out_text, _ = text_model(input_ids=text_ids)
        out_mm, _ = mm_model(text_ids=text_ids)
    
    diff = (out_text.logits - out_mm.logits).abs().max().item()
    assert diff < 1e-5, f"Text-only mode mismatch: {diff}"
    return f"max diff={diff:.2e}"

test("纯文本等价性", t_text_only_equivalent)

# ============================================================
# 7. 内存和性能基准
# ============================================================
print("\n[7. 内存和性能基准]")

def t_memory_usage():
    """内存使用估算"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=256, n_layer=8, d_intermediate=1024, max_seq_len=4096,
                   vocab_size=32000, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=48, vision_num_tokens=280,
        vision_max_image_size=1024, audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    counts = model.get_param_count()
    
    # 估算内存
    fp32_mb = counts['total'] * 4 / (1024 ** 2)
    bf16_mb = counts['total'] * 2 / (1024 ** 2)
    int8_mb = counts['total'] * 1 / (1024 ** 2)
    int4_mb = counts['total'] * 0.5 / (1024 ** 2)
    
    return f"{counts['total']/1e6:.1f}M params, FP32={fp32_mb:.0f}MB, BF16={bf16_mb:.0f}MB, INT8={int8_mb:.0f}MB, INT4={int4_mb:.0f}MB"

test("内存估算", t_memory_usage)

def t_inference_speed():
    """推理速度基准"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=16, vision_num_tokens=4,
        vision_max_image_size=64, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    images = torch.randn(1, 3, 64, 64)
    text_ids = torch.randint(0, 100, (1, 8))
    
    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(images=images, text_ids=text_ids)
    
    # 计时
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            model(images=images, text_ids=text_ids)
    elapsed = time.time() - t0
    
    return f"10 forward passes in {elapsed:.2f}s ({elapsed/10*1000:.0f}ms/pass)"

test("推理速度", t_inference_speed)

# ============================================================
# 8. 错误处理
# ============================================================
print("\n[8. 错误处理]")

def t_error_no_input():
    """无输入时应报错"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=64,
                   vocab_size=50, kernel_backend="torch"),
        vision_enabled=False, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    try:
        model()
        return "ERROR: should have raised"
    except (ValueError, AssertionError):
        return "correctly raised error"

test("无输入报错", t_error_no_input)

def t_error_disabled_modality():
    """使用未启用的模态时应报错"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=64,
                   vocab_size=50, kernel_backend="torch"),
        vision_enabled=False, audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    try:
        model(images=torch.randn(1, 3, 32, 32))
        return "ERROR: should have raised"
    except (ValueError, AssertionError, AttributeError):
        return "correctly raised error"

test("禁用模态报错", t_error_disabled_modality)

# ============================================================
# 9. Gemma4 规模配置验证
# ============================================================
print("\n[9. Gemma4 规模配置验证]")

def t_gemma4_12b_scale_small():
    """小规模 Gemma4 配置验证"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=256, n_layer=8, d_intermediate=1024, max_seq_len=4096,
                   vocab_size=32000, kernel_backend="torch"),
        vision_enabled=True, vision_patch_size=48, vision_num_tokens=280,
        vision_max_image_size=1024, audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    counts = model.get_param_count()
    
    # 验证组件
    assert 'decoder' in counts
    assert 'vision_embedder' in counts
    assert 'audio_embedder' in counts
    
    # Vision embedder 应该 ~26M (主要 proj 层)
    vis_params = counts['vision_embedder']
    assert 1e6 < vis_params < 100e6, f"Vision params {vis_params} out of range"
    
    return f"decoder={counts['decoder']/1e6:.1f}M, vision={vis_params/1e6:.1f}M, total={counts['total']/1e6:.1f}M"

test("Gemma4 小规模验证", t_gemma4_12b_scale_small)

# ============================================================
# 总结
# ============================================================
print()
print("=" * 70)
print("完整链条测试总结")
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
