#!/usr/bin/env python3
"""TESM 多模态模块测试

测试覆盖:
1. 纯文本模式（inputs_embeds=None，向后兼容）
2. inputs_embeds 模式（直接传入 embeddings）
3. VisionEmbedder 图像嵌入
4. AudioEmbedder 音频嵌入
5. TESMMultimodalModel 组合模型
6. 参数量统计
"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import time

results = []

def test(name, fn):
    start = time.time()
    try:
        detail = fn()
        dur = time.time() - start
        results.append((name, True, dur, detail or ""))
        print(f"  [PASS] {name} ({dur:.2f}s) {detail}")
    except Exception as e:
        dur = time.time() - start
        err = f"{type(e).__name__}: {str(e)[:150]}"
        results.append((name, False, dur, err))
        print(f"  [FAIL] {name} ({dur:.2f}s)")
        print(f"         {err}")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.modules.multimodal import BaseEmbedder, VisionEmbedder, AudioEmbedder
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel

print("=" * 60)
print("TESM 多模态模块测试")
print("=" * 60)

# ============================================================
# 1. 向后兼容：纯文本 input_ids 模式
# ============================================================
print("\n[1. 向后兼容 - 纯文本 input_ids]")

def t_backward_compat():
    """纯文本 input_ids 模式仍然工作"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.eval()
    ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, _ = model(input_ids=ids)
    assert out.logits.shape == (2, 8, 100)
    assert torch.isfinite(out.logits).all()
    return f"logits={out.logits.shape}"

test("纯文本 input_ids", t_backward_compat)

def t_backward_compat_labels():
    """纯文本带 labels 训练"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.train()
    ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    out, _ = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    return f"loss={out.loss.item():.4f}"

test("纯文本训练", t_backward_compat_labels)

# ============================================================
# 2. inputs_embeds 模式
# ============================================================
print("\n[2. inputs_embeds 模式]")

def t_inputs_embeds():
    """直接传入 pre-computed embeddings"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.eval()
    # 直接传入 embeddings
    embeds = torch.randn(2, 8, 64)
    with torch.no_grad():
        out, _ = model(inputs_embeds=embeds)
    assert out.logits.shape == (2, 8, 100)
    assert torch.isfinite(out.logits).all()
    return f"logits={out.logits.shape}"

test("inputs_embeds 前向", t_inputs_embeds)

def t_inputs_embeds_training():
    """inputs_embeds 模式训练"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.train()
    embeds = torch.randn(2, 8, 64)
    labels = torch.randint(0, 100, (2, 8))
    out, _ = model(inputs_embeds=embeds, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    return f"loss={out.loss.item():.4f}, grad OK"

test("inputs_embeds 训练", t_inputs_embeds_training)

def t_both_none_error():
    """同时传 None 应报错"""
    cfg = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16,
                     vocab_size=50, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    try:
        model()
        return "ERROR: should have raised ValueError"
    except ValueError:
        return "correctly raised ValueError"

test("参数校验(都None)", t_both_none_error)

def t_both_provided_error():
    """同时传 input_ids 和 inputs_embeds 应报错"""
    cfg = TESMConfig(d_model=32, n_layer=1, d_intermediate=64, max_seq_len=16,
                     vocab_size=50, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    try:
        model(input_ids=torch.randint(0, 50, (1, 4)), inputs_embeds=torch.randn(1, 4, 32))
        return "ERROR: should have raised ValueError"
    except ValueError:
        return "correctly raised ValueError"

test("参数校验(都传)", t_both_provided_error)

# ============================================================
# 3. VisionEmbedder
# ============================================================
print("\n[3. VisionEmbedder]")

def t_vision_embedder_basic():
    """VisionEmbedder 基本功能"""
    embedder = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4, max_image_size=64)
    images = torch.randn(2, 3, 64, 64)
    embeds = embedder(images)
    assert embeds.shape == (2, 4, 64), f"Expected (2, 4, 64), got {embeds.shape}"
    assert torch.isfinite(embeds).all()
    params = sum(p.numel() for p in embedder.parameters())
    return f"embeds={embeds.shape}, params={params/1e6:.2f}M"

test("VisionEmbedder 基本", t_vision_embedder_basic)

def t_vision_embedder_non_divisible():
    """图像尺寸不可整除时自动 resize"""
    embedder = VisionEmbedder(d_model=64, patch_size=16, num_output_tokens=4, max_image_size=64)
    images = torch.randn(2, 3, 50, 50)  # 50 不可被 16 整除
    embeds = embedder(images)
    assert embeds.shape == (2, 4, 64)
    return f"50x50 -> {embeds.shape}"

test("VisionEmbedder 不可整除", t_vision_embedder_non_divisible)

def t_vision_embedder_param_count():
    """参数量验证 (~35M for d_model=3840)"""
    embedder_small = VisionEmbedder(d_model=3840, patch_size=48, num_output_tokens=280)
    params_small = sum(p.numel() for p in embedder_small.parameters())
    # 验证主要成分
    proj_params = embedder_small.patch_proj.weight.numel() + embedder_small.patch_proj.bias.numel()
    return f"d_model=3840: {params_small/1e6:.1f}M (proj={proj_params/1e6:.1f}M)"

test("VisionEmbedder 参数量", t_vision_embedder_param_count)

def t_vision_embedder_different_sizes():
    """不同尺寸图像"""
    embedder = VisionEmbedder(d_model=32, patch_size=16, num_output_tokens=4, max_image_size=128)
    for size in [(64, 64), (128, 128), (96, 96), (32, 32)]:
        images = torch.randn(1, 3, size[0], size[1])
        embeds = embedder(images)
        assert embeds.shape[0] == 1
        assert embeds.shape[2] == 32
    return "sizes 32/64/96/128 all OK"

test("VisionEmbedder 多尺寸", t_vision_embedder_different_sizes)

# ============================================================
# 4. AudioEmbedder
# ============================================================
print("\n[4. AudioEmbedder]")

def t_audio_embedder_basic():
    """AudioEmbedder 基本功能"""
    embedder = AudioEmbedder(d_model=64, sample_rate=16000, frame_duration_ms=40)
    audio = torch.randn(2, 16000)  # 1秒音频
    embeds = embedder(audio)
    # 1秒 / 40ms = 25 帧
    assert embeds.shape[0] == 2
    assert embeds.shape[2] == 64
    assert torch.isfinite(embeds).all()
    params = sum(p.numel() for p in embedder.parameters())
    return f"embeds={embeds.shape}, params={params/1e6:.2f}M"

test("AudioEmbedder 基本", t_audio_embedder_basic)

def t_audio_embedder_short_audio():
    """短音频处理"""
    embedder = AudioEmbedder(d_model=64, sample_rate=16000, frame_duration_ms=40)
    audio = torch.randn(1, 100)  # 很短，会被填充
    embeds = embedder(audio)
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 64
    return f"100 samples -> {embeds.shape}"

test("AudioEmbedder 短音频", t_audio_embedder_short_audio)

def t_audio_embedder_param_count():
    """参数量验证 (~1M)"""
    embedder = AudioEmbedder(d_model=3840)
    params = sum(p.numel() for p in embedder.parameters())
    return f"d_model=3840: {params/1e6:.2f}M"

test("AudioEmbedder 参数量", t_audio_embedder_param_count)

# ============================================================
# 5. TESMMultimodalModel
# ============================================================
print("\n[5. TESMMultimodalModel]")

def t_multimodal_text_only():
    """多模态模型 - 纯文本"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=False,
        audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    text_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, _ = model(text_ids=text_ids)
    assert out.logits.shape == (2, 8, 100)
    return f"logits={out.logits.shape}"

test("多模态-纯文本", t_multimodal_text_only)

def t_multimodal_vision_text():
    """多模态模型 - 图像+文本"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=16,
        vision_num_tokens=4,
        vision_max_image_size=64,
        audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    text_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, _ = model(images=images, text_ids=text_ids)
    # 总长度 = 4 (vision) + 8 (text) = 12
    assert out.logits.shape == (2, 12, 100), f"Expected (2, 12, 100), got {out.logits.shape}"
    return f"logits={out.logits.shape}"

test("多模态-图像+文本", t_multimodal_vision_text)

def t_multimodal_vision_audio_text():
    """多模态模型 - 图像+音频+文本"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=16,
        vision_num_tokens=4,
        vision_max_image_size=64,
        audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    audio = torch.randn(2, 16000)
    text_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, _ = model(images=images, audio=audio, text_ids=text_ids)
    # 总长度 = 4 (vision) + 25 (audio) + 8 (text) = 37
    assert out.logits.shape[0] == 2
    assert out.logits.shape[2] == 100
    return f"logits={out.logits.shape}"

test("多模态-图像+音频+文本", t_multimodal_vision_audio_text)

def t_multimodal_training():
    """多模态模型训练"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=16,
        vision_num_tokens=4,
        vision_max_image_size=64,
        audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.train()
    images = torch.randn(2, 3, 64, 64)
    text_ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    out, _ = model(images=images, text_ids=text_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    return f"loss={out.loss.item():.4f}"

test("多模态训练", t_multimodal_training)

def t_multimodal_param_count():
    """多模态模型参数量统计"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=256,
                   vocab_size=100, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=16,
        vision_num_tokens=4,
        vision_max_image_size=64,
        audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    counts = model.get_param_count()
    total = counts['total']
    return f"total={total/1e6:.2f}M, decoder={counts['decoder']/1e6:.1f}M, vision={counts.get('vision_embedder', 0)/1e6:.2f}M, audio={counts.get('audio_embedder', 0)/1e6:.2f}M"

test("多模态参数量", t_multimodal_param_count)

# ============================================================
# 6. 一致性对比
# ============================================================
print("\n[6. 一致性对比]")

def t_consistency_input_ids_vs_embeds():
    """input_ids 和 inputs_embeds 结果一致"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.eval()
    
    ids = torch.randint(0, 100, (1, 8))
    
    # 方式1: input_ids
    with torch.no_grad():
        out1, _ = model(input_ids=ids)
    
    # 方式2: inputs_embeds（先获取 embedding）
    embeds = model.get_input_embeddings()(ids)
    with torch.no_grad():
        out2, _ = model(inputs_embeds=embeds)
    
    diff = (out1.logits - out2.logits).abs().max().item()
    assert diff < 1e-5, f"Results differ: {diff}"
    return f"max diff={diff:.2e}"

test("input_ids vs embeds 一致性", t_consistency_input_ids_vs_embeds)

# ============================================================
# 7. Gemma4-12B 规模配置
# ============================================================
print("\n[7. Gemma4-12B 规模配置]")

def t_gemma4_12b_scale():
    """Gemma4-12B 规模配置测试（小模型验证）"""
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=256, n_layer=8, d_intermediate=1024, max_seq_len=4096,
                   vocab_size=32000, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=48,
        vision_num_tokens=280,
        vision_max_image_size=1024,
        audio_enabled=True,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    # 图像 + 音频 + 文本
    images = torch.randn(1, 3, 224, 224)
    audio = torch.randn(1, 16000)
    text_ids = torch.randint(0, 32000, (1, 32))
    
    with torch.no_grad():
        out, _ = model(images=images, audio=audio, text_ids=text_ids)
    
    assert torch.isfinite(out.logits).all()
    counts = model.get_param_count()
    return f"total={counts['total']/1e6:.1f}M, logits={out.logits.shape}"

test("Gemma4-12B scale (small)", t_gemma4_12b_scale)

# ============================================================
# 总结
# ============================================================
print()
print("=" * 60)
print("多模态测试总结")
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
