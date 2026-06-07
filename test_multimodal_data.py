#!/usr/bin/env python3
"""多模态数据处理测试"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import numpy as np
import time
import tempfile
import os
from pathlib import Path

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
        err = f"{type(e).__name__}: {str(e)[:200]}"
        results.append((name, False, dur, err))
        print(f"  [FAIL] {name} ({dur:.2f}s)")
        print(f"         {err}")

from tesm_ssm.training.multimodal_data import (
    MultimodalDataConfig, MultimodalDataset,
    multimodal_collate_fn, create_multimodal_dataloader,
    load_image, load_audio, pad_or_truncate_audio, image_augment,
    create_dummy_multimodal_data, create_dummy_image, create_dummy_audio,
)

print("=" * 70)
print("多模态数据处理测试")
print("=" * 70)

# 创建临时目录
tmpdir = tempfile.mkdtemp()

# 创建测试数据
img_dir = Path(tmpdir) / "images"
aud_dir = Path(tmpdir) / "audio"
img_dir.mkdir(exist_ok=True)
aud_dir.mkdir(exist_ok=True)

# 创建示例图像和音频
for i in range(5):
    create_dummy_image(str(img_dir / f"image_{i}.jpg"), size=224)
    create_dummy_audio(str(aud_dir / f"audio_{i}.wav"), duration=1.0)

# 创建示例数据文件
data_file = Path(tmpdir) / "data.jsonl"
create_dummy_multimodal_data(
    str(data_file), num_samples=20,
    image_root=str(img_dir), audio_root=str(aud_dir)
)

# ============================================================
# 1. 图像预处理测试
# ============================================================
print("\n[1. 图像预处理]")

def t_load_image():
    """加载图像"""
    config = MultimodalDataConfig(image_size=224)
    img_path = str(img_dir / "image_0.jpg")
    tensor = load_image(img_path, config)
    
    assert tensor.shape == (3, 224, 224), f"Wrong shape: {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -3.0 and tensor.max() <= 3.0  # normalized
    return f"shape={tensor.shape}, range=[{tensor.min():.2f}, {tensor.max():.2f}]"

test("加载图像", t_load_image)

def t_image_augment():
    """图像数据增强"""
    config = MultimodalDataConfig(image_size=224, image_augment=True)
    tensor = torch.randn(3, 224, 224)
    
    # 多次增强，验证输出形状不变
    for _ in range(5):
        aug = image_augment(tensor, config)
        assert aug.shape == (3, 224, 224)
    
    return "shape preserved after augmentation"

test("图像增强", t_image_augment)

# ============================================================
# 2. 音频预处理测试
# ============================================================
print("\n[2. 音频预处理]")

def t_load_audio():
    """加载音频"""
    config = MultimodalDataConfig(audio_sample_rate=16000)
    aud_path = str(aud_dir / "audio_0.wav")
    waveform = load_audio(aud_path, config)
    
    assert waveform.dim() == 1, f"Wrong dim: {waveform.dim()}"
    assert waveform.dtype == torch.float32
    assert waveform.abs().max() <= 1.0 + 1e-6  # normalized
    return f"shape={waveform.shape}, sr={config.audio_sample_rate}"

test("加载音频", t_load_audio)

def t_pad_truncate_audio():
    """音频填充/截断"""
    # 短音频 -> 填充
    short = torch.randn(8000)
    padded = pad_or_truncate_audio(short, 16000)
    assert padded.shape[0] == 16000
    
    # 长音频 -> 截断
    long = torch.randn(32000)
    truncated = pad_or_truncate_audio(long, 16000)
    assert truncated.shape[0] == 16000
    
    # 正好长度
    exact = torch.randn(16000)
    same = pad_or_truncate_audio(exact, 16000)
    assert same.shape[0] == 16000
    
    return "short->pad, long->truncate, exact->same"

test("音频填充截断", t_pad_truncate_audio)

# ============================================================
# 3. MultimodalDataset 测试
# ============================================================
print("\n[3. MultimodalDataset]")

def t_dataset_basic():
    """数据集基本功能"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=True,
        image_size=224,
        max_seq_len=128,
        vocab_size=100,
    )
    dataset = MultimodalDataset(config)
    
    assert len(dataset) == 20, f"Expected 20, got {len(dataset)}"
    
    # 获取一个样本
    sample = dataset[0]
    assert 'image' in sample
    assert 'audio' in sample
    assert 'input_ids' in sample
    assert 'labels' in sample
    assert 'attention_mask' in sample
    
    return f"{len(dataset)} samples, keys={list(sample.keys())}"

test("Dataset 基本", t_dataset_basic)

def t_dataset_shapes():
    """数据集样本形状"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=True,
        image_size=224,
        audio_max_length=16000,
        max_seq_len=128,
        vocab_size=100,
    )
    dataset = MultimodalDataset(config)
    sample = dataset[0]
    
    assert sample['image'].shape == (3, 224, 224)
    assert sample['audio'].shape[0] == 16000
    assert sample['input_ids'].shape[0] == 128
    assert sample['labels'].shape[0] == 128
    assert sample['attention_mask'].shape[0] == 128
    
    return f"image={sample['image'].shape}, audio={sample['audio'].shape}, text={sample['input_ids'].shape}"

test("Dataset 形状", t_dataset_shapes)

def t_dataset_no_image():
    """不使用图像"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=False,
        use_audio=True,
        max_seq_len=128,
        vocab_size=100,
    )
    dataset = MultimodalDataset(config)
    sample = dataset[0]
    
    assert 'image' not in sample
    assert 'audio' in sample
    assert 'input_ids' in sample
    
    return "image disabled, audio+text only"

test("Dataset 无图像", t_dataset_no_image)

# ============================================================
# 4. Collate Function 测试
# ============================================================
print("\n[4. Collate Function]")

def t_collate_fn():
    """批处理 collate"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=True,
        image_size=224,
        audio_max_length=16000,
        max_seq_len=128,
        vocab_size=100,
    )
    dataset = MultimodalDataset(config)
    
    # 取4个样本
    batch = [dataset[i] for i in range(4)]
    batched = multimodal_collate_fn(batch)
    
    assert batched['images'].shape == (4, 3, 224, 224), f"Wrong image batch: {batched['images'].shape}"
    assert batched['audios'].shape == (4, 16000), f"Wrong audio batch: {batched['audios'].shape}"
    assert batched['input_ids'].shape == (4, 128), f"Wrong text batch: {batched['input_ids'].shape}"
    assert batched['labels'].shape == (4, 128)
    assert batched['attention_mask'].shape == (4, 128)
    assert batched['audio_masks'].shape == (4, 16000)
    
    return f"batch: images={batched['images'].shape}, audios={batched['audios'].shape}, text={batched['input_ids'].shape}"

test("Collate 批处理", t_collate_fn)

# ============================================================
# 5. DataLoader 测试
# ============================================================
print("\n[5. DataLoader]")

def t_dataloader():
    """完整 DataLoader"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=True,
        image_size=224,
        audio_max_length=16000,
        max_seq_len=128,
        vocab_size=100,
    )
    
    dataloader = create_multimodal_dataloader(config, batch_size=4, shuffle=False)
    
    batch = next(iter(dataloader))
    
    assert 'images' in batch
    assert 'audios' in batch
    assert 'input_ids' in batch
    assert 'labels' in batch
    assert batch['images'].shape == (4, 3, 224, 224)
    assert batch['audios'].shape == (4, 16000)
    assert batch['input_ids'].shape == (4, 128)
    
    return f"batch_size=4, keys={list(batch.keys())}"

test("DataLoader", t_dataloader)

def t_dataloader_iteration():
    """DataLoader 多轮迭代"""
    config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=False,
        image_size=224,
        max_seq_len=128,
        vocab_size=100,
    )
    
    dataloader = create_multimodal_dataloader(config, batch_size=4, shuffle=False, num_workers=0)
    
    count = 0
    for batch in dataloader:
        count += 1
        assert batch['images'].shape[0] == 4
        assert torch.isfinite(batch['images']).all()
        assert torch.isfinite(batch['input_ids']).all()
    
    # 20 samples / batch_size 4 = 5 batches (drop_last=True)
    assert count == 5, f"Expected 5 batches, got {count}"
    return f"{count} batches iterated"

test("DataLoader 迭代", t_dataloader_iteration)

# ============================================================
# 6. 与模型集成测试
# ============================================================
print("\n[6. 与模型集成]")

def t_data_to_model():
    """数据 -> 模型的完整链条"""
    from tesm_ssm.models.config_tesm import TESMConfig
    from tesm_ssm.models.multimodal import TESMMultimodalModel, MultimodalConfig
    
    # 创建 DataLoader
    data_config = MultimodalDataConfig(
        data_path=str(data_file),
        image_root=str(img_dir),
        audio_root=str(aud_dir),
        use_image=True,
        use_audio=False,
        image_size=64,  # 小尺寸加速
        max_seq_len=32,
        vocab_size=50,
    )
    dataloader = create_multimodal_dataloader(data_config, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    
    # 创建模型
    mm_cfg = MultimodalConfig.from_tesm_config(
        TESMConfig(d_model=64, n_layer=1, d_intermediate=128, max_seq_len=64,
                   vocab_size=50, kernel_backend="torch"),
        vision_enabled=True,
        vision_patch_size=16,
        vision_num_tokens=4,
        vision_max_image_size=64,
        audio_enabled=False,
    )
    model = TESMMultimodalModel(mm_cfg)
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        out, _ = model(
            images=batch['images'],
            text_ids=batch['input_ids'][:, :16],  # 缩短
        )
    
    assert out.logits.shape[0] == 2  # batch_size
    assert torch.isfinite(out.logits).all()
    
    return f"data -> model: logits={out.logits.shape}"

test("数据到模型", t_data_to_model)

# ============================================================
# 清理
# ============================================================
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ============================================================
# 总结
# ============================================================
print()
print("=" * 70)
print("多模态数据处理测试总结")
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
