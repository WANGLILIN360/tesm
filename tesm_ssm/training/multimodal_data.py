"""TESM 多模态数据处理

支持图像(PIL)、音频(librosa)、文本的多模态数据加载和预处理。
设计为可选模块，纯文本用户不需要导入。

使用方式:
    from tesm_ssm.training.multimodal_data import (
        MultimodalDataset, MultimodalDataConfig, multimodal_collate_fn
    )
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# 多模态数据配置
# =============================================================================

@dataclass
class MultimodalDataConfig:
    """多模态数据配置"""
    
    # 数据路径
    data_path: str = ""  # jsonl 文件路径
    image_root: str = ""  # 图像根目录
    audio_root: str = ""  # 音频根目录
    
    # 模态开关
    use_image: bool = True
    use_audio: bool = False
    use_text: bool = True
    
    # 图像预处理配置
    image_size: int = 224  # 输入图像尺寸
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 音频预处理配置
    audio_sample_rate: int = 16000
    audio_max_length: int = 16000  # 最大音频长度 (1秒)
    audio_normalize: bool = True
    
    # 文本配置
    max_seq_len: int = 512
    vocab_size: int = 32000
    pad_token_id: int = 0
    
    # 训练配置
    image_augment: bool = True  # 是否使用数据增强
    random_crop: bool = True
    random_flip: bool = True


# =============================================================================
# 图像预处理
# =============================================================================

def load_image(path: str, config: MultimodalDataConfig) -> torch.Tensor:
    """加载并预处理图像
    
    Args:
        path: 图像文件路径
        config: 数据配置
        
    Returns:
        tensor: (3, H, W) 图像张量
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Image processing requires Pillow. "
            "Install: pip install Pillow"
        )
    
    img = Image.open(path).convert('RGB')
    
    # Resize
    img = img.resize((config.image_size, config.image_size), Image.BILINEAR)
    
    # To tensor (0-255 -> 0-1)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    
    # Normalize
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor


def image_augment(tensor: torch.Tensor, config: MultimodalDataConfig) -> torch.Tensor:
    """图像数据增强
    
    Args:
        tensor: (3, H, W) 图像张量
        config: 数据配置
        
    Returns:
        tensor: 增强后的图像
    """
    if not config.image_augment:
        return tensor
    
    # Random horizontal flip
    if config.random_flip and random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    
    # Random crop (simulated by random shift)
    if config.random_crop and random.random() > 0.5:
        _, H, W = tensor.shape
        shift_h = random.randint(-H // 20, H // 20)
        shift_w = random.randint(-W // 20, W // 20)
        tensor = torch.roll(tensor, shifts=(shift_h, shift_w), dims=(1, 2))
    
    # Color jitter (simplified)
    if random.random() > 0.8:
        jitter = torch.randn(3, 1, 1) * 0.05
        tensor = tensor + jitter
        tensor = torch.clamp(tensor, -3.0, 3.0)
    
    return tensor


# =============================================================================
# 音频预处理
# =============================================================================

def load_audio(path: str, config: MultimodalDataConfig) -> torch.Tensor:
    """加载并预处理音频
    
    Args:
        path: 音频文件路径
        config: 数据配置
        
    Returns:
        waveform: (T,) 音频波形 @ config.audio_sample_rate
    """
    # 尝试 librosa
    try:
        import librosa
        waveform, sr = librosa.load(path, sr=config.audio_sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).float()
    except ImportError:
        # Fallback: 使用 scipy (通常已安装)
        try:
            from scipy.io import wavfile
            sr, waveform = wavfile.read(path)
            waveform = torch.from_numpy(waveform.astype(np.float32))
            # 重采样到目标采样率 (简单最近邻)
            if sr != config.audio_sample_rate:
                ratio = config.audio_sample_rate / sr
                new_len = int(waveform.shape[0] * ratio)
                waveform = F.interpolate(
                    waveform.unsqueeze(0).unsqueeze(0),
                    size=new_len, mode='linear', align_corners=False
                ).squeeze()
        except ImportError:
            # 最终 fallback: 生成随机波形 (用于测试)
            logger.warning("No audio library found (librosa/scipy), using random waveform")
            duration = 1.0  # assume 1 second
            samples = int(config.audio_sample_rate * duration)
            waveform = torch.randn(samples) * 0.1
            return waveform
    
    # 归一化
    if config.audio_normalize and waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    
    return waveform


def pad_or_truncate_audio(waveform: torch.Tensor, max_length: int) -> torch.Tensor:
    """填充或截断音频到固定长度
    
    Args:
        waveform: (T,) 音频波形
        max_length: 目标长度
        
    Returns:
        waveform: (max_length,) 固定长度音频
    """
    if waveform.shape[0] > max_length:
        # 截断
        start = random.randint(0, waveform.shape[0] - max_length)
        waveform = waveform[start:start + max_length]
    elif waveform.shape[0] < max_length:
        # 填充 (重复)
        repeats = (max_length // waveform.shape[0]) + 1
        waveform = waveform.repeat(repeats)[:max_length]
    
    return waveform


# =============================================================================
# 多模态数据集
# =============================================================================

class MultimodalDataset(Dataset):
    """多模态数据集
    
    支持从 JSONL 文件加载多模态数据，格式:
    ```jsonl
    {"text": "描述这张图片", "image": "path/to/image.jpg", "audio": "path/to/audio.wav", "label": "这是一只猫"}
    {"text": "回答这个问题", "image": null, "audio": null, "label": "答案是42"}
    ```
    
    Args:
        config: MultimodalDataConfig 配置
        tokenizer: 文本 tokenizer (可选)
        transform: 自定义图像变换 (可选)
    """
    
    def __init__(
        self,
        config: MultimodalDataConfig,
        tokenizer: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        
        # 加载数据
        self.samples = self._load_data(config.data_path)
        
        logger.info(f"MultimodalDataset loaded: {len(self.samples)} samples")
        logger.info(f"  Image: {config.use_image}, Audio: {config.use_audio}, Text: {config.use_text}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """从 JSONL 文件加载数据"""
        samples = []
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:100]}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本
        
        Returns:
            dict with keys: image(optional), audio(optional), input_ids, labels, attention_mask
        """
        sample = self.samples[idx]
        result = {}
        
        # 加载图像 (如果启用)
        if self.config.use_image:
            if sample.get('image'):
                image_path = Path(self.config.image_root) / sample['image']
                if image_path.exists():
                    image_tensor = load_image(str(image_path), self.config)
                    if self.config.image_augment:
                        image_tensor = image_augment(image_tensor, self.config)
                    result['image'] = image_tensor
                else:
                    result['image'] = torch.zeros(3, self.config.image_size, self.config.image_size)
            else:
                # 样本没有图像，填充零张量
                result['image'] = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        # 加载音频 (如果启用)
        if self.config.use_audio:
            if sample.get('audio'):
                audio_path = Path(self.config.audio_root) / sample['audio']
                if audio_path.exists():
                    waveform = load_audio(str(audio_path), self.config)
                    waveform = pad_or_truncate_audio(waveform, self.config.audio_max_length)
                    result['audio'] = waveform
                else:
                    result['audio'] = torch.zeros(self.config.audio_max_length)
            else:
                # 样本没有音频，填充零张量
                result['audio'] = torch.zeros(self.config.audio_max_length)
        
        # 处理文本
        text = sample.get('text', '')
        label = sample.get('label', '')
        
        # 如果有 tokenizer，使用 tokenizer
        if self.tokenizer is not None:
            # 构建完整序列: text + label
            full_text = f"{text} {label}".strip()
            tokenized = self.tokenizer(full_text, max_length=self.config.max_seq_len, 
                                       padding='max_length', truncation=True, return_tensors='pt')
            result['input_ids'] = tokenized['input_ids'].squeeze(0)
            result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            
            # labels: 只对 label 部分计算损失
            label_ids = self.tokenizer(label, max_length=self.config.max_seq_len,
                                      padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze(0)
            result['labels'] = label_ids
        else:
            # 没有 tokenizer，使用随机 token ID（用于测试）
            text_ids = torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))
            result['input_ids'] = text_ids
            result['labels'] = text_ids.clone()
            result['attention_mask'] = torch.ones(self.config.max_seq_len)
        
        # 如果有自定义 transform，应用
        if self.transform is not None:
            result = self.transform(result)
        
        return result


# =============================================================================
# Collate Function (批处理)
# =============================================================================

def multimodal_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """多模态批处理 collate 函数
    
    处理不同长度的序列，进行 padding 和 masking。
    
    Args:
        batch: 样本列表
        
    Returns:
        dict: 批处理后的张量
    """
    result = {}
    
    # 图像: 直接堆叠 (固定尺寸)
    if 'image' in batch[0]:
        images = torch.stack([b['image'] for b in batch])
        result['images'] = images
    
    # 音频: 填充到 batch 内最大长度
    if 'audio' in batch[0]:
        audios = [b['audio'] for b in batch]
        max_audio_len = max(a.shape[0] for a in audios)
        padded_audios = []
        audio_masks = []
        for audio in audios:
            if audio.shape[0] < max_audio_len:
                pad_len = max_audio_len - audio.shape[0]
                padded = F.pad(audio, (0, pad_len), value=0.0)
                mask = torch.cat([torch.ones(audio.shape[0]), torch.zeros(pad_len)])
            else:
                padded = audio
                mask = torch.ones(max_audio_len)
            padded_audios.append(padded)
            audio_masks.append(mask)
        result['audios'] = torch.stack(padded_audios)
        result['audio_masks'] = torch.stack(audio_masks)
    
    # 文本: 已经是固定长度 (tokenizer padding)，直接堆叠
    if 'input_ids' in batch[0]:
        result['input_ids'] = torch.stack([b['input_ids'] for b in batch])
    
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([b['labels'] for b in batch])
    
    if 'attention_mask' in batch[0]:
        result['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
    
    return result


# =============================================================================
# 辅助函数
# =============================================================================

def create_multimodal_dataloader(
    config: MultimodalDataConfig,
    tokenizer: Optional[Callable] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """创建多模态 DataLoader
    
    Args:
        config: 数据配置
        tokenizer: 文本 tokenizer
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载 worker 数
        pin_memory: 是否 pin memory
        
    Returns:
        DataLoader
    """
    dataset = MultimodalDataset(config, tokenizer=tokenizer)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )


# =============================================================================
# 示例数据生成 (用于测试)
# =============================================================================

def create_dummy_multimodal_data(
    output_path: str,
    num_samples: int = 100,
    image_root: str = "dummy_images",
    audio_root: str = "dummy_audio",
):
    """创建示例多模态数据 (JSONL 格式)
    
    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
        image_root: 图像目录
        audio_root: 音频目录
    """
    import os
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # 随机决定是否包含图像/音频
            has_image = random.random() > 0.3
            has_audio = random.random() > 0.7
            
            sample = {
                "text": f"Sample question {i}: What do you see?",
                "image": f"image_{i % 10}.jpg" if has_image else None,
                "audio": f"audio_{i % 5}.wav" if has_audio else None,
                "label": f"This is the answer for sample {i}.",
            }
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Dummy multimodal data created: {output_path} ({num_samples} samples)")


def create_dummy_image(path: str, size: int = 224):
    """创建示例图像"""
    try:
        from PIL import Image
    except ImportError:
        return
    
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_dummy_audio(path: str, duration: float = 1.0, sr: int = 16000):
    """创建示例音频"""
    try:
        import soundfile as sf
    except ImportError:
        try:
            import scipy.io.wavfile as wavfile
        except ImportError:
            return
        
        samples = int(duration * sr)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(path, sr, waveform)
        return
    
    samples = int(duration * sr)
    waveform = np.random.randn(samples).astype(np.float32) * 0.1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)
