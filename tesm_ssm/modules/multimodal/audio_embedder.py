"""Gemma4 风格的无编码器音频嵌入模块

~1M 参数，将原始音频波形直接投影到 token 空间
处理流程: Raw Audio (16kHz) -> 40ms Frames -> Linear Projection
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_embedder import BaseEmbedder


class AudioEmbedder(BaseEmbedder):
    """无编码器音频嵌入模块
    
    模拟 Gemma4 12B 的音频嵌入:
    - 将 16kHz 原始音频波形分割为 40ms 帧 (640 采样点)
    - 通过线性投影映射到 d_model 维度
    
    参数量: ~1M
    
    Args:
        d_model: 输出维度 (默认 3840, 匹配 Gemma4 12B)
        sample_rate: 采样率 (默认 16000 Hz)
        frame_duration_ms: 帧长度 (默认 40ms)
        use_norm: 是否使用 LayerNorm (默认 True)
    
    Example:
        >>> embedder = AudioEmbedder(d_model=768)
        >>> audio = torch.randn(2, 16000)  # (B, T) 1秒音频 @ 16kHz
        >>> embeds = embedder(audio)  # (2, 25, 768)  # 25 frames
    """

    def __init__(
        self,
        d_model: int = 3840,
        sample_rate: int = 16000,
        frame_duration_ms: int = 40,
        use_norm: bool = True,
    ):
        super().__init__(d_model)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.use_norm = use_norm

        # 帧大小
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # 线性投影: (B, N, frame_size) -> (B, N, d_model)
        self.frame_proj = nn.Linear(self.frame_size, d_model)

        # 可选的归一化
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.frame_proj.weight, std=0.02)
        nn.init.zeros_(self.frame_proj.bias)

    def _audio_to_frames(self, audio: torch.Tensor) -> torch.Tensor:
        """将音频波形分割为帧
        
        Args:
            audio: (B, T) 原始音频波形
            
        Returns:
            frames: (B, N, frame_size)
        """
        B, T = audio.shape
        frame_size = self.frame_size

        # 确保长度可被帧大小整除
        num_frames = T // frame_size
        if num_frames == 0:
            # 音频太短，零填充到至少一帧
            pad_len = frame_size - T
            audio = F.pad(audio, (0, pad_len))
            num_frames = 1

        # 截断到整数帧
        audio = audio[:, :num_frames * frame_size]

        # reshape: (B, N * frame_size) -> (B, N, frame_size)
        frames = audio.view(B, num_frames, frame_size)

        return frames

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """将音频嵌入到 d_model 维度
        
        Args:
            audio: (B, T) 原始音频波形, 16kHz
            
        Returns:
            embeds: (B, N, d_model) 其中 N = T // frame_size
        """
        # 分割为帧
        frames = self._audio_to_frames(audio)
        # frames: (B, N, frame_size)

        # 线性投影
        embeds = self.frame_proj(frames)  # (B, N, d_model)

        # 归一化
        if self.use_norm:
            embeds = self.norm(embeds)

        return embeds  # (B, N, d_model)

    def estimate_num_tokens(self, input_shape: Tuple[int, ...]) -> int:
        """估算输出 token 数
        
        Args:
            input_shape: (T,) 音频长度
            
        Returns:
            token 数量
        """
        T = input_shape[0]
        return max(1, T // self.frame_size)

    def __repr__(self):
        return (
            f"AudioEmbedder(d_model={self.d_model}, "
            f"sample_rate={self.sample_rate}, "
            f"frame_duration_ms={self.frame_duration_ms}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M)"
        )
