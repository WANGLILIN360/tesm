"""Gemma4 风格的无编码器音频嵌入模块 V2 - 精度对齐版本

与 Gemma4 12B 的 AudioEmbedder 完全对齐:
- mel spectrogram 提取 (20ms窗口/10ms帧移)
- 2x CNN 下采样 (kernel=3, stride=2)
- 线性投影到 d_model

处理流程: raw audio -> mel frames -> Conv2D x2 -> Linear -> LN
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_embedder import BaseEmbedder


class AudioEmbedderV2(BaseEmbedder):
    """Gemma4-对齐的无编码器音频嵌入模块

    完全匹配 Gemma4 12B 的音频处理:
    - 模拟 mel 帧提取 (20ms窗口, 10ms帧移, 半因果padding)
    - 两个 2D 卷积下采样层 (kernel=3, stride=2)
    - 线性投影到 d_model

    参考: vLLM PR #44429 _compute_audio_num_tokens

    Args:
        d_model: 输出维度
        sample_rate: 采样率 (默认 16000)
        frame_length_ms: 帧长度 (默认 20ms)
        hop_length_ms: 帧移 (默认 10ms)
        audio_seq_length: 最大输出序列长度
        n_mels: mel filterbank 数量 (默认 80)

    Example:
        >>> embedder = AudioEmbedderV2(d_model=3840)
        >>> audio = torch.randn(2, 16000)  # 1秒 @ 16kHz
        >>> embeds = embedder(audio)  # (2, ~25, 3840)
    """

    def __init__(
        self,
        d_model: int = 3840,
        sample_rate: int = 16000,
        frame_length_ms: float = 20.0,
        hop_length_ms: float = 10.0,
        audio_seq_length: int = 256,
        n_mels: int = 80,
    ):
        super().__init__(d_model)
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.audio_seq_length = audio_seq_length
        self.n_mels = n_mels

        # 帧参数
        self.frame_length = int(round(sample_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sample_rate * hop_length_ms / 1000.0))

        # Mel 滤波器组 (可学习参数模拟，正权重确保非负输出)
        self.mel_scale = nn.Parameter(
            torch.rand(n_mels, self.frame_length // 2 + 1) * 0.1 + 0.01
        )

        # 两个 2D 卷积下采样层
        # 输入: (B, 1, n_frames, n_mels)
        # 输出: (B, n_mels, t, n_mels) 经过两层 stride=2
        self.conv1 = nn.Conv2d(1, n_mels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(n_mels, n_mels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        # 投影到 d_model
        self.proj = nn.Linear(n_mels, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def _extract_mel_frames(self, audio: torch.Tensor) -> torch.Tensor:
        """提取 mel 帧

        Args:
            audio: (B, T) 原始音频波形

        Returns:
            mel_spec: (B, n_frames, n_mels)
        """
        B, T = audio.shape
        fl = self.frame_length
        hl = self.hop_length

        # 半因果 padding
        pad_left = fl // 2
        audio = F.pad(audio, (pad_left, 0))

        # unfold
        n_frames = max(1, (audio.shape[1] - fl) // hl + 1)
        audio = audio[:, :n_frames * hl + fl - 1]
        frames = audio.unfold(1, fl, hl)  # (B, n_frames, fl)

        # STFT
        window = torch.hann_window(fl, device=audio.device)
        stft = torch.fft.rfft(frames * window, dim=-1)
        magnitude = stft.abs()  # (B, n_frames, fl//2+1)

        # Mel filterbank
        mel_spec = torch.matmul(magnitude, self.mel_scale.t())
        mel_spec = mel_spec.clamp(min=1e-6)
        mel_spec = torch.log(mel_spec)

        return mel_spec

    def _cnn_downsample(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """CNN 下采样

        Args:
            mel_spec: (B, n_frames, n_mels)

        Returns:
            features: (B, n_tokens, n_mels)
        """
        B, T, M = mel_spec.shape
        
        # (B, T, M) -> (B, 1, T, M)
        x = mel_spec.unsqueeze(1)

        # Conv1 + ReLU: (B, 1, T, M) -> (B, n_mels, T//2, M)
        x = F.relu(self.conv1(x))
        
        # Conv2 + ReLU: (B, n_mels, T//2, M) -> (B, n_mels, T//4, M)
        x = F.relu(self.conv2(x))
        
        # (B, n_mels, T//4, M) -> (B, T//4, n_mels, M) -> mean over last dim -> (B, T//4, n_mels)
        x = x.permute(0, 2, 1, 3)  # (B, T//4, n_mels, M)
        x = x.mean(dim=-1)  # (B, T//4, n_mels)

        return x

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """将音频嵌入到 d_model 维度

        处理流程: raw audio -> mel frames -> Conv2D x2 -> Linear -> LN

        Args:
            audio: (B, T) 原始音频波形, 16kHz

        Returns:
            embeds: (B, N, d_model)
        """
        # Step 1: 提取 mel 帧
        mel_spec = self._extract_mel_frames(audio)  # (B, n_frames, n_mels)

        # Step 2: CNN 下采样
        features = self._cnn_downsample(mel_spec)  # (B, n_tokens, n_mels)

        # Step 3: 线性投影
        embeds = self.proj(features)  # (B, n_tokens, d_model)

        # Step 4: 归一化
        embeds = self.norm(embeds)

        return embeds

    def estimate_num_tokens(self, input_shape: Tuple[int, ...]) -> int:
        """估算输出 token 数

        参考 Gemma4 的 _compute_audio_num_tokens
        """
        T = input_shape[0]
        fl = self.frame_length
        hl = self.hop_length
        pad_left = fl // 2
        padded = T + pad_left
        n_frames = max(0, (padded - fl) // hl + 1)
        if n_frames <= 0:
            return 0
        t = n_frames
        for _ in range(2):
            t = (t + 2 - 3) // 2 + 1
        return min(t, self.audio_seq_length)

    def __repr__(self):
        return (
            f"AudioEmbedderV2(d_model={self.d_model}, "
            f"sample_rate={self.sample_rate}, "
            f"frame={self.frame_length_ms}ms/{self.hop_length_ms}ms, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M)"
        )
