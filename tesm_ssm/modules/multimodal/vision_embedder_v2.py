"""Gemma4 风格的无编码器视觉嵌入模块 V2 - 精度对齐版本

与 Gemma4 12B 的 VisionEmbedder 完全对齐:
- 3层 LayerNorm (patch前/投影后/最终)
- Factorized 2D 位置编码 (nn.Parameter, X/Y 独立)
- 无效位置掩码处理

处理流程: raw patches -> LN1 -> Dense -> LN2 -> +factorized_posemb -> LN3
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_embedder import BaseEmbedder


class VisionEmbedderV2(BaseEmbedder):
    """Gemma4-对齐的无编码器视觉嵌入模块

    完全匹配 Gemma4 12B 的架构:
    - patch_ln1: 原始 patches 归一化
    - patch_dense: 密集投影 (ColumnParallelLinear 等价)
    - patch_ln2: 投影后归一化
    - pos_embedding: factorized 2D 位置编码 (X/Y 独立)
    - pos_norm: 最终归一化

    Args:
        d_model: 输出维度
        patch_size: patch 大小 (默认 48, 匹配 Gemma4)
        num_output_tokens: 输出 token 数 (默认 280, 匹配 Gemma4)
        in_channels: 输入通道数 (默认 3, RGB)
        max_image_size: 最大图像尺寸 (默认 1024)
        posemb_size: 位置编码尺寸 (默认 128)
    
    Example:
        >>> embedder = VisionEmbedderV2(d_model=3840)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> embeds = embedder(images)  # (2, 280, 3840)
    """

    def __init__(
        self,
        d_model: int = 3840,
        patch_size: int = 48,
        num_output_tokens: int = 280,
        in_channels: int = 3,
        max_image_size: int = 1024,
        posemb_size: int = 128,
    ):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.num_output_tokens = num_output_tokens
        self.in_channels = in_channels
        self.max_image_size = max_image_size
        self.posemb_size = posemb_size

        patch_dim = in_channels * patch_size * patch_size

        # Step 1: 原始 patches 归一化
        self.patch_ln1 = nn.LayerNorm(patch_dim)

        # Step 2: 密集投影 (使用标准 Linear，ColumnParallelLinear 是分布式优化)
        self.patch_dense = nn.Linear(patch_dim, d_model, bias=True)

        # Step 3: 投影后归一化
        self.patch_ln2 = nn.LayerNorm(d_model)

        # Step 4: Factorized 2D 位置编码
        # Gemma4 使用 (posemb_size, 2, d_model) 的 nn.Parameter
        self.pos_embedding = nn.Parameter(
            torch.zeros(posemb_size, 2, d_model)
        )

        # Step 5: 最终归一化
        self.pos_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.patch_dense.weight, std=0.02)
        nn.init.zeros_(self.patch_dense.bias)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def _image_to_patches(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将图像分割为 patches 并生成坐标

        Args:
            images: (B, C, H, W)

        Returns:
            patches: (B, N, patch_dim)
            positions_xy: (B, N, 2) 坐标 (y, x)，-1 表示 padding
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # 确保尺寸可被 patch_size 整除
        if H % p != 0 or W % p != 0:
            new_H = math.ceil(H / p) * p
            new_W = math.ceil(W / p) * p
            images = F.interpolate(
                images, size=(new_H, new_W), mode='bilinear', align_corners=False
            )
            H, W = new_H, new_W

        # unfold: (B, C, H, W) -> (B, C, H//p, W//p, p, p)
        patches = images.unfold(2, p, p).unfold(3, p, p)
        GH, GW = patches.shape[2], patches.shape[3]
        N = GH * GW

        # 重排: (B, GH, GW, p, p, C) -> (B, N, patch_dim)
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
        patches = patches.view(B, N, -1)

        # 生成坐标 (y, x)
        y_coords = torch.arange(GH, device=images.device)
        x_coords = torch.arange(GW, device=images.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # (N, 2)
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        return patches, positions

    def _factorized_posemb(self, positions_xy: torch.Tensor) -> torch.Tensor:
        """Factorized 2D 位置编码

        对每个坐标轴独立嵌入后相加，无效位置（-1）会被掩码掉。
        完全匹配 Gemma4 的实现。

        Args:
            positions_xy: (B, N, 2) 坐标 (y, x)

        Returns:
            pos_embs: (B, N, d_model)
        """
        clamped_pos = positions_xy.clamp(min=0).long()
        valid_mask = positions_xy != -1  # (B, N, 2)

        pos_embs = torch.zeros(
            *positions_xy.shape[:-1], self.pos_embedding.shape[-1],
            device=positions_xy.device, dtype=self.pos_embedding.dtype,
        )

        # 对 X 和 Y 两个轴独立嵌入
        for i in range(2):
            axis_pe = self.pos_embedding[:, i, :][clamped_pos[..., i]]
            mask = valid_mask[..., i].unsqueeze(-1).to(axis_pe.dtype)
            pos_embs = pos_embs + (axis_pe * mask)

        return pos_embs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """将图像嵌入到 d_model 维度

        处理流程: raw patches -> LN1 -> Dense -> LN2 -> +factorized_posemb -> LN3

        Args:
            images: (B, C, H, W) 图像张量

        Returns:
            embeds: (B, num_output_tokens, d_model)
        """
        # Step 1: 图像 -> patches + 坐标
        patches, positions = self._image_to_patches(images)

        # Step 2: LN1 (原始 patches 归一化)
        patches = self.patch_ln1(patches.to(self.pos_embedding.dtype))

        # Step 3: Dense (密集投影)
        embeds = self.patch_dense(patches)

        # Step 4: LN2 (投影后归一化)
        embeds = self.patch_ln2(embeds)

        # Step 5: +Factorized Positional Embedding
        pos_embs = self._factorized_posemb(positions)
        embeds = embeds + pos_embs

        # Step 6: LN3 (最终归一化)
        embeds = self.pos_norm(embeds)

        # Step 7: 自适应池化到固定 token 数
        if embeds.shape[1] != self.num_output_tokens:
            embeds = embeds.permute(0, 2, 1)  # (B, D, N)
            embeds = F.adaptive_avg_pool1d(embeds, self.num_output_tokens)
            embeds = embeds.permute(0, 2, 1)  # (B, num_tokens, D)

        return embeds

    def estimate_num_tokens(self, input_shape: Tuple[int, ...]) -> int:
        """估算输出 token 数"""
        C, H, W = input_shape
        p = self.patch_size
        gh = math.ceil(H / p)
        gw = math.ceil(W / p)
        return min(self.num_output_tokens, gh * gw)

    def __repr__(self):
        return (
            f"VisionEmbedderV2(d_model={self.d_model}, patch_size={self.patch_size}, "
            f"num_output_tokens={self.num_output_tokens}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M)"
        )
