"""Gemma4 风格的无编码器视觉嵌入模块

~35M 参数，替代传统 550M Vision Encoder
处理流程: Image -> 48x48 Patches -> Linear Proj -> Pos Emb -> Pool
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_embedder import BaseEmbedder


class VisionEmbedder(BaseEmbedder):
    """无编码器视觉嵌入模块
    
    模拟 Gemma4 12B 的轻量级视觉嵌入:
    - 将图像分割为 48x48 patches
    - 通过单层线性投影映射到 d_model 维度
    - 添加因子化坐标位置编码
    - 自适应池化到固定 token 数
    
    参数量: ~35M (vs 传统 Vision Encoder 550M)
    
    Args:
        d_model: 输出维度 (默认 3840, 匹配 Gemma4 12B)
        patch_size: patch 大小 (默认 48)
        num_output_tokens: 输出 token 数 (默认 280)
        in_channels: 输入通道数 (默认 3, RGB)
        max_image_size: 最大图像尺寸 (默认 1024)
        use_norm: 是否使用 LayerNorm (默认 True)
    
    Example:
        >>> embedder = VisionEmbedder(d_model=768)
        >>> images = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
        >>> embeds = embedder(images)  # (2, 280, 768)
    """

    def __init__(
        self,
        d_model: int = 3840,
        patch_size: int = 48,
        num_output_tokens: int = 280,
        in_channels: int = 3,
        max_image_size: int = 1024,
        use_norm: bool = True,
    ):
        super().__init__(d_model)
        self.patch_size = patch_size
        self.num_output_tokens = num_output_tokens
        self.in_channels = in_channels
        self.max_image_size = max_image_size
        self.use_norm = use_norm

        patch_dim = in_channels * patch_size * patch_size

        # Patch 投影: (B, N, patch_dim) -> (B, N, d_model)
        self.patch_proj = nn.Linear(patch_dim, d_model)

        # 因子化坐标位置编码
        # 假设最大图像尺寸 1024, grid = 1024 / 48 = ~21
        max_grid = math.ceil(max_image_size / patch_size)
        self.pos_x = nn.Embedding(max_grid, d_model // 2)
        self.pos_y = nn.Embedding(max_grid, d_model // 2)

        # 可选的归一化
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.normal_(self.pos_x.weight, std=0.02)
        nn.init.normal_(self.pos_y.weight, std=0.02)

    def _image_to_patches(self, images: torch.Tensor) -> torch.Tensor:
        """将图像分割为 patches
        
        Args:
            images: (B, C, H, W)
            
        Returns:
            patches: (B, N, patch_dim) 其中 N = (H//patch)*(W//patch)
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # 确保尺寸可被 patch_size 整除
        if H % p != 0 or W % p != 0:
            # 自适应 resize 到最近的 patch 倍数
            new_H = math.ceil(H / p) * p
            new_W = math.ceil(W / p) * p
            images = F.interpolate(
                images, size=(new_H, new_W), mode='bilinear', align_corners=False
            )
            H, W = new_H, new_W

        # unfold: (B, C, H, W) -> (B, C, H//p, W//p, p, p)
        patches = images.unfold(2, p, p).unfold(3, p, p)
        # (B, C, H//p, W//p, p, p)

        # 重排: (B, H//p, W//p, C, p, p) -> (B, N, patch_dim)
        B, C, GH, GW, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
        patches = patches.view(B, GH * GW, C * p * p)

        return patches, GH, GW

    def _add_positional_embedding(
        self, embeds: torch.Tensor, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        """添加因子化坐标位置编码
        
        Args:
            embeds: (B, N, d_model)
            grid_h: 高度方向的 grid 数
            grid_w: 宽度方向的 grid 数
            
        Returns:
            embeds: (B, N, d_model) 添加了位置编码
        """
        B, N, D = embeds.shape

        # 生成坐标网格
        y_coords = torch.arange(grid_h, device=device)
        x_coords = torch.arange(grid_w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # flatten
        yy = yy.flatten()  # (N,)
        xx = xx.flatten()  # (N,)

        # 获取位置编码
        pos_y_emb = self.pos_y(yy)  # (N, d_model//2)
        pos_x_emb = self.pos_x(xx)  # (N, d_model//2)

        # 拼接
        pos_emb = torch.cat([pos_y_emb, pos_x_emb], dim=-1)  # (N, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)

        return embeds + pos_emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """将图像嵌入到 d_model 维度
        
        Args:
            images: (B, C, H, W) 图像张量, C=3 (RGB)
            
        Returns:
            embeds: (B, num_output_tokens, d_model)
        """
        # Patch 分割
        patches, grid_h, grid_w = self._image_to_patches(images)
        # patches: (B, N, patch_dim)

        # 线性投影
        embeds = self.patch_proj(patches)  # (B, N, d_model)

        # 添加位置编码
        embeds = self._add_positional_embedding(embeds, grid_h, grid_w, images.device)

        # 归一化
        if self.use_norm:
            embeds = self.norm(embeds)

        # 自适应池化到固定 token 数
        if embeds.shape[1] != self.num_output_tokens:
            # (B, N, D) -> (B, D, N) -> pool -> (B, D, num_output_tokens) -> (B, num_output_tokens, D)
            embeds = embeds.permute(0, 2, 1)
            embeds = F.adaptive_avg_pool1d(embeds, self.num_output_tokens)
            embeds = embeds.permute(0, 2, 1)

        return embeds  # (B, num_output_tokens, d_model)

    def estimate_num_tokens(self, input_shape: Tuple[int, ...]) -> int:
        """估算输出 token 数
        
        Args:
            input_shape: (C, H, W)
            
        Returns:
            token 数量
        """
        C, H, W = input_shape
        p = self.patch_size
        gh = math.ceil(H / p)
        gw = math.ceil(W / p)
        # 池化后固定为 num_output_tokens
        return min(self.num_output_tokens, gh * gw)

    def __repr__(self):
        return (
            f"VisionEmbedder(d_model={self.d_model}, patch_size={self.patch_size}, "
            f"num_output_tokens={self.num_output_tokens}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M)"
        )
