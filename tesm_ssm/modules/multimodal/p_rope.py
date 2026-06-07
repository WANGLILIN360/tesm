"""p-RoPE (Proportional RoPE) - 低频率剪枝的位置编码

Gemma4 12B 使用 p-RoPE 替代标准 RoPE，通过丢弃低频率的旋转维度，
在保持长上下文能力的同时减少计算量。

参考: Gemma4 技术文档 "Proportional RoPE on global layers for long-context efficiency"
"""

import torch
import torch.nn as nn
import math


class PRoPE(nn.Module):
    """Proportional RoPE - 低频率剪枝的旋转位置编码

    标准 RoPE 使用等间距频率:
        theta_i = base^(-2i/d)

    p-RoPE 只保留高频部分（i 较小的维度），丢弃低频部分:
        active_dims = int(dim * (1 - prune_ratio))
        theta_i = base^(-2i/active_dims)  for i in [0, active_dims)
        其余维度设为 0（不旋转）

    这样在长上下文时减少低频分量的计算，提高效率。

    Args:
        dim: 头维度
        max_seq_len: 最大序列长度
        base: RoPE 基数 (默认 10000)
        prune_ratio: 剪枝比例 (默认 0.5，即丢弃50%低频)
        device: 设备

    Example:
        >>> prope = PRoPE(dim=64, max_seq_len=2048, prune_ratio=0.5)
        >>> x = torch.randn(1, 8, 64)  # (batch, seq_len, dim)
        >>> x_rotated = prope(x, positions=torch.arange(8))
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        prune_ratio: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.prune_ratio = prune_ratio

        # 活跃维度 = 只保留高频部分
        self.active_dims = int(dim * (1 - prune_ratio))
        # 确保是偶数
        self.active_dims = self.active_dims // 2 * 2

        # 预计算频率 (只计算活跃维度)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.active_dims, 2).float() / self.active_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算位置编码
        self._precompute(max_seq_len)

    def _precompute(self, max_seq_len: int):
        """预计算 cos/sin 缓存"""
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (max_seq_len, active_dims/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, active_dims)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转张量的一半维度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        seq_len: int = None,
    ) -> torch.Tensor:
        """应用 p-RoPE

        Args:
            x: (batch, seq_len, dim) 或 (batch, n_heads, seq_len, dim)
            positions: (seq_len,) 位置索引，None 时使用 [0, 1, ..., seq_len-1]
            seq_len: 序列长度，None 时从 x 推断

        Returns:
            x_rotated: 与 x 相同形状
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        if positions is None:
            positions = torch.arange(seq_len, device=x.device)

        # 获取 cos/sin (只应用到活跃维度)
        cos = self.cos_cached[positions]  # (seq_len, active_dims)
        sin = self.sin_cached[positions]

        # 扩展维度以匹配输入
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # 应用旋转: 只旋转活跃维度，其余保持不变
        if self.active_dims < self.dim:
            # x = [x_active, x_pruned]
            x_active = x[..., :self.active_dims]
            x_pruned = x[..., self.active_dims:]

            # 只旋转活跃部分
            x_active_rotated = x_active * cos + self._rotate_half(x_active) * sin

            # 拼接
            x_out = torch.cat([x_active_rotated, x_pruned], dim=-1)
        else:
            x_out = x * cos + self._rotate_half(x) * sin

        return x_out

    def __repr__(self):
        return (
            f"PRoPE(dim={self.dim}, active_dims={self.active_dims}, "
            f"prune_ratio={self.prune_ratio}, base={self.base})"
        )
