from typing import Optional

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.norm2 = norm_cls(dim)
        self.mlp = mlp_cls(dim)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, prev_state=None, **mixer_kwargs):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        # mixer now returns (y, final_state)
        mixer_out = self.mixer(hidden_states, inference_params=inference_params, prev_state=prev_state, **mixer_kwargs)
        if isinstance(mixer_out, tuple):
            hidden_states, final_state = mixer_out
        else:
            hidden_states = mixer_out
            final_state = None
        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, final_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self.mixer, "allocate_inference_cache"):
            return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None
