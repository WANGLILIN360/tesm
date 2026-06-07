"""多模态 Embedder 抽象基类

所有模态的 Embedder 必须实现此接口，确保与 TESM decoder 的兼容性。
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseEmbedder(nn.Module, ABC):
    """多模态 Embedder 抽象基类
    
    所有自定义 Embedder 需要继承此类并实现以下接口。
    
    Example:
        class MyEmbedder(BaseEmbedder):
            def __init__(self, d_model=768):
                super().__init__(d_model)
                self.proj = nn.Linear(100, d_model)
            
            def forward(self, x):
                return self.proj(x)
    """

    def __init__(self, d_model: int, **kwargs):
        """初始化
        
        Args:
            d_model: 输出维度，必须等于 TESM config.d_model
            **kwargs: 子类可扩展的其他参数
        """
        super().__init__()
        self._d_model = d_model

    @property
    def d_model(self) -> int:
        """输出维度"""
        return self._d_model

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将原始数据嵌入到 d_model 维度
        
        Args:
            x: 原始输入数据，形状由子类定义
            
        Returns:
            embeddings: (batch, num_tokens, d_model)
        """
        raise NotImplementedError

    def estimate_num_tokens(self, input_shape: Tuple[int, ...]) -> int:
        """估算给定输入会产生多少个 token
        
        Args:
            input_shape: 输入张量的形状（不含 batch 维度）
            
        Returns:
            token 数量
        """
        return -1  # 默认未知，子类可覆盖

    def __repr__(self):
        return f"{self.__class__.__name__}(d_model={self.d_model})"
