"""TESM 多模态模块 - 可选的多模态输入处理

使用方式:
    from tesm_ssm.modules.multimodal import VisionEmbedder, AudioEmbedder, BaseEmbedder
"""

from .base_embedder import BaseEmbedder
from .vision_embedder import VisionEmbedder
from .audio_embedder import AudioEmbedder

__all__ = ["BaseEmbedder", "VisionEmbedder", "AudioEmbedder"]
