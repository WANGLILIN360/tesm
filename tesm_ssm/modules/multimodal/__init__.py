"""TESM 多模态模块 - 可选的多模态输入处理

使用方式:
    from tesm_ssm.modules.multimodal import VisionEmbedder, AudioEmbedder, BaseEmbedder
    from tesm_ssm.modules.multimodal import VisionEmbedderV2, AudioEmbedderV2, PRoPE
"""

from .base_embedder import BaseEmbedder
from .vision_embedder import VisionEmbedder
from .audio_embedder import AudioEmbedder
from .vision_embedder_v2 import VisionEmbedderV2
from .audio_embedder_v2 import AudioEmbedderV2
from .p_rope import PRoPE

__all__ = [
    "BaseEmbedder", "VisionEmbedder", "AudioEmbedder",
    "VisionEmbedderV2", "AudioEmbedderV2", "PRoPE",
]
