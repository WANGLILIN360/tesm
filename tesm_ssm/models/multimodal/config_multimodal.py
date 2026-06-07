"""多模态配置"""

from dataclasses import dataclass, field
from typing import Optional

from tesm_ssm.models.config_tesm import TESMConfig


@dataclass
class MultimodalConfig:
    """TESM 多模态配置
    
    组合了 TESM 基础配置和各模态 Embedder 配置。
    
    Example:
        >>> config = MultimodalConfig.from_tesm_config(TESMConfig.small())
        >>> config.vision_enabled = True
        >>> config.audio_enabled = True
    """

    # TESM 基础配置
    tesm: TESMConfig = field(default_factory=TESMConfig)

    # 模态开关
    vision_enabled: bool = True
    audio_enabled: bool = False
    video_enabled: bool = False

    # Vision Embedder 配置
    vision_patch_size: int = 48
    vision_num_tokens: int = 280
    vision_in_channels: int = 3
    vision_max_image_size: int = 1024
    vision_use_norm: bool = True

    # Audio Embedder 配置
    audio_sample_rate: int = 16000
    audio_frame_duration_ms: int = 40
    audio_use_norm: bool = True

    # 模态类型标记（学习able）
    use_modality_embedding: bool = True
    """是否为不同模态添加可学习的模态类型嵌入"""

    # 训练配置
    freeze_embedders: bool = False
    """是否冻结 Embedder 参数（只训练 decoder）"""

    @classmethod
    def from_tesm_config(cls, tesm_config: TESMConfig, **kwargs) -> "MultimodalConfig":
        """从现有 TESMConfig 创建多模态配置
        
        Args:
            tesm_config: TESM 基础配置
            **kwargs: 其他多模态参数
            
        Returns:
            MultimodalConfig
        """
        return cls(tesm=tesm_config, **kwargs)

    def to_dict(self):
        """序列化为字典"""
        return {
            "tesm": self.tesm.to_dict(),
            "vision_enabled": self.vision_enabled,
            "audio_enabled": self.audio_enabled,
            "video_enabled": self.video_enabled,
            "vision_patch_size": self.vision_patch_size,
            "vision_num_tokens": self.vision_num_tokens,
            "vision_in_channels": self.vision_in_channels,
            "vision_max_image_size": self.vision_max_image_size,
            "vision_use_norm": self.vision_use_norm,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_frame_duration_ms": self.audio_frame_duration_ms,
            "audio_use_norm": self.audio_use_norm,
            "use_modality_embedding": self.use_modality_embedding,
            "freeze_embedders": self.freeze_embedders,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultimodalConfig":
        """从字典反序列化"""
        tesm_dict = d.pop("tesm", {})
        return cls(tesm=TESMConfig.from_dict(tesm_dict), **d)
