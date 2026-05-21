"""TESM 训练模块

提供完整的训练基础设施:
    - TrainingConfig: 训练配置
    - TESMTrainer: 训练器
    - TextDataset: 文本数据集
    - SimpleTokenizer/ByteTokenizer: 简单 tokenizer

快速开始:
    >>> from tesm_ssm import TESMConfig
    >>> from tesm_ssm.training import TrainingConfig, TESMTrainer
    >>> 
    >>> model_config = TESMConfig.small()
    >>> train_config = TrainingConfig(
    ...     model_config=model_config,
    ...     data_path='data/train.txt',
    ...     output_dir='outputs/experiment_1',
    ...     num_epochs=3,
    ...     batch_size=4,
    ...     learning_rate=1e-4,
    ... )
    >>> 
    >>> trainer = TESMTrainer(train_config)
    >>> trainer.train()
"""

from .config import TrainingConfig
from .trainer import TESMTrainer
from .dataset import TextDataset, StreamingTextDataset, SimpleTokenizer, ByteTokenizer, collate_fn

__all__ = [
    'TrainingConfig',
    'TESMTrainer',
    'TextDataset',
    'StreamingTextDataset',
    'SimpleTokenizer',
    'ByteTokenizer',
    'collate_fn',
]
