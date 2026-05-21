"""训练配置"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """TESM 训练配置
    
    示例:
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            data_path="data/train.txt",
            output_dir="outputs/experiment_1",
            num_epochs=10,
            batch_size=4,
            learning_rate=1e-4,
        )
    """
    # 模型配置
    model_config: Optional[object] = None
    
    # 数据配置
    data_path: Optional[str] = None  # 训练数据路径 (.txt 或 .jsonl)
    eval_data_path: Optional[str] = None  # 验证数据路径
    max_seq_len: int = 2048  # 最大序列长度
    vocab_size: Optional[int] = None  # 词表大小（默认从tokenizer获取）
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: Optional[int] = None  # 覆盖 num_epochs
    
    # 优化器配置
    optimizer: str = "adamw"  # adamw, adam, sgd
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # linear, cosine, constant, polynomial
    min_lr_ratio: float = 0.1  # 最小学习率比例
    
    # 系统配置
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # 模型类型选择
    model_type: str = "auto"  # auto, siso, mimo - 自动从 model_config.use_mimo 推断
    
    # 加速器/Kernel 后端选择
    accelerator: str = "auto"  # auto, torch, cuda, triton, tilelang
    
    # 混合精度
    use_amp: bool = True  # 自动混合精度
    amp_dtype: str = "bf16"  # fp16, bf16
    
    # 分布式训练
    local_rank: int = -1  # -1 表示非分布式
    world_size: int = 1
    
    # 检查点与日志
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10  # 每N步记录日志
    eval_interval: int = 500  # 每N步验证
    save_interval: int = 1000  # 每N步保存检查点
    keep_last_n_checkpoints: int = 3  # 保留最近N个检查点
    
    # 日志配置
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # 评估配置
    eval_steps: Optional[int] = None  # 验证步数（None表示全部）
    eval_accumulation_steps: int = 1
    
    # 早停
    early_stopping_patience: Optional[int] = None  # None 表示禁用
    early_stopping_threshold: float = 0.001
    
    # 恢复训练
    resume_from_checkpoint: Optional[str] = None
    
    # 梯度检查点
    gradient_checkpointing: bool = False
    
    # 额外配置
    dataloader_drop_last: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        """验证配置"""
        if self.model_config is None:
            raise ValueError("必须提供 model_config")
        
        # 设置最大序列长度
        if hasattr(self.model_config, 'max_seq_len'):
            self.max_seq_len = min(self.max_seq_len, self.model_config.max_seq_len)
        
        # 设置词表大小
        if self.vocab_size is None and hasattr(self.model_config, 'vocab_size'):
            self.vocab_size = self.model_config.vocab_size
        
        # 验证设备选择
        valid_devices = ["auto", "cpu", "cuda", "mps"] + [f"cuda:{i}" for i in range(8)]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"无效的设备选择: {self.device}，可选: {valid_devices}")
        
        # 验证模型类型
        if self.model_type not in ["auto", "siso", "mimo"]:
            raise ValueError(f"无效的模型类型: {self.model_type}，可选: auto, siso, mimo")
        
        # 验证加速器选择
        valid_accelerators = ["auto", "torch", "cuda", "triton", "tilelang"]
        if self.accelerator not in valid_accelerators:
            raise ValueError(f"无效的加速器选择: {self.accelerator}，可选: {valid_accelerators}")
        
        # 验证设备与加速器兼容性
        # cuda/triton/tilelang 加速器需要 CUDA 设备（tensor.is_cuda 检查）
        _gpu_only_accelerators = {"cuda", "triton", "tilelang"}
        _cpu_devices = {"cpu"}
        if self.accelerator in _gpu_only_accelerators and self.device in _cpu_devices:
            raise ValueError(
                f"加速器 '{self.accelerator}' 需要 CUDA 设备，但当前设备为 '{self.device}'。"
                f"请将 device 改为 'cuda' 或 'auto'，或将 accelerator 改为 'torch' 或 'auto'。"
            )
    
    def to_dict(self):
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建"""
        from tesm_ssm import TESMConfig
        
        config_data = data.copy()
        if 'model_config' in config_data and isinstance(config_data['model_config'], dict):
            config_data['model_config'] = TESMConfig.from_dict(config_data['model_config'])
        
        # 过滤掉不支持的字段
        valid_keys = {k for k in cls.__dataclass_fields__.keys()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}
        
        return cls(**filtered_data)
