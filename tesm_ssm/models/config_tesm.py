"""
TESM 模型配置

详细参数说明请参考: tesm_ssm/docs/config_guide.md

关键参数快速参考:
- decay_init_bias: 状态衰减偏置，根据序列长度选择
    * max_seq_len <= 256:  使用 0.0 或 -1.0 (状态保留 50% 或 27%)
    * max_seq_len <= 1024: 使用 1.0 (状态保留 73%)
    * max_seq_len > 4096:  使用 3.0+ (状态保留 95%+)
- entanglement_threshold: 纠缠激活阈值，建议 0.05-0.10
- annealing_steps: 退火步数，设为总训练步数的 10-20%
"""
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class TESMConfig:
    # ==================== 模型架构参数 ====================
    d_model: int = 768
    """隐藏层维度，影响模型容量。推荐: 小模型256-512，中模型512-768，大模型768+"""
    
    n_layer: int = 24
    """Transformer 层数。推荐: 小模型6-12，中模型12-24，大模型24-32"""
    
    d_intermediate: int = 2048
    """FFN 中间层维度，通常为 d_model * 4"""
    
    vocab_size: int = 151936
    """词表大小"""
    
    max_seq_len: int = 2048
    """最大序列长度。注意: 需配合 decay_init_bias 调整！
    
    关键配置:
    - max_seq_len <= 256:  decay_init_bias = 0.0 或 -1.0
    - max_seq_len <= 1024: decay_init_bias = 1.0
    - max_seq_len > 4096:  decay_init_bias = 3.0+
    """
    
    # ==================== Token 配置 ====================
    pad_token_id: int = 0
    """Padding token ID"""
    
    eos_token_id: Optional[int] = None
    """End-of-sequence token ID，用于生成时停止"""
    
    label_ignore_index: int = -100
    """损失计算时忽略的 label 值"""
    
    # ==================== 训练配置 ====================
    gradient_checkpointing: bool = False
    """是否启用梯度检查点以节省显存"""
    
    tie_embeddings: bool = True
    """是否共享输入/输出 embedding"""
    
    dropout: float = 0.0
    """Dropout 比例"""
    
    # ==================== 归一化配置 ====================
    rms_norm: bool = True
    """是否使用 RMSNorm (否则 LayerNorm)"""
    
    residual_in_fp32: bool = True
    """残差连接是否使用 float32"""
    
    norm_epsilon: float = 1e-5
    """归一化层 epsilon"""
    
    initializer_range: float = 0.02
    """权重初始化范围"""
    
    rescale_prenorm_residual: bool = True
    """是否重缩放 pre-norm 残差"""
    
    # ==================== BitLinear 配置 ====================
    bit_eps: float = 1e-5
    """BitLinear 量化 epsilon"""
    
    bit_threshold: float = 0.5
    """BitLinear 量化阈值"""
    
    # ==================== 词表抑制配置 ====================
    vocab_suppression: bool = False
    """是否启用词表抑制：对未激活token施加负向偏置，增强语义连贯性"""
    
    suppression_bias: float = -10.0
    """抑制偏置值：未激活token的logit减去此值。推荐-5.0到-15.0"""
    
    # ==================== 语义相关激活配置 ====================
    semantic_activation: bool = True
    """是否启用语义相关激活：训练时学习token共现关系，推理时自动激活相关token，解决泛化问题"""
    
    semantic_activation_strength: float = 0.5
    """语义相关token的激活强度：0.0-1.0，推荐0.3-0.7"""
    
    semantic_activation_threshold: float = 0.1
    """语义相关激活阈值：保持低值让模型自己学习区分上下文"""
    
    # ==================== MIMO 多头配置 ====================
    use_mimo: bool = False
    """是否使用 MIMO 多头模式。启用后参数量会增加，但表达能力更强"""
    
    n_heads: int = 4
    """MIMO 头数，仅当 use_mimo=True 时生效"""
    
    # ==================== SSM 状态空间配置 ====================
    d_state: int = 256
    """状态空间维度，影响长程记忆能力。推荐: 短序列256，长序列512+"""
    
    expand: int = 2
    """状态扩展因子"""
    
    # ==================== 纠缠机制配置 ====================
    ent_rank: int = 64
    """纠缠查询/键的秩，影响纠缠表达能力。推荐: d_model/8 到 d_model/4"""
    
    entanglement_scale: float = 0.2
    """纠缠混合权重。推荐: 0.1-0.3"""
    
    entanglement_threshold: float = 0.1
    """三值纠缠激活阈值: 0.05高激活, 0.08中等, 0.10低激活"""
    
    entanglement_init: float = 0.3
    """纠缠矩阵初始化标准差"""
    
    entanglement_window: int = 16
    """局部纠缠窗口大小。0=全局纠缠"""
    
    entanglement_block_size: int = 256
    """纠缠块大小"""
    
    state_scan_chunk_size: int = 16
    """状态扫描块大小"""
    
    # ==================== 内核后端配置 ====================
    use_triton_kernels: bool = True
    """是否使用 Triton kernel 加速"""
    
    kernel_backend: str = "auto"
    """Kernel 后端: auto/cuda/triton/tilelang/torch"""
    
    kernel_mode: str = "fast"
    """Kernel 模式: fast/precise"""
    
    # ==================== 状态衰减偏置 ====================
    decay_init_bias: float = 3.0
    """状态衰减偏置，控制历史状态保留率
    sigmoid(bias): -3=5%, -1=27%, 0=50%, 1=73%, 2=88%, 3=95%, 6=99.7%
    推荐: 短序列(<=256)用0.0, 长序列(>1024)用2.0-3.0
    常见错误: 对短序列使用 decay_init_bias=3.0 会导致位置区分困难"""
    
    # ==================== 位置编码配置 ====================
    rope_base: float = 10000.0
    """RoPE 基础频率"""
    
    global_rel_pos_dim: int = 64
    """全局相对位置编码维度（仅全局纠缠模式使用）"""
    
    # ==================== 温度退火配置（原量子退火）====================
    annealing_enabled: bool = True
    """是否启用温度退火 (推荐 True)"""
    
    T_start: float = 10.0
    """起始温度 (推荐 10.0，高温=软纠缠)"""
    
    T_end: float = 0.1
    """终止温度 (推荐 0.1，低温=硬阈值纠缠)"""
    
    annealing_steps: int = 1000
    """退火步数 (推荐: 总训练步数的 10-20%)"""
    
    annealing_schedule: str = "cosine"
    """退火调度: linear/exponential/cosine"""
    
    # ==================== 量子隧穿启发配置 ====================
    quantum_tunneling_enabled: bool = False
    """是否启用量子隧穿启发 (推荐 False，实验性功能)"""
    
    tunneling_strength: float = 0.1
    """隧穿强度，越小隧穿概率越低 (推荐 0.1)"""
    
    num_tunnel_paths: int = 4
    """采样候选路径数 (推荐 4)"""
    
    energy_landscape: str = "entropy"
    """能量景观类型: entropy/variance/hybrid"""
    
    tunneling_schedule: str = "adaptive"
    """隧穿调度策略: fixed/linear/adaptive"""

    # ==================== 预定义配置 ====================
    
    @classmethod
    def tiny(cls) -> "TESMConfig":
        """极小模型 - 快速实验/调试 (max_seq_len=512)
        
        适用场景: 快速验证、小数据集 (<100K)
        参数量: ~180M
        d_model=768: 支持 Qwen3 分词器 (vocab=151936) 的最小可行维度
        """
        return cls(
            d_model=768, n_layer=12, d_intermediate=3072, max_seq_len=1024,
            d_state=384, expand=2, ent_rank=96,
            entanglement_scale=0.25, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=16,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=1.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=500, annealing_schedule="cosine",
            quantum_tunneling_enabled=False,
        )
    
    @classmethod
    def small(cls) -> "TESMConfig":
        """小模型 - 中等规模训练 (max_seq_len=512)
        
        适用场景: 中等数据集 (100K-1M)
        参数量: ~50M
        """
        return cls(
            d_model=512, n_layer=16, d_intermediate=1536, max_seq_len=512,
            d_state=256, expand=2, ent_rank=48,
            entanglement_scale=0.2, entanglement_threshold=0.08,
            entanglement_init=0.3, entanglement_window=0,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=1.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=2000, annealing_schedule="cosine",
        )
    
    @classmethod
    def small_short(cls) -> "TESMConfig":
        """小模型 - 短序列优化版 (max_seq_len=256)
        
        适用场景: 短文本任务、快速收敛
        参数量: ~50M
        特点: decay_init_bias=0.0，适合短序列
        """
        return cls(
            d_model=512, n_layer=16, d_intermediate=1536, max_seq_len=256,
            d_state=256, expand=2, ent_rank=48,
            entanglement_scale=0.25, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=16,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=0.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=1000, annealing_schedule="cosine",
        )

    @classmethod
    def base(cls) -> "TESMConfig":
        """基础模型 - 大规模训练 (max_seq_len=2048)
        
        适用场景: 大规模预训练 (1M-10M)
        参数量: ~200M
        """
        return cls(
            d_model=768, n_layer=24, d_intermediate=2048, max_seq_len=2048,
            d_state=384, expand=2, ent_rank=64,
            entanglement_scale=0.2, entanglement_threshold=0.08,
            entanglement_init=0.3, entanglement_window=32,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=2.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=5000, annealing_schedule="cosine",
        )

    @classmethod
    def medium(cls) -> "TESMConfig":
        """中等模型 - 大规模训练 (max_seq_len=2048)
        
        适用场景: 大规模预训练 (>10M)
        参数量: ~500M
        """
        return cls(
            d_model=1024, n_layer=32, d_intermediate=2816, max_seq_len=2048,
            d_state=512, expand=2, ent_rank=80,
            entanglement_scale=0.25, entanglement_threshold=0.08,
            entanglement_init=0.35, entanglement_window=32,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=2.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=10000, annealing_schedule="cosine",
        )
    
    @classmethod
    def long_context(cls) -> "TESMConfig":
        """长上下文模型 (max_seq_len=16384)
        
        适用场景: 长文档处理、长上下文推理
        参数量: ~50M
        特点: decay_init_bias=6.0，超长记忆
        """
        return cls(
            d_model=256, n_layer=8, d_intermediate=512, max_seq_len=16384,
            d_state=512, expand=2, ent_rank=32,
            entanglement_scale=0.25, entanglement_threshold=0.08,
            entanglement_init=0.3, entanglement_window=32,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=10000, annealing_schedule="cosine",
        )
    
    # ==================== 实验配置 (用于参数对比) ====================
    
    @classmethod
    def exp_decay_comparison(cls, decay_bias: float) -> "TESMConfig":
        """实验配置: decay_init_bias 对比实验
        
        用于对比不同 decay_init_bias 对训练效果的影响
        """
        return cls(
            d_model=256, n_layer=6, d_intermediate=1024, max_seq_len=256,
            d_state=256, expand=2, ent_rank=32,
            entanglement_scale=0.25, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=16,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=decay_bias,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=500, annealing_schedule="cosine",
        )
    
    @classmethod
    def exp_threshold_comparison(cls, threshold: float) -> "TESMConfig":
        """实验配置: entanglement_threshold 对比实验
        
        用于对比不同 entanglement_threshold 对纠缠激活率的影响
        """
        return cls(
            d_model=256, n_layer=6, d_intermediate=1024, max_seq_len=256,
            d_state=256, expand=2, ent_rank=32,
            entanglement_scale=0.25, entanglement_threshold=threshold,
            entanglement_init=0.3, entanglement_window=16,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=0.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=500, annealing_schedule="cosine",
        )
    
    @classmethod
    def exp_scale_comparison(cls, scale: float) -> "TESMConfig":
        """实验配置: entanglement_scale 对比实验
        
        用于对比不同 entanglement_scale 对纠缠贡献的影响
        """
        return cls(
            d_model=256, n_layer=6, d_intermediate=1024, max_seq_len=256,
            d_state=256, expand=2, ent_rank=32,
            entanglement_scale=scale, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=16,
            entanglement_block_size=256, state_scan_chunk_size=16,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=0.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=500, annealing_schedule="cosine",
        )
    
    # ==================== 大规模模型配置 (对标 GLM-5) ====================
    
    @classmethod
    def large_40b(cls) -> "TESMConfig":
        """40B 参数模型 - 等效 GLM-5 激活参数规模
        
        适用场景: 大规模预训练，长上下文任务
        参数量: ~40B (稠密模型，无 MoE)
        显存需求: ~250GB (训练)
        """
        return cls(
            d_model=4096, n_layer=48, d_intermediate=16384, max_seq_len=131072,
            d_state=2048, expand=2, ent_rank=256,
            entanglement_scale=0.2, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=64,
            entanglement_block_size=512, state_scan_chunk_size=32,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=20000, annealing_schedule="cosine",
        )
    
    @classmethod
    def large_70b(cls) -> "TESMConfig":
        """70B 参数模型
        
        适用场景: 大规模预训练
        参数量: ~70B (稠密模型)
        显存需求: ~450GB (训练)
        """
        return cls(
            d_model=6144, n_layer=48, d_intermediate=24576, max_seq_len=131072,
            d_state=3072, expand=2, ent_rank=384,
            entanglement_scale=0.2, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=64,
            entanglement_block_size=512, state_scan_chunk_size=32,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=30000, annealing_schedule="cosine",
        )
    
    @classmethod
    def large_100b(cls) -> "TESMConfig":
        """100B 参数模型 - 推荐 200K 上下文起点
        
        适用场景: 超大规模预训练，200K 上下文
        参数量: ~100B (稠密模型)
        显存需求: ~800GB (训练) → 需要 10x A100-80GB
        """
        return cls(
            d_model=8192, n_layer=56, d_intermediate=32768, max_seq_len=204800,
            d_state=4096, expand=2, ent_rank=512,
            entanglement_scale=0.2, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=64,
            entanglement_block_size=512, state_scan_chunk_size=32,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=50000, annealing_schedule="cosine",
        )
    
    @classmethod
    def large_200b(cls) -> "TESMConfig":
        """200B 参数模型
        
        适用场景: 超大规模预训练
        参数量: ~200B (稠密模型)
        显存需求: ~1.4TB (训练) → 需要 18x A100-80GB
        """
        return cls(
            d_model=10240, n_layer=72, d_intermediate=40960, max_seq_len=204800,
            d_state=5120, expand=2, ent_rank=640,
            entanglement_scale=0.2, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=128,
            entanglement_block_size=512, state_scan_chunk_size=32,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=100000, annealing_schedule="cosine",
        )
    
    @classmethod
    def large_400b(cls) -> "TESMConfig":
        """400B 参数模型 - 接近 GLM-5 总参数规模
        
        适用场景: 超大规模预训练，对标旗舰模型
        参数量: ~400B (稠密模型，无 MoE)
        显存需求: ~2.5TB (训练) → 需要 32x A100-80GB
        
        注意: GLM-5 使用 MoE (744B总参数, 40B激活参数)
              TESM 是稠密模型，400B 即为实际计算量
        """
        return cls(
            d_model=12288, n_layer=96, d_intermediate=49152, max_seq_len=204800,
            d_state=6144, expand=2, ent_rank=768,
            entanglement_scale=0.2, entanglement_threshold=0.05,
            entanglement_init=0.3, entanglement_window=128,
            entanglement_block_size=512, state_scan_chunk_size=32,
            use_triton_kernels=True, kernel_backend="auto", kernel_mode="fast",
            decay_init_bias=6.0,
            annealing_enabled=True, T_start=10.0, T_end=0.1,
            annealing_steps=200000, annealing_schedule="cosine",
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TESMConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
