# TESM (Token-Entangled State Machine)

状态纠缠机 - 一种基于状态纠缠理论的序列建模架构。

## 简介

TESM 是一种创新的序列建模架构，将状态空间模型 (SSM) 与局部纠缠机制结合，实现高效的长序列建模。核心特性：

- **状态纠缠**: 通过局部窗口内的 token 纠缠增强信息流动
- **多后端支持**: PyTorch / CUDA / Triton / TileLang 四种后端
- **INT2 量化**: 支持 BitLinear INT2 量化推理加速
- **灵活配置**: 从 tiny (10M) 到 400B 的完整模型规格

## 安装

### 基础安装 (仅 PyTorch 后端)

```bash
pip install -e .
```

### CUDA 扩展安装 (推荐)

```bash
# 需要 CUDA Toolkit 和 PyTorch with CUDA
pip install -e .  # setup.py 会自动编译 CUDA 扩展
```

### 依赖

- Python >= 3.8
- PyTorch >= 2.0
- (可选) Triton >= 2.1
- (可选) CUDA Toolkit >= 11.8

## 快速开始

```python
from tesm_ssm import TESMConfig, TESMLMHeadModel

# 使用预设配置
config = TESMConfig.small()
model = TESMLMHeadModel(config)

# 或自定义配置
config = TESMConfig(
    d_model=512,
    n_layer=16,
    max_seq_len=512,
    ssm_cfg={
        "d_state": 256,
        "entanglement_window": 16,
        "kernel_backend": "auto",  # auto/cuda/triton/torch
    }
)

# 前向传播
input_ids = torch.randint(0, config.vocab_size, (2, 128))
outputs = model(input_ids)
logits = outputs.logits
```

## 配置预设

| 预设 | 参数量 | max_seq_len | 适用场景 |
|------|--------|-------------|----------|
| `tiny()` | ~10M | 256 | 调试/快速实验 |
| `small()` | ~50M | 512 | 中等数据集 |
| `base()` | ~200M | 2048 | 大规模训练 |
| `medium()` | ~500M | 2048 | 更大容量 |
| `large_40b()` | ~40B | 131K | 对标 GLM-5 |
| `large_400b()` | ~400B | 204K | 旗舰级 |

详细配置说明见 [tesm_ssm/docs/config_guide.md](tesm_ssm/docs/config_guide.md)

## 后端选择

| 后端 | 训练 | 推理 | 说明 |
|------|:----:|:----:|------|
| `torch` | ✅ | ✅ | 默认回退，兼容性最好 |
| `cuda` | ✅ | ✅ | 完整 autograd，需编译 |
| `triton` | ✅ | ✅ | 无需编译，自动 autograd |
| `tilelang` | ✅ | ✅ | MIMO 优化，实验性 |

### 功能支持矩阵

| 功能 | PyTorch | CUDA | Triton | TileLang |
|------|:-------:|:----:|:------:|:--------:|
| **基础版 (SISO)** | | | | |
| BitLinear 量化线性 | ✅ | ✅ | ✅ | ✅ |
| 状态扫描 | ✅ | ✅ | ✅ | ✅ |
| 局部窗口纠缠 | ✅ | ✅ | ✅ | ✅ |
| 全局纠缠 | ✅ | ✅ | ✅ | ✅ |
| 融合输出 | ✅ | ✅ | ✅ | ✅ |
| **MIMO 版 (多头)** | | | | |
| 多头状态扫描 | ✅ | ✅ | ✅ | ✅ |
| 多头局部纠缠 | ✅ | ✅ | ✅ | ✅ |
| 多头全局纠缠 | ✅ | ✅ | ✅ | ✅ |
| 融合 MIMO 前向 | ✅ | ✅ | ✅ | ✅ |

```python
# 自动选择 (推荐)
config = TESMConfig.small()
config.ssm_cfg["kernel_backend"] = "auto"

# 强制 CUDA (训练+推理)
config.ssm_cfg["kernel_backend"] = "cuda"

# 强制 Triton (训练+推理，无需编译)
config.ssm_cfg["kernel_backend"] = "triton"

# TileLang (MIMO 优化)
config.ssm_cfg["kernel_backend"] = "tilelang"

# 纯 PyTorch (兼容性最好)
config.ssm_cfg["kernel_backend"] = "torch"
```

## 核心模块

- `tesm_ssm.modules.tesm.TESM` - 核心 TESM 层
- `tesm_ssm.modules.tesm.BitLinear` - INT2 量化线性层
- `tesm_ssm.modules.tesm_mimo.TESMMIMO_Optimized` - MIMO 多头变体
- `tesm_ssm.ops.cuda` - CUDA 算子
- `tesm_ssm.ops.triton` - Triton 算子 (推理加速)

## 项目结构

```
tesm_ssm/
├── models/          # 模型定义 (TESMLMHeadModel)
├── modules/         # 核心模块 (TESM, BitLinear, Block)
├── ops/             # 算子实现
│   ├── cuda/        # CUDA 后端
│   ├── triton/      # Triton 后端
│   └── tilelang/    # TileLang 后端
├── utils/           # 工具函数
└── docs/            # 配置文档

csrc/tesm_ops/       # CUDA C++ 源码
```

## 作为依赖使用

在 `requirements.txt` 或 `pyproject.toml` 中添加：

```text
# 从本地路径安装
tesm_ssm @ file:///path/to/tesm-main-official-backup
```

或在其他项目中：

```python
# pyproject.toml
[project]
dependencies = [
    "tesm_ssm @ file:///home/lingji/wang/tesm-main-official-backup",
]
```

## 许可证

MIT License
