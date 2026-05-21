# TESM 参数配置指南

本文档详细说明 TESM 模型各参数的含义、适用场景和推荐配置。

---

## 目录

1. [模型架构参数](#模型架构参数)
2. [SSM 核心参数](#ssm-核心参数)
3. [纠缠机制参数](#纠缠机制参数)
4. [量子退火参数](#量子退火参数)
5. [预定义配置场景](#预定义配置场景)
6. [常见问题与调参建议](#常见问题与调参建议)

---

## 模型架构参数

### `d_model`
- **含义**: 模型隐藏层维度
- **影响**: 模型容量、参数量、计算成本
- **推荐值**:
  - 小规模实验: 256-512
  - 中等规模: 512-768
  - 大规模: 768-1024+
- **场景**:
  - `256`: 快速实验、调试、小数据集（<100K样本）
  - `512`: 中等数据集（100K-1M样本）
  - `768+`: 大规模预训练（>1M样本）

### `n_layer`
- **含义**: Transformer 层数
- **影响**: 模型深度、推理速度
- **推荐值**:
  - 小模型: 6-12
  - 中模型: 12-24
  - 大模型: 24-32
- **注意**: 层数过多而 `d_model` 过小会导致训练困难

### `d_intermediate`
- **含义**: FFN 中间层维度
- **影响**: FFN 容量
- **推荐值**: `d_model * 4` (标准 Transformer 配置)
- **示例**: `d_model=256` → `d_intermediate=1024`

### `max_seq_len`
- **含义**: 最大序列长度
- **影响**: 显存占用、长程依赖能力
- **推荐值**:
  - 短文本: 256-512
  - 中等文本: 512-2048
  - 长文本: 2048-16384+
- **注意**: 需要与 `decay_init_bias` 配合调整

---

## SSM 核心参数

### `d_state`
- **含义**: 状态空间维度
- **影响**: 状态记忆容量、长程依赖能力
- **推荐值**: `d_model` 或 `d_model * 2`
- **场景**:
  - `256`: 短序列（<512 tokens）
  - `512`: 中等序列（512-2048 tokens）
  - `768+`: 长序列（>2048 tokens）

### `expand`
- **含义**: 状态扩展因子
- **影响**: 状态维度 = `d_state * expand`
- **推荐值**: `2` (默认)

### `decay_init_bias` ⚠️ 关键参数
- **含义**: 状态衰减初始偏置
- **数学**: `decay = sigmoid(decay_raw + decay_init_bias)`
- **影响**: 历史状态保留率、位置区分能力
- **状态保留率对照表**:

| `decay_init_bias` | `sigmoid(bias)` | 状态保留率 | 适用场景 |
|-------------------|-----------------|------------|----------|
| **-3.0** | 0.05 | 5% | 极短序列（<32），几乎无记忆 |
| **-1.0** | 0.27 | 27% | 短序列（32-64），快速遗忘 |
| **0.0** | 0.50 | 50% | 中等序列（64-256），**推荐起点** |
| **1.0** | 0.73 | 73% | 中长序列（256-1024） |
| **2.0** | 0.88 | 88% | 长序列（1024-4096） |
| **3.0** | 0.95 | 95% | 超长序列（>4096），**不适合短序列** |
| **6.0** | 0.997 | 99.7% | 极长序列（>16384） |

- **重要公式**: 经过 `L` 步后，初始状态保留率为 `sigmoid(bias)^L`
  - `bias=3.0, L=256`: `0.95^256 ≈ 0.00002` (看似衰减，但梯度稀释)
  - `bias=0.0, L=256`: `0.5^256 ≈ 0` (正常衰减)

- **调参建议**:
  ```python
  # 短序列 (<256)
  decay_init_bias = 0.0  # 或 -1.0
  
  # 中等序列 (256-1024)
  decay_init_bias = 1.0
  
  # 长序列 (1024-4096)
  decay_init_bias = 2.0
  
  # 超长序列 (>4096)
  decay_init_bias = 3.0  # 当前默认值
  ```

---

## 纠缠机制参数

### `entanglement_threshold`
- **含义**: 三值纠缠激活阈值
- **数学**: `ternary = +1 if score > threshold, -1 if score < -threshold, else 0`
- **影响**: 纠缠激活率
- **推荐值**:
  - `0.05`: 高激活率（~25-30%），强纠缠
  - `0.08`: 中等激活率（~15-20%），平衡
  - `0.10`: 低激活率（~10-15%），弱纠缠
  - `0.15`: 极低激活率（<10%），几乎无纠缠

- **激活率对照**:
  - 当前日志显示 `ent=+11.2% -11.8% 0=77.0%`
  - 意味着 ~22% 位置激活纠缠，78% 独立状态

### `entanglement_scale`
- **含义**: 纠缠混合权重
- **数学**: `entangled = states + scale * (signed_avg - states)`
- **影响**: 纠缠对最终状态的贡献
- **推荐值**: `0.2-0.3`
- **场景**:
  - `0.1`: 纠缠贡献小，保守
  - `0.25`: 平衡（默认）
  - `0.4`: 纠缠贡献大，激进

### `entanglement_window`
- **含义**: 局部纠缠窗口大小
- **影响**: 纠缠计算范围、显存占用
- **推荐值**:
  - `8`: 极短依赖
  - `16`: 短依赖（默认）
  - `32`: 中等依赖
  - `64`: 长依赖

### `ent_rank`
- **含义**: 纠缠查询/键的秩
- **影响**: 纠缠表达能力
- **推荐值**: `d_model / 8` 到 `d_model / 4`
- **示例**: `d_model=256` → `ent_rank=32-64`

---

## 量子退火参数

### `annealing_enabled`
- **含义**: 是否启用量子退火
- **作用**: 从高温软纠缠平滑过渡到低温硬纠缠
- **推荐**: `True` (默认)

### `T_start` / `T_end`
- **含义**: 起始/终止温度
- **影响**:
  - 高温 (`T > 1.0`): 软纠缠，类似 softmax
  - 低温 (`T <= 1.0`): 硬纠缠，三值阈值
- **推荐值**:
  - `T_start = 10.0`: 初始高温，平滑纠缠
  - `T_end = 0.1`: 最终低温，硬阈值纠缠

### `annealing_steps`
- **含义**: 退火步数
- **影响**: 从高温到低温的过渡速度
- **推荐值**: 总训练步数的 `10-20%`
- **示例**:
  - 总步数 5000 → `annealing_steps = 500-1000`
  - 总步数 50000 → `annealing_steps = 5000-10000`

### `annealing_schedule`
- **含义**: 退火调度
- **选项**:
  - `linear`: 线性降温
  - `exponential`: 指数降温
  - `cosine`: 余弦降温（推荐，更平滑）

---

## 预定义配置场景

### 场景 1: 快速实验 / 调试
```python
TESMConfig(
    d_model=256,
    n_layer=6,
    d_intermediate=1024,
    max_seq_len=256,
    ssm_cfg={
        "d_state": 256,
        "ent_rank": 48,
        "entanglement_threshold": 0.05,
        "entanglement_window": 16,
        "decay_init_bias": 0.0,  # 短序列关键！
        "annealing_steps": 500,
    }
)
```
- **适用**: 快速验证、小数据集（<100K）
- **参数量**: ~10-20M

### 场景 2: 中等规模预训练
```python
TESMConfig(
    d_model=512,
    n_layer=12,
    d_intermediate=2048,
    max_seq_len=512,
    ssm_cfg={
        "d_state": 512,
        "ent_rank": 64,
        "entanglement_threshold": 0.08,
        "entanglement_window": 32,
        "decay_init_bias": 1.0,
        "annealing_steps": 2000,
    }
)
```
- **适用**: 中等数据集（100K-1M）
- **参数量**: ~50-100M

### 场景 3: 大规模预训练
```python
TESMConfig(
    d_model=768,
    n_layer=16,
    d_intermediate=3072,
    max_seq_len=2048,
    ssm_cfg={
        "d_state": 768,
        "ent_rank": 96,
        "entanglement_threshold": 0.08,
        "entanglement_window": 32,
        "decay_init_bias": 2.0,
        "annealing_steps": 5000,
    }
)
```
- **适用**: 大规模预训练（>1M）
- **参数量**: ~200-500M

### 场景 4: 长上下文推理
```python
TESMConfig(
    d_model=256,
    n_layer=8,
    d_intermediate=512,
    max_seq_len=16384,
    ssm_cfg={
        "d_state": 512,
        "ent_rank": 32,
        "entanglement_threshold": 0.08,
        "entanglement_window": 32,
        "decay_init_bias": 6.0,  # 超长记忆
        "annealing_steps": 10000,
    }
)
```
- **适用**: 长文档处理、长上下文推理
- **参数量**: ~30-50M

---

## 常见问题与调参建议

### Q1: 训练 1 epoch 后模型输出不连贯？

**原因**: `decay_init_bias` 设置过高，导致位置区分困难

**解决方案**:
```python
# 检查当前配置
if max_seq_len <= 256:
    decay_init_bias = 0.0  # 或 -1.0
elif max_seq_len <= 1024:
    decay_init_bias = 1.0
else:
    decay_init_bias = 2.0+
```

### Q2: 纠缠激活率过低（<10%）？

**解决方案**: 降低 `entanglement_threshold`
```python
entanglement_threshold = 0.05  # 从 0.08 降低
```

### Q3: 训练速度慢？

**解决方案**:
1. 启用 TileLang kernel: `kernel_backend="auto"`
2. 使用 `kernel_mode="fast"` (默认)
3. 减小 `entanglement_window`

### Q4: 显存不足？

**解决方案**:
1. 减小 `batch_size`
2. 减小 `d_model` / `d_state`
3. 启用梯度检查点: `gradient_checkpointing=True`
4. 减小 `max_seq_len`

### Q5: 如何选择合适的模型规模？

| 数据量 | 推荐 d_model | 推荐 n_layer | 预期参数量 |
|--------|--------------|--------------|------------|
| <100K | 256 | 6 | ~10M |
| 100K-1M | 512 | 12 | ~50M |
| 1M-10M | 768 | 16 | ~200M |
| >10M | 1024+ | 24+ | ~500M+ |

---

## 参数依赖关系图

```
max_seq_len ──────┐
                  ├──> decay_init_bias
d_state ──────────┘

d_model ──────────┐
                  ├──> ent_rank
entanglement_window┘

总训练步数 ────────> annealing_steps
                  └──> annealing_steps = 总步数 * 0.1~0.2

纠缠激活率 ────────> entanglement_threshold
                  └──> 激活率低 → 降低阈值
```

---

## 总结

**最关键的三个参数**:

1. **`decay_init_bias`**: 根据序列长度选择，短序列用小值（0.0），长序列用大值（3.0+）
2. **`entanglement_threshold`**: 控制纠缠激活率，建议从 0.08 开始调整
3. **`annealing_steps`**: 设为总训练步数的 10-20%

**调参优先级**:
1. 先确定 `decay_init_bias` (根据 `max_seq_len`)
2. 再调整 `entanglement_threshold` (观察纠缠激活率)
3. 最后微调 `entanglement_scale` 和 `annealing_steps`
