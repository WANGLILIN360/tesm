# TESM (Token-Entangled State Machine) 测试报告

**测试日期**: 2026-06-03
**测试版本**: GitHub HEAD (wanglilin360/tesm)
**测试环境**: Python 3.12, PyTorch (with CUDA 12.x), CPU/GPU

---

## 一、项目概述

TESM 是一种创新的序列建模架构，结合了状态空间模型 (SSM) 与局部纠缠机制。核心组件包括：

- **TESM_SISO**: 单输入单输出核心层
- **TESMMIMO_Optimized**: 多头扩展版本
- **BitLinear**: INT2 量化线性层
- **TernaryQuantumTunneling**: 三值量子隧穿模块
- **多后端支持**: PyTorch / CUDA / Triton / TileLang

---

## 二、测试覆盖

### 2.1 测试模块

| 模块 | 测试项数 | 通过 | 失败 |
|------|---------|------|------|
| 配置模块 (TESMConfig) | 3 | 3 | 0 |
| BitLinear 量化层 | 4 | 4 | 0 |
| TernaryQuantumTunneling | 3 | 3 | 0 |
| TESM_SISO 核心层 | 6 | 6 | 0 |
| TESMMIMO_Optimized | 3 | 3 | 0 |
| MixerModel / TESMLMHeadModel | 6 | 6 | 0 |
| Block / RMSNorm | 3 | 3 | 0 |
| TrainingConfig | 2 | 2 | 0 |
| INT2 量化工具 | 4 | 4 | 0 |
| PagedStateCache | 4 | 4 | 0 |
| 模型保存/加载 | 1 | 1 | 0 |
| **总计** | **39** | **39** | **0** |

### 2.2 深入测试

| 测试项 | 结果 |
|--------|------|
| 数值稳定性 (5步训练) | PASS |
| 不同 batch size | PASS |
| 长序列 (接近 max_seq_len) | PASS |
| 增量推理 (单步/多步) | PASS |
| Embedding 共享 | PASS |
| 生成方法 (greedy/top-k) | PASS |
| 全局纠缠模式 | PASS |
| INT2 save/load 一致性 | PASS |

---

## 三、发现的 Bug 和问题

### Bug 1: TESM_SISO.forward 缺少序列长度检查 [中等]

**描述**: TESM_SISO.forward() 没有检查输入序列长度是否超过 `max_seq_len`，可能导致缓冲区溢出或未定义行为。

**复现**:
```python
layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                  entanglement_window=4, max_seq_len=8, kernel_backend='torch')
x = torch.randn(1, 16, 64)  # seq_len=16 > max_seq_len=8
out = layer(x)  # 应该报错但没有
```

**影响**: 可能导致 RoPE 位置编码越界、因果掩码失效等问题。

**建议修复**: 在 forward() 开头添加：
```python
if seqlen > self.max_seq_len:
    raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")
```

---

### Bug 2: 空序列输入导致 ValueError [中等]

**描述**: 当输入序列长度为0时，抛出 `ValueError: range() arg 3 must not be zero`。

**复现**:
```python
layer = TESM_SISO(d_model=64, d_state=32, expand=2, ent_rank=8,
                  entanglement_window=4, max_seq_len=32, kernel_backend='torch')
x = torch.randn(1, 0, 64)
out = layer(x)  # ValueError
```

**建议修复**: 添加空序列检查或优雅处理。

---

### Bug 3: d_state=0 导致 IndexError [低]

**描述**: 极端配置下（d_state=0），模型初始化时会产生警告，前向传播时抛出 IndexError。

**复现**:
```python
layer = TESM_SISO(d_model=64, d_state=0, expand=2, ent_rank=8,
                  entanglement_window=4, max_seq_len=32, kernel_backend='torch')
x = torch.randn(1, 4, 64)
out = layer(x)  # IndexError: max(): Expected reduction dim 2 to have non-zero size
```

**建议修复**: 在 __init__ 中添加参数验证，确保 d_state > 0。

---

### Bug 4: MIMO d_head 整数除法问题 [低]

**描述**: TESMMIMO_Optimized 中 `d_head = self.d_inner // n_heads` 使用整数除法。当 `d_inner` 不能整除 `n_heads` 时，d_head 会被截断，可能导致维度不匹配。

**复现**:
```python
layer = TESMMIMO_Optimized(d_model=65, d_state=32, n_heads=4, expand=2, ...)
# d_inner = 130, d_head = 130 // 4 = 32 (实际应该是32.5)
```

**建议修复**: 添加验证或警告：
```python
if self.d_inner % n_heads != 0:
    import warnings
    warnings.warn(f"d_inner ({self.d_inner}) is not divisible by n_heads ({n_heads})")
```

---

### Bug 5: _parallel_prefix_scan 长序列数值稳定性 [低]

**描述**: 在 `_parallel_prefix_scan` 中，当序列很长且衰减因子接近1时，`torch.cumprod(decay, dim=1)` 可能下溢或上溢。

**位置**: `tesm_ssm/modules/tesm.py:1071`

**当前代码**:
```python
A = torch.cumprod(decay, dim=1)  # (B, L, D)
weighted_update = update / A.clamp_min(1e-30)
```

**建议修复**: 考虑使用 log-space 计算或使用更稳定的累积算法。

---

### Bug 6: 无效的 annealing_schedule 静默回退 [低]

**描述**: 当传入无效的 `annealing_schedule` 值时，`get_temperature()` 静默回退到 `T_start`，而不是报错。

**位置**: `tesm_ssm/modules/tesm.py:597-603`

**当前代码**:
```python
else:
    T = self.T_start  # 无效值时静默回退
```

**建议修复**: 添加警告或抛出 ValueError。

---

### Bug 7: TESMMIMO_Optimized._parallel_state_scan_mimo_stable 使用 Python 循环 [低]

**描述**: MIMO 版本的稳定扫描使用 Python 循环而不是向量化操作，效率低下。

**位置**: `tesm_ssm/modules/tesm_mimo.py:356-368`

**建议修复**: 使用 PyTorch 向量化操作或并行前缀和算法。

---

### Bug 8: MixerModel 不处理 seqlen=0 的情况 [低]

**描述**: 虽然 TESM_SISO 会在 seqlen=0 时抛出 ValueError，但 MixerModel 没有前置检查。

**建议修复**: 在 MixerModel.forward 中添加空序列检查。

---

## 四、代码质量问题

### 4.1 文档与注释

| 方面 | 评价 |
|------|------|
| 模块文档字符串 | 良好，大部分类和方法有 docstring |
| 配置参数注释 | 优秀，config_guide.md 非常详细 |
| 复杂算法注释 | 良好，有数学公式说明 |

### 4.2 代码结构

| 方面 | 评价 |
|------|------|
| 模块化程度 | 优秀，各后端隔离清晰 |
| 错误处理 | 良好，大量使用 try-except 处理可选依赖 |
| 类型注解 | 部分有，可加强 |
| 参数验证 | 中等，部分边界情况未覆盖 |

### 4.3 潜在改进点

1. **统一序列长度检查**: 应该在每个层的入口处检查，而不是仅在 MixerModel 中
2. **参数验证**: 建议在 __init__ 中验证关键参数（d_state > 0, entanglement_window >= 0 等）
3. **向后兼容性**: CUDA/Triton kernel 的错误信息可以更具体
4. **测试覆盖**: 缺少对 CUDA kernel 的测试（需要 GPU 环境）

---

## 五、总结

### 5.1 总体评价

TESM 项目代码质量**良好**，架构设计清晰，多后端支持完善。核心功能在 CPU 环境下（PyTorch backend）全部通过测试。

### 5.2 Bug 统计

| 严重程度 | 数量 | 描述 |
|---------|------|------|
| 中等 | 2 | 缺少序列长度检查、空序列处理 |
| 低 | 6 | 边界条件、参数验证、数值稳定性 |

### 5.3 建议优先级

1. **高**: 在 TESM_SISO.forward 中添加 `seqlen > max_seq_len` 检查
2. **中**: 添加空序列处理和参数验证（d_state > 0 等）
3. **低**: 改进错误处理、添加警告信息

---

*报告生成时间: 2026-06-03*
*测试工具: 自定义 pytest 风格测试套件 + 手动代码审查*
