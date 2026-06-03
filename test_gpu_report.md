# TESM GPU 代码审查报告

**环境**: PyTorch 2.8.0+cu128, Triton 3.4.0, 无物理 GPU
**审查日期**: 2026-06-03

---

## 1. GPU 降级路径测试 (19/23 通过)

### 降级行为评估

| 后端 | 降级检查 | 行为 | 评价 |
|------|---------|------|------|
| CUDA | `torch.cuda.is_available()` | 正确返回 None | ✅ 正确 |
| Triton | `torch.cuda.is_available() + triton` | 正确返回 False | ✅ 正确 |
| BitLinear (auto) | 自动选择可用后端 | 正确选择 torch | ✅ 正确 |
| torch.compile | `torch.compiler.disable` | 正确禁用编译 | ✅ 正确 |

### 发现的问题

#### [中等] BitLinear kernel_backend='cuda' 严格报错 (无回退)

**位置**: `tesm_ssm/modules/tesm.py:279-284`

**问题**: 当显式指定 `kernel_backend='cuda'` 但 CUDA 不可用时，BitLinear 直接抛出 RuntimeError，而不是回退到 PyTorch 实现。

```python
# 当前代码 (line 274-285)
if self.kernel_backend in {"auto", "torch"}:
    return F.linear(qinput, qweight, bias)

# 指定的后端不可用，报错
raise RuntimeError(
    f"kernel_backend='{self.kernel_backend}' specified but BitLinear kernel not available. "
    ...
)
```

**影响**: 用户设置 `kernel_backend='cuda'` 但在 CPU 上运行时直接崩溃。

**建议修复**: 添加一个 `strict=False` 参数，或在无 GPU 时自动回退：
```python
if self.kernel_backend in {"auto", "torch"} or not torch.cuda.is_available():
    return F.linear(qinput, qweight, bias)
```

#### [中等] Triton backward kernel 中的 None 检查可能无效

**位置**: `tesm_ssm/ops/triton/tesm_kernels.py:287, 305`

**问题**: `grad_init_ptr is not None` 在 Triton JIT kernel 内部可能不起作用。Triton JIT 编译器会将 Python None 转换为 null pointer 或 0，但 `is not None` 检查在 JIT 编译上下文中可能始终返回 True。

```python
# 第287行
grad_state = tl.load(grad_init_ptr + row_id) if grad_init_ptr is not None else 0.0

# 第305行
if grad_init_ptr is not None:
    tl.store(grad_init_ptr + row_id, grad_state)
```

**影响**: 当 `grad_init_ptr` 实际上不应该被使用时，可能产生不正确的梯度或内存错误。

**建议修复**: 使用一个标志参数代替 None 检查：
```python
# 添加 has_grad_init 参数
grad_state = tl.load(grad_init_ptr + row_id) if has_grad_init else 0.0
```

#### [低] Triton backward kernel 使用 Python 条件

**位置**: `tesm_ssm/ops/triton/tesm_kernels.py:297`

**问题**: `grad_state * state_prev if col > 0 else 0.0` 使用 Python 条件。虽然 Triton JIT 可以编译时评估 `col > 0`（因为 `col` 是 Python int），但这种写法不够清晰，建议使用 `tl.where`。

```python
# 当前代码
grad_decay = grad_state * state_prev if col > 0 else 0.0

# 建议
grad_decay = tl.where(col > 0, grad_state * state_prev, 0.0)
```

---

## 2. torch.compile 兼容性

### Graph Break 问题

**位置**: `tesm_ssm/models/mixer_seq_simple.py:234`

```python
total = layer.mixer._stats_total_buffer.item() if layer.mixer._stats_total_buffer is not None else 1.0
```

**问题**: `Tensor.item()` 调用导致 torch.compile 的 graph break。

**影响**: 不影响功能正确性，但降低了 torch.compile 的优化效果。

**严重性**: 低（纯 PyTorch 代码也正常工作）

---

## 3. CUDA Kernel 接口完整性

### 接口覆盖 (全部正确)

| Kernel | Forward | Backward | Autograd | 评价 |
|--------|---------|----------|----------|------|
| StateScan (SISO) | ✅ | ✅ | ✅ | 完整 |
| LocalEnt (SISO) | ✅ | ✅ | ✅ | 完整 |
| QuantizedLinear | ✅ | ✅ | ✅ | 完整 |
| GlobalEnt (SISO) | ✅ | ✅ | ✅ | 完整 |
| FusedOutput (SISO) | ✅ | ✅ | ✅ | 完整 |
| StateScan (MIMO) | ✅ | ✅ | ✅ | 完整 |
| LocalEnt (MIMO) | ✅ | ✅ | ✅ | 完整 |
| GlobalEnt (MIMO) | ✅ | ✅ | ✅ | 完整 |
| FusedOutput (MIMO) | ✅ | ✅ | ✅ | 完整 |
| INT2 Linear | ✅ | N/A | N/A | 完整 |
| INT8xINT2 Linear | ✅ | N/A | N/A | 完整 |

---

## 4. 建议修复优先级

| 优先级 | 问题 | 文件 | 行号 |
|--------|------|------|------|
| 高 | BitLinear CUDA 回退 | `tesm.py` | 275-285 |
| 中 | Triton None 检查 | `tesm_kernels.py` | 287, 305 |
| 低 | Triton Python 条件 | `tesm_kernels.py` | 297 |
| 低 | torch.compile graph break | `mixer_seq_simple.py` | 234 |
