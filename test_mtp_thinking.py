#!/usr/bin/env python3
"""
MTP (Multi-Token Prediction) 和 Thinking Mode 测试
"""

import sys
sys.path.insert(0, '/mnt/agents/tesm')

import torch
import torch.nn as nn
import time

results = []

def test(name, fn):
    start = time.time()
    try:
        detail = fn()
        dur = time.time() - start
        results.append((name, True, dur, detail or ""))
        print(f"  [PASS] {name} ({dur:.2f}s) {detail}")
    except Exception as e:
        dur = time.time() - start
        err = f"{type(e).__name__}: {str(e)[:200]}"
        results.append((name, False, dur, err))
        print(f"  [FAIL] {name} ({dur:.2f}s)")
        print(f"         {err}")

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel
from tesm_ssm.modules.mtp import MTPHead, SpeculativeDecoder
from tesm_ssm.modules.thinking import (
    ThinkingModeController, ThinkingOutput,
    THINK_TRIGGER, THOUGHT_START, THOUGHT_END,
)

print("=" * 70)
print("MTP + Thinking Mode 测试")
print("=" * 70)

# ============================================================
# 1. MTP Head 基础测试
# ============================================================
print("\n[1. MTP Head 基础测试]")

def t_mtp_head_basic():
    """MTP Head 基本功能"""
    mtp = MTPHead(d_model=64, vocab_size=100, num_pred_tokens=4)
    hidden = torch.randn(2, 8, 64)
    logits_list = mtp(hidden)
    
    assert len(logits_list) == 4, f"Expected 4 logits, got {len(logits_list)}"
    for i, logits in enumerate(logits_list):
        assert logits.shape == (2, 8, 100), f"Logits {i} shape wrong: {logits.shape}"
        assert torch.isfinite(logits).all()
    
    params = sum(p.numel() for p in mtp.parameters())
    return f"4 predictions, each (2,8,100), params={params/1e6:.2f}M"

test("MTP Head 基本", t_mtp_head_basic)

def t_mtp_head_weight_tying():
    """MTP Head weight tying"""
    mtp = MTPHead(d_model=64, vocab_size=100, num_pred_tokens=4)
    embedding = nn.Embedding(100, 64)
    
    hidden = torch.randn(1, 4, 64)
    logits_list = mtp(hidden, input_embeddings=embedding)
    
    assert len(logits_list) == 4
    assert logits_list[0].shape == (1, 4, 100)
    return "weight tying OK"

test("MTP Head weight tying", t_mtp_head_weight_tying)

def t_mtp_head_gradient():
    """MTP Head 梯度流"""
    mtp = MTPHead(d_model=32, vocab_size=50, num_pred_tokens=2)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    logits_list = mtp(hidden)
    loss = sum(l.sum() for l in logits_list)
    loss.backward()
    
    assert hidden.grad is not None
    for name, p in mtp.named_parameters():
        assert p.grad is not None, f"{name} no grad"
    return "all grads OK"

test("MTP Head 梯度", t_mtp_head_gradient)

def t_mtp_head_different_sizes():
    """不同配置"""
    for num_pred in [1, 2, 4, 8]:
        mtp = MTPHead(d_model=64, vocab_size=100, num_pred_tokens=num_pred)
        hidden = torch.randn(1, 4, 64)
        logits_list = mtp(hidden)
        assert len(logits_list) == num_pred
    return "num_pred=1,2,4,8 OK"

test("MTP Head 多配置", t_mtp_head_different_sizes)

# ============================================================
# 2. Thinking Mode 测试
# ============================================================
print("\n[2. Thinking Mode 测试]")

def t_thinking_parse():
    """解析 thinking 输出"""
    controller = ThinkingModeController(enabled=True)
    
    text = f"{THOUGHT_START}让我一步一步思考...\n1. 首先分析\n2. 然后推理{THOUGHT_END}最终答案是 42"
    result = controller.parse_output(text)
    
    assert isinstance(result, ThinkingOutput)
    assert "一步一步思考" in result.reasoning
    assert "最终答案是 42" in result.answer
    assert result.raw_text == text
    return f"reasoning={len(result.reasoning)}chars, answer={len(result.answer)}chars"

test("Thinking 解析", t_thinking_parse)

def t_thinking_no_thinking():
    """无 thinking 内容的输出"""
    controller = ThinkingModeController()
    
    text = "这是一个普通回答，没有 thinking 内容"
    result = controller.parse_output(text)
    
    assert result.reasoning == ""
    assert result.answer == text
    return "no thinking detected"

test("Thinking 无内容", t_thinking_no_thinking)

def t_thinking_trigger():
    """添加 thinking trigger"""
    controller = ThinkingModeController(enabled=True)
    
    prompt = "你是一个有用的助手"
    new_prompt = controller.add_thinking_trigger(prompt)
    
    assert THINK_TRIGGER in new_prompt
    return f"trigger added: {new_prompt[:30]}..."

test("Thinking trigger", t_thinking_trigger)

def t_thinking_format_training():
    """格式化训练样本"""
    controller = ThinkingModeController()
    
    text = controller.format_training_example(
        question="2+2=?",
        reasoning="2加2等于4",
        answer="4",
    )
    
    assert "Question:" in text
    assert THOUGHT_START in text
    assert THOUGHT_END in text
    assert "2加2等于4" in text
    return "training format OK"

test("Thinking 训练格式", t_thinking_format_training)

def t_thinking_enable_disable():
    """启用/禁用切换"""
    controller = ThinkingModeController(enabled=False)
    assert not controller.enabled
    
    controller.enable()
    assert controller.enabled
    
    controller.disable()
    assert not controller.enabled
    return "enable/disable OK"

test("Thinking 开关", t_thinking_enable_disable)

def t_thinking_extract():
    """提取和剥离 thinking"""
    controller = ThinkingModeController()
    
    text = f"{THOUGHT_START}推理过程{THOUGHT_END}答案"
    
    thinking_only = controller.extract_thinking_only(text)
    assert thinking_only == "推理过程"
    
    stripped = controller.strip_thinking(text)
    assert stripped == "答案"
    
    assert controller.is_thinking_content(text)
    assert not controller.is_thinking_content("普通文本")
    return "extract/strip/check OK"

test("Thinking 提取", t_thinking_extract)

# ============================================================
# 3. MTP + Thinking 集成测试
# ============================================================
print("\n[3. MTP + Thinking 集成]")

def t_mtp_with_thinking():
    """MTP + Thinking 集成"""
    from tesm_ssm.modules.thinking import MTPWithThinking
    
    mtp = MTPHead(d_model=64, vocab_size=100, num_pred_tokens=4)
    thinking = ThinkingModeController(enabled=True)
    combined = MTPWithThinking(mtp, thinking)
    
    assert combined.thinking.enabled
    assert combined.mtp_head is mtp
    return "integration OK"

test("MTP+Thinking 集成", t_mtp_with_thinking)

# ============================================================
# 4. 完整端到端测试
# ============================================================
print("\n[4. 完整端到端测试]")

def t_end_to_end_with_mtp():
    """带 MTP 的模型生成"""
    cfg = TESMConfig(d_model=64, n_layer=2, d_intermediate=128, max_seq_len=64,
                     vocab_size=100, kernel_backend="torch")
    model = TESMLMHeadModel(cfg)
    model.eval()
    
    # 创建 MTP head
    mtp = MTPHead(d_model=64, vocab_size=100, num_pred_tokens=2)
    
    # 测试 MTP head 独立工作
    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        outputs, _ = model(input_ids)
        hidden = outputs.hidden_states
        mtp_logits = mtp(hidden)
    
    assert len(mtp_logits) == 2
    assert all(l.shape == (1, 8, 100) for l in mtp_logits)
    return f"MTP generated {len(mtp_logits)} future token predictions"

test("E2E MTP", t_end_to_end_with_mtp)

def t_end_to_end_with_thinking():
    """带 Thinking Mode 的生成和解析"""
    controller = ThinkingModeController(enabled=True)
    
    # 模拟模型输出 (thinking 格式)
    mock_output = (
        f"{THOUGHT_START}"
        f"1. 分析: 这是一个数学问题\n"
        f"2. 计算: 2 + 2 = 4\n"
        f"3. 验证: 4 是正确的"
        f"{THOUGHT_END}"
        f"答案是 4"
    )
    
    result = controller.parse_output(mock_output)
    
    assert "数学问题" in result.reasoning
    assert "答案是 4" in result.answer
    assert controller.is_thinking_content(mock_output)
    return f"reasoning={result.reasoning[:20]}..., answer={result.answer}"

test("E2E Thinking", t_end_to_end_with_thinking)

# ============================================================
# 总结
# ============================================================
print()
print("=" * 70)
print("MTP + Thinking Mode 测试总结")
print("=" * 70)
passed = sum(1 for _, p, _, _ in results if p)
failed = len(results) - passed
total_t = sum(t for _, _, t, _ in results)
print(f"\n总计: {passed}/{len(results)} 通过, {failed} 失败")
print(f"总耗时: {total_t:.2f}s")
if failed:
    print("\n失败项:")
    for n, p, _, d in results:
        if not p: print(f"  - {n}: {d}")
print()
for n, p, _, d in results:
    if p and d: print(f"  {n}: {d}")
