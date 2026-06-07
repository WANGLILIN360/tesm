"""Thinking Mode - 结构化推理模式

Gemma4 12B 风格的链式思维推理:
- 通过 <|think|> 触发
- 推理内容包裹在 <|channel>thought\n...<channel|> 中
- 解析器提取 thinking 内容和最终答案
- 支持启用/禁用切换

参考: Gemma4 vLLM 文档 Thinking/Reasoning Mode
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# 特殊 token 常量 (匹配 Gemma4)
THINK_TRIGGER = "<|think|>"
THOUGHT_START = "<|channel>thought\n"
THOUGHT_END = "<channel|>"


@dataclass
class ThinkingOutput:
    """Thinking Mode 输出结构"""
    reasoning: str  # 推理过程（思考链）
    answer: str     # 最终答案
    raw_text: str   # 原始完整文本


class ThinkingModeController:
    """Thinking Mode 控制器

    管理模型的结构化推理流程:
    1. 通过 system prompt 中的特殊 token 启用/禁用
    2. 解析模型输出的 thinking 分隔符
    3. 分离推理过程和最终答案

    Example:
        >>> controller = ThinkingModeController()
        >>> # 启用 thinking
        >>> prompt = controller.add_thinking_trigger(system_prompt)
        >>> # 生成
        >>> output = model.generate(prompt)
        >>> # 解析
        >>> result = controller.parse_output(output)
        >>> print(result.reasoning)  # 推理过程
        >>> print(result.answer)     # 最终答案
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def enable(self):
        """启用 Thinking Mode"""
        self.enabled = True

    def disable(self):
        """禁用 Thinking Mode"""
        self.enabled = False

    def add_thinking_trigger(self, system_prompt: str) -> str:
        """在 system prompt 中添加 thinking 触发器

        Args:
            system_prompt: 原始 system prompt

        Returns:
            添加了 thinking trigger 的 prompt
        """
        if not self.enabled:
            return system_prompt

        if THINK_TRIGGER not in system_prompt:
            system_prompt = f"{THINK_TRIGGER}\n{system_prompt}"

        return system_prompt

    def parse_output(self, text: str) -> ThinkingOutput:
        """解析模型输出，分离推理过程和答案

        处理格式:
            <|channel>thought\n[推理内容]<channel|>[最终答案]

        Args:
            text: 模型生成的完整文本

        Returns:
            ThinkingOutput (reasoning, answer, raw_text)
        """
        raw_text = text

        # 查找 thought 分隔符
        start_idx = text.find(THOUGHT_START)
        end_idx = text.find(THOUGHT_END)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            # 提取推理内容
            reasoning = text[start_idx + len(THOUGHT_START):end_idx].strip()
            # 提取最终答案（thought 结束后的内容）
            answer = text[end_idx + len(THOUGHT_END):].strip()
        else:
            # 没有找到分隔符，全部作为答案
            reasoning = ""
            answer = text.strip()

        return ThinkingOutput(
            reasoning=reasoning,
            answer=answer,
            raw_text=raw_text,
        )

    def format_training_example(
        self,
        question: str,
        reasoning: str,
        answer: str,
    ) -> str:
        """格式化训练样本（用于 SFT 训练 thinking 能力）

        Args:
            question: 问题
            reasoning: 推理过程
            answer: 最终答案

        Returns:
            格式化后的训练文本
        """
        return (
            f"Question: {question}\n\n"
            f"{THOUGHT_START}{reasoning}{THOUGHT_END}"
            f"{answer}"
        )

    def is_thinking_content(self, text: str) -> bool:
        """检查文本是否包含 thinking 内容"""
        return THOUGHT_START in text and THOUGHT_END in text

    def extract_thinking_only(self, text: str) -> str:
        """只提取 thinking 内容（不含分隔符）"""
        start_idx = text.find(THOUGHT_START)
        end_idx = text.find(THOUGHT_END)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx + len(THOUGHT_START):end_idx].strip()
        return ""

    def strip_thinking(self, text: str) -> str:
        """移除 thinking 内容，只保留答案"""
        result = self.parse_output(text)
        return result.answer


class MTPWithThinking(nn.Module):
    """MTP + Thinking Mode 的组合模块

    在推测解码的同时支持 thinking mode 输出。
    当 thinking 启用时，MTP drafter 会优先预测 thought 分隔符内的 token。
    """

    def __init__(
        self,
        mtp_head: "MTPHead",
        thinking_controller: ThinkingModeController,
    ):
        super().__init__()
        self.mtp_head = mtp_head
        self.thinking = thinking_controller

    def get_special_token_bias(self) -> Optional[torch.Tensor]:
        """获取特殊 token 的 logits 偏置

        当 thinking 启用时，提高 thought 分隔符的预测概率。
        """
        if not self.thinking.enabled:
            return None

        # 在实际实现中，这里会返回一个 vocab_size 的偏置张量
        # 提高 <|channel>thought 相关 token 的概率
        return None

    def should_stop_thinking(self, generated_text: str) -> bool:
        """判断 thinking 过程是否应该结束"""
        if not self.thinking.enabled:
            return True

        # 如果已经生成了 thought 结束标记，则 thinking 结束
        return THOUGHT_END in generated_text
