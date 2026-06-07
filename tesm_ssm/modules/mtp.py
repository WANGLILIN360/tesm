"""MTP (Multi-Token Prediction) 多 token 预测模块

Gemma4 12B 风格的推测解码实现:
- 轻量级单层 dense MTP head
- 同时预测未来 N 个 token
- 与 target model 共享 embeddings
- 推测解码: drafter 生成候选 -> target 并行验证

参考: Google Gemma4 Technical Report Section 4.2
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Multi-Token Prediction Head

    轻量级单层线性头，用于预测未来 token。
    共享 target model 的 input embeddings。

    Args:
        d_model: 模型维度
        vocab_size: 词表大小
        num_pred_tokens: 预测未来 token 数量 (默认 4，匹配 Gemma4)
    """

    def __init__(self, d_model: int, vocab_size: int, num_pred_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_pred_tokens = num_pred_tokens

        # N 个独立的投影层，每个预测一个未来位置
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(num_pred_tokens)
        ])

        # 输出头 (与 target model 共享 embedding 时不用)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: Optional[nn.Module] = None,
    ) -> List[torch.Tensor]:
        """预测未来 N 个 token 的 logits

        Args:
            hidden_states: (B, L, D) target model 的隐藏状态
            input_embeddings: 共享的 input embedding 层 (用于 tying)

        Returns:
            list of N tensors, each (B, L, vocab_size)
        """
        B, L, D = hidden_states.shape
        logits_list = []

        for i, proj in enumerate(self.projections):
            # 投影: (B, L, D) -> (B, L, D)
            pred_hidden = proj(hidden_states)

            # 计算 logits
            if input_embeddings is not None:
                # 使用共享 embedding 的转置作为输出层 (weight tying)
                logits = torch.matmul(pred_hidden, input_embeddings.weight.t())
            else:
                logits = self.output(pred_hidden)

            logits_list.append(logits)

        return logits_list


class SpeculativeDecoder:
    """推测解码器

    使用 MTP head 作为 drafter，target model 作为 verifier。
    流程:
    1. Drafter (MTP) 快速生成 N 个候选 token
    2. Target model 并行验证所有候选
    3. 接受匹配的 token，从第一个不匹配处重新生成

    Args:
        target_model: 目标模型 (大模型)
        mtp_head: MTP head
        num_speculative_tokens: 推测 token 数
        temperature: 采样温度
        top_k: top-k 采样
    """

    def __init__(
        self,
        target_model: nn.Module,
        mtp_head: MTPHead,
        num_speculative_tokens: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        self.target_model = target_model
        self.mtp_head = mtp_head
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        """使用推测解码生成序列

        Args:
            input_ids: (B, L) 输入 token IDs
            max_new_tokens: 最大生成 token 数

        Returns:
            generated_ids: (B, L + max_new_tokens) 生成的完整序列
        """
        self.target_model.eval()
        self.mtp_head.eval()

        generated = input_ids.clone()
        B = generated.shape[0]
        device = generated.device

        num_generated = 0
        while num_generated < max_new_tokens:
            # Step 1: 获取 target model 的当前隐藏状态
            with torch.no_grad():
                outputs, _ = self.target_model(generated)
                hidden = outputs.hidden_states  # (B, L, D)
                last_hidden = hidden[:, -1:, :]  # (B, 1, D)

            # Step 2: MTP drafter 快速生成候选 token
            draft_tokens = self._draft_tokens(last_hidden, self.num_speculative_tokens)
            # draft_tokens: list of (B, 1) token IDs

            # Step 3: 构建完整的候选序列
            candidate_ids = torch.cat([generated] + draft_tokens, dim=1)

            # Step 4: Target model 并行验证
            with torch.no_grad():
                verify_outputs, _ = self.target_model(candidate_ids)
                verify_logits = verify_outputs.logits  # (B, L+N, V)

            # Step 5: 接受/拒绝验证
            accepted = self._verify_tokens(
                generated, draft_tokens, verify_logits
            )

            # Step 6: 扩展生成序列
            generated = torch.cat([generated, accepted.unsqueeze(1)], dim=1)
            num_generated += accepted.shape[1]

            # 检查 EOS
            if (accepted == kwargs.get('eos_token_id', -1)).any():
                break

        return generated

    def _draft_tokens(
        self,
        hidden_state: torch.Tensor,
        num_tokens: int,
    ) -> List[torch.Tensor]:
        """MTP drafter 生成候选 token

        Args:
            hidden_state: (B, 1, D) 当前隐藏状态
            num_tokens: 生成 token 数

        Returns:
            list of (B, 1) token IDs
        """
        draft_tokens = []
        current_hidden = hidden_state

        for i in range(min(num_tokens, self.num_speculative_tokens)):
            # 使用第 i 个 projection
            if i < len(self.mtp_head.projections):
                proj = self.mtp_head.projections[i]
                pred_hidden = proj(current_hidden)

                # 采样
                logits = self.mtp_head.output(pred_hidden)
                logits = logits / max(self.temperature, 1e-6)

                if self.top_k > 0:
                    probs = F.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(
                        probs, min(self.top_k, probs.size(-1))
                    )
                    next_token = top_k_indices.gather(
                        -1, torch.multinomial(top_k_probs.squeeze(1), num_samples=1)
                    ).unsqueeze(-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs.squeeze(1), num_samples=1).unsqueeze(-1)

                draft_tokens.append(next_token)

                # 更新 hidden state (简化: 用 embedding 反馈)
                # 实际应该通过 model forward 获取新 hidden state
                # 这里用投影近似
                current_hidden = pred_hidden

        return draft_tokens

    def _verify_tokens(
        self,
        prefix_ids: torch.Tensor,
        draft_tokens: List[torch.Tensor],
        verify_logits: torch.Tensor,
    ) -> torch.Tensor:
        """验证候选 token

        从 target model 的 logits 中采样，与 draft 比较。
        接受匹配的 token，从第一个不匹配处重新生成。

        Args:
            prefix_ids: (B, L) 已验证的前缀
            draft_tokens: list of (B, 1) draft token
            verify_logits: (B, L+N, V) target model 的 logits

        Returns:
            accepted: (B, num_accepted) 接受的 token
        """
        B = prefix_ids.shape[0]
        prefix_len = prefix_ids.shape[1]

        # 获取 target model 在 draft 位置的 logits
        draft_logits = verify_logits[:, prefix_len-1:prefix_len-1+len(draft_tokens), :]

        # 贪婪采样获取 target model 的预测
        target_tokens = draft_logits.argmax(dim=-1)  # (B, N)

        # 将 draft tokens 堆叠
        draft_stacked = torch.cat(draft_tokens, dim=1)  # (B, N)

        # 比较: 找到第一个不匹配的位置
        matches = (target_tokens == draft_stacked)  # (B, N)

        # 每个 batch 独立接受
        accepted_tokens = []
        for b in range(B):
            match_row = matches[b]
            if match_row.all():
                # 全部接受
                accepted_tokens.append(draft_stacked[b])
            else:
                # 找到第一个不匹配
                first_mismatch = (~match_row).nonzero(as_tuple=True)[0][0].item()
                if first_mismatch == 0:
                    # 第一个就不匹配，只接受 target 的预测
                    accepted_tokens.append(target_tokens[b, :1])
                else:
                    # 接受匹配的部分 + target 在不匹配处的预测
                    accepted = torch.cat([
                        draft_stacked[b, :first_mismatch],
                        target_tokens[b, first_mismatch:first_mismatch+1],
                    ])
                    accepted_tokens.append(accepted)

        # 找到最短的接受序列
        min_len = min(len(t) for t in accepted_tokens)
        accepted = torch.stack([t[:min_len] for t in accepted_tokens])

        return accepted
