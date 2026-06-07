"""TESM 多模态推理生成器

完整的多模态推理 pipeline:
- 增量推理 (每步 O(1))
- 多种采样策略 (top-k, top-p, temperature)
- 重复惩罚
- 流式输出
- 与 DataLoader 集成

使用方式:
    from tesm_ssm.inference.multimodal_generator import MultimodalGenerator
    generator = MultimodalGenerator(model)
    output = generator.generate(images=img, text_ids=txt, max_new_tokens=50)
"""

import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class MultimodalGenerator:
    """多模态生成器

    封装了高效的多模态推理 pipeline:
    1. 预填充: 处理多模态输入 (image/audio/text) 得到初始 hidden states
    2. 增量推理: 每步只处理新 token (O(1) 复杂度)
    3. 采样: top-k + top-p + temperature + repetition penalty
    4. 输出: 支持完整输出或流式输出

    Args:
        model: TESMMultimodalModel 或 TESMLMHeadModel
        device: 运行设备 (None 自动选择)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval().to(self.device)

    @torch.no_grad()
    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict]:
        """生成文本

        支持多模态输入，使用增量推理缓存加速。

        Args:
            text_ids: (B, L) 文本 prompt token IDs
            images: (B, 3, H, W) 图像张量
            audio: (B, T) 音频波形
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度 (0=greedy, >0=random)
            top_k: top-k 采样 (0=disable)
            top_p: nucleus 采样 (1.0=disable)
            repetition_penalty: 重复惩罚 (1.0=no penalty)
            eos_token_id: 结束 token ID
            pad_token_id: 填充 token ID
            use_cache: 使用增量推理缓存
            return_dict: 返回详细信息 (tokens, scores, timing)

        Returns:
            generated_ids: (B, L + new_tokens) 生成的完整序列
            或 dict (if return_dict=True)
        """
        # 设备对齐
        if text_ids is not None:
            text_ids = text_ids.to(self.device)
        if images is not None:
            images = images.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)

        # 获取多模态初始 embeddings
        initial_embeds, generated_ids = self._prefill(text_ids, images, audio)
        B = generated_ids.shape[0]

        if eos_token_id is None:
            tesm_cfg = getattr(self.model, 'tesm_config', None)
            if tesm_cfg is not None and hasattr(tesm_cfg, 'eos_token_id'):
                eos_token_id = tesm_cfg.eos_token_id
            else:
                eos_token_id = -1
        if eos_token_id is None:
            eos_token_id = -1

        # 增量推理缓存
        inference_params = None
        if use_cache and hasattr(self.model, 'decoder'):
            max_seqlen = initial_embeds.shape[1] + max_new_tokens
            cache = self.model.decoder.backbone.allocate_inference_cache(B, max_seqlen)
            inference_params = {'state_cache': cache}

        # 生成
        all_logits = []
        scores = []
        start_time = time.time()

        for step in range(max_new_tokens):
            # 前向传播
            if step == 0 or inference_params is None:
                # 第一次：完整前向
                logits = self._forward(initial_embeds, inference_params)
            else:
                # 增量：只传入新 token
                last_token_embed = self._get_token_embeds(generated_ids[:, -1:])
                logits = self._forward(last_token_embed, inference_params)

            next_logits = logits[:, -1, :]  # (B, V)

            # 重复惩罚
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, repetition_penalty
                )

            # 采样
            next_token, token_score = self._sample(
                next_logits, temperature, top_k, top_p
            )

            all_logits.append(next_logits)
            scores.append(token_score)

            # 拼接
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            # 检查 EOS
            if eos_token_id >= 0 and (next_token == eos_token_id).all():
                break

        elapsed = time.time() - start_time

        if return_dict:
            return {
                'sequences': generated_ids,
                'scores': torch.stack(scores) if scores else None,
                'logits': torch.stack(all_logits) if all_logits else None,
                'num_tokens': generated_ids.shape[1] - (text_ids.shape[1] if text_ids is not None else 0),
                'time': elapsed,
                'tokens_per_sec': (generated_ids.shape[1] - (text_ids.shape[1] if text_ids is not None else 0)) / max(elapsed, 1e-6),
            }

        return generated_ids

    @torch.no_grad()
    def stream_generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Iterator[Tuple[int, float]]:
        """流式生成

        每生成一个 token yield 一次，适合实时输出。

        Yields:
            (token_id, score): 生成的 token ID 和置信度
        """
        # 设备对齐
        if text_ids is not None:
            text_ids = text_ids.to(self.device)
        if images is not None:
            images = images.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)

        initial_embeds, generated_ids = self._prefill(text_ids, images, audio)
        B = generated_ids.shape[0]
        if B != 1:
            raise ValueError("stream_generate only supports batch_size=1")

        # 增量缓存
        max_seqlen = initial_embeds.shape[1] + max_new_tokens
        if hasattr(self.model, 'decoder'):
            cache = self.model.decoder.backbone.allocate_inference_cache(B, max_seqlen)
            inference_params = {'state_cache': cache}
        else:
            inference_params = None

        eos_token_id = eos_token_id or -1

        for step in range(max_new_tokens):
            if step == 0 or inference_params is None:
                logits = self._forward(initial_embeds, inference_params)
            else:
                last_embed = self._get_token_embeds(generated_ids[:, -1:])
                logits = self._forward(last_embed, inference_params)

            next_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, repetition_penalty
                )

            next_token, score = self._sample(next_logits, temperature, top_k, top_p)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            token_id = next_token.item()
            yield token_id, score.item()

            if eos_token_id >= 0 and token_id == eos_token_id:
                break

    def _prefill(
        self,
        text_ids: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预填充：处理多模态输入得到初始 embeddings

        Returns:
            initial_embeds: (B, N, D) 初始多模态 embeddings
            input_ids: (B, L) 输入 token IDs
        """
        embeds_list = []
        device = self.device

        # 使用 model 的多模态处理能力
        if hasattr(self.model, 'vision_embedder') and images is not None:
            vis_embeds = self.model.vision_embedder(images.to(device))
            embeds_list.append(vis_embeds)

        if hasattr(self.model, 'audio_embedder') and audio is not None:
            aud_embeds = self.model.audio_embedder(audio.to(device))
            embeds_list.append(aud_embeds)

        if text_ids is not None:
            txt_embeds = self._get_token_embeds(text_ids.to(device))
            embeds_list.append(txt_embeds)

        combined_embeds = torch.cat(embeds_list, dim=1)

        # 添加模态嵌入
        if hasattr(self.model, 'modality_embedding') and self.model.modality_embedding is not None:
            B = combined_embeds.shape[0]
            N = combined_embeds.shape[1]
            modality_ids = torch.zeros((B, N), dtype=torch.long, device=device)

            offset = 0
            if hasattr(self.model, 'vision_embedder') and images is not None:
                n = self.model.vision_embedder.num_output_tokens if hasattr(self.model.vision_embedder, 'num_output_tokens') else self.model.vision_embedder(images).shape[1]
                modality_ids[:, offset:offset+n] = 0  # VISION
                offset += n
            if hasattr(self.model, 'audio_embedder') and audio is not None:
                n = self.model.audio_embedder(audio).shape[1]
                modality_ids[:, offset:offset+n] = 1  # AUDIO
                offset += n
            if text_ids is not None:
                n = text_ids.shape[1]
                modality_ids[:, offset:offset+n] = 2  # TEXT

            combined_embeds = combined_embeds + self.model.modality_embedding(modality_ids)

        if text_ids is None:
            text_ids = torch.zeros((combined_embeds.shape[0], 0), dtype=torch.long, device=device)

        return combined_embeds, text_ids

    def _forward(self, embeds: torch.Tensor, inference_params=None) -> torch.Tensor:
        """通过模型前向传播 (使用 embeddings)"""
        if hasattr(self.model, 'decoder'):
            hidden, _, _, _ = self.model.decoder.backbone.forward_with_embeds(
                embeds, inference_params=inference_params
            )
            logits = self.model.decoder.lm_head(hidden)
        else:
            # 纯文本模型 fallback
            outputs, _ = self.model(inputs_embeds=embeds)
            logits = outputs.logits
        return logits

    def _get_token_embeds(self, token_ids: torch.Tensor) -> torch.Tensor:
        """获取 token embeddings"""
        if hasattr(self.model, 'text_embedding'):
            embeds = self.model.text_embedding(token_ids)
            if hasattr(self.model, 'modality_embedding') and self.model.modality_embedding is not None:
                mod_ids = torch.full(token_ids.shape, 2, dtype=torch.long, device=token_ids.device)
                embeds = embeds + self.model.modality_embedding(mod_ids)
            return embeds
        elif hasattr(self.model, 'backbone'):
            return self.model.backbone.embedding(token_ids)
        else:
            return self.model.get_input_embeddings()(token_ids)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样下一个 token

        Args:
            logits: (B, V) 未归一化的 logits
            temperature: 温度
            top_k: top-k 阈值
            top_p: nucleus 阈值

        Returns:
            (token, score): 采样的 token ID 和置信度
        """
        B, V = logits.shape

        if temperature <= 0:
            # Greedy
            probs = F.softmax(logits, dim=-1)
            token = logits.argmax(dim=-1)
            score = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
            return token, score

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, V))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # 至少保留一个
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        score = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)

        return token, score

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚

        降低已生成 token 的 logits，减少重复。
        """
        if penalty == 1.0:
            return logits

        score = torch.gather(logits, 1, generated_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, generated_ids, score)
        return logits

    def benchmark(
        self,
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        warmup: int = 3,
        runs: int = 10,
    ) -> Dict[str, float]:
        """推理速度基准测试

        Returns:
            dict: tokens/sec, avg_latency, first_token_latency
        """
        # Warmup
        for _ in range(warmup):
            self.generate(text_ids, images, audio, max_new_tokens=8)

        # Benchmark
        times = []
        for _ in range(runs):
            start = time.time()
            self.generate(text_ids, images, audio, max_new_tokens=max_new_tokens)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        return {
            'avg_time': avg_time,
            'tokens_per_sec': max_new_tokens / avg_time,
            'first_token_latency': min(times),  # approximate
            'min_time': min(times),
            'max_time': max(times),
        }

    def __repr__(self):
        return f"MultimodalGenerator(model={type(self.model).__name__}, device={self.device})"
