"""TESM 多模态组合模型

将各模态的 Embedder 与 TESM Decoder 组合在一起。
纯文本用户不需要使用此类，直接使用 TESMLMHeadModel 即可。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.mixer_seq_simple import TESMLMHeadModel, TESMCausalLMOutput
from tesm_ssm.modules.multimodal.base_embedder import BaseEmbedder
from tesm_ssm.modules.multimodal.vision_embedder import VisionEmbedder
from tesm_ssm.modules.multimodal.audio_embedder import AudioEmbedder

from .config_multimodal import MultimodalConfig


# 模态 ID 常量
VISION = 0
AUDIO = 1
TEXT = 2


class TESMMultimodalModel(nn.Module):
    """TESM 多模态组合模型

    由可选的 Embedder（视觉/音频）和 TESM Decoder 组合而成。
    纯文本用户不需要使用此类。

    Args:
        config: MultimodalConfig 配置
        vision_embedder: 自定义视觉 Embedder（可选）
        audio_embedder: 自定义音频 Embedder（可选）

    Example:
        >>> from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
        >>> config = MultimodalConfig.from_tesm_config(TESMConfig.small())
        >>> config.vision_enabled = True
        >>> model = TESMMultimodalModel(config)
        >>> 
        >>> # 图像 + 文本
        >>> images = torch.randn(2, 3, 224, 224)
        >>> text_ids = torch.randint(0, 1000, (2, 32))
        >>> output, _ = model(images=images, text_ids=text_ids)
    """

    def __init__(
        self,
        config: MultimodalConfig,
        vision_embedder: Optional[BaseEmbedder] = None,
        audio_embedder: Optional[BaseEmbedder] = None,
    ):
        super().__init__()
        self.config = config
        self.tesm_config = config.tesm
        d_model = self.tesm_config.d_model

        # 文本 Embedding（始终存在）
        self.text_embedding = nn.Embedding(
            self.tesm_config.vocab_size, d_model
        )

        # 可选模态 Embedder
        if config.vision_enabled:
            if vision_embedder is not None:
                self.vision_embedder = vision_embedder
            else:
                self.vision_embedder = VisionEmbedder(
                    d_model=d_model,
                    patch_size=config.vision_patch_size,
                    num_output_tokens=config.vision_num_tokens,
                    in_channels=config.vision_in_channels,
                    max_image_size=config.vision_max_image_size,
                    use_norm=config.vision_use_norm,
                )
        else:
            self.vision_embedder = None

        if config.audio_enabled:
            if audio_embedder is not None:
                self.audio_embedder = audio_embedder
            else:
                self.audio_embedder = AudioEmbedder(
                    d_model=d_model,
                    sample_rate=config.audio_sample_rate,
                    frame_duration_ms=config.audio_frame_duration_ms,
                    use_norm=config.audio_use_norm,
                )
        else:
            self.audio_embedder = None

        # 模态类型嵌入（可学习）
        if config.use_modality_embedding:
            num_modalities = 3  # vision, audio, text
            self.modality_embedding = nn.Embedding(num_modalities, d_model)
        else:
            self.modality_embedding = None

        # TESM Decoder
        self.decoder = TESMLMHeadModel(self.tesm_config)

        # 将多模态的 text_embedding 共享给 decoder
        self.decoder.backbone.embedding = self.text_embedding
        if self.tesm_config.tie_embeddings:
            self.decoder.lm_head.weight = self.text_embedding.weight

        # 冻结 Embedder（如果配置要求）
        if config.freeze_embedders:
            self._freeze_embedders()

    def _freeze_embedders(self):
        """冻结所有 Embedder 参数"""
        for name, param in self.named_parameters():
            if "embedder" in name and "decoder" not in name:
                param.requires_grad = False

    def _embed_modality(
        self,
        data: Optional[torch.Tensor],
        embedder: Optional[BaseEmbedder],
        modality_id: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """嵌入单个模态

        Args:
            data: 原始数据
            embedder: Embedder 模块
            modality_id: 模态 ID

        Returns:
            (embeds, modality_ids) 或 (None, None)
        """
        if data is None or embedder is None:
            return None, None

        embeds = embedder(data)  # (B, N, d_model)
        B, N, _ = embeds.shape
        modality_ids = torch.full(
            (B, N), modality_id, dtype=torch.long, device=embeds.device
        )

        return embeds, modality_ids

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[TESMCausalLMOutput, list]:
        """多模态前向传播

        Args:
            text_ids: (B, L) 文本 token IDs
            images: (B, C, H, W) 图像
            audio: (B, T) 原始音频波形 @ 16kHz
            labels: (B, L) 训练标签（只对文本部分计算损失）
            inference_params: 增量推理参数

        Returns:
            (TESMCausalLMOutput, final_states)
        """
        embeds_list = []
        modality_ids_list = []

        # Vision
        vis_embeds, vis_ids = self._embed_modality(
            images, self.vision_embedder, VISION
        )
        if vis_embeds is not None:
            embeds_list.append(vis_embeds)
            modality_ids_list.append(vis_ids)

        # Audio
        aud_embeds, aud_ids = self._embed_modality(
            audio, self.audio_embedder, AUDIO
        )
        if aud_embeds is not None:
            embeds_list.append(aud_embeds)
            modality_ids_list.append(aud_ids)

        # Text
        if text_ids is not None:
            txt_embeds = self.text_embedding(text_ids)  # (B, L, d_model)
            embeds_list.append(txt_embeds)
            txt_ids = torch.full(
                (text_ids.shape[0], text_ids.shape[1]),
                TEXT,
                dtype=torch.long,
                device=txt_embeds.device,
            )
            modality_ids_list.append(txt_ids)

        # 拼接所有模态
        combined_embeds = torch.cat(embeds_list, dim=1)  # (B, total_N, d_model)

        # 添加模态类型嵌入
        if self.modality_embedding is not None:
            all_modality_ids = torch.cat(modality_ids_list, dim=1)  # (B, total_N)
            modality_embeds = self.modality_embedding(all_modality_ids)
            combined_embeds = combined_embeds + modality_embeds

        # 传入 decoder（使用 inputs_embeds）
        # 需要绕过 decoder 的 embedding 层
        output, final_states = self._forward_decoder(
            inputs_embeds=combined_embeds,
            labels=labels,
            inference_params=inference_params,
            **kwargs,
        )

        return output, final_states

    def _forward_decoder(
        self,
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[TESMCausalLMOutput, list]:
        """通过 decoder 前向传播（使用 pre-computed embeddings）

        直接调用 decoder backbone，绕过 embedding 层。
        """
        # 使用 decoder 的 backbone 直接处理 embeddings
        hidden_states, ent_maps, ent_stats, final_states = self.decoder.backbone.forward_with_embeds(
            inputs_embeds=inputs_embeds,
            inference_params=inference_params,
            **kwargs,
        )

        # LM Head
        logits = self.decoder.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            # 找到文本部分的位置（labels 只对文本有效）
            loss = self._compute_multimodal_loss(logits, labels, inputs_embeds.shape[1])

        output = TESMCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            entanglement_maps=ent_maps,
            entanglement_stats=ent_stats,
        )

        return output, final_states

    def _compute_multimodal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        total_seq_len: int,
    ) -> torch.Tensor:
        """计算多模态损失

        labels 只对文本部分有效，其他位置设为 ignore_index。
        对齐方式: labels 对应 logits 的最后 L_labels 个位置（文本部分）。
        """
        B, L_logits, V = logits.shape
        B_labels, L_labels = labels.shape

        # 构建与 logits 对齐的 labels
        # labels 只对应文本部分（logits 的最后 L_labels 个位置）
        aligned_labels = torch.full(
            (B, L_logits), self.tesm_config.label_ignore_index,
            dtype=torch.long, device=logits.device
        )
        
        # 将 labels 放入对齐后的最后 L_labels 个位置
        if L_logits >= L_labels:
            aligned_labels[:, -L_labels:] = labels
        else:
            aligned_labels = labels[:, :L_logits]

        # 标准的 shift 计算
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = aligned_labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tesm_config.label_ignore_index,
        )

        return loss

    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """多模态生成

        先处理多模态输入得到初始 embeddings，然后自回归生成文本。

        Args:
            text_ids: 初始文本 prompt
            images: 图像
            audio: 音频
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样

        Returns:
            生成的 token IDs (包含 prompt)
        """
        self.eval()

        # 获取多模态 embeddings
        with torch.no_grad():
            embeds_list = []

            if images is not None and self.vision_embedder is not None:
                vis_embeds = self.vision_embedder(images)
                embeds_list.append(vis_embeds)

            if audio is not None and self.audio_embedder is not None:
                aud_embeds = self.audio_embedder(audio)
                embeds_list.append(aud_embeds)

            if text_ids is not None:
                txt_embeds = self.text_embedding(text_ids)
                embeds_list.append(txt_embeds)

            combined_embeds = torch.cat(embeds_list, dim=1)

            # 添加模态嵌入
            if self.modality_embedding is not None:
                N_vis = self.vision_embedder.num_output_tokens if images is not None else 0
                N_aud = aud_embeds.shape[1] if audio is not None else 0
                N_txt = text_ids.shape[1] if text_ids is not None else 0
                modality_ids = []
                if N_vis > 0:
                    modality_ids.append(torch.full((1, N_vis), VISION, dtype=torch.long, device=combined_embeds.device))
                if N_aud > 0:
                    modality_ids.append(torch.full((1, N_aud), AUDIO, dtype=torch.long, device=combined_embeds.device))
                if N_txt > 0:
                    modality_ids.append(torch.full((1, N_txt), TEXT, dtype=torch.long, device=combined_embeds.device))
                all_modality_ids = torch.cat(modality_ids, dim=1)
                combined_embeds = combined_embeds + self.modality_embedding(all_modality_ids)

            B = combined_embeds.shape[0]

        # 自回归生成
        generated_embeds = combined_embeds
        generated_ids = text_ids if text_ids is not None else torch.zeros((B, 0), dtype=torch.long, device=combined_embeds.device)

        eos_token_id = eos_token_id or self.tesm_config.eos_token_id

        for _ in range(max_new_tokens):
            # 前向传播
            with torch.no_grad():
                # 重新拼接 embeddings
                if generated_ids.shape[1] > (text_ids.shape[1] if text_ids is not None else 0):
                    # 已有生成的 token，需要嵌入
                    new_text_embeds = self.text_embedding(generated_ids[:, -1:])
                    if self.modality_embedding is not None:
                        new_modality_ids = torch.full((B, 1), TEXT, dtype=torch.long, device=new_text_embeds.device)
                        new_text_embeds = new_text_embeds + self.modality_embedding(new_modality_ids)
                    all_embeds = torch.cat([generated_embeds, new_text_embeds], dim=1)
                else:
                    all_embeds = generated_embeds

                hidden_states, _, _, _ = self.decoder.backbone.forward_with_embeds(all_embeds)
                logits = self.decoder.lm_head(hidden_states[:, -1:, :])

            # 采样
            logits = logits.squeeze(1)
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, num_samples=1)).squeeze(-1)
                else:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            # 扩展 embeddings 用于下一步
            new_embed = self.text_embedding(next_token.unsqueeze(1))
            if self.modality_embedding is not None:
                new_mod_ids = torch.full((B, 1), TEXT, dtype=torch.long, device=new_embed.device)
                new_embed = new_embed + self.modality_embedding(new_mod_ids)
            generated_embeds = torch.cat([generated_embeds, new_embed], dim=1)

            # 检查 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated_ids

    def get_param_count(self) -> Dict[str, int]:
        """获取各组件参数量"""
        result = {"decoder": sum(p.numel() for p in self.decoder.parameters())}
        if self.vision_embedder is not None:
            result["vision_embedder"] = sum(p.numel() for p in self.vision_embedder.parameters())
        if self.audio_embedder is not None:
            result["audio_embedder"] = sum(p.numel() for p in self.audio_embedder.parameters())
        result["text_embedding"] = sum(p.numel() for p in self.text_embedding.parameters())
        if self.modality_embedding is not None:
            result["modality_embedding"] = sum(p.numel() for p in self.modality_embedding.parameters())
        result["total"] = sum(p.numel() for p in self.parameters())
        return result

    def __repr__(self):
        parts = ["TESMMultimodalModel("]
        parts.append(f"  decoder: {sum(p.numel() for p in self.decoder.parameters()) / 1e6:.1f}M")
        if self.vision_embedder is not None:
            parts.append(f"  vision: {sum(p.numel() for p in self.vision_embedder.parameters()) / 1e6:.1f}M")
        if self.audio_embedder is not None:
            parts.append(f"  audio: {sum(p.numel() for p in self.audio_embedder.parameters()) / 1e6:.1f}M")
        parts.append(f"  total: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M")
        parts.append(")")
        return "\n".join(parts)
