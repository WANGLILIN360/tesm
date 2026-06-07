# TESM 多模态架构设计：SDK层 vs 应用层

**核心问题**：多模态 Embedder 应该放在 SDK 内部（架构层）还是让用户自己实现（应用层）？

---

## 1. TESM 的当前定位

```
tesm_ssm/                          ← SDK 包（类似 transformers）
├── models/                        ← 模型定义
│   ├── config_tesm.py             ← TESMConfig（纯文本配置）
│   └── mixer_seq_simple.py        ← TESMLMHeadModel（纯文本模型）
├── modules/                       ← 核心模块
│   ├── tesm.py                    ← TESM_SISO（SSM核心）
│   ├── tesm_mimo.py               ← TESMMIMO（多头版本）
│   ├── block.py                   ← Block + RMSNorm
│   └── [multimodal/]              ← ← ← 新增 Embedder 模块
├── ops/                           ← 内核后端
│   ├── cuda/                      ← CUDA kernels
│   ├── triton/                    ← Triton kernels
│   └── tilelang/                  ← TileLang kernels
├── training/                      ← 训练工具
└── utils/                         ← 工具函数（量化、缓存等）
```

**TESM 是 SDK，不是应用**。类比：
- `tesm_ssm` ≈ `transformers`（库/SDK层）
- 用户的应用 ≈ `chatgpt` / `claude`（应用层）

---

## 2. 三种方案的对比

### 方案A：纯应用层（SDK不做多模态）

```python
# 用户代码
from tesm_ssm import TESMLMHeadModel, TESMConfig
from PIL import Image
import torch

# 用户自己实现 Vision Embedder
class MyVisionEmbedder(torch.nn.Module):
    def forward(self, images):
        ... # 自己处理图像 → tokens

# 用户自己拼接多模态序列
model = TESMLMHeadModel(TESMConfig(...))
vision_embedder = MyVisionEmbedder()

images = load_images(...)  # 用户自己加载
text_tokens = tokenizer(...)

# 用户自己拼接：image_tokens + text_tokens
image_embeds = vision_embedder(images)
text_embeds = model.backbone.embedding(text_tokens)
combined = torch.cat([image_embeds, text_embeds], dim=1)

# 直接传入 decoder hidden_states（绕过 embedding 层）
# ⚠️ 但 TESMLMHeadModel 不支持直接传 hidden_states！
```

**优点**：
- SDK 保持纯粹，只做 SSM

**缺点**：
- ❌ TESMLMHeadModel 不支持直接传入 pre-computed embeddings
- ❌ 每个用户都要重复实现 VisionEmbedder
- ❌ 量化工具（INT2/INT8）无法应用到 Embedder
- ❌ 训练 pipeline（Trainer）不支持多模态数据
- ❌ 增量推理缓存需要用户自己管理模态边界

**结论**：❌ 不可行。SDK 当前 API 不支持这种用法。

---

### 方案B：纯架构层（SDK内置端到端多模态）

```python
# SDK 内部实现
class TESMMultimodalModel(torch.nn.Module):
    def __init__(self, config):
        self.vision_embedder = VisionEmbedder(...)
        self.audio_embedder = AudioEmbedder(...)
        self.text_embedding = nn.Embedding(...)
        self.decoder = TESMLMHeadModel(config)  # 复用
    
    def forward(self, images=None, audio=None, text_ids=None):
        embeds = []
        if images is not None:
            embeds.append(self.vision_embedder(images))
        if audio is not None:
            embeds.append(self.audio_embedder(audio))
        if text_ids is not None:
            embeds.append(self.text_embedding(text_ids))
        combined = torch.cat(embeds, dim=1)
        return self.decoder(inputs_embeds=combined)
```

**优点**：
- ✅ 开箱即用
- ✅ 量化/训练/推理 pipeline 统一

**缺点**：
- ❌ SDK 变重，引入了多模态依赖（PIL, librosa 等）
- ❌ 强制所有用户都安装多模态依赖，即使只用纯文本
- ❌ 不够灵活，用户无法自定义 Embedder

**结论**：⚠️ 可行但不优雅。SDK 不应该强制所有用户承担多模态依赖。

---

### 方案C：混合方案（推荐）✅

**核心原则**：SDK 提供可组合的 building blocks（积木），不强制的端到端。

```
tesm_ssm/
├── models/
│   ├── config_tesm.py             ← 纯文本配置（不变）
│   ├── mixer_seq_simple.py        ← 纯文本模型（不变）
│   └── multimodal/                ← 新增：多模态模型（可选导入）
│       ├── config_multimodal.py   ← 多模态配置
│       ├── model_multimodal.py    ← TESMMultimodalModel
│       └── pipeline.py            ← 多模态推理 pipeline
├── modules/
│   ├── tesm.py                    ← SSM 核心（不变）
│   ├── tesm_mimo.py               ← MIMO（不变）
│   ├── block.py                   ← Block（不变）
│   └── multimodal/                ← 新增：多模态模块（可选导入）
│       ├── vision_embedder.py     ← VisionEmbedder
│       ├── audio_embedder.py      ← AudioEmbedder
│       └── base_embedder.py       ← 抽象基类
```

**关键设计**：

1. **optional import**：多模态是可选的，不强制依赖
2. **composition over inheritance**：Embedder 和 Decoder 是组合关系
3. **统一接口**：所有 Embedder 继承同一个基类

```python
# SDK 提供（可选使用）
from tesm_ssm.modules.multimodal import VisionEmbedder, AudioEmbedder
from tesm_ssm.models.multimodal import TESMMultimodalModel

# 开箱即用
model = TESMMultimodalModel(config)
output = model(images=images, audio=audio, text_ids=text_ids)

# 也可以只用纯文本（不受影响）
from tesm_ssm import TESMLMHeadModel  # 原来的方式
model = TESMLMHeadModel(config)
```

**优点**：
- ✅ 纯文本用户不受影响（完全不导入多模态模块）
- ✅ 多模态用户开箱即用
- ✅ 量化工具可以应用到 Embedder（因为是 SDK 的一部分）
- ✅ 训练 pipeline 可以扩展支持多模态数据
- ✅ 用户可以自定义 Embedder（继承基类即可）
- ✅ 符合 SDK 的 building block 哲学

---

## 3. 具体实现设计

### 3.1 BaseEmbedder 抽象接口

```python
# tesm_ssm/modules/multimodal/base_embedder.py

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEmbedder(nn.Module, ABC):
    """多模态 Embedder 抽象基类
    
    所有模态的 Embedder 必须实现此接口，确保与 TESM decoder 的兼容性。
    """
    
    @abstractmethod
    def output_dim(self) -> int:
        """输出维度，必须等于 TESM config.d_model"""
        pass
    
    @abstractmethod
    def num_tokens(self, input_shape) -> int:
        """给定输入后输出多少个 token"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """输入: 原始数据（图像/音频/视频）
        输出: (batch, num_tokens, d_model) 的 embedding"""
        pass
```

### 3.2 VisionEmbedder（无编码器视觉）

```python
# tesm_ssm/modules/multimodal/vision_embedder.py

class VisionEmbedder(BaseEmbedder):
    """Gemma4-style 无编码器视觉嵌入
    
    ~35M 参数，替代传统的 550M Vision Encoder
    
    处理流程:
        Image (B, 3, H, W) 
        → 48x48 Patches 
        → Flatten (B, N, 6912) 
        → Linear Proj (B, N, d_model)
        → + Positional Embedding (factorized X/Y)
        → Adaptive Pool (B, 280, d_model)
    """
    
    def __init__(self, d_model=3840, patch_size=48, num_output_tokens=280):
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, d_model)
        # 因子化坐标位置编码
        grid_size = 1024 // patch_size  # 21
        self.pos_x = nn.Embedding(grid_size, d_model // 2)
        self.pos_y = nn.Embedding(grid_size, d_model // 2)
        self.norm = RMSNorm(d_model)
        self.num_output_tokens = num_output_tokens
        
    def forward(self, images):
        # images: (B, 3, H, W)
        B = images.shape[0]
        patches = images.unfold(2, 48, 48).unfold(3, 48, 48)
        # (B, 3, H//48, W//48, 48, 48)
        N = patches.shape[2] * patches.shape[3]
        patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(B, N, -1)
        # (B, N, 6912)
        
        # 线性投影
        embeds = self.patch_proj(patches)  # (B, N, d_model)
        
        # 添加位置编码
        pos_x = self.pos_x(torch.arange(patches.shape[2], device=images.device))
        pos_y = self.pos_y(torch.arange(patches.shape[3], device=images.device))
        # ... (位置编码计算)
        
        embeds = self.norm(embeds)
        
        # Pool 到固定 token 数
        embeds = embeds.permute(0, 2, 1)  # (B, d_model, N)
        embeds = F.adaptive_avg_pool1d(embeds, self.num_output_tokens)
        embeds = embeds.permute(0, 2, 1)  # (B, 280, d_model)
        
        return embeds
```

### 3.3 AudioEmbedder（无编码器音频）

```python
# tesm_ssm/modules/multimodal/audio_embedder.py

class AudioEmbedder(BaseEmbedder):
    """Gemma4-style 无编码器音频嵌入
    
    ~1M 参数
    
    处理流程:
        Raw Audio (B, T) @ 16kHz
        → 40ms Frames (640 samples each)
        → Linear Projection (B, N, d_model)
    """
    
    def __init__(self, d_model=3840, sample_rate=16000, frame_duration_ms=40):
        frame_size = sample_rate * frame_duration_ms // 1000  # 640
        self.frame_proj = nn.Linear(frame_size, d_model)
        self.norm = RMSNorm(d_model)
        
    def forward(self, audio):
        # audio: (B, T) raw waveform
        B, T = audio.shape
        frame_size = 640
        # 切分成 40ms 帧
        num_frames = T // frame_size
        audio = audio[:, :num_frames * frame_size]
        frames = audio.reshape(B, num_frames, frame_size)
        # (B, N, 640)
        
        embeds = self.frame_proj(frames)  # (B, N, d_model)
        embeds = self.norm(embeds)
        return embeds
```

### 3.4 TESMMultimodalModel（组合模型）

```python
# tesm_ssm/models/multimodal/model_multimodal.py

class TESMMultimodalModel(nn.Module):
    """TESM 多模态组合模型
    
    由用户指定的 Embedder 组合 + TESM Decoder 构成。
    纯文本用户不需要使用此类。
    """
    
    def __init__(self, config: TESMConfig, 
                 vision_embedder: Optional[BaseEmbedder] = None,
                 audio_embedder: Optional[BaseEmbedder] = None,
                 video_embedder: Optional[BaseEmbedder] = None):
        super().__init__()
        self.config = config
        
        # Text Embedding（始终存在）
        self.text_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 可选模态 Embedder
        self.vision_embedder = vision_embedder
        self.audio_embedder = audio_embedder
        self.video_embedder = video_embedder
        
        # TESM Decoder（复用现有模型）
        self.decoder = TESMLMHeadModel(config)
        
        # 模态类型标记（告诉 decoder 当前 token 属于哪个模态）
        # 类似 <|vision|>, <|audio|>, <|text|> 的特殊 token
        self.num_modality_tokens = 3
        self.modality_embedding = nn.Embedding(self.num_modality_tokens, config.d_model)
        
    def forward(self, text_ids=None, images=None, audio=None, video=None,
                labels=None, inference_params=None):
        """
        Args:
            text_ids: (B, L) text token IDs
            images: (B, 3, H, W) images
            audio: (B, T) raw audio waveform
            video: (B, T, 3, H, W) video frames
            labels: (B, L) labels for loss computation
        """
        embeds_list = []
        modality_ids = []
        
        # Vision
        if images is not None and self.vision_embedder is not None:
            vis_embeds = self.vision_embedder(images)
            embeds_list.append(vis_embeds)
            modality_ids.extend([0] * vis_embeds.shape[1])  # vision=0
        
        # Audio
        if audio is not None and self.audio_embedder is not None:
            aud_embeds = self.audio_embedder(audio)
            embeds_list.append(aud_embeds)
            modality_ids.extend([1] * aud_embeds.shape[1])  # audio=1
        
        # Text
        if text_ids is not None:
            txt_embeds = self.text_embedding(text_ids)
            embeds_list.append(txt_embeds)
            modality_ids.extend([2] * text_ids.shape[1])  # text=2
        
        # 拼接
        combined_embeds = torch.cat(embeds_list, dim=1)
        
        # 添加模态类型嵌入
        modality_embeds = self.modality_embedding(
            torch.tensor(modality_ids, device=combined_embeds.device)
        )
        combined_embeds = combined_embeds + modality_embeds
        
        # 传入 decoder
        # 需要修改 TESMLMHeadModel 支持 inputs_embeds
        return self.decoder(inputs_embeds=combined_embeds, labels=labels,
                          inference_params=inference_params)
```

---

## 4. 对现有代码的修改点

### 4.1 修改 1：TESMLMHeadModel 支持 inputs_embeds

```python
# tesm_ssm/models/mixer_seq_simple.py
# 在 TESMLMHeadModel.forward 中添加 inputs_embeds 参数

def forward(self, input_ids=None, inputs_embeds=None, labels=None, 
            inference_params=None, **kwargs):
    """
    Args:
        input_ids: (B, L) token IDs（与 inputs_embeds 二选一）
        inputs_embeds: (B, L, D) pre-computed embeddings（与 input_ids 二选一）
        labels: (B, L) labels for loss
    """
    assert (input_ids is not None) != (inputs_embeds is not None), \
        "Must provide exactly one of input_ids or inputs_embeds"
    
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.backbone.embedding(input_ids)
    
    # ... rest of the forward pass
```

**影响范围**：纯文本用户不受影响（传 input_ids 即可）。

### 4.2 修改 2：MixerModel 支持 inputs_embeds

同上，MixerModel 也需要支持 inputs_embeds。

### 4.3 新增文件（不修改现有文件）

```
tesm_ssm/modules/multimodal/__init__.py
tesm_ssm/modules/multimodal/base_embedder.py
tesm_ssm/modules/multimodal/vision_embedder.py
tesm_ssm/modules/multimodal/audio_embedder.py
tesm_ssm/models/multimodal/__init__.py
tesm_ssm/models/multimodal/config_multimodal.py
tesm_ssm/models/multimodal/model_multimodal.py
tesm_ssm/models/multimodal/pipeline.py
```

---

## 5. 依赖管理

### 5.1 多模态依赖为 optional

```python
# setup.py
extras_require={
    "multimodal": [
        "Pillow>=9.0",        # 图像处理
        "librosa>=0.10",      # 音频处理
        "torchvision>=0.14",  # 图像预处理
    ],
}
```

用户安装：
```bash
pip install tesm_ssm                # 纯文本
pip install tesm_ssm[multimodal]    # 多模态
```

### 5.2 运行时检查

```python
# vision_embedder.py
class VisionEmbedder(BaseEmbedder):
    def __init__(self, ...):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "VisionEmbedder requires Pillow. "
                "Install with: pip install tesm_ssm[multimodal]"
            )
```

---

## 6. 总结

| 决策 | 结论 |
|------|------|
| **Embedder 放在哪层？** | **架构层（SDK）**，作为可选模块 |
| **是否强制所有用户？** | **否**，optional import，optional dependency |
| **纯文本用户受影响吗？** | **完全不受影响** |
| **现有代码需要大改吗？** | **不需要**，只需添加 `inputs_embeds` 支持（2行代码） |
| **用户能自定义 Embedder 吗？** | **可以**，继承 `BaseEmbedder` 即可 |

### 核心原则

> SDK 提供**可组合的积木**，不是**强制的黑盒**。

| 用户类型 | 使用方式 |
|---------|---------|
| 纯文本用户 | `from tesm_ssm import TESMLMHeadModel`（不变） |
| 多模态用户 | `from tesm_ssm.models.multimodal import TESMMultimodalModel` |
| 自定义模态用户 | 继承 `BaseEmbedder` + `TESMMultimodalModel` |
