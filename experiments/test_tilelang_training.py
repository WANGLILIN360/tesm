"""
TESM TileLang 后端训练测试

使用 TileLang 后端训练 50 条中文短文本，测试模型的复述能力。
TileLang 针对 MIMO 多头模式优化。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tesm_ssm import TESMConfig, TESMLMHeadModel


# ==================== 50个中文测试句子 ====================
TEST_SENTENCES = [
    # 问答类 (10句)
    "什么是人工智能？人工智能是模拟人类智能的技术。",
    "天空为什么是蓝色的？因为大气层散射了阳光。",
    "地球绕太阳转一圈需要多久？大约需要365天。",
    "水的化学式是什么？水的化学式是H2O。",
    "中国的首都在哪里？中国的首都是北京。",
    "一年有几个季节？一年有春夏秋冬四个季节。",
    "人类需要呼吸什么气体？人类需要呼吸氧气。",
    "太阳从哪个方向升起？太阳从东方升起。",
    "鱼儿为什么能在水里呼吸？因为它们有鳃。",
    "什么是友谊？友谊是人与人之间的真挚情感。",
    
    # 对话类 (10句)
    "你好，很高兴认识你。你好，我也很高兴认识你。",
    "今天天气怎么样？今天天气晴朗，适合外出。",
    "你喜欢什么颜色？我喜欢蓝色，因为它像大海。",
    "你喜欢吃什么水果？我最喜欢吃苹果和香蕉。",
    "你周末通常做什么？我通常看书或散步。",
    "你会说几种语言？我会说中文和一点英语。",
    "你家有几口人？我家有四口人。",
    "你最喜欢的运动是什么？我最喜欢打篮球。",
    "你觉得学习重要吗？是的，学习非常重要。",
    "你有什么爱好？我喜欢画画和听音乐。",
    
    # 描述类 (10句)
    "春天来了，花儿开放，小鸟在枝头歌唱。",
    "夏天的太阳很热，人们喜欢去海边游泳。",
    "秋天树叶变黄，纷纷飘落在地上。",
    "冬天寒冷，人们穿上厚厚的棉衣。",
    "早晨的阳光温暖而柔和，照亮了大地。",
    "夜晚的星空美丽，无数星星闪烁着光芒。",
    "大海辽阔无边，波浪拍打着沙滩。",
    "高山巍峨壮观，山顶常年覆盖着白雪。",
    "森林里树木茂密，各种动物在其中生活。",
    "城市里高楼林立，街道上人来人往。",
    
    # 推理类 (10句)
    "如果下雨，地面会变湿。现在地面湿了，所以下雨了。",
    "所有的人都会死。苏格拉底是人，所以苏格拉底会死。",
    "学习使人进步。他学习很努力，所以他进步很快。",
    "运动有益健康。她每天运动，所以她很健康。",
    "读书可以增长知识。他读了很多书，所以知识丰富。",
    "勤奋是成功的关键。他很勤奋，所以他成功了。",
    "节约用水很重要。我们要养成节约用水的好习惯。",
    "保护环境人人有责。每个人都应该爱护环境。",
    "诚实是一种美德。我们要做一个诚实的人。",
    "友谊需要珍惜。好朋友之间要互相信任和帮助。",
    
    # 情感类 (10句)
    "看到家人团聚，我感到非常幸福和温暖。",
    "听到好消息，她激动得跳了起来。",
    "失去宠物后，他伤心了好几天。",
    "考试取得好成绩，同学们都很开心。",
    "收到朋友的礼物，她感到很惊喜。",
    "看到美丽的风景，心情变得格外舒畅。",
    "帮助别人让我感到快乐和满足。",
    "回忆童年时光，心中充满温馨和怀念。",
    "面对困难时，我们要保持乐观的心态。",
    "成功后的喜悦让人难以忘怀。",
]


# ==================== 字符级分词器 ====================
class CharTokenizer:
    """基于字符的简单分词器"""
    
    def __init__(self, sentences):
        chars = set()
        for sentence in sentences:
            chars.update(sentence)
        
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        self.vocab = [self.pad_token, self.eos_token, self.unk_token]
        self.vocab.extend(sorted(chars))
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.vocab_size = len(self.vocab)
        
        print(f"词表大小: {self.vocab_size}")
    
    def encode(self, text, add_eos=True):
        ids = [self.token_to_id.get(c, self.unk_token_id) for c in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        chars = []
        for i in ids:
            if i >= self.vocab_size:
                continue
            token = self.id_to_token[i]
            if skip_special_tokens and token in [self.pad_token, self.eos_token, self.unk_token]:
                continue
            chars.append(token)
        return "".join(chars)


# ==================== 数据集 ====================
class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.encoded = []
        for sentence in sentences:
            ids = tokenizer.encode(sentence, add_eos=True)
            if len(ids) > max_length:
                ids = ids[:max_length]
            self.encoded.append(ids)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        ids = self.encoded[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    input_ids_padded = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_token_id
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long) - 100
    
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        input_ids_padded[i, :len(ids)] = ids
        labels_padded[i, :len(lbls)] = lbls
    
    return {"input_ids": input_ids_padded, "labels": labels_padded}


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_generation(model, tokenizer, device, num_samples=5, max_new_tokens=32):
    model.eval()
    
    test_prompts = []
    for sentence in TEST_SENTENCES[:num_samples]:
        mid = len(sentence) // 2
        prompt = sentence[:mid]
        test_prompts.append(prompt)
    
    print("\n" + "="*60)
    print("生成测试:")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts):
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # 使用低温度和贪婪解码提高稳定性
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=10,
                use_cache=True,
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        print(f"\n[{i+1}] Prompt: {prompt}")
        print(f"    生成: {generated_text}")
        print(f"    原句: {TEST_SENTENCES[i]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查 TileLang 是否可用
    try:
        from tesm_ssm.ops.tilelang import TESM_TILELANG_AVAILABLE
        if TESM_TILELANG_AVAILABLE:
            print("TileLang 后端可用 ✓")
        else:
            print("TileLang 后端不可用，将回退到 PyTorch")
    except ImportError:
        print("TileLang 未安装，将回退到 PyTorch")
    
    # 创建分词器和数据集
    tokenizer = CharTokenizer(TEST_SENTENCES)
    dataset = SentenceDataset(TEST_SENTENCES, tokenizer, max_length=128)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    
    # 创建模型配置 - TileLang MIMO 模式
    config = TESMConfig.tiny()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.max_seq_len = 128
    
    # MIMO 模式（修复后）
    config.use_mimo = True  # 启用 MIMO
    config.n_heads = 4  # 4 头 MIMO

    config.ssm_cfg = {
        "d_state": 256,
        "expand": 2,
        "ent_rank": 32,
        "entanglement_scale": 0.25,
        "entanglement_threshold": 0.05,
        "entanglement_init": 0.3,
        "entanglement_window": 16,
        "entanglement_block_size": 256,
        "state_scan_chunk_size": 16,
        # 后端配置：使用 TileLang 加速
        "use_triton_kernels": False,
        "kernel_backend": "tilelang",
        "kernel_mode": "precise",
        "decay_init_bias": 0.0,
        # 温度退火
        "annealing_enabled": True,
        "T_start": 10.0,
        "T_end": 0.1,
        "annealing_steps": 500,
        "annealing_schedule": "cosine",
    }
    
    print(f"\n模型配置 (TileLang MIMO):")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  use_mimo: {config.use_mimo}")
    print(f"  kernel_backend: {config.ssm_cfg['kernel_backend']}")
    print(f"  vocab_size: {config.vocab_size}")
    
    # 创建模型
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    # 检查 MIMO 配置
    for name, module in model.named_modules():
        if 'TESMMIMO' in type(module).__name__:
            print(f"\n  MIMO 模块找到: {name}")
            print(f"    d_head: {module.d_head}")
            print(f"    d_state_total: {module.d_state_total}")
            print(f"    n_heads: {module.n_heads}")
            break
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 优化器（降低学习率提高稳定性）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # 学习率调度器
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练 (TileLang MIMO 后端)")
    print("="*60)
    
    best_loss = float("inf")
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if (epoch + 1) % 20 == 0:
                evaluate_generation(model, tokenizer, device, num_samples=3, max_new_tokens=20)
    
    # 最终评估
    print("\n" + "="*60)
    print("最终评估")
    print("="*60)
    evaluate_generation(model, tokenizer, device, num_samples=10, max_new_tokens=30)
    
    # 过拟合测试
    print("\n" + "="*60)
    print("过拟合测试（复述训练数据）")
    print("="*60)
    
    model.eval()
    correct = 0
    total = len(TEST_SENTENCES)
    
    for i, sentence in enumerate(TEST_SENTENCES):
        prompt_len = min(5, len(sentence) // 3)
        prompt = sentence[:prompt_len]
        
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=len(sentence) - prompt_len + 10,
                temperature=0.1,
                top_k=1,
                use_cache=True,
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        if sentence in generated_text:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        if i < 10:
            print(f"[{status}] Prompt: {prompt}")
            print(f"    生成: {generated_text}")
            print(f"    原句: {sentence}")
            print()
    
    print(f"\n过拟合率: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # 保存模型
    save_path = os.path.join(os.path.dirname(__file__), "test_tilelang_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "backend": "tilelang",
    }, save_path)
    print(f"\n模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
