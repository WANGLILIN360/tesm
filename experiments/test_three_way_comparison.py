"""
TESM 三方案对比实验

对比三种配置：
1. 基线（无退火、无隧穿）
2. 温度退火（原量子退火）
3. 量子隧穿

使用50条中文句子，训练100个epoch，对比损失收敛和生成质量。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import json

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


class CharTokenizer:
    """简单字符分词器"""
    def __init__(self, sentences):
        chars = set()
        for sentence in sentences:
            chars.update(sentence)
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab = [self.pad_token, self.eos_token] + sorted(chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = len(self.vocab)
        print(f"词表大小: {self.vocab_size}")
    
    def encode(self, text, add_eos=True):
        ids = [self.token_to_id.get(c, 0) for c in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids):
        return "".join(self.id_to_token.get(i, "") for i in ids if i > 1)


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.encoded = [tokenizer.encode(s, add_eos=True)[:max_length] for s in sentences]
    
    def __len__(self):
        return len(self.encoded)
    
    def __getitem__(self, idx):
        ids = torch.tensor(self.encoded[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


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


def train_model(config, dataloader, device, num_epochs, name):
    """训练模型并返回损失历史"""
    print(f"\n{'='*60}")
    print(f"训练: {name}")
    print(f"{'='*60}")
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    losses = []
    tunnel_stats = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 收集隧穿统计
        epoch_tunnel_success = 0
        epoch_tunnel_total = 0
        
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
            
            # 收集隧穿统计
            for layer in model.backbone.layers:
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                    tunneler = layer.mixer.quantum_tunneler
                    if tunneler is not None and tunneler.total_attempts > 0:
                        epoch_tunnel_success += tunneler.tunnel_success_count.item()
                        epoch_tunnel_total += tunneler.total_attempts.item()
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # 计算隧穿成功率
        tunnel_rate = epoch_tunnel_success / epoch_tunnel_total if epoch_tunnel_total > 0 else 0.0
        tunnel_stats.append(tunnel_rate)
        
        if (epoch + 1) % 10 == 0:
            msg = f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}"
            if tunnel_rate > 0:
                msg += f", 隧穿成功率 = {tunnel_rate*100:.1f}%"
            print(msg)
    
    return {
        "losses": losses,
        "tunnel_stats": tunnel_stats,
        "final_loss": losses[-1],
        "model": model,
    }


def evaluate_generation(model, tokenizer, device, test_sentences, num_samples=10):
    """评估生成能力"""
    model.eval()
    correct = 0
    total = min(num_samples, len(test_sentences))
    
    for i in range(total):
        sentence = test_sentences[i]
        prompt_len = min(5, len(sentence) // 3)
        prompt = sentence[:prompt_len]
        
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], device=device)
        
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
    
    return correct / total


def main():
    print("="*60)
    print("TESM 三方案对比实验")
    print("="*60)
    print("\n方案说明:")
    print("  1. 基线: 无退火、无隧穿")
    print("  2. 温度退火: T从10→0.1，softmax→硬阈值")
    print("  3. 量子隧穿: 状态隧穿跳出局部最优")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 数据准备
    tokenizer = CharTokenizer(TEST_SENTENCES)
    dataset = SentenceDataset(TEST_SENTENCES, tokenizer, max_length=128)
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    
    num_epochs = 100
    
    # ==================== 方案1: 基线 ====================
    print("\n" + "="*60)
    print("方案1: 基线（无退火、无隧穿）")
    print("="*60)
    
    config_baseline = TESMConfig.tiny()
    config_baseline.vocab_size = tokenizer.vocab_size
    config_baseline.pad_token_id = tokenizer.pad_token_id
    config_baseline.eos_token_id = tokenizer.eos_token_id
    # 关闭退火和隧穿
    config_baseline.ssm_cfg["annealing_enabled"] = False
    config_baseline.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_baseline = train_model(config_baseline, dataloader, device, num_epochs, "基线")
    
    # ==================== 方案2: 温度退火 ====================
    print("\n" + "="*60)
    print("方案2: 温度退火（原量子退火）")
    print("="*60)
    
    config_annealing = TESMConfig.tiny()
    config_annealing.vocab_size = tokenizer.vocab_size
    config_annealing.pad_token_id = tokenizer.pad_token_id
    config_annealing.eos_token_id = tokenizer.eos_token_id
    # 启用退火，关闭隧穿
    config_annealing.ssm_cfg["annealing_enabled"] = True
    config_annealing.ssm_cfg["T_start"] = 10.0
    config_annealing.ssm_cfg["T_end"] = 0.1
    config_annealing.ssm_cfg["annealing_steps"] = num_epochs * len(dataloader)
    config_annealing.ssm_cfg["annealing_schedule"] = "cosine"
    config_annealing.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_annealing = train_model(config_annealing, dataloader, device, num_epochs, "温度退火")
    
    # ==================== 方案3: 量子隧穿 ====================
    print("\n" + "="*60)
    print("方案3: 量子隧穿")
    print("="*60)
    
    config_tunnel = TESMConfig.tiny()
    config_tunnel.vocab_size = tokenizer.vocab_size
    config_tunnel.pad_token_id = tokenizer.pad_token_id
    config_tunnel.eos_token_id = tokenizer.eos_token_id
    # 关闭退火，启用量子隧穿
    config_tunnel.ssm_cfg["annealing_enabled"] = False
    config_tunnel.ssm_cfg["quantum_tunneling_enabled"] = True
    config_tunnel.ssm_cfg["tunneling_strength"] = 0.1
    config_tunnel.ssm_cfg["num_tunnel_paths"] = 4
    config_tunnel.ssm_cfg["energy_landscape"] = "entropy"
    
    result_tunnel = train_model(config_tunnel, dataloader, device, num_epochs, "量子隧穿")
    
    # ==================== 方案4: 温度退火 + 量子隧穿 ====================
    print("\n" + "="*60)
    print("方案4: 温度退火 + 量子隧穿")
    print("="*60)
    
    config_both = TESMConfig.tiny()
    config_both.vocab_size = tokenizer.vocab_size
    config_both.pad_token_id = tokenizer.pad_token_id
    config_both.eos_token_id = tokenizer.eos_token_id
    # 同时启用退火和隧穿
    config_both.ssm_cfg["annealing_enabled"] = True
    config_both.ssm_cfg["T_start"] = 10.0
    config_both.ssm_cfg["T_end"] = 0.1
    config_both.ssm_cfg["annealing_steps"] = num_epochs * len(dataloader)
    config_both.ssm_cfg["annealing_schedule"] = "cosine"
    config_both.ssm_cfg["quantum_tunneling_enabled"] = True
    config_both.ssm_cfg["tunneling_strength"] = 0.1
    config_both.ssm_cfg["num_tunnel_paths"] = 4
    config_both.ssm_cfg["energy_landscape"] = "entropy"
    
    result_both = train_model(config_both, dataloader, device, num_epochs, "温度退火+量子隧穿")
    
    # ==================== 结果对比 ====================
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    # 评估生成质量
    print("\n评估生成质量...")
    gen_baseline = evaluate_generation(result_baseline["model"], tokenizer, device, TEST_SENTENCES)
    gen_annealing = evaluate_generation(result_annealing["model"], tokenizer, device, TEST_SENTENCES)
    gen_tunnel = evaluate_generation(result_tunnel["model"], tokenizer, device, TEST_SENTENCES)
    gen_both = evaluate_generation(result_both["model"], tokenizer, device, TEST_SENTENCES)
    
    print("\n" + "-"*60)
    print(f"{'方案':<25} {'最终损失':<15} {'生成准确率':<15}")
    print("-"*60)
    print(f"{'基线':<25} {result_baseline['final_loss']:<15.4f} {gen_baseline*100:<14.1f}%")
    print(f"{'温度退火':<25} {result_annealing['final_loss']:<15.4f} {gen_annealing*100:<14.1f}%")
    print(f"{'量子隧穿':<25} {result_tunnel['final_loss']:<15.4f} {gen_tunnel*100:<14.1f}%")
    print(f"{'温度退火+量子隧穿':<25} {result_both['final_loss']:<15.4f} {gen_both*100:<14.1f}%")
    print("-"*60)
    
    # 计算相对改进
    best_loss = min(result_baseline['final_loss'], result_annealing['final_loss'], 
                    result_tunnel['final_loss'], result_both['final_loss'])
    best_gen = max(gen_baseline, gen_annealing, gen_tunnel, gen_both)
    
    print(f"\n最佳损失: {best_loss:.4f}")
    print(f"最佳生成准确率: {best_gen*100:.1f}%")
    
    # 保存结果
    results = {
        "baseline": {
            "final_loss": result_baseline["final_loss"],
            "generation_accuracy": gen_baseline,
            "losses": result_baseline["losses"],
        },
        "annealing": {
            "final_loss": result_annealing["final_loss"],
            "generation_accuracy": gen_annealing,
            "losses": result_annealing["losses"],
        },
        "tunneling": {
            "final_loss": result_tunnel["final_loss"],
            "generation_accuracy": gen_tunnel,
            "losses": result_tunnel["losses"],
            "tunnel_success_rate": result_tunnel["tunnel_stats"][-1] if result_tunnel["tunnel_stats"] else 0,
        },
        "both": {
            "final_loss": result_both["final_loss"],
            "generation_accuracy": gen_both,
            "losses": result_both["losses"],
            "tunnel_success_rate": result_both["tunnel_stats"][-1] if result_both["tunnel_stats"] else 0,
        },
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "three_way_comparison_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {save_path}")
    
    # 绘制对比曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(result_baseline["losses"], 'b-', label='Baseline', linewidth=2)
        plt.plot(result_annealing["losses"], 'g-', label='Temperature Annealing', linewidth=2)
        plt.plot(result_tunnel["losses"], 'r-', label='Quantum Tunneling', linewidth=2)
        plt.plot(result_both["losses"], 'm-', label='Annealing + Tunneling', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        methods = ['Baseline', 'Annealing', 'Tunneling', 'Both']
        losses = [result_baseline["final_loss"], result_annealing["final_loss"],
                  result_tunnel["final_loss"], result_both["final_loss"]]
        accuracies = [gen_baseline*100, gen_annealing*100, gen_tunnel*100, gen_both*100]
        
        x = range(len(methods))
        width = 0.35
        plt.bar([i - width/2 for i in x], losses, width, label='Final Loss', color='steelblue')
        plt.bar([i + width/2 for i in x], accuracies, width, label='Gen Accuracy (%)', color='coral')
        plt.xticks(x, methods)
        plt.ylabel('Value')
        plt.title('Final Results Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'three_way_comparison.png'))
        plt.close()
        print("对比图已保存到: three_way_comparison.png")
    except Exception as e:
        print(f"绘图跳过: {e}")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
