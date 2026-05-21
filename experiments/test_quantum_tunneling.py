"""
TESM 量子隧穿启发模块测试

验证量子隧穿机制对状态优化的效果：
1. 能量景观计算
2. 隧穿概率分布
3. 状态转移效果
4. 与基线对比（无隧穿 vs 有隧穿）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

from tesm_ssm.modules.tesm import QuantumTunneling, TESM
from tesm_ssm import TESMConfig, TESMLMHeadModel


# ==================== 测试数据 ====================
TEST_SENTENCES = [
    "人工智能正在改变世界。",
    "量子计算是未来的方向。",
    "深度学习需要大量数据。",
    "自然语言处理很有趣。",
    "机器学习应用广泛。",
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
    
    def encode(self, text, add_eos=True):
        ids = [self.token_to_id.get(c, 0) for c in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids):
        return "".join(self.id_to_token.get(i, "") for i in ids if i > 1)


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=64):
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


# ==================== 测试函数 ====================

def test_energy_landscape():
    """测试能量景观计算"""
    print("\n" + "="*60)
    print("测试 1: 能量景观计算")
    print("="*60)
    
    d_state = 64
    batch_size = 4
    
    # 创建三种能量景观的隧穿器
    for landscape in ["entropy", "variance", "hybrid"]:
        tunneler = QuantumTunneling(
            d_state=d_state,
            energy_landscape=landscape,
        )
        
        # 生成测试状态
        # 低能量状态：集中分布
        low_energy_state = torch.zeros(batch_size, d_state)
        low_energy_state[:, :8] = 1.0  # 只激活前8维
        
        # 高能量状态：均匀分布
        high_energy_state = torch.ones(batch_size, d_state) / d_state
        
        # 随机状态
        random_state = torch.randn(batch_size, d_state)
        
        # 计算能量
        low_E = tunneler.compute_energy(low_energy_state)
        high_E = tunneler.compute_energy(high_energy_state)
        rand_E = tunneler.compute_energy(random_state)
        
        print(f"\n{landscape} 能量景观:")
        print(f"  低能量状态 (集中): {low_E.mean():.4f}")
        print(f"  高能量状态 (均匀): {high_E.mean():.4f}")
        print(f"  随机状态: {rand_E.mean():.4f}")
        
        # 验证：集中状态应该能量更低
        if landscape == "entropy":
            assert low_E.mean() < high_E.mean(), f"熵能量: 集中状态应低于均匀状态"
            print("  ✓ 验证通过: 集中状态能量更低")


def test_tunneling_probability():
    """测试隧穿概率计算"""
    print("\n" + "="*60)
    print("测试 2: 隧穿概率分布")
    print("="*60)
    
    d_state = 64
    tunneler = QuantumTunneling(d_state=d_state, tunneling_strength=0.1)
    
    # 测试不同势垒高度
    barrier_heights = torch.linspace(0, 2, 100)
    probs = tunneler.get_tunneling_probability(barrier_heights)
    
    print(f"\n隧穿强度: {tunneler.tunneling_strength}")
    print(f"势垒高度范围: [0, 2]")
    print(f"隧穿概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # 验证：势垒越高，概率越低
    assert probs[0] > probs[-1], "势垒越高，隧穿概率应越低"
    print("✓ 验证通过: 势垒越高，隧穿概率越低")
    
    # 绘制概率曲线
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(barrier_heights.detach().numpy(), probs.detach().numpy(), 'b-', linewidth=2)
        plt.xlabel('Barrier Height')
        plt.ylabel('Tunneling Probability')
        plt.title('Quantum-Inspired Tunneling Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'tunneling_probability.png'))
        plt.close()
        print("✓ 概率曲线已保存到 tunneling_probability.png")
    except Exception as e:
        print(f"绘图跳过: {e}")


def test_quantum_tunnel_step():
    """测试单步隧穿操作"""
    print("\n" + "="*60)
    print("测试 3: 单步隧穿操作")
    print("="*60)
    
    d_state = 64
    batch_size = 8
    
    tunneler = QuantumTunneling(
        d_state=d_state,
        tunneling_strength=0.2,
        num_tunnel_paths=4,
    )
    
    # 初始状态
    states = torch.randn(batch_size, d_state)
    
    print(f"\n初始状态形状: {states.shape}")
    print(f"初始状态能量: {tunneler.compute_energy(states).mean():.4f}")
    
    # 执行隧穿
    new_states, tunnel_info = tunneler.quantum_tunnel(states, training=True)
    
    print(f"\n隧穿后状态形状: {new_states.shape}")
    print(f"隧穿后状态能量: {tunneler.compute_energy(new_states).mean():.4f}")
    
    print(f"\n隧穿统计:")
    print(f"  成功率: {tunnel_info['tunnel_success_rate']*100:.1f}%")
    print(f"  平均势垒高度: {tunnel_info['avg_barrier_height']:.4f}")
    print(f"  平均隧穿概率: {tunnel_info['avg_tunnel_prob']:.4f}")
    print(f"  能量变化: {tunnel_info['energy_before']:.4f} → {tunnel_info['energy_after']:.4f}")


def test_tunneling_vs_baseline():
    """对比测试：有隧穿 vs 无隧穿"""
    print("\n" + "="*60)
    print("测试 4: 隧穿效果对比训练")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    tokenizer = CharTokenizer(TEST_SENTENCES)
    dataset = SentenceDataset(TEST_SENTENCES, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    
    num_epochs = 50
    
    # ===== 基线模型（无隧穿）=====
    print("\n--- 基线模型（无隧穿）---")
    config_baseline = TESMConfig.tiny()
    config_baseline.vocab_size = tokenizer.vocab_size
    config_baseline.pad_token_id = tokenizer.pad_token_id
    config_baseline.eos_token_id = tokenizer.eos_token_id
    config_baseline.ssm_cfg["quantum_tunneling_enabled"] = False
    
    model_baseline = TESMLMHeadModel(config_baseline, device=device)
    model_baseline.to(device)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)
    
    baseline_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer_baseline.zero_grad()
            outputs, _ = model_baseline(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer_baseline.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        baseline_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # ===== 隧穿模型 =====
    print("\n--- 隧穿模型（启用量子隧穿）---")
    config_tunnel = TESMConfig.tiny()
    config_tunnel.vocab_size = tokenizer.vocab_size
    config_tunnel.pad_token_id = tokenizer.pad_token_id
    config_tunnel.eos_token_id = tokenizer.eos_token_id
    config_tunnel.ssm_cfg["quantum_tunneling_enabled"] = True
    config_tunnel.ssm_cfg["tunneling_strength"] = 0.15
    config_tunnel.ssm_cfg["num_tunnel_paths"] = 4
    config_tunnel.ssm_cfg["energy_landscape"] = "entropy"
    
    model_tunnel = TESMLMHeadModel(config_tunnel, device=device)
    model_tunnel.to(device)
    optimizer_tunnel = torch.optim.AdamW(model_tunnel.parameters(), lr=1e-3)
    
    tunnel_losses = []
    tunnel_stats = []
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_stats = {"success_rate": 0, "energy_delta": 0, "count": 0}
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer_tunnel.zero_grad()
            outputs, _ = model_tunnel(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer_tunnel.step()
            total_loss += loss.item()
            
            # 收集隧穿统计
            for layer in model_tunnel.backbone.layers:
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                    tunneler = layer.mixer.quantum_tunneler
                    if tunneler is not None and tunneler.total_attempts > 0:
                        epoch_stats["success_rate"] += tunneler.tunnel_success_count.item()
                        epoch_stats["count"] += tunneler.total_attempts.item()
        
        avg_loss = total_loss / len(dataloader)
        tunnel_losses.append(avg_loss)
        
        if epoch_stats["count"] > 0:
            avg_success = epoch_stats["success_rate"] / epoch_stats["count"]
            tunnel_stats.append(avg_success)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, 隧穿成功率 = {tunnel_stats[-1]*100:.1f}%" if tunnel_stats else f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # ===== 对比结果 =====
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"基线最终损失: {baseline_losses[-1]:.4f}")
    print(f"隧穿最终损失: {tunnel_losses[-1]:.4f}")
    improvement = (baseline_losses[-1] - tunnel_losses[-1]) / baseline_losses[-1] * 100
    print(f"改进: {improvement:+.2f}%")
    
    # 绘制对比曲线
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(baseline_losses, 'b-', label='Baseline (No Tunneling)', linewidth=2)
        plt.plot(tunnel_losses, 'r-', label='With Quantum Tunneling', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison: Baseline vs Quantum Tunneling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'tunneling_comparison.png'))
        plt.close()
        print("\n✓ 对比曲线已保存到 tunneling_comparison.png")
    except Exception as e:
        print(f"绘图跳过: {e}")


def test_generation_quality():
    """测试生成质量"""
    print("\n" + "="*60)
    print("测试 5: 生成质量对比")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CharTokenizer(TEST_SENTENCES)
    dataset = SentenceDataset(TEST_SENTENCES, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    
    num_epochs = 100
    
    # 训练隧穿模型
    config = TESMConfig.tiny()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.ssm_cfg["quantum_tunneling_enabled"] = True
    config.ssm_cfg["tunneling_strength"] = 0.1
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("训练中...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs, _ = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}")
    
    # 测试生成
    print("\n生成测试:")
    model.eval()
    for sentence in TEST_SENTENCES[:3]:
        prompt = sentence[:5]
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], device=device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=3,
            use_cache=True,
        )
        gen_text = tokenizer.decode(generated[0].tolist())
        print(f"\n  Prompt: {prompt}")
        print(f"  生成: {gen_text}")
        print(f"  原句: {sentence}")


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("TESM 量子隧穿启发模块测试")
    print("="*60)
    
    # 运行所有测试
    test_energy_landscape()
    test_tunneling_probability()
    test_quantum_tunnel_step()
    test_tunneling_vs_baseline()
    test_generation_quality()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
