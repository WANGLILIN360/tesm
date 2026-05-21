"""
TESM 最终三方案对比实验

基于调参结果，使用最佳配置对比：
1. 基线（无退火、无隧穿）
2. 温度退火（T_start=10, T_end=0.1, cosine调度）
3. 量子隧穿（paths=8, strength=0.1, entropy能量景观）

使用更多数据和训练步数，得出最终结论。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, IterableDataset
import json
from tqdm import tqdm
import numpy as np

from tesm_ssm import TESMConfig, TESMLMHeadModel
import sentencepiece as spm


class PretrainDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, max_length=256, max_samples=None):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
    
    def __iter__(self):
        count = 0
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_samples and count >= self.max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    if not text:
                        continue
                    ids = self.tokenizer.encode(text, out_type=int)[:self.max_length]
                    if len(ids) < 10:
                        continue
                    yield {
                        'input_ids': torch.tensor(ids, dtype=torch.long),
                        'labels': torch.tensor(ids, dtype=torch.long),
                    }
                    count += 1
                except:
                    continue


def collate_fn(batch, pad_token_id, max_length=256):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    max_len = min(max(len(ids) for ids in input_ids), max_length)
    input_ids_padded = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_token_id
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long) - 100
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        seq_len = min(len(ids), max_len)
        input_ids_padded[i, :seq_len] = ids[:seq_len]
        labels_padded[i, :seq_len] = lbls[:seq_len]
    return {'input_ids': input_ids_padded, 'labels': labels_padded}


def train_model(config, tokenizer, device, num_samples, num_steps, batch_size, max_length, name):
    """训练模型"""
    print(f"\n{'='*60}")
    print(f"训练: {name}")
    print(f"{'='*60}")
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    # 创建数据集
    dataset = PretrainDataset(
        "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl",
        tokenizer, max_length, num_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, max_length),
        num_workers=0,
    )
    
    model.train()
    losses = []
    tunnel_rates = []
    
    total_loss = 0
    tunnel_success = 0
    tunnel_total = 0
    step = 0
    
    pbar = tqdm(dataloader, total=num_steps, desc=name)
    for batch in pbar:
        if step >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        total_loss += loss.item()
        
        # 收集隧穿统计
        for layer in model.backbone.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                tunneler = layer.mixer.quantum_tunneler
                if tunneler is not None and tunneler.total_attempts > 0:
                    tunnel_success += tunneler.tunnel_success_count.item()
                    tunnel_total += tunneler.total_attempts.item()
        
        step += 1
        
        if step % 50 == 0:
            avg_loss = total_loss / 50
            rate = tunnel_success / tunnel_total if tunnel_total > 0 else 0
            tunnel_rates.append(rate)
            
            msg = f"Loss={avg_loss:.4f}"
            if rate > 0:
                msg += f", Tunnel={rate*100:.1f}%"
            pbar.set_postfix_str(msg)
            total_loss = 0
    
    pbar.close()
    
    # 计算统计
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    avg_tunnel_rate = np.mean(tunnel_rates) if tunnel_rates else 0
    
    return {
        'losses': losses,
        'final_loss': final_loss,
        'tunnel_rate': avg_tunnel_rate,
        'model': model,
    }


def evaluate_perplexity(model, tokenizer, device, num_samples=500):
    """评估困惑度"""
    model.eval()
    
    dataset = PretrainDataset(
        "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl",
        tokenizer, max_length=256, max_samples=num_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=8,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, 256),
        num_workers=0,
    )
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, _ = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # 计算有效token数
            valid_mask = labels != -100
            num_tokens = valid_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity


def main():
    print("="*60)
    print("TESM 最终三方案对比实验")
    print("="*60)
    print("\n方案配置:")
    print("  1. 基线: 无退火、无隧穿")
    print("  2. 温度退火: T_start=10, T_end=0.1, cosine调度")
    print("  3. 量子隧穿: paths=8, strength=0.1, entropy景观")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("/root/private_data/tesm/tesm_v3/tokenizer_custom/tokenizer.model")
    
    class SPTokenizer:
        def __init__(self, sp):
            self.sp = sp
            self.vocab_size = sp.get_piece_size()
            self.pad_token_id = 0
            self.eos_token_id = 1
        def encode(self, text, out_type=int):
            return self.sp.encode(text, out_type=out_type)
        def decode(self, ids):
            return self.sp.decode(ids)
    
    tokenizer = SPTokenizer(sp)
    print(f"词表大小: {tokenizer.vocab_size}")
    
    # 训练参数
    num_samples = 10000
    num_steps = 3000
    batch_size = 8
    max_length = 256
    
    # ==================== 方案1: 基线 ====================
    config_baseline = TESMConfig.tiny()
    config_baseline.vocab_size = tokenizer.vocab_size
    config_baseline.pad_token_id = tokenizer.pad_token_id
    config_baseline.eos_token_id = tokenizer.eos_token_id
    config_baseline.max_seq_len = max_length
    config_baseline.ssm_cfg["annealing_enabled"] = False
    config_baseline.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_baseline = train_model(
        config_baseline, tokenizer, device,
        num_samples, num_steps, batch_size, max_length,
        "基线"
    )
    
    # ==================== 方案2: 温度退火 ====================
    config_annealing = TESMConfig.tiny()
    config_annealing.vocab_size = tokenizer.vocab_size
    config_annealing.pad_token_id = tokenizer.pad_token_id
    config_annealing.eos_token_id = tokenizer.eos_token_id
    config_annealing.max_seq_len = max_length
    config_annealing.ssm_cfg["annealing_enabled"] = True
    config_annealing.ssm_cfg["T_start"] = 10.0
    config_annealing.ssm_cfg["T_end"] = 0.1
    config_annealing.ssm_cfg["annealing_steps"] = num_steps
    config_annealing.ssm_cfg["annealing_schedule"] = "cosine"
    config_annealing.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_annealing = train_model(
        config_annealing, tokenizer, device,
        num_samples, num_steps, batch_size, max_length,
        "温度退火"
    )
    
    # ==================== 方案3: 量子隧穿（最佳配置）====================
    config_tunnel = TESMConfig.tiny()
    config_tunnel.vocab_size = tokenizer.vocab_size
    config_tunnel.pad_token_id = tokenizer.pad_token_id
    config_tunnel.eos_token_id = tokenizer.eos_token_id
    config_tunnel.max_seq_len = max_length
    config_tunnel.ssm_cfg["annealing_enabled"] = False
    config_tunnel.ssm_cfg["quantum_tunneling_enabled"] = True
    config_tunnel.ssm_cfg["tunneling_strength"] = 0.1
    config_tunnel.ssm_cfg["num_tunnel_paths"] = 8  # 最佳
    config_tunnel.ssm_cfg["energy_landscape"] = "entropy"
    
    result_tunnel = train_model(
        config_tunnel, tokenizer, device,
        num_samples, num_steps, batch_size, max_length,
        "量子隧穿"
    )
    
    # ==================== 评估 ====================
    print("\n" + "="*60)
    print("评估困惑度...")
    print("="*60)
    
    ppl_baseline = evaluate_perplexity(result_baseline['model'], tokenizer, device)
    ppl_annealing = evaluate_perplexity(result_annealing['model'], tokenizer, device)
    ppl_tunnel = evaluate_perplexity(result_tunnel['model'], tokenizer, device)
    
    # ==================== 结果 ====================
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    
    print("\n" + "-"*70)
    print(f"{'方案':<15} {'最终损失':<15} {'困惑度':<15} {'隧穿率':<15}")
    print("-"*70)
    print(f"{'基线':<15} {result_baseline['final_loss']:<15.4f} {ppl_baseline:<15.2f} {'-':<15}")
    print(f"{'温度退火':<15} {result_annealing['final_loss']:<15.4f} {ppl_annealing:<15.2f} {'-':<15}")
    print(f"{'量子隧穿':<15} {result_tunnel['final_loss']:<15.4f} {ppl_tunnel:<15.2f} {result_tunnel['tunnel_rate']*100:.1f}%")
    print("-"*70)
    
    # 排名
    results_list = [
        ('基线', result_baseline['final_loss'], ppl_baseline),
        ('温度退火', result_annealing['final_loss'], ppl_annealing),
        ('量子隧穿', result_tunnel['final_loss'], ppl_tunnel),
    ]
    results_sorted = sorted(results_list, key=lambda x: x[1])
    
    print(f"\n排名（按损失）:")
    for i, (name, loss, ppl) in enumerate(results_sorted, 1):
        print(f"  {i}. {name}: Loss={loss:.4f}, PPL={ppl:.2f}")
    
    # 保存结果
    save_data = {
        'baseline': {'final_loss': result_baseline['final_loss'], 'perplexity': ppl_baseline},
        'annealing': {'final_loss': result_annealing['final_loss'], 'perplexity': ppl_annealing},
        'tunneling': {'final_loss': result_tunnel['final_loss'], 'perplexity': ppl_tunnel, 'tunnel_rate': result_tunnel['tunnel_rate']},
        'ranking': [(name, loss, ppl) for name, loss, ppl in results_sorted],
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "final_comparison_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {save_path}")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax = axes[0]
        window = 50
        for name, losses in [('Baseline', result_baseline['losses']), 
                             ('Temperature Annealing', result_annealing['losses']),
                             ('Quantum Tunneling', result_tunnel['losses'])]:
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=name, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 最终对比
        ax = axes[1]
        methods = ['Baseline', 'Annealing', 'Tunneling']
        losses_final = [result_baseline['final_loss'], result_annealing['final_loss'], result_tunnel['final_loss']]
        ppls = [ppl_baseline, ppl_annealing, ppl_tunnel]
        
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, losses_final, width, label='Final Loss', color='steelblue')
        ax.bar(x + width/2, ppls, width, label='Perplexity', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Value')
        ax.set_title('Final Results Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'final_comparison.png'))
        plt.close()
        print("对比图已保存到: final_comparison.png")
    except Exception as e:
        print(f"绘图跳过: {e}")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
