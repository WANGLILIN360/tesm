"""
TESM 大规模对比实验

对比三种配置：
1. 基线（无退火、无隧穿）
2. 温度退火（高温探索→低温收敛）
3. 量子隧穿（状态隧穿跳出局部最优）

使用 pretrain_hq.jsonl 数据集进行测试。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import math
from tqdm import tqdm

from tesm_ssm import TESMConfig, TESMLMHeadModel
from transformers import AutoTokenizer


# ==================== 数据集 ====================

class PretrainDataset(IterableDataset):
    """流式读取预训练数据"""
    def __init__(self, filepath, tokenizer, max_length=512, max_samples=None):
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
                    
                    # 编码
                    ids = self.tokenizer.encode(text, add_special_tokens=True)[:self.max_length]
                    if len(ids) < 10:  # 过短跳过
                        continue
                    
                    yield {
                        'input_ids': torch.tensor(ids, dtype=torch.long),
                        'labels': torch.tensor(ids, dtype=torch.long),
                    }
                    count += 1
                except Exception as e:
                    continue


def collate_fn(batch, pad_token_id, max_length=512):
    """动态padding"""
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


def train_model(config, dataloader, device, num_steps, name, log_interval=100):
    """训练模型并返回损失历史"""
    print(f"\n{'='*60}")
    print(f"训练: {name}")
    print(f"{'='*60}")
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    model.train()
    losses = []
    tunnel_stats = []
    
    step = 0
    total_loss = 0
    epoch_tunnel_success = 0
    epoch_tunnel_total = 0
    
    pbar = tqdm(dataloader, desc=f"Training {name}", total=num_steps)
    
    for batch in pbar:
        if step >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf loss at step {step}, skipping")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        losses.append(loss.item())
        
        # 收集隧穿统计
        for layer in model.backbone.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                tunneler = layer.mixer.quantum_tunneler
                if tunneler is not None and tunneler.total_attempts > 0:
                    epoch_tunnel_success += tunneler.tunnel_success_count.item()
                    epoch_tunnel_total += tunneler.total_attempts.item()
        
        step += 1
        
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            tunnel_rate = epoch_tunnel_success / epoch_tunnel_total if epoch_tunnel_total > 0 else 0.0
            tunnel_stats.append(tunnel_rate)
            
            msg = f"Step {step}/{num_steps}: Loss={avg_loss:.4f}"
            if tunnel_rate > 0:
                msg += f", Tunnel={tunnel_rate*100:.1f}%"
            pbar.set_postfix_str(msg)
            
            total_loss = 0
            epoch_tunnel_success = 0
            epoch_tunnel_total = 0
    
    pbar.close()
    
    return {
        'losses': losses,
        'tunnel_stats': tunnel_stats,
        'final_loss': losses[-100:] if len(losses) >= 100 else losses,
        'avg_final_loss': sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else sum(losses) / len(losses),
    }


def main():
    print("="*60)
    print("TESM 大规模对比实验")
    print("="*60)
    print("\n方案说明:")
    print("  1. 基线: 无退火、无隧穿")
    print("  2. 温度退火: T从10→0.1，softmax→硬阈值")
    print("  3. 量子隧穿: 状态隧穿跳出局部最优")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 数据路径
    data_path = "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl"
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    # 使用 sentencepiece tokenizer
    tokenizer_path = "/root/private_data/tesm/tesm_v3/tokenizer_custom/tokenizer.model"
    if os.path.exists(tokenizer_path):
        print(f"加载 tokenizer: {tokenizer_path}")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        
        # 包装成类似transformers的接口
        class SPTokenizer:
            def __init__(self, sp):
                self.sp = sp
                self.vocab_size = sp.get_piece_size()
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def encode(self, text, add_special_tokens=True):
                return self.sp.encode(text, out_type=int)
            
            def decode(self, ids):
                return self.sp.decode(ids)
        
        tokenizer = SPTokenizer(sp)
    else:
        print("使用默认 Qwen tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    
    print(f"词表大小: {tokenizer.vocab_size}")
    
    # 训练参数
    num_samples = 10000  # 使用1万条数据
    num_steps = 2000     # 训练2000步
    batch_size = 8
    max_length = 256
    
    # 创建数据集
    dataset = PretrainDataset(
        data_path, tokenizer, 
        max_length=max_length,
        max_samples=num_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, max_length),
        num_workers=0,
    )
    
    # ==================== 方案1: 基线 ====================
    print("\n" + "="*60)
    print("方案1: 基线（无退火、无隧穿）")
    print("="*60)
    
    config_baseline = TESMConfig.tiny()
    config_baseline.vocab_size = tokenizer.vocab_size
    config_baseline.pad_token_id = tokenizer.pad_token_id
    config_baseline.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    config_baseline.max_seq_len = max_length
    # 关闭退火和隧穿
    config_baseline.ssm_cfg["annealing_enabled"] = False
    config_baseline.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_baseline = train_model(config_baseline, dataloader, device, num_steps, "基线")
    
    # ==================== 方案2: 温度退火 ====================
    print("\n" + "="*60)
    print("方案2: 温度退火")
    print("="*60)
    
    config_annealing = TESMConfig.tiny()
    config_annealing.vocab_size = tokenizer.vocab_size
    config_annealing.pad_token_id = tokenizer.pad_token_id
    config_annealing.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    config_annealing.max_seq_len = max_length
    # 启用退火，关闭隧穿
    config_annealing.ssm_cfg["annealing_enabled"] = True
    config_annealing.ssm_cfg["T_start"] = 10.0
    config_annealing.ssm_cfg["T_end"] = 0.1
    config_annealing.ssm_cfg["annealing_steps"] = num_steps
    config_annealing.ssm_cfg["annealing_schedule"] = "cosine"
    config_annealing.ssm_cfg["quantum_tunneling_enabled"] = False
    
    result_annealing = train_model(config_annealing, dataloader, device, num_steps, "温度退火")
    
    # ==================== 方案3: 量子隧穿 ====================
    print("\n" + "="*60)
    print("方案3: 量子隧穿")
    print("="*60)
    
    config_tunnel = TESMConfig.tiny()
    config_tunnel.vocab_size = tokenizer.vocab_size
    config_tunnel.pad_token_id = tokenizer.pad_token_id
    config_tunnel.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    config_tunnel.max_seq_len = max_length
    # 关闭退火，启用量子隧穿
    config_tunnel.ssm_cfg["annealing_enabled"] = False
    config_tunnel.ssm_cfg["quantum_tunneling_enabled"] = True
    config_tunnel.ssm_cfg["tunneling_strength"] = 0.1
    config_tunnel.ssm_cfg["num_tunnel_paths"] = 4
    config_tunnel.ssm_cfg["energy_landscape"] = "entropy"
    
    result_tunnel = train_model(config_tunnel, dataloader, device, num_steps, "量子隧穿")
    
    # ==================== 结果对比 ====================
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    print("\n" + "-"*60)
    print(f"{'方案':<20} {'平均最终损失':<20} {'隧穿成功率':<15}")
    print("-"*60)
    print(f"{'基线':<20} {result_baseline['avg_final_loss']:<20.4f} {'-':<15}")
    print(f"{'温度退火':<20} {result_annealing['avg_final_loss']:<20.4f} {'-':<15}")
    print(f"{'量子隧穿':<20} {result_tunnel['avg_final_loss']:<20.4f} {result_tunnel['tunnel_stats'][-1]*100 if result_tunnel['tunnel_stats'] else 0:.1f}%")
    print("-"*60)
    
    # 保存结果
    results = {
        "baseline": {
            "avg_final_loss": result_baseline['avg_final_loss'],
            "losses": result_baseline['losses'],
        },
        "annealing": {
            "avg_final_loss": result_annealing['avg_final_loss'],
            "losses": result_annealing['losses'],
        },
        "tunneling": {
            "avg_final_loss": result_tunnel['avg_final_loss'],
            "losses": result_tunnel['losses'],
            "tunnel_stats": result_tunnel['tunnel_stats'],
        },
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "large_scale_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {save_path}")
    
    # 绘制对比曲线
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        # 平滑曲线
        def smooth(data, window=50):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(result_baseline['losses']) > 50:
            plt.plot(smooth(result_baseline['losses']), 'b-', label='Baseline', linewidth=2, alpha=0.8)
        if len(result_annealing['losses']) > 50:
            plt.plot(smooth(result_annealing['losses']), 'g-', label='Temperature Annealing', linewidth=2, alpha=0.8)
        if len(result_tunnel['losses']) > 50:
            plt.plot(smooth(result_tunnel['losses']), 'r-', label='Quantum Tunneling', linewidth=2, alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss (Smoothed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最终损失对比
        plt.subplot(1, 2, 2)
        methods = ['Baseline', 'Annealing', 'Tunneling']
        losses = [result_baseline['avg_final_loss'], result_annealing['avg_final_loss'], result_tunnel['avg_final_loss']]
        colors = ['steelblue', 'seagreen', 'coral']
        bars = plt.bar(methods, losses, color=colors)
        plt.ylabel('Average Final Loss')
        plt.title('Final Loss Comparison')
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{loss:.4f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'large_scale_comparison.png'))
        plt.close()
        print("对比图已保存到: large_scale_comparison.png")
    except Exception as e:
        print(f"绘图跳过: {e}")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
