"""
TESM 量子隧穿调参实验

测试不同参数组合：
1. tunneling_strength: 隧穿强度 [0.05, 0.1, 0.2, 0.5]
2. num_tunnel_paths: 候选路径数 [2, 4, 8]
3. energy_landscape: 能量景观 [entropy, variance, hybrid]

共 4 × 3 × 3 = 36 组实验
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
import itertools

from tesm_ssm import TESMConfig, TESMLMHeadModel
import sentencepiece as spm


# ==================== 数据集 ====================

class PretrainDataset(IterableDataset):
    """流式读取预训练数据"""
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
                except Exception as e:
                    continue


def collate_fn(batch, pad_token_id, max_length=256):
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


def train_and_evaluate(config, dataloader, device, num_steps, name):
    """训练并评估"""
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    model.train()
    losses = []
    tunnel_success_rate = 0
    
    step = 0
    total_loss = 0
    tunnel_success = 0
    tunnel_total = 0
    
    for batch in dataloader:
        if step >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
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
                    tunnel_success += tunneler.tunnel_success_count.item()
                    tunnel_total += tunneler.total_attempts.item()
        
        step += 1
    
    avg_loss = total_loss / step if step > 0 else float('inf')
    tunnel_rate = tunnel_success / tunnel_total if tunnel_total > 0 else 0.0
    
    # 计算最后100步的平均损失
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else avg_loss
    
    return {
        'avg_loss': avg_loss,
        'final_loss': final_loss,
        'tunnel_rate': tunnel_rate,
        'total_steps': step,
    }


def main():
    print("="*60)
    print("TESM 量子隧穿调参实验")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载 tokenizer
    tokenizer_path = "/root/private_data/tesm/tesm_v3/tokenizer_custom/tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
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
    
    # 数据
    data_path = "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl"
    num_samples = 5000
    num_steps = 1000
    batch_size = 8
    max_length = 256
    
    dataset = PretrainDataset(data_path, tokenizer, max_length, num_samples)
    
    # 调参网格（简化版）
    tunneling_strengths = [0.05, 0.2, 0.5]
    num_tunnel_paths_list = [2, 4]
    energy_landscapes = ["entropy", "variance"]
    
    # 基线（无隧穿）
    print("\n" + "="*60)
    print("基线实验（无隧穿）")
    print("="*60)
    
    config_baseline = TESMConfig.tiny()
    config_baseline.vocab_size = tokenizer.vocab_size
    config_baseline.pad_token_id = tokenizer.pad_token_id
    config_baseline.eos_token_id = tokenizer.eos_token_id
    config_baseline.max_seq_len = max_length
    config_baseline.ssm_cfg["annealing_enabled"] = False
    config_baseline.ssm_cfg["quantum_tunneling_enabled"] = False
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, max_length),
        num_workers=0,
    )
    
    result_baseline = train_and_evaluate(config_baseline, dataloader, device, num_steps, "基线")
    print(f"基线: 最终损失={result_baseline['final_loss']:.4f}")
    
    # 调参实验
    results = []
    total_exp = len(tunneling_strengths) * len(num_tunnel_paths_list) * len(energy_landscapes)
    
    print(f"\n共 {total_exp} 组调参实验")
    print("="*60)
    
    exp_idx = 0
    best_loss = float('inf')
    best_config = None
    
    for strength in tunneling_strengths:
        for num_paths in num_tunnel_paths_list:
            for landscape in energy_landscapes:
                exp_idx += 1
                
                # 重置数据集
                dataset = PretrainDataset(data_path, tokenizer, max_length, num_samples)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, max_length),
                    num_workers=0,
                )
                
                config_name = f"strength={strength}, paths={num_paths}, landscape={landscape}"
                print(f"\n[{exp_idx}/{total_exp}] {config_name}")
                
                config = TESMConfig.tiny()
                config.vocab_size = tokenizer.vocab_size
                config.pad_token_id = tokenizer.pad_token_id
                config.eos_token_id = tokenizer.eos_token_id
                config.max_seq_len = max_length
                config.ssm_cfg["annealing_enabled"] = False
                config.ssm_cfg["quantum_tunneling_enabled"] = True
                config.ssm_cfg["tunneling_strength"] = strength
                config.ssm_cfg["num_tunnel_paths"] = num_paths
                config.ssm_cfg["energy_landscape"] = landscape
                
                result = train_and_evaluate(config, dataloader, device, num_steps, config_name)
                
                print(f"  最终损失: {result['final_loss']:.4f}, 隧穿率: {result['tunnel_rate']*100:.1f}%")
                
                results.append({
                    'strength': strength,
                    'num_paths': num_paths,
                    'landscape': landscape,
                    'final_loss': result['final_loss'],
                    'tunnel_rate': result['tunnel_rate'],
                })
                
                if result['final_loss'] < best_loss:
                    best_loss = result['final_loss']
                    best_config = config_name
    
    # 结果排序
    results_sorted = sorted(results, key=lambda x: x['final_loss'])
    
    print("\n" + "="*60)
    print("调参结果（按损失排序）")
    print("="*60)
    
    print(f"\n基线损失: {result_baseline['final_loss']:.4f}")
    print(f"\n最佳配置: {best_config}")
    print(f"最佳损失: {best_loss:.4f}")
    
    improvement = (result_baseline['final_loss'] - best_loss) / result_baseline['final_loss'] * 100
    print(f"相对基线改进: {improvement:+.2f}%")
    
    print("\n" + "-"*80)
    print(f"{'排名':<5} {'强度':<10} {'路径数':<10} {'能量景观':<12} {'最终损失':<12} {'隧穿率':<10}")
    print("-"*80)
    
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:<5} {r['strength']:<10} {r['num_paths']:<10} {r['landscape']:<12} {r['final_loss']:<12.4f} {r['tunnel_rate']*100:<9.1f}%")
    
    print("-"*80)
    
    # 保存结果
    save_data = {
        'baseline': result_baseline,
        'best_config': best_config,
        'best_loss': best_loss,
        'improvement': improvement,
        'all_results': results_sorted,
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "quantum_tuning_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {save_path}")
    
    # 绘制热力图
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 按强度和路径数绘制热力图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, landscape in enumerate(energy_landscapes):
            ax = axes[idx]
            
            # 构建矩阵
            matrix = np.zeros((len(tunneling_strengths), len(num_tunnel_paths_list)))
            for r in results:
                if r['landscape'] == landscape:
                    i = tunneling_strengths.index(r['strength'])
                    j = num_tunnel_paths_list.index(r['num_paths'])
                    matrix[i, j] = r['final_loss']
            
            im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
            ax.set_xticks(range(len(num_tunnel_paths_list)))
            ax.set_xticklabels(num_tunnel_paths_list)
            ax.set_yticks(range(len(tunneling_strengths)))
            ax.set_yticklabels(tunneling_strengths)
            ax.set_xlabel('Num Paths')
            ax.set_ylabel('Tunneling Strength')
            ax.set_title(f'Landscape: {landscape}')
            
            # 添加数值标注
            for i in range(len(tunneling_strengths)):
                for j in range(len(num_tunnel_paths_list)):
                    ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
            
            plt.colorbar(im, ax=ax, label='Final Loss')
        
        plt.suptitle('Quantum Tunneling Hyperparameter Tuning', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'quantum_tuning_heatmap.png'))
        plt.close()
        print("热力图已保存到: quantum_tuning_heatmap.png")
    except Exception as e:
        print(f"绘图跳过: {e}")
    
    print("\n" + "="*60)
    print("调参完成！")
    print("="*60)


if __name__ == "__main__":
    main()
