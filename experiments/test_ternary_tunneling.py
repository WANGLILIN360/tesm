"""
测试改进后的三值量子隧穿模块

验证量子隧穿作用于三值纠缠决策的效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 禁用TileLang kernel
os.environ['TILELANG_DISABLE'] = '1'

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
    tunnel_stats = {'to_positive': [], 'to_negative': [], 'to_zero': [], 'boundary': []}
    
    total_loss = 0
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
        
        # 收集三值隧穿统计
        for layer in model.backbone.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                tunneler = layer.mixer.quantum_tunneler
                if tunneler is not None:
                    tunnel_stats['to_positive'].append(tunneler.tunnel_to_positive.item())
                    tunnel_stats['to_negative'].append(tunneler.tunnel_to_negative.item())
                    tunnel_stats['to_zero'].append(tunneler.tunnel_to_zero.item())
                    tunnel_stats['boundary'].append(tunneler.total_boundary.item())
        
        step += 1
        
        if step % 50 == 0:
            avg_loss = total_loss / 50
            pbar.set_postfix_str(f"Loss={avg_loss:.4f}")
            total_loss = 0
    
    pbar.close()
    
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    
    # 计算隧穿统计
    total_tunnel = sum(tunnel_stats['to_positive']) + sum(tunnel_stats['to_negative'])
    total_boundary = sum(tunnel_stats['boundary']) if tunnel_stats['boundary'] else 1
    tunnel_rate = total_tunnel / total_boundary if total_boundary > 0 else 0
    
    return {
        'losses': losses,
        'final_loss': final_loss,
        'tunnel_rate': tunnel_rate,
        'tunnel_to_positive': sum(tunnel_stats['to_positive']),
        'tunnel_to_negative': sum(tunnel_stats['to_negative']),
        'tunnel_to_zero': sum(tunnel_stats['to_zero']),
        'total_boundary': total_boundary,
    }


def test_ternary_tunneling_mechanism():
    """测试三值隧穿机制"""
    print("\n" + "="*60)
    print("测试三值隧穿机制")
    print("="*60)
    
    from tesm_ssm.modules.tesm import TernaryQuantumTunneling
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模块
    tunnel_module = TernaryQuantumTunneling(
        threshold=0.1,
        tunneling_strength=0.1,
        num_tunnel_paths=4,
    ).to(device)
    
    # 测试不同分数范围
    print("\n测试分数分布：")
    
    # 边界分数（应该有隧穿机会）
    boundary_scores = torch.tensor([
        [0.05, 0.08, -0.05, -0.09],  # 接近阈值
        [0.12, 0.15, -0.12, -0.15],  # 刚超过阈值
        [0.01, 0.02, -0.01, -0.02],  # 接近0
    ], device=device)
    
    print(f"边界分数: {boundary_scores}")
    
    ternary_values, tunnel_info = tunnel_module.apply_tunneling(boundary_scores, training=True)
    
    print(f"隧穿后三值: {ternary_values}")
    print(f"隧穿率: {tunnel_info['tunnel_rate']*100:.1f}%")
    print(f"边界率: {tunnel_info['boundary_rate']*100:.1f}%")
    
    # 统计
    print(f"\n隧穿统计:")
    print(f"  隧穿到 +1: {tunnel_module.tunnel_to_positive.item()}")
    print(f"  隧穿到 -1: {tunnel_module.tunnel_to_negative.item()}")
    print(f"  保持 0: {tunnel_module.tunnel_to_zero.item()}")
    print(f"  边界总数: {tunnel_module.total_boundary.item()}")


def main():
    print("="*60)
    print("TESM 三值量子隧穿改进测试")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 测试机制
    test_ternary_tunneling_mechanism()
    
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
    
    tokenizer = SPTokenizer(sp)
    
    # 训练参数
    num_samples = 5000
    num_steps = 1000
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
    config_baseline.ssm_cfg["use_triton_kernels"] = False  # 禁用自定义kernel
    
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
    config_annealing.ssm_cfg["use_triton_kernels"] = False  # 禁用自定义kernel
    
    result_annealing = train_model(
        config_annealing, tokenizer, device,
        num_samples, num_steps, batch_size, max_length,
        "温度退火"
    )
    
    # ==================== 方案3: 三值量子隧穿 ====================
    config_tunnel = TESMConfig.tiny()
    config_tunnel.vocab_size = tokenizer.vocab_size
    config_tunnel.pad_token_id = tokenizer.pad_token_id
    config_tunnel.eos_token_id = tokenizer.eos_token_id
    config_tunnel.max_seq_len = max_length
    # 关键：关闭温度退火，启用量子隧穿
    config_tunnel.ssm_cfg["annealing_enabled"] = False
    config_tunnel.ssm_cfg["T_start"] = 0.1  # 确保低温
    config_tunnel.ssm_cfg["T_end"] = 0.1
    config_tunnel.ssm_cfg["quantum_tunneling_enabled"] = True
    config_tunnel.ssm_cfg["tunneling_strength"] = 0.1
    config_tunnel.ssm_cfg["num_tunnel_paths"] = 4
    config_tunnel.ssm_cfg["use_triton_kernels"] = False  # 禁用自定义kernel
    
    result_tunnel = train_model(
        config_tunnel, tokenizer, device,
        num_samples, num_steps, batch_size, max_length,
        "三值量子隧穿"
    )
    
    # ==================== 结果 ====================
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    print("\n" + "-"*80)
    print(f"{'方案':<15} {'最终损失':<12} {'隧穿率':<12} {'隧穿到+1':<12} {'隧穿到-1':<12}")
    print("-"*80)
    print(f"{'基线':<15} {result_baseline['final_loss']:<12.4f} {'-':<12} {'-':<12} {'-':<12}")
    print(f"{'温度退火':<15} {result_annealing['final_loss']:<12.4f} {'-':<12} {'-':<12} {'-':<12}")
    print(f"{'三值量子隧穿':<15} {result_tunnel['final_loss']:<12.4f} {result_tunnel['tunnel_rate']*100:.1f}%{'':<6} {result_tunnel['tunnel_to_positive']:<12} {result_tunnel['tunnel_to_negative']:<12}")
    print("-"*80)
    
    # 排名
    results_list = [
        ('基线', result_baseline['final_loss']),
        ('温度退火', result_annealing['final_loss']),
        ('三值量子隧穿', result_tunnel['final_loss']),
    ]
    results_sorted = sorted(results_list, key=lambda x: x[1])
    
    print(f"\n排名（按损失）:")
    for i, (name, loss) in enumerate(results_sorted, 1):
        print(f"  {i}. {name}: Loss={loss:.4f}")
    
    # 保存结果
    save_data = {
        'baseline': result_baseline,
        'annealing': result_annealing,
        'ternary_tunneling': result_tunnel,
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "ternary_tunneling_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已保存到: {save_path}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
