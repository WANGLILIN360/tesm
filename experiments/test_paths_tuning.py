"""
测试不同 num_tunnel_paths 对隧穿率的影响
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, IterableDataset
import json
from tqdm import tqdm

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


def test_config(num_paths, strength, tokenizer, device, num_steps=500):
    config = TESMConfig.tiny()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.max_seq_len = 256
    config.ssm_cfg["annealing_enabled"] = False
    config.ssm_cfg["quantum_tunneling_enabled"] = True
    config.ssm_cfg["tunneling_strength"] = strength
    config.ssm_cfg["num_tunnel_paths"] = num_paths
    config.ssm_cfg["energy_landscape"] = "entropy"
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    dataset = PretrainDataset(
        "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl",
        tokenizer, max_length=256, max_samples=3000
    )
    dataloader = DataLoader(
        dataset, batch_size=8,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, 256),
        num_workers=0,
    )
    
    total_loss = 0
    tunnel_success = 0
    tunnel_total = 0
    step = 0
    
    for batch in dataloader:
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
        total_loss += loss.item()
        
        for layer in model.backbone.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'quantum_tunneler'):
                tunneler = layer.mixer.quantum_tunneler
                if tunneler is not None and tunneler.total_attempts > 0:
                    tunnel_success += tunneler.tunnel_success_count.item()
                    tunnel_total += tunneler.total_attempts.item()
        step += 1
    
    avg_loss = total_loss / step
    tunnel_rate = tunnel_success / tunnel_total if tunnel_total > 0 else 0
    
    return avg_loss, tunnel_rate


def main():
    print("="*60)
    print("测试 num_tunnel_paths 对隧穿率的影响")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
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
    
    # 测试不同 paths
    paths_list = [2, 4, 8, 16]
    strength = 0.2
    
    print(f"\n固定 tunneling_strength={strength}")
    print("-"*50)
    print(f"{'Paths':<10} {'平均损失':<15} {'隧穿率':<15}")
    print("-"*50)
    
    results = []
    for paths in paths_list:
        print(f"测试 paths={paths}...", end=" ", flush=True)
        loss, rate = test_config(paths, strength, tokenizer, device)
        print(f"Loss={loss:.4f}, Rate={rate*100:.1f}%")
        results.append({'paths': paths, 'loss': loss, 'rate': rate})
    
    print("-"*50)
    
    # 测试不同 strength
    print(f"\n测试不同 tunneling_strength (paths=8)")
    print("-"*50)
    print(f"{'Strength':<10} {'平均损失':<15} {'隧穿率':<15}")
    print("-"*50)
    
    strengths = [0.05, 0.1, 0.2, 0.5]
    for s in strengths:
        print(f"测试 strength={s}...", end=" ", flush=True)
        loss, rate = test_config(8, s, tokenizer, device)
        print(f"Loss={loss:.4f}, Rate={rate*100:.1f}%")
    
    print("-"*50)
    print("\n实验完成！")


if __name__ == "__main__":
    main()
