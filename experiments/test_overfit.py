"""
过拟合测试：一万条数据训练多少轮才能过拟合

测试基线模型（无温度退火、无量子隧穿）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from tesm_ssm import TESMConfig, TESMLMHeadModel
import sentencepiece as spm


class PretrainDataset(Dataset):
    """预训练数据集"""
    def __init__(self, filepath, tokenizer, max_length=256, max_samples=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if not text:
                        continue
                    # 编码文本
                    ids = tokenizer.encode(text, out_type=int)[:max_length-2]  # 预留BOS/EOS位置
                    if len(ids) < 10:
                        continue
                    # 添加特殊token: [BOS] + text + [EOS]
                    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                    self.data.append(ids)
                except:
                    continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(ids, dtype=torch.long),
        }


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


def compute_perplexity(model, dataloader, device, max_batches=50):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs, _ = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss):
                # 计算有效token数
                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
    
    model.train()
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)


def test_generation(model, tokenizer, device, prompts, max_new_tokens=50):
    """测试模型生成能力"""
    model.eval()
    results = []
    
    for prompt in prompts:
        # 编码prompt
        input_ids = torch.tensor([tokenizer.encode(prompt, out_type=int)], dtype=torch.long, device=device)
        
        # 生成
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=40,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.sp.decode(output_ids[0].tolist())
                results.append({
                    'prompt': prompt,
                    'generated': generated,
                    'new_tokens': len(output_ids[0]) - len(input_ids[0]),
                })
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'generated': f'[生成失败: {e}]',
                    'new_tokens': 0,
                })
    
    model.train()
    return results


def main():
    print("="*60)
    print("过拟合测试：一万条数据")
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
            self.bos_token_id = 1  # BOS token
            self.eos_token_id = 2  # EOS token
        def encode(self, text, out_type=int):
            return self.sp.encode(text, out_type=out_type)
    
    tokenizer = SPTokenizer(sp)
    
    # 加载数据
    print("\n加载数据...")
    dataset = PretrainDataset(
        "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl",
        tokenizer, max_length=256, max_samples=10000
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 最大批次大小
    batch_size = 64  # 适配显存
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, 256),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, 256),
        num_workers=0,
    )
    
    # 创建基线模型 - 使用base_768配置 + MIMO优化
    # base_768: 768维模型，对标 GPT-2 small
    config = TESMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_layer=12,
        d_intermediate=3072,
        max_seq_len=256,
        ssm_cfg=dict(
            d_state=256,
            expand=2,
            ent_rank=48,
            entanglement_scale=0.25,
            entanglement_threshold=0.05,
            entanglement_init=0.3,
            entanglement_window=16,
            entanglement_block_size=256,
            state_scan_chunk_size=16,
            decay_init_bias=0.0,
            annealing_enabled=False,
            T_start=10.0,
            T_end=0.1,
            annealing_steps=1000,
            annealing_schedule="cosine",
            quantum_tunneling_enabled=False,
            use_triton_kernels=True,
            use_mimo=True,
        ),
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    # 检查是否有已保存的模型
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    model_path = os.path.join(save_dir, "best_model.pt")
    
    skip_training = False
    if os.path.exists(model_path):
        print(f"\n发现已保存的模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('best_epoch', '?')
        best_val_ppl = checkpoint.get('best_val_ppl', '?')
        print(f"已加载模型 (Epoch {best_epoch}, Val PPL {best_val_ppl:.2f})")
        skip_training = True
        train_losses = []
        val_ppls = []
        overfit_epoch = None
        best_model_state = None
    else:
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")
        
        # 训练设置
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # 训练多个epoch
        num_epochs = 6  # 只训练6轮
        steps_per_epoch = len(train_loader)
        patience = 2  # 早停耐心值（连续2个epoch验证损失不下降就停止）
        patience_counter = 0
        
        print(f"\n训练配置:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  总步数: {num_epochs * steps_per_epoch}")
        
        train_losses = []
        val_ppls = []
        best_val_ppl = float('inf')
        best_model_state = None
        overfit_epoch = None
        
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
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
                
                epoch_loss += loss.item()
                epoch_steps += 1
                
                pbar.set_postfix_str(f"Loss={loss.item():.4f}")
            
            avg_train_loss = epoch_loss / epoch_steps
            train_losses.append(avg_train_loss)
            
            # 计算验证集困惑度
            val_ppl = compute_perplexity(model, val_loader, device)
            val_ppls.append(val_ppl)
            
            print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val PPL={val_ppl:.2f}")
            
            # 检测过拟合并实现早停
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = epoch + 1
                patience_counter = 0
                # 保存最佳模型
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                # 连续3个epoch验证损失不下降
                if overfit_epoch is None and len(val_ppls) >= 3:
                    if val_ppls[-1] > val_ppls[-2] > val_ppls[-3]:
                        overfit_epoch = epoch + 1
                        print(f"  *** 检测到过拟合开始于 Epoch {overfit_epoch} ***")
                
                # 早停
                if patience_counter >= patience:
                    print(f"  *** 早停: 连续{patience}个epoch验证损失未下降 ***")
                    break
    
    # 保存一些训练样本用于复述测试
    test_samples = []
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]['input_ids'].tolist()
        test_samples.append(sample)
    test_prompts = [tokenizer.sp.decode(s[:10]) for s in test_samples]  # 取前10个token作为提示
    
    # 结果
    print("\n" + "="*60)
    print("训练完成")
    print("="*60)
    
    print(f"\n最佳验证困惑度: {best_val_ppl:.2f} (Epoch {best_epoch})")
    if overfit_epoch:
        print(f"过拟合开始: Epoch {overfit_epoch}")
    else:
        print("未检测到明显过拟合")
    
    # ==================== 保存最佳模型 ====================
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pt")
    
    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'config': config.__dict__,
            'best_epoch': best_epoch,
            'best_val_ppl': best_val_ppl,
        }, model_path)
        print(f"\n最佳模型已保存: {model_path}")
    else:
        print("\n警告: 没有保存最佳模型")
    
    # ==================== 生成测试 ====================
    print("\n" + "="*60)
    print("生成测试（复述训练数据能力）")
    print("="*60)
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"已加载最佳模型 (Epoch {best_epoch})")
    
    # 禁用Triton kernel用于生成（避免推理时的kernel错误）
    model.backbone.layers[0].mixer.use_triton_kernels = False
    
    print("\n测试样本（来自训练数据）:")
    for i, (sample, prompt) in enumerate(zip(test_samples, test_prompts), 1):
        original = tokenizer.sp.decode(sample)
        print(f"\n--- 样本 {i} ---")
        print(f"原文前50字: {original[:50]}...")
        print(f"提示词: {prompt}")
        
        # 生成
        input_ids = torch.tensor([sample[:10]], dtype=torch.long, device=device)
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=40,
                )
                generated = tokenizer.sp.decode(output_ids[0].tolist())
                print(f"生成: {generated}")
                
                # 计算与原文的重叠度
                original_tokens = set(sample[:60])
                generated_tokens = set(output_ids[0].tolist()[:60])
                overlap = len(original_tokens & generated_tokens) / max(len(original_tokens), 1)
                print(f"与原文重叠率: {overlap*100:.1f}%")
            except Exception as e:
                print(f"生成失败: {e}")
    
    # 绘制曲线
    actual_epochs = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 训练损失
    axes[0].plot(range(1, actual_epochs+1), train_losses, 'b-o', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 验证困惑度
    axes[1].plot(range(1, actual_epochs+1), val_ppls, 'r-o', label='Val PPL')
    if overfit_epoch:
        axes[1].axvline(x=overfit_epoch, color='g', linestyle='--', label=f'Overfit (E{overfit_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Validation Perplexity')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "overfit_test_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n图表已保存: {save_path}")
    
    # 保存结果
    results = {
        'num_samples': len(dataset),
        'num_epochs': num_epochs,
        'train_losses': train_losses,
        'val_ppls': val_ppls,
        'best_val_ppl': best_val_ppl,
        'best_epoch': best_epoch,
        'overfit_epoch': overfit_epoch,
    }
    
    save_json = os.path.join(os.path.dirname(__file__), "overfit_test_results.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {save_json}")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    if overfit_epoch:
        print(f"一万条数据在 Epoch {overfit_epoch} 开始过拟合")
        print(f"建议训练轮数: {max(1, overfit_epoch - 1)} epochs")
    else:
        print(f"训练 {num_epochs} epochs 后仍未过拟合")
        print("建议增加训练轮数或减少数据量")


if __name__ == "__main__":
    main()
