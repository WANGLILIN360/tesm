"""直接测试已保存模型的生成能力"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import sentencepiece as spm

from tesm_ssm import TESMConfig, TESMLMHeadModel

def main():
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
            self.bos_token_id = 1
            self.eos_token_id = 2
        def encode(self, text, out_type=int):
            return self.sp.encode(text, out_type=out_type)
        def decode(self, ids):
            return self.sp.decode(ids)
    
    tokenizer = SPTokenizer(sp)
    
    # 加载模型
    model_path = "/root/private_data/tesm/tesm-main-official-backup/experiments/checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        return
    
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 重建配置 - 使用base_768配置
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
            use_triton_kernels=False,  # 禁用Triton避免推理错误
            use_mimo=True,
        ),
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    model = TESMLMHeadModel(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型已加载 (Epoch {checkpoint.get('best_epoch', '?')}, Val PPL {checkpoint.get('best_val_ppl', '?'):.2f})")
    
    # 测试提示词
    test_prompts = [
        "今天天气",
        "人工智能是",
        "中国的发展",
        "学习编程的方法",
        "科技进步对社会",
    ]
    
    print("\n" + "="*60)
    print("生成测试")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"提示词: {prompt}")
        
        # 编码
        input_ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40,
            )
        
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"生成: {generated}")
    
    # 测试复述训练数据
    print("\n" + "="*60)
    print("复述训练数据测试")
    print("="*60)
    
    # 加载一些训练样本
    dataset_path = "/root/private_data/tesm/tesm_v3/dataset/pretrain_hq.jsonl"
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            try:
                item = json.loads(line.strip())
                text = item.get('text', '')
                if text:
                    samples.append(text[:100])
            except:
                continue
    
    for i, text in enumerate(samples, 1):
        print(f"\n--- 样本 {i} ---")
        print(f"原文: {text[:50]}...")
        
        # 取前10个字符作为提示
        prompt = text[:15]
        print(f"提示: {prompt}")
        
        input_ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_k=40,
            )
        
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"生成: {generated}")


if __name__ == "__main__":
    main()
