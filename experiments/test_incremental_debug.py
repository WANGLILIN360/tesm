"""调试增量推理性能"""
import sys
sys.path.insert(0, "d:/tesm-main-official-backup/tesm-main-official-backup")

import torch
import time

from tesm_ssm.modules.tesm import TESM

device = torch.device("cuda")
d_model = 512
d_state = 256
ent_rank = 48
window = 16

print("创建 TESM 模型...")
model = TESM(
    d_model=d_model,
    d_state=d_state,
    expand=2,
    ent_rank=ent_rank,
    entanglement_window=window,
    max_seq_len=2048,
    dropout=0.0,
    device=device,
)
model.eval()

# 分配缓存
batch_size = 1
prompt_len = 256
gen_len = 16

cache = model.allocate_inference_cache(batch_size, prompt_len + gen_len)
inference_params = {'state_cache': cache}

print(f"\n缓存结构: {list(cache.keys())}")
print(f"ent_k_cache shape: {cache['ent_k_cache'].shape}")
print(f"cache_idx: {cache.get('cache_idx', 'N/A')}")

# Prefill
print(f"\n=== Prefill ({prompt_len} tokens) ===")
prompt = torch.randn(batch_size, prompt_len, d_model, device=device)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    out = model(prompt, inference_params=inference_params)
    torch.cuda.synchronize()
    prefill_time = (time.time() - start) * 1000
    
    if isinstance(out, tuple):
        out = out[0]
    print(f"Output shape: {out.shape}")
    print(f"Prefill time: {prefill_time:.2f} ms")
    print(f"After prefill - cache_idx: {cache.get('cache_idx', 'N/A')}, seq_pos: {cache['seq_pos']}")

# Incremental
print(f"\n=== Incremental Generation ({gen_len} tokens) ===")
times = []
for i in range(gen_len):
    token = torch.randn(batch_size, 1, d_model, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    out = model(token, inference_params=inference_params)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    
    if isinstance(out, tuple):
        out = out[0]
    
    if i == 0:
        print(f"First token output shape: {out.shape}")
        print(f"First token - cache_idx: {cache.get('cache_idx', 'N/A')}, seq_pos: {cache['seq_pos']}")

avg_time = sum(times) / len(times)
print(f"\n平均单token时间: {avg_time:.2f} ms")
print(f"各token时间: {[f'{t:.1f}' for t in times]}")

# 对比无缓存版本
print(f"\n=== 无缓存对比 ===")
token = torch.randn(batch_size, 1, d_model, device=device)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    out_no_cache = model(token)  # 无缓存
    torch.cuda.synchronize()
    no_cache_time = (time.time() - start) * 1000
    print(f"无缓存单token时间: {no_cache_time:.2f} ms")

print(f"\n使用缓存 vs 无缓存: {avg_time:.2f} ms vs {no_cache_time:.2f} ms")
