/**
 * TESM MIMO Chunked State Scan Forward Kernel
 * 
 * 多头状态扫描前向: 支持 (B, L, H, D) 格式
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void chunk_state_scan_mimo_fwd_kernel(
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ out,
    const int64_t batch,
    const int64_t seqlen,
    const int64_t n_heads,
    const int64_t d_head
) {
    // 每个 thread 处理一个 (batch, head, d) 的状态序列
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t rows = batch * n_heads * d_head;
    
    if (row >= rows) return;
    
    // 解析索引
    const int64_t b = row / (n_heads * d_head);
    const int64_t remainder = row % (n_heads * d_head);
    const int64_t h = remainder / d_head;
    const int64_t d = remainder % d_head;
    
    // 状态扫描
    float state = 0.0f;
    for (int64_t t = 0; t < seqlen; ++t) {
        const int64_t idx = ((b * seqlen + t) * n_heads + h) * d_head + d;
        const float decay_val = max(static_cast<float>(decay[idx]), 1.0e-12f);
        const float update_val = static_cast<float>(update[idx]);
        state = decay_val * state + update_val;
        out[idx] = static_cast<scalar_t>(state);
    }
}


at::Tensor chunk_state_scan_mimo_fwd_cuda(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size) {
    TORCH_CHECK(decay.dim() == 4, "decay must be [B, L, H, D]");
    TORCH_CHECK(update.sizes() == decay.sizes(), "update must match decay");
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(update.is_cuda(), "update must be CUDA");
    
    const auto batch = decay.size(0);
    const auto seqlen = decay.size(1);
    const auto n_heads = decay.size(2);
    const auto d_head = decay.size(3);
    (void)chunk_size;
    
    auto decay_contig = decay.contiguous();
    auto update_contig = update.contiguous();
    auto out = torch::empty_like(update_contig);
    
    const int threads = 256;
    const int64_t rows = batch * n_heads * d_head;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    
    at::cuda::CUDAGuard device_guard{decay.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, decay.scalar_type(), "chunk_state_scan_mimo_fwd_cuda", [&] {
        chunk_state_scan_mimo_fwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            decay_contig.data_ptr<scalar_t>(),
            update_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seqlen,
            n_heads,
            d_head
        );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
