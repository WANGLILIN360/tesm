/**
 * TESM MIMO Chunked State Scan Backward Kernel
 * 
 * 多头状态扫描反向: 支持 (B, L, H, D) 格式
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void chunk_state_scan_mimo_bwd_kernel(
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ grad_states,
    scalar_t* __restrict__ grad_decay,
    scalar_t* __restrict__ grad_update,
    const int64_t batch,
    const int64_t seqlen,
    const int64_t n_heads,
    const int64_t d_head
) {
    // 每个 thread 处理一个 (batch, head, d) 的反向扫描
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t rows = batch * n_heads * d_head;
    
    if (row >= rows) return;
    
    // 解析索引
    const int64_t b = row / (n_heads * d_head);
    const int64_t remainder = row % (n_heads * d_head);
    const int64_t h = remainder / d_head;
    const int64_t d = remainder % d_head;
    
    // 反向扫描: 从后向前
    float grad_state = 0.0f;
    
    for (int64_t t = seqlen - 1; t >= 0; --t) {
        const int64_t idx = ((b * seqlen + t) * n_heads + h) * d_head + d;
        
        // grad_states 来自上游
        grad_state += static_cast<float>(grad_states[idx]);
        
        // grad_update = grad_state
        grad_update[idx] = static_cast<scalar_t>(grad_state);
        
        // grad_decay = grad_state * prev_state
        float prev_state = (t > 0) ? static_cast<float>(states[((b * seqlen + t - 1) * n_heads + h) * d_head + d]) : 0.0f;
        grad_decay[idx] = static_cast<scalar_t>(grad_state * prev_state);
        
        // 传递梯度: grad_state = grad_state * decay
        const float decay_val = max(static_cast<float>(decay[idx]), 1.0e-12f);
        grad_state = grad_state * decay_val;
    }
}


std::vector<at::Tensor> chunk_state_scan_mimo_bwd_cuda(
    const at::Tensor& decay,
    const at::Tensor& states,
    const at::Tensor& grad_states
) {
    TORCH_CHECK(decay.dim() == 4, "decay must be [B, L, H, D]");
    TORCH_CHECK(states.sizes() == decay.sizes(), "states must match decay");
    TORCH_CHECK(grad_states.sizes() == decay.sizes(), "grad_states must match decay");
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    TORCH_CHECK(grad_states.is_cuda(), "grad_states must be CUDA");
    
    const auto batch = decay.size(0);
    const auto seqlen = decay.size(1);
    const auto n_heads = decay.size(2);
    const auto d_head = decay.size(3);
    
    auto decay_contig = decay.contiguous();
    auto states_contig = states.contiguous();
    auto grad_states_contig = grad_states.contiguous();
    
    auto grad_decay = torch::empty_like(decay_contig);
    auto grad_update = torch::empty_like(decay_contig);
    
    const int threads = 256;
    const int64_t rows = batch * n_heads * d_head;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    
    at::cuda::CUDAGuard device_guard{decay.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, decay.scalar_type(), "chunk_state_scan_mimo_bwd_cuda", [&] {
        chunk_state_scan_mimo_bwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            decay_contig.data_ptr<scalar_t>(),
            states_contig.data_ptr<scalar_t>(),
            grad_states_contig.data_ptr<scalar_t>(),
            grad_decay.data_ptr<scalar_t>(),
            grad_update.data_ptr<scalar_t>(),
            batch,
            seqlen,
            n_heads,
            d_head
        );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_decay, grad_update};
}
