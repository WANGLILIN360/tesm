/**
 * TESM MIMO Local Entanglement Forward Kernel
 * 
 * 多头局部窗口纠缠前向: 支持 (B, L, H, R) 格式
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void local_entanglement_mimo_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    const int64_t batch,
    const int64_t seqlen,
    const int64_t n_heads,
    const int64_t ent_rank,
    const int64_t d_head,
    const int64_t window,
    const double threshold
) {
    // 每个 thread 处理一个 (batch, head, position) 的输出
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seqlen) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    
    // 加载 Q[b, i, h]
    extern __shared__ char shared_mem[];
    scalar_t* q_local = reinterpret_cast<scalar_t*>(shared_mem);
    
    for (int r = threadIdx.x; r < ent_rank; r += blockDim.x) {
        q_local[r] = q[((b * seqlen + i) * n_heads + h) * ent_rank + r];
    }
    __syncthreads();
    
    // 累积器
    float acc[16] = {0.0f};
    float norm = 0.0f;
    
    // 计算窗口范围
    const int64_t j_start = std::max(int64_t(0), static_cast<int64_t>(i - window + 1));
    const int64_t j_end = std::min(static_cast<int64_t>(seqlen), static_cast<int64_t>(i + 1));
    
    // 遍历窗口内的位置 j
    for (int64_t j = j_start; j < j_end; j++) {
        // score = Q[i] @ K[j]^T / sqrt(R) + bias[h, i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t k_val = k[((b * seqlen + j) * n_heads + h) * ent_rank + r];
            score += q_local[r] * k_val;
        }
        score = score * inv_scale + bias[(h * seqlen + i) * seqlen + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // 累积 V[j]
        for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
            scalar_t v_val = v[((b * seqlen + j) * n_heads + h) * d_head + d];
            acc[d / blockDim.x] += ternary * v_val;
        }
        norm += fabsf(ternary);
    }
    
    // 归一化并存储
    norm = fmaxf(norm, 1.0f);
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        out[((b * seqlen + i) * n_heads + h) * d_head + d] = 
            static_cast<scalar_t>(acc[d / blockDim.x] / norm);
    }
}


at::Tensor local_entanglement_mimo_fwd_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& bias,
    double threshold
) {
    TORCH_CHECK(q.dim() == 4, "q must be [B, L, H, R]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
    TORCH_CHECK(v.dim() == 4, "v must be [B, L, H, D]");
    TORCH_CHECK(v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(2) == q.size(2), "v dimensions must match q");
    TORCH_CHECK(bias.dim() == 3, "bias must be [H, L, L]");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    
    const auto batch = q.size(0);
    const auto seqlen = q.size(1);
    const auto n_heads = q.size(2);
    const auto ent_rank = q.size(3);
    const auto d_head = v.size(3);
    const int64_t window = bias.size(2);  // window size from bias shape
    
    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto bias_contig = bias.contiguous();
    auto out = torch::empty_like(v_contig);
    
    const int threads = 128;
    const size_t shared_mem_size = ent_rank * sizeof(float);
    
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "local_entanglement_mimo_fwd_cuda", [&] {
        local_entanglement_mimo_fwd_kernel<scalar_t><<<
            dim3((seqlen + threads - 1) / threads, n_heads, batch), threads, shared_mem_size, stream.stream()
        >>>(
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            v_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seqlen,
            n_heads,
            ent_rank,
            d_head,
            window,
            threshold
        );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
