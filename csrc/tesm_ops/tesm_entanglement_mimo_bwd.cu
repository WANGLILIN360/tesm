/**
 * TESM MIMO Local Entanglement Backward Kernel
 * 
 * 多头局部窗口纠缠反向: 支持 (B, L, H, R) 格式
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
__global__ void local_entanglement_mimo_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ grad_v,
    const int64_t batch,
    const int64_t seqlen,
    const int64_t n_heads,
    const int64_t ent_rank,
    const int64_t d_head,
    const int64_t window,
    const double threshold
) {
    // 每个 thread 处理一个 (batch, head, j, d) 的 grad_V
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t j = blockIdx.x;
    const int64_t d = threadIdx.x;
    
    if (d >= d_head) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    float grad_v_acc = 0.0f;
    
    // 遍历所有 i，累积 grad_V[j]
    // 只有当 j 在 i 的窗口内时才有贡献
    for (int64_t i = 0; i < seqlen; i++) {
        // 检查 j 是否在 i 的窗口内
        if (j < std::max(int64_t(0), static_cast<int64_t>(i - window + 1)) || j > i) continue;
        
        // score = Q[i] @ K[j]^T / sqrt(R) + bias[h, i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t q_val = q[((b * seqlen + i) * n_heads + h) * ent_rank + r];
            scalar_t k_val = k[((b * seqlen + j) * n_heads + h) * ent_rank + r];
            score += q_val * k_val;
        }
        score = score * inv_scale + bias[(h * seqlen + i) * seqlen + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // grad_V[j, d] += ternary * grad_out[i, d]
        scalar_t go = grad_out[((b * seqlen + i) * n_heads + h) * d_head + d];
        grad_v_acc += ternary * go;
    }
    
    grad_v[((b * seqlen + j) * n_heads + h) * d_head + d] = static_cast<scalar_t>(grad_v_acc);
}


std::vector<at::Tensor> local_entanglement_mimo_bwd_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& bias,
    double threshold
) {
    TORCH_CHECK(grad_out.dim() == 4, "grad_out must be [B, L, H, D]");
    TORCH_CHECK(q.dim() == 4, "q must be [B, L, H, R]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
    TORCH_CHECK(v.sizes() == grad_out.sizes(), "v must match grad_out");
    TORCH_CHECK(bias.dim() == 3, "bias must be [H, L, L]");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    
    const auto batch = q.size(0);
    const auto seqlen = q.size(1);
    const auto n_heads = q.size(2);
    const auto ent_rank = q.size(3);
    const auto d_head = v.size(3);
    const int64_t window = bias.size(2);
    
    auto grad_out_contig = grad_out.contiguous();
    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto bias_contig = bias.contiguous();
    
    auto grad_v = torch::zeros_like(v_contig);
    
    const int threads = d_head;
    
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "local_entanglement_mimo_bwd_cuda", [&] {
        local_entanglement_mimo_bwd_kernel<scalar_t><<<
            dim3(seqlen, n_heads, batch), threads, 0, stream.stream()
        >>>(
            grad_out_contig.data_ptr<scalar_t>(),
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            v_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(),
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
    
    // 返回 grad_v (其他梯度用 PyTorch 计算)
    return {grad_v};
}
