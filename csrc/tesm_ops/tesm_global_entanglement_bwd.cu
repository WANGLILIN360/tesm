/**
 * TESM Global Entanglement Backward Kernel
 * 
 * 全局纠缠反向: 计算 dQ, dK, dV
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void tesm_global_entanglement_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ Bias,
    scalar_t* __restrict__ grad_V,
    const int batch, const int seq_len, const int ent_rank, const int d_state,
    const float threshold,
    const int stride_go_b, const int stride_go_s, const int stride_go_d,
    const int stride_q_b, const int stride_q_s, const int stride_q_r,
    const int stride_k_b, const int stride_k_s, const int stride_k_r,
    const int stride_v_b, const int stride_v_s, const int stride_v_d,
    const int stride_bias_s
) {
    // 每个 thread 处理一个 (batch, j, d) 的 grad_V
    const int b = blockIdx.y;
    const int j = blockIdx.x;
    const int d = threadIdx.x;
    
    if (d >= d_state) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    
    // 遍历所有 i，累积 grad_V[j]
    float grad_v = 0.0f;
    
    for (int i = 0; i < seq_len; i++) {
        // score = Q[i] @ K[j]^T / sqrt(R) + Bias[i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t q_val = Q[b * stride_q_b + i * stride_q_s + r];
            scalar_t k_val = K[b * stride_k_b + j * stride_k_s + r];
            score += q_val * k_val;
        }
        score = score * inv_scale + Bias[i * stride_bias_s + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // grad_V[j] += ternary * grad_out[i]
        scalar_t go = grad_out[b * stride_go_b + i * stride_go_s + d];
        grad_v += ternary * go;
    }
    
    grad_V[b * stride_v_b + j * stride_v_s + d] = static_cast<scalar_t>(grad_v);
}

template <typename scalar_t>
__global__ void tesm_global_entanglement_mimo_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ Bias,
    scalar_t* __restrict__ grad_V,
    const int batch, const int seq_len, const int n_heads,
    const int ent_rank, const int d_head,
    const float threshold,
    const int stride_go_b, const int stride_go_s, const int stride_go_h, const int stride_go_d,
    const int stride_q_b, const int stride_q_s, const int stride_q_h, const int stride_q_r,
    const int stride_k_b, const int stride_k_s, const int stride_k_h, const int stride_k_r,
    const int stride_v_b, const int stride_v_s, const int stride_v_h, const int stride_v_d,
    const int stride_bias_h, const int stride_bias_s
) {
    // 每个 thread 处理一个 (batch, head, j, d) 的 grad_V
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int j = blockIdx.x;
    const int d = threadIdx.x;
    
    if (d >= d_head) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    
    float grad_v = 0.0f;
    
    for (int i = 0; i < seq_len; i++) {
        // score = Q[i] @ K[j]^T / sqrt(R) + Bias[h, i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t q_val = Q[b * stride_q_b + i * stride_q_s + h * stride_q_h + r];
            scalar_t k_val = K[b * stride_k_b + j * stride_k_s + h * stride_k_h + r];
            score += q_val * k_val;
        }
        score = score * inv_scale + Bias[h * stride_bias_h + i * stride_bias_s + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // grad_V[j] += ternary * grad_out[i]
        scalar_t go = grad_out[b * stride_go_b + i * stride_go_s + h * stride_go_h + d];
        grad_v += ternary * go;
    }
    
    grad_V[b * stride_v_b + j * stride_v_s + h * stride_v_h + d] = static_cast<scalar_t>(grad_v);
}


// SISO 版本
torch::Tensor tesm_global_entanglement_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Bias,
    float threshold
) {
    const int batch = Q.size(0);
    const int seq_len = Q.size(1);
    const int ent_rank = Q.size(2);
    const int d_state = V.size(2);
    
    auto grad_V = torch::zeros_like(V);
    
    const int threads = d_state;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "tesm_global_entanglement_bwd", ([&] {
        tesm_global_entanglement_bwd_kernel<scalar_t><<<
            dim3(seq_len, batch), threads
        >>>(
            grad_out.data_ptr<scalar_t>(),
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            Bias.data_ptr<scalar_t>(),
            grad_V.data_ptr<scalar_t>(),
            batch, seq_len, ent_rank, d_state, threshold,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            Bias.stride(0)
        );
    }));
    
    return grad_V;
}

// MIMO 版本
torch::Tensor tesm_global_entanglement_mimo_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Bias,
    float threshold
) {
    const int batch = Q.size(0);
    const int seq_len = Q.size(1);
    const int n_heads = Q.size(2);
    const int ent_rank = Q.size(3);
    const int d_head = V.size(3);
    
    auto grad_V = torch::zeros_like(V);
    
    const int threads = d_head;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "tesm_global_entanglement_mimo_bwd", ([&] {
        tesm_global_entanglement_mimo_bwd_kernel<scalar_t><<<
            dim3(seq_len, n_heads, batch), threads
        >>>(
            grad_out.data_ptr<scalar_t>(),
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            Bias.data_ptr<scalar_t>(),
            grad_V.data_ptr<scalar_t>(),
            batch, seq_len, n_heads, ent_rank, d_head, threshold,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Bias.stride(0), Bias.stride(1)
        );
    }));
    
    return grad_V;
}
