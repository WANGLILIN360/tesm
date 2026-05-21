/**
 * TESM Global Entanglement Forward Kernel
 * 
 * 全局纠缠前向: O(L^2) ternary attention
 * 支持 SISO (B, L, R) 和 MIMO (B, L, H, R) 格式
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void tesm_global_entanglement_fwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ Bias,
    scalar_t* __restrict__ Out,
    const int batch, const int seq_len, const int ent_rank, const int d_state,
    const float threshold,
    const int stride_q_b, const int stride_q_s, const int stride_q_r,
    const int stride_k_b, const int stride_k_s, const int stride_k_r,
    const int stride_v_b, const int stride_v_s, const int stride_v_d,
    const int stride_out_b, const int stride_out_s, const int stride_out_d,
    const int stride_bias_s
) {
    // 每个 thread 处理一个 (batch, position) 的输出
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    
    // 加载 Q[i]
    extern __shared__ char shared_mem[];
    scalar_t* q_local = reinterpret_cast<scalar_t*>(shared_mem);
    
    for (int r = threadIdx.x; r < ent_rank; r += blockDim.x) {
        q_local[r] = Q[b * stride_q_b + i * stride_q_s + r];
    }
    __syncthreads();
    
    // 累积器
    float acc[16] = {0.0f};  // 假设 d_state <= 16 * blockDim.x
    float norm = 0.0f;
    
    // 遍历所有位置 j
    for (int j = 0; j < seq_len; j++) {
        // 计算 score = Q[i] @ K[j]^T / sqrt(R) + Bias[i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t k_val = K[b * stride_k_b + j * stride_k_s + r];
            score += q_local[r] * k_val;
        }
        score = score * inv_scale + Bias[i * stride_bias_s + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // 累积 V[j]
        for (int d = threadIdx.x; d < d_state; d += blockDim.x) {
            scalar_t v_val = V[b * stride_v_b + j * stride_v_s + d];
            acc[d / blockDim.x] += ternary * v_val;
        }
        norm += fabsf(ternary);
    }
    
    // 归一化并存储
    norm = fmaxf(norm, 1.0f);
    for (int d = threadIdx.x; d < d_state; d += blockDim.x) {
        Out[b * stride_out_b + i * stride_out_s + d] = 
            static_cast<scalar_t>(acc[d / blockDim.x] / norm);
    }
}

template <typename scalar_t>
__global__ void tesm_global_entanglement_mimo_fwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ Bias,
    scalar_t* __restrict__ Out,
    const int batch, const int seq_len, const int n_heads, 
    const int ent_rank, const int d_head,
    const float threshold,
    const int stride_q_b, const int stride_q_s, const int stride_q_h, const int stride_q_r,
    const int stride_k_b, const int stride_k_s, const int stride_k_h, const int stride_k_r,
    const int stride_v_b, const int stride_v_s, const int stride_v_h, const int stride_v_d,
    const int stride_out_b, const int stride_out_s, const int stride_out_h, const int stride_out_d,
    const int stride_bias_h, const int stride_bias_s
) {
    // 每个 thread 处理一个 (batch, head, position) 的输出
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    const float inv_scale = 1.0f / sqrtf((float)ent_rank);
    
    // 加载 Q[b, i, h]
    extern __shared__ char shared_mem[];
    scalar_t* q_local = reinterpret_cast<scalar_t*>(shared_mem);
    
    for (int r = threadIdx.x; r < ent_rank; r += blockDim.x) {
        q_local[r] = Q[b * stride_q_b + i * stride_q_s + h * stride_q_h + r];
    }
    __syncthreads();
    
    // 累积器
    float acc[16] = {0.0f};
    float norm = 0.0f;
    
    // 遍历所有位置 j
    for (int j = 0; j < seq_len; j++) {
        // score = Q[i] @ K[j]^T / sqrt(R) + Bias[h, i, j]
        float score = 0.0f;
        for (int r = 0; r < ent_rank; r++) {
            scalar_t k_val = K[b * stride_k_b + j * stride_k_s + h * stride_k_h + r];
            score += q_local[r] * k_val;
        }
        score = score * inv_scale + Bias[h * stride_bias_h + i * stride_bias_s + j];
        
        // Ternary weight
        float ternary = 0.0f;
        if (score > threshold) ternary = 1.0f;
        else if (score < -threshold) ternary = -1.0f;
        
        // 累积 V[b, j, h]
        for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
            scalar_t v_val = V[b * stride_v_b + j * stride_v_s + h * stride_v_h + d];
            acc[d / blockDim.x] += ternary * v_val;
        }
        norm += fabsf(ternary);
    }
    
    // 归一化并存储
    norm = fmaxf(norm, 1.0f);
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        Out[b * stride_out_b + i * stride_out_s + h * stride_out_h + d] = 
            static_cast<scalar_t>(acc[d / blockDim.x] / norm);
    }
}


// SISO 版本: Q, K, V, Bias 都是 (B, L, R/D) 或 (L, L)
torch::Tensor tesm_global_entanglement_fwd_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Bias,
    float threshold
) {
    const int batch = Q.size(0);
    const int seq_len = Q.size(1);
    const int ent_rank = Q.size(2);
    const int d_state = V.size(2);
    
    auto Out = torch::empty({batch, seq_len, d_state}, Q.options());
    
    const int threads = 128;
    const int blocks = (seq_len + threads - 1) / threads;
    
    const size_t shared_mem_size = ent_rank * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "tesm_global_entanglement_fwd", ([&] {
        tesm_global_entanglement_fwd_kernel<scalar_t><<<
            dim3(blocks, batch), threads, shared_mem_size
        >>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            Bias.data_ptr<scalar_t>(),
            Out.data_ptr<scalar_t>(),
            batch, seq_len, ent_rank, d_state, threshold,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            Bias.stride(0)
        );
    }));
    
    return Out;
}

// MIMO 版本: Q, K, V 是 (B, L, H, R/D), Bias 是 (H, L, L)
torch::Tensor tesm_global_entanglement_mimo_fwd_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Bias,
    float threshold
) {
    const int batch = Q.size(0);
    const int seq_len = Q.size(1);
    const int n_heads = Q.size(2);
    const int ent_rank = Q.size(3);
    const int d_head = V.size(3);
    
    auto Out = torch::empty({batch, seq_len, n_heads, d_head}, Q.options());
    
    const int threads = 128;
    const size_t shared_mem_size = ent_rank * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "tesm_global_entanglement_mimo_fwd", ([&] {
        tesm_global_entanglement_mimo_fwd_kernel<scalar_t><<<
            dim3((seq_len + threads - 1) / threads, n_heads, batch), threads, shared_mem_size
        >>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            Bias.data_ptr<scalar_t>(),
            Out.data_ptr<scalar_t>(),
            batch, seq_len, n_heads, ent_rank, d_head, threshold,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            Bias.stride(0), Bias.stride(1)
        );
    }));
    
    return Out;
}
