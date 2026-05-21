/**
 * TESM Fused Output Forward Kernel
 * 
 * 融合输出: out = local * gate + state_proj + ent_scale * ent_proj
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tesm_fused_output_fwd_kernel(
    const scalar_t* __restrict__ local,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ state_proj,
    const scalar_t* __restrict__ ent_proj,
    scalar_t* __restrict__ out,
    const float ent_scale,
    const int batch, const int seq_len, const int d_state,
    const int stride_l_b, const int stride_l_s, const int stride_l_d,
    const int stride_g_b, const int stride_g_s, const int stride_g_d,
    const int stride_s_b, const int stride_s_s, const int stride_s_d,
    const int stride_e_b, const int stride_e_s, const int stride_e_d,
    const int stride_o_b, const int stride_o_s, const int stride_o_d
) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    for (int d = threadIdx.x; d < d_state; d += blockDim.x) {
        scalar_t l = local[b * stride_l_b + i * stride_l_s + d];
        scalar_t g = gate[b * stride_g_b + i * stride_g_s + d];
        scalar_t s = state_proj[b * stride_s_b + i * stride_s_s + d];
        scalar_t e = ent_proj[b * stride_e_b + i * stride_e_s + d];
        
        out[b * stride_o_b + i * stride_o_s + d] = 
            l * g + s + static_cast<scalar_t>(ent_scale) * e;
    }
}

template <typename scalar_t>
__global__ void tesm_fused_output_mimo_fwd_kernel(
    const scalar_t* __restrict__ local,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ state_proj,
    const scalar_t* __restrict__ ent_proj,
    scalar_t* __restrict__ out,
    const float ent_scale,
    const int batch, const int seq_len, const int n_heads, const int d_head,
    const int stride_l_b, const int stride_l_s, const int stride_l_h, const int stride_l_d,
    const int stride_g_b, const int stride_g_s, const int stride_g_h, const int stride_g_d,
    const int stride_s_b, const int stride_s_s, const int stride_s_h, const int stride_s_d,
    const int stride_e_b, const int stride_e_s, const int stride_e_h, const int stride_e_d,
    const int stride_o_b, const int stride_o_s, const int stride_o_h, const int stride_o_d
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        scalar_t l = local[b * stride_l_b + i * stride_l_s + h * stride_l_h + d];
        scalar_t g = gate[b * stride_g_b + i * stride_g_s + h * stride_g_h + d];
        scalar_t s = state_proj[b * stride_s_b + i * stride_s_s + h * stride_s_h + d];
        scalar_t e = ent_proj[b * stride_e_b + i * stride_e_s + h * stride_e_h + d];
        
        out[b * stride_o_b + i * stride_o_s + h * stride_o_h + d] = 
            l * g + s + static_cast<scalar_t>(ent_scale) * e;
    }
}


// SISO 版本
torch::Tensor tesm_fused_output_fwd_cuda(
    torch::Tensor local,
    torch::Tensor gate,
    torch::Tensor state_proj,
    torch::Tensor ent_proj,
    float ent_scale
) {
    const int batch = local.size(0);
    const int seq_len = local.size(1);
    const int d_state = local.size(2);
    
    auto out = torch::empty_like(local);
    
    const int threads = 128;
    const int blocks = (seq_len + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(local.scalar_type(), "tesm_fused_output_fwd", ([&] {
        tesm_fused_output_fwd_kernel<scalar_t><<<
            dim3(blocks, batch), threads
        >>>(
            local.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            state_proj.data_ptr<scalar_t>(),
            ent_proj.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            ent_scale,
            batch, seq_len, d_state,
            local.stride(0), local.stride(1), local.stride(2),
            gate.stride(0), gate.stride(1), gate.stride(2),
            state_proj.stride(0), state_proj.stride(1), state_proj.stride(2),
            ent_proj.stride(0), ent_proj.stride(1), ent_proj.stride(2),
            out.stride(0), out.stride(1), out.stride(2)
        );
    }));
    
    return out;
}

// MIMO 版本
torch::Tensor tesm_fused_output_mimo_fwd_cuda(
    torch::Tensor local,
    torch::Tensor gate,
    torch::Tensor state_proj,
    torch::Tensor ent_proj,
    float ent_scale
) {
    const int batch = local.size(0);
    const int seq_len = local.size(1);
    const int n_heads = local.size(2);
    const int d_head = local.size(3);
    
    auto out = torch::empty_like(local);
    
    const int threads = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(local.scalar_type(), "tesm_fused_output_mimo_fwd", ([&] {
        tesm_fused_output_mimo_fwd_kernel<scalar_t><<<
            dim3((seq_len + threads - 1) / threads, n_heads, batch), threads
        >>>(
            local.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            state_proj.data_ptr<scalar_t>(),
            ent_proj.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            ent_scale,
            batch, seq_len, n_heads, d_head,
            local.stride(0), local.stride(1), local.stride(2), local.stride(3),
            gate.stride(0), gate.stride(1), gate.stride(2), gate.stride(3),
            state_proj.stride(0), state_proj.stride(1), state_proj.stride(2), state_proj.stride(3),
            ent_proj.stride(0), ent_proj.stride(1), ent_proj.stride(2), ent_proj.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3)
        );
    }));
    
    return out;
}
