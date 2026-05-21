/**
 * TESM Fused Output Backward Kernel
 * 
 * 融合输出反向: 计算 d_local, d_gate, d_state_proj, d_ent_proj
 * 
 * Copyright (c) 2026, TESM Project
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tesm_fused_output_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ local,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ grad_local,
    scalar_t* __restrict__ grad_gate,
    scalar_t* __restrict__ grad_state_proj,
    scalar_t* __restrict__ grad_ent_proj,
    const float ent_scale,
    const int batch, const int seq_len, const int d_state,
    const int stride_go_b, const int stride_go_s, const int stride_go_d,
    const int stride_l_b, const int stride_l_s, const int stride_l_d,
    const int stride_g_b, const int stride_g_s, const int stride_g_d,
    const int stride_gl_b, const int stride_gl_s, const int stride_gl_d,
    const int stride_gg_b, const int stride_gg_s, const int stride_gg_d,
    const int stride_gs_b, const int stride_gs_s, const int stride_gs_d,
    const int stride_ge_b, const int stride_ge_s, const int stride_ge_d
) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    for (int d = threadIdx.x; d < d_state; d += blockDim.x) {
        scalar_t go = grad_out[b * stride_go_b + i * stride_go_s + d];
        scalar_t l = local[b * stride_l_b + i * stride_l_s + d];
        scalar_t g = gate[b * stride_g_b + i * stride_g_s + d];
        
        // d_local = grad_out * gate
        grad_local[b * stride_gl_b + i * stride_gl_s + d] = go * g;
        // d_gate = grad_out * local
        grad_gate[b * stride_gg_b + i * stride_gg_s + d] = go * l;
        // d_state_proj = grad_out
        grad_state_proj[b * stride_gs_b + i * stride_gs_s + d] = go;
        // d_ent_proj = grad_out * ent_scale
        grad_ent_proj[b * stride_ge_b + i * stride_ge_s + d] = 
            go * static_cast<scalar_t>(ent_scale);
    }
}

template <typename scalar_t>
__global__ void tesm_fused_output_mimo_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ local,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ grad_local,
    scalar_t* __restrict__ grad_gate,
    scalar_t* __restrict__ grad_state_proj,
    scalar_t* __restrict__ grad_ent_proj,
    const float ent_scale,
    const int batch, const int seq_len, const int n_heads, const int d_head,
    const int stride_go_b, const int stride_go_s, const int stride_go_h, const int stride_go_d,
    const int stride_l_b, const int stride_l_s, const int stride_l_h, const int stride_l_d,
    const int stride_g_b, const int stride_g_s, const int stride_g_h, const int stride_g_d,
    const int stride_gl_b, const int stride_gl_s, const int stride_gl_h, const int stride_gl_d,
    const int stride_gg_b, const int stride_gg_s, const int stride_gg_h, const int stride_gg_d,
    const int stride_gs_b, const int stride_gs_s, const int stride_gs_h, const int stride_gs_d,
    const int stride_ge_b, const int stride_ge_s, const int stride_ge_h, const int stride_ge_d
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= seq_len) return;
    
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        scalar_t go = grad_out[b * stride_go_b + i * stride_go_s + h * stride_go_h + d];
        scalar_t l = local[b * stride_l_b + i * stride_l_s + h * stride_l_h + d];
        scalar_t g = gate[b * stride_g_b + i * stride_g_s + h * stride_g_h + d];
        
        grad_local[b * stride_gl_b + i * stride_gl_s + h * stride_gl_h + d] = go * g;
        grad_gate[b * stride_gg_b + i * stride_gg_s + h * stride_gg_h + d] = go * l;
        grad_state_proj[b * stride_gs_b + i * stride_gs_s + h * stride_gs_h + d] = go;
        grad_ent_proj[b * stride_ge_b + i * stride_ge_s + h * stride_ge_h + d] = 
            go * static_cast<scalar_t>(ent_scale);
    }
}


// SISO 版本
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
tesm_fused_output_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor local,
    torch::Tensor gate,
    float ent_scale
) {
    const int batch = local.size(0);
    const int seq_len = local.size(1);
    const int d_state = local.size(2);
    
    auto grad_local = torch::empty_like(local);
    auto grad_gate = torch::empty_like(gate);
    auto grad_state_proj = torch::empty_like(local);
    auto grad_ent_proj = torch::empty_like(local);
    
    const int threads = 128;
    const int blocks = (seq_len + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(local.scalar_type(), "tesm_fused_output_bwd", ([&] {
        tesm_fused_output_bwd_kernel<scalar_t><<<
            dim3(blocks, batch), threads
        >>>(
            grad_out.data_ptr<scalar_t>(),
            local.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            grad_local.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(),
            grad_state_proj.data_ptr<scalar_t>(),
            grad_ent_proj.data_ptr<scalar_t>(),
            ent_scale,
            batch, seq_len, d_state,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            local.stride(0), local.stride(1), local.stride(2),
            gate.stride(0), gate.stride(1), gate.stride(2),
            grad_local.stride(0), grad_local.stride(1), grad_local.stride(2),
            grad_gate.stride(0), grad_gate.stride(1), grad_gate.stride(2),
            grad_state_proj.stride(0), grad_state_proj.stride(1), grad_state_proj.stride(2),
            grad_ent_proj.stride(0), grad_ent_proj.stride(1), grad_ent_proj.stride(2)
        );
    }));
    
    return std::make_tuple(grad_local, grad_gate, grad_state_proj, grad_ent_proj);
}

// MIMO 版本
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
tesm_fused_output_mimo_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor local,
    torch::Tensor gate,
    float ent_scale
) {
    const int batch = local.size(0);
    const int seq_len = local.size(1);
    const int n_heads = local.size(2);
    const int d_head = local.size(3);
    
    auto grad_local = torch::empty_like(local);
    auto grad_gate = torch::empty_like(gate);
    auto grad_state_proj = torch::empty_like(local);
    auto grad_ent_proj = torch::empty_like(local);
    
    const int threads = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(local.scalar_type(), "tesm_fused_output_mimo_bwd", ([&] {
        tesm_fused_output_mimo_bwd_kernel<scalar_t><<<
            dim3((seq_len + threads - 1) / threads, n_heads, batch), threads
        >>>(
            grad_out.data_ptr<scalar_t>(),
            local.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            grad_local.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(),
            grad_state_proj.data_ptr<scalar_t>(),
            grad_ent_proj.data_ptr<scalar_t>(),
            ent_scale,
            batch, seq_len, n_heads, d_head,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            local.stride(0), local.stride(1), local.stride(2), local.stride(3),
            gate.stride(0), gate.stride(1), gate.stride(2), gate.stride(3),
            grad_local.stride(0), grad_local.stride(1), grad_local.stride(2), grad_local.stride(3),
            grad_gate.stride(0), grad_gate.stride(1), grad_gate.stride(2), grad_gate.stride(3),
            grad_state_proj.stride(0), grad_state_proj.stride(1), grad_state_proj.stride(2), grad_state_proj.stride(3),
            grad_ent_proj.stride(0), grad_ent_proj.stride(1), grad_ent_proj.stride(2), grad_ent_proj.stride(3)
        );
    }));
    
    return std::make_tuple(grad_local, grad_gate, grad_state_proj, grad_ent_proj);
}
