#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

template <typename scalar_t>
__global__ void local_entanglement_fwd_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* values,
    const scalar_t* local_bias,
    scalar_t* out,
    int64_t batch,
    int64_t seq_len,
    int64_t ent_rank,
    int64_t d_state,
    int64_t window,
    float threshold,
    float inv_scale
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * seq_len * d_state;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / (seq_len * d_state);
    const int64_t rem = idx % (seq_len * d_state);
    const int64_t q_pos = rem / d_state;
    const int64_t d = rem % d_state;
    float accum = 0.0f;
    float norm = 0.0f;
    for (int64_t w = 0; w < window; ++w) {
        const int64_t k_pos = q_pos - (window - 1) + w;
        if (k_pos < 0 || k_pos >= seq_len) {
            continue;
        }
        float score = 0.0f;
        for (int64_t r = 0; r < ent_rank; ++r) {
            const int64_t q_idx = (b * seq_len + q_pos) * ent_rank + r;
            const int64_t k_idx = (b * seq_len + k_pos) * ent_rank + r;
            score += static_cast<float>(q[q_idx]) * static_cast<float>(k[k_idx]);
        }
        score = score * inv_scale + static_cast<float>(local_bias[w]);
        float ternary = 0.0f;
        if (score > threshold) {
            ternary = 1.0f;
        } else if (score < -threshold) {
            ternary = -1.0f;
        }
        norm += fabsf(ternary);
        if (ternary != 0.0f) {
            const int64_t v_idx = (b * seq_len + k_pos) * d_state + d;
            accum += ternary * static_cast<float>(values[v_idx]);
        }
    }
    if (norm < 1.0f) {
        norm = 1.0f;
    }
    out[idx] = static_cast<scalar_t>(accum / norm);
}


at::Tensor local_entanglement_fwd_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& values, const at::Tensor& local_bias, double threshold) {
    TORCH_CHECK(q.dim() == 3, "q must be [B, L, R]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
    TORCH_CHECK(values.dim() == 3, "values must be [B, L, D]");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(local_bias.is_cuda(), "local_bias must be CUDA");
    const auto batch = q.size(0);
    const auto seq_len = q.size(1);
    const auto ent_rank = q.size(2);
    const auto d_state = values.size(2);
    const auto window = local_bias.numel();
    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = values.contiguous();
    auto bias_contig = local_bias.contiguous();
    auto out = torch::empty_like(v_contig);
    const int threads = 256;
    const int64_t total = batch * seq_len * d_state;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(ent_rank));
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "local_entanglement_fwd_cuda", [&] {
        local_entanglement_fwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            v_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seq_len,
            ent_rank,
            d_state,
            window,
            static_cast<float>(threshold),
            inv_scale
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
