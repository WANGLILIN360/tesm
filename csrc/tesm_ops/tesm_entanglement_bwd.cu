#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void local_entanglement_bwd_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* values,
    const scalar_t* local_bias,
    const scalar_t* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_values,
    float* grad_bias,
    int64_t batch,
    int64_t seq_len,
    int64_t ent_rank,
    int64_t d_state,
    int64_t window,
    float threshold,
    float inv_scale
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * seq_len;
    if (idx >= total) {
        return;
    }
    const int64_t b = idx / seq_len;
    const int64_t q_pos = idx % seq_len;

    float raw_norm = 0.0f;
    float dot_term = 0.0f;

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
        raw_norm += fabsf(ternary);
        float grad_weight = 0.0f;
        for (int64_t d = 0; d < d_state; ++d) {
            const int64_t grad_idx = (b * seq_len + q_pos) * d_state + d;
            const int64_t value_idx = (b * seq_len + k_pos) * d_state + d;
            grad_weight += static_cast<float>(grad_out[grad_idx]) * static_cast<float>(values[value_idx]);
        }
        dot_term += grad_weight * ternary;
    }

    const float norm = raw_norm >= 1.0f ? raw_norm : 1.0f;
    const bool unclamped = raw_norm >= 1.0f;

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
        float grad_weight = 0.0f;
        for (int64_t d = 0; d < d_state; ++d) {
            const int64_t grad_idx = (b * seq_len + q_pos) * d_state + d;
            const int64_t value_idx = (b * seq_len + k_pos) * d_state + d;
            grad_weight += static_cast<float>(grad_out[grad_idx]) * static_cast<float>(values[value_idx]);
            atomicAdd(&grad_values[value_idx], static_cast<float>(grad_out[grad_idx]) * (ternary / norm));
        }
        const float sign = ternary > 0.0f ? 1.0f : (ternary < 0.0f ? -1.0f : 0.0f);
        const float grad_score = unclamped ? (grad_weight / norm - sign * dot_term / (norm * norm)) : grad_weight;
        atomicAdd(&grad_bias[w], grad_score);
        for (int64_t r = 0; r < ent_rank; ++r) {
            const int64_t q_idx = (b * seq_len + q_pos) * ent_rank + r;
            const int64_t k_idx = (b * seq_len + k_pos) * ent_rank + r;
            grad_q[q_idx] += grad_score * static_cast<float>(k[k_idx]) * inv_scale;
            atomicAdd(&grad_k[k_idx], grad_score * static_cast<float>(q[q_idx]) * inv_scale);
        }
    }
}


std::vector<at::Tensor> local_entanglement_bwd_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& values,
    const at::Tensor& local_bias,
    const at::Tensor& grad_out,
    double threshold
) {
    TORCH_CHECK(q.dim() == 3, "q must be [B, L, R]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
    TORCH_CHECK(values.dim() == 3, "values must be [B, L, D]");
    TORCH_CHECK(grad_out.sizes() == values.sizes(), "grad_out must match values");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(local_bias.is_cuda(), "local_bias must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = values.contiguous();
    auto bias_contig = local_bias.contiguous();
    auto grad_contig = grad_out.contiguous();
    const auto batch = q_contig.size(0);
    const auto seq_len = q_contig.size(1);
    const auto ent_rank = q_contig.size(2);
    const auto d_state = v_contig.size(2);
    const auto window = bias_contig.numel();

    auto grad_q_f = torch::zeros(q_contig.sizes(), q_contig.options().dtype(torch::kFloat32));
    auto grad_k_f = torch::zeros(k_contig.sizes(), k_contig.options().dtype(torch::kFloat32));
    auto grad_values_f = torch::zeros(v_contig.sizes(), v_contig.options().dtype(torch::kFloat32));
    auto grad_bias_f = torch::zeros(bias_contig.sizes(), bias_contig.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t total = batch * seq_len;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(ent_rank));
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "local_entanglement_bwd_cuda", [&] {
        local_entanglement_bwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            v_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            grad_contig.data_ptr<scalar_t>(),
            grad_q_f.data_ptr<float>(),
            grad_k_f.data_ptr<float>(),
            grad_values_f.data_ptr<float>(),
            grad_bias_f.data_ptr<float>(),
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

    auto grad_q = grad_q_f.to(q.scalar_type());
    auto grad_k = grad_k_f.to(k.scalar_type());
    auto grad_values = grad_values_f.to(values.scalar_type());
    auto grad_bias = grad_bias_f.to(local_bias.scalar_type());
    return {grad_q, grad_k, grad_values, grad_bias};
}
