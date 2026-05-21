#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void quantized_linear_grad_input_kernel(
    const scalar_t* grad_output,
    const scalar_t* qweight,
    scalar_t* grad_input,
    int64_t m,
    int64_t n,
    int64_t k
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * k;
    if (idx >= total) {
        return;
    }
    const int64_t row = idx / k;
    const int64_t col = idx % k;
    float acc = 0.0f;
    for (int64_t nn = 0; nn < n; ++nn) {
        const int64_t go_idx = row * n + nn;
        const int64_t w_idx = nn * k + col;
        acc += static_cast<float>(grad_output[go_idx]) * static_cast<float>(qweight[w_idx]);
    }
    grad_input[idx] = static_cast<scalar_t>(acc);
}


template <typename scalar_t>
__global__ void quantized_linear_grad_weight_kernel(
    const scalar_t* grad_output,
    const scalar_t* x,
    scalar_t* grad_weight,
    int64_t m,
    int64_t n,
    int64_t k
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * k;
    if (idx >= total) {
        return;
    }
    const int64_t row = idx / k;
    const int64_t col = idx % k;
    float acc = 0.0f;
    for (int64_t mm = 0; mm < m; ++mm) {
        const int64_t go_idx = mm * n + row;
        const int64_t x_idx = mm * k + col;
        acc += static_cast<float>(grad_output[go_idx]) * static_cast<float>(x[x_idx]);
    }
    grad_weight[idx] = static_cast<scalar_t>(acc);
}


template <typename scalar_t>
__global__ void quantized_linear_grad_bias_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_bias,
    int64_t m,
    int64_t n
) {
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int64_t row = 0; row < m; ++row) {
        acc += static_cast<float>(grad_output[row * n + col]);
    }
    grad_bias[col] = static_cast<scalar_t>(acc);
}


std::vector<at::Tensor> quantized_linear_bwd_cuda(const at::Tensor& grad_output, const at::Tensor& x, const at::Tensor& qweight, bool has_bias) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be [N, K]");
    TORCH_CHECK(grad_output.size(-1) == qweight.size(0), "grad_output last dim must match qweight first dim");
    TORCH_CHECK(x.size(-1) == qweight.size(1), "x last dim must match qweight second dim");

    auto grad_output_contig = grad_output.contiguous();
    auto x_contig = x.contiguous();
    auto w_contig = qweight.contiguous();
    auto grad_output_2d = grad_output_contig.reshape({-1, grad_output_contig.size(-1)});
    auto x_2d = x_contig.reshape({-1, x_contig.size(-1)});
    const auto m = grad_output_2d.size(0);
    const auto n = grad_output_2d.size(1);
    const auto k = x_2d.size(1);
    auto grad_input_2d = torch::empty({m, k}, x_contig.options());
    auto grad_weight = torch::empty({n, k}, w_contig.options());

    const int threads = 256;
    const int blocks_input = static_cast<int>(((m * k) + threads - 1) / threads);
    const int blocks_weight = static_cast<int>(((n * k) + threads - 1) / threads);
    const int blocks_bias = static_cast<int>((n + threads - 1) / threads);
    at::cuda::CUDAGuard device_guard{grad_output.device()};
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_output_contig.scalar_type(), "quantized_linear_bwd_cuda", [&] {
        quantized_linear_grad_input_kernel<scalar_t><<<blocks_input, threads, 0, stream.stream()>>>(
            grad_output_2d.data_ptr<scalar_t>(),
            w_contig.data_ptr<scalar_t>(),
            grad_input_2d.data_ptr<scalar_t>(),
            m,
            n,
            k
        );
        quantized_linear_grad_weight_kernel<scalar_t><<<blocks_weight, threads, 0, stream.stream()>>>(
            grad_output_2d.data_ptr<scalar_t>(),
            x_2d.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            m,
            n,
            k
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_input = grad_input_2d.reshape(x_contig.sizes().vec());
    if (has_bias) {
        auto grad_bias = torch::empty({n}, grad_output_contig.options());
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_output_contig.scalar_type(), "quantized_linear_grad_bias_cuda", [&] {
            quantized_linear_grad_bias_kernel<scalar_t><<<blocks_bias, threads, 0, stream.stream()>>>(
                grad_output_2d.data_ptr<scalar_t>(),
                grad_bias.data_ptr<scalar_t>(),
                m,
                n
            );
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {grad_input, grad_weight, grad_bias};
    }
    return {grad_input, grad_weight};
}
