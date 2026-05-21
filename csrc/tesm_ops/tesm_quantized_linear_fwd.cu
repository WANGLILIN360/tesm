#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int BLOCK_K, bool HAS_BIAS>
__global__ void quantized_linear_fwd_kernel(
    const scalar_t* x,
    const scalar_t* qweight,
    const scalar_t* bias,
    scalar_t* out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    __shared__ scalar_t x_tile[BLOCK_M][BLOCK_K];
    __shared__ scalar_t w_tile[BLOCK_K][BLOCK_N];

    const int row = blockIdx.y * BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_N + threadIdx.x;
    if (row >= m || col >= n) {
        if (threadIdx.y < BLOCK_M && threadIdx.x < BLOCK_K) {
            for (int tile_k = threadIdx.x; tile_k < BLOCK_K; tile_k += BLOCK_N) {
                x_tile[threadIdx.y][tile_k] = static_cast<scalar_t>(0);
            }
        }
        if (threadIdx.y < BLOCK_K && threadIdx.x < BLOCK_N) {
            w_tile[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);
        }
    }
    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < k; k0 += BLOCK_K) {
        const int x_col = k0 + threadIdx.x;
        if (row < m && x_col < k) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * k + x_col];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);
        }

        const int w_row = k0 + threadIdx.y;
        if (w_row < k && col < n) {
            w_tile[threadIdx.y][threadIdx.x] = qweight[col * k + w_row];
        } else {
            w_tile[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            acc += static_cast<float>(x_tile[threadIdx.y][kk]) * static_cast<float>(w_tile[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row >= m || col >= n) {
        return;
    }
    if (HAS_BIAS) {
        acc += static_cast<float>(bias[col]);
    }
    out[row * n + col] = static_cast<scalar_t>(acc);
}


namespace {
at::Tensor quantized_linear_fwd_impl(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor* bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be [N, K]");
    TORCH_CHECK(x.size(-1) == qweight.size(1), "x last dim must match qweight second dim");
    if (bias != nullptr) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == qweight.size(0), "bias must match qweight first dim");
    }
    auto x_contig = x.contiguous();
    auto w_contig = qweight.contiguous();
    auto x_2d = x_contig.reshape({-1, x_contig.size(-1)});
    const auto m = x_2d.size(0);
    const auto k = x_2d.size(1);
    const auto n = w_contig.size(0);
    auto out_2d = torch::empty({m, n}, x_contig.options());
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 16;
    dim3 threads(BLOCK_N, BLOCK_M);
    dim3 blocks((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
    at::cuda::CUDAGuard device_guard{x.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x_contig.scalar_type(), "quantized_linear_fwd_cuda", [&] {
        if (bias == nullptr) {
            quantized_linear_fwd_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, false><<<blocks, threads, 0, stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                w_contig.data_ptr<scalar_t>(),
                nullptr,
                out_2d.data_ptr<scalar_t>(),
                m,
                n,
                k
            );
        } else {
            auto bias_contig = bias->contiguous();
            quantized_linear_fwd_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, true><<<blocks, threads, 0, stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                w_contig.data_ptr<scalar_t>(),
                bias_contig.data_ptr<scalar_t>(),
                out_2d.data_ptr<scalar_t>(),
                m,
                n,
                k
            );
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x_contig.sizes().vec();
    out_shape.back() = n;
    return out_2d.reshape(out_shape);
}
}


at::Tensor quantized_linear_fwd_cuda(const at::Tensor& x, const at::Tensor& qweight) {
    return quantized_linear_fwd_impl(x, qweight, nullptr);
}


at::Tensor quantized_linear_fwd_bias_cuda(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& bias) {
    return quantized_linear_fwd_impl(x, qweight, &bias);
}
