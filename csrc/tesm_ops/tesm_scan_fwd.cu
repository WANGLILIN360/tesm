#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void chunk_state_scan_fwd_kernel(
    const scalar_t* decay,
    const scalar_t* update,
    scalar_t* out,
    int64_t batch,
    int64_t seqlen,
    int64_t dstate
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t rows = batch * dstate;
    if (row >= rows) {
        return;
    }
    const int64_t b = row / dstate;
    const int64_t d = row % dstate;
    float state = 0.0f;
    for (int64_t t = 0; t < seqlen; ++t) {
        const int64_t idx = (b * seqlen + t) * dstate + d;
        const float decay_val = max(static_cast<float>(decay[idx]), 1.0e-12f);
        const float update_val = static_cast<float>(update[idx]);
        state = decay_val * state + update_val;
        out[idx] = static_cast<scalar_t>(state);
    }
}


at::Tensor chunk_state_scan_fwd_cuda(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size) {
    TORCH_CHECK(decay.dim() == 3, "decay must be [B, L, D]");
    TORCH_CHECK(update.sizes() == decay.sizes(), "update must match decay");
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(update.is_cuda(), "update must be CUDA");
    const auto batch = decay.size(0);
    const auto seqlen = decay.size(1);
    const auto dstate = decay.size(2);
    (void)chunk_size;
    auto decay_contig = decay.contiguous();
    auto update_contig = update.contiguous();
    auto out = torch::empty_like(update_contig);
    const int threads = 256;
    const int64_t rows = batch * dstate;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    at::cuda::CUDAGuard device_guard{decay.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, decay.scalar_type(), "chunk_state_scan_fwd_cuda", [&] {
        chunk_state_scan_fwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            decay_contig.data_ptr<scalar_t>(),
            update_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seqlen,
            dstate
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
