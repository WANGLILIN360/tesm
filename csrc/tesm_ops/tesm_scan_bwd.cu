#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void chunk_state_scan_bwd_kernel(
    const scalar_t* decay,
    const scalar_t* states,
    const scalar_t* grad_states,
    scalar_t* grad_decay,
    scalar_t* grad_update,
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
    float grad_next = 0.0f;
    for (int64_t t = seqlen - 1; t >= 0; --t) {
        const int64_t idx = (b * seqlen + t) * dstate + d;
        const float grad_state = static_cast<float>(grad_states[idx]) + grad_next;
        grad_update[idx] = static_cast<scalar_t>(grad_state);
        if (t > 0) {
            const int64_t prev_idx = (b * seqlen + (t - 1)) * dstate + d;
            grad_decay[idx] = static_cast<scalar_t>(grad_state * static_cast<float>(states[prev_idx]));
        } else {
            grad_decay[idx] = static_cast<scalar_t>(0.0f);
        }
        grad_next = grad_state * static_cast<float>(decay[idx]);
    }
}


std::vector<at::Tensor> chunk_state_scan_bwd_cuda(const at::Tensor& decay, const at::Tensor& states, const at::Tensor& grad_states) {
    TORCH_CHECK(decay.dim() == 3, "decay must be [B, L, D]");
    TORCH_CHECK(states.sizes() == decay.sizes(), "states must match decay");
    TORCH_CHECK(grad_states.sizes() == decay.sizes(), "grad_states must match decay");
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    TORCH_CHECK(grad_states.is_cuda(), "grad_states must be CUDA");
    const auto batch = decay.size(0);
    const auto seqlen = decay.size(1);
    const auto dstate = decay.size(2);
    auto decay_contig = decay.contiguous();
    auto states_contig = states.contiguous();
    auto grad_states_contig = grad_states.contiguous();
    auto grad_decay = torch::zeros_like(decay_contig);
    auto grad_update = torch::zeros_like(states_contig);
    const int threads = 256;
    const int64_t rows = batch * dstate;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    at::cuda::CUDAGuard device_guard{decay.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, decay.scalar_type(), "chunk_state_scan_bwd_cuda", [&] {
        chunk_state_scan_bwd_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            decay_contig.data_ptr<scalar_t>(),
            states_contig.data_ptr<scalar_t>(),
            grad_states_contig.data_ptr<scalar_t>(),
            grad_decay.data_ptr<scalar_t>(),
            grad_update.data_ptr<scalar_t>(),
            batch,
            seqlen,
            dstate
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_decay, grad_update};
}
