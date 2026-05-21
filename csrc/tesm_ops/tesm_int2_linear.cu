/** INT2 量化线性层 CUDA kernel
 * 
 * 权重格式：4 个 INT2 打包成 1 个 INT8
 * 编码：-1→0b00, 0→0b01, +1→0b10
 * 
 * 计算流程：
 * 1. 加载打包的 INT2 权重
 * 2. 解包为 INT8
 * 3. 使用 DP4A 指令做 INT8 点积
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 解包 4 个 INT2 到 4 个 INT8
__device__ __forceinline__ void decode_int2_to_int8(uint8_t packed, int8_t* decoded) {
    // packed: 0b_v3_v2_v1_v0，每个 vi 是 2 bit
    // 解码：0b00→-1, 0b01→0, 0b10→+1
    decoded[0] = (packed & 0x03) - 1;       // v0
    decoded[1] = ((packed >> 2) & 0x03) - 1; // v1
    decoded[2] = ((packed >> 4) & 0x03) - 1; // v2
    decoded[3] = ((packed >> 6) & 0x03) - 1; // v3
}

// 使用 DP4A 指令的 INT8 点积
__device__ __forceinline__ int dp4a_int8(const int8_t* a, const int8_t* b) {
    int result = 0;
    
    // 方法1：使用内联 PTX dp4a 指令（需要 Volta+ 架构）
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile(
        "dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(*reinterpret_cast<const int*>(a)), 
          "r"(*reinterpret_cast<const int*>(b)), 
          "r"(result)
    );
    #else
    // 方法2：回退到普通计算
    for (int i = 0; i < 4; i++) {
        result += a[i] * b[i];
    }
    #endif
    
    return result;
}

// INT2 线性层前向 kernel
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, bool HAS_BIAS>
__global__ void int2_linear_fwd_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float scale,
    int64_t m,
    int64_t n,
    int64_t k
) {
    // 共享内存
    __shared__ float x_tile[BLOCK_M][BLOCK_K];
    __shared__ int8_t w_tile[BLOCK_K][BLOCK_N];
    
    const int row = blockIdx.y * BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_N + threadIdx.x;
    
    float acc = 0.0f;
    
    // 遍历 K 维度（打包后是 k // 4）
    for (int64_t k0 = 0; k0 < k; k0 += BLOCK_K) {
        // 加载 x 块到共享内存
        const int x_col = k0 + threadIdx.x;
        if (row < m && x_col < k) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * k + x_col];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 加载权重块到共享内存（需要解包）
        const int w_row = k0 + threadIdx.y;
        const int packed_k = k / 4;
        const int packed_row = w_row / 4;
        const int packed_offset = (w_row % 4) * 2;  // 每个 packed 包含 4 个 INT2
        
        if (col < n) {
            if (w_row < k && packed_row < packed_k) {
                uint8_t packed = packed_weight[col * packed_k + packed_row];
                // 解包
                int8_t decoded[4];
                decode_int2_to_int8(packed, decoded);
                // 选择正确的元素
                w_tile[threadIdx.y][threadIdx.x] = decoded[w_row % 4];
            } else {
                w_tile[threadIdx.y][threadIdx.x] = 0;
            }
        } else {
            w_tile[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // 计算点积
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            acc += x_tile[threadIdx.y][kk] * w_tile[kk][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 应用缩放
    acc /= scale;
    
    // 添加偏置
    if (HAS_BIAS && col < n) {
        acc += bias[col];
    }
    
    // 写入输出
    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

// 简化版 kernel：直接在全局内存解包
template <int BLOCK_M, int BLOCK_N, bool HAS_BIAS>
__global__ void int2_linear_simple_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float scale,
    int64_t m,
    int64_t n,
    int64_t k
) {
    const int row = blockIdx.y * BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_N + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float acc = 0.0f;
    const int packed_k = k / 4;
    
    // 遍历 K 维度
    for (int64_t k0 = 0; k0 < k; k0 += 4) {
        const int packed_idx = k0 / 4;
        
        // 加载打包的权重
        uint8_t packed = packed_weight[col * packed_k + packed_idx];
        
        // 解包
        int8_t w[4];
        decode_int2_to_int8(packed, w);
        
        // 点积
        for (int i = 0; i < 4; i++) {
            if (k0 + i < k) {
                acc += x[row * k + k0 + i] * w[i];
            }
        }
    }
    
    // 应用缩放
    acc /= scale;
    
    // 添加偏置
    if (HAS_BIAS) {
        acc += bias[col];
    }
    
    out[row * n + col] = acc;
}

// C++ 接口
at::Tensor int2_linear_fwd_cuda(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scale,
    const at::Tensor* bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "packed_weight must be uint8");
    TORCH_CHECK(scale.numel() == 1, "scale must be scalar");
    
    auto x_contig = x.contiguous();
    auto w_contig = packed_weight.contiguous();
    auto x_2d = x_contig.reshape({-1, x_contig.size(-1)});
    
    const auto m = x_2d.size(0);
    const auto k = x_2d.size(1);
    const auto n = w_contig.size(0);
    
    TORCH_CHECK(k % 4 == 0, "K must be divisible by 4");
    TORCH_CHECK(w_contig.size(1) == k / 4, "packed_weight second dim must be K/4");
    
    auto out = torch::empty({m, n}, x_contig.options());
    
    const float scale_val = scale.item<float>();
    
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    
    dim3 threads(BLOCK_N, BLOCK_M);
    dim3 blocks((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
    
    at::cuda::CUDAGuard device_guard{x.device()};
    auto stream = at::cuda::getCurrentCUDAStream();
    
    if (bias == nullptr) {
        int2_linear_simple_kernel<BLOCK_M, BLOCK_N, false><<<blocks, threads, 0, stream.stream()>>>(
            x_2d.data_ptr<float>(),
            w_contig.data_ptr<uint8_t>(),
            nullptr,
            out.data_ptr<float>(),
            scale_val,
            m, n, k
        );
    } else {
        auto bias_contig = bias->contiguous();
        int2_linear_simple_kernel<BLOCK_M, BLOCK_N, true><<<blocks, threads, 0, stream.stream()>>>(
            x_2d.data_ptr<float>(),
            w_contig.data_ptr<uint8_t>(),
            bias_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            scale_val,
            m, n, k
        );
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    auto out_shape = x_contig.sizes().vec();
    out_shape.back() = n;
    return out.reshape(out_shape);
}
