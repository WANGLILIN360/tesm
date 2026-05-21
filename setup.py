from pathlib import Path
import os

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except Exception:
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None


def _get_ext_modules():
    # 环境变量控制是否编译 CUDA 扩展
    # TESM_BUILD_CUDA=0 可禁用 CUDA 编译
    if CUDAExtension is None or CUDA_HOME is None:
        return []
    if os.environ.get('TESM_BUILD_CUDA', '1') == '0':
        print("TESM_BUILD_CUDA=0, skipping CUDA extension build")
        return []
    
    root = Path(__file__).resolve().parent
    source_root = root / "csrc" / "tesm_ops"
    
    # 分批次编译：将 kernels 拆分成多个扩展
    # 这样可以减少内存峰值，也支持选择性编译
    
    extensions = []
    
    # ========== 统一编译所有 CUDA kernels ==========
    # 不再分批次，避免符号未定义问题
    # 限制并行编译数避免内存耗尽
    os.environ.setdefault('MAX_JOBS', '4')  # 限制并行编译数
    
    extensions.append(CUDAExtension(
        name="tesm_cuda_ops",
        sources=[
            str(source_root / "tesm_ops.cpp"),
            # Core kernels
            str(source_root / "tesm_scan_fwd.cu"),
            str(source_root / "tesm_scan_bwd.cu"),
            str(source_root / "tesm_entanglement_fwd.cu"),
            str(source_root / "tesm_entanglement_bwd.cu"),
            # Quantized linear
            str(source_root / "tesm_quantized_linear_fwd.cu"),
            str(source_root / "tesm_quantized_linear_bwd.cu"),
            str(source_root / "tesm_int2_linear.cu"),
            str(source_root / "tesm_int2_linear_optimized.cu"),
            # Global entanglement
            str(source_root / "tesm_global_entanglement_fwd.cu"),
            str(source_root / "tesm_global_entanglement_bwd.cu"),
            # Fused output
            str(source_root / "tesm_fused_output_fwd.cu"),
            str(source_root / "tesm_fused_output_bwd.cu"),
            # MIMO kernels
            str(source_root / "tesm_scan_mimo_fwd.cu"),
            str(source_root / "tesm_scan_mimo_bwd.cu"),
            str(source_root / "tesm_entanglement_mimo_fwd.cu"),
            str(source_root / "tesm_entanglement_mimo_bwd.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr", "-std=c++17"]
        },
    ))
    
    return extensions


ext_modules = _get_ext_modules()
cmdclass = {"build_ext": BuildExtension} if ext_modules and BuildExtension is not None else {}


setup(
    name="tesm_ssm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require={
        "training": [
            "tensorboard>=2.10",
            "wandb>=0.13",
        ],
        "data": [
            "datasets>=2.0",
            "tokenizers>=0.13",
        ],
        "all": [
            "tensorboard>=2.10",
            "wandb>=0.13",
            "datasets>=2.0",
            "tokenizers>=0.13",
            "pytest>=7.0",
            "numpy>=1.20",
        ],
    },
)
