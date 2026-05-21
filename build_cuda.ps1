# TESM CUDA扩展编译脚本
$ErrorActionPreference = "Stop"

# 设置MSVC编译器环境
$MSVC_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

# 检查路径存在
if (-not (Test-Path $MSVC_PATH)) {
    Write-Host "MSVC路径不存在: $MSVC_PATH" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CUDA_PATH)) {
    Write-Host "CUDA路径不存在: $CUDA_PATH" -ForegroundColor Red
    exit 1
}

# 激活虚拟环境
$VENV_PATH = "d:\tesm-main-official-backup\.venv\Scripts\Activate.ps1"
if (Test-Path $VENV_PATH) {
    & $VENV_PATH
}

# 设置环境变量
$env:PATH = "$MSVC_PATH;$CUDA_PATH\bin;$env:PATH"
$env:INCLUDE = "$MSVC_PATH\..\..\..\include;$CUDA_PATH\include"
$env:LIB = "$MSVC_PATH\..\..\..\lib\x64;$CUDA_PATH\lib\x64"
$env:CUDA_HOME = $CUDA_PATH
$env:CUDA_PATH = $CUDA_PATH

# 验证编译器
Write-Host "`n=== 编译器检查 ===" -ForegroundColor Cyan
Write-Host "cl.exe: " -NoNewline
& cl 2>&1 | Select-Object -First 1

Write-Host "nvcc: " -NoNewline
& nvcc --version 2>&1 | Select-Object -First 1

# 编译CUDA扩展
Write-Host "`n=== 编译TESM CUDA扩展 ===" -ForegroundColor Cyan
Set-Location "d:\tesm-main-official-backup\tesm-main-official-backup"

# 使用PyTorch JIT编译
python -c @"
import torch
from torch.utils.cpp_extension import CUDAExtension, load
import setuptools

# 获取源文件
from pathlib import Path
source_root = Path('csrc/tesm_ops')
sources = [
    str(source_root / 'tesm_ops.cpp'),
    str(source_root / 'tesm_scan_fwd.cu'),
    str(source_root / 'tesm_scan_bwd.cu'),
    str(source_root / 'tesm_entanglement_fwd.cu'),
    str(source_root / 'tesm_entanglement_bwd.cu'),
    str(source_root / 'tesm_quantized_linear_fwd.cu'),
    str(source_root / 'tesm_quantized_linear_bwd.cu'),
    str(source_root / 'tesm_int2_linear.cu'),
    str(source_root / 'tesm_int2_linear_optimized.cu'),
]

print('源文件:')
for s in sources:
    print(f'  {s} - 存在: {Path(s).exists()}')

# 编译
module = load(
    name='tesm_cuda_ops',
    sources=sources,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)
print('编译成功!')
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== 编译成功 ===" -ForegroundColor Green
    
    # 验证
    Write-Host "`n验证CUDA ops:" -ForegroundColor Cyan
    python -c "from tesm_ssm.modules.tesm import tesm_cuda_is_available; print('CUDA ops:', tesm_cuda_is_available())"
} else {
    Write-Host "`n=== 编译失败 ===" -ForegroundColor Red
}
