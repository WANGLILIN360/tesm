"""
测试 Triton kernel constexpr 与 tl.arange 兼容性
"""
import torch

def test_triton_kernel():
    print("测试 Triton kernel constexpr 与 tl.arange 兼容性")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试实际 kernel
    from tesm_ssm.ops.triton.tesm_mimo_kernel import (
        tesm_local_entanglement_triton,
        tesm_local_entanglement_pytorch
    )
    
    # 使用更小的维度便于调试
    B, L, H, D, R, W = 1, 8, 2, 16, 8, 4
    
    # 固定随机种子
    torch.manual_seed(42)
    
    q = torch.randn(B, L, H, R, device=device, dtype=torch.float32).contiguous()
    k = torch.randn(B, L, H, R, device=device, dtype=torch.float32).contiguous()
    v = torch.randn(B, L, H, D, device=device, dtype=torch.float32).contiguous()
    bias = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    
    # 测试 PyTorch 版本
    print("\n--- PyTorch 版本 ---")
    out_pytorch = tesm_local_entanglement_pytorch(q, k, v, bias, threshold=0.5)
    print(f"输出: {out_pytorch.shape}")
    print(f"有效: {torch.isfinite(out_pytorch).all().item()}")
    print(f"输出范围: [{out_pytorch.min().item():.4f}, {out_pytorch.max().item():.4f}]")
    
    # 测试 Triton 版本
    print("\n--- Triton 版本 ---")
    try:
        out_triton = tesm_local_entanglement_triton(q, k, v, bias, threshold=0.5)
        torch.cuda.synchronize()
        print(f"✓ Triton kernel 调用成功")
        print(f"输出: {out_triton.shape}")
        print(f"输出范围: [{out_triton.min().item():.4f}, {out_triton.max().item():.4f}]")
        
        if torch.isfinite(out_triton).all():
            print(f"✓ 输出有效 (无 NaN/Inf)")
            
            # 逐位置比较
            diff = (out_triton - out_pytorch).abs()
            print(f"\n差异统计:")
            print(f"  平均: {diff.mean().item():.6f}")
            print(f"  最大: {diff.max().item():.6f}")
            print(f"  中位数: {diff.median().item():.6f}")
            
            # 找出差异最大的位置
            max_diff_idx = diff.argmax()
            b_idx = max_diff_idx // (L * H * D)
            rem = max_diff_idx % (L * H * D)
            l_idx = rem // (H * D)
            rem = rem % (H * D)
            h_idx = rem // D
            d_idx = rem % D
            
            print(f"\n最大差异位置: b={b_idx}, l={l_idx}, h={h_idx}, d={d_idx}")
            print(f"  PyTorch: {out_pytorch[b_idx, l_idx, h_idx, d_idx].item():.6f}")
            print(f"  Triton:  {out_triton[b_idx, l_idx, h_idx, d_idx].item():.6f}")
            
            # 检查是否是数值精度问题
            rel_diff = diff / (out_pytorch.abs() + 1e-6)
            print(f"\n相对差异:")
            print(f"  平均: {rel_diff.mean().item():.6f}")
            print(f"  最大: {rel_diff.max().item():.6f}")
        else:
            print(f"✗ 输出包含 NaN/Inf")
            
    except Exception as e:
        print(f"✗ Triton kernel 调用失败: {type(e).__name__}")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    test_triton_kernel()
