"""
测试 kernel_backend 严格模式：指定后端不可用时报错
"""
import torch
import sys

def test_siso_backend_strict():
    """测试 SISO 后端严格模式"""
    print("=" * 60)
    print("测试 SISO (TESM) 后端严格模式")
    print("=" * 60)
    
    from tesm_ssm.modules.tesm import TESM
    
    # 测试各种后端设置
    backends_to_test = ["auto", "torch", "cuda", "triton", "tilelang"]
    
    for backend in backends_to_test:
        print(f"\n--- kernel_backend='{backend}' ---")
        try:
            model = TESM(
                d_model=64,
                d_state=32,
                ent_rank=16,
                entanglement_window=8,
                kernel_backend=backend,
            ).cuda()
            
            x = torch.randn(1, 16, 64, device='cuda')
            
            # 推理模式
            with torch.no_grad():
                out = model(x)
            print(f"  推理: ✓ 成功, 输出形状 {out.shape}")
            
            # 训练模式
            model.train()
            x = torch.randn(1, 16, 64, device='cuda', requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            print(f"  训练: ✓ 成功, 反向传播完成")
            
        except RuntimeError as e:
            if "kernel_backend=" in str(e) and "not available" in str(e):
                print(f"  ✓ 正确报错: {e}")
            else:
                print(f"  ✗ 意外错误: {e}")
        except Exception as e:
            print(f"  ✗ 意外错误: {type(e).__name__}: {e}")


def test_mimo_backend_strict():
    """测试 MIMO 后端严格模式"""
    print("\n" + "=" * 60)
    print("测试 MIMO (TESMMIMO) 后端严格模式")
    print("=" * 60)
    
    from tesm_ssm.modules.tesm_mimo import TESMMIMO_Optimized
    
    backends_to_test = ["auto", "torch", "cuda", "triton", "tilelang"]
    
    for backend in backends_to_test:
        print(f"\n--- kernel_backend='{backend}' ---")
        try:
            model = TESMMIMO_Optimized(
                d_model=64,
                d_state=32,
                n_heads=2,
                mimo_rank=16,
                entanglement_window=8,
                kernel_backend=backend,
            ).cuda()
            
            x = torch.randn(1, 16, 64, device='cuda')
            
            # 推理模式
            with torch.no_grad():
                out = model(x)
            print(f"  推理: ✓ 成功, 输出形状 {out.shape}")
            
            # 训练模式
            model.train()
            x = torch.randn(1, 16, 64, device='cuda', requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            print(f"  训练: ✓ 成功, 反向传播完成")
            
        except RuntimeError as e:
            if "kernel_backend=" in str(e) and "not available" in str(e):
                print(f"  ✓ 正确报错: {e}")
            else:
                print(f"  ✗ 意外错误: {e}")
        except Exception as e:
            print(f"  ✗ 意外错误: {type(e).__name__}: {e}")


def test_bitlinear_backend_strict():
    """测试 BitLinear 后端严格模式"""
    print("\n" + "=" * 60)
    print("测试 BitLinear 后端严格模式")
    print("=" * 60)
    
    from tesm_ssm.modules.tesm import BitLinear
    
    backends_to_test = ["auto", "torch", "cuda", "triton"]
    
    for backend in backends_to_test:
        print(f"\n--- kernel_backend='{backend}' ---")
        try:
            layer = BitLinear(64, 128, kernel_backend=backend).cuda()
            
            # 推理模式
            with torch.no_grad():
                x = torch.randn(1, 64, device='cuda')
                out = layer(x)
            print(f"  推理: ✓ 成功, 输出形状 {out.shape}")
            
            # 训练模式
            layer.train()
            x = torch.randn(1, 64, device='cuda', requires_grad=True)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            print(f"  训练: ✓ 成功, 反向传播完成")
            
        except RuntimeError as e:
            if "kernel_backend=" in str(e) and "not available" in str(e):
                print(f"  ✓ 正确报错: {e}")
            else:
                print(f"  ✗ 意外错误: {e}")
        except Exception as e:
            print(f"  ✗ 意外错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        sys.exit(0)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    test_siso_backend_strict()
    test_mimo_backend_strict()
    test_bitlinear_backend_strict()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
