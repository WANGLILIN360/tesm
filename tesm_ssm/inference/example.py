"""多模态推理完整示例

展示如何使用 MultimodalGenerator 进行:
1. 单样本推理
2. 批量推理 (Batch Inference)
3. 流式输出
4. 与 DataLoader 集成
5. Benchmark 性能测试
"""

import torch

from tesm_ssm.models.config_tesm import TESMConfig
from tesm_ssm.models.multimodal import MultimodalConfig, TESMMultimodalModel
from tesm_ssm.inference.multimodal_generator import MultimodalGenerator


def example_1_single_inference():
    """示例1: 单样本推理"""
    print("=" * 60)
    print("示例1: 单样本推理")
    print("=" * 60)

    # 创建模型
    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    config.vision_enabled = True
    config.audio_enabled = True
    model = TESMMultimodalModel(config)

    # 创建生成器
    generator = MultimodalGenerator(model)

    # 准备输入
    images = torch.randn(1, 3, 224, 224)
    audio = torch.randn(1, 16000)  # 1s @ 16kHz
    text_ids = torch.randint(0, tesm_config.vocab_size, (1, 10))

    # 生成
    result = generator.generate(
        images=images,
        audio=audio,
        text_ids=text_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        return_dict=True,
    )

    print(f"生成序列长度: {result['sequences'].shape}")
    print(f"生成速度: {result['tokens_per_sec']:.1f} tokens/sec")
    print(f"生成 token 数: {result['num_tokens']}")
    print()


def example_2_batch_inference():
    """示例2: 批量推理"""
    print("=" * 60)
    print("示例2: 批量推理")
    print("=" * 60)

    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    config.vision_enabled = True
    model = TESMMultimodalModel(config)
    generator = MultimodalGenerator(model)

    # 批量输入 (batch_size=4)
    images = torch.randn(4, 3, 224, 224)
    text_ids = torch.randint(0, tesm_config.vocab_size, (4, 8))

    result = generator.generate(
        images=images,
        text_ids=text_ids,
        max_new_tokens=15,
        temperature=0.7,
        top_p=0.95,
        return_dict=True,
    )

    print(f"Batch size: {result['sequences'].shape[0]}")
    print(f"序列长度: {result['sequences'].shape[1]}")
    print(f"速度: {result['tokens_per_sec']:.1f} tokens/sec")
    print()


def example_3_streaming():
    """示例3: 流式输出"""
    print("=" * 60)
    print("示例3: 流式输出")
    print("=" * 60)

    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    model = TESMMultimodalModel(config)
    generator = MultimodalGenerator(model)

    images = torch.randn(1, 3, 224, 224)
    text_ids = torch.randint(0, tesm_config.vocab_size, (1, 5))

    print("流式生成 (模拟实时输出):")
    tokens = []
    for token_id, score in generator.stream_generate(
        images=images,
        text_ids=text_ids,
        max_new_tokens=10,
        temperature=0.8,
    ):
        tokens.append(token_id)
        print(f"  Token {len(tokens)}: id={token_id}, score={score:.4f}")

    print(f"共生成 {len(tokens)} 个 tokens\n")


def example_4_with_dataloader():
    """示例4: 与 DataLoader 集成"""
    print("=" * 60)
    print("示例4: 与 DataLoader 集成")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from tesm_ssm.training.multimodal_data import (
        MultimodalDataset,
        multimodal_collate_fn,
    )

    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    config.vision_enabled = True
    model = TESMMultimodalModel(config)
    generator = MultimodalGenerator(model)

    # 创建模拟数据集
    dummy_data = [
        {
            "text": "描述这张图片",
            "image": torch.randn(3, 224, 224),  # 预加载的图像张量
        }
        for _ in range(8)
    ]

    # 注意: 实际使用时应从文件路径加载图像
    # dummy_data = [
    #     {"text": "描述这张图片", "image": "/path/to/image1.jpg"}
    #     for _ in range(8)
    # ]

    dataset = MultimodalDataset(
        dummy_data,
        tokenizer=lambda t: torch.randint(0, tesm_config.vocab_size, (len(t.split()),)),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: multimodal_collate_fn(
            batch,
            pad_token_id=0,
            image_size=(224, 224),
        ),
    )

    print("DataLoader 批量推理:")
    for batch_idx, batch in enumerate(dataloader):
        images = batch.get("images")
        text_ids = batch.get("text_ids")

        if images is not None and text_ids is not None:
            result = generator.generate(
                images=images,
                text_ids=text_ids,
                max_new_tokens=10,
                temperature=0.7,
            )
            print(f"  Batch {batch_idx}: 输出形状 {result.shape}")

    print()


def example_5_benchmark():
    """示例5: 性能基准测试"""
    print("=" * 60)
    print("示例5: 性能基准测试")
    print("=" * 60)

    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    config.vision_enabled = True
    config.audio_enabled = True
    model = TESMMultimodalModel(config)
    generator = MultimodalGenerator(model)

    # 准备输入
    images = torch.randn(1, 3, 224, 224)
    audio = torch.randn(1, 16000)
    text_ids = torch.randint(0, tesm_config.vocab_size, (1, 8))

    # Benchmark
    print("运行 Benchmark (warmup=3, runs=10)...")
    result = generator.benchmark(
        images=images,
        audio=audio,
        text_ids=text_ids,
        max_new_tokens=32,
        warmup=3,
        runs=10,
    )

    print(f"平均时间: {result['avg_time']:.3f}s")
    print(f"Tokens/sec: {result['tokens_per_sec']:.1f}")
    print(f"最小时间: {result['min_time']:.3f}s")
    print(f"最大时间: {result['max_time']:.3f}s")
    print()


def example_6_different_configs():
    """示例6: 不同配置对比"""
    print("=" * 60)
    print("示例6: 不同采样配置对比")
    print("=" * 60)

    tesm_config = TESMConfig.small()
    config = MultimodalConfig.from_tesm_config(tesm_config)
    model = TESMMultimodalModel(config)
    generator = MultimodalGenerator(model)

    text_ids = torch.randint(0, tesm_config.vocab_size, (1, 5))

    configs = [
        ("Greedy (temp=0)", {"temperature": 0}),
        ("Random (temp=1.0)", {"temperature": 1.0}),
        ("Top-k (k=10)", {"temperature": 1.0, "top_k": 10}),
        ("Top-p (p=0.5)", {"temperature": 1.0, "top_p": 0.5}),
        ("Combined", {"temperature": 0.8, "top_k": 50, "top_p": 0.9}),
    ]

    for name, cfg in configs:
        result = generator.generate(
            text_ids=text_ids,
            max_new_tokens=5,
            **cfg,
        )
        print(f"  {name}: {result.tolist()}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESM MultimodalGenerator 完整示例")
    print("=" * 60 + "\n")

    # 运行所有示例
    example_1_single_inference()
    example_2_batch_inference()
    example_3_streaming()
    example_4_with_dataloader()
    example_5_benchmark()
    example_6_different_configs()

    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
