#!/usr/bin/env python
"""TESM 训练脚本

使用示例:
    # 基础训练 (SISO)
    python train.py \
        --model_config small \
        --model_type siso \
        --data_path data/train.txt \
        --output_dir outputs/exp1 \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 1e-4

    # MIMO 模式训练
    python train.py \
        --model_config small \
        --model_type mimo \
        --n_heads 4 \
        --data_path data/train.txt \
        --output_dir outputs/mimo_exp \
        --num_epochs 3

    # 指定设备和加速器训练
    python train.py \
        --model_config base \
        --device cuda \
        --accelerator cuda \
        --data_path data/train.jsonl \
        --eval_data_path data/val.jsonl \
        --output_dir outputs/exp2 \
        --num_epochs 10 \
        --batch_size 8 \
        --gradient_accumulation_steps 2 \
        --use_amp \
        --amp_dtype bf16

    # 使用纯 PyTorch 后端 (无CUDA编译)
    python train.py \
        --model_config tiny \
        --device cpu \
        --accelerator torch \
        --data_path data/train.txt \
        --output_dir outputs/torch_backend

    # CPU 训练 (无 GPU)
    python train.py \
        --model_config tiny \
        --device cpu \
        --data_path data/train.txt \
        --output_dir outputs/cpu_exp

    # 从检查点恢复
    python train.py \
        --model_config small \
        --data_path data/train.txt \
        --resume_from_checkpoint outputs/exp1/checkpoints/checkpoint_step_1000.pt \
        --output_dir outputs/exp1_resumed

    # 使用配置文件
    python train.py --config_file config.json
"""

import argparse
import json
import sys
from pathlib import Path

from tesm_ssm import TESMConfig
from tesm_ssm.training import TrainingConfig, TESMTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TESM 模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 模型配置
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        '--model_config',
        type=str,
        default='tiny',
        choices=['tiny', 'small', 'base', 'medium', 'large', 'long_context'],
        help='模型预设配置'
    )
    model_group.add_argument(
        '--model_config_file',
        type=str,
        help='模型配置文件路径（JSON）'
    )
    model_group.add_argument(
        '--model_type',
        type=str,
        default='auto',
        choices=['auto', 'siso', 'mimo'],
        help='模型类型: auto=使用配置中的use_mimo, siso=强制单头, mimo=强制多头'
    )
    model_group.add_argument(
        '--n_heads',
        type=int,
        help='MIMO模式的头数（覆盖配置中的n_heads）'
    )
    
    # 数据配置
    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='训练数据路径（.txt 或 .jsonl）'
    )
    data_group.add_argument(
        '--eval_data_path',
        type=str,
        help='验证数据路径'
    )
    data_group.add_argument(
        '--max_seq_len',
        type=int,
        default=2048,
        help='最大序列长度'
    )
    
    # 训练配置
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='输出目录'
    )
    train_group.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='训练轮数'
    )
    train_group.add_argument(
        '--max_steps',
        type=int,
        help='最大训练步数（覆盖 num_epochs）'
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小'
    )
    train_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='梯度累积步数'
    )
    train_group.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='学习率'
    )
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='权重衰减'
    )
    train_group.add_argument(
        '--warmup_steps',
        type=int,
        default=100,
        help='预热步数'
    )
    train_group.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='梯度裁剪阈值'
    )
    
    # 优化器配置
    optimizer_group = parser.add_argument_group("优化器配置")
    optimizer_group.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adamw', 'adam', 'sgd'],
        help='优化器类型'
    )
    optimizer_group.add_argument(
        '--lr_scheduler',
        type=str,
        default='cosine',
        choices=['linear', 'cosine', 'constant', 'polynomial'],
        help='学习率调度器'
    )
    
    # 系统配置
    system_group = parser.add_argument_group("系统配置")
    system_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    system_group.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='数据加载器工作进程数'
    )
    system_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='训练设备: auto=自动选择, cpu=CPU, cuda=NVIDIA GPU, mps=Apple Silicon'
    )
    system_group.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'torch', 'cuda', 'triton', 'tilelang'],
        help='Kernel加速器: auto=自动选择, torch=纯PyTorch, cuda=CUDA kernel, triton=Triton kernel, tilelang=TileLang kernel'
    )
    
    # 混合精度
    amp_group = parser.add_argument_group("混合精度")
    amp_group.add_argument(
        '--use_amp',
        action='store_true',
        help='使用自动混合精度'
    )
    amp_group.add_argument(
        '--amp_dtype',
        type=str,
        default='bf16',
        choices=['fp16', 'bf16'],
        help='混合精度数据类型'
    )
    
    # 检查点与日志
    checkpoint_group = parser.add_argument_group("检查点与日志")
    checkpoint_group.add_argument(
        '--resume_from_checkpoint',
        type=str,
        help='从检查点恢复训练'
    )
    checkpoint_group.add_argument(
        '--save_interval',
        type=int,
        default=1000,
        help='保存检查点间隔（步）'
    )
    checkpoint_group.add_argument(
        '--eval_interval',
        type=int,
        default=500,
        help='验证间隔（步）'
    )
    checkpoint_group.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='日志记录间隔（步）'
    )
    checkpoint_group.add_argument(
        '--keep_last_n_checkpoints',
        type=int,
        default=3,
        help='保留最近N个检查点'
    )
    checkpoint_group.add_argument(
        '--use_tensorboard',
        action='store_true',
        default=True,
        help='使用 TensorBoard'
    )
    checkpoint_group.add_argument(
        '--use_wandb',
        action='store_true',
        help='使用 Weights & Biases'
    )
    checkpoint_group.add_argument(
        '--wandb_project',
        type=str,
        help='Wandb 项目名称'
    )
    checkpoint_group.add_argument(
        '--wandb_run_name',
        type=str,
        help='Wandb 运行名称'
    )
    
    # 早停
    early_stop_group = parser.add_argument_group("早停")
    early_stop_group.add_argument(
        '--early_stopping_patience',
        type=int,
        help='早停耐心值（None表示禁用）'
    )
    early_stop_group.add_argument(
        '--early_stopping_threshold',
        type=float,
        default=0.001,
        help='早停阈值'
    )
    
    # 配置文件
    config_group = parser.add_argument_group("配置文件")
    config_group.add_argument(
        '--config_file',
        type=str,
        help='训练配置文件路径（JSON）'
    )
    
    return parser.parse_args()


def get_model_config(args):
    """获取模型配置"""
    if args.model_config_file:
        # 从文件加载
        with open(args.model_config_file, 'r') as f:
            config_dict = json.load(f)
        return TESMConfig.from_dict(config_dict)
    
    # 使用预设
    config_map = {
        'tiny': TESMConfig.tiny,
        'small': TESMConfig.small,
        'base': TESMConfig.base,
        'medium': TESMConfig.medium,
        'large': TESMConfig.large_40b,
        'long_context': TESMConfig.long_context,
    }
    
    if args.model_config not in config_map:
        raise ValueError(f"未知的模型配置: {args.model_config}")
    
    return config_map[args.model_config]()


def get_training_config(args, model_config):
    """获取训练配置"""
    # 如果提供了配置文件，优先使用
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        return TrainingConfig.from_dict(config_dict)
    
    # 应用 n_heads 覆盖
    if args.n_heads is not None:
        model_config.n_heads = args.n_heads
    
    # 从命令行参数创建
    return TrainingConfig(
        model_config=model_config,
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        max_seq_len=args.max_seq_len,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        device=args.device,
        model_type=args.model_type,
        accelerator=args.accelerator,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("TESM 训练脚本")
    print("=" * 70)
    
    # 获取配置
    print(f"\n[1/4] 加载模型配置: {args.model_config}")
    model_config = get_model_config(args)
    print(f"  模型参数量: {model_config.d_model * model_config.n_layer / 1e6:.1f}M (估计)")
    
    print(f"\n[2/4] 创建训练配置")
    training_config = get_training_config(args, model_config)
    print(f"  模型类型: {args.model_type} (覆盖: n_heads={args.n_heads if args.n_heads else '默认'})")
    print(f"  训练设备: {args.device}")
    print(f"  加速器: {args.accelerator}")
    print(f"  输出目录: {training_config.output_dir}")
    print(f"  批次大小: {training_config.batch_size}")
    print(f"  学习率: {training_config.learning_rate}")
    print(f"  训练轮数: {training_config.num_epochs}")
    
    # 检查数据文件
    data_path = Path(training_config.data_path)
    if not data_path.exists():
        print(f"\n错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    print(f"\n[3/4] 验证数据")
    print(f"  训练数据: {data_path} ({data_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    if training_config.eval_data_path:
        eval_path = Path(training_config.eval_data_path)
        if eval_path.exists():
            print(f"  验证数据: {eval_path} ({eval_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 创建训练器并开始训练
    print(f"\n[4/4] 初始化训练器")
    trainer = TESMTrainer(training_config)
    
    print(f"\n{'=' * 70}")
    print("开始训练")
    print("=" * 70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存当前检查点
        save_path = Path(training_config.output_dir) / 'interrupted_checkpoint.pt'
        trainer._save_checkpoint('interrupted')
        print(f"中断检查点已保存: {save_path}")
        sys.exit(0)
    
    # 保存最终模型
    print(f"\n{'=' * 70}")
    print("保存最终模型")
    print("=" * 70)
    trainer.save_model()
    
    print(f"\n{'=' * 70}")
    print("训练完成！")
    print(f"模型保存在: {Path(training_config.output_dir) / 'final_model'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
