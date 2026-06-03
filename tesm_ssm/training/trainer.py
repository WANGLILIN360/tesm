"""TESM 训练器"""
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig


logger = logging.getLogger(__name__)


class TESMTrainer:
    """TESM 模型训练器
    
    功能:
        - 支持混合精度训练 (AMP)
        - 支持梯度累积
        - 支持学习率调度 (warmup + cosine/linear)
        - 支持检查点保存/恢复
        - 支持 TensorBoard 和 Wandb 日志
        - 支持早停
    
    示例:
        from tesm_ssm.training import TrainingConfig, TESMTrainer
        
        config = TrainingConfig(
            model_config=TESMConfig.small(),
            data_path="data/train.txt",
            output_dir="outputs",
            num_epochs=3,
        )
        
        trainer = TESMTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 设备选择（auto/cpu/cuda/mps）
        self.device = self._get_device(config.device)
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化模型（支持 SISO/MIMO 模式选择）
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.lr_scheduler = None
        self.num_update_steps_per_epoch = None
        self.max_steps = None
        
        # 初始化 AMP
        self.scaler = None
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 完全延迟初始化日志记录器
        self.tb_writer = None
        self.wandb = None
        self._setup_loggers()
        
        # 早停相关
        self.best_eval_loss = float('inf')
        self.best_train_loss = float('inf')
        self.early_stopping_counter = 0
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("[TESM] Trainer initialized")
        logger.info("  +==================================+")
        logger.info(f"  |  Device    : {self.device!s:>20s} |")
        logger.info(f"  |  Model     : {self._get_model_type():>20s} |")
        logger.info(f"  |  Params    : {param_count:>17.2f}M |")
        logger.info(f"  |  Backend   : {self.config.model_config.kernel_backend:>20s} |")
        logger.info("  +==================================+")
    
    def _setup_loggers(self):
        """完全延迟初始化日志记录器（按需导入）"""
        # TensorBoard
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.output_dir / 'logs')
                logger.info("TensorBoard 日志已启用")
            except ImportError:
                logger.warning("tensorboard 未安装，跳过 TensorBoard 日志")
                logger.info("  安装: pip install tensorboard")
        
        # Weights & Biases
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project or 'tesm-training',
                    name=self.config.wandb_run_name,
                    config=self.config.to_dict(),
                )
                self.wandb = wandb
                logger.info("Wandb 日志已启用")
            except ImportError:
                logger.warning("wandb 未安装，跳过 wandb 日志")
                logger.info("  安装: pip install wandb")
            except Exception as e:
                logger.warning(f"wandb 初始化失败: {e}")
    
    def _get_device(self, device_config: str) -> torch.device:
        """获取设备
        
        Args:
            device_config: auto, cpu, cuda, cuda:0, mps
        
        Returns:
            torch.device
        """
        if device_config == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        elif device_config.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning(f"CUDA不可用，回退到CPU")
                return torch.device('cpu')
            return torch.device(device_config)
        elif device_config == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            logger.warning(f"MPS不可用，回退到CPU")
            return torch.device('cpu')
        else:
            return torch.device(device_config)
    
    def _get_model_type(self) -> str:
        """获取当前模型类型"""
        # TESMLMHeadModel 使用 tesm_config 存储配置
        config = getattr(self.model, 'tesm_config', getattr(self.model, 'config', None))
        if config and hasattr(config, 'use_mimo'):
            return 'MIMO' if config.use_mimo else 'SISO'
        return 'Unknown'
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler(),
            ]
        )
    
    def _create_model(self) -> nn.Module:
        """创建模型（支持 SISO/MIMO 模式选择和加速器配置）"""
        from tesm_ssm import TESMLMHeadModel, TESMConfig
        
        model_config = self.config.model_config
        
        # 根据 model_type 配置强制设置 use_mimo
        if self.config.model_type == 'siso':
            # 强制使用 SISO
            model_config.use_mimo = False
            model_config.n_heads = 1
            logger.info("强制使用 SISO 模式")
        elif self.config.model_type == 'mimo':
            # 强制使用 MIMO
            model_config.use_mimo = True
            if model_config.n_heads <= 1:
                model_config.n_heads = 2  # MIMO 至少需要2个头
            logger.info(f"强制使用 MIMO 模式 (n_heads={model_config.n_heads})")
        else:
            # auto: 使用 model_config 中的配置
            logger.info(f"使用配置中的模式: {'MIMO' if model_config.use_mimo else 'SISO'}")
        
        # 应用加速器选择 (kernel_backend)
        if self.config.accelerator != 'auto':
            model_config.kernel_backend = self.config.accelerator
            logger.info(f"使用加速器: {self.config.accelerator}")
        else:
            logger.info(f"使用自动加速器选择 (当前: {model_config.kernel_backend})")
        
        return TESMLMHeadModel(model_config)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分离 weight decay 和 no weight decay 的参数
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 偏置、归一化层、嵌入层不加 weight decay
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        # 创建优化器
        if self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
        
        return optimizer
    
    def _create_lr_scheduler(self, num_training_steps: int):
        """创建学习率调度器"""
        warmup_steps = self.config.warmup_steps
        
        if self.config.lr_scheduler == 'linear':
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    self.config.min_lr_ratio,
                    float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
                )
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(
                    self.config.min_lr_ratio,
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.lr_scheduler == 'constant':
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.lr_scheduler == 'polynomial':
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(
                    self.config.min_lr_ratio,
                    (1.0 - progress) ** 1.5
                )
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        else:
            raise ValueError(f"不支持的学习率调度器: {self.config.lr_scheduler}")
    
    def _get_dataloader(self, data_path: str, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        from .dataset import TextDataset, collate_fn
        
        # 使用 HuggingFace tokenizer（或从 config 获取）
        tokenizer = getattr(self.config, 'tokenizer', None)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
                tokenizer_path = getattr(self.config, 'tokenizer_path', 'gpt2')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            except Exception:
                # 网络不可用时回退到 SimpleTokenizer
                from .dataset import SimpleTokenizer
                tokenizer = SimpleTokenizer()
        
        dataset = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            shuffle=shuffle,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
            drop_last=self.config.dataloader_drop_last,
        )
    
    def train(self, train_dataloader: Optional[DataLoader] = None, eval_dataloader: Optional[DataLoader] = None):
        """开始训练
        
        Args:
            train_dataloader: 训练数据加载器（可选，默认从 config.data_path 创建）
            eval_dataloader: 验证数据加载器（可选）
        """
        # 创建数据加载器
        if train_dataloader is None:
            if self.config.data_path is None:
                raise ValueError("必须提供 data_path 或 train_dataloader")
            train_dataloader = self._get_dataloader(self.config.data_path, shuffle=True)
        
        if eval_dataloader is None and self.config.eval_data_path:
            eval_dataloader = self._get_dataloader(self.config.eval_data_path, shuffle=False)
        
        # 计算总步数
        self.num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        if self.config.max_steps is not None:
            self.max_steps = self.config.max_steps
            self.config.num_epochs = math.ceil(self.max_steps / self.num_update_steps_per_epoch)
        else:
            self.max_steps = self.num_update_steps_per_epoch * self.config.num_epochs
        
        # 创建学习率调度器
        self.lr_scheduler = self._create_lr_scheduler(self.max_steps)
        
        logger.info("")
        logger.info("  +==============================================+")
        logger.info("  |           [*] TESM Training Started [*]      |")
        logger.info("  +==============================================+")
        logger.info(f"  |  Samples       : {len(train_dataloader.dataset):>28,} |")
        logger.info(f"  |  Batch Size     : {self.config.batch_size:>28} |")
        logger.info(f"  |  Grad Accum     : {self.config.gradient_accumulation_steps:>28} |")
        logger.info(f"  |  Steps/Epoch    : {self.num_update_steps_per_epoch:>28,} |")
        logger.info(f"  |  Total Epochs   : {self.config.num_epochs:>28} |")
        logger.info(f"  |  Total Steps    : {self.max_steps:>28,} |")
        logger.info(f"  |  Learning Rate  : {self.config.learning_rate:>28.2e} |")
        logger.info("  +==============================================+")
        logger.info("")
        
        # 恢复训练
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # 训练循环
        self.state.global_step = 0
        self.state.epoch = 0
        self._train_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            
            # 训练一个 epoch
            train_loss = self._train_epoch(train_dataloader)
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - 训练损失: {train_loss:.4f}")
            
            # 禁用epoch保存最佳模型
            # if train_loss < self.best_train_loss:
            #     self.best_train_loss = train_loss
            #     self._save_checkpoint('best')
            #     logger.info(f"新的最佳训练损失: {train_loss:.4f}")
            
            # 验证
            if eval_dataloader is not None and (epoch + 1) % max(1, self.config.eval_interval // self.num_update_steps_per_epoch) == 0:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - 验证损失: {eval_loss:.4f}")
                
                # 早停检查
                if self._check_early_stopping(eval_loss):
                    logger.info(f"触发早停，最佳验证损失: {self.best_eval_loss:.4f}")
                    break
            
            # 保存检查点
            if (epoch + 1) % max(1, self.config.save_interval // self.num_update_steps_per_epoch) == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}')
        
        # 保存最终模型
        self._save_checkpoint('final')
        total_time = time.time() - getattr(self, '_train_start_time', time.time())
        logger.info("")
        logger.info("  +==============================================+")
        logger.info("  |         [*] TESM Training Complete [*]        |")
        logger.info("  +==============================================+")
        logger.info(f"  |  Total Steps   : {self.state.global_step:>28,} |")
        logger.info(f"  |  Total Time    : {self._format_eta(total_time):>28} |")
        logger.info("  +==============================================+")
        logger.info("")
        
        # 关闭日志
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb:
            self.wandb.finish()
    
    def _compute_loss(self, model, input_ids, labels, loss_mask=None):
        """计算损失（支持 loss mask）
        
        Args:
            model: 模型
            input_ids: 输入 token IDs
            labels: 标签
            loss_mask: 损失掩码 [batch_size, seq_len]，1=计算loss，0=忽略
                      若为 None，则使用 labels 中的 -100 作为忽略标记
        
        Returns:
            torch.Tensor: 标量损失值
        """
        if self.scaler is not None:
            with torch.cuda.amp.autocast(dtype=self._get_amp_dtype()):
                outputs, _ = model(input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs, _ = model(input_ids, labels=labels)
            loss = outputs.loss
        
        # 如果提供了 loss_mask，重新计算带 mask 的损失
        if loss_mask is not None:
            logits = outputs.logits
            # Shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = loss_mask[..., 1:].contiguous().float()
            
            # 逐token计算交叉熵
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
            per_token_loss = per_token_loss.view(shift_labels.size())
            
            # 应用 mask 并求平均
            total_tokens = shift_mask.sum()
            if total_tokens > 0:
                loss = (per_token_loss * shift_mask).sum() / total_tokens
            else:
                loss = per_token_loss.mean()
        
        return loss
    
    def compute_perplexity(self, loss: float) -> float:
        """从损失计算困惑度
        
        Args:
            loss: 交叉熵损失值
        
        Returns:
            float: 困惑度 (PPL)
        """
        import math
        return math.exp(min(loss, 20))  # clamp 防止溢出
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 梯度累积步数计数
        accumulation_steps = 0
        
        # 计时与梯度追踪
        epoch_start = time.time()
        last_log_time = epoch_start
        last_grad_norm = 0.0
        
        for step, batch in enumerate(dataloader):
            # 将数据移到设备
            if step == 0:
                logger.info("[DEBUG] Loading first batch...")
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            loss_mask = batch.get('loss_mask')
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
            if step == 0:
                logger.info(f"[DEBUG] Batch loaded: input_ids={input_ids.shape}, ids range=[{input_ids.min()},{input_ids.max()}]")
            
            # 前向传播（支持 loss mask）
            if step == 0:
                logger.info("[DEBUG] Starting forward pass...")
            loss = self._compute_loss(self.model, input_ids, labels, loss_mask=loss_mask)
            if step == 0:
                logger.info(f"[DEBUG] Forward done, loss={loss.item():.4f}")
            
            # 梯度累积：除以累积步数
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            if step == 0:
                logger.info("[DEBUG] Starting backward pass...")
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if step == 0:
                logger.info("[DEBUG] Backward done")
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            accumulation_steps += 1
            
            # 更新参数
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                # 梯度裁剪（同时记录梯度范数）
                last_grad_norm = 0.0
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    ).item()
                
                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 学习率调度
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # 清空梯度
                self.optimizer.zero_grad()
                
                # 更新状态
                self.state.global_step += 1
                accumulation_steps = 0
                num_batches += 1
                
                # 日志记录
                if self.state.global_step % self.config.log_interval == 0:
                    now = time.time()
                    elapsed = now - epoch_start
                    steps_done = self.state.global_step - (self.state.epoch * self.num_update_steps_per_epoch)
                    steps_done = max(steps_done, 1)
                    eta_secs = (elapsed / steps_done) * (self.max_steps - self.state.global_step)
                    self._log_step(
                        total_loss / max(1, num_batches),
                        grad_norm=last_grad_norm,
                        elapsed=now - last_log_time,
                        eta_secs=eta_secs,
                    )
                    last_log_time = now
                
                # 验证
                if self.config.eval_data_path and self.state.global_step % self.config.eval_interval == 0:
                    eval_dataloader = self._get_dataloader(self.config.eval_data_path, shuffle=False)
                    eval_loss = self._evaluate(eval_dataloader)
                    logger.info(f"Step {self.state.global_step} - 验证损失: {eval_loss:.4f}")
                    
                    if self._check_early_stopping(eval_loss):
                        return total_loss / max(1, num_batches)
                
                # 保存检查点
                if self.state.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f'step_{self.state.global_step}')
                
                # 禁用自动保存最佳模型 - 只在最后保存
                # avg_loss = total_loss / max(1, num_batches)
                # if avg_loss < self.best_train_loss:
                #     self.best_train_loss = avg_loss
                #     self._save_checkpoint('best')
                #     logger.info(f"新的最佳训练损失: {avg_loss:.4f}")
                
                # 检查是否达到最大步数
                if self.state.global_step >= self.max_steps:
                    break
        
        # 处理剩余的梯度
        if accumulation_steps > 0:
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.optimizer.zero_grad()
        
        return total_loss / max(1, num_batches)
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """评估模型
        
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if self.config.eval_steps is not None and step >= self.config.eval_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss_mask = batch.get('loss_mask')
                if loss_mask is not None:
                    loss_mask = loss_mask.to(self.device)
                
                loss = self._compute_loss(self.model, input_ids, labels, loss_mask=loss_mask)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / max(1, num_batches)
        
        # 记录困惑度
        ppl = self.compute_perplexity(avg_loss)
        logger.info(f"评估 - Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")
        
        return avg_loss
    
    def _format_eta(self, seconds: float) -> str:
        """格式化剩余时间"""
        if seconds < 0 or not math.isfinite(seconds):
            return "--:--:--"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _collect_ternary_stats(self) -> Optional[Dict[str, float]]:
        """收集模型三值纠缠统计"""
        stats = {'pos': 0.0, 'neg': 0.0, 'zero': 0.0}
        count = 0
        for module in self.model.modules():
            if hasattr(module, '_ternary_stats_for_logging') and module._ternary_stats_for_logging is not None:
                ternary, total = module._ternary_stats_for_logging
                if total <= 0:
                    continue
                pos = (ternary > 0).float().sum().item() / total
                neg = (ternary < 0).float().sum().item() / total
                zero = (ternary == 0).float().sum().item() / total
                stats['pos'] += pos
                stats['neg'] += neg
                stats['zero'] += zero
                count += 1
        if count == 0:
            return None
        return {k: v / count for k, v in stats.items()}

    def _log_step(self, loss: float, grad_norm: float = 0.0,
                  elapsed: float = 0.0, eta_secs: float = 0.0):
        """记录训练步骤日志"""
        lr = self.optimizer.param_groups[0]['lr']
        ppl = self.compute_perplexity(loss)
        progress = self.state.global_step / max(1, self.max_steps)

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('train/loss', loss, self.state.global_step)
            self.tb_writer.add_scalar('train/ppl', ppl, self.state.global_step)
            self.tb_writer.add_scalar('train/lr', lr, self.state.global_step)
            self.tb_writer.add_scalar('train/grad_norm', grad_norm, self.state.global_step)

        # Wandb
        if self.wandb:
            self.wandb.log({
                'train/loss': loss,
                'train/ppl': ppl,
                'train/lr': lr,
                'train/grad_norm': grad_norm,
                'train/step': self.state.global_step,
            })

        # 三值统计
        ternary = self._collect_ternary_stats()
        ternary_str = ""
        if ternary is not None:
            ternary_str = f" | Ternary +{ternary['pos']:.1%} -{ternary['neg']:.1%} 0:{ternary['zero']:.1%}"

        # 进度条
        bar_len = 20
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)

        # 品牌感控制台输出
        step_str = f"{self.state.global_step}/{self.max_steps}"
        eta_str = self._format_eta(eta_secs)
        speed = f"{elapsed:.1f}s" if elapsed > 0 else ""

        logger.info(
            f"[TESM] [{bar}] {progress:.1%} | Step {step_str} | "
            f"Loss {loss:.4f} | PPL {ppl:.1f} | LR {lr:.2e} | "
            f"Grad {grad_norm:.3f} | ETA {eta_str} {speed}"
            f"{ternary_str}"
        )
    
    def _check_early_stopping(self, eval_loss: float) -> bool:
        """检查是否应该早停
        
        Returns:
            bool: 是否应该停止训练
        """
        if self.config.early_stopping_patience is None:
            return False
        
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        if self.early_stopping_counter >= self.config.early_stopping_patience:
            return True
        
        return False
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pt'
        
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict(),
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.state.epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"检查点已加载: {checkpoint_path}, step={self.state.global_step}")
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近的N个"""
        if self.config.keep_last_n_checkpoints <= 0:
            return
        
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.config.keep_last_n_checkpoints:]:
            if 'best' in checkpoint.stem:
                continue  # 不删除最佳模型检查点
            checkpoint.unlink()
            logger.info(f"删除旧检查点: {checkpoint}")
    
    def _get_amp_dtype(self) -> torch.dtype:
        """获取 AMP 数据类型"""
        if self.config.amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    
    def save_model(self, save_path: Optional[str] = None, save_int2: bool = True, build_cooccurrence: bool = True):
        """保存模型（FP32 + INT2 + 共现矩阵）
        
        Args:
            save_path: 保存路径
            save_int2: 是否同时保存 INT2 量化模型
            build_cooccurrence: 是否构建并保存共现矩阵（用于动态词表激活）
        """
        if save_path is None:
            save_path = self.output_dir / 'final_model'
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 如果存在 best checkpoint，加载最佳权重再保存
        best_ckpt = self.checkpoint_dir / 'checkpoint_best.pt'
        if best_ckpt.exists():
            logger.info(f"加载最佳模型 checkpoint: {best_ckpt}")
            best_state = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(best_state['model_state_dict'])
        
        # 保存 FP32 模型权重
        torch.save(self.model.state_dict(), save_path / 'model.pt')
        logger.info(f"FP32 模型已保存: {save_path / 'model.pt'}")
        
        # 保存 INT2 量化模型
        if save_int2:
            try:
                from tesm_ssm.utils.int2_quantization import save_int2_model
                int2_path = str(save_path / 'model_int2.pt')
                save_int2_model(self.model, int2_path)
                logger.info(f"INT2 模型已保存: {int2_path}")
            except Exception as e:
                logger.warning(f"INT2 模型保存失败: {e}")
        
        # 构建并保存共现矩阵（用于推理时动态词表激活）
        if build_cooccurrence and getattr(self.model, 'semantic_activation', False) and hasattr(self.model, 'build_cooccurrence_from_dataset'):
            try:
                if not getattr(self.model, 'cooccurrence_built', False):
                    # 需要训练数据来构建共现矩阵
                    if self.config.data_path:
                        logger.info("构建 token 共现矩阵...")
                        from torch.utils.data import DataLoader
                        dataloader = self._get_dataloader(self.config.data_path, shuffle=False)
                        self.model.build_cooccurrence_from_dataset(dataloader, max_batches=50)
                if getattr(self.model, 'cooccurrence_built', False):
                    cooccurrence_path = save_path / 'cooccurrence.pt'
                    torch.save({
                        'related_token_ids': self.model.related_token_ids.cpu(),
                        'related_token_strengths': self.model.related_token_strengths.cpu(),
                        'token_freq': self.model.token_freq.cpu(),
                    }, cooccurrence_path)
                    logger.info(f"共现矩阵已保存: {cooccurrence_path}")
            except Exception as e:
                logger.warning(f"共现矩阵构建/保存失败: {e}")
        
        # 保存配置
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"模型已保存: {save_path}")


class TrainingState:
    """训练状态"""
    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_metric: Optional[float] = None
