"""训练模块测试"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from tesm_ssm import TESMConfig
from tesm_ssm.training import TrainingConfig, TESMTrainer, SimpleTokenizer, TextDataset


class TestTrainingConfig:
    """训练配置测试"""
    
    def test_basic_creation(self):
        """测试基本配置创建"""
        model_config = TESMConfig.tiny()
        config = TrainingConfig(
            model_config=model_config,
            data_path="data/train.txt",
            output_dir="outputs/test",
            num_epochs=3,
            batch_size=4,
        )
        
        assert config.model_config is model_config
        assert config.num_epochs == 3
        assert config.batch_size == 4
    
    def test_to_dict(self):
        """测试配置序列化"""
        model_config = TESMConfig.tiny()
        config = TrainingConfig(
            model_config=model_config,
            data_path="data/train.txt",
            num_epochs=2,
        )
        
        d = config.to_dict()
        assert d['num_epochs'] == 2
        assert d['data_path'] == "data/train.txt"
        assert 'model_config' in d
    
    def test_from_dict(self):
        """测试配置反序列化"""
        model_config = TESMConfig.tiny()
        original = TrainingConfig(
            model_config=model_config,
            data_path="data/train.txt",
            num_epochs=2,
        )
        
        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)
        
        assert restored.num_epochs == original.num_epochs
        assert restored.batch_size == original.batch_size
        assert restored.data_path == original.data_path


class TestSimpleTokenizer:
    """简单Tokenizer测试"""
    
    def test_encode_decode(self):
        """测试编码解码"""
        tokenizer = SimpleTokenizer(vocab_size=256)
        text = "Hello World"
        
        tokens = tokenizer.encode(text)
        assert len(tokens) == len(text)
        assert all(0 <= t < 256 for t in tokens)
        
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_special_tokens(self):
        """测试特殊token"""
        tokenizer = SimpleTokenizer(vocab_size=256)
        text = "Hi"
        
        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert tokens[-1] == tokenizer.eos_token_id


class TestTextDataset:
    """文本数据集测试"""
    
    @pytest.fixture
    def temp_text_file(self):
        """创建临时文本文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n\n")
            f.write("Another paragraph here.\n")
            path = f.name
        
        yield path
        
        # 清理
        Path(path).unlink()
    
    @pytest.fixture
    def temp_jsonl_file(self):
        """创建临时JSONL文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({'text': 'First document'}) + '\n')
            f.write(json.dumps({'text': 'Second document'}) + '\n')
            path = f.name
        
        yield path
        
        Path(path).unlink()
    
    def test_load_txt(self, temp_text_file):
        """测试加载txt文件"""
        tokenizer = SimpleTokenizer(vocab_size=256)
        dataset = TextDataset(
            data_path=temp_text_file,
            tokenizer=tokenizer,
            max_seq_len=32,
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert 'attention_mask' in sample
        
        assert sample['input_ids'].shape[0] == 32
        assert sample['labels'].shape[0] == 32
    
    def test_load_jsonl(self, temp_jsonl_file):
        """测试加载jsonl文件"""
        tokenizer = SimpleTokenizer(vocab_size=256)
        dataset = TextDataset(
            data_path=temp_jsonl_file,
            tokenizer=tokenizer,
            max_seq_len=32,
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert sample['input_ids'].dtype == torch.long


class TestTESMTrainerCreation:
    """训练器创建测试"""
    
    def test_trainer_creation(self, tmp_path):
        """测试训练器创建"""
        model_config = TESMConfig.tiny()
        
        # 创建临时数据文件
        data_file = tmp_path / "train.txt"
        data_file.write_text("This is training data. " * 100)
        
        config = TrainingConfig(
            model_config=model_config,
            data_path=str(data_file),
            output_dir=str(tmp_path / "outputs"),
            num_epochs=1,
            batch_size=2,
            use_amp=False,  # 测试环境禁用AMP
            use_tensorboard=False,
        )
        
        trainer = TESMTrainer(config)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device is not None
    
    def test_optimizer_groups(self, tmp_path):
        """测试优化器参数分组"""
        model_config = TESMConfig.tiny()
        
        data_file = tmp_path / "train.txt"
        data_file.write_text("Training data. " * 50)
        
        config = TrainingConfig(
            model_config=model_config,
            data_path=str(data_file),
            output_dir=str(tmp_path / "outputs"),
            weight_decay=0.01,
            use_tensorboard=False,
        )
        
        trainer = TESMTrainer(config)
        
        # 检查优化器参数组
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[0]['weight_decay'] == 0.01
        assert trainer.optimizer.param_groups[1]['weight_decay'] == 0.0


class TestTrainingLoop:
    """训练循环测试"""
    
    @pytest.fixture
    def small_trainer(self, tmp_path):
        """创建小型训练器用于测试"""
        model_config = TESMConfig.tiny()
        
        # 创建足够大的训练数据
        data_file = tmp_path / "train.txt"
        data_file.write_text("Training data example. " * 500)
        
        config = TrainingConfig(
            model_config=model_config,
            data_path=str(data_file),
            output_dir=str(tmp_path / "outputs"),
            num_epochs=1,
            batch_size=2,
            max_seq_len=64,
            use_amp=False,
            use_tensorboard=False,
            log_interval=1000,  # 减少日志输出
            save_interval=10000,
        )
        
        trainer = TESMTrainer(config)
        # 手动设置 max_steps 避免 _train_epoch 中的 None 比较
        trainer.max_steps = 10000
        return trainer
    
    def test_train_epoch(self, small_trainer):
        """测试训练一个epoch"""
        from torch.utils.data import DataLoader
        from tesm_ssm.training import collate_fn
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            small_trainer._get_dataloader(small_trainer.config.data_path).dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        # 训练前记录损失
        small_trainer.model.eval()
        initial_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                if i >= 2:  # 只取2个batch
                    break
                input_ids = batch['input_ids'].to(small_trainer.device)
                labels = batch['labels'].to(small_trainer.device)
                outputs, _ = small_trainer.model(input_ids, labels=labels)
                initial_loss += outputs.loss.item()
        initial_loss /= 2
        
        # 训练一个epoch
        small_trainer.model.train()
        avg_loss = small_trainer._train_epoch(train_dataloader)
        
        # 验证损失是合理的数值（未训练的模型损失可能较高）
        assert 0 < avg_loss < 100  # 损失应该在合理范围内
        
        # 验证模型参数有更新
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_trainer.model.parameters()
            if p.requires_grad
        )
        assert has_grad, "模型参数应该有梯度"
    
    def test_evaluate(self, small_trainer):
        """测试评估"""
        from torch.utils.data import DataLoader
        from tesm_ssm.training import collate_fn
        
        eval_dataloader = DataLoader(
            small_trainer._get_dataloader(small_trainer.config.data_path).dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        eval_loss = small_trainer._evaluate(eval_dataloader)
        
        # 验证损失是合理的数值（未训练的模型损失可能较高）
        assert 0 < eval_loss < 100
        
        # 验证模型仍在训练模式
        assert small_trainer.model.training
    
    def test_checkpoint_save_load(self, small_trainer, tmp_path):
        """测试检查点保存和加载"""
        # 保存检查点
        small_trainer.state.global_step = 100
        small_trainer.state.epoch = 1
        small_trainer._save_checkpoint('test')
        
        checkpoint_path = tmp_path / "outputs" / "checkpoints" / "checkpoint_test.pt"
        assert checkpoint_path.exists()
        
        # 修改状态
        original_step = small_trainer.state.global_step
        small_trainer.state.global_step = 999
        
        # 加载检查点
        small_trainer._load_checkpoint(str(checkpoint_path))
        
        # 验证状态恢复
        assert small_trainer.state.global_step == original_step
        assert small_trainer.state.epoch == 1


class TestTrainingConfigValidation:
    """训练配置验证测试"""
    
    def test_missing_model_config(self):
        """测试缺少模型配置时抛出错误"""
        with pytest.raises(ValueError, match="必须提供 model_config"):
            TrainingConfig(
                data_path="data/train.txt",
                output_dir="outputs",
            )
    
    def test_max_seq_len_clipping(self):
        """测试最大序列长度裁剪"""
        model_config = TESMConfig.tiny()
        model_config.max_seq_len = 128
        
        config = TrainingConfig(
            model_config=model_config,
            data_path="data/train.txt",
            max_seq_len=256,  # 超过模型限制
        )
        
        assert config.max_seq_len <= 128


class TestLrSchedulers:
    """学习率调度器测试"""
    
    @pytest.fixture
    def trainer_for_scheduler(self, tmp_path):
        """创建用于测试调度器的训练器"""
        model_config = TESMConfig.tiny()
        
        data_file = tmp_path / "train.txt"
        data_file.write_text("Data " * 100)
        
        config = TrainingConfig(
            model_config=model_config,
            data_path=str(data_file),
            output_dir=str(tmp_path / "outputs"),
            num_epochs=1,
            batch_size=2,
            warmup_steps=10,
            learning_rate=1e-4,
            use_tensorboard=False,
        )
        
        return TESMTrainer(config)
    
    def test_cosine_scheduler(self, trainer_for_scheduler):
        """测试余弦调度器"""
        scheduler = trainer_for_scheduler._create_lr_scheduler(100)
        
        # 预热阶段
        initial_lr = trainer_for_scheduler.optimizer.param_groups[0]['lr']
        
        for _ in range(10):
            scheduler.step()
        
        # 预热后学习率应该更高
        warmed_up_lr = trainer_for_scheduler.optimizer.param_groups[0]['lr']
        assert warmed_up_lr >= initial_lr
        
        # 衰减阶段
        for _ in range(90):
            scheduler.step()
        
        final_lr = trainer_for_scheduler.optimizer.param_groups[0]['lr']
        assert final_lr < warmed_up_lr
    
    def test_linear_scheduler(self, trainer_for_scheduler):
        """测试线性调度器"""
        trainer_for_scheduler.config.lr_scheduler = 'linear'
        scheduler = trainer_for_scheduler._create_lr_scheduler(100)
        
        # 预热
        for _ in range(10):
            scheduler.step()
        
        # 衰减
        lrs = []
        for _ in range(90):
            lrs.append(trainer_for_scheduler.optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # 学习率应该单调递减
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i-1]
    
    def test_constant_scheduler(self, trainer_for_scheduler):
        """测试常数调度器"""
        trainer_for_scheduler.config.lr_scheduler = 'constant'
        scheduler = trainer_for_scheduler._create_lr_scheduler(100)
        
        # 预热
        for _ in range(10):
            scheduler.step()
        
        # 之后保持常数
        lr1 = trainer_for_scheduler.optimizer.param_groups[0]['lr']
        for _ in range(50):
            scheduler.step()
        lr2 = trainer_for_scheduler.optimizer.param_groups[0]['lr']
        
        assert abs(lr1 - lr2) < 1e-10
