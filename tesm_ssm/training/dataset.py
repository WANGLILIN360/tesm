"""数据集和数据加载器"""
import json
import random
from pathlib import Path
from typing import List, Optional, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


class TextDataset(Dataset):
    """文本数据集 - 支持 .jsonl 格式
    
    使用 load_dataset 高效加载，直接截断（非滑动窗口），
    pad 位置的 label 设为 -100 不参与 loss 计算。
    
    格式:
        .jsonl: 每行一个 JSON 对象，必须包含 "text" 字段
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        shuffle: bool = True,
    ):
        self.data_path = str(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        
        # 使用 datasets 库高效加载
        from datasets import load_dataset
        self.samples = load_dataset('json', data_files=self.data_path, split='train')
        
        # 获取 pad_token_id
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本
        
        Returns:
            dict: {
                'input_ids': torch.Tensor [seq_len],
                'labels': torch.Tensor [seq_len],
            }
        """
        sample = self.samples[idx]
        text = str(sample.get('text', ''))
        
        # 分词 + 截断（不使用滑动窗口）
        tokens = self.tokenizer(text, add_special_tokens=False, 
                                max_length=self.max_seq_len - 2, truncation=True).input_ids
        
        # 添加 BOS / EOS
        bos_id = getattr(self.tokenizer, 'bos_token_id', None)
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        if bos_id is not None:
            tokens = [bos_id] + tokens
        if eos_id is not None:
            tokens = tokens + [eos_id]
        
        # Padding
        pad_len = self.max_seq_len - len(tokens)
        input_ids = tokens + [self.pad_token_id] * pad_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Labels: pad 位置设为 -100
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingTextDataset(IterableDataset):
    """流式文本数据集 - 适用于大数据集
    
    特点:
        - 不一次性加载全部数据到内存
        - 支持多worker并行读取
        - 适用于 TB 级数据
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        buffer_size: int = 10000,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
    
    def _read_lines(self) -> Iterator[str]:
        """逐行读取文件"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    
    def _tokenize_stream(self, lines: Iterator[str]) -> Iterator[List[int]]:
        """流式分词"""
        buffer = []
        
        for line in lines:
            if not line:
                continue
            
            if self.data_path.suffix == '.jsonl':
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            
            if not text:
                continue
            
            tokens = self._tokenize_text(text)
            buffer.extend(tokens)
            
            # 当缓冲区足够大时，产出序列
            while len(buffer) >= self.max_seq_len:
                yield buffer[:self.max_seq_len]
                buffer = buffer[self.max_seq_len // 2:]  # 50% overlap
        
        # 处理剩余buffer
        if len(buffer) >= 10:
            yield buffer + [0] * (self.max_seq_len - len(buffer))
    
    def _tokenize_text(self, text: str) -> List[int]:
        """将文本转换为token ID序列"""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif hasattr(self.tokenizer, 'tokenize'):
            return self.tokenizer.tokenize(text)
        else:
            raise ValueError("tokenizer 必须提供 encode 或 tokenize 方法")
    
    def __iter__(self):
        """迭代器"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # 单worker
            lines = self._read_lines()
        else:
            # 多worker：每个worker读取部分数据
            lines = self._shard_lines(worker_info)
        
        for tokens in self._tokenize_stream(lines):
            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()
            attention_mask = (input_ids != 0).long()
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
    
    def _shard_lines(self, worker_info) -> Iterator[str]:
        """数据分片 - 每个worker读取不同的行"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % worker_info.num_workers == worker_info.id:
                    yield line.strip()


def collate_fn(batch: List[dict]) -> dict:
    """批次整理函数
    
    Args:
        batch: List[dict]，每个dict包含 input_ids, labels
    
    Returns:
        dict: 整理后的批次数据
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'labels': labels,
    }
    
    # 可选的 loss_mask
    if 'loss_mask' in batch[0]:
        result['loss_mask'] = torch.stack([item['loss_mask'] for item in batch])
    
    return result


class SimpleTokenizer:
    """简单字符级 Tokenizer - 用于快速测试
    
    示例:
        tokenizer = SimpleTokenizer(vocab_size=256)
        tokens = tokenizer.encode("Hello World")
        text = tokenizer.decode(tokens)
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """编码文本"""
        # 简单字符级编码（取ASCII码，限制在vocab_size内）
        tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        if add_special_tokens:
            tokens = tokens + [self.eos_token_id]
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """解码token序列"""
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token_id, self.eos_token_id]]
        return ''.join(chr(min(t, 127)) for t in tokens)
    
    def __len__(self):
        return self.vocab_size


class ByteTokenizer:
    """字节级 Tokenizer - GPT-2/LLaMA 风格
    
    使用字节级 BPE，词表大小通常为 32000-100000
    这里提供简化版本，实际应使用 tiktoken 或 HuggingFace tokenizer
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """UTF-8 字节编码"""
        # 简化版本：将UTF-8字节映射到vocab
        bytes_data = text.encode('utf-8')
        tokens = [(b % (self.vocab_size - 3)) + 3 for b in bytes_data]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """解码为文本"""
        if skip_special_tokens:
            tokens = [t for t in tokens if t >= 3]
        # 还原字节（简化处理，可能有错误）
        bytes_data = bytes([(t - 3) % 256 for t in tokens])
        try:
            return bytes_data.decode('utf-8', errors='ignore')
        except:
            return ""
    
    def __len__(self):
        return self.vocab_size
