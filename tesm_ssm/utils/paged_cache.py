"""分页状态缓存：支持超长上下文推理

特性：
1. 状态分页：将状态分成固定大小的页
2. CPU卸载：不活跃的页自动卸载到CPU内存
3. LRU淘汰：显存不足时淘汰最久未使用的页
4. 透明访问：对上层透明，自动处理页换入换出
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class StatePage:
    """单个状态页"""
    page_id: int
    start_pos: int  # 该页对应的起始位置
    end_pos: int    # 该页对应的结束位置
    
    # 状态数据
    state: torch.Tensor          # (batch, d_state) float64
    ent_k_cache: torch.Tensor    # (batch, window, ent_rank) float32
    ent_v_cache: torch.Tensor    # (batch, window, d_state) float32
    
    # 元数据
    device: torch.device
    is_on_gpu: bool = True
    last_access: int = 0  # 访问时间戳，用于LRU


class PagedStateCache:
    """分页状态缓存管理器
    
    使用方式：
        cache = PagedStateCache(
            batch_size=1,
            d_state=256,
            ent_rank=48,
            window=16,
            page_size=512,  # 每页512个token
            max_gpu_pages=100,  # GPU最多存100页
        )
        
        # 预填充阶段：保存状态
        cache.save_state(0, state_dict)
        
        # 增量推理阶段：加载状态
        state = cache.load_state(cur_pos)
    """
    
    def __init__(
        self,
        batch_size: int,
        d_state: int,
        ent_rank: int,
        window: int,
        page_size: int = 512,
        max_gpu_pages: int = 100,
        device: torch.device = None,
    ):
        self.batch_size = batch_size
        self.d_state = d_state
        self.ent_rank = ent_rank
        self.window = window
        self.page_size = page_size
        self.max_gpu_pages = max_gpu_pages
        self.device = device or torch.device('cuda')
        
        # 页存储
        self.pages: Dict[int, StatePage] = OrderedDict()  # page_id -> StatePage
        self.gpu_page_count = 0
        self.access_counter = 0
        
        # 当前活跃状态（GPU上）
        self.active_state: Optional[Dict] = None
        self.active_page_id: Optional[int] = None
        
    def _create_empty_page(self, page_id: int, start_pos: int, device: torch.device) -> StatePage:
        """创建空的状态页"""
        return StatePage(
            page_id=page_id,
            start_pos=start_pos,
            end_pos=start_pos + self.page_size,
            state=torch.zeros(self.batch_size, self.d_state, device=device, dtype=torch.float64),
            ent_k_cache=torch.zeros(self.batch_size, self.window, self.ent_rank, device=device, dtype=torch.float32),
            ent_v_cache=torch.zeros(self.batch_size, self.window, self.d_state, device=device, dtype=torch.float32),
            device=device,
            is_on_gpu=(device.type == 'cuda'),
        )
    
    def _evict_lru_page(self):
        """淘汰最久未使用的GPU页到CPU"""
        if self.gpu_page_count <= 1:
            return  # 至少保留一页在GPU
        
        # 找到最久未访问的GPU页
        lru_page_id = None
        lru_access = float('inf')
        
        for page_id, page in self.pages.items():
            if page.is_on_gpu and page.page_id != self.active_page_id:
                if page.last_access < lru_access:
                    lru_access = page.last_access
                    lru_page_id = page_id
        
        if lru_page_id is not None:
            page = self.pages[lru_page_id]
            # 卸载到CPU
            page.state = page.state.cpu()
            page.ent_k_cache = page.ent_k_cache.cpu()
            page.ent_v_cache = page.ent_v_cache.cpu()
            page.is_on_gpu = False
            page.device = torch.device('cpu')
            self.gpu_page_count -= 1
    
    def _load_page_to_gpu(self, page_id: int):
        """将页加载到GPU"""
        if page_id not in self.pages:
            return
        
        page = self.pages[page_id]
        
        if page.is_on_gpu:
            # 已经在GPU，更新访问时间
            page.last_access = self.access_counter
            self.access_counter += 1
            return
        
        # 需要加载到GPU
        while self.gpu_page_count >= self.max_gpu_pages:
            self._evict_lru_page()
        
        page.state = page.state.to(self.device)
        page.ent_k_cache = page.ent_k_cache.to(self.device)
        page.ent_v_cache = page.ent_v_cache.to(self.device)
        page.is_on_gpu = True
        page.device = self.device
        page.last_access = self.access_counter
        self.access_counter += 1
        self.gpu_page_count += 1
        
        # 移动到OrderedDict末尾（最近使用）
        self.pages.move_to_end(page_id)
    
    def get_page_for_position(self, pos: int) -> int:
        """获取位置对应的页ID"""
        return pos // self.page_size
    
    def save_state(self, pos: int, state_dict: Dict):
        """保存状态到对应位置的页
        
        Args:
            pos: 当前位置
            state_dict: 包含 state, ent_k_cache, ent_v_cache 的字典
        """
        page_id = self.get_page_for_position(pos)
        
        # 如果页不存在，创建新页
        if page_id not in self.pages:
            start_pos = page_id * self.page_size
            self.pages[page_id] = self._create_empty_page(page_id, start_pos, self.device)
            self.gpu_page_count += 1
            
            # 如果超过GPU页数限制，淘汰旧页
            while self.gpu_page_count > self.max_gpu_pages:
                self._evict_lru_page()
        
        # 更新页数据
        page = self.pages[page_id]
        page.state.copy_(state_dict['state'])
        page.ent_k_cache.copy_(state_dict['ent_k_cache'])
        page.ent_v_cache.copy_(state_dict['ent_v_cache'])
        page.last_access = self.access_counter
        self.access_counter += 1
        
        # 更新活跃状态
        self.active_state = state_dict.copy()
        self.active_page_id = page_id
    
    def load_state(self, pos: int) -> Optional[Dict]:
        """加载位置对应的状态
        
        Args:
            pos: 当前位置
            
        Returns:
            状态字典，如果页不存在返回None
        """
        page_id = self.get_page_for_position(pos)
        
        # 如果是当前活跃页，直接返回
        if page_id == self.active_page_id and self.active_state is not None:
            return self.active_state
        
        # 加载对应页
        if page_id not in self.pages:
            return None
        
        self._load_page_to_gpu(page_id)
        page = self.pages[page_id]
        
        self.active_state = {
            'state': page.state,
            'ent_k_cache': page.ent_k_cache,
            'ent_v_cache': page.ent_v_cache,
            'seq_pos': pos - page.start_pos,  # 页内相对位置
        }
        self.active_page_id = page_id
        
        return self.active_state
    
    def get_memory_stats(self) -> Dict:
        """获取内存使用统计"""
        gpu_mem = 0
        cpu_mem = 0
        
        for page in self.pages.values():
            # state: batch * d_state * 8 bytes (float64)
            # ent_k_cache: batch * window * ent_rank * 4 bytes (float32)
            # ent_v_cache: batch * window * d_state * 4 bytes (float32)
            page_mem = (
                self.batch_size * self.d_state * 8 +
                self.batch_size * self.window * self.ent_rank * 4 +
                self.batch_size * self.window * self.d_state * 4
            )
            
            if page.is_on_gpu:
                gpu_mem += page_mem
            else:
                cpu_mem += page_mem
        
        return {
            'gpu_pages': self.gpu_page_count,
            'cpu_pages': len(self.pages) - self.gpu_page_count,
            'total_pages': len(self.pages),
            'gpu_memory_mb': gpu_mem / (1024 ** 2),
            'cpu_memory_mb': cpu_mem / (1024 ** 2),
        }
    
    def clear(self):
        """清空所有缓存"""
        self.pages.clear()
        self.gpu_page_count = 0
        self.active_state = None
        self.active_page_id = None
