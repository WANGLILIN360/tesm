#!/usr/bin/env python3
"""
TESM Bug 修复补丁

使用方法:
    python bug_fixes.py

这会直接在源代码中应用修复。
"""

import sys
from pathlib import Path

TESM_ROOT = Path('/mnt/agents/tesm')


def fix_tesm_siso_seq_len_check():
    """Bug 1: TESM_SISO.forward 添加序列长度检查"""
    
    target_file = TESM_ROOT / 'tesm_ssm/modules/tesm.py'
    content = target_file.read_text()
    
    # 在 forward 方法开头添加检查
    old_code = '''    def forward(self, u, inference_params=None, cross_layer_state=None, prev_state=None, **kwargs):
        # 处理可能的额外参数（如从上层传递的labels等）
        batch, seqlen, _ = u.shape'''
    
    new_code = '''    def forward(self, u, inference_params=None, cross_layer_state=None, prev_state=None, **kwargs):
        # 处理可能的额外参数（如从上层传递的labels等）
        batch, seqlen, _ = u.shape
        
        # 序列长度检查
        if seqlen > self.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")
        
        # 空序列检查
        if seqlen == 0:
            return u, None'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 1: TESM_SISO.forward 序列长度检查")
    else:
        print("  [SKIP] Bug 1: 代码不匹配，可能已修复或结构改变")


def fix_tesm_siso_param_validation():
    """Bug 3: 添加参数验证，确保 d_state > 0 等"""
    
    target_file = TESM_ROOT / 'tesm_ssm/modules/tesm.py'
    content = target_file.read_text()
    
    # 在 TESM_SISO.__init__ 中添加验证
    old_code = '''        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand'''
    
    new_code = '''        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # 参数验证
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if ent_rank <= 0:
            raise ValueError(f"ent_rank must be positive, got {ent_rank}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 3: TESM_SISO.__init__ 参数验证")
    else:
        print("  [SKIP] Bug 3: 代码不匹配")


def fix_mimo_d_head_warning():
    """Bug 4: MIMO d_head 整除警告"""
    
    target_file = TESM_ROOT / 'tesm_ssm/modules/tesm_mimo.py'
    content = target_file.read_text()
    
    # 在 d_head 计算后添加警告
    old_code = '''        self.d_inner = int(expand * d_model)
        self.d_head = self.d_inner // n_heads  # 每头维度
        self.d_state_total = d_state * n_heads  # 总状态维度'''
    
    new_code = '''        self.d_inner = int(expand * d_model)
        self.d_head = self.d_inner // n_heads  # 每头维度
        if self.d_inner % n_heads != 0:
            import warnings
            warnings.warn(f"d_inner ({self.d_inner}) is not divisible by n_heads ({n_heads}). "
                         f"d_head will be truncated to {self.d_head}. "
                         f"Consider using d_model such that (expand * d_model) % n_heads == 0")
        self.d_state_total = d_state * n_heads  # 总状态维度'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 4: MIMO d_head 整除警告")
    else:
        print("  [SKIP] Bug 4: 代码不匹配")


def fix_temperature_invalid_schedule():
    """Bug 6: 无效 annealing_schedule 添加警告"""
    
    target_file = TESM_ROOT / 'tesm_ssm/modules/tesm.py'
    content = target_file.read_text()
    
    # 在 get_temperature 方法中
    old_code = '''        else:
            T = self.T_start'''
    
    new_code = '''        else:
            import warnings
            warnings.warn(f"Unknown annealing_schedule '{self.annealing_schedule}', "
                         f"falling back to constant T_start={self.T_start}. "
                         f"Valid options: 'linear', 'exponential', 'cosine'")
            T = self.T_start'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 6: 无效 annealing_schedule 警告")
    else:
        print("  [SKIP] Bug 6: 代码不匹配")


def fix_parallel_prefix_scan_clamp():
    """Bug 5: 提高 _parallel_prefix_scan 的数值稳定性"""
    
    target_file = TESM_ROOT / 'tesm_ssm/modules/tesm.py'
    content = target_file.read_text()
    
    # 提高 clamp_min 的值
    old_code = '''        A = torch.cumprod(decay, dim=1)  # (B, L, D)
        weighted_update = update / A.clamp_min(1e-30)  # (B, L, D)'''
    
    new_code = '''        A = torch.cumprod(decay, dim=1)  # (B, L, D)
        weighted_update = update / A.clamp_min(1e-12)  # (B, L, D)'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 5: 提高 _parallel_prefix_scan 数值稳定性 (1e-30 -> 1e-12)")
    else:
        print("  [SKIP] Bug 5: 代码不匹配")


def fix_mixer_model_empty_seq():
    """Bug 8: MixerModel 添加空序列检查"""
    
    target_file = TESM_ROOT / 'tesm_ssm/models/mixer_seq_simple.py'
    content = target_file.read_text()
    
    old_code = '''    def forward(self, input_ids, inference_params=None, prev_states=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:'''
    
    new_code = '''    def forward(self, input_ids, inference_params=None, prev_states=None, **mixer_kwargs):
        batch_size, seqlen = input_ids.shape
        if seqlen == 0:
            raise ValueError("Input sequence length is 0")
        if seqlen > self.config.max_seq_len:'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        target_file.write_text(content)
        print("  [FIXED] Bug 8: MixerModel 空序列检查")
    else:
        print("  [SKIP] Bug 8: 代码不匹配")


def main():
    print("="*60)
    print("TESM Bug 修复补丁")
    print("="*60)
    print()
    
    print("[应用修复...]")
    fix_tesm_siso_seq_len_check()
    fix_tesm_siso_param_validation()
    fix_mimo_d_head_warning()
    fix_temperature_invalid_schedule()
    fix_parallel_prefix_scan_clamp()
    fix_mixer_model_empty_seq()
    
    print()
    print("="*60)
    print("修复完成！")
    print("="*60)
    print()
    print("注意:")
    print("  - 修复已直接应用到源代码")
    print("  - 建议重新运行测试验证修复效果")
    print("  - 如需恢复原始代码，请使用 git checkout")


if __name__ == '__main__':
    main()
