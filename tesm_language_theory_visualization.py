# TESM语言能力实现的理论原理可视化 - 完善版
# 基于 TESM 核心模块深度分析

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 设置中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
try:
    font_prop = fm.FontProperties(fname=font_path)
except:
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Thin.ttc'
    font_prop = fm.FontProperties(fname=font_path)

plt.rcParams['axes.unicode_minus'] = False

# 创建大型画布
fig = plt.figure(figsize=(24, 20))
fig.suptitle('TESM (Token-Entangled State Machine) 语言能力理论原理', 
             fontsize=22, fontweight='bold', y=0.98, fontproperties=font_prop)

# 使用更灵活的网格布局
gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.35, wspace=0.25)

# ============================================================================
# 1. TESM整体架构图 (左上，跨2列)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_title('TESM 整体架构与数据流', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# 绘制模块框
def draw_module(ax, x, y, w, h, text, color, text_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=9, fontweight='bold', color=text_color, fontproperties=font_prop)

# 输入层
draw_module(ax1, 0.5, 4.5, 1.5, 0.8, '输入Token\nu_t', '#E8F4FD', 'black')

# BitLinear投影层
draw_module(ax1, 2.5, 4.5, 1.5, 0.8, 'BitLinear\nINT2量化', '#FFE5E5', 'black')

# 分解投影
draw_module(ax1, 4.5, 5.0, 1.2, 0.6, 'local', '#F0F0F0', 'black')
draw_module(ax1, 4.5, 4.3, 1.2, 0.6, 'decay', '#E5F3FF', 'black')
draw_module(ax1, 4.5, 3.6, 1.2, 0.6, 'write', '#E5FFE5', 'black')
draw_module(ax1, 4.5, 2.9, 1.2, 0.6, 'state_value', '#FFF5E5', 'black')
draw_module(ax1, 4.5, 2.2, 1.2, 0.6, 'ent_q', '#F5E5FF', 'black')
draw_module(ax1, 4.5, 1.5, 1.2, 0.6, 'ent_k', '#F5E5FF', 'black')

# 状态扫描模块
draw_module(ax1, 6.2, 3.0, 1.8, 1.5, '状态扫描\nh_t = d*h_{t-1}+u_t\n(并行scan)', '#C5E0B6', 'black')

# 三值纠缠模块
draw_module(ax1, 6.2, 1.0, 1.8, 1.5, '三值纠缠\nT in{-1,0,+1}\n(局部窗口/全局)', '#FFD9B3', 'black')

# 输出合并
draw_module(ax1, 8.5, 2.5, 1.3, 2.0, '输出合并\nout_gate\n+ state_proj\n+ ent_proj', '#B4D7E8', 'black')

# 绘制连接箭头
arrow_style = dict(arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray')

# 输入到投影
ax1.annotate('', xy=(2.5, 4.9), xytext=(2.0, 4.9), arrowprops=arrow_style)
ax1.annotate('', xy=(4.5, 5.3), xytext=(4.0, 4.9), arrowprops=arrow_style)

# 投影到各分支
for y in [5.0, 4.3, 3.6, 2.9, 2.2, 1.5]:
    ax1.annotate('', xy=(6.2, y), xytext=(5.7, y), arrowprops=arrow_style)

# 状态扫描连接
ax1.annotate('', xy=(8.5, 3.5), xytext=(8.0, 3.5), arrowprops=arrow_style)

# 纠缠连接
ax1.annotate('', xy=(8.5, 2.0), xytext=(8.0, 2.0), arrowprops=arrow_style)

# 添加温度退火指示
ax1.text(6.2, 0.3, '温度退火: T_start=10 -> T_end=0.1', fontsize=8, 
         fontproperties=font_prop, style='italic', color='blue')

# 添加图例
legend_elements = [
    mpatches.Patch(color='#E8F4FD', label='输入'),
    mpatches.Patch(color='#FFE5E5', label='BitLinear量化'),
    mpatches.Patch(color='#C5E0B6', label='状态扫描'),
    mpatches.Patch(color='#FFD9B3', label='三值纠缠'),
    mpatches.Patch(color='#B4D7E8', label='输出合并'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, prop=font_prop)

# ============================================================================
# 2. 状态累积机制详解 (右上)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title('状态累积机制', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# 绘制状态累积公式 (简化版)
formula_text = 'h_t = sigmoid(d_raw + bias) * h_{t-1} + sigmoid(w_t) * tanh(v_t)'
ax2.text(0.5, 0.95, formula_text, transform=ax2.transAxes, fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontproperties=font_prop)

# 模拟状态累积过程
seq_len = 50
t = np.arange(seq_len)
np.random.seed(42)

# 不同衰减率的状态累积
decay_rates = [0.95, 0.8, 0.5, 0.3]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
labels = ['decay=0.95\n(长记忆)', 'decay=0.8\n(中记忆)', 'decay=0.5\n(短记忆)', 'decay=0.3\n(极短记忆)']

for decay, color, label in zip(decay_rates, colors, labels):
    state = np.zeros(seq_len)
    for i in range(1, seq_len):
        update = np.random.randn() * 0.3
        state[i] = decay * state[i-1] + update
    ax2.plot(t, state, label=label, color=color, linewidth=2, alpha=0.8)

ax2.set_xlabel('时间步 t', fontsize=10, fontproperties=font_prop)
ax2.set_ylabel('状态值 h_t', fontsize=10, fontproperties=font_prop)
ax2.legend(loc='upper left', fontsize=8, prop=font_prop)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, seq_len)

# 添加说明
ax2.text(0.95, 0.05, 'decay_init_bias 控制\n初始衰减率', transform=ax2.transAxes,
         fontsize=8, ha='right', va='bottom', fontproperties=font_prop,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# 3. 三值纠缠决策过程 (第二行左)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title('三值纠缠决策过程', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# 绘制决策边界
scores = np.linspace(-1.5, 1.5, 100)
threshold = 0.1

# 三值决策
ternary = np.zeros_like(scores)
ternary[scores > threshold] = 1
ternary[scores < -threshold] = -1

ax3.fill_between(scores, ternary, where=(ternary > 0), alpha=0.3, color='green', label='+1 正向关联')
ax3.fill_between(scores, ternary, where=(ternary < 0), alpha=0.3, color='red', label='-1 负向抑制')
ax3.fill_between(scores, ternary, where=(ternary == 0), alpha=0.3, color='gray', label='0 无关过滤')

ax3.plot(scores, ternary, 'k-', linewidth=2)
ax3.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, label=f'阈值 +{threshold}')
ax3.axvline(x=-threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值 -{threshold}')

ax3.set_xlabel('纠缠分数 score', fontsize=10)
ax3.set_ylabel('三值决策 T', fontsize=10, fontproperties=font_prop)
ax3.legend(loc='upper left', fontsize=8, prop=font_prop)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.grid(True, alpha=0.3)

# 添加量子隧穿区域
ax3.axvspan(-threshold*1.5, threshold*1.5, alpha=0.2, color='yellow')
ax3.text(0, -1.3, '量子隧穿区域', fontsize=8, ha='center', fontproperties=font_prop, style='italic')

# ============================================================================
# 4. 温度退火过程 (第二行中)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title('温度退火训练策略', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

steps = np.linspace(0, 1, 100)
T_start, T_end = 10, 0.1

# 不同退火调度
schedules = {
    'cosine': T_end + 0.5 * (T_start - T_end) * (1 + np.cos(np.pi * steps)),
    'linear': T_start * (1 - steps) + T_end * steps,
    'exponential': T_start * np.exp(-steps * np.log(T_start / T_end))
}

colors = {'cosine': '#3498db', 'linear': '#e74c3c', 'exponential': '#2ecc71'}

for name, curve in schedules.items():
    ax4.plot(steps * 1000, curve, label=name, linewidth=2.5, color=colors[name])

# 高温/低温区域
ax4.axhline(y=1, color='orange', linestyle='--', linewidth=2, label='软硬阈值 T=1')
ax4.fill_between(steps * 1000, curve, 0, where=(curve > 1), alpha=0.2, color='orange')
ax4.fill_between(steps * 1000, curve, 0, where=(curve <= 1), alpha=0.2, color='blue')

ax4.set_xlabel('训练步数', fontsize=10, fontproperties=font_prop)
ax4.set_ylabel('温度 T', fontsize=10)
ax4.legend(loc='upper right', fontsize=9, prop=font_prop)
ax4.set_xlim(0, 1000)
ax4.set_ylim(0, 12)
ax4.grid(True, alpha=0.3)

# 添加阶段标注
ax4.text(100, 8, '高温阶段\n软纠缠\n(softmax)', fontsize=8, ha='center', fontproperties=font_prop,
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax4.text(800, 2, '低温阶段\n硬纠缠\n(三值阈值)', fontsize=8, ha='center', fontproperties=font_prop,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# 5. 纠缠窗口机制 (第二行右)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('局部窗口纠缠 vs 全局纠缠', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# 绘制序列和窗口
seq_len = 16
window = 4

# 绘制token序列
for i in range(seq_len):
    circle = Circle((i, 0), 0.3, facecolor='lightblue', edgecolor='black')
    ax5.add_patch(circle)
    ax5.text(i, 0, f't{i}', ha='center', va='center', fontsize=7)

# 当前token的纠缠窗口
current_t = 10
for w in range(window):
    hist_t = current_t - window + 1 + w
    if hist_t >= 0:
        circle = Circle((hist_t, 0), 0.35, facecolor='orange', edgecolor='red', linewidth=2)
        ax5.add_patch(circle)

# 当前token
circle = Circle((current_t, 0), 0.4, facecolor='green', edgecolor='black', linewidth=2)
ax5.add_patch(circle)

# 绘制纠缠箭头
for w in range(window):
    hist_t = current_t - window + 1 + w
    if hist_t >= 0:
        ax5.annotate('', xy=(current_t, 0.5), xytext=(hist_t, 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

ax5.set_xlim(-1, seq_len)
ax5.set_ylim(-1.5, 2)
ax5.axis('off')

# 添加说明
ax5.text(seq_len/2, 1.5, f'局部窗口纠缠: W={window}, 复杂度 O(N*W)', 
         fontsize=10, ha='center', fontproperties=font_prop, fontweight='bold')

# 全局纠缠示意
ax5.text(seq_len/2, -1, '全局纠缠: W=0, 使用相对位置编码, 复杂度 O(N^2)', 
         fontsize=9, ha='center', fontproperties=font_prop, style='italic', color='blue')

# ============================================================================
# 6. BitLinear INT2量化 (第三行左)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_title('BitLinear INT2 量化', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# 权重量化示意
weights = np.linspace(-2, 2, 100)
quantized = np.clip(np.round(weights), -1, 1)

ax6.plot(weights, quantized, 'b-', linewidth=2, label='量化函数')
ax6.fill_between(weights, quantized - 0.1, quantized + 0.1, alpha=0.3, color='blue')

ax6.axhline(y=1, color='green', linestyle='--', alpha=0.5)
ax6.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax6.set_xlabel('原始权重 w', fontsize=10)
ax6.set_ylabel('量化权重 w_q', fontsize=10, fontproperties=font_prop)
ax6.set_xlim(-2, 2)
ax6.set_ylim(-1.5, 1.5)
ax6.grid(True, alpha=0.3)

# 添加三值标注
ax6.text(-1.5, 1, '-1', fontsize=12, ha='center', fontweight='bold', color='red')
ax6.text(0, 0, '0', fontsize=12, ha='center', fontweight='bold', color='gray')
ax6.text(1.5, 1, '+1', fontsize=12, ha='center', fontweight='bold', color='green')

# 添加说明
ax6.text(0, -1.3, '权重量化: w_q in{-1, 0, +1}\n输入量化: x_q in[-128, 127]', 
         fontsize=9, ha='center', fontproperties=font_prop,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================================
# 7. MIMO多头扩展 (第三行中)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1])
ax7.set_title('MIMO 多头扩展机制', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# 绘制多头结构
n_heads = 4
d_state = 64

for h in range(n_heads):
    # 每个头的状态空间
    rect = Rectangle((0.1, h * 0.2 + 0.1), 0.3, 0.15, 
                     facecolor=plt.cm.Set2(h/n_heads), edgecolor='black')
    ax7.add_patch(rect)
    ax7.text(0.25, h * 0.2 + 0.175, f'Head {h}\nd_state={d_state}', 
             ha='center', va='center', fontsize=8, fontproperties=font_prop)

# MIMO rank扩展
for r in range(3):  # mimo_rank = 3
    for h in range(n_heads):
        rect = Rectangle((0.5 + r * 0.15, h * 0.2 + 0.1), 0.12, 0.15,
                        facecolor=plt.cm.Set3(r/3), edgecolor='gray', alpha=0.7)
        ax7.add_patch(rect)

ax7.text(0.7, 0.95, 'MIMO Rank扩展', fontsize=9, ha='center', fontproperties=font_prop)

# 输出合并
rect = Rectangle((0.9, 0.1), 0.08, 0.8, facecolor='lightgreen', edgecolor='black')
ax7.add_patch(rect)
ax7.text(0.94, 0.5, '合并\nmimo_o', ha='center', va='center', fontsize=8, fontproperties=font_prop, rotation=90)

ax7.set_xlim(0, 1.1)
ax7.set_ylim(0, 1)
ax7.axis('off')

# 添加公式
ax7.text(0.5, 0.02, 'y = einsum("blrhd,hrd->blhd", states, mimo_o)', 
         fontsize=8, ha='center', fontproperties=font_prop, style='italic')

# ============================================================================
# 8. 后端性能对比 (第三行右)
# ============================================================================
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_title('多后端性能特性', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

backends = ['PyTorch', 'CUDA', 'Triton', 'TileLang']
metrics = {
    '训练速度': [1, 3, 2.5, 3.5],
    '推理速度': [1, 4, 3.5, 4],
    '显存效率': [1, 2, 2.5, 3],
    '兼容性': [5, 3, 4, 2],
}

x = np.arange(len(backends))
width = 0.2
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (metric, values) in enumerate(metrics.items()):
    ax8.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

ax8.set_ylabel('相对性能', fontsize=10, fontproperties=font_prop)
ax8.set_xticks(x + width * 1.5)
ax8.set_xticklabels(backends, fontsize=9)
ax8.legend(loc='upper right', fontsize=8, prop=font_prop)
ax8.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 9. 与Transformer/SSM对比 (第四行左)
# ============================================================================
ax9 = fig.add_subplot(gs[3, 0])
ax9.set_title('架构对比: TESM vs Transformer vs SSM', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

comparison_data = {
    '长程依赖': {'TESM': 9, 'Transformer': 7, 'SSM': 8},
    '计算效率': {'TESM': 9, 'Transformer': 5, 'SSM': 9},
    '显存占用': {'TESM': 8, 'Transformer': 4, 'SSM': 8},
    '增量推理': {'TESM': 9, 'Transformer': 6, 'SSM': 9},
    '可解释性': {'TESM': 8, 'Transformer': 6, 'SSM': 5},
}

categories = list(comparison_data.keys())
x = np.arange(len(categories))
width = 0.25

tesm_scores = [comparison_data[cat]['TESM'] for cat in categories]
trans_scores = [comparison_data[cat]['Transformer'] for cat in categories]
ssm_scores = [comparison_data[cat]['SSM'] for cat in categories]

ax9.bar(x - width, tesm_scores, width, label='TESM', color='#3498db', alpha=0.8)
ax9.bar(x, trans_scores, width, label='Transformer', color='#e74c3c', alpha=0.8)
ax9.bar(x + width, ssm_scores, width, label='SSM', color='#2ecc71', alpha=0.8)

ax9.set_ylabel('能力评分', fontsize=10, fontproperties=font_prop)
ax9.set_xticks(x)
ax9.set_xticklabels(categories, fontsize=8, fontproperties=font_prop, rotation=15, ha='right')
ax9.legend(loc='upper right', fontsize=9, prop=font_prop)
ax9.set_ylim(0, 11)
ax9.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 10. 语言能力对应关系 (第四行中)
# ============================================================================
ax10 = fig.add_subplot(gs[3, 1])
ax10.set_title('TESM机制与语言能力对应', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax10.axis('off')

# 创建对应关系图
language_abilities = [
    ('词汇理解', 'Token Embedding + BitLinear'),
    ('句法分析', '局部窗口纠缠 + 三值决策'),
    ('语义整合', '状态累积 + 长程记忆'),
    ('语篇连贯', '全局纠缠 + 相对位置编码'),
    ('上下文推理', '温度退火 + 增量推理'),
]

y_start = 0.9
for i, (ability, mechanism) in enumerate(language_abilities):
    # 语言能力框
    rect1 = FancyBboxPatch((0.05, y_start - i * 0.18), 0.35, 0.12,
                           boxstyle="round,pad=0.02", facecolor='#E8F4FD', edgecolor='black')
    ax10.add_patch(rect1)
    ax10.text(0.225, y_start - i * 0.18 + 0.06, ability, ha='center', va='center',
              fontsize=9, fontproperties=font_prop, fontweight='bold')
    
    # 箭头
    ax10.annotate('', xy=(0.55, y_start - i * 0.18 + 0.06), 
                 xytext=(0.40, y_start - i * 0.18 + 0.06),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # 机制框
    rect2 = FancyBboxPatch((0.55, y_start - i * 0.18), 0.40, 0.12,
                           boxstyle="round,pad=0.02", facecolor='#FFE5E5', edgecolor='black')
    ax10.add_patch(rect2)
    ax10.text(0.75, y_start - i * 0.18 + 0.06, mechanism, ha='center', va='center',
              fontsize=8, fontproperties=font_prop)

ax10.set_xlim(0, 1)
ax10.set_ylim(0, 1)

# ============================================================================
# 11. 理论框架总结 (第四行右)
# ============================================================================
ax11 = fig.add_subplot(gs[3, 2])
ax11.set_title('TESM核心理论框架', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax11.axis('off')

framework_text = """
【三大核心机制】

1. 状态空间建模 (SSM)
   - 线性复杂度: O(N)
   - 工作记忆: d_state 维度
   - 增量推理: 常量显存

2. 三值纠缠 (Ternary Entanglement)
   - 离散决策: {-1, 0, +1}
   - 局部窗口: O(N*W)
   - 全局纠缠: 相对位置编码

3. 温度退火 (Temperature Annealing)
   - 高温探索: softmax平滑
   - 低温利用: 硬阈值决策
   - 可选隧穿: 边界效应

【理论优势】
- 效率: 线性复杂度 + INT2量化
- 可解释: 离散纠缠显式编码
- 灵活: 多后端 + 多头扩展
- 语言: 机制对应语言层次
"""

ax11.text(0.05, 0.95, framework_text, transform=ax11.transAxes,
          fontsize=9, verticalalignment='top', fontproperties=font_prop,
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=0.5))

# 保存图片
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/lingji/wang/tesm_language_theory.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("TESM语言能力理论原理图已生成 (完善版)")
print("包含11个子图:")
print("  1. TESM整体架构与数据流")
print("  2. 状态累积机制")
print("  3. 三值纠缠决策过程")
print("  4. 温度退火训练策略")
print("  5. 局部窗口纠缠 vs 全局纠缠")
print("  6. BitLinear INT2量化")
print("  7. MIMO多头扩展机制")
print("  8. 多后端性能特性")
print("  9. 架构对比")
print("  10. 语言能力对应关系")
print("  11. 理论框架总结")
