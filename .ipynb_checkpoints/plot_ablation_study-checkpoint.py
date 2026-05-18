import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import matplotlib.patches as mpatches

# 1. 消融实验数据
data = """Dataset,Variant,F1,eTaF1
BATADAL,w/o Macro-Phy,78.6,56.2
BATADAL,w/o Micro-Attention,73.5,86.7
BATADAL,DecouplePhy,86.9,89.2
HAI,w/o Macro-Phy,71.4,78.2
HAI,w/o Micro-Attention,73.8,77.2
HAI,DecouplePhy,74.6,76.8
SWaT,w/o Macro-Phy,46.4,20.8
SWaT,w/o Micro-Attention,79.9,60.6
SWaT,DecouplePhy,83.3,67.7"""

df = pd.read_csv(StringIO(data))

# 2. 全局学术风格设置 (Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.15
})

df_melted = df.melt(id_vars=['Dataset', 'Variant'], value_vars=['F1', 'eTaF1'], 
                    var_name='Metric', value_name='Score')

# 配色策略：保持与主实验一致的低饱和度蓝灰色系
# F1 = 深海蓝, eTaF1 = 浅蓝灰
consistent_palette = ["#4A6984", "#95A9BA"] 

# 3. 绘图执行
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
datasets = ['BATADAL', 'HAI', 'SWaT']
# 排序：将 DecouplePhy (完整版) 放在最上方，形成“向下坠落”的对比感
variant_order = ['DecouplePhy', 'w/o Micro-Attention', 'w/o Macro-Phy']

for i, ds in enumerate(datasets):
    ax = axes[i]
    ds_data = df_melted[df_melted['Dataset'] == ds]
    
    # 绘制横向柱状图 (orient='h')
    sns.barplot(
        data=ds_data, y='Variant', x='Score', hue='Metric', 
        order=variant_order, hue_order=['F1', 'eTaF1'],
        ax=ax, palette=consistent_palette, 
        edgecolor='#2C3E50', linewidth=0.8, alpha=0.9,
        orient='h'
    )
    
    # 针对 DecouplePhy 增加物理纹理强调
    # 在横向图中，前两个 patch 对应 DecouplePhy 的 F1 和 eTaF1
    for j, bar in enumerate(ax.patches):
        # 索引 0 (Metric1: DecouplePhy) 和 3 (Metric2: DecouplePhy)
        if j == 0 or j == 3: 
            bar.set_hatch('////' if j == 0 else '....')
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    # 4. 细节修饰
    ax.set_title(f"Ablation: {ds}", fontweight='bold', pad=15)
    ax.set_xlabel("Score (%)")
    ax.set_ylabel("")
    
    # 仅最左侧子图显示变体名称，保持图面整洁
    if i > 0:
        ax.set_yticklabels([])
    
    ax.set_xlim(0, 110)
    ax.grid(axis='x', linestyle='--', alpha=0.15)
    sns.despine(ax=ax)

    # 移除局部子图图例
    if ax.get_legend(): ax.get_legend().remove()

# 5. Figure 水平布局调整：为顶部全局图例留出空间
fig.subplots_adjust(top=0.82, bottom=0.15, left=0.15, right=0.95, wspace=0.25)

# --- 6. 创建单 Figure-level 全局图例 (靠左对齐) ---
f1_patch = mpatches.Patch(facecolor=consistent_palette[0], label='F1 (Baselines)', alpha=0.9, edgecolor='#2C3E50')
eta_patch = mpatches.Patch(facecolor=consistent_palette[1], label='eTaF1 (Baselines)', alpha=0.9, edgecolor='#2C3E50')
full_f1 = mpatches.Patch(facecolor=consistent_palette[0], label='DecouplePhy (F1)', hatch='////', edgecolor='black', linewidth=1.2)
full_eta = mpatches.Patch(facecolor=consistent_palette[1], label='DecouplePhy (eTaF1)', hatch='....', edgecolor='black', linewidth=1.2)

fig.legend(handles=[f1_patch, full_f1, eta_patch, full_eta], 
           loc='upper left', ncol=2, frameon=False,
           bbox_to_anchor=(0.15, 0.98)) 

# 7. 保存
plt.savefig('Ablation_Study_Horizontal.png', dpi=600, bbox_inches='tight')
print("✅ 消融实验横向柱状图已生成：Ablation_Study_Horizontal.png")