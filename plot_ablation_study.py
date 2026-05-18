import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import matplotlib.patches as mpatches


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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.15
})

df_melted = df.melt(id_vars=['Dataset', 'Variant'], value_vars=['F1', 'eTaF1'],
                    var_name='Metric', value_name='Score')

consistent_palette = ["#4A6984", "#95A9BA"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
datasets = ['BATADAL', 'HAI', 'SWaT']
variant_order = ['DecouplePhy', 'w/o Micro-Attention', 'w/o Macro-Phy']

for i, ds in enumerate(datasets):
    ax = axes[i]
    ds_data = df_melted[df_melted['Dataset'] == ds]
    sns.barplot(data=ds_data, y='Variant', x='Score', hue='Metric',
                order=variant_order, hue_order=['F1', 'eTaF1'],
                ax=ax, palette=consistent_palette,
                edgecolor='#2C3E50', linewidth=0.8, alpha=0.9, orient='h')

    for j, bar in enumerate(ax.patches):
        if j == 0 or j == 3:
            bar.set_hatch('////' if j == 0 else '....')
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_title(f"Ablation: {ds}", fontweight='bold', pad=15)
    ax.set_xlabel("Score (%)")
    ax.set_ylabel("")
    if i > 0: ax.set_yticklabels([])
    ax.set_xlim(0, 110)
    ax.grid(axis='x', linestyle='--', alpha=0.15)
    sns.despine(ax=ax)
    if ax.get_legend(): ax.get_legend().remove()


fig.subplots_adjust(top=0.75, bottom=0.15, left=0.15, right=0.95, wspace=0.25)


f1_patch = mpatches.Patch(facecolor=consistent_palette[0], label='F1 Score (Baselines)', alpha=0.9, edgecolor='#2C3E50')
eta_patch = mpatches.Patch(facecolor=consistent_palette[1], label='eTaF1 Score (Baselines)', alpha=0.9,
                           edgecolor='#2C3E50')
full_f1 = mpatches.Patch(facecolor=consistent_palette[0], label='DecouplePhy (F1)', hatch='////', edgecolor='black',
                         linewidth=1.2)
full_eta = mpatches.Patch(facecolor=consistent_palette[1], label='DecouplePhy (eTaF1)', hatch='....', edgecolor='black',
                          linewidth=1.2)

fig.legend(handles=[f1_patch, full_f1, eta_patch, full_eta],
           loc='upper left', ncol=2, frameon=False,
           bbox_to_anchor=(0.15, 0.98))

plt.savefig('Figure6.pdf', dpi=2000, bbox_inches='tight')