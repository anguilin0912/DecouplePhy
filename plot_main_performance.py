import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import matplotlib.patches as mpatches


data = """Dataset,IDS,F1,eTaF1
BATADAL,GeCo,82.3,92.4
BATADAL,SIMPLE,47.2,45.7
BATADAL,TABOR,12.7,24.1
BATADAL,Invariant,34.0,29.3
BATADAL,Seq2SeqNN,9.6,11.0
BATADAL,PASAD,29.1,14.1
BATADAL,DecouplePhy,86.9,89.2
HAI,GeCo,63.9,69.4
HAI,SIMPLE,54.7,71.7
HAI,TABOR,8.7,0.0
HAI,Invariant,16.2,38.3
HAI,Seq2SeqNN,6.0,4.2
HAI,PASAD,5.3,1.3
HAI,DecouplePhy,74.6,76.8
SWaT,GeCo,86.2,70.2
SWaT,SIMPLE,77.9,52.3
SWaT,TABOR,77.9,27.3
SWaT,Invariant,80.8,38.6
SWaT,Seq2SeqNN,17.5,44.9
SWaT,PASAD,44.6,7.5
SWaT,DecouplePhy,83.3,67.7"""

df = pd.read_csv(StringIO(data))


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.1
})

df_melted = df.melt(id_vars=['Dataset', 'IDS'], value_vars=['F1', 'eTaF1'],
                    var_name='Metric', value_name='Score')

consistent_palette = ["#4A6984", "#95A9BA"]


fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
datasets = ['BATADAL', 'HAI', 'SWaT']
model_order = ['GeCo', 'SIMPLE', 'TABOR', 'Invariant', 'Seq2SeqNN', 'PASAD', 'DecouplePhy']

for i, ds in enumerate(datasets):
    ax = axes[i]
    ds_data = df_melted[df_melted['Dataset'] == ds]

    sns.barplot(
        data=ds_data, x='IDS', y='Score', hue='Metric',
        order=model_order, hue_order=['F1', 'eTaF1'],
        ax=ax, palette=consistent_palette,
        edgecolor='#2C3E50', linewidth=0.7, alpha=0.9
    )

    for j, bar in enumerate(ax.patches):
        if j == 6 or j == 13:
            bar.set_hatch('////' if j == 6 else '....')
            bar.set_edgecolor('black')
            bar.set_linewidth(1.2)

    ax.set_title(f"Dataset: {ds}", fontweight='bold', pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Detection Score (%)" if i == 0 else "")
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.15)
    sns.despine(ax=ax)
    if ax.get_legend(): ax.get_legend().remove()


fig.subplots_adjust(top=0.75, bottom=0.22, left=0.08, right=0.95, wspace=0.2)


f1_patch = mpatches.Patch(facecolor=consistent_palette[0], label='F1 Score (Baselines)', alpha=0.9, edgecolor='#2C3E50')
eta_patch = mpatches.Patch(facecolor=consistent_palette[1], label='eTaF1 Score (Baselines)', alpha=0.9,
                           edgecolor='#2C3E50')
prop_f1 = mpatches.Patch(facecolor=consistent_palette[0], label='DecouplePhy (F1)', hatch='////', edgecolor='black',
                         linewidth=1.2)
prop_eta = mpatches.Patch(facecolor=consistent_palette[1], label='DecouplePhy (eTaF1)', hatch='....', edgecolor='black',
                          linewidth=1.2)

fig.legend(handles=[f1_patch, prop_f1, eta_patch, prop_eta],
           loc='upper left', ncol=2, frameon=False,
           bbox_to_anchor=(0.08, 0.98))

plt.savefig('Figure5.pdf', dpi=2000, bbox_inches='tight')