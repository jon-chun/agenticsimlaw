#!/usr/bin/env python3
"""Generate Figure 3: BRD comparison bar chart with 3 dataset panels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 10
data = {
    'NLSY97 (base rate 36%)': {
        'models': ['Gemini 2.5\nFlash', 'GPT-4.1\nMini', 'GPT-4o\nMini'],
        'zs':     [0.460, 0.360, 0.470],
        'sc':     [0.460, 0.380, 0.490],
        'debate': [0.191, 0.303, 0.023],
    },
    'COMPAS (base rate 45%)': {
        'models': ['Gemini 2.5\nFlash', 'GPT-4.1\nMini', 'GPT-4o\nMini'],
        'zs':     [0.110, 0.080, 0.070],
        'sc':     [0.110, 0.060, 0.080],
        'debate': [0.098, 0.073, 0.077],
    },
    'Credit Default (base rate 22%)': {
        'models': ['GPT-4.1\nMini', 'GPT-4o\nMini'],
        'zs':     [0.040, 0.020],
        'sc':     [0.040, 0.020],
        'debate': [0.013, 0.090],
    },
}

# Colors matching the original figure
c_zs     = '#a6cee3'  # light blue
c_sc     = '#4a86c8'  # medium blue
c_debate = '#1a2744'  # dark navy

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                         gridspec_kw={'width_ratios': [3, 3, 2]})
fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18, left=0.06, right=0.98)

for ax, (title, d) in zip(axes, data.items()):
    models = d['models']
    n = len(models)
    x = np.arange(n)
    w = 0.25

    bars_zs = ax.bar(x - w, d['zs'], w, color=c_zs, edgecolor='white', linewidth=0.5)
    bars_sc = ax.bar(x,     d['sc'], w, color=c_sc, edgecolor='white', linewidth=0.5)
    bars_db = ax.bar(x + w, d['debate'], w, color=c_debate, edgecolor='white', linewidth=0.5)

    # Value labels
    for bars in [bars_zs, bars_sc, bars_db]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.55)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.tick_params(axis='y', labelsize=9)

axes[0].set_ylabel('Base-Rate Deviation (BRD)', fontsize=11)

# Shared legend at top
fig.legend(
    [plt.Rectangle((0,0),1,1, fc=c_zs),
     plt.Rectangle((0,0),1,1, fc=c_sc),
     plt.Rectangle((0,0),1,1, fc=c_debate)],
    ['Zero-shot', 'SC-K=59', 'Debate'],
    loc='upper center', ncol=3, fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, 0.99)
)

out = '../agenticsimlaw-main/papers/tmlr2026/images/figure_brd_comparison.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
