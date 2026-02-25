#!/usr/bin/env python3
"""Generate cooperative vs adversarial BRD bar chart for Appendix I."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 17
labels = [
    'GPT-4o Mini\nNLSY97',
    'GPT-4.1 Mini\nNLSY97',
    'GPT-4o Mini\nCOMPAS',
    'GPT-4.1 Mini\nCOMPAS',
    'GPT-4o Mini\nCredit Default',
    'GPT-4.1 Mini\nCredit Default',
]
adv_brd  = [0.023, 0.303, 0.077, 0.073, 0.090, 0.013]
coop_brd = [0.090, 0.287, 0.107, 0.130, 0.003, 0.043]

# Colors: adversarial = dark navy (same as debate), cooperative = medium teal
c_adv  = '#1a2744'
c_coop = '#6baed6'

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.97)

x = np.arange(len(labels))
w = 0.32

bars_adv  = ax.bar(x - w/2, adv_brd,  w, color=c_adv,  edgecolor='white', linewidth=0.5)
bars_coop = ax.bar(x + w/2, coop_brd, w, color=c_coop, edgecolor='white', linewidth=0.5)

# Value labels
for bars in [bars_adv, bars_coop]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

# Star the winners
for i in range(len(labels)):
    if adv_brd[i] < coop_brd[i]:
        winner_bar = bars_adv[i]
    else:
        winner_bar = bars_coop[i]
    bx = winner_bar.get_x() + winner_bar.get_width()/2
    by = winner_bar.get_height()
    # bold the winner value label is enough

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Base-Rate Deviation (BRD)', fontsize=11)
ax.set_title('Adversarial vs. Cooperative Debate Calibration', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.38)
ax.set_yticks(np.arange(0, 0.40, 0.05))
ax.tick_params(axis='y', labelsize=9)

# Separator lines between datasets
for sep in [1.5, 3.5]:
    ax.axvline(sep, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)

# Dataset labels
for xpos, label in [(0.5, 'NLSY97'), (2.5, 'COMPAS'), (4.5, 'Credit Default')]:
    ax.text(xpos, 0.36, label, ha='center', va='top', fontsize=9,
            fontstyle='italic', color='grey')

ax.legend(
    [plt.Rectangle((0,0),1,1, fc=c_adv),
     plt.Rectangle((0,0),1,1, fc=c_coop)],
    ['Adversarial', 'Cooperative'],
    loc='upper center', fontsize=10, frameon=True, framealpha=0.9
)

out = '../agenticsimlaw-main/papers/tmlr2026/images/figure_cooperative_ablation.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
