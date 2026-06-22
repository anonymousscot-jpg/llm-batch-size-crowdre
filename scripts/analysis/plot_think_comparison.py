"""Figure: Think ON vs Think OFF, Macro F1 vs batch size (RQ5).

Standalone matplotlib + seaborn generator (no plotly/kaleido dependency), in the
same visual style as make_dualaxis_fig.py and plot_failure_rate.py. Two panels
(Quaternary, Quinary), one line per reasoning variant of the two models that
expose a Think toggle. Each line is averaged over BOTH prompts at each batch
size. Colour encodes the model (green Gemma, cyan Nemotron, matching the
per-model palette elsewhere); line style encodes the reasoning mode
(solid = Think OFF, dashed = Think ON).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'
TASKS = ['Quaternary', 'Quinary']
BATCHES = [1, 2, 4, 8, 16, 32, 64]

# (display label, result folder, colour, think-on?) for each reasoning variant.
SERIES = [
    ('Gemma-4-31B (Think OFF)',         'gemma4-31b-cloud-batch',                          '#2ca02c', False),
    ('Gemma-4-31B (Think ON)',          'gemma4-31b-cloud-batch-ton',                      '#2ca02c', True),
    ('Nemotron-Ultra-253B (Think OFF)', 'llama-3.1-nemotron-ultra-253b-v1-batch-off',      '#17becf', False),
    ('Nemotron-Ultra-253B (Think ON)',  'llama-3.1-nemotron-ultra-253b-v1-batch-final',    '#17becf', True),
]


def curve(folder, task):
    """Macro F1 per batch size, averaged over both prompts."""
    d = pd.read_csv(RESULTS / folder / task / 'batch_summary_aggregated.csv')
    g = d.groupby('Batch Size')['Avg Macro F1'].mean()
    return g.reindex(BATCHES).values


def main():
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    x = np.arange(len(BATCHES))

    for ax, task in zip(axes, TASKS):
        for label, folder, color, think_on in SERIES:
            ax.plot(x, curve(folder, task), color=color, lw=2.4,
                    ls='--' if think_on else '-',
                    marker='^' if think_on else 'o', ms=7,
                    mfc='white' if think_on else color, mec=color, mew=2,
                    zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(BATCHES)
        ax.set_xlabel('Batch Size', fontsize=13)
        ax.set_title(task, fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel('Average Macro $F_1$ (both prompts)', fontsize=13)

    # Legend: colour = model, line style/marker = reasoning mode.
    handles = [
        Line2D([0], [0], color='#2ca02c', lw=2.4, label='Gemma-4-31B'),
        Line2D([0], [0], color='#17becf', lw=2.4, label='Nemotron-Ultra-253B'),
        Line2D([0], [0], color='gray', lw=2.4, ls='-', marker='o', mfc='gray',
               label='Think OFF (solid)'),
        Line2D([0], [0], color='gray', lw=2.4, ls='--', marker='^', mfc='white',
               label='Think ON (dashed)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Think ON vs Think OFF: Macro $F_1$ vs Batch Size',
                 fontsize=15, fontweight='bold')
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = BASE / 'analysis_output' / 'Global_Averages' / 'Think_OnOff_Comparison.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
