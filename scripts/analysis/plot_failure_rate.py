"""Figure: structural batch failure rate vs batch size, split by prompt.

Standalone matplotlib + seaborn generator (no plotly/kaleido dependency), in
the same visual style as make_dualaxis_fig.py. The structural failure rate is
averaged over the five default (reasoning-off) models and both target tasks,
split by prompt. Each value label is placed on the side that moves it AWAY from
the other series (higher value above, lower below, decided per batch), so the
numbers stay readable even where the two lines nearly touch (batches 2 and 4)
or coincide exactly (both 1% at a batch of 32).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'

# Five default (reasoning-off) model result folders.
MODEL_DIRS = {
    'Gemma-4-31B': 'gemma4-31b-cloud-batch',
    'Mixtral-8x22B': 'mixtral_8x22b',
    'Nemotron-Ultra-253B': 'llama-3.1-nemotron-ultra-253b-v1-batch-off',
    'Llama-4-Maverick': 'meta-llama-4-maverick-17b-batch',
    'DeepSeek-V3.2': 'deepseek-v3.2-cloud-batch',
}
TASKS = ['Quaternary', 'Quinary']
BATCHES = [1, 2, 4, 8, 16, 32, 64]
EZS, FSR = 'Enhanced_Zero_Shot', 'Few_Shot_with_Reasoning'
LABELS = {EZS: 'Enhanced Zero-Shot', FSR: 'Few-Shot with Reasoning'}
STYLE = {
    EZS: dict(color='#1f77b4', marker='o', mfc='#1f77b4'),   # filled circle
    FSR: dict(color='#d62728', marker='D', mfc='white'),     # open diamond
}


def load():
    """Mean failure rate (%) over the five models and both tasks, per (batch, prompt)."""
    frames = [pd.read_csv(RESULTS / folder / task / 'batch_summary_aggregated.csv')
              for folder in MODEL_DIRS.values() for task in TASKS]
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(['Batch Size', 'Prompt'])['Avg Failure Rate'].mean().reset_index()
    g['Failure Rate (%)'] = g['Avg Failure Rate'] * 100.0
    return g.pivot(index='Batch Size', columns='Prompt',
                   values='Failure Rate (%)').reindex(BATCHES)


def main():
    pivot = load()
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(BATCHES))

    for prompt in (EZS, FSR):
        st = STYLE[prompt]
        ax.plot(x, pivot[prompt].values, color=st['color'], lw=2.5,
                marker=st['marker'], ms=11, mfc=st['mfc'], mec=st['color'],
                mew=2, label=LABELS[prompt], zorder=3)

    # Per batch: label the higher point above and the lower point below, so the
    # two numbers never collide. Ties (e.g. batch 32) split deterministically.
    ezs, fsr = pivot[EZS].values, pivot[FSR].values
    for i in range(len(BATCHES)):
        ezs_above = ezs[i] >= fsr[i]
        for prompt, val, above in ((EZS, ezs[i], ezs_above), (FSR, fsr[i], not ezs_above)):
            dy, va = (10, 'bottom') if above else (-10, 'top')
            ax.annotate(f'{val:.1f}%', (x[i], val), textcoords='offset points',
                        xytext=(0, dy), ha='center', va=va, fontsize=12,
                        fontweight='bold', color=STYLE[prompt]['color'])

    ax.set_xticks(x); ax.set_xticklabels(BATCHES)
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Failure Rate (%)', fontsize=14)
    ax.set_title('Structural Batch Failure Rate vs Batch Size '
                 '(Quaternary and Quinary)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.4, 2.6)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=True, fontsize=12, loc='upper left')
    sns.despine()
    fig.tight_layout()

    out = BASE / 'analysis_output' / 'Global_Averages' / 'Failure_Rate_by_Batch.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
