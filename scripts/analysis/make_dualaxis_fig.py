import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'
DIRS = {'DeepSeek-V3.2': ['DeepSeek', 'deepseek-v3.2-cloud-batch'],
        'Gemma-4-31B': ['Gemma', 'gemma4-31b-cloud-batch'],
        'Llama-4-Maverick': ['Nvidia', 'meta-llama-4-maverick-17b-batch'],
        'Mixtral-8x22B': ['Mixtral', 'mixtral_8x22b'],
        'Nemotron-Ultra-253B': ['Nvidia', 'llama-3.1-nemotron-ultra-253b-v1-batch-off']}
ORDER = ['Gemma-4-31B', 'Mixtral-8x22B', 'Nemotron-Ultra-253B', 'Llama-4-Maverick', 'DeepSeek-V3.2']
COLORS = {'Gemma-4-31B': '#2ca02c', 'Mixtral-8x22B': '#ff7f0e', 'Nemotron-Ultra-253B': '#17becf',
          'Llama-4-Maverick': '#d62728', 'DeepSeek-V3.2': '#1f77b4'}
TASKS = [('Quaternary_Easy', 'Quaternary'), ('Quinary', 'Quinary')]
BATCHES = [1, 2, 4, 8, 16, 32, 64]

# Both rows average over BOTH prompts (never a single prompt).
#   Top row    : two overall-average lines, one over all five models and one
#                with the Mixtral outlier removed, to expose the general trend.
#   Bottom row : kept per model -> five lines per task.
ALL5_F1 = '#000000'
NOMIX_F1 = '#1f77b4'
OVERALL_TOK = '#888888'


def load(model, task):
    """Per-model curve, averaged over both prompts at each batch size."""
    p = RESULTS / DIRS[model][1] / task / 'batch_summary_aggregated.csv'
    d = pd.read_csv(p)
    d = d.groupby('Batch Size', as_index=False).mean(numeric_only=True)
    return d.set_index('Batch Size').reindex(BATCHES)


def task_mean(task, exclude=()):
    """Average across models (each already averaged over both prompts)."""
    models = [m for m in ORDER if m not in exclude]
    f1 = [load(m, task)['Avg Macro F1'].values for m in models]
    tok = [load(m, task)['Avg Tokens/Req'].values for m in models]
    return np.nanmean(f1, axis=0), np.nanmean(tok, axis=0)


# Per-row, per-task F1 limits.
F1_YLIM_OVERALL = {'Quaternary': (0.805, 0.850), 'Quinary': (0.685, 0.735)}
F1_YLIM_PERMODEL = {'Quaternary': (0.70, 0.89), 'Quinary': (0.58, 0.77)}

x = np.arange(len(BATCHES))
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))

# --- Top row: overall average (all five, and excluding Mixtral) ---
for c, (tkey, tname) in enumerate(TASKS):
    ax = axes[0, c]; ax2 = ax.twinx()
    f1_all, tok_all = task_mean(tkey)
    f1_nomix, _ = task_mean(tkey, exclude=('Mixtral-8x22B',))
    ax.plot(x, f1_all, color=ALL5_F1, lw=2.6, marker='o', ms=6, zorder=4)
    ax.plot(x, f1_nomix, color=NOMIX_F1, lw=2.4, marker='^', ms=6, zorder=4)
    ax2.plot(x, tok_all, color=OVERALL_TOK, lw=2.0, ls='--', marker='s', ms=5, zorder=2)
    ax.set_ylim(*F1_YLIM_OVERALL[tname])
    ax.set_title(tname, fontsize=13, fontweight='bold')
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    if c == 0:
        ax.set_ylabel('Overall average\n(both prompts)\n\nMacro $F_1$-score', fontsize=11)
    if c == 1:
        ax2.set_ylabel('Tokens / Requirement', fontsize=11)

# --- Bottom row: per-model lines, each averaged over both prompts ---
for c, (tkey, tname) in enumerate(TASKS):
    ax = axes[1, c]; ax2 = ax.twinx()
    for m in ORDER:
        d = load(m, tkey)
        ax.plot(x, d['Avg Macro F1'], color=COLORS[m], lw=2, marker='o', ms=4)
        ax2.plot(x, d['Avg Tokens/Req'], color=COLORS[m], lw=1.6, ls='--', alpha=0.8)
    ax.set_ylim(*F1_YLIM_PERMODEL[tname])
    if c == 0:
        ax.set_ylabel('Per model\n(average of both prompts)\n\nMacro $F_1$-score', fontsize=11)
    if c == 1:
        ax2.set_ylabel('Tokens / Requirement', fontsize=11)

for ax in axes.ravel():
    ax.set_xticks(x); ax.set_xticklabels(BATCHES)
    ax.grid(True, alpha=0.3)
for c in (0, 1):
    axes[1, c].set_xlabel('Batch Size', fontsize=12)

# Legend for the top row (placed above it).
top_handles = [
    Line2D([0], [0], color=ALL5_F1, lw=2.6, marker='o', label='Macro $F_1$, all five models'),
    Line2D([0], [0], color=NOMIX_F1, lw=2.4, marker='^', label='Macro $F_1$, excluding Mixtral'),
    Line2D([0], [0], color=OVERALL_TOK, lw=2.0, ls='--', marker='s', label='Tokens/Req (right axis)')]
leg_top = fig.legend(handles=top_handles, loc='upper center', ncol=3,
                     bbox_to_anchor=(0.5, 1.06), fontsize=10, frameon=False)
fig.add_artist(leg_top)

# Legend for the per-model bottom row (placed below it).
model_handles = [Line2D([0], [0], color=COLORS[m], lw=2.2, label=m) for m in ORDER]
style_handles = [
    Line2D([0], [0], color='black', lw=2.2, ls='-', label='Macro $F_1$ (solid, left)'),
    Line2D([0], [0], color='black', lw=1.8, ls='--', label='Tokens/Req (dashed, right)')]
fig.legend(handles=model_handles + style_handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False)

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
out = BASE / 'analysis_output' / 'Global_Averages' / 'DualAxis_F1_Tokens_Trends.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)
