"""Three-objective trade-off scatter in 3D (matplotlib).

Replaces the plotly export, which left the plot in the top-left with a side
legend and a lot of empty space. This version centres the cube and puts the
legend below it, in a taller frame, so the figure fills the column. Each point
is one model at one batch size, with accuracy, token cost, and execution time
averaged over both tasks and both prompts.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'
DIRS = {'Gemma-4-31B': 'gemma4-31b-cloud-batch',
        'Mixtral-8x22B': 'mixtral_8x22b',
        'Nemotron-Ultra-253B': 'llama-3.1-nemotron-ultra-253b-v1-batch-off',
        'Llama-4-Maverick': 'meta-llama-4-maverick-17b-batch',
        'DeepSeek-V3.2': 'deepseek-v3.2-cloud-batch'}
COLORS = {'Gemma-4-31B': '#2ca02c', 'Mixtral-8x22B': '#ff7f0e',
          'Nemotron-Ultra-253B': '#17becf', 'Llama-4-Maverick': '#d62728',
          'DeepSeek-V3.2': '#1f77b4'}
ORDER = list(DIRS)
TASKS = ['Quaternary', 'Quinary']


def load():
    rows = []
    for m, d in DIRS.items():
        for t in TASKS:
            df = pd.read_csv(RESULTS / d / t / 'batch_summary_aggregated.csv')
            df['Model'] = m
            rows.append(df)
    A = pd.concat(rows)
    return A.groupby(['Model', 'Batch Size'])[
        ['Avg Macro F1', 'Avg Tokens/Req', 'Avg Time (s)']].mean().reset_index()


def main():
    st = load()
    fig = plt.figure(figsize=(6.6, 6.3))
    ax = fig.add_subplot(111, projection='3d')
    for m in ORDER:
        s = st[st.Model == m]
        ax.scatter(s['Avg Tokens/Req'], s['Avg Time (s)'], s['Avg Macro F1'],
                   color=COLORS[m], s=55, edgecolor='white', linewidth=0.6,
                   depthshade=False, label=m)
    ax.set_xlabel('Tokens / Requirement', fontsize=10, labelpad=8)
    ax.set_ylabel('Execution Time (s)', fontsize=10, labelpad=10)
    ax.set_zlabel('Macro $F_1$', fontsize=10, labelpad=4)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=18, azim=-58)
    ax.set_box_aspect((1, 1, 0.82))
    fig.legend(loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.12)

    out = BASE / 'analysis_output' / 'Global_Averages' / 'Interactive_3D_Pareto_QuaternaryEasy_Quinary.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print('saved', out)


if __name__ == '__main__':
    main()
