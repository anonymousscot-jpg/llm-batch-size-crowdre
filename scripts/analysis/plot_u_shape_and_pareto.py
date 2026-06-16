import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'

MODEL_DIRS = {
    'DeepSeek': ['DeepSeek', 'deepseek-v3.2-cloud-batch'],
    'Gemma (Think OFF)': ['Gemma', 'gemma4-31b-cloud-batch'],
    'Gemma (Think ON)': ['Gemma', 'gemma4-31b-cloud-batch-ton'],
    'Mixtral': ['Mixtral', 'mixtral_8x22b'],
    'Nemotron (Think OFF)': ['Nvidia', 'llama-3.1-nemotron-ultra-253b-v1-batch-off'],
    'Nemotron (Think ON)': ['Nvidia', 'llama-3.1-nemotron-ultra-253b-v1-batch-final'],
    'Llama-Maverick': ['Nvidia', 'meta-llama-4-maverick-17b-batch']
}

DISPLAY_NAME_MAP = {
    'DeepSeek': 'DeepSeek-V3.2',
    'Gemma (Think OFF)': 'Gemma-4-31B (Think OFF)',
    'Gemma (Think ON)': 'Gemma-4-31B (Think ON)',
    'Mixtral': 'Mixtral-8x22B',
    'Nemotron (Think OFF)': 'Nemotron-Ultra-253B (Think OFF)',
    'Nemotron (Think ON)': 'Nemotron-Ultra-253B (Think ON)',
    'Llama-Maverick': 'Llama-4-Maverick'
}

# Think OFF is the default consumer configuration. The main report figures use
# these five models; the Think ON variants are reserved for the RQ5 comparison.
DEFAULT_MODELS = [
    'DeepSeek-V3.2',
    'Gemma-4-31B (Think OFF)',
    'Llama-4-Maverick',
    'Mixtral-8x22B',
    'Nemotron-Ultra-253B (Think OFF)',
]

TARGET_TASKS = ['Quaternary_Easy', 'Quinary']
TASK_DISPLAY_MAP = {'Quaternary_Easy': 'Quaternary', 'Quinary': 'Quinary'}
OUTPUT_DIR = BASE_DIR / 'analysis_output' / 'Global_Averages'


def load_data():
    results = []
    for model_name, path_parts in MODEL_DIRS.items():
        display_name = DISPLAY_NAME_MAP.get(model_name, model_name)
        model_path = RESULTS_DIR / path_parts[1]
        if not model_path.exists():
            continue
        for task_dir in [d for d in model_path.iterdir() if d.is_dir()]:
            csv_file = task_dir / 'batch_summary_aggregated.csv'
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    required_cols = ["Batch Size", "Prompt", "Avg Macro F1", "Avg Tokens/Req", "Avg Failure Rate", "Avg Time (s)"]
                    if all(col in df.columns for col in required_cols):
                        subset_cols = ["Batch Size", "Prompt"]
                        if df.duplicated(subset=subset_cols).any():
                            df = df.groupby(subset_cols, as_index=False).mean(numeric_only=True)
                        df['Model'] = display_name
                        df['Task'] = task_dir.name
                        results.append(df)
                except Exception:
                    pass
    if not results:
        return None
    return pd.concat(results, ignore_index=True)


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_DIR / f'{name}.html'))
    try:
        fig.write_image(str(OUTPUT_DIR / f'{name}.png'), scale=2)
        print(f"Saved {name}.html and {name}.png")
    except Exception as e:
        print(f"Saved {name}.html (PNG export failed: {e})")


def plot_u_shape(df, models, name, title):
    f = df[(df['Task'].isin(TARGET_TASKS)) & (df['Model'].isin(models))].copy()
    avg_f1 = f.groupby(['Task', 'Model', 'Batch Size'])['Avg Macro F1'].mean().reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=[TASK_DISPLAY_MAP[t] for t in TARGET_TASKS])
    colors = px.colors.qualitative.Plotly
    model_list = [m for m in models if m in avg_f1['Model'].unique()]
    for i, task in enumerate(TARGET_TASKS):
        td = avg_f1[avg_f1['Task'] == task]
        for j, model in enumerate(model_list):
            md = td[td['Model'] == model].sort_values('Batch Size')
            fig.add_trace(go.Scatter(
                x=md['Batch Size'], y=md['Avg Macro F1'], mode='lines+markers',
                name=model, legendgroup=model, showlegend=(i == 1),
                line=dict(color=colors[j % len(colors)], width=2), marker=dict(size=8)
            ), row=1, col=i + 1)
    for c in (1, 2):
        fig.update_xaxes(type="log", tickvals=[1, 2, 4, 8, 16, 32, 64], title_text="Batch Size (Log Scale)", row=1, col=c)
    fig.update_yaxes(title_text="Average Macro F1", row=1, col=1)
    fig.update_layout(title_text=title, title_x=0.5, height=500, width=1000, template="plotly_white")
    _save(fig, name)


def plot_pareto(df, models, name):
    f = df[(df['Task'].isin(TARGET_TASKS)) & (df['Model'].isin(models))].copy()
    avg = f.groupby(['Model', 'Batch Size'])[['Avg Macro F1', 'Avg Tokens/Req', 'Avg Time (s)']].mean().reset_index()
    fig = px.scatter_3d(
        avg, x='Avg Time (s)', y='Avg Tokens/Req', z='Avg Macro F1', color='Model',
        hover_data=['Batch Size'], title='3D Pareto Frontier (Quaternary and Quinary)',
        opacity=0.85)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(scene=dict(
        xaxis_title='Execution Time (s)', yaxis_title='Tokens per Requirement', zaxis_title='Macro F1 Score',
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))), margin=dict(l=0, r=0, b=0, t=40))
    _save(fig, name)


def plot_failure_rate(df, models, name):
    f = df[(df['Task'].isin(TARGET_TASKS)) & (df['Model'].isin(models))].copy()
    # Failure rate (%) averaged across the default models and the two tasks, split by prompt.
    g = f.groupby(['Batch Size', 'Prompt'])['Avg Failure Rate'].mean().reset_index()
    g['Failure Rate (%)'] = g['Avg Failure Rate'] * 100.0
    prompt_label = {'Enhanced_Zero_Shot': 'Enhanced Zero-Shot', 'Few_Shot_with_Reasoning': 'Few-Shot with Reasoning'}
    g['Prompt'] = g['Prompt'].map(prompt_label).fillna(g['Prompt'])
    batches = [1, 2, 4, 8, 16, 32, 64]
    # Distinct colours, marker shapes, and label positions so the two series stay
    # readable even where they coincide (both are exactly 1% at a batch of 32).
    style = {
        'Enhanced Zero-Shot':      dict(color='#1f77b4', symbol='circle',          textpos='top center'),
        'Few-Shot with Reasoning': dict(color='#d62728', symbol='diamond-open',    textpos='bottom center'),
    }
    fig = go.Figure()
    for prompt in ['Enhanced Zero-Shot', 'Few-Shot with Reasoning']:
        sub = g[g['Prompt'] == prompt].set_index('Batch Size').reindex(batches).reset_index()
        st = style[prompt]
        fig.add_trace(go.Scatter(
            x=sub['Batch Size'].astype(str), y=sub['Failure Rate (%)'],
            mode='lines+markers+text', name=prompt,
            text=[f"{v:.1f}%" for v in sub['Failure Rate (%)']],
            textposition=st['textpos'], textfont=dict(size=13, color=st['color']),
            line=dict(color=st['color'], width=2.5),
            marker=dict(size=12, symbol=st['symbol'],
                        line=dict(width=2, color=st['color']))))
    fig.update_layout(
        title='Structural Batch Failure Rate vs Batch Size (Quaternary and Quinary)',
        xaxis_title='Batch Size', yaxis_title='Failure Rate (%)',
        height=520, width=900, template='plotly_white',
        font=dict(size=14), legend=dict(font=dict(size=14)))
    fig.update_yaxes(range=[-0.25, 2.5])
    _save(fig, name)


def plot_think_comparison(df, name):
    models = ['Gemma-4-31B (Think OFF)', 'Gemma-4-31B (Think ON)',
              'Nemotron-Ultra-253B (Think OFF)', 'Nemotron-Ultra-253B (Think ON)']
    plot_u_shape(df, models, name, 'Think ON vs Think OFF: Macro F1 vs Batch Size')


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    if df is None:
        print("No data found!")
    else:
        # Main report figures use the Think OFF default model set.
        plot_u_shape(df, DEFAULT_MODELS, 'U_Shape_F1_Curve', 'Macro F1 vs Batch Size (Quaternary and Quinary)')
        plot_pareto(df, DEFAULT_MODELS, 'Interactive_3D_Pareto_QuaternaryEasy_Quinary')
        plot_failure_rate(df, DEFAULT_MODELS, 'Failure_Rate_by_Batch')
        # RQ5 reasoning comparison keeps both ON and OFF variants.
        plot_think_comparison(df, 'Think_OnOff_Comparison')
