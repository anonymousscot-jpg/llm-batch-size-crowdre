import pandas as pd
import numpy as np
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
                except Exception as e:
                    pass
    if not results:
        return None
    return pd.concat(results, ignore_index=True)

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True 
            dominated = np.all(costs >= c, axis=1) & np.any(costs > c, axis=1)
            is_efficient[dominated] = False
    return is_efficient

def analyze_all_batches():
    df = load_data()
    target_tasks = ['Quaternary', 'Quinary']
    default_models = [
        'DeepSeek-V3.2',
        'Gemma-4-31B (Think OFF)',
        'Llama-4-Maverick',
        'Mixtral-8x22B',
        'Nemotron-Ultra-253B (Think OFF)',
    ]
    filtered_df = df[(df['Task'].isin(target_tasks)) & (df['Model'].isin(default_models))].copy()
    
    # Average across tasks and prompts for every Model AND Batch Size combination
    stats = filtered_df.groupby(['Model', 'Batch Size'])[['Avg Macro F1', 'Avg Tokens/Req', 'Avg Time (s)']].mean().reset_index()
    
    # Normalize globally to calculate a fair 3D score
    f1_min, f1_max = stats['Avg Macro F1'].min(), stats['Avg Macro F1'].max()
    tok_min, tok_max = stats['Avg Tokens/Req'].min(), stats['Avg Tokens/Req'].max()
    time_min, time_max = stats['Avg Time (s)'].min(), stats['Avg Time (s)'].max()
    
    stats['F1_Norm'] = (stats['Avg Macro F1'] - f1_min) / (f1_max - f1_min)
    stats['Tok_Norm'] = (stats['Avg Tokens/Req'] - tok_min) / (tok_max - tok_min)
    stats['Time_Norm'] = (stats['Avg Time (s)'] - time_min) / (time_max - time_min)
    
    stats['3D Score'] = stats['F1_Norm'] - 0.5 * (stats['Tok_Norm'] + stats['Time_Norm'])
    
    # Pareto dominance across ALL configurations
    costs = np.column_stack((
        stats['Avg Tokens/Req'].values,
        stats['Avg Time (s)'].values,
        -stats['Avg Macro F1'].values
    ))
    
    stats['Pareto'] = is_pareto_efficient(costs)
    
    # We want ALL models, sorted by 3D score
    all_configs = stats.sort_values('3D Score', ascending=False)
    
    final_cols = ['Model', 'Batch Size', 'Avg Macro F1', 'Avg Tokens/Req', 'Avg Time (s)', '3D Score', 'Pareto']
    out_table = all_configs[final_cols].copy()
    
    out_table['Avg Macro F1'] = out_table['Avg Macro F1'].round(3)
    out_table['Avg Tokens/Req'] = out_table['Avg Tokens/Req'].round(1)
    out_table['Avg Time (s)'] = out_table['Avg Time (s)'].round(1)
    out_table['3D Score'] = out_table['3D Score'].round(3)
    
    # Map Pareto boolean to Yes/No
    out_table['Pareto'] = out_table['Pareto'].map({True: 'Yes', False: 'No'})
    out_table.rename(columns={'Pareto': 'On Pareto Frontier?'}, inplace=True)
    
    print("\n" + "="*80)
    print("ALL CONFIGURATIONS (Ranked by 3D Efficiency Score)")
    print("="*80)
    # Print top 20 in terminal so it's readable
    pd.set_option('display.max_rows', None)
    print(out_table.to_string(index=False))
    
    global_best = out_table.iloc[0]
    print("\n" + "="*80)
    print("THE GLOBAL BEST OPERATING POINT:")
    print(f"Model: {global_best['Model']}")
    print(f"Batch Size: {global_best['Batch Size']}")
    print(f"Macro F1: {global_best['Avg Macro F1']}")
    print(f"Tokens/Req: {global_best['Avg Tokens/Req']}")
    print(f"Time (s): {global_best['Avg Time (s)']}")
    print(f"Score: {global_best['3D Score']}")
    print("="*80)
    
    latex_table = out_table.to_latex(
        index=False,
        longtable=True,
        caption="Comprehensive analysis of all models across all batch sizes for Quaternary and Quinary tasks. Configurations are ranked by their composite 3D Efficiency Score. Only a fraction of configurations survive strict Pareto dominance.",
        label="tab:global_efficiency_all_models"
    )
    out_tex_dir = BASE_DIR / 'analysis_output'
    out_tex_dir.mkdir(exist_ok=True)
    with open(out_tex_dir / 'Global_Pareto_All_Batches.tex', 'w') as f:
        f.write(latex_table)
        
if __name__ == "__main__":
    analyze_all_batches()
