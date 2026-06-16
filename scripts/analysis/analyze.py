import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_theme(style="whitegrid")

# Configuration
# Repo root is two levels up from scripts/analysis/. Results are read from
# results/; generated figures are written to analysis_output/.
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = BASE_DIR / 'analysis_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIRS = {
    'DeepSeek': ['DeepSeek', 'deepseek-v3.2-cloud-batch'],
    'Gemma (Think OFF)': ['Gemma', 'gemma4-31b-cloud-batch'],
    'Gemma (Think ON)': ['Gemma', 'gemma4-31b-cloud-batch-ton'],
    'Mixtral': ['Mixtral', 'mixtral_8x22b'],
    'Nemotron (Think OFF)': ['Nvidia', 'llama-3.1-nemotron-ultra-253b-v1-batch-off'],
    'Nemotron (Think ON)': ['Nvidia', 'llama-3.1-nemotron-ultra-253b-v1-batch-final'], 
    'Llama-Maverick': ['Nvidia', 'meta-llama-4-maverick-17b-batch']
}

# Mapping for full academic names in graphs/legends
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
            print(f"Warning: Model directory missing for {model_name}: {model_path}")
            continue
        
        for task_dir in [d for d in model_path.iterdir() if d.is_dir()]:
            csv_file = task_dir / 'batch_summary_aggregated.csv'
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    required_cols = ["Batch Size", "Prompt", "Avg Macro F1", "Avg Tokens/Req", "Avg Failure Rate", "Avg Time (s)"]
                    if all(col in df.columns for col in required_cols):
                        # --- DUPLICATE HANDLING START ---
                        subset_cols = ["Batch Size", "Prompt"]
                        if df.duplicated(subset=subset_cols).any():
                            n_dupes = df.duplicated(subset=subset_cols).sum()
                            print(f"  [!] Found {n_dupes} duplicate rows in {task_dir.name} ({display_name}). Aggregating...")
                            # Aggregate duplicates by taking the mean of all numeric values
                            df = df.groupby(subset_cols, as_index=False).mean(numeric_only=True)
                        # --- DUPLICATE HANDLING END ---
                        
                        df['Model'] = display_name
                        df['Task'] = task_dir.name
                        results.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

    if not results:
        return None
    return pd.concat(results, ignore_index=True)


def plot_graphs(df: pd.DataFrame):
    print(f"\nGenerating plots in {OUTPUT_DIR}/ ...")
    GLOBAL_DIR = OUTPUT_DIR / 'Global_Averages'
    GLOBAL_DIR.mkdir(exist_ok=True)
    
    # We will average the results across all 5-7 tasks for general batch scaling curves
    avg_df = df.groupby(['Model', 'Batch Size', 'Prompt']).mean(numeric_only=True).reset_index()

    # 1. Macro F1 vs Batch Size (Grouped by Model)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_df, x='Batch Size', y='Avg Macro F1', hue='Model', style='Prompt', markers=True, dashes=True)
    plt.xscale('log', base=2)
    plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
    plt.title('Average Macro F1-Score vs. Batch Size (Quaternary and Quinary)')
    plt.ylabel('Average Macro F1')
    plt.xlabel('Batch Size (Log Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'F1_vs_Batch_Size.png', dpi=300)
    plt.close()

    # 2. Token Efficiency / Cost vs Batch Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_df, x='Batch Size', y='Avg Tokens/Req', hue='Model', style='Prompt', markers=True)
    plt.xscale('log', base=2)
    plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
    plt.title('Token Cost Amortization: Avg Tokens/Req vs. Batch Size')
    plt.ylabel('Average Tokens per Requirement')
    plt.xlabel('Batch Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Tokens_vs_Batch_Size.png', dpi=300)
    plt.close()

    # 3. Pareto Frontier: Cost (Tokens) vs Accuracy (F1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=avg_df, x='Avg Tokens/Req', y='Avg Macro F1', hue='Model', style='Prompt', s=100)
    plt.title('Pareto Frontier: Accuracy vs. Cost')
    plt.ylabel('Average Macro F1')
    plt.xlabel('Average Tokens per Requirement')
    # Optional: annotate batch sizes
    for i, row in avg_df.iterrows():
        # Only annotate to avoid crowding, or annotate all if few models
        if row['Prompt'] == 'Enhanced_Zero_Shot':
            plt.annotate(f"B{row['Batch Size']}", (row['Avg Tokens/Req']+5, row['Avg Macro F1']+0.002), fontsize=8, alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Cost_vs_Accuracy_Pareto.png', dpi=300)
    plt.close()

    # 3.5 3D Pareto Frontier: Time vs Cost (Tokens) vs Accuracy (F1)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    models = avg_df['Model'].unique()
    prompts = avg_df['Prompt'].unique()
    
    colors = sns.color_palette("tab10", len(models))
    color_map = dict(zip(models, colors))
    markers = ['o', '^', 's', 'p', '*', 'D', 'v']
    marker_map = dict(zip(prompts, markers[:len(prompts)]))
    
    for _, row in avg_df.iterrows():
        ax.scatter(row['Avg Time (s)'], row['Avg Tokens/Req'], row['Avg Macro F1'],
                   color=color_map[row['Model']], marker=marker_map[row['Prompt']], s=60)
                   
    import matplotlib.lines as mlines
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='o', linestyle='None', markersize=8, label=m) for m in models]
    prompt_handles = [mlines.Line2D([], [], color='gray', marker=marker_map[p], linestyle='None', markersize=8, label=p) for p in prompts]
    
    first_legend = ax.legend(handles=model_handles, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=prompt_handles, title='Prompt', bbox_to_anchor=(1.05, 0.5), loc='upper left')

    ax.set_xlabel('Avg Time (s)')
    ax.set_ylabel('Avg Tokens/Req')
    ax.set_zlabel('Avg Macro F1')
    ax.set_title('3D Pareto Frontier: Time vs Tokens vs Accuracy')
    
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Cost_Time_Accuracy_3D_Pareto.png', dpi=300)
    plt.close()

    # 4. Task-Specific Performance Heatmap (F1 Score)
    PROMPT_SHORT = {
        'Enhanced_Zero_Shot': 'EZS',
        'Few_Shot_with_Reasoning': 'FSR',
    }
    # Find the row that achieved the maximum F1 for each model/task combo
    best_idx = df.groupby(['Task', 'Model'])['Avg Macro F1'].idxmax()
    best_rows = df.loc[best_idx, ['Task', 'Model', 'Avg Macro F1', 'Batch Size', 'Prompt']]
    
    # Create the pivot table for the heatmap values
    best_f1_df = best_rows.pivot(index='Task', columns='Model', values='Avg Macro F1')

    # Build an annotation matrix of the same shape as best_f1_df
    # We iterate through the pivot table's index and columns to ensure order matches
    annot_matrix = []
    for task in best_f1_df.index:
        row_annots = []
        for model in best_f1_df.columns:
            # Find the specific row for this task and model
            match = best_rows[(best_rows['Task'] == task) & (best_rows['Model'] == model)]
            if not match.empty:
                f1 = match.iloc[0]['Avg Macro F1']
                bs = int(match.iloc[0]['Batch Size'])
                pr = PROMPT_SHORT.get(match.iloc[0]['Prompt'], match.iloc[0]['Prompt'])
                row_annots.append(f"{f1:.3f}\nB{bs}·{pr}")
            else:
                row_annots.append("")
        annot_matrix.append(row_annots)

    # Height scales with the number of task rows so the cells are not stretched.
    n_rows = max(1, best_f1_df.shape[0])
    plt.figure(figsize=(13, 1.6 * n_rows + 2.2))
    sns.heatmap(best_f1_df, annot=annot_matrix, fmt="", cmap="YlGnBu",
                annot_kws={"size": 9, "va": "center"}, square=False,
                cbar_kws={"shrink": 0.6})
    plt.title('Best F1-Score by Model vs Task (F1 | Batch · Prompt)', fontsize=13)
    plt.ylabel('Classification Task')
    plt.xlabel('Model')
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Heatmap_Task_vs_Model_F1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Dual-Axis Line Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    sns.lineplot(data=avg_df, x='Batch Size', y='Avg Macro F1', hue='Model', style='Prompt', ax=ax1, legend=False)
    sns.lineplot(data=avg_df, x='Batch Size', y='Avg Tokens/Req', hue='Model', style='Prompt', ax=ax2, legend=False, linestyle='--')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax1.set_ylabel('Average Macro F1 (Solid Lines)')
    ax2.set_ylabel('Average Tokens/Req (Dashed Lines)')
    plt.title('Trade-Off Curve: Macro F1 and Token Cost vs Batch Size')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Dual_Axis_F1_and_Tokens.png', dpi=300)
    plt.close()

    # 6. Aggregate Execution Time per Model (total hours across the target tasks)
    # Sized and fonted so that, when scaled to a single ACM column, the value
    # labels and ticks stay readable.
    total_time_h = (df.groupby('Model')['Avg Time (s)'].sum() / 3600.0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=total_time_h.values, y=total_time_h.index, palette='viridis', ax=ax)
    xmax = max(total_time_h.values)
    for i, v in enumerate(total_time_h.values):
        ax.text(v + xmax * 0.015, i, f"{v:.1f} h", va='center', fontsize=16, fontweight='bold')
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel('Total Execution Time (hours)', fontsize=16)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.set_title('Aggregate Execution Time by Model', fontsize=16)
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Execution_Time_BarChart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # generate task-specific graphs
    print("Generating task-specific graphs...")
    tasks = df['Task'].unique()
    for task in tasks:
        TASK_DIR = OUTPUT_DIR / 'Task_Specific' / task
        TASK_DIR.mkdir(parents=True, exist_ok=True)
        task_df = df[df['Task'] == task]
        task_avg = task_df.groupby(['Model', 'Batch Size', 'Prompt']).mean(numeric_only=True).reset_index()
        
        # 1. Macro F1 vs Batch Size for specific task
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=task_avg, x='Batch Size', y='Avg Macro F1', hue='Model', style='Prompt', markers=True, dashes=True)
        plt.xscale('log', base=2)
        plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
        plt.title(f'[{task}] Macro F1-Score vs. Batch Size')
        plt.ylabel('Average Macro F1')
        plt.xlabel('Batch Size (Log Scale)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_F1_vs_Batch_Size.png', dpi=300)
        plt.close()

        # 2. Token Efficiency / Cost vs Batch Size for specific task
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=task_avg, x='Batch Size', y='Avg Tokens/Req', hue='Model', style='Prompt', markers=True)
        plt.xscale('log', base=2)
        plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
        plt.title(f'[{task}] Token Cost Amortization vs. Batch Size')
        plt.ylabel('Average Tokens per Requirement')
        plt.xlabel('Batch Size')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_Tokens_vs_Batch_Size.png', dpi=300)
        plt.close()

        # 3. Pareto Frontier
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=task_avg, x='Avg Tokens/Req', y='Avg Macro F1', hue='Model', style='Prompt', s=100)
        plt.title(f'[{task}] Pareto Frontier: Accuracy vs. Cost')
        plt.ylabel('Average Macro F1')
        plt.xlabel('Average Tokens per Requirement')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_Cost_vs_Accuracy_Pareto.png', dpi=300)
        plt.close()

        # 3.5 3D Pareto Frontier (Task specific)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        models_task = task_avg['Model'].unique()
        prompts_task = task_avg['Prompt'].unique()
        
        colors_task = sns.color_palette("tab10", len(models_task))
        color_map_task = dict(zip(models_task, colors_task))
        markers_task = ['o', '^', 's', 'p', '*', 'D', 'v']
        marker_map_task = dict(zip(prompts_task, markers_task[:len(prompts_task)]))
        
        for _, row in task_avg.iterrows():
            ax.scatter(row['Avg Time (s)'], row['Avg Tokens/Req'], row['Avg Macro F1'],
                       color=color_map_task[row['Model']], marker=marker_map_task[row['Prompt']], s=60)
                       
        import matplotlib.lines as mlines
        model_handles_task = [mlines.Line2D([], [], color=color_map_task[m], marker='o', linestyle='None', markersize=8, label=m) for m in models_task]
        prompt_handles_task = [mlines.Line2D([], [], color='gray', marker=marker_map_task[p], linestyle='None', markersize=8, label=p) for p in prompts_task]
        
        first_legend_task = ax.legend(handles=model_handles_task, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.add_artist(first_legend_task)
        ax.legend(handles=prompt_handles_task, title='Prompt', bbox_to_anchor=(1.05, 0.5), loc='upper left')

        ax.set_xlabel('Avg Time (s)')
        ax.set_ylabel('Avg Tokens/Req')
        ax.set_zlabel('Avg Macro F1')
        ax.set_title(f'[{task}] 3D Pareto Frontier: Time vs Tokens vs Accuracy')
        
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_Cost_Time_Accuracy_3D_Pareto.png', dpi=300)
        plt.close()

        # 4. Dual-Axis Line Chart (Task specific)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        sns.lineplot(data=task_avg, x='Batch Size', y='Avg Macro F1', hue='Model', style='Prompt', ax=ax1, legend=False)
        sns.lineplot(data=task_avg, x='Batch Size', y='Avg Tokens/Req', hue='Model', style='Prompt', ax=ax2, legend=False, linestyle='--')
        ax1.set_xscale('log', base=2)
        ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax1.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        ax1.set_ylabel('Average Macro F1 (Solid Lines)')
        ax2.set_ylabel('Average Tokens/Req (Dashed Lines)')
        plt.title(f'[{task}] Trade-Off Curve: F1 vs Tokens')
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_Dual_Axis_F1_and_Tokens.png', dpi=300)
        plt.close()

        # 5. Execution Speed Grouped Bar Chart (Task specific)
        plt.figure(figsize=(12, 6))
        task_avg_by_prompt = task_avg.groupby(['Batch Size', 'Model']).mean(numeric_only=True).reset_index()
        sns.barplot(data=task_avg_by_prompt, x='Batch Size', y='Avg Time (s)', hue='Model')
        plt.title(f'[{task}] Execution Speed vs Batch Size')
        plt.ylabel('Average Execution Time (s)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(TASK_DIR / f'{task}_Execution_Time_BarChart.png', dpi=300)
        plt.close()

    print("✔ Graphs successfully saved.")


def print_terminal_analysis(df: pd.DataFrame):
    print(f"\nTotal Data Rows Extracted: {len(df)}")
    print(f"Tasks Discovered: {df['Task'].unique().tolist()}")
    
    print("\n" + "="*80)
    print("1. OPTIMAL BATCH SIZE (Avg F1 across all tasks)")
    print("="*80)
    opt_batch = df.groupby(['Model', 'Batch Size'])['Avg Macro F1'].mean().unstack()
    print(opt_batch.round(4).to_string())
    
    print("\n" + "="*80)
    print("2. HIGHEST REACHED F1 SCORE PER MODEL / TASK")
    print("="*80)
    task_df = df.groupby(['Model', 'Task'])['Avg Macro F1'].max().unstack()
    print(task_df.round(4).to_string())

    print("\n" + "="*80)
    print("3. TOKENS PER REQUIREMENT (COST SCALING)")
    print("="*80)
    tokens_df = df.groupby(['Model', 'Batch Size'])['Avg Tokens/Req'].mean().unstack()
    print(tokens_df.round(1).to_string())


def efficiency_analysis(df: pd.DataFrame):
    """
    Find the batch size per model & task that optimises for speed (Avg Time/s)
    and token efficiency (Avg Tokens/Req).

    Composite Efficiency Score = 0.5 * norm(tokens) + 0.5 * norm(time)
    where norm() scales each metric to [0,1] so lower is cheaper/faster.
    The batch with the LOWEST composite score is the most efficient.
    """
    GLOBAL_DIR = OUTPUT_DIR / 'Global_Averages'
    GLOBAL_DIR.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("5. MOST EFFICIENT BATCH SIZE PER TASK PER MODEL (Token + Time)")
    print("="*80)

    # Average across prompt strategies (we care about the scaling behaviour)
    grp = df.groupby(['Model', 'Task', 'Batch Size'])[
        ['Avg Tokens/Req', 'Avg Time (s)']
    ].mean().reset_index()

    # --- Normalise within each (Model, Task) group so metrics are comparable ---
    def norm_group(g):
        for col in ['Avg Tokens/Req', 'Avg Time (s)']:
            mn, mx = g[col].min(), g[col].max()
            g[col + '_norm'] = (g[col] - mn) / (mx - mn) if mx > mn else 0.0
        g['Efficiency_Score'] = 0.5 * g['Avg Tokens/Req_norm'] + 0.5 * g['Avg Time (s)_norm']
        return g

    grp = grp.groupby(['Model', 'Task'], group_keys=False)[grp.columns.tolist()].apply(norm_group)

    # Best = lowest composite efficiency score
    best_eff = grp.loc[grp.groupby(['Model', 'Task'])['Efficiency_Score'].idxmin()]

    # ── Best batch by Tokens only ──────────────────────────────────────────────
    best_tok = grp.loc[grp.groupby(['Model', 'Task'])['Avg Tokens/Req'].idxmin()]

    # ── Best batch by Time only ────────────────────────────────────────────────
    best_time = grp.loc[grp.groupby(['Model', 'Task'])['Avg Time (s)'].idxmin()]

    # ── Print tables ──────────────────────────────────────────────────────────
    pivot_eff  = best_eff.pivot( index='Model', columns='Task', values='Batch Size').astype('Int64')
    pivot_tok  = best_tok.pivot( index='Model', columns='Task', values='Batch Size').astype('Int64')
    pivot_time = best_time.pivot(index='Model', columns='Task', values='Batch Size').astype('Int64')
    pivot_tok_val  = best_tok.pivot( index='Model', columns='Task', values='Avg Tokens/Req')
    pivot_time_val = best_time.pivot(index='Model', columns='Task', values='Avg Time (s)')

    print("\nBest Batch Size (Combined Token + Time Efficiency):")
    print(pivot_eff.to_string())
    print("\nBest Batch Size for Minimum Tokens/Req:")
    print(pivot_tok.to_string())
    print("\nBest Batch Size for Minimum Time (s):")
    print(pivot_time.to_string())

    # ── Overall efficiency per model ──────────────────────────────────────────
    print("\n" + "-"*60)
    print("Overall most efficient batch per model (avg across tasks):")
    overall_grp = df.groupby(['Model', 'Batch Size'])[
        ['Avg Tokens/Req', 'Avg Time (s)']
    ].mean().reset_index()

    def norm_overall(g):
        for col in ['Avg Tokens/Req', 'Avg Time (s)']:
            mn, mx = g[col].min(), g[col].max()
            g[col + '_norm'] = (g[col] - mn) / (mx - mn) if mx > mn else 0.0
        g['Efficiency_Score'] = 0.5 * g['Avg Tokens/Req_norm'] + 0.5 * g['Avg Time (s)_norm']
        return g

    overall_grp = overall_grp.groupby('Model', group_keys=False)[overall_grp.columns.tolist()].apply(norm_overall)
    best_overall = overall_grp.loc[overall_grp.groupby('Model')['Efficiency_Score'].idxmin()].set_index('Model')
    print(best_overall[['Batch Size', 'Avg Tokens/Req', 'Avg Time (s)', 'Efficiency_Score']].round(2).to_string())

    # ── Heatmap 1: Combined efficiency batch ───────────────────────────────────
    fig_w = max(10, len(pivot_eff.columns) * 1.4)
    fig_h = max(5, len(pivot_eff.index) * 0.8)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot_eff.astype(float), annot=True, fmt='.0f',
                cmap='YlOrRd_r', linewidths=0.5,
                cbar_kws={'label': 'Most Efficient Batch Size'})
    plt.title('Most Efficient Batch Size per Model & Task\n(Combined: Tokens + Time, lower score = better)', fontsize=13)
    plt.xlabel('Task'); plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Efficient_Batch_Combined_Heatmap.png', dpi=300)
    plt.close()

    # ── Heatmap 2: Best Token batch ────────────────────────────────────────────
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot_tok.astype(float), annot=True, fmt='.0f',
                cmap='Blues_r', linewidths=0.5,
                cbar_kws={'label': 'Best Token-Efficient Batch Size'})
    plt.title('Best Batch Size for Minimum Tokens/Req per Model & Task', fontsize=13)
    plt.xlabel('Task'); plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Efficient_Batch_Tokens_Heatmap.png', dpi=300)
    plt.close()

    # ── Heatmap 3: Minimum tokens value ───────────────────────────────────────
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot_tok_val, annot=True, fmt='.0f',
                cmap='Blues_r', linewidths=0.5,
                cbar_kws={'label': 'Min Avg Tokens/Req'})
    plt.title('Minimum Tokens/Req Achieved per Model & Task', fontsize=13)
    plt.xlabel('Task'); plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Min_Tokens_Heatmap.png', dpi=300)
    plt.close()

    # ── Heatmap 4: Minimum time value ─────────────────────────────────────────
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot_time_val, annot=True, fmt='.1f',
                cmap='Oranges_r', linewidths=0.5,
                cbar_kws={'label': 'Min Avg Time (s)'})
    plt.title('Minimum Execution Time Achieved per Model & Task', fontsize=13)
    plt.xlabel('Task'); plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Min_Time_Heatmap.png', dpi=300)
    plt.close()

    # ── Bar: overall most efficient batch per model ────────────────────────────
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette('tab10', len(best_overall))
    ax = best_overall['Batch Size'].plot(kind='bar', color=colors)
    for i, (b, tok, t) in enumerate(zip(
        best_overall['Batch Size'],
        best_overall['Avg Tokens/Req'],
        best_overall['Avg Time (s)']
    )):
        ax.text(i, b + 0.2, f'{tok:.0f}tok\n{t:.1f}s', ha='center', va='bottom', fontsize=8)
    plt.title('Most Efficient Batch Size per Model\n(Minimising Tokens + Time)', fontsize=13)
    plt.xlabel('Model'); plt.ylabel('Best Batch Size')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Efficient_Batch_Bar.png', dpi=300)
    plt.close()

    print("\n✔ Efficiency analysis saved to Analysis_Graphs/Global_Averages/")


def best_batch_analysis(df: pd.DataFrame):
    """For each task and model, find the batch size that yields the highest Macro F1."""
    print("\n" + "="*80)
    print("4. BEST BATCH SIZE PER TASK PER MODEL (by Avg Macro F1)")
    print("="*80)

    # For each model+task combination, find the batch that gives best F1
    idx = df.groupby(['Model', 'Task', 'Batch Size'])['Avg Macro F1'].mean().reset_index()
    best = idx.loc[idx.groupby(['Model', 'Task'])['Avg Macro F1'].idxmax()]
    pivot = best.pivot(index='Model', columns='Task', values='Batch Size').astype('Int64')
    print("\nBest Batch Size (highest Macro F1):")
    print(pivot.to_string())

    # F1 scores at those best batch sizes
    best_f1 = best.pivot(index='Model', columns='Task', values='Avg Macro F1')
    print("\nCorresponding Best F1 Scores:")
    print(best_f1.round(4).to_string())

    # Overall best batch per model (averaged across all tasks)
    print("\n" + "-"*60)
    print("Overall best batch size per model (avg across tasks):")
    overall = df.groupby(['Model', 'Batch Size'])['Avg Macro F1'].mean()
    overall_best = overall.groupby('Model').idxmax().apply(lambda x: x[1])
    overall_best_f1 = overall.groupby('Model').max()
    summary = pd.DataFrame({'Best Batch Size': overall_best, 'Best Avg F1': overall_best_f1.round(4)})
    print(summary.to_string())

    # --- Visualise: heatmap of best batch sizes ---
    GLOBAL_DIR = OUTPUT_DIR / 'Global_Averages'
    GLOBAL_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(max(10, len(pivot.columns) * 1.4), max(5, len(pivot.index) * 0.8)))
    sns.heatmap(
        pivot.astype(float),
        annot=True, fmt='.0f', cmap='YlOrRd_r',
        linewidths=0.5, cbar_kws={'label': 'Optimal Batch Size'}
    )
    plt.title('Optimal Batch Size per Model & Task (by Macro F1)', fontsize=14)
    plt.xlabel('Task')
    plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Best_Batch_Size_Heatmap.png', dpi=300)
    plt.close()

    # --- Visualise: heatmap of the corresponding best F1 scores ---
    plt.figure(figsize=(max(10, len(best_f1.columns) * 1.4), max(5, len(best_f1.index) * 0.8)))
    sns.heatmap(
        best_f1,
        annot=True, fmt='.3f', cmap='YlGnBu',
        linewidths=0.5, vmin=0.5, vmax=1.0,
        cbar_kws={'label': 'Best Macro F1'}
    )
    plt.title('Best Reached F1 per Model & Task (at Optimal Batch Size)', fontsize=14)
    plt.xlabel('Task')
    plt.ylabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Best_F1_at_Optimal_Batch_Heatmap.png', dpi=300)
    plt.close()

    # --- Visualise: Multi-panel subplot – Best Batch & F1 per Task ---
    tasks = sorted(df['Task'].unique())
    models = summary.index.tolist()
    n_tasks = len(tasks)
    n_cols = 2
    n_rows = math.ceil(n_tasks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4.8))
    axes = axes.flatten()

    # Pre-calculate best batch F1 per task/model
    task_best = df.groupby(['Task', 'Model', 'Batch Size'])['Avg Macro F1'].mean().reset_index()
    task_best = task_best.loc[task_best.groupby(['Task', 'Model'])['Avg Macro F1'].idxmax()]

    # Pre-calculate overall avg F1 per task/model (across all batches)
    task_avg_f1 = df.groupby(['Task', 'Model'])['Avg Macro F1'].mean().reset_index()
    task_avg_f1.columns = ['Task', 'Model', 'Avg F1 (all batches)']

    model_colors = dict(zip(models, sns.color_palette('tab10', len(models))))

    for i, task in enumerate(tasks):
        ax1 = axes[i]
        tdf = task_best[task_best['Task'] == task].set_index('Model').reindex(models)
        tavg = task_avg_f1[task_avg_f1['Task'] == task].set_index('Model').reindex(models)

        x = np.arange(len(models))
        bar_colors = [model_colors[m] for m in models]

        # Bars: Best Batch Size (left axis)
        bars = ax1.bar(x, tdf['Batch Size'], color=bar_colors, alpha=0.65, width=0.6)
        ax1.set_ylabel('Best Batch Size', fontsize=10)
        ax1.set_title(f'Task: {task}', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=22, ha='right', fontsize=7.5)

        # Annotate bars with batch number
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        # Right axis: two F1 lines
        ax2 = ax1.twinx()

        best_f1_vals = tdf['Avg Macro F1'].values
        avg_f1_vals  = tavg['Avg F1 (all batches)'].values

        f1_all = np.concatenate([v for v in [best_f1_vals, avg_f1_vals] if not np.all(np.isnan(v))])
        y_min = max(0.5, np.nanmin(f1_all) - 0.05)

        line_best, = ax2.plot(x, best_f1_vals, color='steelblue', marker='D', markersize=6,
                              linewidth=2, label='Best Batch F1')
        line_avg,  = ax2.plot(x, avg_f1_vals,  color='darkorange', marker='o', markersize=6,
                              linewidth=1.8, linestyle='--', label='Avg F1 (all batches)')
        ax2.set_ylabel('Macro F1', fontsize=10)
        ax2.set_ylim(y_min, 1.05)

        # Annotate best-batch F1 values (top)
        for xi, val in enumerate(best_f1_vals):
            if not np.isnan(val):
                ax2.text(xi, val + 0.008, f'{val:.3f}',
                         ha='center', va='bottom', fontsize=7.5, color='steelblue', fontweight='bold')

        # Annotate avg F1 values (below the point to avoid overlap)
        for xi, val in enumerate(avg_f1_vals):
            if not np.isnan(val):
                ax2.text(xi, val - 0.012, f'{val:.3f}',
                         ha='center', va='top', fontsize=7.5, color='darkorange')

        # Per-subplot mini legend (top-right)
        ax2.legend(handles=[line_best, line_avg], fontsize=7.5, loc='lower right',
                   framealpha=0.7)

    # Hide unused axes (last cell if n_tasks is odd)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Best Batch Size, Best-Batch F1 & Overall Avg F1 per Task × Model',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Best_Batch_Size_Bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Best F1 vs Optimal-Cost F1: per model, per task ---
    # For every model × task:
    #   "Best batch"         = batch with highest avg F1 (across prompts) for that task
    #   "Cost-optimal batch" = batch with lowest token+time efficiency score for that model (global)

    # Step 1: cost-optimal batch per model (same logic as efficiency_analysis)
    eff_grp = df.groupby(['Model', 'Batch Size'])[['Avg Tokens/Req', 'Avg Time (s)']].mean().reset_index()

    def _norm(g):
        for col in ['Avg Tokens/Req', 'Avg Time (s)']:
            mn, mx = g[col].min(), g[col].max()
            g[col + '_n'] = (g[col] - mn) / (mx - mn) if mx > mn else 0.0
        g['eff'] = 0.5 * g['Avg Tokens/Req_n'] + 0.5 * g['Avg Time (s)_n']
        return g

    eff_grp = eff_grp.groupby('Model', group_keys=False)[eff_grp.columns.tolist()].apply(_norm)
    cost_opt_batch = (
        eff_grp.loc[eff_grp.groupby('Model')['eff'].idxmin()]
        .set_index('Model')['Batch Size'].astype(int)
    )

    # Step 2: per model × task – find best batch and its F1 vs cost-opt batch F1
    # grp_mean: avg F1 per (model, task, batch) → used to pick best batch fairly
    # grp_max:  max F1 per (model, task, batch) → displayed value (matches raw single-prompt data)
    grp_mean = df.groupby(['Model', 'Task', 'Batch Size'])['Avg Macro F1'].mean().reset_index()
    grp_max  = df.groupby(['Model', 'Task', 'Batch Size'])['Avg Macro F1'].max().reset_index()

    rows = []
    for model in grp_mean['Model'].unique():
        opt_b = cost_opt_batch.get(model)
        for task in grp_mean['Task'].unique():
            sub_mean = grp_mean[(grp_mean['Model'] == model) & (grp_mean['Task'] == task)]
            sub_max  = grp_max[ (grp_max['Model']  == model) & (grp_max['Task']  == task)]
            if sub_mean.empty:
                continue

            # Best batch: chosen by highest mean-across-prompts
            best_b  = int(sub_mean.loc[sub_mean['Avg Macro F1'].idxmax(), 'Batch Size'])
            # F1 value shown: max single-prompt F1 at that batch
            best_f1_arr = sub_max.loc[sub_max['Batch Size'] == best_b, 'Avg Macro F1'].values
            best_f1 = float(best_f1_arr[0]) if len(best_f1_arr) else np.nan

            # Opt batch: max single-prompt F1 at the cost-optimal batch
            opt_f1_arr = sub_max[sub_max['Batch Size'] == opt_b]['Avg Macro F1'].values
            opt_f1 = float(opt_f1_arr[0]) if len(opt_f1_arr) else np.nan

            rows.append({
                'Model':       model,
                'Task':        task,
                'Best Batch':  best_b,
                'Best F1':     round(best_f1, 4),
                'Opt Batch':   opt_b,
                'Opt F1':      round(opt_f1, 4) if not np.isnan(opt_f1) else np.nan,
                'F1 Drop (%)': round((best_f1 - opt_f1) / best_f1 * 100, 2) if not np.isnan(opt_f1) else np.nan,
            })

    comp_df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("BEST BATCH F1 vs COST-OPTIMAL BATCH F1 — per Model × Task")
    print("="*80)
    # Pivot for readability
    for col in ['Best Batch', 'Best F1', 'Opt Batch', 'Opt F1', 'F1 Drop (%)']:
        print(f"\n--- {col} ---")
        piv = comp_df.pivot(index='Model', columns='Task', values=col)
        print(piv.round(4).to_string())

    # --- Plot: one subplot per model, tasks on x-axis, 2 bars ---
    models   = comp_df['Model'].unique()
    tasks    = comp_df['Task'].unique()
    n_models = len(models)
    n_cols   = 2
    n_rows   = math.ceil(n_models / n_cols)

    task_colors = dict(zip(tasks, sns.color_palette('tab10', len(tasks))))
    bar_w = 0.35

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 9, n_rows * 5),
                             sharey=False)
    axes = axes.flatten()

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        mdf = comp_df[comp_df['Model'] == model].sort_values('Task')
        task_list = mdf['Task'].tolist()
        xi = np.arange(len(task_list))

        bars_best = ax.bar(xi - bar_w/2, mdf['Best F1'].values,
                           width=bar_w, label='Best F1',
                           color=[task_colors[t] for t in task_list], alpha=0.85)
        bars_opt  = ax.bar(xi + bar_w/2, mdf['Opt F1'].values,
                           width=bar_w, label='Cost-Optimal F1',
                           color=[task_colors[t] for t in task_list], alpha=0.45, hatch='//')

        # Annotate using named column access (avoid fragile positional _N indexing)
        for bar, (_, row) in zip(bars_best, mdf.iterrows()):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f'B{int(row["Best Batch"])}\n{h:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

        for bar, (_, row) in zip(bars_opt, mdf.iterrows()):
            h = bar.get_height()
            drop = row['F1 Drop (%)']
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f'B{int(row["Opt Batch"])}\n{h:.3f}\n↓{drop:.1f}%',
                    ha='center', va='bottom', fontsize=6.5, color='#444')

        ax.set_xticks(xi)
        ax.set_xticklabels(task_list, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Macro F1', fontsize=10)
        ax.set_title(model, fontsize=11, fontweight='bold')
        f1_floor = mdf['Opt F1'].min()
        ax.set_ylim(max(0.5, f1_floor - 0.08), 1.02)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for ax_idx in range(n_models, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle('Best-Batch F1 vs Cost-Optimal-Batch F1 per Model × Task\n'
                 '(solid = best accuracy batch | hatched = cheapest batch)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    plt.savefig(GLOBAL_DIR / 'Best_vs_Optimal_F1_Bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✔ Best-batch analysis saved to Analysis_Graphs/Global_Averages/")


def generate_radar_charts():
    print("\nGenerating radar charts from detailed logs...")
    for model_name, path_parts in MODEL_DIRS.items():
        model_path = RESULTS_DIR / path_parts[1]
        if not model_path.exists(): continue
        
        for task_dir in [d for d in model_path.iterdir() if d.is_dir()]:
            json_file = task_dir / 'batch_detailed_log.json'
            if not json_file.exists(): continue
            
            try:
                with open(json_file, 'r') as f:
                    logs = json.load(f)
            except:
                continue
                
            # We want to compare all Batch Sizes for Enhanced_Zero_Shot
            b_sizes = [1, 2, 4, 8, 16, 32, 64]
            y_data = {b: {'true': [], 'pred': []} for b in b_sizes}
            
            for batch in logs:
                b_size = batch.get("batch_size")
                prompt = batch.get("prompt")
                if prompt != "Enhanced_Zero_Shot": continue
                if b_size not in b_sizes: continue
                
                true_labels_dict = batch.get("true_labels")
                if not true_labels_dict: continue
                preds_dict = batch.get("predictions", {})
                
                for rid, true_label in true_labels_dict.items():
                    y_data[b_size]['true'].append(true_label)
                    y_data[b_size]['pred'].append(preds_dict.get(rid, ""))
            
            # Use batch 1 (or the first available) to determine classes
            classes_set = set()
            for b_data in y_data.values():
                if b_data['true']:
                    classes_set.update(b_data['true'])
            
            classes = sorted(list(classes_set))
            if len(classes) < 3: continue # Radar chart needs at least 3 axes to look like a radar
            
            def get_f1_scores(truths, preds, classes):
                f1s = []
                for cls in classes:
                    tp = sum(1 for t, p in zip(truths, preds) if t == cls and p == cls)
                    fp = sum(1 for t, p in zip(truths, preds) if t != cls and p == cls)
                    fn = sum(1 for t, p in zip(truths, preds) if t == cls and p != cls)
                    p = tp / (tp + fp) if tp + fp > 0 else 0
                    r = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
                    f1s.append(f1)
                return f1s

            # Calculate F1s
            f1_scores = {}
            has_data = False
            for b in b_sizes:
                if y_data[b]['true']:
                    raw_f1 = get_f1_scores(y_data[b]['true'], y_data[b]['pred'], classes)
                    f1_scores[b] = raw_f1 + raw_f1[:1]
                    has_data = True
            
            if not has_data: continue

            # Plot Radar Chart
            N = len(classes)
            angles = [n / float(N) * 2 * math.pi for n in range(N)]
            angles += angles[:1]
            
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], classes)
            
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
            plt.ylim(0, 1)
            
            # Color map for batches to distinguish
            cmap = plt.get_cmap('tab10')
            for idx, b in enumerate(b_sizes):
                if b in f1_scores:
                    ax.plot(angles, f1_scores[b], linewidth=1.5, linestyle='solid', label=f'Batch {b}', color=cmap(idx))
                    # Fill might be too confusing with 7 overlaps, so we just use lines.
            
            display_name = DISPLAY_NAME_MAP.get(model_name, model_name)
            plt.title(f'[{display_name}] Radar Chart: {task_dir.name} (All Batches)', size=15, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            safe_model_name = display_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            plt.tight_layout()
            radar_dir = OUTPUT_DIR / 'Radar_Charts' / task_dir.name
            radar_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(radar_dir / f'{task_dir.name}_{safe_model_name}_RadarChart_AllBatches.png', dpi=300)
            plt.close()

def main():
    print("Scanning directories for batch experiment results...")
    df = load_data()

    if df is None or df.empty:
        print("❌ No valid experiment results found in the target directories.")
        return

    # Scope the analysis to the two target tasks for the focused report.
    TARGET_TASKS = ['Quaternary_Easy', 'Quinary']
    df = df[df['Task'].isin(TARGET_TASKS)].copy()
    if df.empty:
        print("❌ No data for the target tasks (Quaternary_Easy, Quinary).")
        return
    # Display label: drop the "Easy" suffix for the report.
    df['Task'] = df['Task'].replace({'Quaternary_Easy': 'Quaternary'})

    # Think OFF is the default consumer configuration. The main report figures
    # use these five models; the Think ON variants are analyzed separately in RQ5.
    DEFAULT_MODELS = [
        'DeepSeek-V3.2',
        'Gemma-4-31B (Think OFF)',
        'Llama-4-Maverick',
        'Mixtral-8x22B',
        'Nemotron-Ultra-253B (Think OFF)',
    ]
    df = df[df['Model'].isin(DEFAULT_MODELS)].copy()
    print(f"Scoped analysis to target tasks: {TARGET_TASKS} and default models: {DEFAULT_MODELS}")

    print_terminal_analysis(df)
    best_batch_analysis(df)
    efficiency_analysis(df)
    plot_graphs(df)
    generate_radar_charts()

if __name__ == "__main__":
    main()
