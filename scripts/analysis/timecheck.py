"""
timecheck.py
============
For each model × task, validates that:
  sum(time_seconds) across all batch-size / prompt / repetition rows
  equals the total wall-clock experiment time recorded in batch_results.csv.

Also cross-checks that `Avg Time (s)` in the aggregated CSV matches the
average of `time_seconds` from batch_results.csv.

Prints a clear summary table and flags any mismatches.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path('/Users/SONY/Documents/College/Study_pdeu/Sem_8/Expriment/Batch')

MODEL_DIRS = {
    'DeepSeek':             ['DeepSeek', 'deepseek-v3.2-cloud-batch'],
    'Gemma (Think OFF)':    ['Gemma',    'gemma4-31b-cloud-batch'],
    'Gemma (Think ON)':     ['Gemma',    'gemma4-31b-cloud-batch-ton'],
    'Mixtral':              ['Mixtral',  'mixtral_8x22b'],
    'Nemotron (Think OFF)': ['Nvidia',   'llama-3.1-nemotron-ultra-253b-v1-batch-off'],
    'Nemotron (Think ON)':  ['Nvidia',   'llama-3.1-nemotron-ultra-253b-v1-batch-final'],
}

SEP = "=" * 100

records = []

for model_name, path_parts in MODEL_DIRS.items():
    model_path = BASE_DIR / path_parts[0] / path_parts[1]
    if not model_path.exists():
        print(f"[MISSING DIR] {model_name}: {model_path}")
        continue

    for task_dir in sorted([d for d in model_path.iterdir() if d.is_dir()]):
        results_csv   = task_dir / 'batch_results.csv'
        aggregated_csv = task_dir / 'batch_summary_aggregated.csv'

        if not results_csv.exists():
            print(f"[NO batch_results.csv] {model_name} / {task_dir.name}")
            continue

        # --- Ground truth from batch_results.csv ---
        res = pd.read_csv(results_csv)
        if 'time_seconds' not in res.columns:
            print(f"[NO time_seconds col] {model_name} / {task_dir.name}")
            continue

        total_time_sum = res['time_seconds'].sum()  # grand total across all runs
        n_runs         = len(res)

        # Per batch-size / prompt expected avg (mean over repetitions, should be 1 here)
        per_combo = (
            res.groupby(['batch_size', 'prompt'])['time_seconds']
            .mean()
            .reset_index()
            .rename(columns={'time_seconds': 'time_from_results'})
        )

        # --- Cross-check against aggregated CSV ---
        if aggregated_csv.exists():
            agg = pd.read_csv(aggregated_csv).rename(columns={
                'Batch Size': 'batch_size',
                'Prompt': 'prompt',
                'Avg Time (s)': 'avg_time_agg'
            })
            merged = per_combo.merge(agg[['batch_size', 'prompt', 'avg_time_agg']],
                                     on=['batch_size', 'prompt'], how='left')
            merged['diff'] = (merged['time_from_results'] - merged['avg_time_agg']).abs()
            max_diff = merged['diff'].max()
            agg_match = '✅ MATCH' if max_diff < 1.0 else f'⚠ MISMATCH (max diff={max_diff:.2f}s)'

            agg_total = agg['avg_time_agg'].sum()
        else:
            agg_match = '—  (no aggregated.csv)'
            agg_total = None
            merged    = per_combo

        records.append({
            'Model':           model_name,
            'Task':            task_dir.name,
            'Runs':            n_runs,
            'Sum Time (s)':    round(total_time_sum, 1),
            'Sum Time (min)':  round(total_time_sum / 60, 2),
            'Agg CSV Match':   agg_match,
        })

        print(f"\n{SEP}")
        print(f"  {model_name}  ·  {task_dir.name}")
        print(SEP)
        print(f"  Total runs in batch_results.csv : {n_runs}")
        print(f"  Sum of time_seconds             : {total_time_sum:.1f}s  "
              f"({total_time_sum/60:.2f} min)")
        if agg_total is not None:
            print(f"  Sum of Avg Time (s) [agg csv]  : {agg_total:.1f}s  "
                  f"({agg_total/60:.2f} min)  — this is across all B×P combos")
        print(f"  Aggregated CSV time match       : {agg_match}")
        print()
        print(merged[['batch_size', 'prompt', 'time_from_results', 'avg_time_agg', 'diff']
                     if 'avg_time_agg' in merged.columns else
                     ['batch_size', 'prompt', 'time_from_results']].to_string(index=False))

# ── Final summary table ──────────────────────────────────────────────────────
print(f"\n\n{'='*100}")
print("SUMMARY: Total Experiment Time per Model × Task")
print('='*100)
summary = pd.DataFrame(records)
if not summary.empty:
    print(summary.to_string(index=False))

    print(f"\n{'='*100}")
    print("Total time per model (sum across all tasks):")
    print('='*100)
    model_total = summary.groupby('Model')['Sum Time (s)'].sum().reset_index()
    model_total['Sum Time (min)'] = (model_total['Sum Time (s)'] / 60).round(2)
    model_total['Sum Time (hr)']  = (model_total['Sum Time (s)'] / 3600).round(3)
    print(model_total.to_string(index=False))
    grand = summary['Sum Time (s)'].sum()
    print(f"\nGrand total across all models & tasks: {grand:.1f}s  "
          f"({grand/60:.1f} min  /  {grand/3600:.2f} hr)")
