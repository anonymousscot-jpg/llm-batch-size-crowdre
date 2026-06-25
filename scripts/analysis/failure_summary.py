"""Structural-failure summary for the RQ4 / Section 5 analysis.

Reads the per-run records in results/<model>/<task>/batch_results.csv for the
five default models and both target tasks, and reports the structural failure
counts and rates two ways:

  * the per-batch rate, and its unweighted mean across the seven batch sizes
    (this over-weights the sparse large batches); and
  * the pooled rate (total failed batches / total batches), i.e. the fraction
    of calls that actually fail.

These reproduce the RQ4 numbers (pooled 0.22% for EZS and 0.31% for FSR) and the
Section 5 distribution (28 of the 34 structural failures occur at a batch of four
or smaller, where there are far more batches). The unweighted mean (0.53% / 0.44%)
is printed for transparency, since the paper deliberately reports the pooled rate
instead.
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'
MODEL_DIRS = [
    'deepseek-v3.2-cloud-batch',
    'gemma4-31b-cloud-batch',
    'meta-llama-4-maverick-17b-batch',
    'mixtral_8x22b',
    'llama-3.1-nemotron-ultra-253b-v1-batch-off',
]
TASKS = ['Quaternary', 'Quinary']
PROMPTS = {'Enhanced_Zero_Shot': 'EZS', 'Few_Shot_with_Reasoning': 'FSR'}


def load():
    frames = [pd.read_csv(RESULTS / d / t / 'batch_results.csv')
              for d in MODEL_DIRS for t in TASKS
              if (RESULTS / d / t / 'batch_results.csv').exists()]
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    total_failed = int(df['failed_batches'].sum())
    total_batches = int(df['total_batches'].sum())
    print(f'Structural failures: {total_failed} of {total_batches} batches\n')

    print('By batch size (all prompts):')
    g = df.groupby('batch_size').agg(failed=('failed_batches', 'sum'),
                                     total=('total_batches', 'sum'))
    g['rate_%'] = (g.failed / g.total * 100).round(3)
    print(g.to_string())
    small = int(g.loc[[1, 2, 4], 'failed'].sum())
    print(f'  failures at batch <= 4: {small} of {total_failed}\n')

    print('By prompt:')
    for key, lab in PROMPTS.items():
        sub = df[df.prompt == key]
        gb = sub.groupby('batch_size').agg(f=('failed_batches', 'sum'),
                                           t=('total_batches', 'sum'))
        mean_rate = (gb.f / gb.t * 100).mean()
        pooled = sub.failed_batches.sum() / sub.total_batches.sum() * 100
        print(f'  {lab}: {int(sub.failed_batches.sum())} failed of '
              f'{int(sub.total_batches.sum())} batches  '
              f'-> pooled {pooled:.2f}%  (unweighted mean {mean_rate:.2f}%)')


if __name__ == '__main__':
    main()
