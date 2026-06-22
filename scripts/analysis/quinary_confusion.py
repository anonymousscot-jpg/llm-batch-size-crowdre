"""Pooled Quinary confusion matrix and the RQ2 ``Other''-sink statistics.

For each batch size it builds the confusion matrix (true class x predicted
class) for the Quinary task, summed over the five default models and both
prompts, and writes results/Quinary_confusion_pooled.csv. It then prints the
derived ``Other''-sink numbers reported in RQ2 so the derivation is explicit:
the count of items predicted ``Other'', the precision and recall of ``Other'',
and the share of all misclassifications that are assigned to ``Other''.

The precomputed CSV is shipped in the repository so the RQ2 numbers can be
verified directly. Regenerating it requires the per-requirement detailed logs
(batch_detailed_log.json), which are part of the full archived artifact rather
than this trimmed repository; place them under results/<model>/Quinary/ to
re-run. The ``(invalid)'' predicted column holds items from batches that failed
structural validation and therefore carry no usable label.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / 'results'

# Five default (reasoning-off) model result folders.
MODEL_DIRS = [
    'deepseek-v3.2-cloud-batch',
    'gemma4-31b-cloud-batch',
    'meta-llama-4-maverick-17b-batch',
    'mixtral_8x22b',
    'llama-3.1-nemotron-ultra-253b-v1-batch-off',
]
CLASSES = ['Energy', 'Entertainment', 'Health', 'Other', 'Safety']
BATCHES = [1, 2, 4, 8, 16, 32, 64]
INVALID = '(invalid)'


def build():
    conf = {b: defaultdict(lambda: defaultdict(int)) for b in BATCHES}
    found = False
    for folder in MODEL_DIRS:
        log = RESULTS / folder / 'Quinary' / 'batch_detailed_log.json'
        if not log.exists():
            continue
        found = True
        for rec in json.load(open(log)):
            b = rec.get('batch_size')
            if b not in conf:
                continue
            true = rec.get('true_labels') or {}
            pred = rec.get('predictions') or {}
            for rid, t in true.items():
                p = pred.get(rid)
                if p is None:
                    continue
                if p not in CLASSES:
                    p = INVALID
                conf[b][t][p] += 1
    if not found:
        raise SystemExit(
            'No batch_detailed_log.json found. These per-requirement logs are part '
            'of the full archived artifact; place them under results/<model>/Quinary/ '
            'to regenerate. The precomputed Quinary_confusion_pooled.csv is shipped '
            'for direct verification.')
    return conf


def write_csv(conf):
    cols = CLASSES + [INVALID]
    out = RESULTS / 'Quinary_confusion_pooled.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Batch Size', 'True Class'] + ['Pred ' + c for c in cols])
        for b in BATCHES:
            for t in CLASSES:
                w.writerow([b, t] + [conf[b][t].get(c, 0) for c in cols])
    print('wrote', out)


def report(conf):
    print('Quinary "Other"-sink (pooled over five models and both prompts):')
    for b in (1, 64):
        pred_other = sum(conf[b][t]['Other'] for t in CLASSES)
        corr_other = conf[b]['Other']['Other']
        other_total = sum(conf[b]['Other'].values())
        errors = sum(conf[b][t][p] for t in CLASSES for p in conf[b][t] if p != t)
        err_to_other = sum(conf[b][t]['Other'] for t in CLASSES if t != 'Other')
        print(f'  B={b:<2}  predicted Other={pred_other}  '
              f'precision={corr_other / pred_other:.3f}  '
              f'recall={corr_other / other_total:.3f}  '
              f'errors->Other share={err_to_other / errors:.3f}')


if __name__ == '__main__':
    conf = build()
    write_csv(conf)
    report(conf)
