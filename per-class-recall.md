# Per-Class Recall on the Quinary Task

Supporting material for **RQ2** of *Finding the Sweet Spot: Optimizing Batch Size for
Accurate and Cost-Efficient LLM-Based Crowd Requirements Classification*
(Appendix B of the online appendix).

RQ2 reports that larger batches redistribute predictions from the specific sectors into the
ambiguous catch-all `Other` class. This document gives the per-class recall behind that claim.

## Table: per-class recall at B=1 and B=64

Five-model average (Gemma-4-31B, Mixtral-8x22B, Nemotron-Ultra-253B, Llama-4-Maverick,
DeepSeek-V3.2), broken down by prompting strategy.

| B | Prompt | Energy | Entertainment | Health | Safety | Other |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | EZS | 0.74 | 0.69 | 0.67 | 0.89 | 0.56 |
| 1 | FSR | 0.81 | 0.64 | 0.54 | 0.82 | 0.73 |
| 1 | **Avg** | **0.78** | **0.66** | **0.60** | **0.85** | **0.64** |
| 64 | EZS | 0.71 | 0.59 | 0.54 | 0.82 | 0.70 |
| 64 | FSR | 0.73 | 0.57 | 0.52 | 0.84 | 0.71 |
| 64 | **Avg** | **0.72** | **0.58** | **0.53** | **0.83** | **0.71** |

## Change from B=1 to B=64 (average rows)

| Sector | B=1 | B=64 | Δ |
| --- | --- | --- | --- |
| Energy | 0.78 | 0.72 | −0.06 |
| Entertainment | 0.66 | 0.58 | **−0.08** |
| Health | 0.60 | 0.53 | **−0.07** |
| Safety | 0.85 | 0.83 | −0.02 |
| Other (catch-all) | 0.64 | 0.71 | **+0.07** |

Every well-defined class loses recall as the batch grows, with Entertainment and Health falling
the most. The catch-all `Other` class is the only one that gains.

## Supporting counts

From the pooled prediction counts over both prompts
(`results/Quinary_confusion_pooled.csv`):

| Quantity | B=1 | B=64 |
| --- | --- | --- |
| Items predicted `Other` | 940 | 1,090 (+16%) |
| Share of all misclassifications falling into `Other` | 56% | 61% |
| Precision of `Other` | 0.44 | 0.42 |

## Derivation of the RQ2 claims

Source: `results/Quinary_confusion_pooled.csv`, Quinary task, pooled over
5 models × 2 prompts × 320 items = **3,200 predictions per batch size**.

| Quantity | How it is computed | B=1 | B=64 | Change |
| --- | --- | --- | --- | --- |
| Total predictions | 5 models × 2 prompts × 320 | 3,200 | 3,200 | 0 |
| Correct | sum of confusion diagonal | 2,265 | 2,156 | −109 |
| Invalid / unparseable | `Pred (invalid)` column | 2 | 93 | +91 |
| Total misclassifications | total − correct | 935 | 1,044 | +109 |
| Predicted `Other` | `Pred Other` column sum | 940 | 1,090 | +16.0% |
| &nbsp;&nbsp;of which correct | True `Other` → Pred `Other` | 412 | 453 | +41 |
| &nbsp;&nbsp;of which wrong | 940−412, 1090−453 | 528 | 637 | +109 |
| Precision of `Other` | 412/940, 453/1090 | 0.438 | 0.416 | 0.44 → 0.42 |
| Share of misclassifications into `Other` | 528/935, 637/1044 | 56.5% | 61.0% | 56% → 61% |

### Where the extra `Other` predictions come from

| True class | B=1 | B=64 | Change |
| --- | --- | --- | --- |
| Health | 201 | 241 | +40 |
| Entertainment | 162 | 200 | +38 |
| Energy | 92 | 125 | +33 |
| Safety | 73 | 71 | −2 |
| `Other` (correct) | 412 | 453 | +41 |
| **Total** | **940** | **1,090** | **+150** |

### Reconciliation

The three error components account exactly for the change in total misclassifications:

| Component | B=1 | B=64 | Change |
| --- | --- | --- | --- |
| Invalid / unparseable | 2 | 93 | +91 |
| Wrong, predicted `Other` | 528 | 637 | +109 |
| Wrong, predicted some other specific class | 405 | 314 | −91 |
| **Total misclassifications** | **935** | **1,044** | **+109** |

Two distinct effects are visible here. The **+91 invalid predictions** are the structural
failure mode analysed under RQ4 — whole batches lost to unparseable output, which the
zero-tolerance policy scores as entirely incorrect. Separately, errors that previously landed
on a wrong *specific* sector (−91) migrate into the catch-all, while `Other` absorbs +109 more
wrong predictions overall. The `Other` class therefore grows both by attracting new errors and
by displacing them from the specific sectors, without becoming any more precise (0.44 → 0.42).

## Interpretation

Larger batches redistribute predictions from the specific sectors into a low-precision
catch-all. Because Macro F1 is the mean of the per-class scores, these recall losses are
exactly what the aggregate accuracy drop reported in RQ2 reflects — a drop of 2.7 percentage
points on the Quinary task, from 0.721 at the peak to 0.694 at B=64.

The Quaternary task, which by design contains no catch-all class, shows no counterpart:
misclassifications stay spread across the four specific sectors, with no single class taking
more than about a third of them at either batch size, so no sink forms. This is why the two
tasks lose a similar amount of accuracy on average, but only the Quinary task concentrates
that loss in one low-precision class.

## Notes

- Recall is computed per class over the 320-requirement class-balanced Quinary sample
  (64 per class).
- `Avg` rows are the mean of the EZS and FSR rows.
- Under the zero-tolerance validation policy, a batch that fails structural validation after
  three retries has all of its items scored as incorrect, so these recall figures reflect
  production-usable accuracy rather than ideal behaviour.
