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
