# Hypothesized Failure Mechanisms and Supporting Data

This document is supplementary material for the paper *"How Many Requirements
Should an LLM Classify at Once? A Cost and Accuracy Study of Batch Size for
CrowdRE Sector Classification."*

The paper evaluates the models as **black-box services** and does not claim to
measure their internal behaviour. This file collects (1) the observations that
characterise the large-batch degradation and (2) candidate mechanisms that are
consistent with those observations and with the published literature. The
mechanisms are offered as **hypotheses for future white-box testing, not as
established findings.** All numbers below are reproducible from the released
result CSVs and detailed logs (`results/<model>/<task>/`).

---

## 1. What the experiments establish (observations only)

- Accuracy does not fall monotonically with batch size; it degrades most at the
  largest batch (B = 64), and the degradation is concentrated in **one model**
  (Mixtral-8x22B), not spread evenly across the pool.
- The degradation is **not** caused by exceeding the context window: total
  tokens per call stayed far below every model's limit at every batch size
  (Section 2.3).
- On the five-class (Quinary) task, the accuracy lost to batching is
  redistributed into the catch-all "Other" class (the "Other sink", reported in
  RQ2 of the paper); the four-class (Quaternary) task has no such class and its
  errors stay spread across the specific sectors.

---

## 2. Supporting evidence

### 2.1 Structural failure rate by model

Structural failure rate (%) = failed batches / total batches, averaged over both
tasks and both prompts. A batch "fails" when its response is not valid JSON, has
the wrong item count, or has mismatched identifiers after three retries.

| Model                | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|----------------------|----:|----:|----:|----:|-----:|-----:|-----:|
| Gemma-4-31B          |  0  |  0  |  0  |  0  |   0  |   0  |   0  |
| Mixtral-8x22B        |  0  |  0  |  1  |  1  |   1  | **5**| **5**|
| Nemotron-Ultra-253B  |  0  |  0  |  0  |  0  |   0  |   0  |   0  |
| Llama-4-Maverick     |  0  |  0  |  0  |  0  |   0  |   0  |   0  |
| DeepSeek-V3.2        |  1  |  1  |  1  |  0  |   0  |   0  |   0  |

**Observation.** Mixtral-8x22B, the smallest Mixture-of-Experts (MoE) model, is
the only model with a non-trivial failure rate at large batches. The other four
models, including the much larger MoE model DeepSeek-V3.2, stay at or near 0%.
Parameter count therefore does not predict batch resilience in this pool.

### 2.2 Mixtral failure detail (by task and prompt)

Failure rate (%) for Mixtral-8x22B only:

| Task        | Prompt | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|-------------|--------|----:|----:|----:|----:|-----:|-----:|-----:|
| Quaternary  | EZS    |  0  |  0  |  0  |  0  |   0  |  10  |   0  |
| Quaternary  | FSR    |  0  |  0  |  4  |  5  |   5  |  10  |   0  |
| Quinary     | EZS    |  0  |  0  |  0  |  0  |   0  |   0  |  20  |
| Quinary     | FSR    |  0  |  0  |  1  |  0  |   0  |   0  |   0  |

**Observation.** The single largest spike is 20% (one batch in five) on the
five-class Quinary task under the concise Enhanced Zero-Shot prompt at B = 64.
The reasoning-heavy Few-Shot prompt suppresses this particular spike (0% at
B = 64) but introduces its own smaller failures at mid-range Quaternary batches.

### 2.3 Context overflow is ruled out

The output budget scales with the batch size as

    max_tokens(B) = 4096 + 180 * B

so at B = 64 the budget is 15,616 output tokens. The measured **total** tokens
per call (prompt + output) at B = 64 ranged from about **4,077** (Mixtral) to
about **9,030** (DeepSeek-V3.2) on the Quinary task, far below every model's
context window (64k for Mixtral up to 1M for Llama-4-Maverick). The degradation
therefore cannot be attributed to truncation by context overflow.

---

## 3. Candidate mechanisms (hypotheses)

The following are consistent with the observations above and with the cited
literature. They are **not** measured here and would require white-box access to
test.

### 3.1 Attention dilution

Self-attention scales quadratically with sequence length [Vaswani et al., 2017].
Packing many distinct requirements into one prompt increases the number of
competing interactions, and the softmax normalisation may spread attention too
thinly to resolve individual requirement boundaries, which would push the model
toward broad keyword heuristics. This is consistent with the "lost in the
middle" effect, in which information placed deep inside a long context is used
less reliably [Liu et al., 2023].

### 3.2 Generation-state drift

Producing a long, valid array of items requires the model to maintain its place
across hundreds of output tokens. As the key-value cache fills, serving systems
may apply windowing or quantization that could erode the model's awareness of
the array structure mid-generation. This is consistent with the loop/format
failures observed only at the largest batches.

### 3.3 Routing stress in Mixture-of-Experts models

A sparse MoE model routes each token to a small subset of expert sub-networks
through a learned gating function [Shazeer et al., 2017]. A single prompt that
mixes many unrelated topics may force frequent switching among experts, which
could dilute their specialisation. The disproportionate collapse of the smallest
MoE model, Mixtral [Jiang et al., 2024] (Section 2.1), is **consistent with, but
not proof of**, this mechanism; the larger MoE model DeepSeek-V3.2 does not show
the same fragility, so any such effect is not a simple function of the MoE
architecture alone.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL.
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Jiang et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
