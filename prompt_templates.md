# Prompt Templates

Full prompt templates used in *Finding the Sweet Spot: Optimizing Batch Size for
Accurate and Cost-Efficient LLM-Based Crowd Requirements Classification*.

Two strategies are compared, each applied at every batch size
(*B* = 1, 2, 4, 8, 16, 32, 64) and on both classification tasks:

| Strategy | Description |
| --- | --- |
| **EZS** — Enhanced Zero-Shot | Concise, instruction-only. States the task, the label set, and a strict output format. No examples. |
| **FSR** — Few-Shot with Reasoning | Same task statement and output contract as EZS, prepended with one worked example per category carrying an explicit reasoning trace, and requesting the reasoning before a closing `FINAL_JSON` array. |

`{req_lines}` marks where the batch of requirements is inserted. Each requirement
is supplied on its own line, prefixed with its identifier (`REQ_001`, `REQ_002`, …).

The **Quinary** task uses the five-label set below. The **Quaternary** task uses
exactly the same templates with the catch-all `Other` label removed from every
label list, leaving `Energy | Entertainment | Health | Safety`.

---

## 1. Enhanced Zero-Shot (EZS) — Quinary

```text
SOFTWARE REQUIREMENT CLASSIFICATION (BATCH)

Classify each of the following requirements into exactly one category:
Energy | Entertainment | Health | Other | Safety

Requirements:
{req_lines}

IMPORTANT: Return ONLY a valid JSON array with one object per requirement.
Each object must have exactly two keys: "id" and "category". The "id" must
match the requirement ID exactly. The "category" must be one of:
Energy | Entertainment | Health | Other | Safety.

Example output format:
[{"id": "REQ_001", "category": "Energy"},
 {"id": "REQ_002", "category": "Safety"}]

Output:
```

---

## 2. Few-Shot with Reasoning (FSR) — Quinary

```text
CLASSIFICATION WITH REASONING (BATCH)

EXAMPLES:
Example 1:
Requirement: "The smart thermostat shall learn user preferences and optimize
heating schedules to minimize energy consumption while maintaining comfort."
Reasoning: The emphasis is clearly on reducing energy usage via automated
scheduling.
Category: Energy

<!-- TODO: paste Examples 2-5 verbatim from the experiment source before
     publishing. One worked example per remaining category, in this order:
     Entertainment, Health, Other, Safety. Each follows the exact
     Requirement / Reasoning / Category structure shown in Example 1. -->

TASK: Classify each requirement below into one
of [Energy, Entertainment, Health, Other, Safety]:

Requirements:
{req_lines}

IMPORTANT: After your reasoning, you MUST end your
response with ONLY a valid JSON array. Each object
must have exactly two keys: "id" and "category".
The "id" must match the requirement ID exactly.
The "category" must be one of: Energy |
Entertainment | Health | Other | Safety.

Format your final answer as:
FINAL_JSON:
[{"id": "REQ_001", "category": "Energy"},
 {"id": "REQ_002", "category": "Safety"}]

Output:
```

---

## Decoding settings

All runs use identical decoding settings across every model, prompt, and batch size:

| Parameter | Value |
| --- | --- |
| temperature | 1.0 (provider default) |
| top-p | 0.95 |
| frequency penalty | none |
| presence penalty | none |
| max output tokens | `4096 + 180 * B` |

## Response validation

Every response must pass a three-part check: valid JSON, exactly *B* items, and
identifiers matching the input exactly. A failing response is retried up to three
times; a batch still failing after that is scored as entirely incorrect.
