#!/usr/bin/env python3
"""
Unified batch-size experiment runner for LLM-based CrowdRE sector classification.

This single script replaces the five per-model runners. It sweeps a chosen model
over batch sizes 1..64 and both prompting strategies (Enhanced Zero-Shot and
Few-Shot with Reasoning), validates every response, scores it, and writes the
per-run metrics, an aggregated summary, and a detailed log.

No API keys are hardcoded. Provide them via environment variables:
  NVIDIA_API_KEYS   comma-separated keys for the NVIDIA-hosted models (round-robin)
  OPENAI_BASE_URL   endpoint for the OpenAI-compatible / Ollama-hosted models
  OPENAI_API_KEY    key (or the literal "ollama") for that endpoint

Examples
  python run_experiment.py --list
  python run_experiment.py --model llama-4-maverick --task Quaternary
  python run_experiment.py --model gemma-4-31b --task Quinary --dry-run
"""
import os
import re
import csv
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch-experiment")

# Repository root and default data directory (this file lives in <repo>/scripts/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Fixed inference settings (identical across all runs).
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 0.3

# =============================================================================
# TASKS — each maps to a dataset file and its label set.
# =============================================================================
TASKS = {
    "Quaternary": ("Quaternary/quaternary_320.csv", ["Energy", "Entertainment", "Health", "Safety"]),
    "Quinary":         ("Quinary_Batch/quinary_batch_320.csv",     ["Energy", "Entertainment", "Health", "Other", "Safety"]),
}

# =============================================================================
# MODELS — one entry per evaluated configuration.
#   provider "nvidia"  : NVIDIA-hosted; round-robin keys; reasoning via system msg.
#   provider "openai"  : OpenAI-compatible / Ollama endpoint; reasoning via extra_body.
# =============================================================================
MODELS = {
    "gemma-4-31b": dict(
        provider="openai", model="gemma4:31b-cloud", think=False,
        system="You are a precise software requirements classifier. "
               "Follow instructions exactly and return valid JSON.",
        extra_body={"num_ctx": 262144, "think": False}),
    "gemma-4-31b-think": dict(
        provider="openai", model="gemma4:31b-cloud", think=True,
        system="You are a precise software requirements classifier. "
               "Follow instructions exactly and return valid JSON.",
        extra_body={"num_ctx": 262144, "think": True}),
    "mixtral-8x22b": dict(
        provider="openai", model="mixtral:8x22b", think=False,
        system="You are a precise software requirements classifier. "
               "Follow instructions exactly and return valid JSON.",
        extra_body=None),
    "deepseek-v3.2": dict(
        provider="openai", model="deepseek-v3.2:cloud", think=False,
        system="You are a precise software requirements classifier. "
               "Follow instructions exactly and return valid JSON.",
        extra_body=None),
    "llama-4-maverick": dict(
        provider="nvidia", model="meta/llama-4-maverick-17b-128e-instruct",
        think=False, system="detailed thinking off", extra_body=None),
    "nemotron-ultra-253b": dict(
        provider="nvidia", model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        think=False, system="detailed thinking off", extra_body=None),
    "nemotron-ultra-253b-think": dict(
        provider="nvidia", model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        think=True, system="detailed thinking on", extra_body=None),
}

# =============================================================================
# FEW-SHOT EXAMPLES — one per category, used by the FSR prompt.
# =============================================================================
FSR_EXAMPLES = {
    "Energy": {
        "requirement": "The smart thermostat shall learn user preferences and optimize heating schedules to minimize energy consumption while maintaining comfort.",
        "reasoning": "The emphasis is clearly on reducing energy usage via automated scheduling."},
    "Entertainment": {
        "requirement": "The platform shall recommend personalized playlists based on the user's listening history and mood preferences.",
        "reasoning": "Core functionality relates to media consumption and entertainment personalization."},
    "Health": {
        "requirement": "The fitness app shall provide real-time heart rate monitoring and alert users when heart rate exceeds safe thresholds during exercise.",
        "reasoning": "Core functionality relates to health monitoring and user safety during workouts."},
    "Other": {
        "requirement": "The system shall allow administrators to configure user roles and access permissions through a centralized dashboard.",
        "reasoning": "This is a general administrative/system management requirement that does not fall into a specific domain category."},
    "Safety": {
        "requirement": "The autonomous vehicle shall detect pedestrians within a 50-meter radius and initiate emergency braking if a collision is imminent.",
        "reasoning": "The requirement directly addresses collision avoidance and pedestrian safety."},
}

# =============================================================================
# PROMPT TEMPLATES (EZS and FSR).
# =============================================================================
def get_batch_prompt_ezs(requirements: List[Tuple[str, str]], categories: List[str]) -> str:
    category_list = " | ".join(categories)
    req_lines = "\n".join([f'[{rid}] "{text}"' for rid, text in requirements])
    return f"""SOFTWARE REQUIREMENT CLASSIFICATION (BATCH)

Classify each of the following requirements into exactly one category:
{category_list}

Requirements:
{req_lines}

IMPORTANT: Return ONLY a valid JSON array with one object per requirement.
Each object must have exactly two keys: "id" and "category".
The "id" must match the requirement ID exactly. The "category" must be one of: {category_list}.

Example output format:
[{{"id": "REQ_001", "category": "{categories[0]}"}}, {{"id": "REQ_002", "category": "{categories[-1]}"}}]

Output:"""


def get_batch_prompt_fsr(requirements: List[Tuple[str, str]], categories: List[str]) -> str:
    category_list = " | ".join(categories)
    category_bracket = f"[{', '.join(categories)}]"
    req_lines = "\n".join([f'[{rid}] "{text}"' for rid, text in requirements])
    examples_text = ""
    for idx, cat in enumerate(categories, 1):
        ex = FSR_EXAMPLES[cat]
        examples_text += (f"Example {idx}:\nRequirement: \"{ex['requirement']}\"\n"
                          f"Reasoning: {ex['reasoning']}\nCategory: {cat}\n\n")
    return f"""CLASSIFICATION WITH REASONING (BATCH)

EXAMPLES:
{examples_text}TASK: Classify each requirement below into one of {category_bracket}:

Requirements:
{req_lines}

IMPORTANT: After your reasoning, you MUST end your response with ONLY a valid JSON array.
Each object must have exactly two keys: "id" and "category".
The "id" must match the requirement ID exactly. The "category" must be one of: {category_list}.

Format your final answer as:
FINAL_JSON:
[{{"id": "REQ_001", "category": "{categories[0]}"}}, {{"id": "REQ_002", "category": "{categories[-1]}"}}]

Output:"""


PROMPT_STRATEGIES = {
    "Enhanced_Zero_Shot": get_batch_prompt_ezs,
    "Few_Shot_with_Reasoning": get_batch_prompt_fsr,
}


# =============================================================================
# UNIFIED CLIENT — handles both the NVIDIA and OpenAI-compatible providers.
# =============================================================================
class Client:
    def __init__(self, model_cfg: Dict):
        self.cfg = model_cfg
        self.provider = model_cfg["provider"]
        self._n = 0
        if self.provider == "nvidia":
            keys = [k.strip() for k in os.getenv("NVIDIA_API_KEYS", "").split(",") if k.strip()]
            if not keys:
                raise RuntimeError("Set NVIDIA_API_KEYS (comma-separated) before running.")
            self.clients = [OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=k) for k in keys]
        else:  # openai-compatible / Ollama
            base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
            key = os.getenv("OPENAI_API_KEY", "ollama")
            self.clients = [OpenAI(base_url=base, api_key=key)]
        logger.info(f"Client ready: provider={self.provider}, model={model_cfg['model']}, "
                    f"keys/endpoints={len(self.clients)}")

    def _next(self) -> OpenAI:
        c = self.clients[self._n % len(self.clients)]
        self._n += 1
        return c

    def classify_batch(self, prompt: str, max_tokens: int) -> Dict:
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = dict(
                    model=self.cfg["model"],
                    messages=[{"role": "system", "content": self.cfg["system"]},
                              {"role": "user", "content": prompt}],
                    temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens,
                    frequency_penalty=0.0, presence_penalty=0.0)
                if self.cfg.get("extra_body"):
                    kwargs["extra_body"] = self.cfg["extra_body"]
                resp = self._next().chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                tokens = resp.usage.total_tokens if resp.usage else 0
                return {"raw_response": raw, "total_tokens": tokens, "success": True, "error": None}
            except Exception as e:
                logger.warning(f"API attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
        return {"raw_response": "", "total_tokens": 0, "success": False,
                "error": f"Failed after {MAX_RETRIES} retries"}


# =============================================================================
# VALIDATION (valid JSON, exact count, exact IDs).
# =============================================================================
def extract_json_from_response(raw: str) -> Optional[List[Dict]]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if not cleaned:
        return None
    final = re.search(r"FINAL_JSON\s*:\s*(\[.*\])", cleaned, re.DOTALL)
    if final:
        try:
            return json.loads(final.group(1))
        except json.JSONDecodeError:
            pass
    found = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == '[':
            depth = 0
            for j in range(i, len(cleaned)):
                if cleaned[j] == '[':
                    depth += 1
                elif cleaned[j] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(cleaned[i:j + 1])
                            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                                found.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
        i += 1
    return found[-1] if found else None


def validate_batch_response(parsed, expected_ids: List[str]) -> Dict:
    res = {"json_valid": False, "count_valid": False, "ids_valid": False,
           "all_valid": False, "failure_reason": None}
    if parsed is None:
        res["failure_reason"] = "JSON parse failure"; return res
    if not isinstance(parsed, list):
        res["failure_reason"] = "Response is not a JSON array"; return res
    for item in parsed:
        if not isinstance(item, dict) or "id" not in item or "category" not in item:
            res["failure_reason"] = "JSON items missing 'id' or 'category' keys"; return res
    res["json_valid"] = True
    if len(parsed) != len(expected_ids):
        res["failure_reason"] = f"Count mismatch: expected {len(expected_ids)}, got {len(parsed)}"; return res
    res["count_valid"] = True
    if {item["id"] for item in parsed} != set(expected_ids):
        res["failure_reason"] = "ID mismatch"; return res
    res["ids_valid"] = True
    res["all_valid"] = True
    return res


# =============================================================================
# METRICS (macro precision / recall / F1, accuracy, tokens/req, failure rate).
# =============================================================================
def compute_metrics(all_true, all_pred, total_tokens, total_items,
                    failed_batches, total_batches, categories,
                    batch_size, prompt_name, repetition) -> Dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(all_true, all_pred):
        if p in categories and p == t:
            tp[t] += 1
        elif p in categories and p != t:
            fp[p] += 1; fn[t] += 1
        else:
            fn[t] += 1
    pr, rc, f1s = [], [], []
    for c in categories:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        pr.append(p); rc.append(r); f1s.append(f)
    return {
        "batch_size": batch_size, "prompt": prompt_name, "repetition": repetition,
        "macro_precision": round(sum(pr) / len(pr), 4) if pr else 0,
        "macro_recall": round(sum(rc) / len(rc), 4) if rc else 0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0,
        "tokens_per_requirement": round(total_tokens / total_items, 2) if total_items else 0,
        "failure_rate": round(failed_batches / total_batches, 4) if total_batches else 0,
        "failed_batches": failed_batches, "total_batches": total_batches,
        "total_tokens": total_tokens, "total_items": total_items,
        "accuracy": round(sum(tp.values()) / total_items, 4) if total_items else 0,
    }


# =============================================================================
# RUNNER
# =============================================================================
def run_combination(df, categories, client, batch_size, prompt_name, repetition):
    prompt_fn = PROMPT_STRATEGIES[prompt_name]
    max_tokens = 4096 + batch_size * 180  # dynamic budget scales with batch size
    shuffled = df.sample(frac=1, random_state=repetition * 42).reset_index(drop=True)
    num_batches = len(shuffled) // batch_size

    all_true, all_pred, batch_records = [], [], []
    total_tokens = total_items = total_correct = failed_batches = 0

    for b in range(num_batches):
        s = b * batch_size
        chunk = shuffled.iloc[s:s + batch_size]
        req_ids = [f"REQ_{s + j + 1:03d}" for j in range(batch_size)]
        requirements = list(zip(req_ids, chunk["requirements"].tolist()))
        true_labels = dict(zip(req_ids, chunk["class"].tolist()))

        api = client.classify_batch(prompt_fn(requirements, categories), max_tokens)
        total_tokens += api["total_tokens"]

        parsed, validation, predictions = None, {"all_valid": False, "failure_reason": "API call failed"}, {}
        if api["success"]:
            parsed = extract_json_from_response(api["raw_response"])
            validation = validate_batch_response(parsed, req_ids)

        if not validation["all_valid"]:
            failed_batches += 1
            for rid in req_ids:
                all_true.append(true_labels[rid]); all_pred.append("__FAILED__")
                predictions[rid] = "__FAILED__"; total_items += 1
        else:
            pred_map = {it["id"]: it["category"] for it in parsed}
            for rid in req_ids:
                pred = pred_map.get(rid, "__MISSING__").strip().capitalize()
                if pred not in categories:
                    pred = "__INVALID__"
                all_true.append(true_labels[rid]); all_pred.append(pred)
                predictions[rid] = pred
                total_correct += (pred == true_labels[rid]); total_items += 1

        batch_records.append({
            "batch_idx": b, "batch_size": batch_size, "prompt": prompt_name,
            "repetition": repetition, "req_ids": req_ids, "true_labels": true_labels,
            "predictions": predictions, "validation": validation,
            "batch_failed": not validation["all_valid"],
            "raw_response": api["raw_response"][:2000],
            "tokens": api["total_tokens"], "api_success": api["success"]})
        time.sleep(DELAY_BETWEEN_REQUESTS)

    metrics = compute_metrics(all_true, all_pred, total_tokens, total_items,
                              failed_batches, num_batches, categories,
                              batch_size, prompt_name, repetition)
    logger.info(f"  B={batch_size} {prompt_name}: F1={metrics['macro_f1']:.4f} "
                f"tokens/req={metrics['tokens_per_requirement']:.1f} "
                f"fail={metrics['failure_rate']:.2%}")
    return metrics, batch_records


def aggregate_and_write(run_metrics, detailed_log, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # per-run metrics
    cols = list(run_metrics[0].keys())
    with open(out_dir / "batch_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["time_seconds"])
        w.writeheader()
        for m in run_metrics:
            w.writerow(m)
    # detailed log
    with open(out_dir / "batch_detailed_log.json", "w") as f:
        json.dump(detailed_log, f, indent=2, default=str)
    # aggregated summary (mean over repetitions, per batch size and prompt)
    agg = defaultdict(list)
    for m in run_metrics:
        agg[(m["batch_size"], m["prompt"])].append(m)
    rows = []
    for (bs, pr), runs in sorted(agg.items()):
        n = len(runs)
        mean = lambda k: round(sum(r[k] for r in runs) / n, 4)
        rows.append({
            "Batch Size": bs, "Prompt": pr,
            "Avg Macro F1": mean("macro_f1"), "Std Macro F1": "",
            "Avg Precision": mean("macro_precision"), "Avg Recall": mean("macro_recall"),
            "Avg Tokens/Req": mean("tokens_per_requirement"),
            "Avg Failure Rate": mean("failure_rate"), "Avg Accuracy": mean("accuracy"),
            "Avg Time (s)": round(sum(r.get("time_seconds", 0) for r in runs) / n, 2)})
    with open(out_dir / "batch_summary_aggregated.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote results to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Unified LLM batch-size experiment runner.")
    ap.add_argument("--model", choices=list(MODELS), help="model configuration to run")
    ap.add_argument("--task", choices=list(TASKS), help="classification task to run")
    ap.add_argument("--prompts", nargs="+", choices=list(PROMPT_STRATEGIES),
                    default=list(PROMPT_STRATEGIES), help="prompt strategies (default: both)")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
                    help="batch sizes to sweep (default: 1 2 4 8 16 32 64)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per configuration")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="datasets directory")
    ap.add_argument("--out", default=None, help="output directory (default: results/<model>/<task>)")
    ap.add_argument("--dry-run", action="store_true", help="run only batch size 1, one rep")
    ap.add_argument("--list", action="store_true", help="list available models and tasks, then exit")
    args = ap.parse_args()

    if args.list or not (args.model and args.task):
        print("Models:", ", ".join(MODELS))
        print("Tasks: ", ", ".join(TASKS))
        if not args.list:
            print("\nProvide --model and --task to run. See --help.")
        return

    import pandas as pd
    csv_name, categories = TASKS[args.task]
    df = pd.read_csv(Path(args.data_dir) / csv_name)
    df.dropna(subset=["requirements", "class"], inplace=True)
    df["class"] = df["class"].str.strip().str.capitalize()

    client = Client(MODELS[args.model])
    batch_sizes = [1] if args.dry_run else args.batch_sizes
    reps = 1 if args.dry_run else args.reps
    out_dir = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / args.model / args.task

    run_metrics, detailed_log = [], []
    for bs in batch_sizes:
        for prompt in args.prompts:
            for rep in range(1, reps + 1):
                t0 = time.time()
                metrics, records = run_combination(df, categories, client, bs, prompt, rep)
                metrics["time_seconds"] = round(time.time() - t0, 2)
                run_metrics.append(metrics)
                detailed_log.extend(records)

    aggregate_and_write(run_metrics, detailed_log, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
