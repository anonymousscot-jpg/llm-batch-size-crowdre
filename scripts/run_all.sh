#!/usr/bin/env bash
#
# Replication guide: reproduce the full experiment from the paper.
#
# This script runs every model over both target tasks, sweeping all batch sizes
# and both prompts. Results are written to results/<model>/<task>/.
#
# PREREQUISITES
#   1. pip install -r ../requirements.txt
#   2. Export your API credentials (no keys are stored in this repo):
#        export NVIDIA_API_KEYS="key1,key2"     # NVIDIA-hosted models
#        export OPENAI_BASE_URL="http://localhost:11434/v1"  # Ollama / local endpoint
#        export OPENAI_API_KEY="ollama"          # or your endpoint key
#   3. (Optional) try one tiny run first:
#        python run_experiment.py --model gemma-4-31b --task Quinary --dry-run
#
# USAGE
#   bash run_all.sh              # the two target tasks from the paper
#
set -euo pipefail
cd "$(dirname "$0")"

# The five default (reasoning-off) models plus the two reasoning-on variants.
MODELS=(
  "gemma-4-31b"
  "gemma-4-31b-think"
  "mixtral-8x22b"
  "nemotron-ultra-253b"
  "nemotron-ultra-253b-think"
  "llama-4-maverick"
  "deepseek-v3.2"
)

# The two tasks analyzed in the paper.
TASKS=(Quaternary_Easy Quinary)

echo "Models: ${MODELS[*]}"
echo "Tasks:  ${TASKS[*]}"
echo

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "==============================================================="
    echo ">> Running model=$model  task=$task"
    echo "==============================================================="
    python run_experiment.py --model "$model" --task "$task"
  done
done

echo
echo "All runs complete. Now regenerate the tables and figures:"
echo "  python analysis/analyze.py"
echo "  python analysis/plot_u_shape_and_pareto.py"
echo "  python analysis/pareto_all_batches_4_5.py"
echo "  python analysis/make_dualaxis_fig.py"
