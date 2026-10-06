#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=8192
#SBATCH -p gpu-preempt
#SBATCH -t 4:00:00
#SBATCH -o ./slurm_output_j/slurm-%j.out
#SBATCH -e ./slurm_output_j/slurm-%j-error.out
#SBATCH --qos=short

set -euo pipefail

# Submit from the repository root, or set COMPREHEND_REPO_DIR explicitly.
cd "${COMPREHEND_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "Start time: $(date)"

export CUDA_LAUNCH_BLOCKING=1
export BLOSSOM_PROVIDER=gemma
export SUMMARY_GENERATION_PROVIDER=gemma
export EVALUATION_PROVIDER=openai
export INTERNAL_EVAL_MAX_WORKERS="${INTERNAL_EVAL_MAX_WORKERS:-6}"
export POSTHOC_MAX_WORKERS="${POSTHOC_MAX_WORKERS:-6}"
export EVALTREE_SUMMARY_MAX_WORKERS="${EVALTREE_SUMMARY_MAX_WORKERS:-1}"

if [[ ! -f myenv/bin/activate ]]; then
    python3 -m venv myenv
fi
source myenv/bin/activate
echo "Installing requirements"
python -m pip install -r requirements.txt
echo "Finished installing requirements"

# main.py reads the repository's .env; verify the existing server and inputs.
python - <<'PY'
from pathlib import Path

from openai import OpenAI
from main import API_KEYS
from tree import Tree

gemma = API_KEYS["gemma"]
if "<" in gemma["base_url"] or "<" in gemma["model"]:
    raise SystemExit("Set GEMMA_BASE_URL and GEMMA_MODEL in .env first.")
key = API_KEYS["openai"]
if not key or key.endswith("..."):
    raise SystemExit("Set OPENAI_API_KEY in .env for GPT evaluation.")
client = OpenAI(
    api_key=gemma["api_key"], base_url=gemma["base_url"], timeout=30,
)
model_ids = {model.id for model in client.models.list().data}
if gemma["model"] not in model_ids:
    raise SystemExit(f"GEMMA_MODEL is not served; available model IDs: {sorted(model_ids)}")
for dataset in ("WB", "UF"):
    source = Path(f"comparison_256_outputs/{dataset}_current_k2/evaltree_capability_base.json")
    tree = Tree.load(source)
    if len(tree.levels) != 1 or len(tree.levels[0].nodes) != 256:
        raise SystemExit(f"Expected exactly 256 source leaves: {source}")
    leaves = tree.levels[0].nodes
    if any(not node.capability or node.mainIdea for node in leaves):
        raise SystemExit(f"Expected fixed capabilities and fresh main ideas: {source}")
print(f"Serving model: {gemma['model']}")
print("Inputs verified: WB256 and UF256; Gemma generation, OpenAI evaluation.")
PY

# Change GEMMA_RUN_TAG for a different checkpoint; retain it to resume this run.
OUTPUT_ROOT="hpc_gemma_outputs/${GEMMA_RUN_TAG:-gemma256}"
for DATASET in WB UF; do
    echo "Starting ${DATASET}256: COMPREHEND -> EvalTree(k=2) -> GPT evaluation"
    python -u -m experiments.evaltree_k_sensitivity \
        "comparison_256_outputs/${DATASET}_current_k2/evaltree_capability_base.json" \
        --output-dir "${OUTPUT_ROOT}/${DATASET}256" \
        --k 2 \
        --num-eval-samples 2 \
        --seed 42 \
        --blossom-prompt-variant current
done

echo "Results: ${OUTPUT_ROOT}/{WB256,UF256}/comparison.json and REPORT.md"
echo "End time: $(date)"
