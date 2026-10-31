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
export EVALTREE_SUMMARY_MAX_WORKERS="${EVALTREE_SUMMARY_MAX_WORKERS:-8}"
# Concurrent Gemma requests for main ideas and pair summaries.
export GEMMA_CONCURRENCY="${GEMMA_CONCURRENCY:-8}"

if [[ ! -f myenv/bin/activate ]]; then
    python3 -m venv myenv
fi
source myenv/bin/activate
echo "Installing requirements"
python -m pip install -r requirements.txt
echo "Finished installing requirements"

# main.py reads the repository's .env; verify the existing server and inputs.
python -u -m experiments.local_model_preflight

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
