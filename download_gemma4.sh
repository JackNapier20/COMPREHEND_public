#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash download_gemma4.sh [MODEL_ID] [DESTINATION]\n'
    printf 'Default model: google/gemma-4-31B-it\n'
    printf 'Override DOWNLOAD_ROOT, DOWNLOAD_VENV, MODEL_REVISION, or PYTHON_BIN as needed.\n'
    exit 0
fi
if [[ $# -gt 2 ]]; then
    printf 'Usage: bash download_gemma4.sh [MODEL_ID] [DESTINATION]\n' >&2
    exit 2
fi

MODEL_ID="${1:-${MODEL_ID:-google/gemma-4-31B-it}}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/gypsum/work1/zamani/sivaraaman/models}"
MODEL_DIR="${2:-${MODEL_DIR:-${DOWNLOAD_ROOT}/${MODEL_ID##*/}}}"
DOWNLOAD_VENV="${DOWNLOAD_VENV:-/gypsum/work1/zamani/sivaraaman/venvs/hf-download}"
MODEL_REVISION="${MODEL_REVISION:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Model: ${MODEL_ID}"
echo "Revision: ${MODEL_REVISION}"
echo "Destination: ${MODEL_DIR}"
echo "Download only; no GPU allocation is required."

# Do not upgrade huggingface_hub inside the existing training/serving environment.
if [[ ! -x "${DOWNLOAD_VENV}/bin/python" ]]; then
    mkdir -p "$(dirname "${DOWNLOAD_VENV}")"
    "${PYTHON_BIN}" -m venv "${DOWNLOAD_VENV}"
fi
"${DOWNLOAD_VENV}/bin/python" -m pip install --upgrade huggingface_hub hf_xet

mkdir -p "${MODEL_DIR}"
df -h "${MODEL_DIR}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
# Keep the Xet cache off the home quota, without relocating saved HF credentials.
export HF_XET_CACHE="${HF_XET_CACHE:-${DOWNLOAD_ROOT}/.xet-cache}"

# HF_TOKEN or a saved Hugging Face login is used automatically when available.
# Preserve .cache/huggingface in MODEL_DIR so interrupted downloads can resume.
"${DOWNLOAD_VENV}/bin/hf" download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"

echo "Download complete: ${MODEL_DIR}"
echo "Includes weights, tokenizer, processor, configuration, and native chat template."
echo "Gemma 4 training/serving must use its own template, not Gemma 3 turn tokens."
