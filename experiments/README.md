# EvalTree k-sensitivity experiment

This experiment addresses the reviewer concern that EvalTree was evaluated
only with `k=2`. Here, `k` is the maximum number of child clusters produced by
each recursive K-means split; it is not the number of leaves in a final cluster.

The default run constructs one fixed Blossom baseline and EvalTree variants at
`k = 2, 4, 6, 8` from the same leakage-checked 64-node WildBench sample. It
then applies the same post-hoc leaf-sampled evaluator to all five trees.

```bash
python -m experiments.evaltree_k_sensitivity \
  WB_new_random_64.json \
  --output-dir k_sensitivity_outputs \
  --k 2 4 6 8 \
  --num-eval-samples 2 \
  --seed 42
```

The runner caches the 64 EvalTree capability annotations once and reuses them
for every `k`. Completed compatible trees and evaluation files are reused by
default. Use `--overwrite` to regenerate everything, `--evaluation-only` to
evaluate already-built trees, or `--report-only` to rebuild reports from
existing evaluated files.

The standalone runner constrains BLAS, OpenMP, and scikit-learn to one worker
to avoid macOS mutex failures when transformer embeddings and K-means run in
the same process. Claude generations retry empty responses three times; a
persistent refusal falls back to `gpt-4o-mini`, with every Blossom fallback
recorded in the tree's `config.provider_fallbacks` field.

Outputs include each constructed tree, each shared-evaluator JSON,
`comparison.json`, a long-form `comparison.csv`, and `REPORT.md`. In the
specificity columns, lower is better: they measure how well a parent summary
is supported by a deliberately selected outsider leaf.

# HPC Gemma comparison on WB256 and UF256

`script_gemma_256.sh` runs COMPREHEND and EvalTree (`k=2`) on WB256,
then UF256, using Gemma for main-task extraction and parent summaries and
OpenAI for construction decisions and post-hoc evaluation. It uses the same
precomputed leaf capabilities as the DeepSeek/GLM runs, two descendant samples,
seed 42, and the current summary prompt. Gemma must already be serving an
OpenAI-compatible endpoint reachable from the batch job.

Set these values in the repository's `.env` (unquoted, as read by `main.py`):

```dotenv
GEMMA_API_KEY=DUMMY
GEMMA_BASE_URL=http://YOUR_MODEL_SERVER:8000/v1
GEMMA_MODEL=YOUR_EXACT_SERVED_MODEL_NAME
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

Use `localhost` only when the server runs on the same node as the pipeline.
The script sets `BLOSSOM_PROVIDER=gemma`, `SUMMARY_GENERATION_PROVIDER=gemma`,
and `EVALUATION_PROVIDER=openai`; no manual edits to `main.py` are required.
It checks the served model and both 256-leaf inputs before construction.
The preflight prints the effective endpoint and client node, and retries
connection failures for up to ten minutes to allow vLLM to finish loading.
Set `GEMMA_WAIT_SECONDS` to change this limit. A model-name mismatch or
authentication error stops immediately. An already-exported `GEMMA_BASE_URL`
or `GEMMA_MODEL` takes precedence over the value in `.env`.

For a standalone endpoint test with one real model response:

```bash
python -u debug_gemma_endpoint.py --base-url http://gpu014:5050/v1
```

This prints the client hostname, lists the served model IDs, then sends one
short chat prompt to the sole served model. It only requires the `openai`
package and does not read `.env`; the endpoint comes from `--base-url` and
the optional authentication key from the `GEMMA_API_KEY` environment variable
(default `DUMMY`). Use `--model` if the server lists multiple models, or
`--base-url http://127.0.0.1:5050/v1` when testing on the serving node itself.

For the pipeline's input and server readiness checks:

```bash
python -u -m experiments.local_model_preflight
```

Submit from the repository root so Slurm can open its log files:

```bash
mkdir -p slurm_output_j
sbatch script_gemma_256.sh
```

Outputs go to `hpc_gemma_outputs/gemma256/WB256` and `UF256`. Each contains
the constructed trees, evaluated JSONs, `comparison.json`, `comparison.csv`
(including standard errors), and `REPORT.md`. These generated outputs are
ignored by Git. To give a different checkpoint its own output directory:

```bash
GEMMA_RUN_TAG=gemma3-27b-it sbatch script_gemma_256.sh
```

The script retains the supplied four-hour limit. Resubmitting with the same
tag reuses completed artifacts and resumes partial COMPREHEND/post-hoc
evaluation checkpoints; an interrupted EvalTree construction is rebuilt.
The exact served model name is recorded in the tree configuration and affects
cache compatibility. Keep OpenAI network access available throughout the run.

# EvalTree faithfulness-specificity gap diagnostics

The diagnostic runner explains the paired
`faithfulness1 - specificity_faithfulness` margin. It investigates
aggregation, outsider policy, embedding representation, recursive
summarization, judge calibration, descendant sampling, and
capability-annotation stability without modifying the completed WB256 or UF256
trees.

Generate the offline audit and a preflight report without provider calls:

```bash
python -m experiments.specificity_diagnostics --report-only
```

Run the complete resumable experiment with its default 1,100-call ceiling:

```bash
python -m experiments.specificity_diagnostics \
  --phases all \
  --max-api-calls 1100
```

Phases can be run separately with `--phases candidates summaries judge
sampling annotations`. Every provider request is cached by its content in the
output directory. `--report-only` rebuilds `diagnostics.json`,
`diagnostics.csv`, `REPORT.md`, and `blind_review.csv` from the available cache;
`--overwrite` deliberately repeats the selected phase while leaving source
trees untouched.
