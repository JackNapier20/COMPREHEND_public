# EvalTree k-sensitivity experiment

On branch `codex/evaltree-mpnet`, EvalTree construction uses
`all-mpnet-base-v2` instead of `all-MiniLM-L6-v2`. Capability annotations,
K-means settings, summary prompts/models and evaluation embeddings are unchanged.
Constructed trees record `config.clustering_embedding_model`; comparison and
ablation runners reject previous MiniLM or unsigned tree caches. Use a new output
directory for the MPNet comparison to preserve the earlier baseline. Shared leaf
annotation caches and Blossom outputs remain compatible. No experiments are
launched merely by checking out this branch.

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

## Fresh WB256 Claude teacher tree for fine-tuning

```bash
python -m dataset_prep.sample_random_dataset \
  --source dataset_WB_with_scores.csv --dataset-name "WildBench finetune v5" \
  --count 256 --seed 42 --output finetune_v5_outputs/WB_train_random_256.json \
  --exclude WB_new_random_256.json UF_new_random_256.json \
    comparison_256_outputs/WB_current_k2/evaltree_capability_base.json
python -u -m experiments.finetune_tree finetune_v5_outputs/WB_train_random_256.json
```

The sampler excludes exact/normalized evaluation prompts and dataset source IDs,
as well as current summary demonstrations. Its signed audit is checked again
before any API calls. The teacher runner forces Claude main ideas and summaries,
OpenAI construction/post-hoc judges, and no GPT summary fallback. Defaults for
reassignment, regeneration, two descendant samples and seed 42 remain unchanged.
Strict Claude summaries and main-idea batches each use four workers, while GPT
judging uses six; environment variables can lower these limits.

Outputs are separate from evaluation artifacts under `finetune_v5_outputs`.
The tree retains every node's `mainIdea`; `main_ideas.jsonl` exports all extracted
ideas and `leaf_main_ideas.jsonl` contains the 256 raw leaf prompt/idea targets
with their source identities. `sampling_audit.json` and `MANIFEST.json` record
disjointness and protected-file hashes. This stage does not overwrite any
existing `finetuneDatasetV5.jsonl` or choose a student chat-template format.
Rerunning resumes compatible partial/completed trees and evaluations;
`--export-only` requires existing complete artifacts and makes no API calls.

## Downloading Gemma 4 for new training runs

`bash download_gemma4.sh` downloads the official `google/gemma-4-31B-it`
repository to `/gypsum/work1/zamani/sivaraaman/models/gemma-4-31B-it`.
It uses an isolated downloader environment, requires no GPU, and can be rerun
to resume interrupted downloads. Run on a cluster node where large downloads
and outbound internet are permitted. Use `HF_TOKEN` or a saved HF login if
authentication is needed; credentials are not printed or stored in the script.

An optional model ID and destination can select another variant:

```bash
bash download_gemma4.sh google/gemma-4-26B-A4B-it \
  /gypsum/work1/zamani/sivaraaman/models/gemma-4-26B-A4B-it
```

`MODEL_REVISION` can pin a Hub commit for reproducibility. `DOWNLOAD_ROOT`,
`DOWNLOAD_VENV`, and `PYTHON_BIN` override local paths and the Python executable.
The downloaded native template must be used for both training and inference.
Gemma 4 uses different turn tokens and supports native system turns; this
script does not migrate the Gemma 3-specific pipeline settings or serving stack.

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
Gemma generation uses an explicit `GEMMA_TIMEOUT_SECONDS` deadline (default
600 seconds), output cap `GEMMA_MAX_TOKENS` (1024), and two same-model attempts.
Gemma requests explicitly use temperature 0, preserve system instructions in
the first user turn, and send `ignore_eos=false` and the `<end_of_turn>` stop.
Main-task extraction defaults to `GEMMA_MAIN_IDEA_BATCH_SIZE=1`, which calls
individual plain-text extraction without requiring a JSON array. Set a size
of 2 or more only after confirming the model follows the batched JSON format.
It uses
`GEMMA_MAIN_IDEA_MAX_TOKENS=1024` for batches, with at most 256 output tokens
for an individual fallback. Completed main-task batches are checkpointed
immediately and reused after interruption. These settings can be changed in
`.env`; GPT evaluation still uses its existing configuration.
`GEMMA_CONCURRENCY` controls parallel main-task and COMPREHEND summary
requests (default 8). The Slurm script exports this setting; override it with
`GEMMA_CONCURRENCY=2 sbatch script_gemma_256.sh` when needed. Its EvalTree
summary worker count is also 8, controlled by `EVALTREE_SUMMARY_MAX_WORKERS`.
Main-task results are assigned by their original leaf IDs and checkpointed
as batches finish. On a failed batch, new submissions stop and successful
in-flight batches are saved before the failure is raised.
Gemma responses ending at the output-token limit are rejected, and trees
with missing/error summaries are not marked complete. The endpoint debug
script prints the actual response and finish reason to check model behavior
before a full run, particularly if server logs contain detokenization warnings.

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

Then exercise the actual main-idea, EvalTree summary, and COMPREHEND current
few-shot summary paths on two toy inputs:

```bash
GEMMA_MAIN_IDEA_BATCH_SIZE=1 python -u -m experiments.gemma_smoke_test \
  --base-url http://YOUR_MODEL_SERVER:5050/v1 \
  --model YOUR_EXACT_SERVED_MODEL_NAME
```

This uses only Gemma and writes no artifacts. With batch size 1 and no retries,
there are four generation requests. Inspect every printed result for coherence;
a successful endpoint response does not prove the longer pipeline prompts work.
`--main-idea-batch-size 2` can separately test JSON batching. Keep the same
`GEMMA_MAIN_IDEA_BATCH_SIZE` in the full run as in the successful smoke test.
Use `--checks evaltree-summary` or `--checks comprehend-summary` to test either
summary generator independently when extraction is failing.

If a short capability prompt works but extraction repeats text or hits its
token cap, compare the actual request with the available V4 training format:

```bash
python -u -m experiments.gemma_prompt_diagnostics \
  --output hpc_gemma_outputs/gemma_prompt_diagnostics.json
```

This makes exactly seven generation calls, with one attempt per condition and
128 output tokens per call. It repeats the known-working capability prompt,
tests the current extraction prompt without its editor prefix, crosses current
versus V4 instruction wording with current versus V4 task/instruction ordering,
and tests a V4-style pair summary. Decoding and roles remain fixed. Failures
and truncated responses are retained, not retried or accepted as pipeline output.
Full prompts, hashes, responses, stop reasons, and timings are saved to the new
JSON file; existing files are refused. This does not change production prompts
or invoke the OpenAI judge. The V4 reference comes from the available training
dataset; confirm checkpoint provenance before attributing failures to training.

The generation policy is recorded in comparison cache identities; older
server-default-sampling artifacts are not compatible. After a request-policy
change, use a new `GEMMA_RUN_TAG` to retain failed/partial prior runs untouched.

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
