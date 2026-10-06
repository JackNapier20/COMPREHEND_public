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
