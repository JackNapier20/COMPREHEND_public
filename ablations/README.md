# Standalone ablation runner

Run one experiment from the repository root:

```bash
python -m ablations.runner Results/UF_random_prompts_64_base.json \
  --ablation 1 \
  --output-dir ablation_outputs/random \
  --score-file dataset_UF_with_scores.csv \
  --posthoc-eval
```

The source tree is copied before any work begins. Final trees and viewer JSON
files are written beneath `--output-dir`; the optional post-hoc result is
written to its `eval_outputs/` child directory.

| ID | Hierarchy | Leaf representation | Pairing / summary changes |
|---:|---|---|---|
| 1 | Blossom | raw | text + main-idea matching |
| 2 | Blossom | raw | text-only matching |
| 3 | Blossom | main idea | text-only matching over extracted ideas |
| 4 | Blossom | capability | text-only matching over capability phrases |
| 5 | EvalTree | capability | recursive 2-means + EvalTree gerund summaries |
| 6 | Blossom | capability | one summary pass; no reassignment |
| 7 | EvalTree | raw | recursive 2-means + Blossom common summaries |
| 8 | EvalTree | main idea | recursive 2-means + Blossom common summaries |
| 9 | Blossom | raw | text-only, one summary pass, no reassignment |

`--posthoc-eval` is the comparable leaf-based evaluation path: each internal
node samples descendant leaves for entailment metrics and selects a
non-descendant leaf for specificity metrics.

Run the complete resumable WB256 suite and regenerate the comparison report
after every completed variant:

```bash
python -m ablations.suite \
  comparison_256_outputs/WB_current_k2/evaltree_capability_base.json \
  --output-dir ablation_outputs/WB256 \
  --score-file dataset_WB_with_scores.csv
```

Run the corresponding UF256 suite with a separate shared cache and report:

```bash
python -m ablations.suite \
  comparison_256_outputs/UF_current_k2/evaltree_capability_base.json \
  --output-dir ablation_outputs/UF256 \
  --dataset-label UF256 \
  --score-file dataset_UF_with_scores.csv
```

The suite prepares one validated capability/main-idea leaf cache and shares it
across all nine conditions. EvalTree's redundant internal judging is skipped
by default because it does not affect construction; every condition still
uses the same post-hoc evaluator. Pass `--with-evaltree-internal-eval` only
when those unused in-pipeline fields are required.

`--skip-blossom-internal-metrics` skips only optional diagnostic metrics.
A1-A4 still compute the construction entailment scores required for
regeneration and assignment. Assignment retains the original legacy score
(the mean of node-to-summary and outsider-to-summary entailment), with
threshold 0.7; it is not the post-hoc specificity gap. Assignment-enabled
outputs now record `assignment_scoring_version`, so older affected trees
cannot be silently reused. Use a fresh output directory to preserve them.

The targeted WB256/UF256 correction run is resumable:

```bash
python -m experiments.assignment_rerun \
  --project /absolute/path/to/COMPREHEND_public \
  --output /absolute/path/to/COMPREHEND_public/ablation_outputs/assignment_fixed
```

It rebuilds A1-A4 with strict Claude generation and OpenAI evaluation,
retains A5-A9 from the completed MPNet results, and reports scores, ordinary
standard errors, summary word lengths, and verified assignment-edge counts.
