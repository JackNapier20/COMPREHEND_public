# Agentic weakness-report workflow

This workflow turns aggregate scores on hierarchy nodes into a qualitative
weakness report, uses that report to predict held-out leaf scores, and compares
the predictions with ground truth. All paths, score columns, model labels, and
output files are CLI arguments; no experiment paths are hardcoded.

Generate one report for each hierarchy:

```bash
python -m agentic.workflow report ablation_outputs/Ablation1_tree_final.json \
  --target llama \
  --score-key score_Llama-2-7b-chat-hf \
  --output agentic_outputs/llama_blossom.md

python -m agentic.workflow report ablation_outputs/Ablation5_tree_final.json \
  --target llama \
  --score-key score_Llama-2-7b-chat-hf \
  --output agentic_outputs/llama_evaltree.md
```

Predict every leaf from each report in a single comparable call:

```bash
python -m agentic.workflow predict Results/UF_random_prompts_64_base.json \
  --target llama \
  --report blossom=agentic_outputs/llama_blossom.md \
  --report evaltree=agentic_outputs/llama_evaltree.md \
  --score-min 1 --score-max 10 \
  --output agentic_outputs/llama_predictions.json
```

Set the score range to match the source dataset (for example, 1-5 for an
UltraFeedback score or 1-10 for the WildBench-style experiment).

Evaluate those predictions against attached leaf scores:

```bash
python -m agentic.workflow evaluate \
  agentic_outputs/llama_predictions.json \
  ablation_outputs/Ablation1_tree_final.json \
  --score-key score_Llama-2-7b-chat-hf \
  --output agentic_outputs/llama_metrics.json
```

The report and prediction stages require `OPENAI_API_KEY`; the model defaults
to `OPENAI_MODEL` or `gpt-5.4-mini`. The evaluation stage is entirely local.
