"""Run WB256/UF256 COMPREHEND and EvalTree with Together-hosted generators."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("INTERNAL_EVAL_MAX_WORKERS", "6")
os.environ.setdefault("POSTHOC_MAX_WORKERS", "6")
os.environ.setdefault("EVALTREE_SUMMARY_MAX_WORKERS", "8")

from common_summary_generation.prompt_variants import CURRENT_VARIANT
from experiments.evaltree_k_sensitivity import build_trees, evaluate_trees
from experiments.reporting import collect_result, write_reports
from main import API_KEYS, MODEL_CONFIG
from tree import Tree


MODELS = {
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "glm-5.2": "zai-org/GLM-5.2",
}

DATASETS = {
    "wb256": Path(
        "comparison_256_outputs/WB_current_k2/evaltree_capability_base.json"
    ),
    "uf256": Path(
        "comparison_256_outputs/UF_current_k2/evaltree_capability_base.json"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source(path: Path) -> Tree:
    tree = Tree.load(path)
    if len(tree.levels) != 1 or len(tree.levels[0].nodes) != 256:
        raise ValueError(f"Expected one level with 256 leaves: {path}")
    missing_capabilities = [
        node.id for node in tree.levels[0].nodes if not node.capability
    ]
    if missing_capabilities:
        raise ValueError(
            f"Source lacks fixed capability annotations: {missing_capabilities[:10]}"
        )
    seeded_main_ideas = [
        node.id for node in tree.levels[0].nodes if node.mainIdea
    ]
    if seeded_main_ideas:
        raise ValueError(
            "Source already contains main ideas, so Together would not generate "
            f"them: {seeded_main_ideas[:10]}"
        )
    return tree


def _configure_together(model_id: str) -> None:
    if not API_KEYS.get("together", {}).get("api_key"):
        raise ValueError("TOGETHER_API_KEY is missing from .env")
    MODEL_CONFIG.update({
        "blossom": "together",
        "summary_generation": "together",
        "evaluation": "openai",
    })
    API_KEYS["together"]["model"] = model_id
    API_KEYS["together"]["reasoning_effort"] = "low"


def _config(source: Path, source_tree: Tree, model_id: str) -> dict:
    return {
        "experiment": "summary_model_comparison",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "leaf_count": 256,
        "annotation_model": source_tree.config.get("annotation_model", "claude"),
        "summary_provider": "together",
        "summary_model": model_id,
        "evaluation_model": "openai",
        "blossom_provider": "together",
        "blossom_model": model_id,
        "together_reasoning_effort": "low",
        "blossom_summary_prompt_variant": CURRENT_VARIANT,
        "num_eval_samples": 2,
        "seed": 42,
    }


def _audit_trees(trees: list[tuple[str, int | None, Path]], model_id: str) -> None:
    for method, _, path in trees:
        tree = Tree.load(path)
        if tree.config.get("summary_model") != model_id:
            raise RuntimeError(f"Wrong summary model recorded in {path}")
        if method == "blossom" and tree.config.get("blossom_model") != model_id:
            raise RuntimeError(f"Wrong Blossom model recorded in {path}")
        unknown = [
            node.id
            for level in tree.levels[1:]
            for node in level.nodes
            if not (node.summary or node.prompt)
            or "unknown capability" in (node.summary or "").lower()
        ]
        if unknown:
            raise RuntimeError(f"Missing/unknown summaries in {path}: {unknown[:10]}")
        tree.config.update({
            "together_concurrency": API_KEYS["together"].get("concurrency", 1),
            "together_max_tokens": API_KEYS["together"].get(
                "max_tokens", 4096
            ),
            "together_empty_retry_max_tokens": max(
                int(API_KEYS["together"].get("max_tokens", 4096)),
                int(API_KEYS["together"].get(
                    "empty_retry_max_tokens", 16384
                )),
            ),
        })
        tree.dump(path)


def run_condition(
    dataset: str,
    model_slug: str,
    output_root: Path,
    *,
    overwrite: bool,
    num_eval_samples: int,
    seed: int,
) -> None:
    source = DATASETS[dataset]
    source_tree = _validate_source(source)
    model_id = MODELS[model_slug]
    _configure_together(model_id)
    output_dir = output_root / dataset.upper() / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _config(source, source_tree, model_id)
    config["num_eval_samples"] = num_eval_samples
    config["seed"] = seed
    print(
        f"\n=== {dataset.upper()} | {model_id} | "
        "COMPREHEND then EvalTree(k=2) ==="
    )
    trees = build_trees(
        source,
        output_dir,
        (2,),
        config,
        overwrite,
        skip_blossom=False,
        blossom_only=False,
        main_idea_seed_tree=None,
        num_eval_samples=num_eval_samples,
    )
    _audit_trees(trees, model_id)
    evaluated = evaluate_trees(
        trees,
        output_dir,
        num_eval_samples,
        seed,
        overwrite,
    )
    results = [
        collect_result(path, method, k) for method, k, path in evaluated
    ]
    write_reports(
        results,
        output_dir,
        title=f"{dataset.upper()} {model_id} summary-model comparison",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("summary_model_outputs")
    )
    parser.add_argument("--num-eval-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_eval_samples < 1:
        parser.error("--num-eval-samples must be at least 1")

    print("Sequential execution plan:")
    for model_slug in args.models:
        for dataset in args.datasets:
            print(
                f"- {MODELS[model_slug]}: {dataset.upper()} "
                "COMPREHEND -> EvalTree(k=2) -> shared OpenAI evaluation"
            )
    for model_slug in args.models:
        for dataset in args.datasets:
            run_condition(
                dataset,
                model_slug,
                args.output_root,
                overwrite=args.overwrite,
                num_eval_samples=args.num_eval_samples,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
