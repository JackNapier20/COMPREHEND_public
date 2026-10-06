"""Compare one Blossom baseline with EvalTree branching factors 2, 4, 6, 8."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path

# Avoid macOS OpenMP/loky mutex failures when sentence-transformers and
# scikit-learn are used in the same long-running process.
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

from ablations.runner import prepare_leaves, run_blossom, run_evaltree
from evaluation.evaluate import evaluate_tree_file
from experiments.reporting import collect_result, write_reports
from main import API_KEYS, MODEL_CONFIG
from tree import Tree
from common_summary_generation.prompt_variants import CURRENT_VARIANT, VALID_VARIANTS


DEFAULT_K = (2, 4, 6, 8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_path(tree_path: Path, output_dir: Path, samples: int, seed: int) -> Path:
    evaluator_path = Path(inspect.getsourcefile(evaluate_tree_file) or "")
    identity = {
        "tree_sha256": _sha256(tree_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "model": MODEL_CONFIG["evaluation"],
        "samples": samples,
        "seed": seed,
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    model = MODEL_CONFIG["evaluation"].replace("/", "_")
    return (output_dir / "eval_outputs" /
            f"{tree_path.stem}_eval_{model}_n{samples}_s{seed}_{signature}.json")


def _is_complete(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        config = Tree.load(path).config
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return config.get("status") == "complete" and all(
        config.get(key) == value for key, value in expected.items()
    )


def _is_compatible(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        config = Tree.load(path).config
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return all(config.get(key) == value for key, value in expected.items())


def _valid_evaluation(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(data.get("levels"))
        and data.get("config", {}).get("posthoc_evaluation_status") == "complete"
    )


def _configured_model(role: str) -> str:
    provider = MODEL_CONFIG[role]
    if provider in {"gemma", "together"}:
        model = API_KEYS[provider].get("model", "")
        if not model or "<" in model:
            raise ValueError(f"Configure the served model for provider {provider!r}")
        return model
    return provider


def _base_config(
    source: Path,
    source_hash: str,
    leaf_count: int,
    main_idea_seed_tree: Path | None,
    source_tree: Tree | None = None,
) -> dict:
    annotation_model = _configured_model("summary_generation")
    if (source_tree is not None and source_tree.levels
            and all(node.capability for node in source_tree.levels[0].nodes)):
        annotation_model = source_tree.config.get("annotation_model", "precomputed")
    config = {
        "experiment": "evaltree_k_sensitivity",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "leaf_count": leaf_count,
        "annotation_model": annotation_model,
        "summary_provider": MODEL_CONFIG["summary_generation"],
        "summary_model": _configured_model("summary_generation"),
        "evaluation_model": _configured_model("evaluation"),
        "blossom_provider": MODEL_CONFIG["blossom"],
        "blossom_model": _configured_model("blossom"),
        "blossom_summary_prompt_variant": os.environ.get(
            "BLOSSOM_SUMMARY_PROMPT_VARIANT", CURRENT_VARIANT
        ),
    }
    if main_idea_seed_tree is not None:
        config["main_idea_seed_tree"] = str(main_idea_seed_tree.resolve())
        config["main_idea_seed_sha256"] = _sha256(main_idea_seed_tree)
    return config


def _write_complete(tree: Tree, path: Path) -> None:
    if MODEL_CONFIG["summary_generation"] == "gemma":
        invalid = [
            node.id for level in tree.levels[1:] for node in level.nodes
            if not (node.summary or node.prompt)
            or (node.summary or node.prompt).strip().lower() == "unknown capability"
            or (node.summary or node.prompt).startswith("Error:")
        ]
        if invalid:
            raise RuntimeError(
                f"Gemma produced missing/error summaries for nodes {invalid[:10]}; "
                "the tree will not be marked complete. Inspect model output before rerunning."
            )
    tree.config["status"] = "complete"
    tree.dump(path)


def seed_level_zero_main_ideas(tree: Tree, seed_path: Path) -> None:
    seed_tree = Tree.load(seed_path)
    if not seed_tree.levels:
        raise ValueError(f"Main-idea seed tree has no levels: {seed_path}")
    seed_nodes = {node.id: node for node in seed_tree.levels[0].nodes}
    for node in tree.levels[0].nodes:
        seed = seed_nodes.get(node.id)
        if seed is None or seed.prompt != node.prompt:
            raise ValueError(
                f"Main-idea seed mismatch for level-0 node {node.id}: {seed_path}"
            )
        if not seed.mainIdea:
            raise ValueError(
                f"Main-idea seed is missing node {node.id}: {seed_path}"
            )
        node.mainIdea = seed.mainIdea


def build_capability_cache(
    source: Path, output_dir: Path, config: dict, overwrite: bool
) -> Path:
    path = output_dir / "evaltree_capability_base.json"
    expected = {**config, "artifact": "capability_cache"}
    if not overwrite and _is_complete(path, expected):
        print(f"Reusing capability cache: {path}")
        return path
    tree = Tree.load(source)
    expected_count = config["leaf_count"]
    if len(tree.levels) != 1 or len(tree.levels[0].nodes) != expected_count:
        raise ValueError(
            f"The experiment source must contain exactly {expected_count} level-0 nodes"
        )
    tree.config = expected
    prepare_leaves(tree, "capability")
    _write_complete(tree, path)
    return path


def build_trees(
    source: Path,
    output_dir: Path,
    k_values: tuple[int, ...],
    config: dict,
    overwrite: bool,
    skip_blossom: bool,
    blossom_only: bool,
    main_idea_seed_tree: Path | None,
    num_eval_samples: int,
) -> list[tuple[str, int | None, Path]]:
    built = []
    if not skip_blossom:
        path = output_dir / "blossom_baseline.json"
        expected = {**config, "method": "blossom", "k": None}
        if not overwrite and _is_complete(path, expected):
            print(f"Reusing completed tree: {path}")
        else:
            if not overwrite and _is_compatible(path, expected):
                print(f"Resuming partial Blossom tree: {path}")
                tree = Tree.load(path)
            else:
                if path.exists() and not overwrite:
                    raise RuntimeError(
                        f"Existing Blossom output is incompatible: {path}; "
                        "rerun with --overwrite"
                    )
                shutil.copyfile(source, path)
                tree = Tree.load(path)
                tree.config = expected
                if main_idea_seed_tree is not None:
                    seed_level_zero_main_ideas(tree, main_idea_seed_tree)
                tree.dump(path)
            run_blossom(tree, path, leaf_type="raw", custom=True)
            _write_complete(tree, path)
        built.append(("blossom", None, path))

    if blossom_only:
        return built

    capability_base = build_capability_cache(source, output_dir, config, overwrite)
    for k in k_values:
        path = output_dir / f"evaltree_k{k}.json"
        expected = {**config, "method": "evaltree", "k": k}
        if overwrite or not _is_complete(path, expected):
            tree = Tree.load(capability_base)
            tree.config = expected
            run_evaltree(
                tree,
                path,
                leaf_type="capability",
                summary_type="evaltree",
                num_eval_samples=num_eval_samples,
                k=k,
                run_internal_evaluation=False,
            )
            _write_complete(tree, path)
        else:
            print(f"Reusing completed tree: {path}")
        built.append(("evaltree", k, path))
    return built


def evaluate_trees(
    trees: list[tuple[str, int | None, Path]],
    output_dir: Path,
    num_samples: int,
    seed: int,
    overwrite: bool,
) -> list[tuple[str, int | None, Path]]:
    eval_dir = output_dir / "eval_outputs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for method, k, tree_path in trees:
        output_path = _evaluation_path(tree_path, output_dir, num_samples, seed)
        if overwrite or not _valid_evaluation(output_path):
            evaluate_tree_file(
                tree_path,
                model_name=MODEL_CONFIG["evaluation"],
                api_keys=API_KEYS,
                num_samples=num_samples,
                seed=seed,
                output_path=output_path,
            )
        else:
            print(f"Reusing shared evaluation: {output_path}")
        evaluated.append((method, k, output_path))
    return evaluated


def _discover_trees(
    output_dir: Path, k_values: tuple[int, ...], skip_blossom: bool,
    blossom_only: bool,
) -> list[tuple[str, int | None, Path]]:
    trees = []
    if not skip_blossom:
        trees.append(("blossom", None, output_dir / "blossom_baseline.json"))
    if not blossom_only:
        trees.extend(("evaltree", k, output_dir / f"evaltree_k{k}.json")
                     for k in k_values)
    missing = [str(path) for _, _, path in trees if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing constructed trees: " + ", ".join(missing))
    return trees


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path("WB_new_random_64.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("k_sensitivity_outputs"))
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--num-eval-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-blossom", action="store_true")
    parser.add_argument("--blossom-only", action="store_true")
    parser.add_argument("--blossom-prompt-variant", choices=VALID_VARIANTS,
                        default=CURRENT_VARIANT)
    parser.add_argument("--main-idea-seed-tree", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if any(k < 2 for k in args.k) or len(set(args.k)) != len(args.k):
        parser.error("--k values must be unique integers of at least 2")
    if args.num_eval_samples < 1:
        parser.error("--num-eval-samples must be at least 1")
    if args.evaluation_only and args.report_only:
        parser.error("--evaluation-only and --report-only are mutually exclusive")
    if args.skip_blossom and args.blossom_only:
        parser.error("--skip-blossom and --blossom-only are mutually exclusive")
    if not args.source.exists() and not args.report_only:
        parser.error(f"source does not exist: {args.source}")
    if args.main_idea_seed_tree is not None and not args.main_idea_seed_tree.exists():
        parser.error(f"main-idea seed tree does not exist: {args.main_idea_seed_tree}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BLOSSOM_SUMMARY_PROMPT_VARIANT"] = args.blossom_prompt_variant
    k_values = tuple(args.k)
    if args.report_only:
        evaluated = [
            (method, k, _evaluation_path(
                path, args.output_dir, args.num_eval_samples, args.seed
            ))
            for method, k, path in _discover_trees(
                args.output_dir, k_values, args.skip_blossom, args.blossom_only
            )
        ]
    else:
        if args.evaluation_only:
            trees = _discover_trees(
                args.output_dir, k_values, args.skip_blossom, args.blossom_only
            )
        else:
            source_hash = _sha256(args.source)
            source_tree = Tree.load(args.source)
            if len(source_tree.levels) != 1 or len(source_tree.levels[0].nodes) < 2:
                parser.error("source must contain one level with at least two nodes")
            leaf_count = len(source_tree.levels[0].nodes)
            config = _base_config(
                args.source, source_hash, leaf_count, args.main_idea_seed_tree,
                source_tree=source_tree,
            )
            trees = build_trees(
                args.source,
                args.output_dir,
                k_values,
                config,
                args.overwrite,
                args.skip_blossom,
                args.blossom_only,
                args.main_idea_seed_tree,
                args.num_eval_samples,
            )
        evaluated = evaluate_trees(
            trees,
            args.output_dir,
            args.num_eval_samples,
            args.seed,
            args.overwrite,
        )

    missing = [str(path) for _, _, path in evaluated if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluated trees: " + ", ".join(missing))
    results = [collect_result(path, method, k) for method, k, path in evaluated]
    outputs = write_reports(results, args.output_dir)
    print("Reports: " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
