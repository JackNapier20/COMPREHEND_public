"""Run the nine hierarchy ablations from a public-repo base tree.

The runner is deliberately separate from ``main.py``: every experiment starts
from the supplied base JSON and writes only beneath ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from contextlib import contextmanager
from pathlib import Path

from common_summary_generation.summarize import generate_summary, get_entail_2_score
from evaluation.evaluate import evaluate_tree_file
from evaluation.simple_eval import (
    evaluate_common_summary,
    promote_to_next_level,
    propagate_attrs_up,
    reassign,
    record_stat,
)
from evaltree.annotate import annotate_tree
from evaltree.clustering import RecursiveKMeans, build_hierarchy
from evaltree.evaluate import run_evaluation_on_tree
from evaltree.summarize import summarize_tree
from main import API_KEYS, MODEL_CONFIG
from retrieval.blossom_matching import extract_main_ideas, get_nearest_neighbors_blossom
from retrieval.third_neighbor import find_third
from score_retrieval import add_scores, build_scores
from tree import Level, Node, Tree
from visualisation.visualise_helper import generate_dag_json_from_tree


ABLATIONS = {
    1: dict(method="blossom", leaf_type="raw", custom=True),
    2: dict(method="blossom", leaf_type="raw", custom=False),
    3: dict(method="blossom", leaf_type="main_idea", custom=False),
    4: dict(method="blossom", leaf_type="capability", custom=False),
    5: dict(method="evaltree", leaf_type="capability", summary_type="evaltree"),
    6: dict(
        method="blossom",
        leaf_type="capability",
        custom=False,
        max_iterations=1,
        threshold=-1.0,
        reassign=False,
    ),
    7: dict(method="evaltree", leaf_type="raw", summary_type="common"),
    8: dict(method="evaltree", leaf_type="main_idea", summary_type="common"),
    9: dict(
        method="blossom",
        leaf_type="raw",
        custom=False,
        max_iterations=1,
        threshold=-1.0,
        reassign=False,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_leaves(tree: Tree, leaf_type: str) -> None:
    leaves = tree.levels[0].nodes
    if leaf_type == "raw":
        for node in leaves:
            node.capability = node.prompt
    elif leaf_type == "main_idea":
        ideas = extract_main_ideas(
            leaves,
            llm_model_name=MODEL_CONFIG["blossom"],
            api_keys=API_KEYS,
        )
        for node, idea in zip(leaves, ideas):
            node.capability = idea
    elif leaf_type == "capability":
        annotate_tree(
            tree,
            model_name=MODEL_CONFIG["summary_generation"],
            api_keys=API_KEYS,
        )
    else:
        raise ValueError(f"Unknown leaf type: {leaf_type}")


@contextmanager
def leaf_representation(tree: Tree, level_idx: int, leaf_type: str):
    """Temporarily expose the selected leaf representation as ``prompt``."""
    if level_idx != 0 or leaf_type == "raw":
        yield
        return

    leaves = tree.levels[0].nodes
    original_prompts = [node.prompt for node in leaves]
    for node in leaves:
        node.prompt = node.mainIdea if leaf_type == "main_idea" else node.capability
    try:
        yield
    finally:
        for node, prompt in zip(leaves, original_prompts):
            node.prompt = prompt


def attach_external_scores(tree: Tree, score_file: Path | None) -> None:
    if score_file is None:
        return
    if not score_file.exists():
        raise FileNotFoundError(f"Score file not found: {score_file}")
    add_scores(tree, str(score_file))
    build_scores(tree)


def run_blossom(
    tree: Tree,
    output_path: Path,
    *,
    leaf_type: str,
    custom: bool,
    max_iterations: int = 2,
    threshold: float = 0.4,
    reassign_nodes: bool = True,
    score_file: Path | None = None,
    collect_internal_metrics: bool = True,
) -> Path:
    prepare_leaves(tree, leaf_type)
    level_idx = len(tree.levels) - 1
    while len(tree.levels[level_idx].nodes) > 1:
        print(f"\n--- Blossom level {level_idx} ---")
        progress = tree.config.get("blossom_progress")
        if not progress or progress.get("level") != level_idx:
            progress = {"level": level_idx, "phase": "start", "iteration": 0}
            tree.config["blossom_progress"] = progress
            tree.dump(output_path)

        phase = progress.get("phase", "start")
        if phase == "start":
            with leaf_representation(tree, level_idx, leaf_type):
                get_nearest_neighbors_blossom(
                    tree,
                    level_idx,
                    use_custom_score=custom,
                    llm_model_name=MODEL_CONFIG["blossom"],
                    api_keys=API_KEYS,
                    checkpoint_path=output_path,
                )

            # Specificity compares original prompts and excludes the match.
            find_third(tree, level_idx)
            progress.update(phase="matched", iteration=0)
            tree.dump(output_path)

        if progress["phase"] in {"matched", "entailment"}:
            iteration = int(progress.get("iteration", 0))
            needs_regeneration = [True] if iteration == 0 else [
                score <= threshold
                for score in get_entail_2_score(tree, level_idx)
            ]
            while iteration < max_iterations and any(needs_regeneration):
                with leaf_representation(tree, level_idx, leaf_type):
                    generate_summary(
                        tree,
                        level_idx,
                        threshold,
                        model_name=MODEL_CONFIG["summary_generation"],
                        api_keys=API_KEYS,
                        checkpoint_path=output_path,
                    )
                decisions_need_scores = (
                    collect_internal_metrics
                    or max_iterations > 1
                    or threshold >= 0
                )
                if decisions_need_scores:
                    evaluate_common_summary(
                        tree,
                        level_idx,
                        model_name=MODEL_CONFIG["evaluation"],
                        api_keys=API_KEYS,
                        checkpoint_path=output_path,
                        eval_faith_info=collect_internal_metrics,
                    )
                    needs_regeneration = [
                        score <= threshold
                        for score in get_entail_2_score(tree, level_idx)
                    ]
                else:
                    needs_regeneration = [False]
                iteration += 1
                progress.update(phase="entailment", iteration=iteration)
                tree.dump(output_path)
            progress["phase"] = "specificity"
            tree.dump(output_path)

        if progress["phase"] == "specificity":
            if collect_internal_metrics:
                evaluate_common_summary(
                    tree,
                    level_idx,
                    eval_specificity=True,
                    model_name=MODEL_CONFIG["evaluation"],
                    api_keys=API_KEYS,
                    checkpoint_path=output_path,
                )
            progress["phase"] = "promote"
            tree.dump(output_path)

        record_stat(tree, level_idx)
        promote_to_next_level(tree, level_idx)
        if reassign_nodes:
            reassign(
                tree,
                level_idx,
                0.7,
                use_specificity_for_unmatched=True,
                model_name=MODEL_CONFIG["evaluation"],
                api_keys=API_KEYS,
            )
        tree.config.pop("blossom_progress", None)
        tree.dump(output_path)
        level_idx += 1

    propagate_attrs_up(tree, attrs=["score", "evaluation", "entail_1", "entail_2"])
    attach_external_scores(tree, score_file)
    tree.dump(output_path)
    return output_path


def _common_pair_summary(texts: list[str]) -> str:
    """Use the public Blossom prompt to reduce one cluster to one summary."""
    if not texts:
        return "Unknown summary"
    current = texts[0]
    for other in texts[1:]:
        pair = [
            Node(id=0, prompt=current, match=1, score=1.0, alive=True),
            Node(id=1, prompt=other, match=0, score=1.0, alive=False),
        ]
        scratch = Tree(levels=[Level(level=0, nodes=pair)], next_id=2)
        generate_summary(
            scratch,
            0,
            -1.0,
            model_name=MODEL_CONFIG["summary_generation"],
            api_keys=API_KEYS,
        )
        current = pair[0].summary or current
    return current


def summarize_evaltree_common(tree: Tree) -> None:
    node_map = {node.id: node for level in tree.levels for node in level.nodes}
    for level in tree.levels[1:]:
        for node in level.nodes:
            child_texts = []
            for child_id in node.children:
                child = node_map[child_id]
                text = child.summary or child.capability or child.prompt
                if text:
                    child_texts.append(text)
            node.summary = _common_pair_summary(child_texts)
            node.prompt = node.summary


def run_evaltree(
    tree: Tree,
    output_path: Path,
    *,
    leaf_type: str,
    summary_type: str,
    num_eval_samples: int,
    k: int = 2,
    run_internal_evaluation: bool = True,
    score_file: Path | None = None,
) -> Path:
    summary_provider = MODEL_CONFIG["summary_generation"]
    if summary_provider == "claude":
        fallback_policy = "claude-sonnet-4-6 retries=3, then gpt-4o-mini"
    elif summary_provider == "together":
        fallback_policy = (
            f"{API_KEYS['together'].get('model', 'Together model')} "
            "retries=3; no cross-provider fallback"
        )
    else:
        fallback_policy = f"{summary_provider}; provider-default retry policy"
    tree.config["provider_fallback_policy"] = fallback_policy
    leaves = tree.levels[0].nodes
    for leaf in leaves:
        leaf.parent = []
        leaf.children = []
    tree.levels = [Level(level=0, nodes=leaves)]
    tree.next_id = max(node.id for node in leaves) + 1
    prepare_leaves(tree, leaf_type)

    clustering = RecursiveKMeans(max_children=k, max_depth=50)
    embeddings = clustering.embed([node.capability or node.prompt for node in leaves])
    result = clustering.cluster_recursive([node.id for node in leaves], embeddings)
    build_hierarchy(tree, result)

    if summary_type == "evaltree":
        summarize_tree(
            tree,
            model_name=MODEL_CONFIG["summary_generation"],
            api_keys=API_KEYS,
        )
    elif summary_type == "common":
        summarize_evaltree_common(tree)
    else:
        raise ValueError(f"Unknown summary type: {summary_type}")

    if run_internal_evaluation:
        for idx, level in enumerate(tree.levels):
            if len(level.nodes) > 1:
                find_third(tree, idx)
        run_evaluation_on_tree(
            tree,
            num_eval_samples=num_eval_samples,
            model_name=MODEL_CONFIG["evaluation"],
            api_keys=API_KEYS,
        )
    attach_external_scores(tree, score_file)
    tree.dump(output_path)
    return output_path


def run_ablation(
    base_json: Path,
    ablation: int,
    output_dir: Path,
    *,
    num_eval_samples: int = 2,
    score_file: Path | None = None,
    posthoc_eval: bool = False,
    seed: int = 42,
    overwrite: bool = False,
    run_internal_evaluation: bool = True,
    collect_internal_metrics: bool = True,
) -> Path:
    if ablation not in ABLATIONS:
        raise ValueError(f"Ablation must be one of {sorted(ABLATIONS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Ablation{ablation}_{base_json.stem}_final.json"
    config = ABLATIONS[ablation].copy()
    identity = {
        "experiment": "ablation_suite",
        "ablation": ablation,
        "source": str(base_json.resolve()),
        "source_sha256": _sha256(base_json),
        **config,
    }
    if output_path.exists() and not overwrite:
        tree = Tree.load(output_path)
        mismatches = {
            key: (tree.config.get(key), value)
            for key, value in identity.items()
            if tree.config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing output is incompatible: {output_path}; "
                f"mismatches={mismatches}. Use --overwrite to replace it."
            )
        print(f"Resuming ablation {ablation} from {output_path}")
    else:
        shutil.copyfile(base_json, output_path)
        tree = Tree.load(output_path)
        tree.config = {**identity, "construction_status": "partial"}
        tree.dump(output_path)

    runtime_options = {
        "collect_internal_metrics": collect_internal_metrics,
        "evaltree_internal_evaluation": run_internal_evaluation,
    }
    option_mismatches = {
        key: (tree.config.get(key), value)
        for key, value in runtime_options.items()
        if tree.config.get(key) not in (None, value)
    }
    if option_mismatches:
        raise RuntimeError(
            f"Existing output used different runtime options: {option_mismatches}"
        )
    tree.config.update(runtime_options)
    tree.dump(output_path)

    if tree.config.get("construction_status") != "complete":
        if config["method"] == "blossom":
            run_blossom(
                tree,
                output_path,
                leaf_type=config["leaf_type"],
                custom=config["custom"],
                max_iterations=config.get("max_iterations", 2),
                threshold=config.get("threshold", 0.4),
                reassign_nodes=config.get("reassign", True),
                score_file=score_file,
                collect_internal_metrics=collect_internal_metrics,
            )
        else:
            run_evaltree(
                tree,
                output_path,
                leaf_type=config["leaf_type"],
                summary_type=config["summary_type"],
                num_eval_samples=num_eval_samples,
                k=2,
                run_internal_evaluation=run_internal_evaluation,
                score_file=score_file,
            )
        tree.config["construction_status"] = "complete"
        tree.dump(output_path)
    else:
        print(f"Reusing completed ablation tree: {output_path}")

    generate_dag_json_from_tree(
        output_path, output_dir / f"Ablation{ablation}_{base_json.stem}_vis.json"
    )
    if posthoc_eval:
        evaluate_tree_file(
            output_path,
            model_name=MODEL_CONFIG["evaluation"],
            api_keys=API_KEYS,
            num_samples=num_eval_samples,
            seed=seed,
        )
    print(f"Ablation {ablation} saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("base_json", type=Path)
    parser.add_argument("--ablation", type=int, choices=sorted(ABLATIONS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_outputs"))
    parser.add_argument("--score-file", type=Path)
    parser.add_argument("--num-eval-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-internal-eval",
        action="store_true",
        help="Skip EvalTree's redundant in-pipeline evaluation; post-hoc evaluation is unaffected",
    )
    parser.add_argument(
        "--skip-blossom-internal-metrics",
        action="store_true",
        help="Keep only construction-critical Blossom judgments before post-hoc evaluation",
    )
    parser.add_argument(
        "--posthoc-eval",
        action="store_true",
        help="Also run leaf-sampled faithfulness/informativeness/specificity evaluation",
    )
    args = parser.parse_args()
    run_ablation(
        args.base_json,
        args.ablation,
        args.output_dir,
        num_eval_samples=args.num_eval_samples,
        score_file=args.score_file,
        posthoc_eval=args.posthoc_eval,
        seed=args.seed,
        overwrite=args.overwrite,
        run_internal_evaluation=not args.skip_internal_eval,
        collect_internal_metrics=not args.skip_blossom_internal_metrics,
    )


if __name__ == "__main__":
    main()
