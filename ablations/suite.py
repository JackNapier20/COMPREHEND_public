"""Run and report the complete nine-condition ablation suite."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
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
os.environ.setdefault("POSTHOC_MAX_WORKERS", "4")

from ablations.runner import ABLATIONS, prepare_leaves, run_ablation
from retrieval.blossom_matching import extract_main_ideas
from tree import Tree


LABELS = {
    1: "Blossom raw + main-idea matching",
    2: "Blossom raw, text-only matching",
    3: "Blossom main-idea representation",
    4: "Blossom capability representation",
    5: "EvalTree capability baseline",
    6: "Blossom capability, one pass/no reassignment",
    7: "EvalTree raw + common summaries",
    8: "EvalTree main ideas + common summaries",
    9: "Blossom raw, one pass/no reassignment",
}

METRICS = (
    "faithfulness1",
    "faithfulness2",
    "informativeness1",
    "informativeness2",
    "specificity_faithfulness",
    "specificity_informativeness",
    "faithfulness_gap",
    "informativeness_gap",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0, "mean": None, "std": None, "stderr": None,
            "min": None, "max": None,
        }
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _suspicious_main_idea(value: str) -> bool:
    stripped = (value or "").strip()
    return (
        not stripped
        or stripped.startswith("\x60\x60\x60")
        or stripped in {"[", "]", "{", "}"}
        or stripped.lower() == "unknown main idea"
    )


def prepare_shared_base(
    source: Path,
    output_dir: Path,
    overwrite: bool,
    dataset_label: str = "WB256",
) -> Path:
    """Create one capability/main-idea leaf cache shared by every variant."""
    path = output_dir / f"{dataset_label}_shared_leaf_base.json"
    source_hash = _sha256(source)
    if path.exists() and not overwrite:
        tree = Tree.load(path)
        if (
            tree.config.get("artifact") == "ablation_shared_leaf_base"
            and tree.config.get("source_sha256") == source_hash
        ):
            if tree.config.get("status") == "complete":
                return path
            print(f"Resuming shared leaf cache: {path}")
        else:
            raise RuntimeError(
                f"Incompatible shared base at {path}; use --overwrite to replace it."
            )
    else:
        tree = Tree.load(source)
        if len(tree.levels) != 1 or len(tree.levels[0].nodes) != 256:
            raise ValueError("The 256-node ablation suite requires 256 level-0 leaves")
        tree.config = {
            "experiment": f"{dataset_label}_ablation_suite",
            "dataset_label": dataset_label,
            "artifact": "ablation_shared_leaf_base",
            "source": str(source.resolve()),
            "source_sha256": source_hash,
            "status": "partial",
        }
        tree.dump(path)

    prepare_leaves(tree, "capability")
    tree.dump(path)
    leaves = tree.levels[0].nodes
    api_keys = __import__("main").API_KEYS
    for start in range(0, len(leaves), 8):
        chunk = leaves[start:start + 8]
        pending = [
            node for node in chunk
            if not node.mainIdea or _suspicious_main_idea(node.mainIdea)
        ]
        if not pending:
            continue
        for node in pending:
            node.mainIdea = None
        extract_main_ideas(
            pending,
            llm_model_name="claude",
            api_keys=api_keys,
        )
        for node in pending:
            for _ in range(2):
                if not _suspicious_main_idea(node.mainIdea):
                    break
                node.mainIdea = None
                extract_main_ideas(
                    [node],
                    llm_model_name="claude",
                    api_keys=api_keys,
                )
        tree.dump(path)
    suspicious = [
        node.id for node in leaves
        if _suspicious_main_idea(node.mainIdea)
    ]
    if suspicious:
        raise RuntimeError(f"Invalid main ideas for leaf IDs: {suspicious}")
    tree.config["status"] = "complete"
    tree.dump(path)
    return path


def evaluated_path(tree_path: Path) -> Path:
    return tree_path.parent / "eval_outputs" / f"{tree_path.stem}_eval.json"


def collect_result(ablation: int, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    internal = [
        node
        for level in data.get("levels", [])
        if level.get("level", 0) > 0
        for node in level.get("nodes", [])
    ]
    values = {metric: [] for metric in METRICS}
    for node in internal:
        for metric in METRICS[:6]:
            if node.get(metric) is not None:
                values[metric].append(float(node[metric]))
        if (
            node.get("faithfulness1") is not None
            and node.get("specificity_faithfulness") is not None
        ):
            values["faithfulness_gap"].append(
                float(node["faithfulness1"])
                - float(node["specificity_faithfulness"])
            )
        if (
            node.get("informativeness1") is not None
            and node.get("specificity_informativeness") is not None
        ):
            values["informativeness_gap"].append(
                float(node["informativeness1"])
                - float(node["specificity_informativeness"])
            )
    levels = data.get("levels", [])
    return {
        "ablation": ablation,
        "label": LABELS[ablation],
        "tree_method": ABLATIONS[ablation]["method"],
        "evaluated_file": str(path.resolve()),
        "parents": len(internal),
        "levels": len(levels),
        "max_level": max(
            (level.get("level", 0) for level in levels), default=0
        ),
        "metrics": {metric: _summary(values[metric]) for metric in METRICS},
    }


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def write_reports(
    results: list[dict], output_dir: Path, dataset_label: str = "WB256"
) -> None:
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    report_path = output_dir / "REPORT.md"
    json_path.write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "ablation", "label", "tree_method", "parents", "levels", "metric",
            "n", "mean", "std", "stderr", "min", "max",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for metric, stats in result["metrics"].items():
                writer.writerow({
                    "ablation": result["ablation"],
                    "label": result["label"],
                    "tree_method": result["tree_method"],
                    "parents": result["parents"],
                    "levels": result["levels"],
                    "metric": metric,
                    **stats,
                })

    lines = [
        f"# {dataset_label} Full Ablation Results",
        "",
        f"All variants use the same 256 {dataset_label} source leaves, seed 42, "
        "two sampled "
        "descendant leaves per parent, and the shared post-hoc evaluator. Values "
        "are pooled across internal parents and shown as mean (standard error).",
        "",
        "| Ablation | Variant | Parents | Faith avg | Faith min | Info avg | "
        "Info min | Spec faith | Spec info | Faith gap | Info gap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result["metrics"]
        cells = []
        for name in METRICS:
            stats = metric[name]
            cells.append(
                f"{_fmt(stats['mean'])} ({_fmt(stats['stderr'])})"
            )
        lines.append(
            f"| {result['ablation']} | {result['label']} | "
            f"{result['parents']} | " + " | ".join(cells) + " |"
        )

    if len(results) == len(ABLATIONS):
        lines.extend(["", "## Metric Leaders", ""])
        directions = {
            "faithfulness1": max,
            "faithfulness2": max,
            "informativeness1": max,
            "informativeness2": max,
            "specificity_faithfulness": min,
            "specificity_informativeness": min,
            "faithfulness_gap": max,
            "informativeness_gap": max,
        }
        for metric, chooser in directions.items():
            eligible = [
                result for result in results
                if result["metrics"][metric]["mean"] is not None
            ]
            winner = chooser(
                eligible, key=lambda result: result["metrics"][metric]["mean"]
            )
            lines.append(
                f"- **{metric}:** A{winner['ablation']} "
                f"({_fmt(winner['metrics'][metric]['mean'])})."
            )
    lines.extend([
        "",
        "Higher is better for faithfulness, informativeness, and the paired "
        "gaps. Lower is better for the two outsider-specificity columns. "
        "These pooled parent means are descriptive because variants can have "
        "different numbers and depth distributions of internal nodes.",
        "",
        "Machine-readable statistics are in comparison.json; long-form "
        "mean, standard deviation, standard error, range, and N are in "
        "comparison.csv.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_results(output_dir: Path, base_stem: str) -> list[dict]:
    results = []
    for ablation in sorted(ABLATIONS):
        tree_path = output_dir / f"Ablation{ablation}_{base_stem}_final.json"
        eval_path = evaluated_path(tree_path)
        if not eval_path.exists():
            continue
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        if data.get("config", {}).get("posthoc_evaluation_status") == "complete":
            results.append(collect_result(ablation, eval_path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ablation_outputs/WB256")
    )
    parser.add_argument(
        "--dataset-label",
        help="Label used in shared-cache filenames and report headings "
        "(defaults to the output directory name)",
    )
    parser.add_argument(
        "--score-file", type=Path, default=Path("dataset_WB_with_scores.csv")
    )
    parser.add_argument(
        "--ablations", type=int, nargs="+", choices=sorted(ABLATIONS),
        default=sorted(ABLATIONS),
    )
    parser.add_argument("--num-eval-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--with-evaltree-internal-eval",
        action="store_true",
        help="Also run EvalTree's non-construction in-pipeline judgments",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = args.dataset_label or args.output_dir.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_label or ""):
        parser.error("--dataset-label must be a non-empty filename-safe label")
    shared_base = args.output_dir / f"{dataset_label}_shared_leaf_base.json"
    if not args.report_only:
        shared_base = prepare_shared_base(
            args.source, args.output_dir, args.overwrite, dataset_label
        )
        for ablation in args.ablations:
            run_ablation(
                shared_base,
                ablation,
                args.output_dir,
                num_eval_samples=args.num_eval_samples,
                score_file=args.score_file,
                posthoc_eval=True,
                seed=args.seed,
                overwrite=args.overwrite,
                run_internal_evaluation=args.with_evaltree_internal_eval,
                collect_internal_metrics=False,
            )
            write_reports(
                completed_results(args.output_dir, shared_base.stem),
                args.output_dir,
                dataset_label,
            )
    else:
        if not shared_base.exists():
            parser.error(f"shared base does not exist: {shared_base}")
        write_reports(
            completed_results(args.output_dir, shared_base.stem),
            args.output_dir,
            dataset_label,
        )


if __name__ == "__main__":
    main()
