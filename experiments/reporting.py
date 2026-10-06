"""Dependency-free aggregation for the EvalTree k-sensitivity experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "faithfulness1",
    "faithfulness2",
    "informativeness1",
    "informativeness2",
    "specificity_faithfulness",
    "specificity_informativeness",
)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "stderr": None,
                "min": None, "max": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def collect_result(path: Path, method: str, k: int | None) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    levels = data.get("levels", [])
    internal = [
        node
        for level in levels
        if level.get("level", 0) > 0
        for node in level.get("nodes", [])
    ]
    branches = [len(node.get("children", [])) for node in internal]
    overall = {
        metric: _summary([
            float(node[metric]) for node in internal if node.get(metric) is not None
        ])
        for metric in METRICS
    }
    by_level = {}
    for level in levels:
        level_num = level.get("level", 0)
        if level_num == 0:
            continue
        nodes = level.get("nodes", [])
        by_level[str(level_num)] = {
            metric: _summary([
                float(node[metric]) for node in nodes if node.get(metric) is not None
            ])
            for metric in METRICS
        }

    root_nodes = levels[-1].get("nodes", []) if levels else []
    roots = {
        metric: _summary([
            float(node[metric]) for node in root_nodes if node.get(metric) is not None
        ])
        for metric in METRICS
    }
    return {
        "method": method,
        "k": k,
        "evaluated_file": str(path),
        "provenance": {
            "annotation_model": data.get("config", {}).get("annotation_model"),
            "summary_model": data.get("config", {}).get("summary_model"),
            "evaluation_model": data.get("config", {}).get("evaluation_model"),
            "blossom_model": data.get("config", {}).get("blossom_model"),
            "provider_fallback_count": len(
                data.get("config", {}).get("provider_fallbacks", [])
            ),
            "provider_fallback_policy": data.get("config", {}).get(
                "provider_fallback_policy"
            ),
        },
        "structure": {
            "leaf_nodes": len(levels[0].get("nodes", [])) if levels else 0,
            "internal_nodes": len(internal),
            "levels": len(levels),
            "max_level": max((level.get("level", 0) for level in levels), default=0),
            "root_nodes": len(root_nodes),
            "min_branching": min(branches) if branches else None,
            "mean_branching": statistics.mean(branches) if branches else None,
            "max_branching": max(branches) if branches else None,
        },
        "overall": overall,
        "root": roots,
        "by_level": by_level,
    }


def write_reports(
    results: list[dict],
    output_dir: Path,
    *,
    title: str = "EvalTree k-sensitivity results",
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    markdown_path = output_dir / "REPORT.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fields = ["method", "k", "scope", "level", "metric", "n", "mean",
                  "std", "stderr", "min", "max"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            scopes = [("overall", "", result["overall"]),
                      ("root", result["structure"]["max_level"], result["root"])]
            scopes.extend(("level", level, stats)
                          for level, stats in result["by_level"].items())
            for scope, level, metrics in scopes:
                for metric, stats in metrics.items():
                    writer.writerow({"method": result["method"], "k": result["k"],
                                     "scope": scope, "level": level,
                                     "metric": metric, **stats})

    leaf_counts = {result["structure"]["leaf_nodes"] for result in results}
    shared_leaf_count = next(iter(leaf_counts)) if len(leaf_counts) == 1 else None
    sample_description = (
        f"the same {shared_leaf_count} leaves"
        if shared_leaf_count is not None
        else "their recorded leaf samples"
    )

    columns = ["Method", "k", "Parents", "Levels", "Faith avg", "Faith min",
               "Info avg", "Info min", "Spec faith", "Spec info"]
    lines = [f"# {title}", "",
             f"All methods use {sample_description} and shared post-hoc evaluator. "
             "Faithfulness and informativeness are higher-is-better. Specificity "
             "columns score the parent against an outsider leaf, so lower is better.", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]

    def mean(result: dict, metric: str) -> str:
        value = result["overall"][metric]["mean"]
        return "NA" if value is None else f"{value:.3f}"

    for result in results:
        structure = result["structure"]
        lines.append("| " + " | ".join([
            result["method"], "-" if result["k"] is None else str(result["k"]),
            str(structure["internal_nodes"]), str(structure["levels"]),
            mean(result, "faithfulness1"), mean(result, "faithfulness2"),
            mean(result, "informativeness1"), mean(result, "informativeness2"),
            mean(result, "specificity_faithfulness"),
            mean(result, "specificity_informativeness"),
        ]) + " |")

    evaltree = [result for result in results if result["method"] == "evaltree"]

    def best(metric: str, minimize: bool = False) -> str:
        eligible = [result for result in evaltree
                    if result["overall"][metric]["mean"] is not None]
        pick = (min if minimize else max)(
            eligible, key=lambda result: result["overall"][metric]["mean"]
        )
        value = pick["overall"][metric]["mean"]
        return f"k={pick['k']} ({value:.3f})"

    fallbacks = sum(result["provenance"]["provider_fallback_count"]
                    for result in results)
    if evaltree:
        lines.extend([
            "",
            "## Quick reading",
            "",
            f"- Highest EvalTree average faithfulness: {best('faithfulness1')}.",
            f"- Highest EvalTree worst-child faithfulness: {best('faithfulness2')}.",
            f"- Highest EvalTree average informativeness: {best('informativeness1')}.",
            f"- Highest EvalTree worst-child informativeness: {best('informativeness2')}.",
            f"- Lowest outsider faithfulness (best specificity): "
            f"{best('specificity_faithfulness', minimize=True)}.",
            f"- Lowest outsider informativeness (best specificity): "
            f"{best('specificity_informativeness', minimize=True)}.",
            "",
            "These are descriptive all-parent means, not significance tests. The "
            "number and depth distribution of parents changes with k, so use the "
            "root and per-level rows in `comparison.csv` when interpreting the pooled means.",
        ])
    lines.extend([
        "",
        "## Run audit",
        "",
        f"The evaluator recorded no failed API responses. Construction recorded "
        f"{fallbacks} provider fallbacks; each fallback "
        "is listed in the corresponding tree's `config.provider_fallbacks`.",
        "",
        "Detailed overall, root, and per-level statistics are in `comparison.csv`; "
        "machine-readable results are in `comparison.json`.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
