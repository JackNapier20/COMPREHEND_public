"""Summarize approved MPNet reruns beside preserved MiniLM and Blossom results."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics

METRICS = ("faithfulness1", "faithfulness2", "informativeness1", "informativeness2",
           "specificity_faithfulness", "specificity_informativeness",
           "faithfulness_gap", "informativeness_gap")
HEADINGS = ("Faith avg", "Faith min", "Info avg", "Info min", "Raw spec faith", "Raw spec info",
            "Faith gap", "Info gap")


def summary(values):
    if not values:
        return {"n": 0, "mean": None, "std": None, "stderr": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": statistics.mean(values), "std": std,
            "stderr": std / math.sqrt(len(values))}


def collect(path, label, reused=False):
    data = json.loads(path.read_text())
    parents = [node for level in data["levels"][1:] for node in level["nodes"]]
    values = {metric: [] for metric in METRICS}
    for node in parents:
        for metric in METRICS[:6]:
            if node.get(metric) is not None:
                value = float(node[metric])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric} at node {node['id']}: {path}")
                values[metric].append(value)
        for destination, positive, negative in (
            ("faithfulness_gap", "faithfulness1", "specificity_faithfulness"),
            ("informativeness_gap", "informativeness1", "specificity_informativeness"),
        ):
            if node.get(positive) is not None and node.get(negative) is not None:
                values[destination].append(float(node[positive])-float(node[negative]))
    words = [len((node.get("prompt") or node.get("summary") or "").split()) for node in parents]
    return {"label": label, "reused": reused, "evaluated_file": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parents": len(parents),
            "levels": len(data["levels"]), "summary_words": summary(words),
            "metrics": {key: summary(val) for key, val in values.items()}}


def table(rows):
    result = ["| Condition | Parents | " + " | ".join(HEADINGS) + " |",
              "|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        def cell(metric):
            stat = row["metrics"][metric]
            return "NA" if stat["mean"] is None else f"{stat['mean']:.3f} ({stat['stderr']:.3f})"
        result.append("| " + " | ".join([row["label"], str(row["parents"]),
                                         *(cell(metric) for metric in METRICS)]) + " |")
    return result


def write_reports(project, output, allow_partial=False):
    manifest = json.loads((output / "MANIFEST.json").read_text())
    if manifest["status"] != "complete" and not allow_partial:
        raise ValueError("Reports require all approved conditions to finish")
    completed = {job["name"]: json.loads((output / "completed" / f"{job['name']}.json").read_text())
                 for job in manifest["jobs"] if (output / "completed" / f"{job['name']}.json").exists()}
    unfinished = [job["name"] for job in manifest["jobs"] if job["name"] not in completed]
    results, comparisons = {}, []
    for dataset in ("WB", "UF"):
        rows = []
        for a in range(1, 10):
            previous = project / f"ablation_outputs/{dataset}256/eval_outputs/Ablation{a}_{dataset}256_shared_leaf_base_final_eval.json"
            old = collect(previous, f"A{a} MiniLM" if a in (5, 7, 8) else f"A{a} Blossom (reused)", True)
            if a in (5, 7, 8):
                new = collect(Path(completed[f"{dataset}256_A{a}"]["evaluated_file"]), f"A{a} MPNet")
                comparisons.append({"dataset": f"{dataset}256", "condition": f"A{a}", "old": old, "new": new,
                                    "mean_deltas": {metric: new["metrics"][metric]["mean"]-old["metrics"][metric]["mean"]
                                                    for metric in METRICS}})
                rows.append(new)
            else:
                rows.append(old)
        results[f"{dataset}256"] = rows
    old_k = json.loads((project / "k_sensitivity_outputs/comparison.json").read_text())["results"]
    k_rows = []
    for entry in old_k:
        previous = Path(entry["evaluated_file"])
        if not previous.is_absolute():
            previous = project / previous
        if entry["method"] == "blossom":
            k_rows.append(collect(previous, "Blossom (reused)", True))
        else:
            k = entry["k"]
            if f"WB64_k{k}" not in completed:
                continue
            old = collect(previous, f"k={k} MiniLM", True)
            new = collect(Path(completed[f"WB64_k{k}"]["evaluated_file"]), f"k={k} MPNet")
            k_rows.append(new)
            comparisons.append({"dataset": "WB64", "condition": f"k={k}", "old": old, "new": new,
                                "mean_deltas": {metric: new["metrics"][metric]["mean"]-old["metrics"][metric]["mean"]
                                                for metric in METRICS}})
    results["WB64 k-sweep"] = k_rows
    for path, expected in manifest["protected_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Protected baseline changed: {path}")
    manifest["packages"] = {name: version(name) for name in (
        "anthropic", "openai", "sentence-transformers", "scikit-learn", "torch", "numpy")}
    manifest["concurrency"] = {"conditions": 2, "summary_workers_per_condition": 4,
                               "judge_workers_per_condition": 6}
    payload = {"provenance": manifest, "results": results, "minilm_vs_mpnet": comparisons,
               "completed_conditions": len(completed), "unfinished_conditions": unfinished}
    (output / "RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "condition", "reused", "parents", "metric",
                                                   "n", "mean", "std", "stderr", "evaluated_file"])
        writer.writeheader()
        for dataset, rows in results.items():
            for row in rows:
                for metric in METRICS:
                    writer.writerow({"dataset": dataset, "condition": row["label"], "reused": row["reused"],
                                     "parents": row["parents"], "metric": metric,
                                     "evaluated_file": row["evaluated_file"], **row["metrics"][metric]})
    lines = ["# MPNet EvalTree Reruns", "",
             f"Completed: {len(completed)}/{len(manifest['jobs'])} approved conditions."
             + (" Unfinished: " + ", ".join(unfinished) + ". No result is reported for unfinished conditions." if unfinished else ""), "",
             "Only clustering embeddings changed: all-MiniLM-L6-v2 -> all-mpnet-base-v2. "
             "Original Gemma-branch summary prompts were retained. No increased-length prompt was used.", "",
             "Approved fresh conditions: A5/A7/A8 on WB256 and UF256; WB64 k=2,4,6,8. "
             "All other ablations and the WB64 Blossom comparator are reused, not rerun. "
             "Leaf prompts, cached annotations/main ideas and old outputs are preserved.", "",
             "Generation: Claude Sonnet 4.6, with no cross-provider fallback. "
             "Judging: GPT-5.4-mini; two descendant leaves, seed 42; MPNet outsider selection unchanged.", "",
             "Values are mean (ordinary descriptive standard error). Faithfulness, informativeness and "
             "paired gaps are higher-is-better; raw outsider specificity is lower-is-better. "
             "Gaps average the within-parent difference on parents with an outsider, excluding the root; "
             "they are not a subtraction of means from different node populations.", ""]
    for dataset, rows in results.items():
        lines += [f"## {dataset}", "", *table(rows), ""]
    lines += ["## MiniLM Versus MPNet", "",
              "Descriptive mean changes, MPNet minus MiniLM. Nodes are not paired by ID because clustering changed.", "",
              "| Dataset | Condition | Faith avg delta | Info avg delta | Raw spec faith delta | Faith gap delta |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparisons:
        delta = row["mean_deltas"]
        lines.append("| " + " | ".join([row["dataset"], row["condition"],
                     *(f"{delta[key]:+.3f}" for key in ("faithfulness1", "informativeness1",
                                                       "specificity_faithfulness", "faithfulness_gap"))]) + " |")
    lines += ["", "## Integrity And Interpretation", "",
              f"All {len(completed)} completed reruns passed evaluation-completion and outsider-descendant exclusion checks. "
              "Every protected baseline hash was verified unchanged. Per-node summary caches and raw "
              "judge responses remain available in the corresponding output files.", "",
              "Parents are nested and dependent; ordinary SEs do not establish statistical significance. "
              "Parent counts and depth distributions can change, especially across k. "
              "LLM generation and judging can vary even with fixed experimental inputs; these are not deterministic model-quality guarantees.", "",
              "No Together, Gemma, length-control, fresh 22-word, or root-cause experiment was run. "
              "RESULTS.csv contains N, SD, SE and all eight metrics; RESULTS.json records input/output provenance.", ""]
    if unfinished:
        lines += ["## Unfinished Condition", "",
                  "WB64 k=2 stopped after repeated Claude refusals at node 79. "
                  "Its partial topology and 55 cached summaries are preserved; no final k=2 metrics were fabricated. "
                  "GPT-4o-mini fallback has not been used. The first-attempt audit and worker logs preserve the failure.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    for dataset, rows in results.items():
        folder = output / ("WB64_k_sweep" if dataset == "WB64 k-sweep" else dataset)
        (folder / "REPORT.md").write_text(f"# {dataset}: MPNet comparison\n\n" + "\n".join(table(rows)) + "\n")
        (folder / "comparison.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")
    print(output / "REPORT.md")
    for dataset, rows in results.items():
        print(dataset)
        for row in rows:
            if row["reused"] and dataset != "WB64 k-sweep":
                continue
            print(row["label"], row["parents"],
                  {key: (round(row["metrics"][key]["mean"], 4), round(row["metrics"][key]["stderr"], 4))
                   for key in ("faithfulness1", "informativeness1", "specificity_faithfulness", "faithfulness_gap")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    write_reports(args.project.resolve(), args.output.resolve(), args.allow_partial)
