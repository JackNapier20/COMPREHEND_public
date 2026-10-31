"""Fresh, resumable A1-A4 reruns with mandatory construction assignment scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading

CODE = Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[name] = "1"
os.environ.update(TOKENIZERS_PARALLELISM="false", JOBLIB_MULTIPROCESSING="0",
                  BLOSSOM_PROVIDER="claude", SUMMARY_GENERATION_PROVIDER="claude",
                  EVALUATION_PROVIDER="openai", BLOSSOM_SUMMARY_PROMPT_VARIANT="current",
                  CLAUDE_SUMMARY_MAX_WORKERS="4", INTERNAL_EVAL_MAX_WORKERS="6",
                  POSTHOC_MAX_WORKERS="6")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def configure(project):
    for line in (project / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    from main import API_KEYS, MODEL_CONFIG
    MODEL_CONFIG.update(blossom="claude", summary_generation="claude", evaluation="openai")
    if not API_KEYS["claude"] or not API_KEYS["openai"]:
        raise ValueError("Claude and OpenAI API keys must be configured")


def prepare(project, output):
    from ablations.runner import ABLATIONS, ASSIGNMENT_SCORING_VERSION
    from tree import Tree
    previous = project / "ablation_outputs/mpnet_WB64_k2_fresh_20261018T055004Z/combined_results/RESULTS.json"
    prior = json.loads(previous.read_text())
    protected = {str(previous): digest(previous)}
    for dataset in ("WB256", "UF256"):
        directory = project / "ablation_outputs" / dataset
        for path in directory.rglob("*"):
            if path.is_file():
                protected[str(path)] = digest(path)
        for row in prior["results"][dataset]:
            path = Path(row["evaluated_file"])
            protected[str(path)] = digest(path)
    jobs = []
    for a in (1, 2, 3, 4):
        assert ABLATIONS[a]["method"] == "blossom" and ABLATIONS[a].get("reassign", True)
        for dataset in ("WB", "UF"):
            source = project / f"ablation_outputs/{dataset}256/{dataset}256_shared_leaf_base.json"
            tree = Tree.load(source)
            raw = project / f"{dataset}_new_random_256.json"
            canonical = Tree.load(raw)
            assert len(tree.levels) == 1 and len(tree.levels[0].nodes) == 256
            assert {n.id: n.prompt for n in tree.levels[0].nodes} == {n.id: n.prompt for n in canonical.levels[0].nodes}
            assert all(n.capability and n.mainIdea for n in tree.levels[0].nodes)
            assert all(n.specificity is None and not n.parent and not n.children for n in tree.levels[0].nodes)
            protected[str(raw)] = digest(raw)
            target = output / f"{dataset}256/Ablation{a}_{source.stem}_final.json"
            jobs.append({"name": f"{dataset}256_A{a}", "dataset": dataset, "ablation": a,
                         "source": str(source), "source_sha256": digest(source),
                         "output": str(target), "previous_evaluation": prior["results"][f"{dataset}256"][a-1]["evaluated_file"]})
    code = {name: digest(CODE / name) for name in (
        "ablations/runner.py", "evaluation/simple_eval.py", "evaluation/evaluate.py",
        "retrieval/blossom_matching.py", "retrieval/third_neighbor.py", "retrieval/similarity.py",
        "common_summary_generation/summarize.py", "evaltree/prompts.py", "llm_utils.py",
        "experiments/assignment_rerun.py", "experiments/mpnet_report.py")}
    identity = {"jobs": jobs, "protected_sha256": protected, "code_sha256": code,
                "previous_results": str(previous), "assignment_scoring_version": ASSIGNMENT_SCORING_VERSION}
    path = output / "MANIFEST.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in identity:
            if old[key] != identity[key]:
                raise ValueError(f"Resume identity changed: {key}")
        return old
    manifest = {**identity, "status": "preflight_complete", "summary_model": "claude-sonnet-4-6",
                "judge_model": "gpt-5.4-mini", "matching_embedding_model": "all-mpnet-base-v2",
                "num_samples": 2, "seed": 42, "fresh_parent_summaries": True,
                "fresh_leaf_annotations": 0, "fresh_main_ideas": 0,
                "collect_internal_metrics": False, "assignment_scoring_required": True,
                "unchanged_assignment_threshold": 0.7, "unchanged_regeneration_threshold": 0.4,
                "unchanged_max_generation_rounds": 2, "strict_claude_no_fallback": True,
                "assignment_score_semantics": "Existing legacy mean of node-to-summary and outsider-to-summary entailment; unchanged",
                "condition_workers": 4, "summary_workers_per_condition": 4,
                "judge_workers_per_condition": 6,
                "initial_generation_calls": 2040,
                "initial_child_judge_calls": 4080,
                "assignment_judge_calls": 8128,
                "posthoc_judge_call_upper_bound": 12224,
                "additional_regeneration_call_upper_bound_before_retries": 6120}
    save(path, manifest)
    shutil.copy2(__file__, output / "RUNNER.py")
    for name in ("ablations/runner.py", "evaluation/simple_eval.py"):
        destination = output / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, destination)
    return manifest


def verify(manifest):
    for path, expected in manifest["protected_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Protected previous artifact changed: {path}")
    for name, expected in manifest["code_sha256"].items():
        if digest(CODE / name) != expected:
            raise ValueError(f"Run code changed: {name}")


def audit(path, job):
    from evaluation.simple_eval import safe_extract_score
    from experiments.specificity_diagnostics import TreeArtifact
    a = TreeArtifact.load(job["name"], path)
    config = a.payload["config"]
    from ablations.runner import ASSIGNMENT_SCORING_VERSION
    assert config["assignment_scoring_version"] == ASSIGNMENT_SCORING_VERSION
    assert config["posthoc_evaluation_status"] == "complete"
    assert config["construction_status"] == "complete"
    assert config["strict_summary_provider"] and not config.get("provider_fallbacks")
    source = json.loads(Path(job["source"]).read_text())
    assert {n["id"]: n["prompt"] for n in source["levels"][0]["nodes"]} == {
        n["id"]: n["prompt"] for n in a.levels[0]["nodes"]}
    decisions = high = raw_responses = 0
    for level in a.levels[:-1]:
        for node in level["nodes"]:
            third = node.get("third")
            if third is None or not node.get("summary"):
                continue
            assert third != node.get("match")
            assert node["specificity"] is not None, (node["id"], "missing assignment decision score")
            response = node.get("evaluation_specificity") or ""
            matches = re.fullmatch(r"Eval A:\n(.*?)\nEval B:\n(.*)", response, flags=re.S)
            assert matches, (node["id"], "missing raw assignment judgments")
            scores = [safe_extract_score(text) for text in matches.groups()]
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            assert math.isclose(sum(scores)/2, node["specificity"], abs_tol=1e-12)
            decisions += 1
            raw_responses += 2
            if node["specificity"] > 0.7:
                primary = node["parent"][0]
                outsider = level["nodes"][third]
                assert primary in outsider["parent"] and outsider["id"] in a.by_id[primary]["children"]
                high += 1
    judge_responses = 0
    for node in a.internal_nodes(scored_only=False):
        assert node.get("posthoc_evaluation_complete")
        desc = set(a.descendants(node["id"]))
        sampled = []
        for text, avg, minimum in (("evaluation_faithfulness", "faithfulness1", "faithfulness2"),
                                   ("evaluation_informativeness", "informativeness1", "informativeness2")):
            matches = re.findall(r"(?:^|\n\n)Child (\d+):\n(.*?)(?=\n\nChild \d+:\n|\Z)", node[text], re.S)
            assert len(matches) == 2
            sampled.append([int(i) for i, _ in matches])
            assert all(i in desc and i in a.leaf_ids for i in sampled[-1])
            scores = [safe_extract_score(s) for _, s in matches]
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            assert math.isclose(sum(scores)/2, node[avg], abs_tol=1e-12)
            assert math.isclose(min(scores), node[minimum], abs_tol=1e-12)
            assert all("Error:" not in s for _, s in matches)
            judge_responses += 2
        assert sampled[0] == sampled[1]
        for score, response in (("specificity_faithfulness", "evaluation_specificity_faithfulness"),
                                ("specificity_informativeness", "evaluation_specificity_informativeness")):
            if node[score] is not None:
                assert node["third_leaf"] in a.leaf_ids and node["third_leaf"] not in desc
                value = safe_extract_score(node[response])
                assert 0 <= value <= 1 and "Error:" not in node[response]
                assert math.isclose(value, node[score], abs_tol=1e-12)
                judge_responses += 1
    return {"assignment_decisions_scored": decisions, "assignment_raw_responses": raw_responses,
            "above_threshold_decisions_verified": high,
            "extra_assignment_edges": sum(max(0, len(n.get("parent") or [])-1) for n in a.by_id.values()),
            "multi_parent_nodes": sum(len(n.get("parent") or []) > 1 for n in a.by_id.values()),
            "posthoc_raw_responses_verified": judge_responses,
            "descendant_outsiders": 0, "score_mismatches": 0, "provider_fallbacks": 0}


def worker(project, output, name):
    from ablations import runner
    manifest = json.loads((output / "MANIFEST.json").read_text())
    verify(manifest)
    job = next(j for j in manifest["jobs"] if j["name"] == name)
    generate = runner.generate_summary
    def strict(tree, *args, **kwargs):
        tree.config.update(strict_summary_provider=True, summary_model="claude-sonnet-4-6",
                           judge_model="gpt-5.4-mini", provider_fallback_policy="Strict Claude; no cross-provider fallback")
        return generate(tree, *args, **kwargs)
    runner.generate_summary = strict
    target = Path(job["output"])
    runner.run_ablation(Path(job["source"]), job["ablation"], target.parent,
                        score_file=project / f"dataset_{job['dataset']}_with_scores.csv",
                        posthoc_eval=True, num_eval_samples=2, seed=42,
                        run_internal_evaluation=False, collect_internal_metrics=False)
    evaluated = target.parent / "eval_outputs" / f"{target.stem}_eval.json"
    checks = audit(evaluated, job)
    verify(manifest)
    save(output / "completed" / f"{name}.json", {"job": job, "evaluated_file": str(evaluated),
         "tree_sha256": digest(target), "evaluated_sha256": digest(evaluated), "audit": checks})
    print("COMPLETE", name, json.dumps(checks), flush=True)


def report(output):
    from experiments.mpnet_report import METRICS, collect, table
    manifest = json.loads((output / "MANIFEST.json").read_text())
    verify(manifest)
    prior = json.loads(Path(manifest["previous_results"]).read_text())
    results, comparisons = {}, []
    lines = ["# Assignment-Corrected Claude Ablations", "",
             "A1-A4 were rebuilt from the same fixed WB256/UF256 leaves. Claude Sonnet 4.6 generation, "
             "OpenAI GPT-5.4-mini construction and posthoc judges, original prompts, two descendant leaves, seed 42. "
             "Assignment uses its existing legacy score and threshold 0.7; optional split internal metrics remain off. "
             "No provider fallback. A5-A9 are retained from the previous completed MPNet suite.", "",
             "Mean (ordinary descriptive SE). Raw outsider specificity is lower-is-better. Paired gaps use only parents "
             "with valid outsiders, excluding the root and any other full-coverage DAG nodes. "
             "Summary length counts words in the evaluated parent prompt, including the root. "
             "Nested parents are not independent; these are not significance tests.", ""]
    audits = []
    for dataset in ("WB256", "UF256"):
        rows = []
        for index, old in enumerate(prior["results"][dataset], 1):
            job = next((j for j in manifest["jobs"] if j["name"] == f"{dataset}_A{index}"), None)
            completed = output / "completed" / f"{dataset}_A{index}.json"
            if job and not completed.exists():
                continue
            if job:
                saved = json.loads(completed.read_text())
                assert saved["job"] == job and digest(saved["evaluated_file"]) == saved["evaluated_sha256"]
                assert digest(job["output"]) == saved["tree_sha256"]
                row = collect(Path(saved["evaluated_file"]), f"A{index} corrected")
                comparisons.append({"dataset": dataset, "ablation": index, "previous": old, "corrected": row})
                audits.append({"condition": job["name"], **saved["audit"]})
            else:
                row = collect(Path(old["evaluated_file"]), f"A{index} (retained)", True)
            rows.append(row)
        results[dataset] = rows
        lines += [f"## {dataset}", "", *table(rows), "",
                  "| Condition | Summary words (SE) |", "|---|---:|"]
        for row in rows:
            s = row["summary_words"]
            lines.append(f"| {row['label']} | {s['mean']:.2f} ({s['stderr']:.2f}) |")
        lines += [""]
    lines += ["## Assignment Audit", "", "| Condition | Scored decisions | Added edges | Multi-parent nodes |",
              "|---|---:|---:|---:|"]
    for row in audits:
        lines.append(f"| {row['condition']} | {row['assignment_decisions_scored']} | {row['extra_assignment_edges']} | {row['multi_parent_nodes']} |")
    lines += ["", f"{len(audits)}/8 affected conditions complete. All protected baseline hashes verified unchanged.", ""]
    save(output / "RESULTS.json", {"provenance": manifest, "results": results,
                                  "previous_vs_corrected": comparisons, "assignment_audits": audits})
    with (output / "RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "condition", "reused", "parents", "metric",
                                                   "n", "mean", "std", "stderr", "evaluated_file"])
        writer.writeheader()
        for dataset, rows in results.items():
            for row in rows:
                for metric in (*METRICS, "summary_words"):
                    stats = row["summary_words"] if metric == "summary_words" else row["metrics"][metric]
                    writer.writerow({"dataset": dataset, "condition": row["label"], "reused": row["reused"],
                                     "parents": row["parents"], "metric": metric,
                                     "evaluated_file": row["evaluated_file"], **stats})
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if args.report_only:
        report(output)
        return
    configure(project)
    if args.worker:
        worker(project, output, args.worker)
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare(project, output)
    print("Eight fresh A1-A4 conditions; at most 26,472 initial generation/decision/posthoc calls, "
          "plus up to 6,120 regeneration calls and provider retries. Four conditions in parallel.", flush=True)
    if args.preflight_only:
        return
    manifest.update(status="running", packages={name: version(name) for name in (
        "anthropic", "openai", "sentence-transformers", "scikit-learn", "torch", "numpy", "networkx")})
    save(output / "MANIFEST.json", manifest)
    stop = threading.Event()
    def launch(job):
        if stop.is_set():
            raise RuntimeError("Stopped launching after a failed condition")
        complete = output / "completed" / f"{job['name']}.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            assert saved["job"] == job and digest(saved["evaluated_file"]) == saved["evaluated_sha256"]
            assert digest(job["output"]) == saved["tree_sha256"]
            audit(Path(saved["evaluated_file"]), job)
            return
        log = output / "logs" / f"{job['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print("START", job["name"], flush=True)
        with log.open("a") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.assignment_rerun", "--project", str(project),
                            "--output", str(output), "--worker", job["name"]],
                           cwd=CODE, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("DONE", job["name"], flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(launch, job) for job in manifest["jobs"]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    stop.set()
                    for queued in futures:
                        queued.cancel()
                    raise
        verify(manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "MANIFEST.json", manifest)
        report(output)
    print("All eight affected conditions complete; previous artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
