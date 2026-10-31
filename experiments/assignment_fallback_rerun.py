"""Isolated WB256 A4 rebuild with the existing documented summary fallback."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

from experiments.assignment_rerun import CODE, configure, digest, save


def validate(path, source):
    from evaluation.simple_eval import safe_extract_score
    from experiments.specificity_diagnostics import TreeArtifact

    artifact = TreeArtifact.load("WB256_A4", path)
    config = artifact.payload["config"]
    assert config["construction_status"] == config["posthoc_evaluation_status"] == "complete"
    assert config["strict_summary_provider"] is False
    original = json.loads(source.read_text())
    assert {n["id"]: n["prompt"] for n in original["levels"][0]["nodes"]} == {
        n["id"]: n["prompt"] for n in artifact.levels[0]["nodes"]}
    decisions = responses = 0
    for level in artifact.levels[:-1]:
        for node in level["nodes"]:
            third = node.get("third")
            if third is None or not node.get("summary"):
                continue
            assert third != node.get("match")
            match = re.fullmatch(r"Eval A:\n(.*?)\nEval B:\n(.*)",
                                 node.get("evaluation_specificity") or "", re.DOTALL)
            assert match, node["id"]
            scores = [safe_extract_score(text) for text in match.groups()]
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            assert math.isclose(sum(scores) / 2, node["specificity"], abs_tol=1e-12)
            if node["specificity"] > 0.7:
                parent = node["parent"][0]
                outsider = level["nodes"][third]
                assert parent in outsider["parent"]
                assert outsider["id"] in artifact.by_id[parent]["children"]
            decisions += 1
    for node in artifact.internal_nodes(scored_only=False):
        assert node.get("posthoc_evaluation_complete")
        descendants = set(artifact.descendants(node["id"]))
        sampled = []
        for field, avg, minimum in (
            ("evaluation_faithfulness", "faithfulness1", "faithfulness2"),
            ("evaluation_informativeness", "informativeness1", "informativeness2"),
        ):
            pairs = re.findall(r"(?:^|\n\n)Child (\d+):\n(.*?)(?=\n\nChild \d+:\n|\Z)",
                               node[field], re.DOTALL)
            assert len(pairs) == 2
            sampled.append([int(i) for i, _ in pairs])
            assert all(i in descendants and i in artifact.leaf_ids for i in sampled[-1])
            scores = [safe_extract_score(text) for _, text in pairs]
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            assert math.isclose(sum(scores) / 2, node[avg], abs_tol=1e-12)
            assert math.isclose(min(scores), node[minimum], abs_tol=1e-12)
            assert all("Error:" not in text for _, text in pairs)
            responses += 2
        assert sampled[0] == sampled[1]
        for score, field in (
            ("specificity_faithfulness", "evaluation_specificity_faithfulness"),
            ("specificity_informativeness", "evaluation_specificity_informativeness"),
        ):
            if node.get(score) is not None:
                assert node["third_leaf"] in set(artifact.leaf_ids) - descendants
                value = safe_extract_score(node[field])
                assert math.isfinite(value) and 0 <= value <= 1
                assert "Error:" not in node[field]
                assert math.isclose(value, node[score], abs_tol=1e-12)
                responses += 1
    fallbacks = config.get("provider_fallbacks", [])
    assert all(f["from"] == "claude-sonnet-4-6" and f["to"] == "gpt-4o-mini"
               and f["stage"] == "blossom_summary" for f in fallbacks)
    return {"assignment_decisions_scored": decisions,
            "assignment_raw_responses": decisions * 2,
            "posthoc_raw_responses_verified": responses,
            "extra_assignment_edges": sum(max(0, len(n.get("parent") or []) - 1)
                                          for n in artifact.by_id.values()),
            "descendant_outsiders": 0, "score_mismatches": 0,
            "provider_fallback_events": fallbacks,
            "fallback_node_levels": sorted({(f["level"], f["node_id"]) for f in fallbacks})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = project / "ablation_outputs/WB256/WB256_shared_leaf_base.json"
    manifest_path = output / "MANIFEST.json"
    names = ("ablations/runner.py", "evaluation/simple_eval.py", "evaluation/evaluate.py",
             "retrieval/blossom_matching.py", "retrieval/third_neighbor.py", "retrieval/similarity.py",
             "common_summary_generation/summarize.py", "evaltree/prompts.py", "llm_utils.py",
             "experiments/assignment_fallback_rerun.py")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source"] == str(source)
    else:
        protected = {str(source): digest(source)}
        for directory in (project / "ablation_outputs/WB256",
                          project / "ablation_outputs/assignment_fixed_20261025"):
            for path in directory.rglob("*"):
                if path.is_file():
                    protected[str(path)] = digest(path)
        manifest = {"status": "prepared", "condition": "WB256_A4",
                    "source": str(source), "source_sha256": digest(source),
                    "protected_sha256": protected,
                    "code_sha256": {name: digest(CODE / name) for name in names},
                    "summary_model": "claude-sonnet-4-6", "fallback_model": "gpt-4o-mini",
                    "fallback_policy": "Existing default: Claude retries, then OpenAI on RuntimeError",
                    "judge_model": "gpt-5.4-mini", "embedding_model": "all-mpnet-base-v2",
                    "fresh_tree": True, "cached_leaf_annotations": True,
                    "assignment_threshold": 0.7, "regeneration_threshold": 0.4,
                    "max_generation_rounds": 2, "num_eval_samples": 2, "seed": 42}
        save(manifest_path, manifest)
        for name in names:
            target = output / "code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE / name, target)

    def verify():
        for path, expected in manifest["protected_sha256"].items():
            assert digest(path) == expected, f"Protected artifact changed: {path}"
        for name, expected in manifest["code_sha256"].items():
            assert digest(CODE / name) == expected, f"Code changed: {name}"

    script = "experiments/assignment_fallback_rerun.py"
    if args.report_only and manifest["code_sha256"][script] != digest(CODE / script):
        # Preserve the executed snapshot; reporting fixes must not change construction code.
        for name, expected in manifest["code_sha256"].items():
            if name != script:
                assert digest(CODE / name) == expected, f"Construction code changed: {name}"
        manifest.setdefault("reporting_code_amendments", []).append({
            "original_sha256": manifest["code_sha256"][script],
            "updated_sha256": digest(CODE / script),
            "reason": "Audit-only fix: cast leaf ID list to a set for descendant exclusion",
            "additional_api_calls": 0,
        })
        manifest["code_sha256"][script] = digest(CODE / script)
        shutil.copy2(CODE / script, output / "REPORT_RUNNER.py")
        save(manifest_path, manifest)
    verify()
    target = output / "WB256/Ablation4_WB256_shared_leaf_base_final.json"
    evaluated = target.parent / "eval_outputs" / f"{target.stem}_eval.json"
    if not args.report_only:
        configure(project)
        from ablations import runner
        generate = runner.generate_summary

        def with_fallback(tree, *a, **kw):
            tree.config.update(strict_summary_provider=False,
                               summary_model="claude-sonnet-4-6", judge_model="gpt-5.4-mini",
                               provider_fallback_policy=manifest["fallback_policy"])
            return generate(tree, *a, **kw)

        runner.generate_summary = with_fallback
        manifest["status"] = "running"
        save(manifest_path, manifest)
        try:
            runner.run_ablation(source, 4, target.parent,
                                score_file=project / "dataset_WB_with_scores.csv",
                                posthoc_eval=True, num_eval_samples=2, seed=42,
                                run_internal_evaluation=False, collect_internal_metrics=False)
        except Exception as exc:
            manifest.update(status="failed_resumable", error=f"{type(exc).__name__}: {exc}")
            save(manifest_path, manifest)
            raise
    checks = validate(evaluated, source)
    verify()
    from experiments.mpnet_report import collect, table
    row = collect(evaluated, "WB256 A4 Claude + documented GPT fallback")
    manifest.update(status="complete", tree_sha256=digest(target), evaluated_sha256=digest(evaluated))
    save(manifest_path, manifest)
    save(output / "AUDIT.json", checks)
    save(output / "RESULTS.json", {"provenance": manifest, "result": row, "audit": checks})
    with (output / "RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "n", "mean", "std", "stderr"])
        writer.writeheader()
        for metric, stat in {**row["metrics"], "summary_words": row["summary_words"]}.items():
            writer.writerow({"metric": metric, **stat})
    words = row["summary_words"]
    lines = ["# WB256 A4 Assignment-Corrected Fallback Rerun", "",
             ("Fresh construction from the fixed 256 leaves and cached capability annotations. "
             "Claude Sonnet 4.6 primary summaries; GPT-4o-mini fallback after exhausted Claude retries. "
             "GPT-5.4-mini construction/reassignment and posthoc evaluation; MPNet; seed 42; two descendant leaves."),
             "", ("Mean (ordinary descriptive SE); nested parents are dependent. "
             "Specificity is the raw outsider score. Gaps are paired within parents with valid outsiders."),
             "", *table([row]), "",
             f"Average summary words: {words['mean']:.3f} ({words['stderr']:.3f}).", "",
             (f"Fallback events: {len(checks['provider_fallback_events'])}. "
             f"Node/level locations: {checks['fallback_node_levels']}. "
             "Events can include regeneration of the same pair; the event list is not a final-summary count."),
             "", (f"Extra assignment edges: {checks['extra_assignment_edges']}. "
             f"All {len(manifest['protected_sha256'])} protected files unchanged; "
             "zero descendant outsiders or score mismatches.")]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", json.dumps({"result": row, "audit": checks}), flush=True)


if __name__ == "__main__":
    main()
