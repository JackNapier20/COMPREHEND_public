"""Build a strict Claude COMPREHEND teacher tree and retain reusable main ideas."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from dataset_prep.sample_random_dataset import overlap_audit
from dataset_prep.sample_wb_random import active_summary_prompt, demonstration_texts, leakage_reason, normalize
from tree import Tree


DEFAULT_PROTECTED = (
    Path("WB_new_random_256.json"), Path("UF_new_random_256.json"),
    Path("comparison_256_outputs/WB_current_k2/evaltree_capability_base.json"),
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_sample(source: Path, protected_paths: tuple[Path, ...], count: int = 256) -> dict:
    sample = json.loads(source.read_text(encoding="utf-8"))
    if len(sample["levels"]) != 1 or len(sample["levels"][0]["nodes"]) != count:
        raise ValueError(f"Expected a fresh {count}-leaf sample: {source}")
    nodes = sample["levels"][0]["nodes"]
    if len({node["prompt"] for node in nodes}) != count:
        raise ValueError("Training prompts must be unique")
    if len({normalize(node["prompt"]) for node in nodes}) != count:
        raise ValueError("Training prompts must be unique after normalization")
    prompt = active_summary_prompt(Path(__file__).resolve().parents[1] / "common_summary_generation/summarize.py")
    examples = demonstration_texts(prompt)
    leaks = [{"id": node["id"], "reason": reason} for node in nodes
             if (reason := leakage_reason(node["prompt"], prompt, examples))]
    if leaks:
        raise ValueError(f"Training prompts overlap summary demonstrations: {leaks}")
    results = []
    for path in protected_paths:
        protected = json.loads(path.read_text(encoding="utf-8"))
        result = {"path": str(path.resolve()), "sha256": file_hash(path),
                  **overlap_audit(sample, protected)}
        if any(result[key] for key in (
            "exact_prompt_overlap", "normalized_prompt_overlap", "source_id_overlap",
        )):
            raise ValueError(f"Training/evaluation overlap: {result}")
        results.append(result)
    return {"sample_sha256": file_hash(source), "sample_size": count, "protected_files": results,
            "demonstration_text_count": len(examples), "demonstration_overlap_count": len(leaks)}


def export_main_ideas(tree_path: Path, source: Path, output_dir: Path) -> dict:
    tree = Tree.load(tree_path)
    sample = json.loads(source.read_text(encoding="utf-8"))
    source_nodes = sample["levels"][0]["nodes"]
    sources = {node["id"]: index for index, node in enumerate(source_nodes)}
    source_prompts = {node["id"]: node["prompt"] for node in source_nodes}
    leaves = tree.levels[0].nodes
    if ({node.id for node in leaves} != set(sources)
            or any(node.prompt != source_prompts.get(node.id) for node in leaves)):
        raise ValueError("The built tree's leaves differ from its audited source")
    missing = [node.id for node in leaves if not node.mainIdea or not node.mainIdea.strip()]
    if missing:
        raise ValueError(f"Missing leaf main ideas: {missing}")
    source_config = sample["config"]
    sample_signature = file_hash(source)
    records = []
    for level in tree.levels:
        for node in level.nodes:
            if not node.mainIdea:
                continue
            record = {
                "node_id": node.id, "level": level.level, "is_leaf": level.level == 0,
                "prompt": node.prompt, "main_idea": node.mainIdea,
                "teacher_model": "claude-sonnet-4-6",
                "source_sample_sha256": sample_signature,
            }
            if level.level == 0:
                position = sources[node.id]
                for field, label in (("selected_ids", "wildbench_id"),
                                     ("selected_session_ids", "session_id"),
                                     ("selected_source_rows", "source_row")):
                    values = source_config.get(field, [])
                    if values:
                        record[label] = values[position]
            records.append(record)
    leaf_records = [record for record in records if record["is_leaf"]]
    for name, rows in (("main_ideas.jsonl", records), ("leaf_main_ideas.jsonl", leaf_records)):
        with (output_dir / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"stored_main_ideas": len(records), "leaf_main_ideas": len(leaf_records),
            "levels": len(tree.levels),
            "internal_nodes": sum(len(level.nodes) for level in tree.levels[1:]),
            "tree_sha256": file_hash(tree_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("finetune_v5_outputs/WB_claude"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=list(DEFAULT_PROTECTED))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    audit = audit_sample(args.source, tuple(args.exclude))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "sampling_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {audit['sample_size']} training prompts; all held-out overlap checks are zero.", flush=True)

    os.environ.update(BLOSSOM_PROVIDER="claude", SUMMARY_GENERATION_PROVIDER="claude",
                      EVALUATION_PROVIDER="openai", BLOSSOM_SUMMARY_PROMPT_VARIANT="current")
    os.environ.setdefault("MAIN_IDEA_MAX_WORKERS", "4")
    os.environ.setdefault("CLAUDE_SUMMARY_MAX_WORKERS", "4")
    os.environ.setdefault("INTERNAL_EVAL_MAX_WORKERS", "6")
    os.environ.setdefault("POSTHOC_MAX_WORKERS", "6")
    from experiments import evaltree_k_sensitivity as runner
    from experiments.reporting import collect_result, write_reports
    from main import API_KEYS, MODEL_CONFIG

    MODEL_CONFIG.update(blossom="claude", summary_generation="claude", evaluation="openai")
    config = runner._base_config(
        args.source, audit["sample_sha256"], audit["sample_size"], None,
        source_tree=Tree.load(args.source),
    )
    config.update(
        experiment="finetune_v5_teacher", annotation_model="raw_prompts",
        blossom_model="claude-sonnet-4-6", summary_model="claude-sonnet-4-6",
        strict_summary_provider=True, evaluation_judge="gpt-5.4-mini",
        provider_fallback_policy="Claude retries=3; no cross-provider generation fallback",
        sampling_provenance=json.loads(args.source.read_text(encoding="utf-8"))["config"],
    )
    if not args.export_only:
        if not API_KEYS["claude"] or not API_KEYS["openai"]:
            raise SystemExit("Configure ANTHROPIC_API_KEY and OPENAI_API_KEY in .env")
        trees = runner.build_trees(
            args.source, args.output_dir, (2,), config, False, False, True, None, args.samples,
        )
    else:
        trees = runner._discover_trees(args.output_dir, (2,), False, True)
        if not runner._is_complete(trees[0][2], {**config, "method": "blossom", "k": None}):
            raise SystemExit("Export-only requires a complete compatible teacher tree")
    tree_path = trees[0][2]
    tree = Tree.load(tree_path)
    if tree.config.get("provider_fallbacks"):
        raise RuntimeError("Teacher tree contains cross-provider fallback generations")
    exports = export_main_ideas(tree_path, args.source, args.output_dir)
    print(f"Stored {exports['leaf_main_ideas']} leaf main ideas and {exports['stored_main_ideas']} overall.", flush=True)
    if args.export_only:
        evaluated = [(method, k, runner._evaluation_path(path, args.output_dir, args.samples, args.seed))
                     for method, k, path in trees]
        if not all(runner._valid_evaluation(path) for _, _, path in evaluated):
            raise SystemExit("Export-only requires an existing complete evaluation; no API calls made")
    else:
        evaluated = runner.evaluate_trees(trees, args.output_dir, args.samples, args.seed, False)
    results = [collect_result(path, "comprehend", None) for _, _, path in evaluated]
    write_reports(results, args.output_dir, title="WB256 COMPREHEND Claude teacher results")
    for protected in audit["protected_files"]:
        if file_hash(Path(protected["path"])) != protected["sha256"]:
            raise RuntimeError(f"Protected evaluation artifact changed: {protected['path']}")
    manifest = {"status": "complete", "sampling_audit": audit, "exports": exports,
                "tree": str(tree_path.resolve()), "evaluated_tree": str(evaluated[0][2].resolve()),
                "providers": dict(MODEL_CONFIG), "strict_claude_teacher": True}
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Teacher tree, main ideas, evaluation and reports complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
