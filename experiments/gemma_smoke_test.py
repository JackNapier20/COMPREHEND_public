"""Exercise real Gemma extraction and summary paths without writing artifacts."""

from __future__ import annotations

import argparse
import os

from common_summary_generation.prompt_variants import CURRENT_VARIANT
from common_summary_generation.summarize import generate_summary
from evaltree.summarize import generate_cluster_summary
from retrieval.blossom_matching import _extract_main_idea_llm
from tree import Level, Node, Tree


def _check_text(text: str | None, label: str) -> str:
    if (not text or text.strip().lower() in {"unknown capability", "unknown main idea"}
            or text.startswith("Error:")):
        raise RuntimeError(f"{label} returned a missing/error result: {text!r}")
    return text


def run_smoke_test(api_keys: dict, *, batch_size: int | None = None) -> dict:
    config = dict(api_keys["gemma"])
    size = batch_size if batch_size is not None else int(config.get("main_idea_batch_size", 1))
    if size < 1:
        raise ValueError("Main-idea batch size must be positive")
    config["concurrency"] = 1
    keys = {"gemma": config}
    tasks = ["Write Python code to read CSV files.", "Write Python code to filter CSV rows."]
    print(f"Endpoint: {config['base_url']}", flush=True)
    print(f"Model: {config['model']}; main-idea batch size: {size}", flush=True)
    print("Checking main-idea extraction...", flush=True)
    ideas = _extract_main_idea_llm(
        tasks, batch_size=size, llm_model_name="gemma", api_keys=keys,
    )
    if len(ideas) != len(tasks):
        raise RuntimeError("Main-idea output count does not match inputs")
    for index, idea in enumerate(ideas):
        print(f"Main idea {index + 1}: {_check_text(idea, 'Main idea')}", flush=True)

    print("Checking EvalTree summary generation...", flush=True)
    summary = _check_text(generate_cluster_summary(
        ["Reading tabular data using Python", "Filtering tabular data using Python"],
        model_name="gemma", api_keys=keys,
    ), "EvalTree summary")
    print(f"EvalTree summary: {summary}", flush=True)

    print("Checking COMPREHEND's current few-shot summary prompt...", flush=True)
    tree = Tree(levels=[Level(level=0, nodes=[
        Node(id=0, prompt=tasks[0], match=1, score=0.9, alive=True),
        Node(id=1, prompt=tasks[1], match=0, score=0.9, alive=False),
    ])], next_id=2)
    generate_summary(tree, 0, threshold=-1.0, model_name="gemma", api_keys=keys)
    common = _check_text(tree.levels[0].nodes[0].summary, "COMPREHEND summary")
    if common != tree.levels[0].nodes[1].summary:
        raise RuntimeError("Pair summaries were not assigned consistently")
    print(f"COMPREHEND summary: {common}", flush=True)
    print("Smoke checks completed. Inspect all outputs for coherence before a full run.", flush=True)
    return {"main_ideas": ideas, "evaltree_summary": summary, "comprehend_summary": common}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Overrides GEMMA_BASE_URL/.env")
    parser.add_argument("--model", help="Overrides GEMMA_MODEL/.env")
    parser.add_argument("--main-idea-batch-size", type=int, help="Defaults to configured size")
    args = parser.parse_args()
    if args.main_idea_batch_size is not None and args.main_idea_batch_size < 1:
        parser.error("--main-idea-batch-size must be positive")
    if args.base_url:
        os.environ["GEMMA_BASE_URL"] = args.base_url
    if args.model:
        os.environ["GEMMA_MODEL"] = args.model
    os.environ["BLOSSOM_SUMMARY_PROMPT_VARIANT"] = CURRENT_VARIANT
    from main import API_KEYS

    try:
        run_smoke_test(API_KEYS, batch_size=args.main_idea_batch_size)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Smoke check FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
