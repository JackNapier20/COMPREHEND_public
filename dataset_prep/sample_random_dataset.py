"""Create reproducible, leakage-checked random samples from an instructions CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from dataset_prep.sample_wb_random import (
    active_summary_prompt,
    demonstration_texts,
    leakage_reason,
    normalize,
)
from common_summary_generation.prompt_variants import (
    CURRENT_VARIANT,
    VALID_VARIANTS,
    apply_summary_prompt_variant,
)

def source_ids(data: dict) -> set[str]:
    """Use dataset identities, never the tree-local 0..N node IDs."""
    config = data.get("config", {})
    return {
        str(value) for field in ("selected_ids", "selected_wildbench_ids", "selected_session_ids")
        for value in config.get(field, []) if value is not None and str(value).strip()
    }


def overlap_audit(sample: dict, protected: dict) -> dict:
    sample_prompts = [node.get("prompt") or "" for node in sample["levels"][0]["nodes"]]
    protected_prompts = [node.get("prompt") or "" for node in protected["levels"][0]["nodes"]]
    return {
        "exact_prompt_overlap": len(set(sample_prompts) & set(protected_prompts)),
        "normalized_prompt_overlap": len(
            {normalize(text) for text in sample_prompts} &
            {normalize(text) for text in protected_prompts}
        ),
        "source_id_overlap": len(source_ids(sample) & source_ids(protected)),
    }


def sample_dataset(
    source: Path,
    summary_source: Path,
    output: Path,
    *,
    dataset_name: str,
    count: int = 256,
    seed: int = 42,
    prompt_variant: str = CURRENT_VARIANT,
    exclude_trees: tuple[Path, ...] = (),
    overwrite: bool = False,
) -> dict:
    if count < 1:
        raise ValueError("Sample count must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Sample already exists; reuse it or explicitly overwrite: {output}")
    protected = []
    excluded_prompts = set()
    excluded_ids = set()
    for path in exclude_trees:
        raw = path.read_bytes()
        data = json.loads(raw)
        nodes = data["levels"][0]["nodes"]
        excluded_prompts.update(normalize(node.get("prompt") or "") for node in nodes)
        excluded_ids.update(source_ids(data))
        protected.append((path, hashlib.sha256(raw).hexdigest(), data))
    prompt = apply_summary_prompt_variant(
        active_summary_prompt(summary_source), prompt_variant
    )
    examples = demonstration_texts(prompt)

    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "instructions" not in rows[0]:
        raise ValueError(f"{source} does not contain an instructions column")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = []
    rejected = []
    seen = set()
    for row_index in order:
        row = rows[row_index]
        instruction = row.get("instructions", "")
        normalized = normalize(instruction)
        row_ids = {str(row[field]) for field in ("id", "session_id") if row.get(field)}
        if normalized in excluded_prompts or row_ids & excluded_ids:
            rejected.append({"row": row_index, "reason": "protected evaluation prompt/source ID"})
            continue
        if normalized in seen:
            rejected.append({"row": row_index, "reason": "duplicate instruction"})
            continue
        reason = leakage_reason(instruction, prompt, examples)
        if reason:
            rejected.append({"row": row_index, "reason": reason})
            continue
        seen.add(normalized)
        selected.append((row_index, row))
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"Only found {len(selected)} eligible prompts after scanning {len(rows)} rows"
        )

    id_fields = [field for field in ("id", "session_id", "source")
                 if field in rows[0]]
    config = {
        "dataset": dataset_name,
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_seed": seed,
        "sample_size": count,
        "summary_prompt_source": str(summary_source),
        "summary_prompt_variable": "prompt14",
        "summary_prompt_variant": prompt_variant,
        "summary_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "demonstration_text_count": len(examples),
        "rejected_before_completion": rejected,
        "selected_source_rows": [row_index for row_index, _ in selected],
    }
    for field in id_fields:
        config[f"selected_{field}s"] = [row.get(field) for _, row in selected]

    data = {
        "config": config,
        "levels": [{
            "level": 0,
            "nodes": [
                {"id": idx, "prompt": row["instructions"], "alive": True}
                for idx, (_, row) in enumerate(selected)
            ],
        }],
        "next_id": count,
    }
    config["excluded_trees"] = []
    for path, signature, protected_data in protected:
        audit = overlap_audit(data, protected_data)
        if any(audit.values()):
            raise RuntimeError(f"Sample overlaps protected evaluation data: {path}: {audit}")
        config["excluded_trees"].append({
            "path": str(path.resolve()), "sha256": signature,
            "leaf_count": len(protected_data["levels"][0]["nodes"]), **audit,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-source",
        type=Path,
        default=Path("common_summary_generation/summarize.py"),
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-variant", choices=VALID_VARIANTS,
                        default=CURRENT_VARIANT)
    parser.add_argument("--exclude", type=Path, nargs="+", default=[],
                        help="Protect every level-0 prompt and source ID in these trees")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data = sample_dataset(
        args.source,
        args.summary_source,
        args.output,
        dataset_name=args.dataset_name,
        count=args.count,
        seed=args.seed,
        prompt_variant=args.prompt_variant,
        exclude_trees=tuple(args.exclude),
        overwrite=args.overwrite,
    )
    config = data["config"]
    print(f"Wrote {config['sample_size']} prompts to {args.output}")
    print(f"Extracted {config['demonstration_text_count']} demonstration texts")
    print(f"Rejected before completion: {len(config['rejected_before_completion'])}")
    for excluded in config["excluded_trees"]:
        print(f"No overlap with {excluded['path']}: "
              f"exact={excluded['exact_prompt_overlap']}, "
              f"normalized={excluded['normalized_prompt_overlap']}, "
              f"source IDs={excluded['source_id_overlap']}")


if __name__ == "__main__":
    main()
