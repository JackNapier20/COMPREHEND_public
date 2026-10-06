"""Create reproducible, leakage-checked random samples from an instructions CSV."""

from __future__ import annotations

import argparse
import csv
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


def sample_dataset(
    source: Path,
    summary_source: Path,
    output: Path,
    *,
    dataset_name: str,
    count: int = 256,
    seed: int = 42,
    prompt_variant: str = CURRENT_VARIANT,
) -> dict:
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
        "sample_seed": seed,
        "sample_size": count,
        "summary_prompt_source": str(summary_source),
        "summary_prompt_variable": "prompt14",
        "summary_prompt_variant": prompt_variant,
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
    args = parser.parse_args()
    data = sample_dataset(
        args.source,
        args.summary_source,
        args.output,
        dataset_name=args.dataset_name,
        count=args.count,
        seed=args.seed,
        prompt_variant=args.prompt_variant,
    )
    config = data["config"]
    print(f"Wrote {config['sample_size']} prompts to {args.output}")
    print(f"Extracted {config['demonstration_text_count']} demonstration texts")
    print(f"Rejected before completion: {len(config['rejected_before_completion'])}")


if __name__ == "__main__":
    main()
