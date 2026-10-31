"""Create a reproducible WildBench sample excluding summary demonstrations."""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import json
import random
import re
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def active_summary_prompt(source_path: Path) -> str:
    """Extract the literal portion of the active ``prompt14`` f-string."""
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    value = None
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "prompt14"
               for target in node.targets):
            value = node.value
            break
    if not isinstance(value, ast.JoinedStr):
        raise ValueError(f"Could not find prompt14 f-string in {source_path}")

    pieces = []
    for part in value.values:
        if isinstance(part, ast.Constant):
            pieces.append(str(part.value))
        elif isinstance(part, ast.FormattedValue):
            pieces.append("{RUNTIME_PROMPT}")
    return "".join(pieces)


def demonstration_texts(prompt: str) -> list[str]:
    """Return every Prompt A, B, and C from the demonstration portion."""
    demonstrations = prompt.split("Now carry out this task", 1)[0]
    pattern = re.compile(
        r"Prompt A(?: \([^\n]+\))?:?\s*\n\"?(.*?)\"?\s*\n"
        r"Prompt B(?: \([^\n]+\))?:?\s*\n\"?(.*?)\"?\s*\nPrompt C",
        re.S,
    )
    examples = []
    for prompt_a, prompt_b in pattern.findall(demonstrations):
        examples.extend([prompt_a.strip().strip('"'), prompt_b.strip().strip('"')])
    prompt_c_pattern = re.compile(
        r"Prompt C(?: \([^\n]+\))?:?\s*\n(.*?)"
        r"(?=\n(?:\d+\.\s*\(|Example \d+|Now carry out))",
        re.S,
    )
    examples.extend(text.strip().strip('"')
                    for text in prompt_c_pattern.findall(demonstrations + "\nNow carry out"))
    if not examples:
        raise ValueError("No summary demonstrations were extracted")
    return examples


def ngrams(tokens: list[str], size: int = 4) -> set[tuple[str, ...]]:
    return {tuple(tokens[idx:idx + size])
            for idx in range(max(0, len(tokens) - size + 1))}


def leakage_reason(candidate: str, full_prompt: str,
                   examples: list[str]) -> str | None:
    """Detect exact examples and close modifications of example children."""
    candidate_norm = normalize(candidate)
    if not candidate_norm:
        return "empty prompt"

    demonstration_norm = normalize(full_prompt.split("Now carry out this task", 1)[0])
    if len(candidate_norm) >= 20 and candidate_norm in demonstration_norm:
        return "exact/substring match in the summary demonstrations"

    candidate_tokens = candidate_norm.split()
    candidate_grams = ngrams(candidate_tokens)
    for index, example in enumerate(examples, start=1):
        example_norm = normalize(example)
        sequence = difflib.SequenceMatcher(None, candidate_norm, example_norm).ratio()
        example_grams = ngrams(example_norm.split())
        shared_grams = (
            len(candidate_grams & example_grams) / min(len(candidate_grams), len(example_grams))
            if candidate_grams and example_grams else 0.0
        )
        if sequence >= 0.72:
            return f"near-verbatim match to demonstration child {index} (ratio={sequence:.3f})"
        if shared_grams >= 0.35:
            return f"shared phrase match to demonstration child {index} (4-gram={shared_grams:.3f})"
    return None


def sample_wildbench(source: Path, summary_source: Path, output: Path,
                     *, count: int = 64, seed: int = 42) -> dict:
    prompt = active_summary_prompt(summary_source)
    examples = demonstration_texts(prompt)

    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if "instructions" not in (rows[0] if rows else {}):
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

    data = {
        "config": {
            "dataset": "allenai/WildBench v2 test",
            "source_file": str(source),
            "sample_seed": seed,
            "sample_size": count,
            "summary_prompt_source": str(summary_source),
            "summary_prompt_variable": "prompt14",
            "demonstration_text_count": len(examples),
            "rejected_before_completion": rejected,
            "selected_source_rows": [row_index for row_index, _ in selected],
            "selected_wildbench_ids": [row.get("id") for _, row in selected],
            "selected_session_ids": [row.get("session_id") for _, row in selected],
        },
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
    parser.add_argument("--source", type=Path, default=Path("dataset_WB_with_scores.csv"))
    parser.add_argument(
        "--summary-source", type=Path,
        default=Path("common_summary_generation/summarize.py"),
    )
    parser.add_argument("--output", type=Path, default=Path("WB_new_random_64.json"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = sample_wildbench(
        args.source, args.summary_source, args.output,
        count=args.count, seed=args.seed,
    )
    config = data["config"]
    print(f"Wrote {config['sample_size']} prompts to {args.output}")
    print(f"Extracted {config['demonstration_text_count']} demonstration texts")
    print(f"Rejected before completion: {len(config['rejected_before_completion'])}")


if __name__ == "__main__":
    main()
