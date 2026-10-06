"""Generate weakness reports, predict leaf scores, and evaluate predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from openai import OpenAI


REPORT_SYSTEM = """You analyze an LLM's strengths and weaknesses from scores attached to internal nodes of a prompt hierarchy. Identify concrete capability patterns, score contrasts, failure conditions, and testable hypotheses. The report will be used to predict scores for unseen leaf prompts, so prioritize evidence that supports prediction. Do not discuss node IDs. Stay grounded in the supplied summaries and scores."""

REPORT_USER = """Hierarchy method: {method}
Evaluated model: {target}

Internal nodes and aggregate scores:
{nodes}

Write a structured weakness report of at most {max_words} words. Explain both strengths and weaknesses, important exceptions, and how the observed patterns should affect predictions for new prompts."""

PREDICT_SYSTEM = """You predict an evaluated LLM's performance on leaf prompts using a weakness report derived only from hierarchy-level evidence. Return valid JSON. Give each leaf an integer score within the requested range and a concise evidence-based reason. Do not omit or reorder leaves."""


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def levels(data: dict) -> list[dict]:
    return data.get("levels", [])


def score_value(node: dict, score_key: str):
    if score_key in node:
        return node.get(score_key)
    return (node.get("wb_scores") or {}).get(score_key)


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=VALUE")
    name, assigned = value.split("=", 1)
    if not name or not assigned:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=VALUE")
    return name, assigned


def extract_json(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    start = min((idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0), default=-1)
    if start < 0:
        raise ValueError("Model response did not contain JSON")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(candidate[start:])
    return parsed


def openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def generate_report(
    tree_path: Path,
    target: str,
    score_key: str,
    output_path: Path,
    *,
    model: str,
    max_words: int,
) -> Path:
    data = load_json(tree_path)
    method = (data.get("config") or {}).get("method", "unknown")
    payload = []
    for level in levels(data):
        if level.get("level", 0) == 0:
            continue
        for node in level.get("nodes", []):
            score = score_value(node, score_key)
            text = node.get("summary") or node.get("prompt")
            if text and score is not None:
                payload.append({"level": level.get("level"), "text": text, "score": score})
    if not payload:
        raise ValueError(
            f"No internal nodes contain score {score_key!r}; attach and aggregate scores first"
        )

    response = openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {
                "role": "user",
                "content": REPORT_USER.format(
                    method=method,
                    target=target,
                    nodes=json.dumps(payload, indent=2, ensure_ascii=False),
                    max_words=max_words,
                ),
            },
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.choices[0].message.content, encoding="utf-8")
    print(f"Saved weakness report to {output_path}")
    return output_path


def predict_leaf_scores(
    leaves_path: Path,
    reports: list[tuple[str, str]],
    target: str,
    output_path: Path,
    *,
    model: str,
    score_min: int,
    score_max: int,
) -> Path:
    data = load_json(leaves_path)
    leaf_level = next((level for level in levels(data) if level.get("level", 0) == 0), None)
    if leaf_level is None:
        raise ValueError("Input tree has no level-0 leaves")
    leaf_payload = [
        {"id": node.get("id"), "text": node.get("prompt") or node.get("summary") or ""}
        for node in leaf_level.get("nodes", [])
    ]

    merged = {
        node["id"]: {"id": node["id"], "text": node["text"], "predictions": {}}
        for node in leaf_payload
    }
    client = openai_client()
    for method, report_file in reports:
        report = Path(report_file).read_text(encoding="utf-8")
        user_prompt = (
            f"Evaluated model: {target}\nHierarchy method: {method}\n\n"
            f"Weakness report:\n{report}\n\n"
            f"Leaf prompts:\n{json.dumps(leaf_payload, indent=2, ensure_ascii=False)}\n\n"
            f"Use integer scores from {score_min} to {score_max}, inclusive.\n"
            "Return exactly this shape: "
            '{"predictions": [{"id": <original id>, "score": <integer>, '
            '"reason": "<brief reason>"}, ...]}'
        )
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PREDICT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = extract_json(response.choices[0].message.content)
        predictions = parsed.get("predictions", [])
        if len(predictions) != len(leaf_payload):
            raise ValueError(
                f"{method} returned {len(predictions)} predictions for {len(leaf_payload)} leaves"
            )
        for prediction in predictions:
            leaf_id = prediction.get("id")
            if leaf_id not in merged:
                raise ValueError(f"{method} returned unknown leaf id {leaf_id}")
            score = int(prediction["score"])
            if not score_min <= score <= score_max:
                raise ValueError(
                    f"Score for leaf {leaf_id} is outside {score_min}-{score_max}: {score}"
                )
            merged[leaf_id]["predictions"][method] = {
                "score": score,
                "reason": prediction.get("reason", ""),
            }

    output = {"target": target, "reports": dict(reports), "leaves": list(merged.values())}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved leaf predictions to {output_path}")
    return output_path


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def evaluate_predictions(
    predictions_path: Path,
    ground_truth_path: Path,
    score_key: str,
    output_path: Path | None = None,
) -> dict:
    predictions = load_json(predictions_path)
    truth_data = load_json(ground_truth_path)
    leaf_level = next(level for level in levels(truth_data) if level.get("level", 0) == 0)
    truth = {
        node["id"]: float(score_value(node, score_key))
        for node in leaf_level.get("nodes", [])
        if score_value(node, score_key) is not None
    }

    methods = sorted(
        {method for leaf in predictions["leaves"] for method in leaf["predictions"]}
    )
    results = {}
    for method in methods:
        pairs = [
            (truth[leaf["id"]], float(leaf["predictions"][method]["score"]))
            for leaf in predictions["leaves"]
            if leaf["id"] in truth and method in leaf["predictions"]
        ]
        if not pairs:
            continue
        actual, predicted = map(list, zip(*pairs))
        results[method] = {
            "n": len(pairs),
            "mae": sum(abs(p - t) for t, p in pairs) / len(pairs),
            "rmse": math.sqrt(sum((p - t) ** 2 for t, p in pairs) / len(pairs)),
            "exact_accuracy": sum(p == t for t, p in pairs) / len(pairs),
            "pearson": pearson(actual, predicted),
        }

    result = {"score_key": score_key, "methods": results}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Generate one weakness report")
    report_parser.add_argument("tree", type=Path)
    report_parser.add_argument("--target", required=True)
    report_parser.add_argument("--score-key", required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--max-words", type=int, default=1000)

    predict_parser = subparsers.add_parser("predict", help="Predict leaves from reports")
    predict_parser.add_argument("leaves", type=Path)
    predict_parser.add_argument("--target", required=True)
    predict_parser.add_argument(
        "--report", type=parse_assignment, action="append", required=True,
        metavar="METHOD=PATH",
    )
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--score-min", type=int, default=1)
    predict_parser.add_argument("--score-max", type=int, default=10)

    eval_parser = subparsers.add_parser("evaluate", help="Compare predictions to scores")
    eval_parser.add_argument("predictions", type=Path)
    eval_parser.add_argument("ground_truth", type=Path)
    eval_parser.add_argument("--score-key", required=True)
    eval_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "report":
        generate_report(
            args.tree,
            args.target,
            args.score_key,
            args.output,
            model=args.model,
            max_words=args.max_words,
        )
    elif args.command == "predict":
        predict_leaf_scores(
            args.leaves,
            args.report,
            args.target,
            args.output,
            model=args.model,
            score_min=args.score_min,
            score_max=args.score_max,
        )
    else:
        evaluate_predictions(
            args.predictions, args.ground_truth, args.score_key, args.output
        )


if __name__ == "__main__":
    main()
