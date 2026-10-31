"""Diagnose why EvalTree's faithfulness-minus-specificity gap clusters near 0.2.

The runner is additive: it reads completed post-hoc EvalTree JSON artifacts,
caches every new provider request by content, and never modifies source trees.
It supports a fully offline audit as well as bounded candidate, summary, judge,
sampling, and annotation experiments.

Example:
    python -m experiments.specificity_diagnostics --phases all
    python -m experiments.specificity_diagnostics --report-only
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREES = {
    "WB": ROOT / "comparison_256_outputs/WB_current_k2/eval_outputs/"
    "evaltree_k2_eval_openai_n2_s42_a932fc4bd33e.json",
    "UF": ROOT / "comparison_256_outputs/UF_current_k2/eval_outputs/"
    "evaltree_k2_eval_openai_n2_s42_749f8f1846dd.json",
}
DEFAULT_OUTPUT = ROOT / "specificity_diagnostics_outputs"
ALL_PHASES = ("candidates", "summaries", "judge", "sampling", "annotations")

BOUNDARY_SYSTEM = (
    "You identify precise common capability categories. Return only one concise "
    "gerund phrase and do not add an explanation."
)
BOUNDARY_PROMPT = """Given these capability descriptions, write the most specific
single capability category that is supported by every description, or by the
largest coherent majority when no detail is shared by all of them.

Do not introduce a domain, task, constraint, or purpose that the descriptions
do not support. Avoid generic phrases such as \"applying expertise\", \"solving
problems\", \"creating content\", or \"analyzing information\" when a more
discriminating shared capability is available. Return one gerund phrase.

Descriptions:
{descriptions}

Specific common capability:"""

DEANCHORED_PROMPT = """Assess whether category B is supported by candidate A.
Consider only claims made by B. Omissions from B do not count against it.
Return JSON only as {{\"score\": NUMBER}}, where NUMBER is from 0 to 100 and is
the percentage of B's distinct claims that are present in or implied by A.

Candidate A:
{prompt_a}

Category B:
{prompt_b}
"""

BINARY_PROMPT = """Does candidate A belong under category B without category B
adding an unsupported domain, task, or constraint? Return JSON only as
{{\"applies\": true or false, \"confidence\": NUMBER}}, where confidence is from
0 to 1.

Candidate A:
{prompt_a}

Category B:
{prompt_b}
"""

INSTRUCTION_ONLY_ANNOTATION_PROMPT = """Given an instruction, summarize the
specific skill or capability needed to fulfill it effectively. Return one
gerund phrase.

Instruction: {instruction}

Capability:"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = _stable_hash([seed, *parts])[:16]
    return int(digest, 16)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _api_keys() -> dict[str, Any]:
    _load_env()
    return {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {
            "n": 0, "mean": None, "median": None, "q25": None, "q75": None,
            "std": None, "stderr": None, "min": None, "max": None,
        }
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
        "std": std,
        "stderr": std / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _trimmed_mean(values: Sequence[float], proportion: float = 0.1) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    trim = int(len(ordered) * proportion)
    kept = ordered[trim:len(ordered) - trim] if trim else ordered
    return statistics.mean(kept)


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = statistics.mean(first)
    second_mean = statistics.mean(second)
    first_delta = [value - first_mean for value in first]
    second_delta = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in first_delta)
        * sum(value * value for value in second_delta)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(first_delta, second_delta)) / denominator


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    return _pearson(_rank(first), _rank(second))


def _bootstrap_mean_ci(values: Sequence[float], seed: int = 42,
                       iterations: int = 2000) -> list[float | None]:
    values = list(values)
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _bootstrap_spearman_ci(first: Sequence[float], second: Sequence[float],
                           seed: int = 42,
                           iterations: int = 1000) -> list[float | None]:
    if len(first) != len(second) or len(first) < 3:
        return [None, None]
    rng = random.Random(seed)
    correlations = []
    for _ in range(iterations):
        indices = [rng.randrange(len(first)) for _ in first]
        correlation = _spearman(
            [first[index] for index in indices],
            [second[index] for index in indices],
        )
        if correlation is not None:
            correlations.append(correlation)
    return [_quantile(correlations, 0.025), _quantile(correlations, 0.975)]


def _clustered_stderr(rows: Sequence[dict[str, Any]], score_key: str,
                      cluster_key: str) -> float | None:
    usable = [row for row in rows if row.get(score_key) is not None
              and row.get(cluster_key) is not None]
    if len(usable) < 2:
        return None
    mean = statistics.mean(float(row[score_key]) for row in usable)
    clusters: dict[Any, list[float]] = defaultdict(list)
    for row in usable:
        clusters[row[cluster_key]].append(float(row[score_key]))
    if len(clusters) < 2:
        return None
    meat = sum(
        sum(value - mean for value in values) ** 2
        for values in clusters.values()
    )
    n = len(usable)
    correction = len(clusters) / (len(clusters) - 1)
    return math.sqrt(correction * meat / (n * n))


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    return _normalize([
        statistics.mean(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ])


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _token_jaccard(first: str, second: str) -> float:
    left = set(_normalized_text(first).split())
    right = set(_normalized_text(second).split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _extract_explicit_score(text: str) -> float | None:
    matches = re.findall(
        r"(?im)^\s*\**\s*score\b[\s:=*]*?(\d+(?:\.\d+)?)", text or ""
    )
    return float(matches[0]) if matches else None


def _extract_descendant_scores(text: str) -> list[tuple[int, float]]:
    pattern = re.compile(
        r"Child\s+(\d+):\s*\n.*?(?=\n\nChild\s+\d+:|\Z)", re.DOTALL
    )
    extracted = []
    for match in pattern.finditer(text or ""):
        score = _extract_explicit_score(match.group(0))
        if score is not None:
            extracted.append((int(match.group(1)), score))
    return extracted


@dataclass
class TreeArtifact:
    label: str
    path: Path
    payload: dict[str, Any]
    levels: list[dict[str, Any]] = field(init=False)
    by_id: dict[int, dict[str, Any]] = field(init=False)
    level_by_id: dict[int, int] = field(init=False)
    leaf_ids: list[int] = field(init=False)
    _leaf_memo: dict[int, tuple[int, ...]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.levels = self.payload.get("levels", [])
        self.by_id = {
            int(node["id"]): node
            for level in self.levels
            for node in level.get("nodes", [])
        }
        self.level_by_id = {
            int(node["id"]): int(level.get("level", 0))
            for level in self.levels
            for node in level.get("nodes", [])
        }
        self.leaf_ids = [int(node["id"]) for node in self.levels[0]["nodes"]]

    @classmethod
    def load(cls, label: str, path: Path) -> "TreeArtifact":
        return cls(label, path, json.loads(path.read_text(encoding="utf-8")))

    def descendants(self, node_id: int) -> tuple[int, ...]:
        if node_id in self._leaf_memo:
            return self._leaf_memo[node_id]
        node = self.by_id[node_id]
        children = [int(value) for value in node.get("children", [])]
        if not children:
            leaves = (node_id,)
        else:
            found = {
                leaf
                for child_id in children
                for leaf in self.descendants(child_id)
            }
            leaves = tuple(sorted(found))
        self._leaf_memo[node_id] = leaves
        return leaves

    def internal_nodes(self, *, scored_only: bool = True) -> list[dict[str, Any]]:
        nodes = [node for level in self.levels[1:] for node in level["nodes"]]
        if scored_only:
            nodes = [node for node in nodes
                     if node.get("specificity_faithfulness") is not None]
        return nodes

    def raw_text(self, leaf_id: int) -> str:
        return self.by_id[leaf_id].get("prompt") or ""

    def capability_text(self, leaf_id: int) -> str:
        node = self.by_id[leaf_id]
        return node.get("capability") or node.get("prompt") or ""

    def summary_text(self, node_id: int) -> str:
        node = self.by_id[node_id]
        return node.get("summary") or node.get("prompt") or ""

    def same_height_pool(self, node_id: int) -> list[int]:
        level_num = self.level_by_id[node_id]
        descendants = set(self.descendants(node_id))
        found = {
            leaf
            for peer in self.levels[level_num]["nodes"]
            if int(peer["id"]) != node_id
            for leaf in self.descendants(int(peer["id"]))
            if leaf not in descendants
        }
        return sorted(found)

    def true_sibling_pool(self, node_id: int) -> list[int]:
        node = self.by_id[node_id]
        found: set[int] = set()
        for parent_id in node.get("parent", []) or []:
            parent = self.by_id[int(parent_id)]
            for sibling_id in parent.get("children", []):
                sibling_id = int(sibling_id)
                if sibling_id != node_id:
                    found.update(self.descendants(sibling_id))
        found.difference_update(self.descendants(node_id))
        return sorted(found)

    def global_pool(self, node_id: int) -> list[int]:
        descendants = set(self.descendants(node_id))
        return [leaf_id for leaf_id in self.leaf_ids if leaf_id not in descendants]


class EmbeddingRegistry:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._spaces: dict[tuple[str, str, str], dict[int, list[float]]] = {}

    def _model(self, model_name: str) -> Any:
        if model_name not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]

    def space(self, artifact: TreeArtifact, representation: str,
              model_name: str) -> dict[int, list[float]]:
        key = (artifact.label, representation, model_name)
        if key not in self._spaces:
            if representation == "raw":
                texts = [artifact.raw_text(leaf_id) for leaf_id in artifact.leaf_ids]
            elif representation == "capability":
                texts = [artifact.capability_text(leaf_id)
                         for leaf_id in artifact.leaf_ids]
            else:
                raise ValueError(f"Unknown representation: {representation}")
            encoded = self._model(model_name).encode(texts, convert_to_numpy=True)
            self._spaces[key] = {
                leaf_id: _normalize(vector.tolist())
                for leaf_id, vector in zip(artifact.leaf_ids, encoded)
            }
        return self._spaces[key]

    def encode_texts(self, texts: Sequence[str], model_name: str) -> list[list[float]]:
        encoded = self._model(model_name).encode(list(texts), convert_to_numpy=True)
        return [_normalize(vector.tolist()) for vector in encoded]


def select_hard_candidate(
    artifact: TreeArtifact,
    node_id: int,
    pool: Sequence[int],
    vectors: dict[int, Sequence[float]],
) -> tuple[int | None, float | None]:
    descendants = [leaf_id for leaf_id in artifact.descendants(node_id)
                   if leaf_id in vectors]
    candidates = sorted(leaf_id for leaf_id in pool if leaf_id in vectors
                        and leaf_id not in descendants)
    if not descendants or not candidates:
        return None, None
    center = _centroid([vectors[leaf_id] for leaf_id in descendants])
    scored = [(_dot(vectors[leaf_id], center), leaf_id) for leaf_id in candidates]
    similarity, leaf_id = max(scored, key=lambda item: (item[0], -item[1]))
    return leaf_id, similarity


def select_distant_candidate(
    artifact: TreeArtifact,
    node_id: int,
    vectors: dict[int, Sequence[float]],
) -> tuple[int | None, float | None]:
    descendants = list(artifact.descendants(node_id))
    candidates = artifact.global_pool(node_id)
    if not descendants or not candidates:
        return None, None
    center = _centroid([vectors[leaf_id] for leaf_id in descendants])
    scored = [(_dot(vectors[leaf_id], center), leaf_id) for leaf_id in candidates]
    similarity, leaf_id = min(scored, key=lambda item: (item[0], item[1]))
    return leaf_id, similarity


def _level_band(level_num: int) -> str:
    if level_num <= 2:
        return "L1-2"
    if level_num <= 5:
        return "L3-5"
    return "L6+"


def select_causal_panel(artifact: TreeArtifact, seed: int = 42,
                        per_band: int = 8) -> list[int]:
    selected: list[int] = []
    for band in ("L1-2", "L3-5", "L6+"):
        nodes = [node for node in artifact.internal_nodes()
                 if _level_band(artifact.level_by_id[int(node["id"])]) == band]
        low = [node for node in nodes
               if float(node["specificity_faithfulness"]) <= 0.25]
        high = [node for node in nodes
                if float(node["specificity_faithfulness"]) > 0.25]
        rng = random.Random(_stable_seed(seed, artifact.label, band))
        rng.shuffle(low)
        rng.shuffle(high)
        high_count = min(per_band // 2, len(high))
        chosen = high[:high_count] + low[:per_band - high_count]
        if len(chosen) < per_band:
            remaining = [node for node in high[high_count:] + low[per_band-high_count:]
                         if node not in chosen]
            chosen.extend(remaining[:per_band - len(chosen)])
        selected.extend(int(node["id"]) for node in chosen)
    return selected


def select_summary_panel(artifact: TreeArtifact, causal_ids: Sequence[int],
                         per_band: int = 4) -> list[int]:
    selected = []
    for band in ("L1-2", "L3-5", "L6+"):
        candidates = [artifact.by_id[node_id] for node_id in causal_ids
                      if _level_band(artifact.level_by_id[node_id]) == band]
        candidates.sort(
            key=lambda node: (-float(node["specificity_faithfulness"]), int(node["id"]))
        )
        high = [node for node in candidates
                if float(node["specificity_faithfulness"]) > 0.25]
        low = [node for node in candidates
               if float(node["specificity_faithfulness"]) <= 0.25]
        chosen = high[:per_band // 2] + low[:per_band - min(per_band // 2, len(high))]
        if len(chosen) < per_band:
            leftovers = [node for node in candidates if node not in chosen]
            chosen.extend(leftovers[:per_band - len(chosen)])
        selected.extend(int(node["id"]) for node in chosen[:per_band])
    return selected


def select_sampling_panel(artifact: TreeArtifact, seed: int = 42,
                          count: int = 12) -> list[int]:
    candidates = [node for node in artifact.internal_nodes()
                  if len(artifact.descendants(int(node["id"]))) >= 5]
    mid = [node for node in candidates
           if _level_band(artifact.level_by_id[int(node["id"])]) == "L3-5"]
    high = [node for node in candidates
            if _level_band(artifact.level_by_id[int(node["id"])]) == "L6+"]
    rng = random.Random(_stable_seed(seed, artifact.label, "sampling"))
    rng.shuffle(mid)
    rng.shuffle(high)
    half = count // 2
    chosen = mid[:half] + high[:count-half]
    return [int(node["id"]) for node in chosen]


@dataclass
class CallCache:
    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CallCache":
        if not path.exists():
            return cls(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, payload.get("entries", {}))

    def successful(self, key: str) -> bool:
        return self.entries.get(key, {}).get("status") == "complete"

    def save(self) -> None:
        _atomic_json(self.path, {"version": 1, "entries": self.entries})


def _task_request(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task.get(key)
        for key in (
            "kind", "model", "prompt_type", "prompt_a", "prompt_b",
            "prompt_b_ref", "prompt_template_hash", "user_prompt",
            "system_prompt", "replicate",
        )
        if task.get(key) is not None
    }


def task_key(task: dict[str, Any]) -> str:
    return _stable_hash(_task_request(task))


def _legacy_task_key(task: dict[str, Any]) -> str:
    request = _task_request(task)
    request.pop("prompt_template_hash", None)
    return _stable_hash(request)


def migrate_legacy_cache(tasks: Sequence[dict[str, Any]], cache: CallCache) -> int:
    """Alias successful pre-template-hash entries created by early diagnostics."""
    migrated = 0
    for task in tasks:
        if cache.successful(task["key"]):
            continue
        legacy_key = _legacy_task_key(task)
        if legacy_key != task["key"] and cache.successful(legacy_key):
            cache.entries[task["key"]] = dict(cache.entries[legacy_key])
            migrated += 1
    if migrated:
        cache.save()
    return migrated


def _task(kind: str, model: str, condition: str, metadata: dict[str, Any],
          **request: Any) -> dict[str, Any]:
    task = {
        "kind": kind,
        "model": model,
        "condition": condition,
        "metadata": metadata,
        **request,
    }
    task["key"] = task_key(task)
    return task


def _current_faith_prompt() -> str:
    from evaluation.evaluate import FAITHFULNESS_PROMPT

    return FAITHFULNESS_PROMPT


def _format_judge_prompt(prompt_type: str, prompt_a: str, prompt_b: str) -> str:
    if prompt_type == "current":
        return _current_faith_prompt().format(prompt_a=prompt_a, prompt_b=prompt_b)
    if prompt_type == "deanchored":
        return DEANCHORED_PROMPT.format(prompt_a=prompt_a, prompt_b=prompt_b)
    if prompt_type == "binary":
        return BINARY_PROMPT.format(prompt_a=prompt_a, prompt_b=prompt_b)
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def _prompt_template_hash(prompt_type: str) -> str:
    if prompt_type == "current":
        template = _current_faith_prompt()
    elif prompt_type == "deanchored":
        template = DEANCHORED_PROMPT
    elif prompt_type == "binary":
        template = BINARY_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return _stable_hash(template)


def _parse_task_score(prompt_type: str, text: str) -> float:
    if prompt_type == "current":
        from evaluation.simple_eval import safe_extract_score

        return float(safe_extract_score(text))
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"Expected JSON response, got: {text[:160]!r}")
    payload = json.loads(match.group(0))
    if prompt_type == "deanchored":
        return max(0.0, min(1.0, float(payload["score"]) / 100.0))
    applies = payload["applies"]
    if isinstance(applies, str):
        applies = applies.strip().lower() == "true"
    confidence = max(0.0, min(1.0, float(payload["confidence"])))
    return confidence if applies else 1.0 - confidence


def execute_tasks(tasks: Sequence[dict[str, Any]], cache: CallCache,
                  max_calls: int, overwrite: bool = False) -> None:
    unique: dict[str, dict[str, Any]] = {}
    for task in tasks:
        unique.setdefault(task["key"], task)
    pending = [task for key, task in unique.items()
               if overwrite or not cache.successful(key)]
    print(f"Preflight: {len(unique)} unique logical tasks, "
          f"{len(pending)} uncached provider calls")
    if len(pending) > max_calls:
        raise RuntimeError(
            f"Refusing {len(pending)} uncached calls; --max-api-calls={max_calls}"
        )
    keys = _api_keys()
    for index, task in enumerate(pending, start=1):
        key = task["key"]
        try:
            if task["kind"] == "generate":
                from evaltree.llm import generate_text

                text = generate_text(
                    task["user_prompt"], task["system_prompt"],
                    model_name=task["model"], api_keys=keys,
                ).strip().strip('"')
                result = {"status": "complete", "text": text, "score": None}
            elif task["kind"] == "judge":
                from evaluation.simple_eval import _judge_call

                prompt_b = task.get("prompt_b")
                if prompt_b is None:
                    dependency = cache.entries.get(task["prompt_b_ref"], {})
                    if dependency.get("status") != "complete":
                        raise RuntimeError(
                            f"Missing generated dependency {task['prompt_b_ref']}"
                        )
                    prompt_b = dependency["text"]
                user_prompt = _format_judge_prompt(
                    task["prompt_type"], task["prompt_a"], prompt_b
                )
                text = _judge_call(user_prompt, model_name=task["model"], api_keys=keys)
                result = {
                    "status": "complete",
                    "text": text,
                    "score": _parse_task_score(task["prompt_type"], text),
                }
            else:
                raise ValueError(f"Unknown task kind: {task['kind']}")
            cache.entries[key] = result
        except Exception as exc:
            cache.entries[key] = {"status": "error", "error": repr(exc)}
            cache.save()
            raise
        cache.save()
        if index % 10 == 0 or index == len(pending):
            print(f"Completed {index}/{len(pending)} uncached calls")


def _judge_task(
    artifact: TreeArtifact,
    node_id: int,
    condition: str,
    prompt_a: str,
    prompt_b: str | None,
    candidate_id: int | None,
    *,
    model: str = "openai",
    prompt_type: str = "current",
    prompt_b_ref: str | None = None,
    metric: str = "outsider_faithfulness",
    replicate: int | None = None,
) -> dict[str, Any]:
    metadata = {
        "dataset": artifact.label,
        "node_id": node_id,
        "level": artifact.level_by_id[node_id],
        "condition": condition,
        "metric": metric,
        "candidate_id": candidate_id,
    }
    return _task(
        "judge", model, condition, metadata,
        prompt_type=prompt_type, prompt_a=prompt_a, prompt_b=prompt_b,
        prompt_b_ref=prompt_b_ref,
        prompt_template_hash=_prompt_template_hash(prompt_type),
        replicate=replicate,
    )


def _candidate_record(node_id: int, policy: str, candidate_id: int | None,
                      similarity: float | None) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "policy": policy,
        "candidate_id": candidate_id,
        "similarity": similarity,
    }


def compute_candidates(
    artifact: TreeArtifact, registry: EmbeddingRegistry, seed: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    raw_mpnet = registry.space(artifact, "raw", "all-mpnet-base-v2")
    cap_minilm = registry.space(artifact, "capability", "all-MiniLM-L6-v2")
    raw_minilm = registry.space(artifact, "raw", "all-MiniLM-L6-v2")
    cap_mpnet = registry.space(artifact, "capability", "all-mpnet-base-v2")
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for node in artifact.internal_nodes():
        node_id = int(node["id"])
        global_pool = artifact.global_pool(node_id)
        same_height = artifact.same_height_pool(node_id) or global_pool
        true_sibling = artifact.true_sibling_pool(node_id)
        policies = {
            "current_same_height_raw_mpnet": (same_height, raw_mpnet),
            "same_height_capability_minilm": (same_height, cap_minilm),
            "same_height_raw_minilm": (same_height, raw_minilm),
            "same_height_capability_mpnet": (same_height, cap_mpnet),
            "true_sibling_raw_mpnet": (true_sibling, raw_mpnet),
            "true_sibling_capability_minilm": (true_sibling, cap_minilm),
            "global_raw_mpnet": (global_pool, raw_mpnet),
        }
        node_result: dict[str, dict[str, Any]] = {}
        for policy, (pool, vectors) in policies.items():
            candidate_id, similarity = select_hard_candidate(
                artifact, node_id, pool, vectors
            )
            node_result[policy] = _candidate_record(
                node_id, policy, candidate_id, similarity
            )
        rng = random.Random(_stable_seed(seed, artifact.label, node_id, "random"))
        random_ids = rng.sample(global_pool, min(3, len(global_pool)))
        for index, candidate_id in enumerate(random_ids, start=1):
            policy = f"random_global_{index}"
            node_result[policy] = _candidate_record(
                node_id, policy, candidate_id, None
            )
        distant_id, distant_similarity = select_distant_candidate(
            artifact, node_id, raw_mpnet
        )
        node_result["distant_global_raw_mpnet"] = _candidate_record(
            node_id, "distant_global_raw_mpnet", distant_id, distant_similarity
        )
        result[node_id] = node_result
    return result


def build_candidate_tasks(
    artifacts: Sequence[TreeArtifact], panels: dict[str, list[int]],
    candidates: dict[str, dict[int, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    tasks = []
    primary = (
        "same_height_capability_minilm",
        "true_sibling_raw_mpnet",
        "true_sibling_capability_minilm",
        "random_global_1",
        "random_global_2",
        "random_global_3",
    )
    decomposition = ("same_height_raw_minilm", "same_height_capability_mpnet")
    for artifact in artifacts:
        panel = panels[artifact.label]
        subpanel = panel[:12]
        for node_id in panel:
            summary = artifact.summary_text(node_id)
            for policy in primary + (decomposition if node_id in subpanel else ()):
                candidate_id = candidates[artifact.label][node_id][policy]["candidate_id"]
                if candidate_id is None:
                    continue
                tasks.append(_judge_task(
                    artifact, node_id, policy, artifact.raw_text(candidate_id),
                    summary, candidate_id,
                ))
    return tasks


def _farthest_first(ids: Sequence[int], vectors: dict[int, Sequence[float]],
                    limit: int) -> list[int]:
    ids = sorted(ids)
    if len(ids) <= limit:
        return ids
    center = _centroid([vectors[node_id] for node_id in ids])
    first = max(ids, key=lambda node_id: (_dot(vectors[node_id], center), -node_id))
    chosen = [first]
    remaining = set(ids) - {first}
    while remaining and len(chosen) < limit:
        pick = min(
            remaining,
            key=lambda node_id: (
                max(_dot(vectors[node_id], vectors[chosen_id])
                    for chosen_id in chosen),
                node_id,
            ),
        )
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _summary_generation_task(
    artifact: TreeArtifact,
    node_id: int,
    input_mode: str,
    prompt_mode: str,
    descriptions: Sequence[str],
) -> dict[str, Any]:
    description_block = "\n".join(f"- {value}" for value in descriptions)
    if prompt_mode == "current":
        from evaltree.prompts import SUMMARY_PROMPT, SUMMARY_SYSTEM

        user_prompt = SUMMARY_PROMPT.format(descriptions=description_block)
        system_prompt = (
            SUMMARY_SYSTEM + "\nCRITICAL: DO NOT copy the input descriptions "
            "verbatim. You must synthesize them into a new gerund phrase."
        )
    else:
        user_prompt = BOUNDARY_PROMPT.format(descriptions=description_block)
        system_prompt = BOUNDARY_SYSTEM
    condition = f"summary_{input_mode}_{prompt_mode}"
    return _task(
        "generate", "claude", condition,
        {
            "dataset": artifact.label,
            "node_id": node_id,
            "level": artifact.level_by_id[node_id],
            "condition": condition,
            "metric": "summary_generation",
            "candidate_id": None,
        },
        user_prompt=user_prompt, system_prompt=system_prompt,
    )


def build_summary_tasks(
    artifacts: Sequence[TreeArtifact], summary_panels: dict[str, list[int]],
    candidates: dict[str, dict[int, dict[str, dict[str, Any]]]],
    registry: EmbeddingRegistry,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for artifact in artifacts:
        vectors = registry.space(artifact, "capability", "all-MiniLM-L6-v2")
        for node_id in summary_panels[artifact.label]:
            node = artifact.by_id[node_id]
            child_descriptions = []
            for child_id in node.get("children", []):
                child = artifact.by_id[int(child_id)]
                text = child.get("summary") or child.get("capability") or child.get("prompt")
                if text:
                    child_descriptions.append(text)
            descendant_ids = _farthest_first(
                artifact.descendants(node_id), vectors, limit=64
            )
            direct_descriptions = [artifact.capability_text(leaf_id)
                                   for leaf_id in descendant_ids]
            generation_tasks = [
                _summary_generation_task(
                    artifact, node_id, "recursive", "boundary", child_descriptions
                ),
                _summary_generation_task(
                    artifact, node_id, "direct", "current", direct_descriptions
                ),
                _summary_generation_task(
                    artifact, node_id, "direct", "boundary", direct_descriptions
                ),
            ]
            tasks.extend(generation_tasks)
            stored_descendants = _extract_descendant_scores(
                node.get("evaluation_faithfulness") or ""
            )
            descendant_ids_for_judging = [leaf_id for leaf_id, _ in stored_descendants]
            if len(descendant_ids_for_judging) < 2:
                descendant_ids_for_judging = list(artifact.descendants(node_id))[:2]
            outsider_policies = (
                "current_same_height_raw_mpnet",
                "true_sibling_capability_minilm",
            )
            for generation_task in generation_tasks:
                summary_ref = generation_task["key"]
                condition = generation_task["condition"]
                for leaf_id in descendant_ids_for_judging[:2]:
                    tasks.append(_judge_task(
                        artifact, node_id, condition, artifact.raw_text(leaf_id),
                        None, leaf_id, prompt_b_ref=summary_ref,
                        metric="descendant_faithfulness",
                    ))
                for policy in outsider_policies:
                    candidate_id = candidates[artifact.label][node_id][policy]["candidate_id"]
                    if candidate_id is None:
                        continue
                    tasks.append(_judge_task(
                        artifact, node_id, f"{condition}__{policy}",
                        artifact.raw_text(candidate_id), None, candidate_id,
                        prompt_b_ref=summary_ref,
                    ))
    return tasks


def select_calibration_panel(artifacts: Sequence[TreeArtifact],
                             count_per_band: int = 8) -> list[tuple[TreeArtifact, int]]:
    bands = (
        ("low", lambda value: value <= 0.10),
        ("middle", lambda value: 0.10 < value < 0.60),
        ("high", lambda value: value >= 0.60),
    )
    selected = []
    for band_name, predicate in bands:
        per_dataset = count_per_band // len(artifacts)
        for artifact in artifacts:
            nodes = [node for node in artifact.internal_nodes()
                     if predicate(float(node["specificity_faithfulness"]))]
            nodes.sort(key=lambda node: (
                _stable_seed(42, "calibration", artifact.label, band_name,
                             int(node["id"])),
                int(node["id"]),
            ))
            selected.extend((artifact, int(node["id"]))
                            for node in nodes[:per_dataset])
    return selected


def build_judge_tasks(
    artifacts: Sequence[TreeArtifact],
    candidates: dict[str, dict[int, dict[str, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    panel = select_calibration_panel(artifacts)
    tasks: list[dict[str, Any]] = []
    for artifact, node_id in panel:
        node = artifact.by_id[node_id]
        candidate_id = int(node["third_leaf"])
        prompt_a = artifact.raw_text(candidate_id)
        prompt_b = artifact.summary_text(node_id)
        for replicate in (1, 2):
            tasks.append(_judge_task(
                artifact, node_id, f"judge_current_repeat_{replicate}",
                prompt_a, prompt_b, candidate_id, replicate=replicate,
            ))
        tasks.append(_judge_task(
            artifact, node_id, "judge_deanchored", prompt_a, prompt_b,
            candidate_id, prompt_type="deanchored",
        ))
        tasks.append(_judge_task(
            artifact, node_id, "judge_binary", prompt_a, prompt_b,
            candidate_id, prompt_type="binary",
        ))
        tasks.append(_judge_task(
            artifact, node_id, "judge_claude_current", prompt_a, prompt_b,
            candidate_id, model="claude",
        ))

    for artifact in artifacts:
        artifact_panel = [node_id for current, node_id in panel
                          if current.label == artifact.label][:6]
        for node_id in artifact_panel:
            summary = artifact.summary_text(node_id)
            distant_id = candidates[artifact.label][node_id][
                "distant_global_raw_mpnet"
            ]["candidate_id"]
            controls = [("identity_control", summary, node_id)]
            if distant_id is not None:
                controls.append((
                    "distant_control", artifact.raw_text(distant_id), distant_id
                ))
            for condition, prompt_a, candidate_id in controls:
                for model in ("openai", "claude"):
                    tasks.append(_judge_task(
                        artifact, node_id, f"{condition}_{model}", prompt_a,
                        summary, candidate_id, model=model,
                    ))
    return tasks, [(artifact.label, node_id) for artifact, node_id in panel]


def build_sampling_tasks(
    artifacts: Sequence[TreeArtifact], sampling_panels: dict[str, list[int]],
    seed: int,
) -> list[dict[str, Any]]:
    tasks = []
    for artifact in artifacts:
        for node_id in sampling_panels[artifact.label]:
            node = artifact.by_id[node_id]
            existing = {leaf_id for leaf_id, _ in _extract_descendant_scores(
                node.get("evaluation_faithfulness") or ""
            )}
            available = [leaf_id for leaf_id in artifact.descendants(node_id)
                         if leaf_id not in existing]
            rng = random.Random(_stable_seed(seed, artifact.label, node_id, "extra"))
            selected = rng.sample(available, min(3, len(available)))
            for index, leaf_id in enumerate(selected, start=1):
                tasks.append(_judge_task(
                    artifact, node_id, f"extra_descendant_{index}",
                    artifact.raw_text(leaf_id), artifact.summary_text(node_id),
                    leaf_id, metric="descendant_faithfulness",
                ))
    return tasks


def select_annotation_leaves(artifact: TreeArtifact, seed: int,
                             count: int = 12) -> list[int]:
    frequencies = Counter(
        int(node["third_leaf"]) for node in artifact.internal_nodes()
        if node.get("third_leaf") is not None
    )
    frequent = [leaf_id for leaf_id, _ in sorted(
        frequencies.items(), key=lambda item: (-item[1], item[0])
    )[:count // 2]]
    remaining = [leaf_id for leaf_id in artifact.leaf_ids if leaf_id not in frequent]
    rng = random.Random(_stable_seed(seed, artifact.label, "annotations"))
    random_ids = rng.sample(remaining, count - len(frequent))
    return frequent + random_ids


def build_annotation_tasks(
    artifacts: Sequence[TreeArtifact], annotation_leaves: dict[str, list[int]],
) -> list[dict[str, Any]]:
    from evaltree.prompts import ANNOTATION_PROMPT, ANNOTATION_SYSTEM

    tasks = []
    for artifact in artifacts:
        for leaf_id in annotation_leaves[artifact.label]:
            instruction = artifact.raw_text(leaf_id)
            metadata = {
                "dataset": artifact.label,
                "node_id": leaf_id,
                "level": 0,
                "metric": "capability_annotation",
                "candidate_id": None,
            }
            for condition, user_prompt in (
                ("annotation_current_repeat", ANNOTATION_PROMPT.format(
                    instruction=instruction, response="N/A"
                )),
                ("annotation_instruction_only", INSTRUCTION_ONLY_ANNOTATION_PROMPT.format(
                    instruction=instruction
                )),
            ):
                tasks.append(_task(
                    "generate", "claude", condition,
                    {**metadata, "condition": condition},
                    user_prompt=user_prompt, system_prompt=ANNOTATION_SYSTEM,
                ))
    return tasks


def audit_artifact(
    artifact: TreeArtifact,
    candidate_map: dict[int, dict[str, dict[str, Any]]],
    registry: EmbeddingRegistry,
) -> dict[str, Any]:
    cap_vectors = registry.space(artifact, "capability", "all-MiniLM-L6-v2")
    rows = []
    parser_mismatches = 0
    parser_missing = 0
    descendant_leaks = 0
    current_candidate_mismatches = 0
    for node in artifact.internal_nodes():
        node_id = int(node["id"])
        score = float(node["specificity_faithfulness"])
        explicit = _extract_explicit_score(
            node.get("evaluation_specificity_faithfulness") or ""
        )
        if explicit is None:
            parser_missing += 1
        elif not math.isclose(explicit, score, abs_tol=1e-12):
            parser_mismatches += 1
        outsider_id = int(node["third_leaf"])
        descendants = artifact.descendants(node_id)
        if outsider_id in descendants:
            descendant_leaks += 1
        current_candidate = candidate_map[node_id][
            "current_same_height_raw_mpnet"
        ]["candidate_id"]
        if current_candidate != outsider_id:
            current_candidate_mismatches += 1
        true_sibling = outsider_id in artifact.true_sibling_pool(node_id)
        center = _centroid([cap_vectors[leaf_id] for leaf_id in descendants])
        inside_similarities = [_dot(cap_vectors[leaf_id], center)
                               for leaf_id in descendants]
        aligned = candidate_map[node_id]["true_sibling_capability_minilm"]
        aligned_similarity = aligned.get("similarity")
        embedding_margin = None
        if inside_similarities and aligned_similarity is not None:
            embedding_margin = statistics.mean(inside_similarities) - aligned_similarity
        outsider_text = artifact.raw_text(outsider_id)
        lexical_similarities = [
            _token_jaccard(outsider_text, artifact.raw_text(leaf_id))
            for leaf_id in descendants
        ]
        max_lexical = max(lexical_similarities, default=0.0)
        normalized_outsider = _normalized_text(outsider_text)
        exact_duplicate = any(
            normalized_outsider == _normalized_text(artifact.raw_text(leaf_id))
            for leaf_id in descendants
        )
        faith = node.get("faithfulness1")
        row = {
            "dataset": artifact.label,
            "node_id": node_id,
            "level": artifact.level_by_id[node_id],
            "level_band": _level_band(artifact.level_by_id[node_id]),
            "cluster_size": len(descendants),
            "summary_words": len(artifact.summary_text(node_id).split()),
            "faithfulness": float(faith) if faith is not None else None,
            "specificity": score,
            "gap": float(faith) - score if faith is not None else None,
            "outsider_id": outsider_id,
            "outsider_similarity": node.get("third_leaf_sim"),
            "true_sibling": true_sibling,
            "embedding_margin": embedding_margin,
            "max_lexical_jaccard": max_lexical,
            "exact_descendant_duplicate": exact_duplicate,
            "near_descendant_duplicate": max_lexical >= 0.8,
        }
        rows.append(row)

    specificity_values = [row["specificity"] for row in rows]
    faithfulness_values = [row["faithfulness"] for row in rows
                           if row["faithfulness"] is not None]
    gap_values = [row["gap"] for row in rows if row["gap"] is not None]
    specificity_level_means = []
    faithfulness_level_means = []
    gap_level_means = []
    by_level = {}
    for level in sorted({row["level"] for row in rows}):
        level_rows = [row for row in rows if row["level"] == level]
        level_specificity = [row["specificity"] for row in level_rows]
        level_faithfulness = [row["faithfulness"] for row in level_rows
                              if row["faithfulness"] is not None]
        level_gaps = [row["gap"] for row in level_rows if row["gap"] is not None]
        by_level[str(level)] = {
            "specificity": _summary(level_specificity),
            "faithfulness": _summary(level_faithfulness),
            "gap": _summary(level_gaps),
        }
        specificity_level_means.append(statistics.mean(level_specificity))
        if level_faithfulness:
            faithfulness_level_means.append(statistics.mean(level_faithfulness))
        if level_gaps:
            gap_level_means.append(statistics.mean(level_gaps))
    specificity_summary = _summary(specificity_values)
    faithfulness_summary = _summary(faithfulness_values)
    gap_summary = _summary(gap_values)
    specificity_clustered = _clustered_stderr(
        rows, "specificity", "outsider_id"
    )
    gap_clustered = _clustered_stderr(rows, "gap", "outsider_id")
    features = (
        "level", "cluster_size", "summary_words", "outsider_similarity",
        "embedding_margin", "max_lexical_jaccard",
    )
    correlations = {}
    for feature in features:
        usable = [row for row in rows if row.get(feature) is not None]
        feature_values = [float(row[feature]) for row in usable]
        feature_specificity_values = [row["specificity"] for row in usable]
        gap_feature_values = [float(row[feature]) for row in usable
                              if row.get("gap") is not None]
        feature_gap_values = [row["gap"] for row in usable
                              if row.get("gap") is not None]
        correlations[feature] = {
            "specificity_spearman": _spearman(
                feature_values, feature_specificity_values,
            ),
            "specificity_ci95": _bootstrap_spearman_ci(
                feature_values, feature_specificity_values,
                seed=_stable_seed(42, artifact.label, feature, "specificity"),
            ),
            "gap_spearman": _spearman(
                gap_feature_values, feature_gap_values,
            ),
            "gap_ci95": _bootstrap_spearman_ci(
                gap_feature_values, feature_gap_values,
                seed=_stable_seed(42, artifact.label, feature, "gap"),
            ),
        }
    policy_overlap = {}
    baseline_policy = "current_same_height_raw_mpnet"
    policies = sorted(next(iter(candidate_map.values())).keys())
    for policy in policies:
        compared = [
            candidate_map[node_id][policy]["candidate_id"]
            == candidate_map[node_id][baseline_policy]["candidate_id"]
            for node_id in candidate_map
            if policy in candidate_map[node_id]
        ]
        policy_overlap[policy] = statistics.mean(compared) if compared else None

    outsider_counts = Counter(row["outsider_id"] for row in rows)
    return {
        "tree_path": str(artifact.path.resolve()),
        "tree_sha256": _sha256_path(artifact.path),
        "integrity": {
            "evaluated_nodes": len(rows),
            "parser_mismatches": parser_mismatches,
            "parser_missing": parser_missing,
            "descendant_leaks": descendant_leaks,
            "current_candidate_mismatches": current_candidate_mismatches,
        },
        "distribution": {
            **specificity_summary,
            "trimmed_mean_10pct": _trimmed_mean(specificity_values),
            "level_macro_mean": statistics.mean(specificity_level_means),
            "fraction_le_025": statistics.mean(
                value <= 0.25 for value in specificity_values
            ),
            "fraction_ge_060": statistics.mean(
                value >= 0.60 for value in specificity_values
            ),
            "fraction_exact_zero": statistics.mean(
                value == 0 for value in specificity_values
            ),
            "outsider_clustered_stderr": specificity_clustered,
            "clustered_to_naive_se_ratio": (
                specificity_clustered / specificity_summary["stderr"]
                if specificity_clustered is not None
                and specificity_summary["stderr"] else None
            ),
        },
        "faithfulness_distribution": {
            **faithfulness_summary,
            "trimmed_mean_10pct": _trimmed_mean(faithfulness_values),
            "level_macro_mean": statistics.mean(faithfulness_level_means),
        },
        "gap_distribution": {
            **gap_summary,
            "mean_ci95": _bootstrap_mean_ci(
                gap_values, seed=_stable_seed(42, artifact.label, "gap_mean")
            ),
            "trimmed_mean_10pct": _trimmed_mean(gap_values),
            "level_macro_mean": statistics.mean(gap_level_means),
            "fraction_positive": statistics.mean(value > 0 for value in gap_values),
            "fraction_negative": statistics.mean(value < 0 for value in gap_values),
            "fraction_gt_020": statistics.mean(value > 0.20 for value in gap_values),
            "outsider_clustered_stderr": gap_clustered,
            "clustered_to_naive_se_ratio": (
                gap_clustered / gap_summary["stderr"]
                if gap_clustered is not None and gap_summary["stderr"] else None
            ),
        },
        "by_level": by_level,
        "correlations": correlations,
        "candidate_overlap_with_current": policy_overlap,
        "outsider_reuse": {
            "unique": len(outsider_counts),
            "max": max(outsider_counts.values(), default=0),
            "most_common": outsider_counts.most_common(10),
        },
        "duplicate_rates": {
            "exact": statistics.mean(row["exact_descendant_duplicate"] for row in rows),
            "near_jaccard_080": statistics.mean(
                row["near_descendant_duplicate"] for row in rows
            ),
        },
        "rows": rows,
    }


def task_rows(tasks: Sequence[dict[str, Any]], cache: CallCache) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        result = cache.entries.get(task["key"], {})
        row = {
            **task.get("metadata", {}),
            "task_key": task["key"],
            "kind": task["kind"],
            "model": task["model"],
            "prompt_type": task.get("prompt_type"),
            "status": result.get("status", "pending"),
            "score": result.get("score"),
            "raw_response": result.get("text"),
            "error": result.get("error"),
        }
        rows.append(row)
    return rows


def _task_score_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], float]:
    index = {}
    for row in rows:
        if row.get("score") is not None:
            index[(row["dataset"], int(row["node_id"]), row["condition"])] = float(
                row["score"]
            )
    return index


def _paired_effect(
    baseline: dict[tuple[str, int], float],
    comparison: dict[tuple[str, int], float],
    seed: int = 42,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(comparison))
    differences = [comparison[key] - baseline[key] for key in keys]
    return {
        "n": len(keys),
        "baseline_mean": _mean([baseline[key] for key in keys]),
        "comparison_mean": _mean([comparison[key] for key in keys]),
        "mean_difference": _mean(differences),
        "mean_absolute_difference": _mean([abs(value) for value in differences]),
        "difference_ci95": _bootstrap_mean_ci(differences, seed),
        "spearman": _spearman(
            [baseline[key] for key in keys], [comparison[key] for key in keys]
        ),
    }


def analyze_candidates(
    artifacts: Sequence[TreeArtifact], panels: dict[str, list[int]],
    candidates: dict[str, dict[int, dict[str, dict[str, Any]]]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    task_scores = _task_score_index(rows)
    baseline = {}
    for artifact in artifacts:
        for node_id in panels[artifact.label]:
            baseline[(artifact.label, node_id)] = float(
                artifact.by_id[node_id]["specificity_faithfulness"]
            )
    conditions = (
        "same_height_capability_minilm",
        "true_sibling_raw_mpnet",
        "true_sibling_capability_minilm",
        "same_height_raw_minilm",
        "same_height_capability_mpnet",
    )
    effects = {}
    for condition in conditions:
        compared = {
            key: score for (dataset, node_id, current), score in task_scores.items()
            if current == condition and (key := (dataset, node_id)) in baseline
        }
        effects[condition] = _paired_effect(baseline, compared)
    random_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (dataset, node_id, condition), score in task_scores.items():
        if condition.startswith("random_global_"):
            random_scores[(dataset, node_id)].append(score)
    random_mean = {key: statistics.mean(values) for key, values in random_scores.items()}
    effects["random_global_mean"] = _paired_effect(baseline, random_mean)

    effects["by_dataset"] = {}
    for artifact in artifacts:
        dataset = artifact.label
        dataset_baseline = {
            key: value for key, value in baseline.items() if key[0] == dataset
        }
        dataset_effects = {}
        for condition in conditions:
            compared = {
                (current_dataset, node_id): score
                for (current_dataset, node_id, current), score in task_scores.items()
                if current_dataset == dataset and current == condition
            }
            dataset_effects[condition] = _paired_effect(
                dataset_baseline, compared,
                seed=_stable_seed(42, dataset, condition),
            )
        dataset_effects["random_global_mean"] = _paired_effect(
            dataset_baseline,
            {key: value for key, value in random_mean.items() if key[0] == dataset},
            seed=_stable_seed(42, dataset, "random"),
        )
        effects["by_dataset"][dataset] = dataset_effects

    replacement = []
    for artifact in artifacts:
        for node in artifact.internal_nodes():
            node_id = int(node["id"])
            current = candidates[artifact.label][node_id][
                "current_same_height_raw_mpnet"
            ]["candidate_id"]
            aligned = candidates[artifact.label][node_id][
                "true_sibling_capability_minilm"
            ]["candidate_id"]
            replacement.append(current != aligned)
    effects["aligned_replacement_rate_all_nodes"] = statistics.mean(replacement)
    return effects


def _summary_condition_metrics(
    artifacts: Sequence[TreeArtifact], summary_panels: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[tuple[str, int], dict[str, float]]]:
    grouped: dict[str, dict[tuple[str, int], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"inside": [], "outside": []})
    )
    for row in rows:
        condition = row.get("condition", "")
        if not condition.startswith("summary_") or row.get("score") is None:
            continue
        summary_condition = condition.split("__", 1)[0]
        key = (row["dataset"], int(row["node_id"]))
        bucket = "inside" if row.get("metric") == "descendant_faithfulness" else "outside"
        grouped[summary_condition][key][bucket].append(float(row["score"]))

    baseline_name = "summary_recursive_current"
    for artifact in artifacts:
        for node_id in summary_panels[artifact.label]:
            node = artifact.by_id[node_id]
            key = (artifact.label, node_id)
            grouped[baseline_name][key]["inside"].append(float(node["faithfulness1"]))
            grouped[baseline_name][key]["outside"].append(
                float(node["specificity_faithfulness"])
            )
            aligned = next((row for row in rows
                            if row.get("dataset") == artifact.label
                            and row.get("node_id") == node_id
                            and row.get("condition") == "true_sibling_capability_minilm"
                            and row.get("score") is not None), None)
            if aligned:
                grouped[baseline_name][key]["outside"].append(float(aligned["score"]))

    result = {}
    for condition, by_node in grouped.items():
        result[condition] = {}
        for key, values in by_node.items():
            if not values["inside"] or not values["outside"]:
                continue
            inside = statistics.mean(values["inside"])
            outside = statistics.mean(values["outside"])
            result[condition][key] = {
                "inside": inside, "outside": outside, "gap": inside - outside,
            }
    return result


def analyze_summaries(
    artifacts: Sequence[TreeArtifact], summary_panels: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _summary_condition_metrics(artifacts, summary_panels, rows)
    output = {"conditions": {}}
    for condition, by_node in metrics.items():
        output["conditions"][condition] = {
            metric: _summary([values[metric] for values in by_node.values()])
            for metric in ("inside", "outside", "gap")
        }
    comparisons = {
        "broadness_recursive": ("summary_recursive_current",
                                  "summary_recursive_boundary"),
        "recursive_drift_current_prompt": ("summary_recursive_current",
                                             "summary_direct_current"),
        "combined_direct_boundary": ("summary_recursive_current",
                                      "summary_direct_boundary"),
    }
    output["comparisons"] = {}
    for name, (baseline_name, comparison_name) in comparisons.items():
        baseline_nodes = metrics.get(baseline_name, {})
        comparison_nodes = metrics.get(comparison_name, {})
        output["comparisons"][name] = {
            metric: _paired_effect(
                {key: values[metric] for key, values in baseline_nodes.items()},
                {key: values[metric] for key, values in comparison_nodes.items()},
            )
            for metric in ("inside", "outside", "gap")
        }
    output["comparisons_by_dataset"] = {}
    for artifact in artifacts:
        dataset = artifact.label
        output["comparisons_by_dataset"][dataset] = {}
        for name, (baseline_name, comparison_name) in comparisons.items():
            baseline_nodes = {
                key: values for key, values in metrics.get(baseline_name, {}).items()
                if key[0] == dataset
            }
            comparison_nodes = {
                key: values for key, values in metrics.get(comparison_name, {}).items()
                if key[0] == dataset
            }
            output["comparisons_by_dataset"][dataset][name] = {
                metric: _paired_effect(
                    {key: values[metric] for key, values in baseline_nodes.items()},
                    {key: values[metric] for key, values in comparison_nodes.items()},
                    seed=_stable_seed(42, dataset, name, metric),
                )
                for metric in ("inside", "outside", "gap")
            }
    return output


def analyze_judge(
    artifacts: Sequence[TreeArtifact], calibration_panel: Sequence[tuple[str, int]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    artifact_map = {artifact.label: artifact for artifact in artifacts}
    baseline = {
        (dataset, node_id): float(
            artifact_map[dataset].by_id[node_id]["specificity_faithfulness"]
        )
        for dataset, node_id in calibration_panel
    }
    scores = _task_score_index(rows)
    conditions = (
        "judge_current_repeat_1", "judge_current_repeat_2",
        "judge_deanchored", "judge_binary", "judge_claude_current",
    )
    effects = {}
    for condition in conditions:
        compared = {
            (dataset, node_id): score
            for (dataset, node_id, current), score in scores.items()
            if current == condition
        }
        effects[condition] = _paired_effect(baseline, compared)
    repeats = defaultdict(list)
    for (dataset, node_id, condition), score in scores.items():
        if condition.startswith("judge_current_repeat_"):
            repeats[(dataset, node_id)].append(score)
    repeat_means = {key: statistics.mean(values) for key, values in repeats.items()}
    effects["judge_current_repeat_mean"] = _paired_effect(baseline, repeat_means)
    effects["controls"] = {
        condition: _summary([
            float(row["score"]) for row in rows
            if row.get("condition") == condition and row.get("score") is not None
        ])
        for condition in (
            "identity_control_openai", "identity_control_claude",
            "distant_control_openai", "distant_control_claude",
        )
    }
    effects["by_dataset"] = {}
    for artifact in artifacts:
        dataset = artifact.label
        dataset_baseline = {
            key: value for key, value in baseline.items() if key[0] == dataset
        }
        effects["by_dataset"][dataset] = {}
        for condition in conditions:
            compared = {
                (current_dataset, node_id): score
                for (current_dataset, node_id, current), score in scores.items()
                if current_dataset == dataset and current == condition
            }
            effects["by_dataset"][dataset][condition] = _paired_effect(
                dataset_baseline, compared,
                seed=_stable_seed(42, dataset, condition),
            )
    return effects


def analyze_sampling(
    artifacts: Sequence[TreeArtifact], sampling_panels: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    task_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("condition", "").startswith("extra_descendant_") \
                and row.get("score") is not None:
            task_scores[(row["dataset"], int(row["node_id"]))].append(float(row["score"]))
    differences = []
    records = []
    for artifact in artifacts:
        for node_id in sampling_panels[artifact.label]:
            original = _extract_descendant_scores(
                artifact.by_id[node_id].get("evaluation_faithfulness") or ""
            )
            original_scores = [score for _, score in original]
            extra = task_scores.get((artifact.label, node_id), [])
            if not original_scores or not extra:
                continue
            two_mean = statistics.mean(original_scores)
            five_mean = statistics.mean(original_scores + extra)
            difference = five_mean - two_mean
            differences.append(difference)
            records.append({
                "dataset": artifact.label, "node_id": node_id,
                "two_leaf_mean": two_mean, "five_leaf_mean": five_mean,
                "difference": difference,
            })
    output = {
        "n": len(records),
        "mean_difference": _mean(differences),
        "mean_absolute_difference": _mean([abs(value) for value in differences]),
        "fraction_abs_gt_005": (
            statistics.mean(abs(value) > 0.05 for value in differences)
            if differences else None
        ),
        "spearman": _spearman(
            [row["two_leaf_mean"] for row in records],
            [row["five_leaf_mean"] for row in records],
        ),
        "records": records,
    }
    output["by_dataset"] = {}
    for artifact in artifacts:
        dataset_records = [row for row in records if row["dataset"] == artifact.label]
        dataset_differences = [row["difference"] for row in dataset_records]
        output["by_dataset"][artifact.label] = {
            "n": len(dataset_records),
            "mean_difference": _mean(dataset_differences),
            "mean_absolute_difference": _mean(
                [abs(value) for value in dataset_differences]
            ),
            "fraction_abs_gt_005": (
                statistics.mean(abs(value) > 0.05 for value in dataset_differences)
                if dataset_differences else None
            ),
        }
    return output


def analyze_annotations(
    artifacts: Sequence[TreeArtifact], annotation_leaves: dict[str, list[int]],
    tasks: Sequence[dict[str, Any]], cache: CallCache,
    registry: EmbeddingRegistry,
) -> dict[str, Any]:
    task_lookup = {
        (task["metadata"]["dataset"], int(task["metadata"]["node_id"]),
         task["condition"]): cache.entries.get(task["key"], {})
        for task in tasks if task["condition"].startswith("annotation_")
    }
    records = []
    for artifact in artifacts:
        original_space = registry.space(
            artifact, "capability", "all-MiniLM-L6-v2"
        )
        level_one = artifact.levels[1]["nodes"] if len(artifact.levels) > 1 else []
        cluster_centers = {
            int(node["id"]): _centroid([
                original_space[leaf_id]
                for leaf_id in artifact.descendants(int(node["id"]))
            ])
            for node in level_one
        }
        for leaf_id in annotation_leaves[artifact.label]:
            current = task_lookup.get((artifact.label, leaf_id,
                                       "annotation_current_repeat"), {})
            instruction = task_lookup.get((artifact.label, leaf_id,
                                           "annotation_instruction_only"), {})
            if current.get("status") != "complete" or instruction.get("status") != "complete":
                continue
            generated = registry.encode_texts(
                [current["text"], instruction["text"]], "all-MiniLM-L6-v2"
            )
            original = original_space[leaf_id]

            def nearest_leaf(vector: Sequence[float]) -> int:
                candidates = [candidate for candidate in artifact.leaf_ids
                              if candidate != leaf_id]
                return max(candidates, key=lambda candidate: (
                    _dot(vector, original_space[candidate]), -candidate
                ))

            def nearest_cluster(vector: Sequence[float]) -> int | None:
                if not cluster_centers:
                    return None
                return max(cluster_centers, key=lambda node_id: (
                    _dot(vector, cluster_centers[node_id]), -node_id
                ))

            records.append({
                "dataset": artifact.label,
                "leaf_id": leaf_id,
                "original_repeat_cosine": _dot(original, generated[0]),
                "original_instruction_only_cosine": _dot(original, generated[1]),
                "repeat_nearest_neighbor_stable": (
                    nearest_leaf(original) == nearest_leaf(generated[0])
                ),
                "instruction_nearest_neighbor_stable": (
                    nearest_leaf(original) == nearest_leaf(generated[1])
                ),
                "repeat_cluster_stable": (
                    nearest_cluster(original) == nearest_cluster(generated[0])
                ),
                "instruction_cluster_stable": (
                    nearest_cluster(original) == nearest_cluster(generated[1])
                ),
            })
    if not records:
        return {"n": 0, "records": []}
    return {
        "n": len(records),
        "original_repeat_cosine": _summary(
            [row["original_repeat_cosine"] for row in records]
        ),
        "original_instruction_only_cosine": _summary(
            [row["original_instruction_only_cosine"] for row in records]
        ),
        "repeat_nearest_neighbor_stability": statistics.mean(
            row["repeat_nearest_neighbor_stable"] for row in records
        ),
        "instruction_nearest_neighbor_stability": statistics.mean(
            row["instruction_nearest_neighbor_stable"] for row in records
        ),
        "repeat_cluster_stability": statistics.mean(
            row["repeat_cluster_stable"] for row in records
        ),
        "instruction_cluster_stability": statistics.mean(
            row["instruction_cluster_stable"] for row in records
        ),
        "records": records,
    }


def _verdict(supported: bool | None, partial: bool = False,
             evidence: str = "") -> dict[str, str]:
    if supported is None:
        status = "pending"
    elif supported:
        status = "supported"
    elif partial:
        status = "partially supported"
    else:
        status = "unsupported"
    return {"status": status, "evidence": evidence}


def build_verdicts(analysis: dict[str, Any]) -> dict[str, dict[str, str]]:
    verdicts = {}
    audits = analysis["offline"]
    aggregate_shifts = []
    dependence_ratios = []
    for audit in audits.values():
        distribution = audit["gap_distribution"]
        mean = distribution["mean"]
        aggregate_shifts.extend([
            abs(distribution["median"] - mean),
            abs(distribution["trimmed_mean_10pct"] - mean),
            abs(distribution["level_macro_mean"] - mean),
        ])
        if distribution["clustered_to_naive_se_ratio"] is not None:
            dependence_ratios.append(distribution["clustered_to_naive_se_ratio"])
    verdicts["aggregation"] = _verdict(
        max(aggregate_shifts, default=0) >= 0.05,
        evidence=f"maximum alternative-aggregation shift={max(aggregate_shifts, default=0):.3f}",
    )
    verdicts["dependence"] = _verdict(
        max(dependence_ratios, default=0) >= 1.25,
        evidence=f"maximum clustered/naive SE ratio={max(dependence_ratios, default=0):.2f}",
    )

    candidate = analysis.get("candidates", {})
    sibling = candidate.get("true_sibling_raw_mpnet", {})
    sibling_diff = sibling.get("mean_difference")
    sibling_ci = sibling.get("difference_ci95", [None, None])
    sibling_supported = (
        sibling_diff is not None and abs(sibling_diff) >= 0.05
        and None not in sibling_ci and (sibling_ci[0] > 0 or sibling_ci[1] < 0)
    )
    verdicts["candidate_pool"] = _verdict(
        sibling_supported if sibling_diff is not None else None,
        partial=sibling_diff is not None and abs(sibling_diff) >= 0.05,
        evidence=(f"true-sibling F-out shift={sibling_diff}, implied gap "
                  f"shift={-sibling_diff if sibling_diff is not None else None}"),
    )
    aligned = candidate.get("same_height_capability_minilm", {})
    aligned_diff = aligned.get("mean_difference")
    aligned_ci = aligned.get("difference_ci95", [None, None])
    replacement = candidate.get("aligned_replacement_rate_all_nodes")
    representation_supported = (
        aligned_diff is not None and abs(aligned_diff) >= 0.05
        and None not in aligned_ci and (aligned_ci[0] > 0 or aligned_ci[1] < 0)
    )
    verdicts["representation"] = _verdict(
        representation_supported if aligned_diff is not None else None,
        partial=replacement is not None and replacement >= 0.40,
        evidence=(f"aligned F-out shift={aligned_diff}, implied gap "
                  f"shift={-aligned_diff if aligned_diff is not None else None}, "
                  f"candidate replacement={replacement}"),
    )
    random_effect = candidate.get("random_global_mean", {})
    random_diff = random_effect.get("mean_difference")
    verdicts["hard_negative"] = _verdict(
        (-random_diff >= 0.05) if random_diff is not None else None,
        evidence=(f"random-minus-hard F-out shift={random_diff}, implied gap "
                  f"shift={-random_diff if random_diff is not None else None}"),
    )

    summaries = analysis.get("summaries", {}).get("comparisons", {})
    broad = summaries.get("broadness_recursive", {})
    broad_gap = broad.get("gap", {}).get("mean_difference")
    broad_inside = broad.get("inside", {}).get("mean_difference")
    verdicts["broad_summary"] = _verdict(
        (broad_gap >= 0.05 and broad_inside >= -0.02)
        if broad_gap is not None and broad_inside is not None else None,
        evidence=f"gap shift={broad_gap}, descendant-faith shift={broad_inside}",
    )
    drift = summaries.get("recursive_drift_current_prompt", {})
    drift_gap = drift.get("gap", {}).get("mean_difference")
    drift_inside = drift.get("inside", {}).get("mean_difference")
    verdicts["recursive_drift"] = _verdict(
        (max(drift_gap, drift_inside) >= 0.05)
        if drift_gap is not None and drift_inside is not None else None,
        evidence=f"gap shift={drift_gap}, descendant-faith shift={drift_inside}",
    )

    judge = analysis.get("judge", {})
    deanchored = judge.get("judge_deanchored", {})
    calibration_shift = deanchored.get("mean_difference")
    binary_shift = judge.get("judge_binary", {}).get("mean_difference")
    calibration_effects = [abs(value) for value in (calibration_shift, binary_shift)
                           if value is not None]
    verdicts["judge_calibration"] = _verdict(
        max(calibration_effects) >= 0.05 if calibration_effects else None,
        evidence=(f"de-anchored shift={calibration_shift}, "
                  f"binary-applicability shift={binary_shift}"),
    )
    claude = judge.get("judge_claude_current", {})
    judge_shift = claude.get("mean_absolute_difference")
    judge_rank = claude.get("spearman")
    verdicts["judge_model"] = _verdict(
        (judge_shift >= 0.10 or (judge_rank is not None and judge_rank < 0.70))
        if judge_shift is not None else None,
        evidence=f"GPT/Claude MAD={judge_shift}, Spearman={judge_rank}",
    )

    sampling = analysis.get("sampling", {})
    sampling_mad = sampling.get("mean_absolute_difference")
    sampling_fraction = sampling.get("fraction_abs_gt_005")
    verdicts["descendant_sampling"] = _verdict(
        (sampling_mad >= 0.05 or sampling_fraction > 0.25)
        if sampling_mad is not None and sampling_fraction is not None else None,
        evidence=f"two-vs-five MAD={sampling_mad}, fraction >.05={sampling_fraction}",
    )

    annotations = analysis.get("annotations", {})
    repeat_cos = annotations.get("original_repeat_cosine", {}).get("mean")
    instruction_cos = annotations.get("original_instruction_only_cosine", {}).get("mean")
    repeat_cluster = annotations.get("repeat_cluster_stability")
    instruction_cluster = annotations.get("instruction_cluster_stability")
    annotation_supported = None
    if None not in (repeat_cos, instruction_cos, repeat_cluster, instruction_cluster):
        annotation_supported = (
            repeat_cos - instruction_cos >= 0.05
            or repeat_cluster - instruction_cluster >= 0.10
        )
    verdicts["annotation_prompt"] = _verdict(
        annotation_supported,
        evidence=(f"cosine repeat/instruction={repeat_cos}/{instruction_cos}, "
                  f"cluster stability={repeat_cluster}/{instruction_cluster}"),
    )
    return verdicts


def _format_number(value: Any, digits: int = 3) -> str:
    return "pending" if value is None else f"{float(value):.{digits}f}"


def write_outputs(
    output_dir: Path,
    artifacts: Sequence[TreeArtifact],
    audits: dict[str, dict[str, Any]],
    panels: dict[str, list[int]],
    summary_panels: dict[str, list[int]],
    sampling_panels: dict[str, list[int]],
    annotation_leaves: dict[str, list[int]],
    candidates: dict[str, dict[int, dict[str, dict[str, Any]]]],
    tasks: Sequence[dict[str, Any]],
    cache: CallCache,
    calibration_panel: Sequence[tuple[str, int]],
    registry: EmbeddingRegistry,
) -> tuple[Path, Path, Path, Path]:
    rows = task_rows(tasks, cache)
    analysis = {
        "offline": audits,
        "candidates": analyze_candidates(artifacts, panels, candidates, rows),
        "summaries": analyze_summaries(artifacts, summary_panels, rows),
        "judge": analyze_judge(artifacts, calibration_panel, rows),
        "sampling": analyze_sampling(artifacts, sampling_panels, rows),
        "annotations": analyze_annotations(
            artifacts, annotation_leaves, tasks, cache, registry
        ),
    }
    analysis["verdicts"] = build_verdicts(analysis)
    artifact_map = {artifact.label: artifact for artifact in artifacts}
    diagnostics = {
        "config": {
            "seed": 42,
            "panel_size_per_dataset": 24,
            "summary_panel_size_per_dataset": 12,
            "sampling_panel_size_per_dataset": 12,
            "annotation_leaf_count_per_dataset": 12,
            "source_hashes": {
                artifact.label: _sha256_path(artifact.path) for artifact in artifacts
            },
        },
        "panels": {
            "causal": panels,
            "summary": summary_panels,
            "sampling": sampling_panels,
            "annotations": annotation_leaves,
            "calibration": calibration_panel,
        },
        "candidate_policies": candidates,
        "tasks": [
            {**task, "result": cache.entries.get(task["key"], {"status": "pending"})}
            for task in tasks
        ],
        "analysis": analysis,
    }
    json_path = output_dir / "diagnostics.json"
    _atomic_json(json_path, diagnostics)

    csv_rows = []
    for audit in audits.values():
        for row in audit["rows"]:
            csv_rows.append({
                **row,
                "condition": "current_baseline",
                "metric": "specificity_faithfulness",
                "score": row["specificity"],
                "status": "complete",
                "model": "openai",
                "task_key": "",
            })
    csv_rows.extend(rows)
    csv_path = output_dir / "diagnostics.csv"
    fields = sorted({key for row in csv_rows for key in row} - {"raw_response"})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    blind_path = output_dir / "blind_review.csv"
    with blind_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "node_id", "candidate_id", "pair_type",
                  "candidate_prompt", "parent_summary"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, node_id in calibration_panel:
            artifact = artifact_map[dataset]
            node = artifact.by_id[node_id]
            candidate_id = int(node["third_leaf"])
            writer.writerow({
                "dataset": dataset, "node_id": node_id,
                "candidate_id": candidate_id, "pair_type": "observed_outsider",
                "candidate_prompt": artifact.raw_text(candidate_id),
                "parent_summary": artifact.summary_text(node_id),
            })

    report_path = output_dir / "REPORT.md"
    lines = [
        "# EvalTree Faithfulness-Specificity Gap Root-Cause Study", "",
        "The primary outcome is the paired margin `faithfulness1 - "
        "specificity_faithfulness` for each evaluated parent. A larger positive "
        "margin means the summary applies more strongly to descendants than to "
        "the selected outsider.", "",
        "## Baseline Paired Gap", "",
        "| Dataset | n | Faith avg | Specificity avg | Gap mean [95% CI] | Gap median | 10% trimmed | Level macro | Naive SE | Clustered SE |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label, audit in audits.items():
        faith = audit["faithfulness_distribution"]
        specificity = audit["distribution"]
        gap = audit["gap_distribution"]
        ci = gap["mean_ci95"]
        lines.append(
            f"| {label} | {gap['n']} | {_format_number(faith['mean'])} | "
            f"{_format_number(specificity['mean'])} | {_format_number(gap['mean'])} "
            f"[{_format_number(ci[0])}, {_format_number(ci[1])}] | "
            f"{_format_number(gap['median'])} | "
            f"{_format_number(gap['trimmed_mean_10pct'])} | "
            f"{_format_number(gap['level_macro_mean'])} | "
            f"{_format_number(gap['stderr'])} | "
            f"{_format_number(gap['outsider_clustered_stderr'])} |"
        )
    lines.extend(["", "### Gap By Level", "",
                  "| Dataset | Level | n | Faith avg | Specificity avg | Gap mean |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label, audit in audits.items():
        for level, metrics in audit["by_level"].items():
            lines.append(
                f"| {label} | {level} | {metrics['gap']['n']} | "
                f"{_format_number(metrics['faithfulness']['mean'])} | "
                f"{_format_number(metrics['specificity']['mean'])} | "
                f"{_format_number(metrics['gap']['mean'])} |"
            )
    lines.extend(["", "## Hypothesis Verdicts", "",
                  "| Hypothesis | Verdict | Evidence |", "|---|---|---|"])
    for name, verdict in analysis["verdicts"].items():
        lines.append(
            f"| {name.replace('_', ' ').title()} | {verdict['status']} | "
            f"{verdict['evidence']} |"
        )
    lines.extend(["", "## Integrity", ""])
    for label, audit in audits.items():
        integrity = audit["integrity"]
        lines.append(
            f"- **{label}:** {integrity['evaluated_nodes']} nodes; "
            f"parser mismatches={integrity['parser_mismatches']}; "
            f"missing explicit scores={integrity['parser_missing']}; "
            f"descendant outsiders={integrity['descendant_leaks']}; "
            f"recomputed-current candidate mismatches="
            f"{integrity['current_candidate_mismatches']}."
        )
    lines.extend(["", "## Candidate Experiments", "",
                  "These conditions change only the outsider. Therefore, the "
                  "implied gap shift is the negative of the outsider-score shift.", "",
                  "| Condition | n | Baseline F-out | Comparison F-out | F-out shift | Implied gap shift | 95% CI for F-out |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for condition, result in analysis["candidates"].items():
        if not isinstance(result, dict) or "n" not in result:
            continue
        ci = result.get("difference_ci95", [None, None])
        lines.append(
            f"| {condition} | {result['n']} | {_format_number(result.get('baseline_mean'))} | "
            f"{_format_number(result.get('comparison_mean'))} | "
            f"{_format_number(result.get('mean_difference'))} | "
            f"{_format_number(-result['mean_difference']) if result.get('mean_difference') is not None else 'pending'} | "
            f"[{_format_number(ci[0])}, {_format_number(ci[1])}] |"
        )
    lines.extend(["", "### Candidate Effects By Dataset", "",
                  "| Dataset | Condition | n | F-out shift | Implied gap shift |",
                  "|---|---|---:|---:|---:|"])
    for dataset, effects in analysis["candidates"].get("by_dataset", {}).items():
        for condition in (
            "same_height_capability_minilm", "true_sibling_raw_mpnet",
            "true_sibling_capability_minilm", "random_global_mean",
        ):
            result = effects.get(condition, {})
            lines.append(
                f"| {dataset} | {condition} | {result.get('n', 0)} | "
                f"{_format_number(result.get('mean_difference'))} | "
                f"{_format_number(-result['mean_difference']) if result.get('mean_difference') is not None else 'pending'} |"
            )
    lines.extend(["", "## Summary Experiment", ""])
    for name, result in analysis["summaries"].get("comparisons", {}).items():
        gap = result["gap"]
        inside = result["inside"]
        lines.append(
            f"- **{name}:** gap shift {_format_number(gap.get('mean_difference'))}; "
            f"descendant-faithfulness shift {_format_number(inside.get('mean_difference'))}."
        )
    judge = analysis["judge"]
    lines.extend(["", "## Judge And Sampling", "",
                  "Judge effects below change F-out only, so their implied gap "
                  "effects have the opposite sign when F-in is held fixed."])
    lines.append(
        f"- De-anchored prompt shift: "
        f"{_format_number(judge.get('judge_deanchored', {}).get('mean_difference'))}."
    )
    lines.append(
        f"- Claude mean absolute difference: "
        f"{_format_number(judge.get('judge_claude_current', {}).get('mean_absolute_difference'))}; "
        f"rank correlation: {_format_number(judge.get('judge_claude_current', {}).get('spearman'))}."
    )
    sampling = analysis["sampling"]
    lines.append(
        f"- Two-versus-five descendant MAD: "
        f"{_format_number(sampling.get('mean_absolute_difference'))}; fraction changing "
        f">0.05: {_format_number(sampling.get('fraction_abs_gt_005'))}."
    )
    lines.extend(["", "### Judge Effects By Dataset", "",
                  "| Dataset | De-anchored F-out shift | Binary F-out shift | Claude MAD |",
                  "|---|---:|---:|---:|"])
    for dataset, effects in judge.get("by_dataset", {}).items():
        lines.append(
            f"| {dataset} | "
            f"{_format_number(effects.get('judge_deanchored', {}).get('mean_difference'))} | "
            f"{_format_number(effects.get('judge_binary', {}).get('mean_difference'))} | "
            f"{_format_number(effects.get('judge_claude_current', {}).get('mean_absolute_difference'))} |"
        )
    lines.extend(["", "## Offline Correlations", "",
                  "| Dataset | Feature | Specificity rho [95% CI] | Gap rho [95% CI] |",
                  "|---|---|---|---|"])
    for dataset, audit in audits.items():
        for feature, result in audit["correlations"].items():
            specificity_ci = result["specificity_ci95"]
            gap_ci = result["gap_ci95"]
            lines.append(
                f"| {dataset} | {feature} | "
                f"{_format_number(result['specificity_spearman'])} "
                f"[{_format_number(specificity_ci[0])}, {_format_number(specificity_ci[1])}] | "
                f"{_format_number(result['gap_spearman'])} "
                f"[{_format_number(gap_ci[0])}, {_format_number(gap_ci[1])}] |"
            )
    completed = sum(row.get("status") == "complete" for row in rows)
    pending = sum(row.get("status") == "pending" for row in rows)
    errors = sum(row.get("status") == "error" for row in rows)
    lines.extend(["", "## Run Status", "",
                  f"- Completed task records: {completed}",
                  f"- Pending task records: {pending}",
                  f"- Error task records: {errors}",
                  "- Baseline trees were opened read-only and their SHA-256 hashes are recorded in diagnostics.json."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, report_path, blind_path


def _parse_phases(values: Sequence[str]) -> tuple[str, ...]:
    phases = []
    for value in values:
        phases.extend(part.strip() for part in value.split(",") if part.strip())
    if "all" in phases:
        return ALL_PHASES
    invalid = sorted(set(phases) - set(ALL_PHASES))
    if invalid:
        raise ValueError(f"Unknown phases: {', '.join(invalid)}")
    return tuple(dict.fromkeys(phases))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wb-tree", type=Path, default=DEFAULT_TREES["WB"])
    parser.add_argument("--uf-tree", type=Path, default=DEFAULT_TREES["UF"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", nargs="+", default=["all"],
                        help="all or any of: " + ", ".join(ALL_PHASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-api-calls", type=int, default=1100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    try:
        phases = _parse_phases(args.phases)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_api_calls < 0:
        parser.error("--max-api-calls must be non-negative")
    paths = {"WB": args.wb_tree, "UF": args.uf_tree}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        parser.error("missing evaluated tree(s): " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [TreeArtifact.load(label, path) for label, path in paths.items()]
    source_hashes_before = {artifact.label: _sha256_path(artifact.path)
                            for artifact in artifacts}
    registry = EmbeddingRegistry()
    candidates = {
        artifact.label: compute_candidates(artifact, registry, args.seed)
        for artifact in artifacts
    }
    panels = {artifact.label: select_causal_panel(artifact, args.seed)
              for artifact in artifacts}
    summary_panels = {
        artifact.label: select_summary_panel(artifact, panels[artifact.label])
        for artifact in artifacts
    }
    sampling_panels = {
        artifact.label: select_sampling_panel(artifact, args.seed)
        for artifact in artifacts
    }
    annotation_leaves = {
        artifact.label: select_annotation_leaves(artifact, args.seed)
        for artifact in artifacts
    }
    audits = {
        artifact.label: audit_artifact(
            artifact, candidates[artifact.label], registry
        )
        for artifact in artifacts
    }

    tasks_by_phase = {
        "candidates": build_candidate_tasks(artifacts, panels, candidates),
        "summaries": build_summary_tasks(
            artifacts, summary_panels, candidates, registry
        ),
        "sampling": build_sampling_tasks(artifacts, sampling_panels, args.seed),
        "annotations": build_annotation_tasks(artifacts, annotation_leaves),
    }
    judge_tasks, calibration_panel = build_judge_tasks(artifacts, candidates)
    tasks_by_phase["judge"] = judge_tasks
    all_tasks = [task for phase in ALL_PHASES for task in tasks_by_phase[phase]]
    selected_tasks = [task for phase in phases for task in tasks_by_phase[phase]]
    cache = CallCache.load(args.output_dir / "call_cache.json")
    migrated = migrate_legacy_cache(all_tasks, cache)
    if migrated:
        print(f"Migrated {migrated} successful legacy cache entries")

    unique_selected = {task["key"]: task for task in selected_tasks}
    uncached = [key for key in unique_selected
                if args.overwrite or not cache.successful(key)]
    print("Selected panels: " + ", ".join(
        f"{label}={len(node_ids)}" for label, node_ids in panels.items()
    ))
    print(f"Selected phases: {', '.join(phases) if phases else 'offline only'}")
    print(f"Exact preflight: {len(unique_selected)} unique tasks, "
          f"{len(uncached)} uncached logical API calls")
    if not args.report_only and len(uncached) > args.max_api_calls:
        raise RuntimeError(
            f"Refusing {len(uncached)} calls; --max-api-calls={args.max_api_calls}"
        )
    if not args.report_only and selected_tasks:
        execute_tasks(selected_tasks, cache, args.max_api_calls, args.overwrite)

    outputs = write_outputs(
        args.output_dir, artifacts, audits, panels, summary_panels,
        sampling_panels, annotation_leaves, candidates, all_tasks, cache,
        calibration_panel, registry,
    )
    source_hashes_after = {artifact.label: _sha256_path(artifact.path)
                           for artifact in artifacts}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A baseline evaluated tree changed during diagnostics")
    print("Outputs: " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
