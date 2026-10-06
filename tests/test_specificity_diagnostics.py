import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from experiments.specificity_diagnostics import (
    CallCache,
    TreeArtifact,
    _bootstrap_mean_ci,
    _bootstrap_spearman_ci,
    _clustered_stderr,
    _format_judge_prompt,
    _legacy_task_key,
    _parse_phases,
    _task,
    build_verdicts,
    compute_candidates,
    execute_tasks,
    migrate_legacy_cache,
    select_causal_panel,
    select_hard_candidate,
)


def _artifact() -> TreeArtifact:
    leaves = [
        {
            "id": node_id,
            "children": [],
            "parent": ([6] if node_id < 2 else [7] if node_id < 4 else [10]),
            "prompt": f"raw prompt {node_id}",
            "capability": f"capability {node_id}",
        }
        for node_id in range(6)
    ]
    nodes = {
        6: {"id": 6, "children": [0, 1], "parent": [8]},
        7: {"id": 7, "children": [2, 3], "parent": [8]},
        10: {"id": 10, "children": [4, 5], "parent": [9]},
        8: {"id": 8, "children": [6, 7], "parent": [9]},
        9: {"id": 9, "children": [8, 10], "parent": []},
    }
    scores = {6: 0.1, 7: 0.8, 10: 0.2, 8: 0.7}
    for node_id, node in nodes.items():
        node.update({
            "prompt": f"summary {node_id}",
            "summary": f"summary {node_id}",
            "faithfulness1": 0.6,
            "specificity_faithfulness": scores.get(node_id),
        })
    payload = {
        "levels": [
            {"level": 0, "nodes": leaves},
            {"level": 1, "nodes": [nodes[6], nodes[7], nodes[10]]},
            {"level": 2, "nodes": [nodes[8]]},
            {"level": 3, "nodes": [nodes[9]]},
        ]
    }
    return TreeArtifact("TEST", Path("fixture.json"), payload)


class SpecificityDiagnosticTests(unittest.TestCase):
    def test_true_sibling_pool_uses_shared_parent(self):
        artifact = _artifact()

        self.assertEqual(artifact.true_sibling_pool(6), [2, 3])
        self.assertEqual(artifact.same_height_pool(6), [2, 3, 4, 5])
        self.assertTrue(set(artifact.true_sibling_pool(6)).isdisjoint(
            artifact.descendants(6)
        ))

    def test_hard_candidate_is_non_descendant_and_deterministic(self):
        artifact = _artifact()
        vectors = {
            0: [1.0, 0.0], 1: [1.0, 0.0],
            2: [0.9, 0.1], 3: [0.0, 1.0],
            4: [0.8, 0.2], 5: [-1.0, 0.0],
        }

        candidate, _ = select_hard_candidate(
            artifact, 6, artifact.same_height_pool(6), vectors
        )

        self.assertEqual(candidate, 2)
        self.assertNotIn(candidate, artifact.descendants(6))

    def test_sparse_panel_selection_is_reproducible_and_balanced(self):
        artifact = _artifact()

        first = select_causal_panel(artifact, seed=42, per_band=2)
        second = select_causal_panel(artifact, seed=42, per_band=2)

        self.assertEqual(first, second)
        self.assertIn(7, first)  # high-specificity L1-2 node
        self.assertTrue(any(node_id in first for node_id in (6, 10)))

    def test_task_keys_deduplicate_requests_but_not_replicates(self):
        first = _task(
            "judge", "openai", "condition-a", {"node_id": 1},
            prompt_type="current", prompt_a="A", prompt_b="B",
        )
        relabeled = _task(
            "judge", "openai", "condition-b", {"node_id": 2},
            prompt_type="current", prompt_a="A", prompt_b="B",
        )
        repeated = _task(
            "judge", "openai", "condition-a", {"node_id": 1},
            prompt_type="current", prompt_a="A", prompt_b="B", replicate=1,
        )

        self.assertEqual(first["key"], relabeled["key"])
        self.assertNotEqual(first["key"], repeated["key"])

    def test_call_cache_only_counts_complete_entries_as_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            cache = CallCache(path, {
                "ok": {"status": "complete", "score": 0.1},
                "bad": {"status": "error", "error": "failure"},
            })
            cache.save()
            loaded = CallCache.load(path)

        self.assertTrue(loaded.successful("ok"))
        self.assertFalse(loaded.successful("bad"))
        self.assertFalse(loaded.successful("missing"))

    def test_legacy_cache_migration_preserves_completed_request(self):
        task = _task(
            "judge", "openai", "condition", {"node_id": 1},
            prompt_type="current", prompt_a="A", prompt_b="B",
            prompt_template_hash="new-template-hash",
        )
        legacy_key = _legacy_task_key(task)
        with tempfile.TemporaryDirectory() as directory:
            cache = CallCache(Path(directory) / "cache.json", {
                legacy_key: {"status": "complete", "score": 0.4, "text": "Score: 0.4"}
            })
            migrated = migrate_legacy_cache([task], cache)

        self.assertEqual(migrated, 1)
        self.assertTrue(cache.successful(task["key"]))
        self.assertEqual(cache.entries[task["key"]]["score"], 0.4)

    def test_call_ceiling_is_checked_before_provider_import(self):
        cache = CallCache(Path("unused.json"))
        tasks = [
            _task(
                "judge", "openai", f"condition-{index}", {"node_id": index},
                prompt_type="current", prompt_a=f"A{index}", prompt_b="B",
            )
            for index in range(2)
        ]

        with self.assertRaises(RuntimeError):
            execute_tasks(tasks, cache, max_calls=1)

        self.assertEqual(cache.entries, {})

    def test_provider_failure_is_checkpointed_for_resume(self):
        task = _task(
            "judge", "openai", "failure", {"node_id": 1},
            prompt_type="current", prompt_a="A", prompt_b="B",
        )
        fake_module = SimpleNamespace(
            _judge_call=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("provider failed")
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            cache = CallCache(path)
            with patch.dict("sys.modules", {"evaluation.simple_eval": fake_module}), \
                    patch("experiments.specificity_diagnostics._api_keys",
                          return_value={}), \
                    patch("experiments.specificity_diagnostics._format_judge_prompt",
                          return_value="prompt"):
                with self.assertRaises(RuntimeError):
                    execute_tasks([task], cache, max_calls=1)
            stored = CallCache.load(path)

        self.assertEqual(stored.entries[task["key"]]["status"], "error")

    def test_embedding_policy_dispatch_builds_every_candidate_family(self):
        artifact = _artifact()

        class FakeRegistry:
            def __init__(self):
                self.calls = []

            def space(self, current, representation, model_name):
                self.calls.append((representation, model_name))
                return {
                    leaf_id: [1.0 - leaf_id / 10, leaf_id / 10]
                    for leaf_id in current.leaf_ids
                }

        registry = FakeRegistry()
        candidates = compute_candidates(artifact, registry, seed=42)

        self.assertEqual(len(set(registry.calls)), 4)
        self.assertIn("same_height_capability_minilm", candidates[6])
        self.assertIn("true_sibling_raw_mpnet", candidates[6])
        self.assertIn("global_raw_mpnet", candidates[6])
        self.assertIn("random_global_3", candidates[6])
        for record in candidates[6].values():
            if record["candidate_id"] is not None:
                self.assertNotIn(record["candidate_id"], artifact.descendants(6))

    def test_clustered_stderr_and_bootstrap_are_deterministic(self):
        rows = [
            {"score": 0.0, "cluster": 1},
            {"score": 0.2, "cluster": 1},
            {"score": 0.8, "cluster": 2},
            {"score": 1.0, "cluster": 2},
        ]

        stderr = _clustered_stderr(rows, "score", "cluster")
        first = _bootstrap_mean_ci([0.1, 0.2, 0.8, 0.9], seed=42, iterations=100)
        second = _bootstrap_mean_ci([0.1, 0.2, 0.8, 0.9], seed=42, iterations=100)
        rank_ci = _bootstrap_spearman_ci(
            [1, 2, 3, 4], [1, 2, 4, 3], seed=42, iterations=100
        )

        self.assertGreater(stderr, 0)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])
        self.assertLessEqual(rank_ci[0], rank_ci[1])

    def test_aggregation_and_dependence_verdicts_use_paired_gap(self):
        analysis = {
            "offline": {
                "TEST": {
                    "distribution": {
                        "mean": 0.2,
                        "median": 0.0,
                        "trimmed_mean_10pct": 0.0,
                        "level_macro_mean": 0.0,
                        "clustered_to_naive_se_ratio": 2.0,
                    },
                    "gap_distribution": {
                        "mean": 0.2,
                        "median": 0.2,
                        "trimmed_mean_10pct": 0.2,
                        "level_macro_mean": 0.2,
                        "clustered_to_naive_se_ratio": 1.0,
                    },
                }
            }
        }

        verdicts = build_verdicts(analysis)

        self.assertEqual(verdicts["aggregation"]["status"], "unsupported")
        self.assertEqual(verdicts["dependence"]["status"], "unsupported")

    def test_phase_parser_supports_csv_and_all(self):
        self.assertEqual(
            _parse_phases(["candidates,judge", "sampling"]),
            ("candidates", "judge", "sampling"),
        )
        self.assertIn("annotations", _parse_phases(["all"]))
        with self.assertRaises(ValueError):
            _parse_phases(["unknown"])

    def test_json_judge_prompts_format_literal_schema_braces(self):
        deanchored = _format_judge_prompt("deanchored", "candidate", "category")
        binary = _format_judge_prompt("binary", "candidate", "category")

        self.assertIn('{"score": NUMBER}', deanchored)
        self.assertIn('{"applies": true or false', binary)


if __name__ == "__main__":
    unittest.main()
