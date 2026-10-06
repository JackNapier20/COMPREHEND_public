import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments import evaltree_k_sensitivity as runner
from tree import Level, Node, Tree


class GemmaExperimentConfigTests(unittest.TestCase):
    def source(self):
        return Tree(
            levels=[Level(level=0, nodes=[Node(id=0, prompt="prompt", capability="capability")])],
            config={"annotation_model": "claude"},
        )

    def config(self):
        return runner._base_config(
            Path("source.json"), "source-hash", 1, None, source_tree=self.source(),
        )

    def test_changed_checkpoint_is_incompatible_with_completed_tree(self):
        with patch.dict(runner.MODEL_CONFIG, {
            "blossom": "gemma", "summary_generation": "gemma", "evaluation": "openai",
        }), patch.dict(runner.API_KEYS, {"gemma": {"model": "gemma-checkpoint-a"}}):
            first = self.config()
            runner.API_KEYS["gemma"]["model"] = "gemma-checkpoint-b"
            second = self.config()
        self.assertEqual(first["summary_model"], "gemma-checkpoint-a")
        self.assertEqual(first["blossom_model"], "gemma-checkpoint-a")
        self.assertEqual(first["annotation_model"], "claude")
        self.assertEqual(first["evaluation_model"], "openai")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "complete_tree.json"
            tree = self.source()
            tree.config = {**first, "status": "complete"}
            tree.dump(path)
            self.assertTrue(runner._is_complete(path, first))
            self.assertFalse(runner._is_complete(path, second))

    def test_unknown_summary_is_not_marked_complete(self):
        with patch.dict(runner.MODEL_CONFIG, {"summary_generation": "gemma"}), \
                tempfile.TemporaryDirectory() as tmp_dir:
            tree = self.source()
            tree.levels.append(Level(level=1, nodes=[
                Node(id=1, prompt="Unknown capability", summary="Unknown capability"),
            ]))
            with self.assertRaisesRegex(RuntimeError, "not be marked complete"):
                runner._write_complete(tree, Path(tmp_dir) / "tree.json")
            self.assertNotEqual(tree.config.get("status"), "complete")

    def test_changed_request_policy_cannot_reuse_an_old_completed_tree(self):
        with patch.dict(runner.MODEL_CONFIG, {
            "blossom": "gemma", "summary_generation": "gemma", "evaluation": "openai",
        }), patch.dict(runner.API_KEYS, {"gemma": {"model": "gemma-v4"}}):
            expected = self.config()
            policy = expected["gemma_generation_policy"]
            self.assertEqual(policy["temperature"], 0)
            self.assertFalse(policy["ignore_eos"])
            self.assertEqual(policy["main_idea_batch_size"], 1)
            self.assertTrue(policy["request_version"])
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "complete_tree.json"
                tree = self.source()
                tree.config = {**expected, "status": "complete"}
                tree.config.pop("gemma_generation_policy")
                tree.dump(path)
                self.assertFalse(runner._is_complete(path, expected))
                self.assertFalse(runner._is_compatible(path, expected))

    def test_non_gemma_config_is_unchanged(self):
        with patch.dict(runner.MODEL_CONFIG, {
            "blossom": "claude", "summary_generation": "claude", "evaluation": "openai",
        }):
            self.assertNotIn("gemma_generation_policy", self.config())
    def test_rejects_unconfigured_served_model_name(self):
        with patch.dict(runner.MODEL_CONFIG, {"summary_generation": "gemma"}), \
                patch.dict(runner.API_KEYS, {"gemma": {"model": "<MODEL>"}}):
            with self.assertRaisesRegex(ValueError, "Configure the served model"):
                self.config()


if __name__ == "__main__":
    unittest.main()
