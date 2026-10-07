import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.finetune_tree import audit_sample, export_main_ideas
from tree import Level, Node, Tree


class FinetuneTreeTests(unittest.TestCase):
    def sample(self):
        return {"config": {"selected_ids": ["wb-a", "wb-b"],
                           "selected_session_ids": ["session-a", "session-b"],
                           "selected_source_rows": [5, 10]},
                "levels": [{"level": 0, "nodes": [
                    {"id": 0, "prompt": "Inspect volcanic strata in basalt"},
                    {"id": 1, "prompt": "Compare astronomical spectral absorption"},
                ]}]}

    @patch("experiments.finetune_tree.leakage_reason", return_value=None)
    def test_overlap_is_rejected_before_building(self, _leak):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, protected = root / "sample.json", root / "evaluation.json"
            source.write_text(json.dumps(self.sample()), encoding="utf-8")
            protected.write_text(json.dumps(self.sample()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Training/evaluation overlap"):
                audit_sample(source, (protected,), count=2)

    def test_export_preserves_leaf_ids_targets_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "sample.json", root / "tree.json"
            sample = self.sample()
            source.write_text(json.dumps(sample), encoding="utf-8")
            tree = Tree(levels=[
                Level(level=0, nodes=[Node(**node, mainIdea=f"idea {node['id']}")
                                      for node in sample["levels"][0]["nodes"]]),
                Level(level=1, nodes=[Node(id=2, prompt="Explain scientific observations", mainIdea="science")]),
            ])
            tree.dump(target)
            info = export_main_ideas(target, source, root)
            self.assertEqual(info["leaf_main_ideas"], 2)
            self.assertEqual(info["stored_main_ideas"], 3)
            leaves = [json.loads(line) for line in (root / "leaf_main_ideas.jsonl").read_text().splitlines()]
            self.assertEqual(leaves[0]["wildbench_id"], "wb-a")
            self.assertEqual(leaves[1]["session_id"], "session-b")
            self.assertEqual(leaves[1]["source_row"], 10)
            self.assertEqual(leaves[0]["main_idea"], "idea 0")
            self.assertEqual(leaves[0]["teacher_model"], "claude-sonnet-4-6")

    def test_export_refuses_missing_main_ideas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "sample.json", root / "tree.json"
            sample = self.sample()
            source.write_text(json.dumps(sample), encoding="utf-8")
            tree = Tree(levels=[Level(level=0, nodes=[Node(**node)
                                                      for node in sample["levels"][0]["nodes"]])])
            tree.dump(target)
            with self.assertRaisesRegex(ValueError, "Missing leaf main ideas"):
                export_main_ideas(target, source, root)
            self.assertFalse((root / "leaf_main_ideas.jsonl").exists())
