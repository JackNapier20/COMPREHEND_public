import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.evaluate import evaluate_tree_file


class _EmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[index + 1.0, 1.0] for index, _ in enumerate(texts)])


class EvaluationResumeTests(unittest.TestCase):
    @patch("evaluation.evaluate.get_embed_model", return_value=_EmbeddingModel())
    @patch("evaluation.evaluate._get_score", return_value=(0.5, "Score: 0.5"))
    def test_completed_nodes_are_skipped_on_resume(self, get_score, _model):
        tree = {
            "config": {},
            "levels": [
                {"level": 0, "nodes": [
                    {"id": 0, "prompt": "leaf zero", "children": []},
                    {"id": 1, "prompt": "leaf one", "children": []},
                ]},
                {"level": 1, "nodes": [
                    {"id": 2, "prompt": "parent", "children": [0, 1]},
                ]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tree.json"
            output = Path(directory) / "tree_eval.json"
            source.write_text(json.dumps(tree), encoding="utf-8")

            evaluate_tree_file(source, api_keys={}, output_path=output)
            first_call_count = get_score.call_count
            evaluate_tree_file(source, api_keys={}, output_path=output)

            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(first_call_count, 4)
        self.assertEqual(get_score.call_count, first_call_count)
        self.assertEqual(
            result["config"]["posthoc_evaluation_status"], "complete"
        )
        self.assertTrue(result["levels"][1]["nodes"][0]["posthoc_evaluation_complete"])


if __name__ == "__main__":
    unittest.main()
