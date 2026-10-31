import tempfile
import unittest
from pathlib import Path

from experiments.reporting import collect_result, write_reports


class SensitivityReportingTests(unittest.TestCase):
    def test_collect_result_aggregates_structure_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tree_eval.json"
            path.write_text(
                """{
                  "levels": [
                    {"level": 0, "nodes": [{"id": 0}, {"id": 1}]},
                    {"level": 1, "nodes": [{
                      "id": 2, "children": [0, 1],
                      "faithfulness1": 0.8, "faithfulness2": 0.6,
                      "informativeness1": 0.7, "informativeness2": 0.5,
                      "specificity_faithfulness": 0.2,
                      "specificity_informativeness": 0.1
                    }]}
                  ]
                }""",
                encoding="utf-8",
            )
            result = collect_result(path, "evaltree", 4)

        self.assertEqual(result["structure"], {
            "leaf_nodes": 2,
            "internal_nodes": 1,
            "levels": 2,
            "max_level": 1,
            "root_nodes": 1,
            "min_branching": 2,
            "mean_branching": 2,
            "max_branching": 2,
        })
        self.assertEqual(result["overall"]["faithfulness1"]["mean"], 0.8)
        self.assertEqual(result["root"]["specificity_faithfulness"]["mean"], 0.2)

    def test_report_uses_recorded_leaf_count(self):
        result = {
            "method": "evaltree",
            "k": 2,
            "provenance": {"provider_fallback_count": 0},
            "structure": {"leaf_nodes": 256, "internal_nodes": 1,
                          "levels": 2, "max_level": 1},
            "overall": {
                metric: {"n": 1, "mean": 0.5, "std": 0.0,
                         "stderr": 0.0, "min": 0.5, "max": 0.5}
                for metric in (
                    "faithfulness1", "faithfulness2", "informativeness1",
                    "informativeness2", "specificity_faithfulness",
                    "specificity_informativeness",
                )
            },
            "root": {},
            "by_level": {},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, report_path = write_reports([result], Path(tmp_dir))
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("All methods use the same 256 leaves", report)


if __name__ == "__main__":
    unittest.main()
