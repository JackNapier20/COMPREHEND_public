import unittest
from unittest.mock import patch

from experiments import gemma_smoke_test as smoke


class GemmaSmokeTests(unittest.TestCase):
    def setUp(self):
        self.keys = {
            "gemma": {"base_url": "http://gpu013:5050/v1", "model": "gemma-v4"},
            "openai": "unused-judge-key",
        }

    @patch("experiments.gemma_smoke_test.generate_summary")
    @patch("experiments.gemma_smoke_test.generate_cluster_summary")
    @patch("experiments.gemma_smoke_test._extract_main_idea_llm")
    def test_probe_uses_gemma_only_and_no_checkpoint(self, extract, cluster, common):
        extract.return_value = ["Reading CSV files", "Filtering CSV rows"]
        cluster.return_value = "Processing tabular data with Python"

        def summarize(tree, *_args, **_kwargs):
            for node in tree.levels[0].nodes:
                node.summary = "Write Python code to process CSV data."

        common.side_effect = summarize
        result = smoke.run_smoke_test(self.keys)
        self.assertEqual(result["evaltree_summary"], cluster.return_value)
        self.assertEqual(extract.call_args.kwargs["batch_size"], 1)
        for call in (extract.call_args, cluster.call_args, common.call_args):
            self.assertEqual(set(call.kwargs["api_keys"]), {"gemma"})
            self.assertNotIn("checkpoint_path", call.kwargs)
        self.assertNotIn("concurrency", self.keys["gemma"])

    @patch("experiments.gemma_smoke_test.generate_summary")
    @patch("experiments.gemma_smoke_test.generate_cluster_summary")
    @patch("experiments.gemma_smoke_test._extract_main_idea_llm")
    def test_unknown_evaltree_summary_stops_before_common_summary(self, extract, cluster, common):
        extract.return_value = ["idea A", "idea B"]
        cluster.return_value = "Unknown capability"
        with self.assertRaisesRegex(RuntimeError, "EvalTree summary"):
            smoke.run_smoke_test(self.keys, batch_size=2)
        self.assertEqual(extract.call_args.kwargs["batch_size"], 2)
        common.assert_not_called()

    @patch("experiments.gemma_smoke_test._extract_main_idea_llm")
    def test_invalid_batch_size_stops_before_requests(self, extract):
        with self.assertRaisesRegex(ValueError, "batch size must be positive"):
            smoke.run_smoke_test(self.keys, batch_size=0)
        extract.assert_not_called()

    @patch("experiments.gemma_smoke_test.generate_summary")
    @patch("experiments.gemma_smoke_test.generate_cluster_summary")
    @patch("experiments.gemma_smoke_test._extract_main_idea_llm")
    def test_summary_can_be_tested_without_running_extraction(self, extract, cluster, common):
        cluster.return_value = "Processing CSV data using Python"
        result = smoke.run_smoke_test(self.keys, checks=("evaltree-summary",))
        self.assertEqual(result, {"evaltree_summary": cluster.return_value})
        extract.assert_not_called()
        common.assert_not_called()

    @patch("experiments.gemma_smoke_test._extract_main_idea_llm")
    def test_unknown_check_fails_before_any_requests(self, extract):
        with self.assertRaisesRegex(ValueError, "Select checks"):
            smoke.run_smoke_test(self.keys, checks=("unknown",))
        extract.assert_not_called()

if __name__ == "__main__":
    unittest.main()
