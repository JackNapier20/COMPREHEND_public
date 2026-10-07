import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_prep.sample_random_dataset import sample_dataset


class RandomDatasetSamplingTests(unittest.TestCase):
    def test_samples_requested_count_with_generic_provenance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "source", "instructions"])
                writer.writeheader()
                for index in range(6):
                    writer.writerow({
                        "id": f"item-{index}",
                        "source": "test",
                        "instructions": f"Explain unique geological process number {index}.",
                    })
            summary = root / "summary.py"
            summary.write_text(
                'prompt14 = f"""Example 1\nPrompt A:\nAlpha task\nPrompt B:\n'
                'Beta task\nPrompt C:\nGamma task\nNow carry out this task\n{story1}"""\n',
                encoding="utf-8",
            )
            output = root / "sample.json"
            data = sample_dataset(
                source,
                summary,
                output,
                dataset_name="Test Dataset",
                count=4,
                seed=42,
            )

        self.assertEqual(len(data["levels"][0]["nodes"]), 4)
        self.assertEqual(data["config"]["dataset"], "Test Dataset")
        self.assertEqual(len(data["config"]["selected_ids"]), 4)
        self.assertEqual(len(data["config"]["selected_sources"]), 4)

    def test_protects_normalized_prompts_and_source_ids_not_local_node_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "instructions"])
                writer.writeheader()
                for index in range(8):
                    writer.writerow({"id": f"item-{index}",
                                     "instructions": f"Describe planetary mineral number {index}."})
            summary = root / "summary.py"
            summary.write_text(
                'prompt14 = f"""Example 1\nPrompt A:\nAlpha task\nPrompt B:\n'
                'Beta task\nPrompt C:\nGamma task\nNow carry out this task\n{story1}"""\n',
                encoding="utf-8",
            )
            protected = root / "evaluation.json"
            protected.write_text(json.dumps({
                "config": {"selected_ids": ["item-3"]},
                "levels": [{"nodes": [{"id": 0, "prompt": "DESCRIBE planetary mineral number 0!"}]}],
            }), encoding="utf-8")
            args = dict(dataset_name="test", count=4, seed=42, exclude_trees=(protected,))
            first = sample_dataset(source, summary, root / "first.json", **args)
            second = sample_dataset(source, summary, root / "second.json", **args)
            self.assertEqual(first, second)
            self.assertNotIn(0, first["config"]["selected_source_rows"])
            self.assertNotIn(3, first["config"]["selected_source_rows"])
            audit = first["config"]["excluded_trees"][0]
            self.assertEqual(audit["source_id_overlap"], 0)
            self.assertEqual(audit["normalized_prompt_overlap"], 0)
            self.assertEqual(len(audit["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                sample_dataset(source, summary, root / "first.json", **args)
