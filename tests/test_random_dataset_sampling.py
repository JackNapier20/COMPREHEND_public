import csv
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
