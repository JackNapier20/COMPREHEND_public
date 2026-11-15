import json
import tempfile
import unittest
from pathlib import Path

from gemma_prompts import gemma_main_idea_messages
from train_gemma4_paper import V6_SUFFIX, V6_SYSTEM, pair_smoke_case, read_dataset


def row(system, user, target):
    return {"messages": [
        {"role": role, "content": content}
        for role, content in zip(("system", "user", "assistant"), (system, user, target))
    ]}


class GemmaPaperTrainingTests(unittest.TestCase):
    def test_reformats_extraction_only_for_json_and_jsonl(self):
        text = "Task with\na second line"
        extraction = row(V6_SYSTEM, f"Text:\n{text}{V6_SUFFIX}", "The target.")
        summary = row("Summarize the pair.", "Prompt A / Prompt B", "Shared task.")
        records = [summary, extraction]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "train.json"
            for raw in (json.dumps(records), "\n".join(map(json.dumps, records))):
                data.write_text(raw)
                self.assertEqual(read_dataset(data), [
                    summary, {"messages": [*gemma_main_idea_messages(text), extraction["messages"][-1]]},
                ])
                self.assertEqual(data.read_text(), raw)

    def test_rejects_unrecognized_v6_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "train.json"
            data.write_text(json.dumps([row(V6_SYSTEM, "Wrong format", "Target")]))
            with self.assertRaisesRegex(ValueError, "verified V6 extraction format"):
                read_dataset(data)

    def test_pair_smoke_captures_pipeline_messages(self):
        name, messages, cap = pair_smoke_case()
        self.assertEqual((name, cap), ("comprehend_pair_summary", 1024))
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("bus arrivals", messages[-1]["content"])
        self.assertIn("train departures", messages[-1]["content"])
