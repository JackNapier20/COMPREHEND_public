import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common_summary_generation.summarize import generate_summary
from llm_utils import call_with_deadline, gemma_chat_text
from retrieval.blossom_matching import _extract_main_idea_llm, extract_main_ideas
from tree import Level, Node, Tree


class GemmaGenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_key": "DUMMY", "base_url": "http://gpu014:5050/v1",
            "model": "gemma-v4", "timeout_seconds": 600, "max_tokens": 1024,
        }
        self.response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="short answer"), finish_reason="stop",
        )])

    @patch("llm_utils.time.sleep")
    @patch("llm_utils.OpenAI")
    @patch("llm_utils.call_with_deadline", wraps=call_with_deadline)
    def test_local_retry_keeps_model_output_cap_and_explicit_deadline(
        self, deadline, openai, _sleep,
    ):
        client = openai.return_value
        client.chat.completions.create.side_effect = [TimeoutError("slow"), self.response]
        text = gemma_chat_text(
            self.config, messages=[{"role": "user", "content": "prompt"}], max_tokens=512,
        )
        self.assertEqual(text, "short answer")
        self.assertEqual(openai.call_args.kwargs["timeout"].read, 600)
        self.assertEqual(openai.call_args.kwargs["timeout"].connect, 10)
        self.assertEqual(openai.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        for request in client.chat.completions.create.call_args_list:
            self.assertEqual(request.kwargs["model"], "gemma-v4")
            self.assertEqual(request.kwargs["max_tokens"], 512)
            self.assertEqual(request.kwargs["stop"], ["<end_of_turn>"])
        for invocation in deadline.call_args_list:
            self.assertEqual(invocation.kwargs["timeout_seconds"], 600)
        client.close.assert_called_once()

    @patch("llm_utils.time.sleep")
    @patch("llm_utils.OpenAI")
    def test_persistent_timeout_is_bounded_and_closes_client(self, openai, _sleep):
        client = openai.return_value
        client.chat.completions.create.side_effect = TimeoutError("slow")
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            gemma_chat_text(self.config, messages=[{"role": "user", "content": "prompt"}])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        client.close.assert_called_once()

    @patch("llm_utils.time.sleep")
    @patch("llm_utils.OpenAI")
    def test_truncated_generation_is_not_accepted_as_a_summary(self, openai, _sleep):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="unfinished summary"), finish_reason="length",
        )])
        client = openai.return_value
        client.chat.completions.create.return_value = response
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts") as failure:
            gemma_chat_text(self.config, messages=[{"role": "user", "content": "prompt"}])
        self.assertIn("output-token limit", str(failure.exception.__cause__))
        client.close.assert_called_once()

    @patch("retrieval.blossom_matching.gemma_chat_text")
    def test_main_idea_dispatch_bounds_batch_and_single_outputs(self, generate):
        generate.side_effect = ['["idea A", "idea B"]', "idea C"]
        keys = {"gemma": {**self.config, "main_idea_max_tokens": 768}}
        self.assertEqual(_extract_main_idea_llm(
            ["A", "B"], llm_model_name="gemma", api_keys=keys,
        ), ["idea A", "idea B"])
        self.assertEqual(_extract_main_idea_llm(
            "C", llm_model_name="gemma", api_keys=keys,
        ), "idea C")
        self.assertEqual(generate.call_args_list[0].kwargs["max_tokens"], 768)
        self.assertEqual(generate.call_args_list[1].kwargs["max_tokens"], 256)

    @patch("retrieval.blossom_matching._extract_main_idea_llm")
    def test_completed_batches_survive_failure_and_are_skipped_on_resume(self, extract):
        tree = Tree(levels=[Level(level=0, nodes=[
            Node(id=0, prompt="A", mainIdea="existing A"),
            Node(id=1, prompt="B"), Node(id=2, prompt="C"),
            Node(id=3, prompt="D"), Node(id=4, prompt="E"),
        ])])
        keys = {"gemma": {"main_idea_batch_size": 2}}
        extract.side_effect = [["idea B", "idea C"], TimeoutError("slow")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "partial.json"
            with self.assertRaises(TimeoutError):
                extract_main_ideas(
                    tree.levels[0].nodes, llm_model_name="gemma", api_keys=keys,
                    checkpoint_callback=lambda: tree.dump(path),
                )
            saved = Tree.load(path)
            self.assertEqual(
                [node.mainIdea for node in saved.levels[0].nodes],
                ["existing A", "idea B", "idea C", None, None],
            )
            extract.reset_mock()
            extract.side_effect = None
            extract.return_value = ["idea D", "idea E"]
            ideas = extract_main_ideas(
                saved.levels[0].nodes, llm_model_name="gemma", api_keys=keys,
                checkpoint_callback=lambda: saved.dump(path),
            )
            self.assertEqual(extract.call_args.args[0], ["D", "E"])
            extract.assert_called_once()
            self.assertEqual(ideas, ["existing A", "idea B", "idea C", "idea D", "idea E"])

    @patch("retrieval.blossom_matching._extract_main_idea_llm")
    def test_concurrent_batches_keep_order_and_checkpoint_each_batch(self, extract):
        nodes = [Node(id=i, prompt=text) for i, text in enumerate("ABCDEF")]
        extract.side_effect = lambda chunk, **_: [f"idea {text}" for text in chunk]
        checkpoints = []
        ideas = extract_main_ideas(
            nodes, llm_model_name="gemma",
            api_keys={"gemma": {"main_idea_batch_size": 2, "concurrency": 3}},
            checkpoint_callback=lambda: checkpoints.append(
                [node.mainIdea for node in nodes]
            ),
        )
        self.assertEqual(ideas, [f"idea {text}" for text in "ABCDEF"])
        self.assertEqual(extract.call_count, 3)
        self.assertEqual(len(checkpoints), 3)
        for index, snapshot in enumerate(checkpoints):
            self.assertEqual(sum(idea is not None for idea in snapshot), (index + 1) * 2)
            for position, idea in enumerate(snapshot):
                if idea is not None:
                    self.assertEqual(idea, f"idea {'ABCDEF'[position]}")

    @patch("common_summary_generation.summarize.gemma_chat_text")
    def test_concurrent_pair_summaries_land_on_their_own_pairs(self, generate):
        generate.side_effect = lambda _config, *, messages: "(start)summary of {}(end)".format(
            re.search(r"pair\d A", messages[1]["content"]).group()
        )
        nodes = []
        for pair in range(3):
            a, b = 2 * pair, 2 * pair + 1
            nodes.append(Node(id=a, prompt=f"pair{pair} A", match=b, score=0.9, alive=True))
            nodes.append(Node(id=b, prompt=f"pair{pair} B", match=a, score=0.9, alive=False))
        tree = Tree(levels=[Level(level=0, nodes=nodes)], next_id=6)
        generate_summary(
            tree, 0, threshold=-1.0, model_name="gemma",
            api_keys={"gemma": {**self.config, "concurrency": 3}},
        )
        self.assertEqual(generate.call_count, 3)
        for pair in range(3):
            a, b = nodes[2 * pair], nodes[2 * pair + 1]
            self.assertIn(f"pair{pair} A", a.summary)
            self.assertEqual(a.summary, b.summary)

    @patch("common_summary_generation.summarize.gemma_chat_text")
    def test_parallel_pair_failure_preserves_other_pair_summaries(self, generate):
        def summarize(_config, *, messages):
            label = re.search(r"pair\d A", messages[1]["content"]).group()
            if label == "pair0 A":
                raise TimeoutError("slow pair")
            return f"(start)summary of {label}(end)"

        generate.side_effect = summarize
        nodes = []
        for pair in range(3):
            a, b = 2 * pair, 2 * pair + 1
            nodes.append(Node(id=a, prompt=f"pair{pair} A", match=b, score=0.9, alive=True))
            nodes.append(Node(id=b, prompt=f"pair{pair} B", match=a, score=0.9, alive=False))
        tree = Tree(levels=[Level(level=0, nodes=nodes)], next_id=6)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "partial.json"
            with self.assertRaisesRegex(RuntimeError, "gemma failed to summarize 1 nodes"):
                generate_summary(
                    tree, 0, threshold=-1.0, model_name="gemma",
                    api_keys={"gemma": {**self.config, "concurrency": 3}},
                    checkpoint_path=path,
                )
            saved = Tree.load(path).levels[0].nodes
        self.assertIsNone(saved[0].summary)
        self.assertIsNone(saved[1].summary)
        for pair in (1, 2):
            self.assertEqual(saved[2 * pair].summary, f"summary of pair{pair} A")
            self.assertEqual(saved[2 * pair].summary, saved[2 * pair + 1].summary)

    @patch("retrieval.blossom_matching._extract_main_idea_llm")
    def test_parallel_batches_keep_main_ideas_attached_to_the_correct_nodes(self, extract):
        nodes = [Node(id=index, prompt=text) for index, text in enumerate(["A", "B", "C", "D"])]
        rendezvous = threading.Barrier(2)

        def batch(texts, **_kwargs):
            rendezvous.wait(timeout=5)
            return [f"idea {text}" for text in texts]

        extract.side_effect = batch
        checkpoints = []
        ideas = extract_main_ideas(
            nodes, llm_model_name="gemma",
            api_keys={"gemma": {"main_idea_batch_size": 2, "concurrency": 2}},
            checkpoint_callback=lambda: checkpoints.append([node.mainIdea for node in nodes]),
        )
        self.assertEqual(ideas, ["idea A", "idea B", "idea C", "idea D"])
        self.assertEqual(sum(idea is not None for idea in checkpoints[0]), 2)
        self.assertEqual(checkpoints[1], ideas)

    @patch("retrieval.blossom_matching._extract_main_idea_llm")
    def test_successful_parallel_batch_is_saved_even_when_first_batch_fails(self, extract):
        nodes = [Node(id=index, prompt=text) for index, text in enumerate("ABCD")]
        tree = Tree(levels=[Level(level=0, nodes=nodes)])
        rendezvous = threading.Barrier(2)

        def batch(texts, **_kwargs):
            rendezvous.wait(timeout=5)
            if texts[0] == "A":
                raise TimeoutError("first batch failed")
            return [f"idea {text}" for text in texts]

        extract.side_effect = batch
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "partial.json"
            with self.assertRaisesRegex(TimeoutError, "first batch failed"):
                extract_main_ideas(
                    nodes, llm_model_name="gemma",
                    api_keys={"gemma": {"main_idea_batch_size": 2, "concurrency": 2}},
                    checkpoint_callback=lambda: tree.dump(path),
                )
            saved = Tree.load(path)
        self.assertEqual([node.mainIdea for node in saved.levels[0].nodes],
                         [None, None, "idea C", "idea D"])

    @patch("retrieval.blossom_matching._extract_main_idea_llm")
    def test_invalid_batch_size_fails_before_request(self, extract):
        with self.assertRaisesRegex(ValueError, "batch size must be positive"):
            extract_main_ideas(
                [Node(id=0, prompt="A")], llm_model_name="gemma",
                api_keys={"gemma": {"main_idea_batch_size": 0}},
            )
        extract.assert_not_called()
