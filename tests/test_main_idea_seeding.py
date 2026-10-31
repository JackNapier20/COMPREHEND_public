import tempfile
import unittest
from pathlib import Path

from experiments.evaltree_k_sensitivity import seed_level_zero_main_ideas
from tree import Level, Node, Tree


class MainIdeaSeedingTests(unittest.TestCase):
    def test_copies_main_ideas_when_ids_and_prompts_match(self):
        target = Tree(levels=[Level(level=0, nodes=[Node(id=0, prompt="same")])])
        seed = Tree(levels=[Level(level=0, nodes=[
            Node(id=0, prompt="same", mainIdea="fixed idea")
        ])])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seed.json"
            seed.dump(path)
            seed_level_zero_main_ideas(target, path)
        self.assertEqual(target.levels[0].nodes[0].mainIdea, "fixed idea")

    def test_rejects_prompt_mismatch(self):
        target = Tree(levels=[Level(level=0, nodes=[Node(id=0, prompt="target")])])
        seed = Tree(levels=[Level(level=0, nodes=[
            Node(id=0, prompt="different", mainIdea="fixed idea")
        ])])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seed.json"
            seed.dump(path)
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                seed_level_zero_main_ideas(target, path)
