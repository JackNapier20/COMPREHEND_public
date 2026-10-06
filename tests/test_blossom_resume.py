import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ablations.runner import run_blossom
from tree import Level, Node, Tree


class BlossomResumeTests(unittest.TestCase):
    def test_specificity_phase_does_not_rematch_or_resummarize(self):
        nodes = [
            Node(id=0, prompt="A", match=1, summary="S", alive=True),
            Node(id=1, prompt="B", match=0, summary="S", alive=False),
        ]
        tree = Tree(
            levels=[Level(level=0, nodes=nodes)],
            next_id=2,
            config={
                "blossom_progress": {
                    "level": 0,
                    "phase": "specificity",
                    "iteration": 1,
                }
            },
        )

        def promote(current_tree, _level_idx):
            current_tree.levels.append(
                Level(level=1, nodes=[Node(id=2, prompt="S")])
            )

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch("ablations.runner.get_nearest_neighbors_blossom") as match, \
                patch("ablations.runner.find_third") as third, \
                patch("ablations.runner.generate_summary") as summarize, \
                patch("ablations.runner.evaluate_common_summary") as evaluate, \
                patch("ablations.runner.record_stat"), \
                patch("ablations.runner.promote_to_next_level", side_effect=promote):
            run_blossom(
                tree,
                Path(tmp_dir) / "tree.json",
                leaf_type="raw",
                custom=True,
                reassign_nodes=False,
            )

        match.assert_not_called()
        third.assert_not_called()
        summarize.assert_not_called()
        evaluate.assert_called_once()
        self.assertTrue(evaluate.call_args.kwargs["eval_specificity"])
        self.assertNotIn("blossom_progress", tree.config)


if __name__ == "__main__":
    unittest.main()
