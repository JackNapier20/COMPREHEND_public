import unittest

from common_summary_generation.summarize import generate_summary
from tree import Level, Node, Tree


class SummaryResumeTests(unittest.TestCase):
    def test_existing_unscored_summary_is_reused_without_provider_call(self):
        first = Node(
            id=0,
            prompt="First prompt",
            match=1,
            score=0.8,
            summary="Existing summary",
            alive=True,
        )
        second = Node(
            id=1,
            prompt="Second prompt",
            match=0,
            score=0.8,
            alive=False,
        )
        tree = Tree(levels=[Level(level=0, nodes=[first, second])], next_id=2)

        generate_summary(tree, 0, threshold=0.5, model_name="openai", api_keys={})

        self.assertEqual(first.summary, "Existing summary")
        self.assertEqual(second.summary, "Existing summary")


if __name__ == "__main__":
    unittest.main()
