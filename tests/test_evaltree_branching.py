import unittest

from evaltree.branching import split_count


class EvalTreeBranchingTests(unittest.TestCase):
    def test_split_count_honors_requested_branching_factor(self):
        for k in (2, 4, 6, 8):
            with self.subTest(k=k):
                self.assertEqual(split_count(64, k), k)

    def test_split_count_caps_k_at_cluster_size(self):
        self.assertEqual(split_count(3, 8), 3)
        self.assertEqual(split_count(1, 8), 1)

    def test_split_count_rejects_non_branching_values(self):
        for k in (-1, 0, 1):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "at least 2"):
                split_count(64, k)


if __name__ == "__main__":
    unittest.main()
