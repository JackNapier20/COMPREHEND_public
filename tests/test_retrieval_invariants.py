import random
import tempfile
import unittest
from pathlib import Path

from retrieval.embedding_cache import (
    embedding_cache_path,
    load_or_create_embeddings,
    sentences_signature,
)
from retrieval.similarity import blend_similarities, closest_outsider
from retrieval.blossom_matching import _parse_main_idea_batch
from evaltree.sampling import sample_descendants


class RetrievalInvariantTests(unittest.TestCase):
    def test_main_idea_batch_parser_accepts_fenced_json_only(self):
        parsed = _parse_main_idea_batch(
            '```json\n["first idea", "second idea"]\n```',
            expected=2,
        )

        self.assertEqual(parsed, ["first idea", "second idea"])
        self.assertIsNone(_parse_main_idea_batch('["one"]\ncommentary', expected=2))

    def test_cache_path_identifies_model_level_count_and_content(self):
        first = embedding_cache_path("org/model", 2, ["a", "b"], "/tmp/cache")
        changed_level = embedding_cache_path("org/model", 3, ["a", "b"], "/tmp/cache")
        changed_content = embedding_cache_path("org/model", 2, ["a", "c"], "/tmp/cache")

        self.assertEqual(first.parent, Path("/tmp/cache"))
        self.assertIn("org_model_L2_N2_", first.name)
        self.assertNotEqual(first, changed_level)
        self.assertNotEqual(first, changed_content)
        self.assertEqual(sentences_signature(["a", "b"]), sentences_signature(["a", "b"]))

    def test_main_idea_similarity_changes_the_edge_weight(self):
        self.assertAlmostEqual(blend_similarities(0.5, 1.0, 0.2), 0.6)
        self.assertAlmostEqual(blend_similarities(0.5, 0.0, 0.2), 0.4)

    def test_outsider_excludes_self_and_selected_match(self):
        scores = [1.0, 0.95, 0.8, 0.7]
        self.assertEqual(closest_outsider(scores, self_idx=0, match_idx=1), 2)

    def test_outsider_is_missing_when_only_a_pair_exists(self):
        self.assertIsNone(closest_outsider([1.0, 0.9], self_idx=0, match_idx=1))

    def test_entailment_samples_only_from_descendant_leaves(self):
        descendants = [11, 12, 13, 14]
        sampled = sample_descendants(descendants, 2, random.Random(42))
        self.assertEqual(len(sampled), 2)
        self.assertTrue(set(sampled).issubset(descendants))

    def test_exact_cache_hit_does_not_reencode(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in this interpreter")

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def encode(self, sentences, **kwargs):
                self.calls += 1
                return np.asarray([[idx + 1.0, 1.0] for idx, _ in enumerate(sentences)])

        model = FakeModel()
        with tempfile.TemporaryDirectory() as cache_dir:
            first = load_or_create_embeddings(
                model, ["a", "b"], "model", 0, cache_dir=cache_dir
            )
            second = load_or_create_embeddings(
                model, ["a", "b"], "model", 0, cache_dir=cache_dir
            )
        self.assertEqual(model.calls, 1)
        self.assertEqual(first[2], second[2])
        self.assertTrue(np.allclose(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
