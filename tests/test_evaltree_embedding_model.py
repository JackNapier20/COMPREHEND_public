import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ablations import runner as ablations
from evaltree.clustering import CLUSTERING_EMBEDDING_MODEL, RecursiveKMeans
from experiments import evaltree_k_sensitivity as comparison
from main import _tree_config
from tree import Level, Node, Tree


class EvalTreeEmbeddingModelTests(unittest.TestCase):
    def leaves(self):
        return Tree(levels=[Level(level=0, nodes=[
            Node(id=0, prompt="Task A", capability="Capability A"),
            Node(id=1, prompt="Task B", capability="Capability B"),
        ])], next_id=2)

    def test_recursive_kmeans_loads_mpnet_and_preserves_k(self):
        with patch("evaltree.clustering.SentenceTransformer") as embedder:
            clustering = RecursiveKMeans(max_children=4)
            embedder.assert_called_once_with("all-mpnet-base-v2")
            self.assertEqual(clustering.find_optimal_k([1, 2, 3, 4, 5]), 4)
            clustering.embed(["A", "B"])
            embedder.return_value.encode.assert_called_once_with(["A", "B"])

    def test_main_records_embedding_only_for_evaltree(self):
        self.assertEqual(_tree_config("evaltree")["clustering_embedding_model"], CLUSTERING_EMBEDDING_MODEL)
        self.assertNotIn("clustering_embedding_model", _tree_config("blossom"))

    def test_runner_records_model_without_changing_summary_or_internal_eval(self):
        tree = self.leaves()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("ablations.runner.RecursiveKMeans") as constructor, \
                patch("ablations.runner.prepare_leaves") as prepare, \
                patch("ablations.runner.build_hierarchy"), \
                patch("ablations.runner.summarize_tree") as summarize, \
                patch("ablations.runner.find_third") as third, \
                patch("ablations.runner.run_evaluation_on_tree") as evaluate:
            ablations.run_evaltree(tree, Path(tmp) / "tree.json", leaf_type="capability",
                                  summary_type="evaltree", num_eval_samples=2,
                                  run_internal_evaluation=False)
        self.assertEqual(tree.config["clustering_embedding_model"], CLUSTERING_EMBEDDING_MODEL)
        constructor.assert_called_once_with(max_children=2, max_depth=50)
        prepare.assert_called_once_with(tree, "capability")
        summarize.assert_called_once()
        third.assert_not_called()
        evaluate.assert_not_called()

    def test_comparison_refuses_legacy_or_minilm_tree_without_overwrite(self):
        for model in (None, "all-MiniLM-L6-v2"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path = directory / "evaltree_k2.json"
                tree = self.leaves()
                tree.config = {"method": "evaltree", "k": 2, "status": "complete"}
                if model is not None:
                    tree.config["clustering_embedding_model"] = model
                tree.dump(path)
                before = path.read_bytes()
                with patch.object(comparison, "build_capability_cache") as cache, \
                        patch.object(comparison, "run_evaltree") as run:
                    cache.return_value = directory / "capabilities.json"
                    with self.assertRaisesRegex(RuntimeError, "current clustering model"):
                        comparison.build_trees(Path("source.json"), directory, (2,), {}, False,
                                               True, False, None, 2)
                run.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_comparison_reuses_mpnet_tree_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "evaltree_k2.json"
            tree = self.leaves()
            tree.config = {"method": "evaltree", "k": 2, "status": "complete",
                           "clustering_embedding_model": CLUSTERING_EMBEDDING_MODEL}
            tree.dump(path)
            with patch.object(comparison, "build_capability_cache"), \
                    patch.object(comparison, "run_evaltree") as run:
                results = comparison.build_trees(Path("source.json"), directory, (2,), {}, False,
                                                 True, False, None, 2)
            run.assert_not_called()
            self.assertEqual(results, [("evaltree", 2, path)])

    def test_explicit_overwrite_can_rebuild_legacy_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "evaltree_k2.json"
            self.leaves().dump(path)
            capabilities = directory / "capabilities.json"
            self.leaves().dump(capabilities)
            with patch.object(comparison, "build_capability_cache", return_value=capabilities), \
                    patch.object(comparison, "run_evaltree") as run, \
                    patch.dict(comparison.MODEL_CONFIG, {"summary_generation": "claude"}):
                comparison.build_trees(Path("source.json"), directory, (2,), {}, True,
                                       True, False, None, 2)
            run.assert_called_once()
            self.assertEqual(Tree.load(path).config["clustering_embedding_model"], CLUSTERING_EMBEDDING_MODEL)

    def test_ablation_refuses_old_completed_evaltree(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "source.json"
            self.leaves().dump(source)
            output = directory / "Ablation5_source_final.json"
            tree = self.leaves()
            tree.config = {"experiment": "ablation_suite", "ablation": 5,
                           "source": str(source.resolve()), "source_sha256": ablations._sha256(source),
                           **ablations.ABLATIONS[5], "construction_status": "complete"}
            tree.dump(output)
            before = output.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "clustering_embedding_model"):
                ablations.run_ablation(source, 5, directory)
            self.assertEqual(output.read_bytes(), before)

    def test_blossom_identity_and_existing_outputs_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "blossom_baseline.json"
            tree = self.leaves()
            tree.config = {"method": "blossom", "k": None, "status": "complete"}
            tree.dump(path)
            before = path.read_bytes()
            with patch.object(comparison, "run_blossom") as run:
                comparison.build_trees(Path("source.json"), directory, (2,), {}, False,
                                       False, True, None, 2)
            run.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
