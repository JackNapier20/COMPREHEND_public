import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablations.runner import ASSIGNMENT_SCORING_VERSION, run_ablation, run_blossom
from evaluation.simple_eval import evaluate_common_summary, promote_to_next_level, reassign
from tree import Level, Node, Tree


def fixture():
    return Tree(levels=[Level(level=0, nodes=[
        Node(id=0, prompt='Task A', match=1, third=2, summary='Shared capability', alive=True),
        Node(id=1, prompt='Task B', match=0, third=2, summary='Shared capability', alive=False),
        Node(id=2, prompt='Task C', match=3, third=0, summary='Other capability', alive=True),
        Node(id=3, prompt='Task D', match=2, third=0, summary='Other capability', alive=False),
    ])], next_id=4, config={'blossom_progress': {
        'level': 0, 'phase': 'specificity', 'iteration': 1}})


class AssignmentScoringTests(unittest.TestCase):
    def test_raw_assignment_score_is_computed_and_resumed_without_optional_metrics(self):
        tree = fixture()
        with patch('evaluation.simple_eval._judge_calls', return_value=['Score: 0.9', 'Score: 0.8']) as judge:
            evaluate_common_summary(tree, 0, eval_specificity=True, eval_faith_info=False)
            calls = judge.call_count
            evaluate_common_summary(tree, 0, eval_specificity=True, eval_faith_info=False)
            self.assertEqual(calls, 4)
            self.assertEqual(judge.call_count, calls)
        self.assertAlmostEqual(tree.levels[0].nodes[0].specificity, 0.85)
        self.assertIsNone(tree.levels[0].nodes[0].specificity_faithfulness)
        promote_to_next_level(tree, 0)
        reassign(tree, 0, 0.7)
        self.assertGreater(len(tree.levels[0].nodes[2].parent), 1)
        self.assertIn(2, tree.levels[1].nodes[0].children)

    def test_zero_is_a_valid_cached_assignment_score(self):
        tree = fixture()
        for node in tree.levels[0].nodes:
            node.specificity = 0.0
        with patch('evaluation.simple_eval._judge_calls') as judge:
            evaluate_common_summary(tree, 0, eval_specificity=True, eval_faith_info=False)
        judge.assert_not_called()

    def test_runner_keeps_decision_scoring_when_metrics_are_off(self):
        for assignment in (False, True):
            with self.subTest(assignment=assignment), tempfile.TemporaryDirectory() as directory:
                tree = fixture()
                def promote(current, level):
                    current.levels.append(Level(level=1, nodes=[Node(id=4, prompt='Root')]))
                with patch('ablations.runner.evaluate_common_summary') as score, \
                     patch('ablations.runner.record_stat'), \
                     patch('ablations.runner.promote_to_next_level', side_effect=promote), \
                     patch('ablations.runner.reassign') as assign:
                    run_blossom(tree, Path(directory)/'tree.json', leaf_type='raw', custom=False,
                                reassign_nodes=assignment, collect_internal_metrics=False)
                self.assertEqual(score.call_count, int(assignment))
                if assignment:
                    self.assertTrue(score.call_args.kwargs['eval_specificity'])
                    self.assertFalse(score.call_args.kwargs['eval_faith_info'])
                    assign.assert_called_once()
                    self.assertEqual(tree.config['assignment_scoring_version'], ASSIGNMENT_SCORING_VERSION)
                else:
                    assign.assert_not_called()

    def test_affected_old_outputs_are_not_silently_reused(self):
        from ablations.runner import ABLATIONS, _sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'
            Tree(levels=[Level(level=0, nodes=[Node(id=0, prompt='Task')])]).dump(source)
            target=root/'Ablation1_source_final.json'
            tree=Tree.load(source)
            tree.config={'experiment':'ablation_suite','ablation':1,'source':str(source.resolve()),
                         'source_sha256':_sha256(source),**ABLATIONS[1], 'construction_status':'complete'}
            tree.dump(target)
            with self.assertRaisesRegex(RuntimeError, 'assignment_scoring_version'):
                run_ablation(source, 1, root)
            self.assertNotIn('assignment_scoring_version', json.loads(target.read_text())['config'])
