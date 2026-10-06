import pandas as pd

import score_retrieval
from tree import Level, Node, Tree


def test_add_scores_uses_exact_match_before_edit_distance(tmp_path, monkeypatch):
    score_path = tmp_path / "scores.csv"
    pd.DataFrame({
        "instructions": ["first prompt", "target prompt"],
        "score_model_a": [1.0, 4.25],
        "score_model_b": [2.0, 3.5],
    }).to_csv(score_path, index=False)
    tree = Tree(levels=[Level(level=0, nodes=[
        Node(id=0, prompt="target prompt")
    ])])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("edit-distance fallback should not run")

    monkeypatch.setattr(score_retrieval.editdistance, "eval", fail_if_called)

    score_retrieval.add_scores(tree, score_path)

    assert tree.levels[0].nodes[0].wb_scores == {
        "score_model_a": 4.25,
        "score_model_b": 3.5,
    }
