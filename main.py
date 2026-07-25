from pathlib import Path
import argparse
import os


def _load_env(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()
from common_summary_generation.summarize import generate_summary, get_entail_2_score
from evaluation.evaluate import evaluate_common_summary, promote_to_next_level, propagate_attrs_up, record_stat, reassign
from tree import Tree
from retrieval.blossom_matching import get_nearest_neighbors_blossom
from retrieval.third_neighbor import find_third
from visualisation.visualise_helper import (
    generate_dag_json_from_tree,
    generate_json_from_tree,
    launch_tree_viewer,
)
from score_retrieval import (
    add_scores,
    build_scores
)

# --------------------------------------------------------------------------- #
DATA_FILE = Path("Results/Gemma/FINAL_random_prompts_64.json")
SCORE_FILE = "dataset_WB_with_scores.csv"   # for UltraFeedback: dataset_UF_with_scores.csv
VISUALISE = True                 # flip to False to skip viewer step
MODEL_CONFIG = {
    "blossom": "claude",  # Options: "openai", "claude", "gemma", "gemini"
    "summary_generation": "claude",
    "evaluation": "openai"  # Options: "openai", "claude", "gemma", "gemini"
}
# Credentials come from .env (see .env.example); never hardcode keys here.
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gemma": {
        "api_key": os.environ.get("GEMMA_API_KEY", "DUMMY"),
        "base_url": os.environ.get("GEMMA_BASE_URL", "http://<GPU_HOST>:8000/v1"),
    },
    "gemini": {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}
# --------------------------------------------------------------------------- #


def build_hierarchy_tree(json_file: Path = DATA_FILE, visualise: bool = VISUALISE) -> None:
    """Runs the full hierarchy pipeline until one node remains."""
    tree = Tree.load(json_file)
    level_idx = 0
    use_custom_score = False

    while len(tree.levels[level_idx].nodes) > 1:
        # finds third nearest neighbor, make sure this function goes before get_nearest_neighbors_blossom
        find_third(tree, level_idx)

        # — Pair active nodes with blossom
        get_nearest_neighbors_blossom(tree, level_idx, custom_threshold=0.5, llm_model_name=MODEL_CONFIG["blossom"], api_keys=API_KEYS)

        # entail2 = [0]
        # while any((score < 0.3 for score in entail2)){ 
            # — Generate common summaries for survivors
        
        iteration = 0
        entail_2s = [True]
        threshold = 0.4
        while (iteration < 2 and any(entail_2s)):
            print("redoing: ")
            generate_summary(tree, level_idx, threshold, model_name=MODEL_CONFIG["summary_generation"], api_keys=API_KEYS)

            # — Evaluate summaries
            evaluate_common_summary(tree, level_idx, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)
            print(tree.levels[level_idx].nodes[0].summary)
            print(tree.levels[level_idx].nodes[0].entail_2)
            
            entail_2s = [score <= threshold for score in get_entail_2_score(tree, level_idx)]
            iteration +=1
        # get scores
        # print(f"ENTAIL2: {entail2}")
        # }
        evaluate_common_summary(tree, level_idx, eval_specificity=True, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)
        # — Show round statistics
        survivors = sum(1 for n in tree.levels[level_idx].nodes if n.alive)
        total     = len(tree.levels[level_idx].nodes)
        print(f"Level {level_idx} │ total: {total} │ survivors: {survivors}")

        record_stat(tree, level_idx)
        
        # — Promote winners to the next level
        promote_to_next_level(tree, level_idx)

        # - Reassignment
        reassign(tree, level_idx, 0.7, use_specificity_for_unmatched=True, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)
        
        # — Persist current state
        tree.dump(json_file)

        level_idx += 1
        
    propagate_attrs_up(tree, attrs=["score", "evaluation", "entail_1", "entail_2"])
    
    # attach scores to tree at the base
    add_scores(tree, SCORE_FILE)
    
    # bubble up the scores by averaging the children up
    build_scores(tree)
    
    tree.dump(json_file)

    # Optional viewer export / launch
    if visualise:
        viewer_json = generate_dag_json_from_tree(json_file)
        launch_tree_viewer(viewer_json)
        
def score_all_nodes(json_file: Path = DATA_FILE) -> None:
    """Re-evaluates all node summaries across all levels."""
    tree = Tree.load(json_file)
    for level_idx in range(len(tree.levels)):
        print(f"Scoring level {level_idx}...")
        evaluate_common_summary(tree, level_idx, force=True, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)
        evaluate_common_summary(tree, level_idx, eval_specificity=True, force=True, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)
        record_stat(tree, level_idx)

    propagate_attrs_up(tree, attrs=["score", "evaluation", "entail_1", "entail_2"])
    add_scores(tree, SCORE_FILE)
    build_scores(tree)
    
    tree.dump(json_file)
    print("Scoring complete. Output saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build hierarchy JSON and/or launch the visualiser"
    )
    parser.add_argument(
        "mode",
        choices=["run", "view", "score", "view-json"],
        nargs="?",
        default="run",
        help=(
            "'run' (default): build the hierarchy then open the viewer; "
            "'view': skip building and only open the current JSON in the viewer; "
            "'score': evaluate common summary for all nodes even if scores are present; "
            "'view-json': serve visualisation/ and open viewer for a DAG JSON basename (requires json_file)"
        ),
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="For view-json: DAG JSON file under visualisation/ (basename or path; only basename is used)",
    )
    args = parser.parse_args()

    if args.mode == "run":
        build_hierarchy_tree()
    elif args.mode == "score":
        score_all_nodes(Path(args.json_file) if args.json_file else DATA_FILE)
    elif args.mode == "view":
        # tree = Tree.load(DATA_FILE)
        # add_scores(tree, SCORE_FILE)
        # build_scores(tree)
        # tree.dump(DATA_FILE)
        # target_file = Path(args.json_file) if args.json_file else DATA_FILE
        # viewer_json = generate_dag_json_from_tree(target_file)
        # launch_tree_viewer(viewer_json)

        viewer_json = generate_dag_json_from_tree(DATA_FILE)
        launch_tree_viewer(viewer_json)

    elif args.mode == "view-json":
        if not args.json_file:
            parser.error("view-json requires a JSON path (e.g. python main.py "
                         "view-json demo_trees/wb_random.json)")
        target = Path(args.json_file)
        if target.exists():
            # repo-relative path (e.g. demo_trees/foo.json) — served from repo root
            data_ref = target.as_posix()
        else:
            # legacy: bare basename of a file already under visualisation/
            data_ref = target.name
        launch_tree_viewer(Path("."), viewer_data_basename=data_ref)