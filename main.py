# ruff: noqa: E402
from pathlib import Path
import argparse
import json
import os


def _load_env(path: str = "") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments.
    Resolves .env next to this file, so it works regardless of the cwd."""
    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        print(f"[WARN] no .env found at {path} -- API keys will be empty")
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()
from common_summary_generation.summarize import generate_summary, get_entail_2_score
from evaluation.evaluate import evaluate_tree_file
from evaluation.simple_eval import evaluate_common_summary, promote_to_next_level, propagate_attrs_up, record_stat, reassign
from tree import Tree, Level
from evaltree.annotate import annotate_tree
from evaltree.clustering import RecursiveKMeans, build_hierarchy
from evaltree.summarize import summarize_tree
from evaltree.evaluate import run_evaluation_on_tree as run_evaltree_evaluation
from retrieval.blossom_matching import get_nearest_neighbors_blossom
from retrieval.third_neighbor import find_third
from visualisation.visualise_helper import (
    generate_dag_json_from_tree,
    launch_tree_viewer,
)
from score_retrieval import (
    add_scores,
    build_scores
)

# --------------------------------------------------------------------------- #
DATA_FILE = Path("Results/UF_random_prompts_64_base.json")
SCORE_FILE = "dataset_UF_with_scores.csv"   # for UltraFeedback: dataset_UF_with_scores.csv
VISUALISE = True                 # flip to False to skip viewer step
MODEL_CONFIG = {
    "blossom": os.environ.get("BLOSSOM_PROVIDER", "claude"),
    "summary_generation": os.environ.get(
        "SUMMARY_GENERATION_PROVIDER", "claude"
    ),
    "evaluation": os.environ.get("EVALUATION_PROVIDER", "openai"),
}
# Credentials come from .env (see .env.example); never hardcode keys here.
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gemma": {
        "api_key": os.environ.get("GEMMA_API_KEY", "DUMMY"),
        "base_url": os.environ.get("GEMMA_BASE_URL", "http://<GPU_HOST>:8000/v1"),
        # must match the vLLM server's registered name (its --model path,
        # unless --served-model-name is set)
        "model": os.environ.get("GEMMA_MODEL", "gemma-3-finetuned-merged-bf16"),
        "max_tokens": int(os.environ.get("GEMMA_MAX_TOKENS", "1024")),
        "timeout_seconds": float(os.environ.get("GEMMA_TIMEOUT_SECONDS", "600")),
        "main_idea_max_tokens": int(os.environ.get("GEMMA_MAIN_IDEA_MAX_TOKENS", "1024")),
        "main_idea_batch_size": int(os.environ.get("GEMMA_MAIN_IDEA_BATCH_SIZE", "1")),
        # in-flight requests; vLLM batches them on the server GPU
        "concurrency": int(os.environ.get("GEMMA_CONCURRENCY", "8")),
    },
    "gemini": {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "together": {
        "api_key": os.environ.get("TOGETHER_API_KEY", ""),
        "base_url": os.environ.get(
            "TOGETHER_BASE_URL", "https://api.together.xyz/v1"
        ),
        "model": os.environ.get("TOGETHER_MODEL", ""),
        "max_tokens": int(os.environ.get("TOGETHER_MAX_TOKENS", "4096")),
        "empty_retry_max_tokens": int(
            os.environ.get("TOGETHER_EMPTY_RETRY_MAX_TOKENS", "16384")
        ),
        "concurrency": int(os.environ.get("TOGETHER_CONCURRENCY", "8")),
        "reasoning_effort": os.environ.get(
            "TOGETHER_REASONING_EFFORT", "low"
        ),
    },
}
# --------------------------------------------------------------------------- #


def _with_suffix(json_file: Path, suffix: str) -> Path:
    """<name>.json -> <name><suffix>.json (unchanged if already suffixed)."""
    if json_file.stem.endswith(suffix):
        return json_file
    return json_file.with_name(json_file.stem + suffix + ".json")


def _tree_config(method: str) -> dict:
    cfg = {"method": method,
           "summary_generation": MODEL_CONFIG["summary_generation"],
           "evaluation": MODEL_CONFIG["evaluation"]}
    if method == "blossom":
        cfg["blossom"] = MODEL_CONFIG["blossom"]
    return cfg


def build_hierarchy_tree(json_file: Path = DATA_FILE, visualise: bool = VISUALISE) -> None:
    """Runs the full hierarchy pipeline until one node remains.

    The base file is never modified: work happens on a <stem>_blossom.json
    copy, whose "config" header records the providers used. An existing copy
    is resumed (completed levels are kept)."""
    out_file = _with_suffix(json_file, "_blossom")
    src = out_file if out_file.exists() else json_file
    if src is out_file and out_file != json_file:
        print(f"Resuming existing {out_file}")
    tree = Tree.load(src)
    tree.config = _tree_config("blossom")
    tree.dump(out_file)
    # A persisted tree contains every completed level. Resume from the highest
    # one instead of re-pairing level 0 and duplicating internal nodes.
    level_idx = len(tree.levels) - 1
    while len(tree.levels[level_idx].nodes) > 1:
        # — Pair active nodes with blossom
        get_nearest_neighbors_blossom(
            tree,
            level_idx,
            use_custom_score=True,
            llm_model_name=MODEL_CONFIG["blossom"],
            api_keys=API_KEYS,
        )

        # Resolve the specificity outsider after matching so the chosen match
        # is explicitly excluded from the candidate set.
        find_third(tree, level_idx)

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
        tree.dump(out_file)

        level_idx += 1
        
    propagate_attrs_up(tree, attrs=["score", "evaluation", "entail_1", "entail_2"])
    
    # attach scores to tree at the base
    add_scores(tree, SCORE_FILE)
    
    # bubble up the scores by averaging the children up
    build_scores(tree)
    
    tree.dump(out_file)

    # Optional viewer export / launch
    if visualise:
        viewer_json = generate_dag_json_from_tree(out_file)
        launch_tree_viewer(viewer_json)
        
def build_evaltree_tree(
    json_file: Path = DATA_FILE,
    visualise: bool = VISUALISE,
    k: int = 2,
) -> Path:
    """Builds an EvalTree hierarchy from the level-0 leaves of json_file:
    annotate each leaf with a capability gerund phrase, cluster the leaves
    with recursive k-means, summarize each cluster bottom-up, assign third
    neighbors, and judge every summary (entailment + specificity).
    Saves to <stem>_evaltree.json for k=2 (the historical default), or
    <stem>_evaltree_k<k>.json for other values, so runs never overwrite one
    another or a Blossom tree built from the same base."""
    if k < 2:
        raise ValueError("EvalTree k must be at least 2")
    suffix = "_evaltree" if k == 2 else f"_evaltree_k{k}"
    out_file = _with_suffix(json_file, suffix)
    base = Tree.load(json_file)
    leaves = base.levels[0].nodes
    for leaf in leaves:            # links are rebuilt by the clustering step
        leaf.parent = []
        leaf.children = []
    tree = Tree(levels=[Level(level=0, nodes=leaves)],
                next_id=max(n.id for n in leaves) + 1,
                config={**_tree_config("evaltree"), "k": k})
    tree.dump(out_file)

    # — Annotate leaves with capability phrases
    annotate_tree(tree, model_name=MODEL_CONFIG["summary_generation"], api_keys=API_KEYS)

    # — Recursive k-means clustering over capability embeddings
    rkm = RecursiveKMeans(max_children=k, max_depth=50)
    embeddings = rkm.embed([n.capability or n.prompt for n in leaves])
    clustering = rkm.cluster_recursive([n.id for n in leaves], embeddings)
    build_hierarchy(tree, clustering)

    # — Summarize every cluster, bottom-up
    summarize_tree(tree, model_name=MODEL_CONFIG["summary_generation"], api_keys=API_KEYS)

    # — Third neighbors (used by the post-hoc specificity evaluation)
    for level_idx in range(len(tree.levels)):
        if len(tree.levels[level_idx].nodes) > 1:
            find_third(tree, level_idx)

    # — Judge summaries: entailment of sampled leaves + sibling-outsider specificity
    run_evaltree_evaluation(tree, num_eval_samples=2,
                            model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)

    tree.dump(out_file)
    print(f"EvalTree hierarchy saved to {out_file}")

    if visualise:
        viewer_json = generate_dag_json_from_tree(out_file)
        launch_tree_viewer(viewer_json)
    return out_file


def score_tree(json_file: Path) -> None:
    """Post-hoc evaluation of a built tree (blossom or evaltree): faithfulness,
    informativeness and both specificity variants via evaluation/evaluate.py.
    For blossom trees, external per-model scores are (re)attached and
    aggregated up the tree first, when SCORE_FILE exists."""
    with open(json_file, encoding="utf-8") as fh:
        raw = json.load(fh)
    method = (raw.get("config") or {}).get("method")
    if method is None:  # older files without a config header
        method = "evaltree" if "_evaltree" in Path(json_file).stem else "blossom"

    if method == "blossom":
        if Path(SCORE_FILE).exists():
            tree = Tree.load(json_file)
            add_scores(tree, SCORE_FILE)
            build_scores(tree)
            tree.dump(json_file)
            print(f"Attached {SCORE_FILE} scores to {json_file}")
        else:
            print(f"[WARN] score file {SCORE_FILE} not found -- skipping score attachment")

    evaluate_tree_file(json_file, model_name=MODEL_CONFIG["evaluation"], api_keys=API_KEYS)


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
        choices=["run", "view", "score", "simplescore", "view-json"],
        nargs="?",
        default="run",
        help=(
            "'run' (default): build the hierarchy then open the viewer; "
            "'view': skip building and only open the current JSON in the viewer; "
            "'score': post-hoc faithfulness/informativeness/specificity evaluation of a "
            "built tree (blossom trees also get external scores attached first); "
            "'simplescore': re-run the in-pipeline (match-based) evaluation on all nodes; "
            "'view-json': serve visualisation/ and open viewer for a DAG JSON basename (requires json_file)"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["blossom", "evaltree", "both"],
        default="blossom",
        help="For run: which hierarchy method to build (default: blossom)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="EvalTree K-means branching factor (default: 2; ignored by Blossom)",
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="For view-json: DAG JSON file under visualisation/ (basename or path; only basename is used)",
    )
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be at least 2")

    if args.mode == "run":
        run_blossom = args.method in ("blossom", "both")
        run_evaltree = args.method in ("evaltree", "both")
        if run_blossom:
            # for "both", hold the (blocking) viewer until the last build
            build_hierarchy_tree(visualise=VISUALISE and not run_evaltree)
        if run_evaltree:
            build_evaltree_tree(k=args.k)
    elif args.mode == "score":
        if args.json_file:
            target = Path(args.json_file)
        else:
            blossom_copy = _with_suffix(DATA_FILE, "_blossom")
            target = blossom_copy if blossom_copy.exists() else DATA_FILE
        score_tree(target)
    elif args.mode == "simplescore":
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
