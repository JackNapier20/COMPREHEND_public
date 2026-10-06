import numpy as np
from sentence_transformers import SentenceTransformer
from tree import Tree
from retrieval.embedding_cache import load_or_create_embeddings
from retrieval.similarity import closest_outsider

def load_model(model_name='all-mpnet-base-v2'):
    print(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

def find_third(tree: Tree, level_idx: int, model_name='all-mpnet-base-v2'):
    """Assign each node's closest outsider, excluding itself and its match."""
    model = load_model(model_name)
    level = tree.levels[level_idx]

    node_idxs = []
    corpus_sentences = []
    for idx, node in enumerate(level.nodes):
        val = node.prompt
        if val and not str(val).startswith("Error:"):
            node_idxs.append(idx)
            corpus_sentences.append(val)

    if len(node_idxs) < 2:
        for node in level.nodes:
            node.third = None
            node.third_score = None
        return

    _, embeddings, cache_path = load_or_create_embeddings(
        model, corpus_sentences, model_name, level_idx
    )
    print(f"Third-neighbor embedding cache: {cache_path}")
    similarities = np.matmul(embeddings, embeddings.T)
    real_to_local = {
        real_idx: local_idx for local_idx, real_idx in enumerate(node_idxs)
    }

    for local_idx, real_idx in enumerate(node_idxs):
        node = level.nodes[real_idx]
        match_local = real_to_local.get(node.match)
        best_local = closest_outsider(
            similarities[local_idx], local_idx, match_local
        )
        if best_local is None:
            node.third = None
            node.third_score = None
            continue
        node.third = node_idxs[best_local]
        node.third_score = float(similarities[local_idx, best_local])
