import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from tree import Tree
from openai import OpenAI
import json
import re
import torch
import anthropic

from llm_utils import anthropic_text, together_chat_text
from retrieval.embedding_cache import load_or_create_embeddings
from retrieval.similarity import blend_similarities

# --- refusal/template detector (only when the prompt STARTS like a refusal) ---
_REFUSAL_START_PATTERNS = [
    r"^\s*i\s*apologize\b",                 # "I apologize ..."
    r"^\s*i(?:'|’)m\s*sorry\b",            # "I'm sorry ..."
    r"^\s*i\s*am\s*sorry\b",              # "I am sorry ..."
    r"^\s*i\s*(?:can(?:not|'t))\b",        # "I cannot / I can't ..."
    r"^\s*i\s*(?:will\s*not|won't)\b",    # "I will not / I won't ..."
    r"^\s*i\s*(?:do\s*not|don't)\s*feel\s*(?:comfortable|able)\b",  # "I don't feel comfortable ..."
    r"^\s*as\s+an\s+ai\b.*\b(?:can(?:not|'t)|will\s*not|won't)\b", # "As an AI, I cannot ..."
    r"^\s*this\s+request\b.*\b(?:violates|goes\s+against)\b.*\b(policy|guidelines)\b",
]
def _starts_with_refusal(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    for pat in _REFUSAL_START_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False

# Shared OpenAI client dynamic initialization in functions

# ---- Embedding device / VRAM knobs ----
USE_CUDA = False  # set True to use GPU if available
EMBED_BATCH_SIZE = 64  # lower this (e.g., 16/32) if you hit CUDA OOM; raise on CPU
IDEA_WEIGHT = 0.2

mainIdeaExtractPrompt = (
    """
    "You will be given a text. Extract the single most important main idea in one short sentence."
    "Return only that sentence."
    "Text:\n{txt}"
    """
)

mainIdeaExtractBatchPrompt = (
    """
You will be given a JSON object with key \"texts\" whose value is an array of texts.
For EACH text, extract the single most important main idea in ONE short sentence.

Output MUST be a JSON array of strings, same length and same order as the input texts.
Do NOT include any extra keys or commentary.

Input JSON:\n{payload}
    """
)


def _parse_main_idea_batch(raw: str, expected: int) -> list[str] | None:
    """Parse a batch response without assigning JSON formatting to nodes."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates = [text]
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(parsed, list)
            and len(parsed) == expected
            and all(isinstance(item, str) and item.strip() for item in parsed)
        ):
            return [item.strip() for item in parsed]
    return None


def _extract_main_idea_llm(text, batch_size: int = 8, llm_model_name: str = "openai", api_keys: dict = None):
    """Call the LLM to extract main idea(s).

    - If `text` is a string: returns a single string.
    - If `text` is a list of strings: returns a list of strings (batched via JSON prompt).

    Provider/model switching is still done by editing the `client...create(...)` call(s) below.
    """

    # --- batched path ---
    if isinstance(text, list):
        if not text:
            return []

        out: list[str] = []
        for start in range(0, len(text), batch_size):
            print(f"Extracting main ideas for batch {start} to {min(start + batch_size, len(text))}...")
            chunk = text[start:start + batch_size]
            payload = json.dumps({"texts": chunk}, ensure_ascii=False)

            if llm_model_name == "openai":
                client = OpenAI(
                    api_key=api_keys["openai"]
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional editor. Follow instructions exactly."},
                        {"role": "user", "content": mainIdeaExtractBatchPrompt.format(payload=payload)},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            elif llm_model_name == "claude":
                client = anthropic.Anthropic(
                    api_key=api_keys["claude"]
                )
                raw = anthropic_text(
                    client,
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system="You are a professional editor. Follow instructions exactly.",
                    messages=[
                        {"role": "user", "content": mainIdeaExtractBatchPrompt.format(payload=payload)},
                    ],
                )
            elif llm_model_name == "together":
                raw = together_chat_text(
                    api_keys["together"],
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional editor. Follow "
                                "instructions exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": mainIdeaExtractBatchPrompt.format(
                                payload=payload
                            ),
                        },
                    ],
                )
            elif llm_model_name == "gemma":
                client = OpenAI(
                    api_key=api_keys["gemma"]["api_key"],
                    base_url=api_keys["gemma"]["base_url"]
                )
                resp = client.chat.completions.create(
                    model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                    stop=["<end_of_turn>"],
                    messages=[
                        {"role": "system", "content": "You are a professional editor. Follow instructions exactly."},
                        {"role": "user", "content": mainIdeaExtractBatchPrompt.format(payload=payload)},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            elif llm_model_name == "gemini":
                client = OpenAI(
                    api_key=api_keys["gemini"]["api_key"],
                    base_url=api_keys["gemini"]["base_url"],
                )
                resp = client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": "You are a professional editor. Follow instructions exactly."},
                        {"role": "user", "content": mainIdeaExtractBatchPrompt.format(payload=payload)},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unknown llm_model_name: {llm_model_name}")

            parsed = _parse_main_idea_batch(raw, len(chunk))
            if parsed is None:
                print(
                    "Batch main-idea response was not a valid JSON array; "
                    "falling back to individual extraction."
                )
                parsed = [
                    _extract_main_idea_llm(
                        item,
                        batch_size=batch_size,
                        llm_model_name=llm_model_name,
                        api_keys=api_keys,
                    ).strip()
                    for item in chunk
                ]

            out.extend(parsed)

        return out

    if llm_model_name == "openai":
        client = OpenAI(
            api_key=api_keys["openai"]
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                {"role": "user", "content": mainIdeaExtractPrompt.format(txt=text)},
            ],
        )
        return resp.choices[0].message.content.strip()
    elif llm_model_name == "claude":
        client = anthropic.Anthropic(
            api_key=api_keys["claude"]
        )
        return anthropic_text(
            client,
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a professional editor. Read the following prompt carefully and respond to the best of your ability.",
            messages=[
                {"role": "user", "content": mainIdeaExtractPrompt.format(txt=text)},
            ],
        )
    elif llm_model_name == "together":
        return together_chat_text(
            api_keys["together"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional editor. Read the following "
                        "prompt carefully and respond to the best of your ability."
                    ),
                },
                {
                    "role": "user",
                    "content": mainIdeaExtractPrompt.format(txt=text),
                },
            ],
        )
    elif llm_model_name == "gemma":
        client = OpenAI(
            api_key=api_keys["gemma"]["api_key"],
            base_url=api_keys["gemma"]["base_url"]
        )
        resp = client.chat.completions.create(
            model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                    stop=["<end_of_turn>"],
            messages=[
                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                {"role": "user", "content": mainIdeaExtractPrompt.format(txt=text)},
            ]
        )
        return resp.choices[0].message.content.strip()
    elif llm_model_name == "gemini":
        client = OpenAI(
            api_key=api_keys["gemini"]["api_key"],
            base_url=api_keys["gemini"]["base_url"],
        )
        resp = client.chat.completions.create(
            model="gemini-3.1-flash-lite",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                {"role": "user", "content": mainIdeaExtractPrompt.format(txt=text)},
            ],
        )
        return resp.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown llm_model_name: {llm_model_name}")


def load_model(model_name='all-mpnet-base-v2', *, use_cuda: bool = False):
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer model: {model_name} on device={device} (USE_CUDA={use_cuda}, cuda_available={torch.cuda.is_available()})")
    m = SentenceTransformer(model_name, device=device)
    # Keep explicit `.to(...)` style, but don't fail if unsupported
    try:
        m = m.to(device)
    except Exception:
        pass
    return m

def extract_main_ideas(nodes, *, llm_model_name="openai", api_keys=None):
    """Populate and return leaf main ideas, reusing values already on nodes."""
    ideas = [None] * len(nodes)
    missing_positions = []
    missing_texts = []
    for pos, node in enumerate(nodes):
        if node.mainIdea:
            ideas[pos] = node.mainIdea
        else:
            missing_positions.append(pos)
            missing_texts.append(node.prompt or "")

    if missing_texts:
        extracted = _extract_main_idea_llm(
            missing_texts,
            batch_size=8,
            llm_model_name=llm_model_name,
            api_keys=api_keys,
        )
        for pos, idea in zip(missing_positions, extracted):
            cleaned = idea.strip() or "Unknown main idea"
            nodes[pos].mainIdea = cleaned
            ideas[pos] = cleaned

    return [idea or "Unknown main idea" for idea in ideas]


def get_nearest_neighbors_blossom(
    tree: Tree,
    level_idx: int,
    model_name: str = 'all-mpnet-base-v2',
    *,
    use_custom_score: bool = True,
    llm_model_name: str = "openai",
    api_keys: dict = None,
):
    model = load_model(model_name, use_cuda=USE_CUDA)
    level = tree.levels[level_idx]

    level_is_zero = getattr(tree.levels[level_idx], "level", level_idx) == 0
    # Apply custom scoring only on leaf level when enabled via `use_custom_score`.
    do_custom = bool(use_custom_score) and level_is_zero

    print("do_custom: ", do_custom)
    print("level_is_zero: ", level_is_zero)

    active_idxs = []                      # ← node indexes, not sheet rows
    corpus_sentences = []

    print("\nProcessing rows:")
    for idx, node in enumerate(level.nodes):
        if not node.alive:
            continue 
        val = node.prompt
        # print(f"Idx {idx}, Value: {val}")
        if val:
            if str(val).startswith("Error:"):
                continue
                # print(f"Skipping row {row} due to Error: prefix")
            else:
                # Flag-only: prompt starts like a refusal/template, but DO NOT exclude yet
                if _starts_with_refusal(val):
                    level.nodes[idx].summary_blocked = True
                active_idxs.append(idx)
                corpus_sentences.append(val)
                # print(f"Added row {row} to active_rows")

    print(f"\nTotal active rows: {len(active_idxs)}")
    print(f"Active row numbers: {active_idxs}")
    # print(f"Corpus sentences: {corpus_sentences}\n")
    
    if len(active_idxs) < 2:
        print("Not enough active nodes to match.")
        return

    corpus_sentences, corpus_embeddings, cache_path = load_or_create_embeddings(
        model,
        corpus_sentences,
        model_name,
        level_idx,
        batch_size=EMBED_BATCH_SIZE,
    )
    print(f"Embedding cache: {cache_path}")
    
    num_sentences = len(corpus_sentences)
    similarity_matrix = np.matmul(corpus_embeddings, corpus_embeddings.T)

    idea_similarity_matrix = None
    combined_similarity_matrix = similarity_matrix
    if do_custom:
        active_nodes = [level.nodes[idx] for idx in active_idxs]
        ideas = extract_main_ideas(
            active_nodes,
            llm_model_name=llm_model_name,
            api_keys=api_keys,
        )
        idea_embeddings = model.encode(
            ideas,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        denom = np.linalg.norm(idea_embeddings, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        idea_embeddings = idea_embeddings / denom
        idea_similarity_matrix = np.matmul(idea_embeddings, idea_embeddings.T)
        combined_similarity_matrix = blend_similarities(
            similarity_matrix, idea_similarity_matrix, IDEA_WEIGHT
        )

    G = nx.Graph()

    print("\nBuilding similarity graph:")
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            similarity = combined_similarity_matrix[i][j]
            G.add_edge(i, j, weight=float(similarity))
            # print(f"Edge between sentence {i} (row {active_idxs[i]}) and sentence {j} (row {active_idxs[j]}) with similarity {similarity}")
            
    print("\nRunning blossom algorithm")
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    # print(f"Matching pairs: {matching}")
    
    idx_mapping = {i: active_idxs[i] for i in range(len(active_idxs))}
    print(f"\nID mapping: { {i: level.nodes[j].id for i, j in idx_mapping.items()} }")

    # --- Handle unmatched nodes (virtual dummy pairing) -------------------
    # Blossom on an odd count leaves exactly one local index unmatched.
    # We flag it as paired-with-dummy and keep it alive so it promotes upward,
    # but we set match=None so downstream summary generation can skip it.
    all_local = set(range(len(active_idxs)))
    used_local = set()
    for a, b in matching:
        used_local.add(a)
        used_local.add(b)
    unmatched_local = sorted(all_local - used_local)

    for k in unmatched_local:
        real_idx = idx_mapping[k]
        n = level.nodes[real_idx]
        n.match = None                      # no real partner; treated as dummy-paired
        n.text_similarity = None
        n.idea_similarity = None
        n.score = None
        n.score2 = None
        n.alive = True                      # must remain alive to promote upward
        n.unmatched = True
        # (Promotion will use n.summary or fall back to n.prompt.)

    if unmatched_local:
        print(f"Unmatched (dummy-paired) nodes at level {level_idx}: {[idx_mapping[k] for k in unmatched_local]}")

    print("\nWriting results:")
    for i, j in matching:
        idx_i = idx_mapping[i]
        idx_j = idx_mapping[j]

        # Flag-only: if either child prompt starts like a refusal, mark for next-level summary flagging
        if _starts_with_refusal(level.nodes[idx_i].prompt) or _starts_with_refusal(level.nodes[idx_j].prompt):
            level.nodes[idx_i].next_summary_flagged = True
            level.nodes[idx_j].next_summary_flagged = True
        sim_score = float(similarity_matrix[i][j])

        # decide winner / loser – keep the lower index alive for stability
        if idx_i < idx_j:
            winner_idx, loser_idx = idx_i, idx_j
        else:
            winner_idx, loser_idx = idx_j, idx_i

        # Default values
        idea_sim = None
        custom_score = sim_score

        # Always compute main-idea similarity for leaf nodes (no threshold).
        if do_custom and idea_similarity_matrix is not None:
            try:
                idea_sim = float(idea_similarity_matrix[i][j])
                custom_score = float(combined_similarity_matrix[i][j])
            except Exception as e:
                # Keep graceful fallback; do not fail the pairing step
                print(f"[custom_score] Skipping idea similarity due to error: {e}")

        # Use custom score in place of sim_score for leaf nodes; otherwise keep sim_score.
        assigned_score = custom_score if do_custom else sim_score

        print(f"Pair: Node {level.nodes[winner_idx].id} (row {winner_idx}) ↔ Node {level.nodes[loser_idx].id} (row {loser_idx}) | "
              f"Text Sim: {sim_score:.4f} | Idea Sim: {idea_sim if idea_sim is not None else 'N/A'} | "
              f"Assigned Score: {assigned_score:.4f}")

        level.nodes[winner_idx].match = loser_idx
        level.nodes[winner_idx].score = sim_score
        level.nodes[winner_idx].score2 = assigned_score
        level.nodes[winner_idx].alive = True
        level.nodes[winner_idx].text_similarity = sim_score
        level.nodes[winner_idx].idea_similarity = idea_sim

        level.nodes[loser_idx].match = winner_idx
        level.nodes[loser_idx].score = sim_score
        level.nodes[loser_idx].score2 = assigned_score
        level.nodes[loser_idx].alive = False
        level.nodes[loser_idx].text_similarity = sim_score
        level.nodes[loser_idx].idea_similarity = idea_sim

    # wb.save(file_path)
    # tree.dump("10NNofASpecificity.json")
    # print(f"Results written to {file_path}")

    # return col
