"""Signature-validated embedding caches shared by retrieval stages."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any


def sentences_signature(sentences: list[str]) -> str:
    """Return a stable digest for an ordered collection of sentences."""
    digest = hashlib.sha1()
    for sentence in sentences:
        digest.update(str(sentence).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def embedding_cache_path(
    model_name: str,
    level_idx: int,
    sentences: list[str],
    cache_dir: str | Path = "outputs",
) -> Path:
    """Build a cache name that cannot alias another level or node set."""
    safe_model = model_name.replace("/", "_")
    signature = sentences_signature(sentences)
    return Path(cache_dir) / (
        f"embeddings_{safe_model}_L{level_idx}_N{len(sentences)}_{signature[:8]}.pkl"
    )


def _normalize(embeddings):
    import numpy as np

    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def load_or_create_embeddings(
    model,
    sentences: list[str],
    model_name: str,
    level_idx: int,
    *,
    cache_dir: str | Path = "outputs",
    batch_size: int = 64,
    rebuild: bool = False,
) -> tuple[list[str], Any, Path]:
    """Load an exact cache hit or encode and persist normalized embeddings."""
    expected_sentences = list(sentences)
    expected_signature = sentences_signature(expected_sentences)
    cache_path = embedding_cache_path(
        model_name, level_idx, expected_sentences, cache_dir=cache_dir
    )

    if cache_path.exists() and not rebuild:
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            embeddings = cached.get("embeddings")
            valid = (
                cached.get("signature") == expected_signature
                and cached.get("sentences") == expected_sentences
                and embeddings is not None
                and len(embeddings) == len(expected_sentences)
            )
            if valid:
                return expected_sentences, _normalize(embeddings), cache_path
            print(f"[cache] Ignoring mismatched embedding cache: {cache_path}")
        except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
            print(f"[cache] Rebuilding unreadable embedding cache {cache_path}: {exc}")

    encoded = model.encode(
        expected_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = _normalize(encoded)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model_name,
                "level": level_idx,
                "sentences": expected_sentences,
                "embeddings": embeddings,
                "signature": expected_signature,
            },
            handle,
        )
    print(f"[cache] Stored {len(expected_sentences)} embeddings at {cache_path}")
    return expected_sentences, embeddings, cache_path
