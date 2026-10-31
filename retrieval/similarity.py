"""Small, dependency-free similarity helpers."""


def blend_similarities(text_similarity, idea_similarity, idea_weight: float = 0.2):
    if not 0.0 <= idea_weight <= 1.0:
        raise ValueError("idea_weight must be between 0 and 1")
    return (1.0 - idea_weight) * text_similarity + idea_weight * idea_similarity


def closest_outsider(similarities, self_idx: int, match_idx: int | None = None):
    excluded = {self_idx}
    if match_idx is not None:
        excluded.add(match_idx)
    candidates = [idx for idx in range(len(similarities)) if idx not in excluded]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: similarities[idx])
