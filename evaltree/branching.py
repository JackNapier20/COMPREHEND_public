"""Dependency-free helpers for EvalTree branching configuration."""


def split_count(num_samples: int, max_children: int) -> int:
    """Return the number of K-means clusters for one recursive split."""
    if max_children < 2:
        raise ValueError("EvalTree k/max_children must be at least 2")
    if num_samples <= 1:
        return 1
    return min(max_children, num_samples)
