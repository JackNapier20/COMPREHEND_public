"""Shared helpers for reproducible descendant-leaf evaluation."""


def sample_descendants(items, count: int, rng, *, repeat_when_short: bool = False):
    items = list(items)
    if count <= 0 or not items:
        return []
    if len(items) >= count:
        return rng.sample(items, count)
    if repeat_when_short:
        return [items[0]] * count
    return rng.sample(items, len(items))
