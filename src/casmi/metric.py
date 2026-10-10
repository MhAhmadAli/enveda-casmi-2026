"""Competition metric: mean reciprocal rank at 25 over tautomer-canonical InChIKey14."""
from __future__ import annotations

from typing import Iterable, Sequence

K = 25


def reciprocal_rank(ranked_keys: Sequence[str | None], true_key: str, k: int = K) -> float:
    """First hit wins; duplicates upstream of the hit still consume a rank slot."""
    for i, key in enumerate(ranked_keys[:k]):
        if key is not None and key == true_key:
            return 1.0 / (i + 1)
    return 0.0


def mrr(ranked: Iterable[Sequence[str | None]], truth: Iterable[str], k: int = K) -> float:
    rrs = [reciprocal_rank(r, t, k) for r, t in zip(ranked, truth)]
    return sum(rrs) / len(rrs) if rrs else 0.0


def topk_accuracy(ranked: Iterable[Sequence[str | None]], truth: Iterable[str],
                  ks: Sequence[int] = (1, 5, 10, 25)) -> dict[int, float]:
    ranked, truth = list(ranked), list(truth)
    out = {}
    for k in ks:
        out[k] = sum(1.0 for r, t in zip(ranked, truth) if t in set(r[:k])) / max(len(truth), 1)
    return out
