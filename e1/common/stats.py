"""Exact paired tests used across E1/E2 analyses."""
from __future__ import annotations

import math


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial test of b vs c discordant counts under p = 0.5.

    This is the exact McNemar test (on paired correct/incorrect) and the sign test (on paired
    gains/losses); both reduce to the same formula."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


mcnemar_exact = binom_two_sided
sign_test = binom_two_sided


def paired_counts(ref: dict, arm: dict, ids, key: str = "score_bool") -> tuple[int, int]:
    """(b, c): b = ref right & arm wrong, c = ref wrong & arm right."""
    b = sum(bool(ref[q][key]) and not arm[q][key] for q in ids)
    c = sum((not ref[q][key]) and bool(arm[q][key]) for q in ids)
    return b, c
