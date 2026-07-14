"""Pareto-frontier utilities for justifying EMAR (net-new; no source repo had these).

The argument the paper wants to make is that EMAR's single-scalar ranking of methods agrees
with multi-objective Pareto dominance on the (accuracy, compute) plane. These helpers compute
the non-dominated frontier and check that ranking by EMAR never contradicts dominance.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


def dominates(a: Sequence[float], b: Sequence[float], maximize: Sequence[bool]) -> bool:
    """True if point ``a`` Pareto-dominates point ``b`` under the per-objective senses.

    ``maximize[k]`` is True if objective ``k`` is better when larger (e.g. accuracy) and
    False if better when smaller (e.g. MACs).
    """
    if not (len(a) == len(b) == len(maximize)):
        raise ValueError("Point and maximize lengths must match.")
    at_least_as_good = all((x >= y) if m else (x <= y) for x, y, m in zip(a, b, maximize))
    strictly_better = any((x > y) if m else (x < y) for x, y, m in zip(a, b, maximize))
    return at_least_as_good and strictly_better


def pareto_frontier(points: Sequence[Sequence[float]], maximize: Sequence[bool]) -> List[int]:
    """Return the indices of the non-dominated points."""
    n = len(points)
    return [
        i for i in range(n)
        if not any(dominates(points[j], points[i], maximize) for j in range(n) if j != i)
    ]


def pareto_emar_consistency(
    objectives: Mapping[str, Sequence[float]],
    emar: Mapping[str, float],
    maximize: Sequence[bool],
) -> Dict[str, object]:
    """Check that EMAR ranking is consistent with Pareto dominance across methods.

    For every ordered pair (a, b) where ``a`` dominates ``b`` on ``objectives``, EMAR should
    rank ``a`` at least as high as ``b``. Returns the Pareto-optimal method names, any
    violating pairs, and an overall consistency flag.
    """
    names = list(objectives)
    points = [objectives[name] for name in names]
    frontier_idx = set(pareto_frontier(points, maximize))

    violations: List[Tuple[str, str]] = []
    for a in names:
        for b in names:
            if a == b:
                continue
            if dominates(objectives[a], objectives[b], maximize) and emar[a] < emar[b]:
                violations.append((a, b))

    return {
        "frontier": [names[i] for i in sorted(frontier_idx)],
        "violations": violations,
        "consistent": len(violations) == 0,
    }
