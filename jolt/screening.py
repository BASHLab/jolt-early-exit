"""Leaderboard logic for the candidate-screening funnel.

Two ranking protocols live here:

- ``rank_candidates`` (legacy): per-cell composite = mean(EMAR rank desc, AURC rank asc).
  Backward-compat with Stage-1b runs and existing tests.
- ``rank_candidates_hv_primary`` + ``suite_winner`` (v4): per-cell ranking by 2D hypervolume
  on (accuracy, MACs); cross-cell winner = top-1 HV in ceil(N/2)+ cells AND on Pareto frontier
  in ceil(N/2)+ cells. EMAR and AURC are reported alongside at the operating point (the curve
  row maximising EMAR(p=2)). See docs_archive/research_sweep_prompt_v4.md Section 4 for the
  reviewer-defensibility framing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from .metrics import hypervolume_2d, per_sample_macs
from .pareto import pareto_frontier


def summarize_run(metrics: dict) -> Dict[str, float]:
    """Summarize one run's metrics.json over its curve: best EMAR/accuracy, lowest AURC/ECE/etc."""
    curve = metrics.get("curve", [])
    if not curve:
        return {"accuracy": 0.0, "emar": 0.0, "aurc": 1.0, "ece": 1.0, "nll": float("inf"), "brier": 2.0}
    return {
        "accuracy": max(r["accuracy"] for r in curve),
        "emar": max(r["emar"] for r in curve),
        "aurc": min(r.get("aurc", 1.0) for r in curve),
        "ece": min(r["ece"] for r in curve),
        "nll": min(r["nll"] for r in curve),
        "brier": min(r.get("brier", 2.0) for r in curve),
    }


def aggregate_seed_summaries(summaries: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Mean each metric across seeds for one candidate."""
    if not summaries:
        return {}
    keys = summaries[0].keys()
    return {k: float(sum(s[k] for s in summaries) / len(summaries)) for k in keys}


def _rank(names: List[str], values: Mapping[str, float], *, higher_better: bool) -> Dict[str, float]:
    """Dense-ish competition ranks (1 = best). Ties share the average rank."""
    ordered = sorted(names, key=lambda n: values[n], reverse=higher_better)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # average of positions [i..j], 1-indexed
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg_rank
        i = j + 1
    return ranks


def rank_candidates(candidate_summaries: Mapping[str, Mapping[str, float]]) -> List[Dict[str, float]]:
    """Rank candidates by composite = mean(EMAR rank desc, AURC rank asc); ties break on accuracy.

    Returns rows sorted best-first, each with name, the summarized metrics, the two ranks, and the
    composite score (lower is better).
    """
    names = list(candidate_summaries)
    if not names:
        return []
    emar = {n: candidate_summaries[n]["emar"] for n in names}
    aurc = {n: candidate_summaries[n]["aurc"] for n in names}
    emar_rank = _rank(names, emar, higher_better=True)
    aurc_rank = _rank(names, aurc, higher_better=False)
    composite = {n: (emar_rank[n] + aurc_rank[n]) / 2.0 for n in names}

    ordered = sorted(names, key=lambda n: (composite[n], -candidate_summaries[n]["accuracy"]))
    rows = []
    for n in ordered:
        row = {"candidate": n, "composite": composite[n], "emar_rank": emar_rank[n], "aurc_rank": aurc_rank[n]}
        row.update(candidate_summaries[n])
        rows.append(row)
    return rows


def top_k(ranked_rows: Sequence[Mapping], k: int) -> List[str]:
    """Names of the top-k candidates from a ranked leaderboard."""
    return [row["candidate"] for row in ranked_rows[:k]]


def collect_candidate_summaries(root) -> Dict[str, Dict[str, float]]:
    """Aggregate every ``<root>/<candidate>/seed*/metrics.json`` into per-candidate seed-mean
    summaries."""
    import json
    from pathlib import Path

    root = Path(root)
    candidates: Dict[str, Dict[str, float]] = {}
    for cand_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seed_summaries = [
            summarize_run(json.loads(mfile.read_text()))
            for mfile in sorted(cand_dir.glob("seed*/metrics.json"))
        ]
        if seed_summaries:
            candidates[cand_dir.name] = aggregate_seed_summaries(seed_summaries)
    return candidates


def write_leaderboard(root, out_csv) -> List[Dict[str, float]]:
    """Build and write the ranked leaderboard CSV; return the rows."""
    import csv
    from pathlib import Path

    rows = rank_candidates(collect_candidate_summaries(root))
    if rows:
        fieldnames = list(rows[0].keys())
        with Path(out_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------------------
# Hypervolume-primary ranking (v4 protocol)
# --------------------------------------------------------------------------------------

def summarize_run_extended(metrics: dict) -> Dict[str, Any]:
    """Per-seed summary that also exposes the (accuracy, MACs) curve points and the operating
    point at the EMAR-best row. Required for the v4 HV-primary protocol.

    Returns the same scalar fields as ``summarize_run`` plus:
    - ``op_accuracy`` / ``op_macs``: accuracy and average per-sample MACs at the curve row
      that maximises EMAR(p=2) (the deployment operating point used for AURC and EMAR
      reporting in the v4 protocol).
    - ``curve_points``: list of ``(accuracy, per_sample_macs)`` tuples over the full curve,
      consumed by ``hypervolume_2d``.
    """
    base = summarize_run(metrics)
    curve = metrics.get("curve", [])
    per_exit_macs = list(metrics.get("per_exit_macs", []))
    if not curve:
        return {**base, "op_accuracy": 0.0, "op_macs": 0.0, "curve_points": []}
    points = []
    for r in curve:
        ec = r.get("exit_counts") or []
        macs = per_sample_macs(ec, per_exit_macs) if (per_exit_macs and ec) else 0.0
        points.append((float(r.get("accuracy", 0.0)), float(macs)))
    # OP selection: prefer val_emar (Tier 2 honest selection) when available; fall back to
    # test_emar for backward compatibility with pre-Tier-2 metrics.json files.
    has_val = any("val_emar" in r for r in curve)
    if has_val:
        op_idx = max(range(len(curve)), key=lambda i: curve[i].get("val_emar", 0.0))
    else:
        op_idx = max(range(len(curve)), key=lambda i: curve[i].get("emar", 0.0))
    op_acc = float(curve[op_idx].get("accuracy", 0.0))
    op_macs = points[op_idx][1]
    return {**base, "op_accuracy": op_acc, "op_macs": op_macs, "curve_points": points,
            "op_selection": "val_emar" if has_val else "test_emar"}


def collect_candidate_summaries_extended(root) -> Dict[str, Dict[str, Any]]:
    """Per-method aggregation that preserves per-seed curve_points lists for HV computation.

    Scalar fields are seed-mean; ``curve_points_per_seed`` is the per-seed list of curves so
    downstream HV-per-seed averaging stays honest.
    """
    import json
    from pathlib import Path

    root = Path(root)
    candidates: Dict[str, Dict[str, Any]] = {}
    for cand_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_seed = [
            summarize_run_extended(json.loads(mfile.read_text()))
            for mfile in sorted(cand_dir.glob("seed*/metrics.json"))
        ]
        if not per_seed:
            continue
        scalar_keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
        agg: Dict[str, Any] = {
            k: float(sum(s[k] for s in per_seed) / len(per_seed)) for k in scalar_keys
        }
        agg["curve_points_per_seed"] = [s.get("curve_points", []) for s in per_seed]
        candidates[cand_dir.name] = agg
    return candidates


def _per_cell_hypervolumes(candidate_summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """For one cell, compute each method's HV: average over seeds of HV of that seed's
    (accuracy, MACs) curve relative to the cell-wide reference (0, max_op_macs * 1.01)."""
    names = list(candidate_summaries)
    if not names:
        return {}
    all_op_macs = [float(candidate_summaries[n].get("op_macs", 0.0)) for n in names]
    ref_macs = (max(all_op_macs) * 1.01) if all_op_macs and max(all_op_macs) > 0 else 1.0
    ref = (0.0, ref_macs)
    hv: Dict[str, float] = {}
    for n in names:
        curves = candidate_summaries[n].get("curve_points_per_seed") or [
            candidate_summaries[n].get("curve_points", [])
        ]
        if not curves:
            hv[n] = 0.0
            continue
        seed_hvs = [hypervolume_2d(pts, ref) for pts in curves if pts]
        hv[n] = float(sum(seed_hvs) / len(seed_hvs)) if seed_hvs else 0.0
    return hv


def rank_candidates_hv_primary(
    candidate_summaries: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """v4 per-cell ranking: hypervolume desc, accuracy desc, EMAR desc, AURC asc; tags
    ``on_frontier`` for the (op_accuracy, op_macs) Pareto check.

    ``candidate_summaries`` is the output of ``collect_candidate_summaries_extended`` (or any
    Mapping with the same keys: ``accuracy``, ``emar``, ``aurc``, ``op_accuracy``, ``op_macs``,
    and either ``curve_points`` or ``curve_points_per_seed``).
    """
    names = list(candidate_summaries)
    if not names:
        return []
    hv = _per_cell_hypervolumes(candidate_summaries)
    op_points = [
        (float(candidate_summaries[n].get("op_accuracy", 0.0)),
         float(candidate_summaries[n].get("op_macs", 0.0)))
        for n in names
    ]
    frontier_idx = set(pareto_frontier(op_points, (True, False)))
    on_frontier = {n: (i in frontier_idx) for i, n in enumerate(names)}

    ordered = sorted(
        names,
        key=lambda n: (
            -hv.get(n, 0.0),
            -float(candidate_summaries[n].get("accuracy", 0.0)),
            -float(candidate_summaries[n].get("emar", 0.0)),
            float(candidate_summaries[n].get("aurc", 1.0)),
        ),
    )
    rows: List[Dict[str, Any]] = []
    for n in ordered:
        row: Dict[str, Any] = {
            "candidate": n,
            "hypervolume": hv.get(n, 0.0),
            "on_frontier": on_frontier[n],
        }
        for k, v in candidate_summaries[n].items():
            if k != "curve_points_per_seed" and k != "curve_points":
                row[k] = v
        rows.append(row)
    return rows


def suite_winner(
    per_cell_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """v4 cross-cell HV + Pareto conjunctive gate.

    A method WINS the suite iff (a) it is top-1 on hypervolume in >= ceil(N/2) cells (ties
    allow multiple methods to share top-1 in a tied cell) AND (b) it sits on the (op_accuracy,
    op_macs) Pareto frontier in >= ceil(N/2) cells. Tiebreaker among multiple satisfying
    methods: HV mean rank across cells, lower better. ``per_cell_summaries`` maps cell name to
    that cell's ``collect_candidate_summaries_extended`` output.
    """
    methods = sorted({m for s in per_cell_summaries.values() for m in s})
    N = len(per_cell_summaries)
    threshold = math.ceil(N / 2) if N > 0 else 0
    hv_top_count = {m: 0 for m in methods}
    frontier_count = {m: 0 for m in methods}
    hv_rank_sum = {m: 0.0 for m in methods}
    hv_per_cell: Dict[str, Dict[str, float]] = {}

    for cell, summary in per_cell_summaries.items():
        cell_methods = list(summary)
        hv = _per_cell_hypervolumes(summary)
        hv_per_cell[cell] = hv
        if hv:
            best_hv = max(hv.values())
            for m in cell_methods:
                if hv[m] >= best_hv - 1e-12:
                    hv_top_count[m] = hv_top_count.get(m, 0) + 1
        op_points = [
            (float(summary[m].get("op_accuracy", 0.0)), float(summary[m].get("op_macs", 0.0)))
            for m in cell_methods
        ]
        frontier_idx = set(pareto_frontier(op_points, (True, False)))
        for i, m in enumerate(cell_methods):
            if i in frontier_idx:
                frontier_count[m] = frontier_count.get(m, 0) + 1
        ranks = _rank(cell_methods, hv, higher_better=True)
        for m, r in ranks.items():
            hv_rank_sum[m] = hv_rank_sum.get(m, 0.0) + r

    winners = [
        m for m in methods
        if hv_top_count.get(m, 0) >= threshold and frontier_count.get(m, 0) >= threshold
    ]
    winners.sort(key=lambda m: hv_rank_sum.get(m, float("inf")))
    return {
        "winners": winners,
        "hv_top_count": hv_top_count,
        "frontier_count": frontier_count,
        "hv_mean_rank": {
            m: (hv_rank_sum.get(m, 0.0) / N if N > 0 else 0.0) for m in methods
        },
        "threshold": threshold,
        "n_cells": N,
        "hv_per_cell": hv_per_cell,
    }
