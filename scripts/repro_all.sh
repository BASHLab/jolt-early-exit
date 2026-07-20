#!/usr/bin/env bash
# Regenerate every paper table and figure from the evaluation outputs, then
# report the paper's derived numbers. Train and evaluate first (see README:
# scripts/screen.py for training, scripts/*_eval.py for budgeted / shift
# evaluation) so that outputs/ is populated for all cells.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts:${PYTHONPATH:-}"

# Tables (spliced into tables/) and figures (rendered into figures/).
python scripts/generate_budget_table.py    --insert
python scripts/generate_ablation_tables.py  --insert
python scripts/generate_shift_tables.py     --insert --figure
python scripts/generate_budget_curves_figure.py
python scripts/generate_policy_drift.py                   # policy overspend json
python scripts/make_shift_fig.py                          # five-policy overspend panel
python scripts/make_pareto_grid.py                        # per-comparison Pareto grid

# Report the paper's derived numbers under the symmetric tuned-baseline regime
# (win/tie/loss partition, per-baseline deficit, shift envelope, interaction,
# calibration, Pareto membership, severest-shift ahead count).
python scripts/paper_numbers.py

echo "Done. Tables in tables/, figures in figures/."
