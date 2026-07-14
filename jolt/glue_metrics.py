"""Official GLUE per-task metrics.

The GLUE score is the mean over tasks of each task's *primary* metric, which is NOT accuracy
for every task: CoLA uses Matthews correlation, MRPC/QQP use F1, STS-B (a regression task)
uses the mean of Pearson and Spearman, and the rest use accuracy. Reporting plain accuracy
everywhere (and treating STS-B as classification) produces a meaningless, deflated aggregate.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Majority-class dev accuracy for WNLI (Wang et al., 2018). BERT-family fine-tunes rarely beat
# this, so the GLUE paper and the MobileBERT paper exclude WNLI from the aggregate.
WNLI_MAJORITY_DEV_ACCURACY = 0.563

REGRESSION_TASKS = {"stsb"}

TASK_PRIMARY_METRIC = {
    "cola": "matthews",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "stsb": "pearson_spearman",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
}


def is_regression(task: str) -> bool:
    return task.lower() in REGRESSION_TASKS


def glue_primary_metric(task: str, preds: np.ndarray, labels: np.ndarray) -> Tuple[float, str]:
    """Return ``(metric_value, metric_name)`` using the task's official GLUE metric."""
    task = task.lower()
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    metric = TASK_PRIMARY_METRIC.get(task, "accuracy")

    if metric == "matthews":
        return float(matthews_corrcoef(labels, preds)), "matthews"
    if metric == "f1":
        return float(f1_score(labels, preds)), "f1"
    if metric == "pearson_spearman":
        pearson = pearsonr(preds, labels)[0]
        spearman = spearmanr(preds, labels)[0]
        return float(np.nanmean([pearson, spearman])), "pearson_spearman"
    return float(accuracy_score(labels, preds)), "accuracy"


def glue_aggregate(task_scores: Mapping[str, float], *, exclude_wnli: bool = True) -> float:
    """Mean GLUE score across tasks. WNLI is excluded by convention (the GLUE and MobileBERT
    papers report an 8-task average), since few methods beat its majority-class baseline.
    Including a naive WNLI fine-tune drags the mean by several points."""
    scores = {task.lower(): value for task, value in task_scores.items()}
    if exclude_wnli:
        scores.pop("wnli", None)
    if not scores:
        return 0.0
    return float(sum(scores.values()) / len(scores))
