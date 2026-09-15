"""Metrics: how good is a set of labels, compared to the human benchmark?

The spec deliberately keeps this simple — only three metrics:

1. accuracy    = fraction of tasks where our label matches the human
                 benchmark label
2. error       = 1 - accuracy
3. improvement = baseline error minus our error, in percentage points
                 (positive = we are better than the baseline)

Note on wording: the human label is a *benchmark reference*, not perfect
ground truth. So "error" always means "disagreement with the human
benchmark", nothing more.
"""

import pandas as pd


def accuracy(labels: pd.Series, human_labels: pd.Series) -> float:
    """Fraction of tasks where `labels` match the human benchmark labels."""
    return float((labels == human_labels).mean())


def error_rate(labels: pd.Series, human_labels: pd.Series) -> float:
    """Fraction of tasks where `labels` differ from the human benchmark."""
    return 1.0 - accuracy(labels, human_labels)


def improvement_over_baseline(our_error: float, baseline_error: float) -> float:
    """Error reduction compared to a baseline, in percentage points.

    Example: baseline error 22%, our error 17% -> improvement = +5.0
    (a positive number means we are better).
    """
    return round(100 * (baseline_error - our_error), 2)
