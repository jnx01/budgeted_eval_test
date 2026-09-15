"""Budget simulation: run each policy at several strong-evaluation budgets.

What this does, in plain terms:

We ask: "if we could afford strong evaluation on only X% of the 100 tasks,
how accurate would each policy be?"

Budgets tested: 0%, 10%, 20%, 30%, 40% of tasks (i.e. 0, 10, 20, 30, 40
strong evaluations out of 100 tasks).

- Policy A (scouts only) never uses the strong evaluator, so its result is
  the same at every budget. It is the baseline.
- Policy B (random) picks tasks at random, so its result depends on luck.
  We repeat it 500 times with different random seeds and report the average
  error and a 95% confidence interval.
- Policy C (disagreement-aware) is deterministic up to 7% (it always picks
  the same 7 disagreement tasks). Above that, leftover slots are filled
  randomly, so we also repeat it 500 times and average.

This is cheap to run because all evaluator labels already exist in the
dataset — no model is ever called.
"""

import numpy as np
import pandas as pd

from src import metrics, policies

# Budgets as fractions of the task count: 0%, 10%, 20%, 30%, 40%.
BUDGET_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]

# How many times we repeat the random policies at each budget.
N_TRIALS = 500

# Master seed for the whole simulation, so results are reproducible.
SIMULATION_SEED = 123


def _summarize_trials(errors: list[float]) -> dict:
    """Turn 500 trial errors into an average plus a 95% confidence interval.

    The confidence interval tells us how much the result varies due to the
    random task selection: we take the 2.5th and 97.5th percentiles of the
    trial errors (the middle 95%).
    """
    errors = np.array(errors)
    return {
        "mean_error": float(errors.mean()),
        "ci_low": float(np.percentile(errors, 2.5)),
        "ci_high": float(np.percentile(errors, 97.5)),
    }


def run_simulation(
    tasks: pd.DataFrame,
    budget_fractions: list[float] = BUDGET_FRACTIONS,
    n_trials: int = N_TRIALS,
    seed: int = SIMULATION_SEED,
) -> pd.DataFrame:
    """Run all three policies at all budgets. Returns one long DataFrame
    with one row per (policy, budget) combination."""
    human = tasks["human_label"]
    n_tasks = len(tasks)

    # The scouts-only baseline: one number, same at every budget.
    baseline_labels = policies.final_labels(tasks, policies.policy_scouts_only(tasks))
    baseline_error = metrics.error_rate(baseline_labels, human)

    # One random generator for the whole run, seeded once -> reproducible.
    rng = np.random.default_rng(seed)

    rows = []
    for fraction in budget_fractions:
        # Convert the fraction into a number of tasks (e.g. 0.2 * 100 = 20).
        budget = int(round(fraction * n_tasks))

        # --- Policy A: scouts only (baseline, no randomness) ---
        rows.append({
            "policy": "scouts_only",
            "budget_pct": int(round(100 * fraction)),
            "budget_n": budget,
            "mean_error": baseline_error,
            "ci_low": baseline_error,
            "ci_high": baseline_error,
        })

        # --- Policy B: random selection (repeated n_trials times) ---
        random_errors = []
        for _ in range(n_trials):
            chosen = policies.policy_random(tasks, budget, rng)
            labels = policies.final_labels(tasks, chosen)
            random_errors.append(metrics.error_rate(labels, human))
        rows.append({
            "policy": "random",
            "budget_pct": int(round(100 * fraction)),
            "budget_n": budget,
            **_summarize_trials(random_errors),
        })

        # --- Policy C: disagreement-aware (repeated; only matters above 7%) ---
        disagree_errors = []
        for _ in range(n_trials):
            chosen = policies.policy_disagreement_aware(tasks, budget, rng)
            labels = policies.final_labels(tasks, chosen)
            disagree_errors.append(metrics.error_rate(labels, human))
        rows.append({
            "policy": "disagreement_aware",
            "budget_pct": int(round(100 * fraction)),
            "budget_n": budget,
            **_summarize_trials(disagree_errors),
        })

    results = pd.DataFrame(rows)

    # Add the improvement-over-baseline column (in percentage points).
    results["improvement_vs_scouts_only"] = [
        metrics.improvement_over_baseline(err, baseline_error)
        for err in results["mean_error"]
    ]
    return results


def agreement_error_analysis(tasks: pd.DataFrame) -> dict:
    """Experiment 1: is scout disagreement predictive of error?

    Compares the scouts' error rate on tasks where they AGREE vs tasks
    where they DISAGREE. If disagreement cases have much higher error,
    disagreement is a useful warning signal.
    """
    scout_labels = policies.scout_label(tasks)
    human = tasks["human_label"]

    # A scout "error" = scout label differs from the human benchmark label.
    scout_wrong = scout_labels != human

    agree_mask = ~tasks["scout_disagree"]
    disagree_mask = tasks["scout_disagree"]

    return {
        "n_agree": int(agree_mask.sum()),
        "n_disagree": int(disagree_mask.sum()),
        "error_when_agree": float(scout_wrong[agree_mask].mean()),
        "error_when_disagree": float(scout_wrong[disagree_mask].mean()),
        # Also useful context: how good is the strong evaluator overall?
        "strong_error_overall": float((tasks["strong_label"] != human).mean()),
        "scout_error_overall": float(scout_wrong.mean()),
    }
