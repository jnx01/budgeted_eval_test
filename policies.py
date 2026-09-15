"""Evaluation policies: decide which tasks receive strong evaluation.

The big picture, in plain terms:

Every policy answers one question:

    "Given a budget of N strong evaluations, WHICH tasks get them?"

Once that choice is made, the final label for each task is simple:
  - if the task got a strong evaluation -> use the strong evaluator's label
  - otherwise                           -> use the scouts' label
    (their common label if they agree; scout 1's label if they disagree —
    a fixed, deterministic tie-break so results are reproducible)

The three policies:
    A. scouts_only          - never use the strong evaluator (budget 0)
    B. random_selection     - pick N tasks completely at random
    C. disagreement_aware   - pick scout-disagreement tasks first; if the
                              budget is bigger than the number of
                              disagreements, fill the rest at random
"""

import numpy as np
import pandas as pd


def scout_label(tasks: pd.DataFrame) -> pd.Series:
    """The label we use when a task gets NO strong evaluation.

    If the two scouts agree, use their common label.
    If they disagree, use scout 1's label (the spec's tie-break rule).
    """
    return tasks["scout_1_label"]


def final_labels(tasks: pd.DataFrame, strong_chosen: pd.Series) -> pd.Series:
    """Combine scout and strong labels into one final label per task.

    `strong_chosen` is a True/False column saying which tasks received a
    strong evaluation. Chosen tasks use the strong label; all others use
    the scout label.
    """
    return pd.Series(
        np.where(strong_chosen, tasks["strong_label"], scout_label(tasks)),
        index=tasks.index,
    )


def policy_scouts_only(tasks: pd.DataFrame) -> pd.Series:
    """Policy A: spend nothing. No task gets a strong evaluation."""
    return pd.Series(False, index=tasks.index)


def policy_random(tasks: pd.DataFrame, budget: int, rng: np.random.Generator) -> pd.Series:
    """Policy B: choose `budget` tasks at random for strong evaluation.

    `rng` is a numpy random generator passed in by the caller, so the
    simulation controls the seed and can repeat this many times.
    """
    chosen = pd.Series(False, index=tasks.index)
    if budget > 0:
        # Randomly pick `budget` row positions without replacement.
        positions = rng.choice(len(tasks), size=budget, replace=False)
        chosen.iloc[positions] = True
    return chosen


def policy_disagreement_aware(
    tasks: pd.DataFrame, budget: int, rng: np.random.Generator
) -> pd.Series:
    """Policy C: strong-evaluate disagreement cases first.

    Step 1: all tasks where the scouts disagree get strong evaluation
            (up to the budget).
    Step 2: if budget remains after that, fill it with randomly chosen
            agreement tasks. (Needed because disagreement is rare in our
            sample — only 7 of 100 tasks — so budgets above 7% always
            have leftover slots.)
    """
    chosen = pd.Series(False, index=tasks.index)
    if budget == 0:
        return chosen

    # Positions of disagreement tasks and agreement tasks.
    disagree_positions = np.flatnonzero(tasks["scout_disagree"].to_numpy())
    agree_positions = np.flatnonzero(~tasks["scout_disagree"].to_numpy())

    # Step 1: disagreement tasks first (all of them, or as many as fit).
    n_from_disagree = min(budget, len(disagree_positions))
    chosen.iloc[disagree_positions[:n_from_disagree]] = True

    # Step 2: fill any leftover budget with random agreement tasks.
    leftover = budget - n_from_disagree
    if leftover > 0:
        fill = rng.choice(agree_positions, size=leftover, replace=False)
        chosen.iloc[fill] = True

    return chosen
