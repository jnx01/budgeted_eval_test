"""Data loading: load SummEval-LLMEval and build one row per comparison task.

What this module does, in plain terms:

1. Downloads the dataset from Hugging Face (only the first time; after that
   it is read from a local cache).
2. The raw dataset has 6 rows per task (one row per evaluator mode).
   We reshape ("pivot") it so each task becomes ONE row containing:
       task_id, premise, generator_1, generator_2,
       human_label, scout_1_label, scout_2_label, strong_label
3. We pick 100 tasks: one task from each of the 100 different premises,
   so the sample stays diverse (no premise is over-represented).
4. We add a `scout_disagree` column: True when the two scout evaluators
   gave different labels.

A "label" here means: which generator was preferred (0 = generator_1,
1 = generator_2). The human label is the human benchmark label — a reference,
not unquestionable truth.
"""

import pandas as pd
from datasets import load_dataset

# --- Names of things, defined once so the rest of the code never retypes them ---

DATASET_NAME = "bay-calibration-llm-evaluators/summeval-annotated-latest"

# The three evaluator modes we use, exactly as they appear in the dataset's
# "worker" column. (Verified with explore_dataset.py.)
SCOUT_1_MODE = "w_gpt-3.5-turbo-0125-score-only"
SCOUT_2_MODE = "w_gpt-3.5-turbo-0125-rate-explain"
STRONG_MODE = "w_gemini-pro-analyze-rate"

# Alternative scout pair for Experiment 4 (robustness check): one GPT-3.5
# mode and one Gemini mode, both using the simple "score-only" prompt.
ALT_SCOUT_1_MODE = "w_gpt-3.5-turbo-0125-score-only"
ALT_SCOUT_2_MODE = "w_gemini-pro-score-only"

# Fixed random seed so that "pick one task per premise" gives the same 100
# tasks every time anyone runs the code. This makes results reproducible.
SELECTION_SEED = 42


def load_raw_dataset() -> pd.DataFrame:
    """Download (or read from cache) the full dataset and return it as a
    pandas DataFrame with 9,000 rows: 1,500 tasks x 6 evaluator modes."""
    dataset = load_dataset(DATASET_NAME, split="train")
    return dataset.to_pandas()


def build_task_table(
    raw: pd.DataFrame,
    scout_1_mode: str = SCOUT_1_MODE,
    scout_2_mode: str = SCOUT_2_MODE,
    strong_mode: str = STRONG_MODE,
) -> pd.DataFrame:
    """Reshape the raw data from 6-rows-per-task to 1-row-per-task.

    For each task we keep:
      - premise, generator_1, generator_2  (what was being compared)
      - human_label                        (the human benchmark label)
      - scout_1_label, scout_2_label       (the two cheap evaluators)
      - strong_label                       (the expensive evaluator)

    The three evaluator modes are parameters so Experiment 4 can rerun the
    same pipeline with a different scout pair.
    """
    # Keep only the rows from the three evaluator modes we care about.
    modes_we_need = [scout_1_mode, scout_2_mode, strong_mode]
    rows = raw[raw["worker"].isin(modes_we_need)].copy()

    # Pivot: one row per task, one column per evaluator mode.
    # The value in each cell is that evaluator's label for that task.
    labels = rows.pivot(index="task", columns="worker", values="llm_label")
    labels = labels.rename(columns={
        scout_1_mode: "scout_1_label",
        scout_2_mode: "scout_2_label",
        strong_mode: "strong_label",
    })

    # Task-level facts that are the same on all 6 raw rows of a task
    # (premise, generators, human label). We take them from the first row
    # of each task. We also check the human label really is consistent.
    task_facts = raw.groupby("task").agg(
        premise=("premise", "first"),
        generator_1=("generator_1", "first"),
        generator_2=("generator_2", "first"),
        human_label=("human_label", "first"),
        # If a task has more than 1 distinct human label, something is wrong.
        _human_label_variants=("human_label", "nunique"),
    )

    # Data quirk we discovered while building this: 51 of the 1,500 tasks
    # have a human label that differs between their 6 raw rows (e.g. one row
    # says humans preferred generator_1, another row says generator_2, for
    # the exact same task). The human label is supposed to be a fixed
    # property of the task, so these rows contradict each other and we
    # cannot know which one is right. We simply drop those 51 tasks.
    # This is safe: every premise has 15 tasks and loses at most 4, so we
    # can still pick one task per premise for all 100 premises.
    inconsistent = task_facts["_human_label_variants"] > 1
    n_dropped = int(inconsistent.sum())
    if n_dropped > 0:
        print(f"[data] Dropping {n_dropped} tasks with inconsistent human labels "
              f"({len(task_facts) - n_dropped} tasks remain)")
    task_facts = task_facts[~inconsistent].drop(columns="_human_label_variants")

    # Join the task facts with the three evaluator labels.
    tasks = task_facts.join(labels).reset_index().rename(columns={"task": "task_id"})

    # Safety check: every task must have all three evaluator labels.
    label_columns = ["scout_1_label", "scout_2_label", "strong_label"]
    if tasks[label_columns].isna().any().any():
        raise ValueError("Some tasks are missing evaluator labels.")

    # Labels are read as floats by pandas; make them plain integers (0 or 1).
    for column in label_columns + ["human_label"]:
        tasks[column] = tasks[column].astype(int)

    # The disagreement flag: True when the two scouts gave different labels.
    tasks["scout_disagree"] = tasks["scout_1_label"] != tasks["scout_2_label"]

    return tasks


def select_tasks(tasks: pd.DataFrame, seed: int = SELECTION_SEED) -> pd.DataFrame:
    """Pick one task from each premise (100 premises -> 100 tasks).

    Why one per premise: the dataset has 15 tasks per premise, and picking
    100 tasks fully at random could over-represent some source articles.
    One per premise keeps the sample diverse. The choice within each premise
    is random but fixed by the seed, so it is reproducible.
    """
    selected = tasks.groupby("premise", group_keys=False).sample(n=1, random_state=seed)
    return selected.sort_values("task_id").reset_index(drop=True)


def load_tasks(
    scout_1_mode: str = SCOUT_1_MODE,
    scout_2_mode: str = SCOUT_2_MODE,
    strong_mode: str = STRONG_MODE,
) -> pd.DataFrame:
    """One-call helper: raw data -> one row per task -> 100 selected tasks.

    The evaluator modes are parameters so Experiment 4 can reuse the exact
    same pipeline with a different scout pair.
    """
    raw = load_raw_dataset()
    tasks = build_task_table(raw, scout_1_mode, scout_2_mode, strong_mode)
    return select_tasks(tasks)


def disagreement_summary(tasks: pd.DataFrame) -> dict:
    """Basic counts describing how often the two scouts disagree."""
    n_disagree = int(tasks["scout_disagree"].sum())
    n_total = len(tasks)
    return {
        "n_tasks": n_total,
        "n_disagree": n_disagree,
        "pct_disagree": round(100 * n_disagree / n_total, 1),
    }


if __name__ == "__main__":
    # Quick manual check: run `python -m src.data` from the repo root.
    selected = load_tasks()
    print(f"Selected {len(selected)} tasks from {selected['premise'].nunique()} premises")
    print(disagreement_summary(selected))
    print()
    print(selected.head())
