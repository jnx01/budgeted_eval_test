"""Run all experiments and save results.

Run from the repo root:

    .venv/bin/python experiments/run_experiments.py

What it produces:
    results/results.csv                      - error of each policy at each budget
    results/agreement_analysis.csv           - Experiment 1 numbers
    results/figures/plot1_agreement_error.png   - agree vs disagree error bars
    results/figures/plot2_budget_curve.png      - error vs budget (main figure)
    results/figures/plot3_allocation_table.png  - allocation efficiency table
"""

import sys
from pathlib import Path

# Make the repo root importable, so `from src...` works no matter where
# this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# Use a backend that saves files without needing a display window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import simulate
from src.data import (
    ALT_SCOUT_1_MODE,
    ALT_SCOUT_2_MODE,
    disagreement_summary,
    load_tasks,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Consistent names/colors for the three policies across all plots.
POLICY_STYLE = {
    "scouts_only": {"label": "Scouts only", "color": "#888888"},
    "random": {"label": "Random", "color": "#1f77b4"},
    "disagreement_aware": {"label": "Disagreement-aware", "color": "#d62728"},
}


def plot1_agreement_error(analysis: dict, path: Path) -> None:
    """Plot 1: scout error rate when the two scouts agree vs disagree.

    Answers: are disagreement cases actually harder / less reliable?
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Scouts agree", "Scouts disagree"]
    errors = [analysis["error_when_agree"], analysis["error_when_disagree"]]
    counts = [analysis["n_agree"], analysis["n_disagree"]]

    bars = ax.bar(categories, errors, color=["#4c9f70", "#d62728"], width=0.5)
    ax.set_ylabel("Scout error rate vs human benchmark")
    ax.set_ylim(0, max(errors) * 1.25)
    ax.set_title("Disagreement cases are harder for the scouts")

    # Label each bar with the exact value and the sample size behind it.
    for bar, err, n in zip(bars, errors, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{err:.1%}\n(n={n})", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot2_budget_curve(results: pd.DataFrame, path: Path) -> None:
    """Plot 2 (main figure): evaluation error vs strong-evaluation budget.

    One line per policy. Shaded band = 95% confidence interval over the
    500 random trials (only visible for the two randomized policies).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy, style in POLICY_STYLE.items():
        subset = results[results["policy"] == policy].sort_values("budget_pct")
        ax.plot(subset["budget_pct"], subset["mean_error"],
                marker="o", color=style["color"], label=style["label"])
        ax.fill_between(subset["budget_pct"], subset["ci_low"], subset["ci_high"],
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Strong-evaluation budget (% of tasks)")
    ax.set_ylabel("Evaluation error vs human benchmark")
    ax.set_title("Error vs strong-evaluation budget")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot3_allocation_table(results: pd.DataFrame, path: Path) -> None:
    """Plot 3: allocation efficiency as a table figure.

    For each budget above 0%: random error, disagreement-aware error, and
    the difference (positive = disagreement-aware is better).
    """
    random_rows = results[results["policy"] == "random"].set_index("budget_pct")
    disagree_rows = results[results["policy"] == "disagreement_aware"].set_index("budget_pct")

    budgets = [b for b in sorted(random_rows.index) if b > 0]
    table_rows = []
    for b in budgets:
        r_err = random_rows.loc[b, "mean_error"]
        d_err = disagree_rows.loc[b, "mean_error"]
        table_rows.append([f"{b}%", f"{r_err:.1%}", f"{d_err:.1%}",
                           f"{100 * (r_err - d_err):+.1f} pp"])

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Budget", "Random error", "Disagreement-aware error",
                   "Difference (pp)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Allocation efficiency: same budget, different selection", pad=12)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tasks...")
    tasks = load_tasks()
    print("Disagreement summary:", disagreement_summary(tasks))

    # --- Experiment 1: is disagreement predictive of error? ---
    analysis = simulate.agreement_error_analysis(tasks)
    print("\nExperiment 1 - agreement vs disagreement error:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    pd.DataFrame([analysis]).to_csv(RESULTS_DIR / "agreement_analysis.csv",
                                    index=False)

    # --- Experiments 2 & 3: policy comparison across budgets ---
    print("\nRunning budget simulation (500 trials per policy/budget)...")
    results = simulate.run_simulation(tasks)
    results.to_csv(RESULTS_DIR / "results.csv", index=False)
    print(results.to_string(index=False))

    # --- Plots ---
    plot1_agreement_error(analysis, FIGURES_DIR / "plot1_agreement_error.png")
    plot2_budget_curve(results, FIGURES_DIR / "plot2_budget_curve.png")
    plot3_allocation_table(results, FIGURES_DIR / "plot3_allocation_table.png")
    print(f"\nSaved results to {RESULTS_DIR}/ and figures to {FIGURES_DIR}/")

    # --- Experiment 4 (robustness): same study with a different scout pair ---
    # Main study scouts: two GPT-3.5 modes. Here we swap scout 2 for a
    # Gemini mode, so the two scouts are different MODELS, not just
    # different prompts. If the disagreement pattern survives this change,
    # the finding is more convincing. The adjudicator stays the same.
    print("\n" + "=" * 60)
    print("Experiment 4 - robustness with alternative scout pair")
    print(f"  scout 1: {ALT_SCOUT_1_MODE}")
    print(f"  scout 2: {ALT_SCOUT_2_MODE}")
    print("=" * 60)

    alt_tasks = load_tasks(scout_2_mode=ALT_SCOUT_2_MODE)
    print("Disagreement summary:", disagreement_summary(alt_tasks))

    alt_analysis = simulate.agreement_error_analysis(alt_tasks)
    print("\nAgreement vs disagreement error (alternative pair):")
    for key, value in alt_analysis.items():
        print(f"  {key}: {value}")
    pd.DataFrame([alt_analysis]).to_csv(
        RESULTS_DIR / "agreement_analysis_exp4.csv", index=False)

    alt_results = simulate.run_simulation(alt_tasks)
    alt_results.to_csv(RESULTS_DIR / "results_exp4.csv", index=False)
    print(alt_results.to_string(index=False))

    plot2_budget_curve(alt_results, FIGURES_DIR / "plot2_budget_curve_exp4.png")
    print(f"\nSaved Experiment 4 results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
