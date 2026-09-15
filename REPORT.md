# Report: Budgeted Disagreement-Aware Evaluation

## The test, in plain words

> If we can only afford strong evaluation on a few tasks, is it better to
> pick the tasks where two cheap evaluators disagree, rather than picking
> tasks at random?

**How we test it:** We do not call any AI model. We use a public dataset
that already contains the answers of several AI evaluators on 1,500 tasks.
Using these answers, we simulate what would have happened if we had followed 
this budget rule. This makes the whole study free, fast, and repeatable.

**Important note:** We compare every evaluator against the **human
benchmark label**: the answer human annotators picked for each task. We
call it a "benchmark" and not "ground truth," because humans can be wrong or 
split too. It is a reference point we measure against.

## Setup

**The data:** Each of the 1,500 tasks is a comparison: two different AI
systems summarized the same news article; which summary is better? The
dataset (`bay-calibration-llm-evaluators/summeval-annotated-latest`) gives
us, for every task, the human benchmark answer plus the answers of 6
different AI evaluator variants.

One structural detail matters: **every single task is GPT-2 versus one of
15 other summarizers** (BART, T5, Pegasus, LEAD-3, and so on). GPT-2, an
old, weak model, is always one of the two options, and humans preferred
the other system's summary in about 68% of tasks. So every task is the same
kind of judgment: "is this summary better than GPT-2's?" The evaluators are
never comparing two equally strong systems; they are always grading against a fixed
weak baseline. Keep this in mind when interpreting the results.

**Our sample:** We use 100 tasks: one per news article. Here is why. The
1,500 tasks come from only 100 different news articles. Each article was
summarized by several AI systems, and the dataset compares those summaries
in pairs, giving 15 different comparison tasks per article (100 articles x
15 tasks = 1,500). If we picked 100 tasks fully at random, we might get
many tasks about the same few articles and our results would mostly
reflect those few articles. Picking one task per article keeps the sample
diverse.

While preparing the data we found 51 tasks whose human label was recorded
inconsistently: the same task marked as "generator 1 wins" in one row and
"generator 2 wins" in another. After dropping those, every article still had
plenty of tasks left.

**The evaluators:** Think of them as three graders with different
personalities:

| Role | Who it is | Analogy |
|---|---|---|
| Scout 1 (cheap) | GPT-3.5, asked for just a score | Quick helper, no explanation |
| Scout 2 (cheap) | GPT-3.5, asked to rate and explain | Quick helper who explains |
| Strong (adjudicator) | Gemini Pro, asked to analyze then rate | The "expert" we spend budget on |

We call the third one "strong" and not "better" on purpose because whether it
actually deserves that name is part of what we test.

**The three spending rules (policies):** Each rule decides which tasks get
sent to the strong evaluator. Every other task keeps the scouts' answer
(when the two scouts disagree and we're over-budget, we use scout 1's answer — a fixed tie-break,
like "when in doubt, trust helper 1"):

- **Scouts only** — spend nothing. The baseline.
- **Random** — pick tasks for the expert at random, like a lottery.
- **Disagreement-aware** — send the disagreement cases to the expert first.
  If budget is left over, fill it with random tasks.

**Budgets:** We test spending the strong evaluator on 0%, 10%, 20%, 30%,
and 40% of the 100 tasks. Why 0%? At 0% no tasks are sent to the
strong evaluator, so all three rules produce the exact same answers = the
scouts' answers. That's what tells us the error rate we start from *before* spending
anything, so we can see whether spending budget actually helps or hurts.

Because the random rule depends on luck, we
replay it 500 times per budget and report the average (plus a 95%
confidence interval — the range the result lands in 95% of the time).

## Experiment 1 — Do the scouts actually struggle where they disagree?

Before using disagreement as a signal, we check it means something. If the
scouts make mistakes just as often whether they agree or not, disagreement
is useless as a signal.

| Cases | How many | Scout error rate |
|---|---:|---:|
| Scouts agree | 93 | 30.1% |
| Scouts disagree | 7 | 57.1% |

**The signal is real:** Where the scouts disagree, they are nearly twice as
likely to clash with the human benchmark. Disagreement is like a smoke
alarm: it does not tell you exactly what is wrong, but it does point to the existence of trouble.

Two honest caveats. First, the alarm rarely rings: only 7 of 100 tasks, so
"57.1%" is really "4 out of 7"; a small sample. Second, remember that
every task in this dataset is "X versus GPT-2." The hard judgment the
scouts split on seems to be "this summary is better than GPT-2's, but is it
*enough* better?"; a judgment about degree, not about two evenly matched
options.

## Experiment 2 — Does spending on disagreements beat spending at random?

The percentages below are the error rate of the *final answers across all 100 tasks*.
Here is what that means using the first row as an example:

1. We pick 10 tasks to send to the strong evaluator — either 10 random
   tasks, or the disagreement tasks first.
2. Those 10 tasks get the strong evaluator's answer. The other 90 tasks
   keep the scouts' answer.
3. We now have 100 final answers. We check each one against the human
   benchmark and count how many clash with it.
4. That count, as a percentage, is the number in the table. Lower = better.

So "33.8%" means: *out of the 100 final answers produced under the random
rule, about 34 disagreed with the human benchmark.*

| Budget | Random error | Disagreement-aware error | Difference |
|---:|---:|---:|---:|
| 10% | 33.8% | 31.6% | +2.2 pp |
| 20% | 36.0% | 33.9% | +2.1 pp |
| 30% | 37.9% | 36.2% | +1.7 pp |
| 40% | 39.9% | 38.5% | +1.4 pp |

("pp" = percentage points. +2.2 pp means the disagreement-aware rule's
error is 2.2 points lower — roughly 2 fewer wrong answers out of 100.)

**Disagreement-aware wins at every budget.** The edge is modest but
consistent, and biggest at small budgets, which is expected
because at a 10% budget the disagreement cases make up most of what the
expert sees.

## Experiment 3 — The surprise: the "expert" is worse than the helpers

Now the plot twist. Look at the main figure
(`results/figures/plot2_budget_curve.png`): as we spend more budget, error
goes **up**, not down. Why?

| Evaluator | Error vs human benchmark |
|---|---:|
| Scouts (combined) | 32.0% |
| "Strong" evaluator | **52.0%** |

**The expensive evaluator is much worse than the cheap ones on this data.**
52% error is basically a coin flip. Going back to our analogy: it is like
paying for an expert grader who turns out to be worse than your two free
helpers. Every time we "spend budget," we replace a decent answer with a
coin flip, so no clever spending rule can make that a good deal.

And yet the disagreement-aware rule still beats random at every budget,
because it sends those tasks to the expert where the scouts are weakest
anyway.

## Experiment 4 — Does the signal work with a different pair of scouts?

The main study's two scouts are the same model (GPT-3.5) asked in two
different ways. What if the two scouts are *different models* — GPT-3.5 and
Gemini, both asked the same simple way? If disagreement is a general "this
task is hard" signal, the pattern should survive the swap.

**It does not.**

| | Same-model scouts | Cross-model scouts |
|---|---:|---:|
| How often they disagree | 7% | 29% |
| Scout error when agreeing | 30.1% | 32.4% |
| Scout error when disagreeing | **57.1%** | **31.0%** |

With two different models, disagreements become four times more common —
and stop meaning anything. The budget simulation agrees: the
disagreement-aware rule now *loses* to the random rule.

A way to think about it: if two teaching assistants trained by the same
professor disagree about a grade, the question is probably genuinely
tricky. If two strangers with different grading habits disagree, they might
just grade differently in general — their disagreement tells you nothing
about *this* essay. Cross-model disagreement is mostly that second kind:
noise, not signal.

So we say it plainly: **the disagreement signal is pair-specific, not
universal.** Anyone wanting to use it in practice would first need to check
that *their* pair of evaluators produces informative disagreements.

## Limitations

- **This is a replay, not a live system.** We used answers that already
  exist in a dataset.
- **Small numbers.** Only 7 disagreement cases in the main study.
- **One particular sample.** We picked one task per article with a fixed
  random seed. A different pick could shift the numbers.
- **Older models.** GPT-3.5 and Gemini Pro are dated. Newer models might
  disagree less, or give more informative disagreements.
- **The human benchmark is not perfect.** Humans disagree too, and 51
  tasks had contradictory recorded human labels. Sometimes disagreement
  means "this is genuinely ambiguous," not "someone made a mistake."
- **The signal depends on the pair.** Experiment 4 showed it disappears
  with a different scout pair.
- **Every task is "X vs GPT-2."** All comparisons grade against the same
  weak baseline, and humans preferred the other system 68% of the time.
  Our results describe this specific kind of comparison and will probably not carry
  over to comparing two evenly matched, modern systems.

## Conclusion

So, is disagreement a useful signal for spending a limited evaluation
budget? Our answer has three parts:

1. **Yes, the signal can be real.** For our main scout pair, disagreement
   marked the tasks where the scouts were nearly twice as likely to be
   wrong.
2. **But a signal is only as good as what it routes to.** Our "expert" was
   worse than the helpers, so the best strategy overall was to spend
   nothing. A smoke alarm is only useful if the fire department actually
   puts out fires.
3. **And the signal is fragile.** With a different scout pair, disagreement
   became common and meaningless.

For anyone building budgeted evaluation systems, the lesson is: do not just
ask "where do my evaluators disagree?" Ask two more questions:
"does disagreement actually predict mistakes for *my* evaluator pair?" and
"is my expensive evaluator actually better on those cases?" This project
gives a small, cheap, fully reproducible way to answer all three before
spending a single dollar on live evaluation.
