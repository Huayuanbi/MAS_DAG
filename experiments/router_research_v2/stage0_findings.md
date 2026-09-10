# Stage-0 findings

Run date: 2026-08-31

Input: `data/gpqa/GPQA-train-mean5.json` (100 questions, 16 candidates,
five nominal trials per cell).

## Result

The exploratory progression gate failed.

| Comparison | Fold wins | Mean OOF selected-reward difference | Question bootstrap 95% interval |
|---|---:|---:|---:|
| Direct q,g minus fold-trained candidate prior | 3/5 | -0.010 | [-0.054, 0.034] |
| Direct q,g minus fixed `parallel_solvers_verify` | 1/5 | -0.088 | [-0.144, -0.034] |

Mean selected observed reward across the 100 out-of-fold questions was:

- direct q,g scorer: 0.416;
- fold-trained candidate prior: 0.426;
- fixed `parallel_solvers_verify`: 0.504;
- question-shuffled negative control: 0.402.

The direct scorer improves cell-level Brier/log loss relative to simple priors in
several folds, but that calibration improvement does not translate into better
top-1 action selection. This reproduces the earlier observation that ranking or
probability metrics and selected policy value can move in different directions.

## Interpretation boundary

This result does **not** prove that task-conditioned routing is impossible. It
shows that a regularized TF-IDF question-candidate interaction model cannot
extract a stable routing gain from the present 100-question labels. It also shows
that the previously reported 0.60 result on one repeatedly inspected 20-question
split does not survive a stronger grouped-CV baseline ladder.

The result is especially unfavorable because the fixed family beats the direct
scorer by 0.088 with a question-bootstrap interval below zero. Therefore the
agreed protocol does not authorize prospective arbitrary-graph generation yet.

## Immediate consequence

Do not spend the next GPU budget on another pairwise hyperparameter sweep or on
uniformly repeating the same 100 x 16 cells. The next useful work is:

1. audit whether the five nominal trials represent controlled stochastic draws;
2. build a larger reduced-factorial dataset across MMLU-Pro, AIME, and coding;
3. use a simpler fallback action space first: tuned SingleAgent versus one robust
   fixed MAS, with matched calls/tokens and no-communication controls;
4. revisit richer q,g models only after independent-question count and factor
   identifiability improve.

The machine-readable report is generated at
`artifacts/stage0_baselines.json` and is intentionally ignored by Git.
