# Research plan: task-conditioned MAS routing

## 1. Reflection on the completed stage

The current work establishes a learnable association, not a validated routing
policy. On the reused internal 20-question subset, the best checkpoint selected
graphs with mean reward 0.60 versus 0.58 for the fixed
`parallel_solvers_verify`; the difference is less than one question at this sample
size. Other checkpoints ranged from 0.43 to 0.52. Pair accuracy improved under
cost-aware or strong-gap training while top-1 reward often worsened, proving an
objective mismatch.

The five nominal trials per graph produce reward steps of 0.2. Near probability
0.5, five independent Bernoulli trials have standard error about 0.224; a one-step
gap is therefore not a reliable hard preference. Moreover, trial independence is
not yet proven because decoding seed, temperature, model revision, batching, and
prompt hashes were not all retained in the compact mean-five records.

The current internal 80/20 split is useful for development but has been reused for
early stopping, hyperparameter comparison, and final reporting. It is exhausted
as a confirmatory set. The separate GPQA split manifest provides 100 train, 20
validation, and 78 frozen test questions; the latter two have not been used by the
mean-five selector study.

The cost-aligned control adds an important qualification. MMLU-Pro SingleAgent
with thinking reached 0.743 at about 2.6K tokens, slightly above retrospectively
selected fixed MAS families around 0.71--0.73 at 5.6K--6.2K tokens. On GPQA,
thinking SingleAgent fell to 0.39 with 65% truncation. Thus topology benefits are
dataset- and runtime-dependent, and long reasoning under a cap is a confounded
control unless truncation is addressed.

## 2. Claims and estimands

Keep four claims separate:

1. **Finite-library routing:** choose among declared graphs for a new question.
2. **Family transfer:** generalize to a held-out workflow family.
3. **Topology causality:** edges matter after roles, aggregation, calls, and token
   caps are controlled.
4. **Arbitrary graph generation:** generate a new graph outside the candidate
   library.

The current evidence concerns only claim 1, and only exploratorily.

Primary estimand for a declared execution distribution:

\[
p_{qg}=P(Y=1\mid q,g).
\]

Primary deployment result is selected expected correctness at a declared token
budget. Cost is a constraint or Pareto axis. A scalar
`correctness - lambda * tokens` is secondary sensitivity analysis, never the
headline result.

## 3. Staged program

| Stage | Falsifiable hypothesis | Required controls | Gate |
|---|---|---|---|
| 0: existing-data audit | Question text improves new-question selection beyond candidate/family priors. | Candidate prior, family prior, predeclared fixed family, question-shuffled negative control, direct q,g scorer. | Direct scorer beats both the fold-trained candidate prior and fixed family in at least 4/5 outer folds. |
| 1: label stability | Five-trial rankings are sufficiently reproducible for routing. | Audited stochastic seeds, randomized execution order, deeper repeats on an ambiguity panel. | Top-candidate agreement >=70% and nominal gap >=0.4 order agreement >=80%; otherwise abandon hard ranks. |
| 2: prospective GPQA validation | Frozen router has positive paired gain over the fixed prior policy. | Tuned Single, best-of-k/self-consistency, fixed BestMAS, no-communication ensemble, matched tokens and calls. | Nonpositive point estimate stops arbitrary graph work; positive result only authorizes replication. |
| 3: reduced-factorial datasets | Edges and roles have effects after compute and aggregation are controlled. | Orthogonal role, edge, aggregation, token/call factors. | Continue topology claims only when controlled edge contrasts are practically nontrivial and consistent. |
| 4: multi-dataset routing | Task-conditioned routing replicates beyond GPQA. | Dataset-specific models, pooled prior, leave-one-dataset-out transfer, matched-cost controls. | Gain must replicate on multiple domains with question-level uncertainty. |
| 5: final lock | The complete policy generalizes to untouched GPQA. | Frozen code, one primary budget, fixed seeds and baselines. | Evaluate 78-question test once and report regardless of outcome. |

## 4. Stage-0 model ladder

Run all models under identical outer question folds:

1. Random candidate.
2. Cheapest declared policy (diagnostic only when cost is realized post-treatment).
3. Global candidate-index prior.
4. Global family prior.
5. Candidate-only probabilistic scorer.
6. Text-only difficulty model; it must not route by itself.
7. Direct question-candidate interaction scorer.
8. Question-shuffled negative control.
9. Existing ideal-template/BCE scorer.
10. Direct probabilistic scorer plus posterior-Monte-Carlo listwise auxiliary loss.

Use a binomial likelihood or proper Brier/log-loss objective for absolute success
probability. Exclude graph IDs from the primary model. Use candidate indices only
as a declared finite-library prior. Evaluate top-1 selected reward, Brier/log loss,
posterior regret, selection stability, token use, and the Pareto frontier. All
uncertainty and resampling are clustered by question.

## 5. Data-generation redesign

### More questions before uniform repeats

The next marginal GPU budget should primarily increase independent questions.
Recommended order:

1. MMLU-Pro for domain breadth and subject-conditioned routing.
2. AIME for verification/revision behavior and exact evaluation.
3. HumanEval plus MBPP-like execution-scored coding for tester/critic roles.
4. GPQA as hard science validation and final transfer target.
5. GSM8K only as a cheap pipeline/saturation control.

Target 500--2,000 diverse questions across datasets before substantially
increasing model capacity.

### Trial provenance

Every execution must store:

- dataset/question ID and split;
- candidate specification and factor levels;
- model/tokenizer/runtime revisions;
- temperature, decoding seed, request seed, execution order and batch context;
- full prompt or stable prompt hash;
- planned per-node caps and realized input/output tokens;
- finish reason, retries, errors, latency, answer, and correctness.

Repeated deterministic outputs count as effective duplicates, not automatically
as independent trials.

### Adaptive replication

Start with 3--5 audited trials per cell. Add trials to training cells whose
posterior uncertainty can change the action or Pareto frontier. Preserve a random
replication tranche so rare interactions are not excluded by an exploitative
allocation rule. Never adapt replication using prospective validation or test
outcomes.

### Reduced factorial candidates

Do not bundle all design choices into family names. Construct an orthogonal or
fractional-factorial library across:

- role set/diversity;
- active-agent count;
- edge pattern/density;
- aggregation rule;
- per-node and total token/call cap.

Include matched controls: edge shuffle within legal DAGs, role permutation,
communication disabled with identical solver calls, voting without message
passing, single self-consistency, and best-of-k.

## 6. Training cautions

- Split and bootstrap by question, never by graph or pair.
- Nested CV must include preprocessing, hyperparameter selection, architecture
  choice, and threshold selection.
- Pair counts are not independent sample sizes.
- Do not delete ties or convert 0.2 gaps into unquestioned hard labels.
- Do not select checkpoints by pair accuracy; use out-of-fold selected value and
  calibration.
- Separate candidate prior signal from question-candidate interaction with
  negative controls.
- Do not interpret realized token correlations causally; short failed/truncated
  runs can look cheap.
- Record multiple comparisons and predeclare the primary model/budget before the
  prospective gate.
- Keep the 78 GPQA test inaccessible to exploratory scripts.
- Distinguish “routing among known graphs” from “generating new graphs” in every
  result and paper claim.

## 7. Decision tree

```text
Does q,g beat candidate/family priors in >=4/5 grouped CV folds?
├── No: route only between tuned Single and one fixed MAS; collect more questions.
└── Yes: freeze one procedure and run prospective GPQA validation.
    ├── Nonpositive paired gain: stop arbitrary graph generation.
    └── Positive paired gain: run reduced-factorial multi-dataset replication.
        ├── No controlled replication: retain finite-library routing claim only.
        └── Replicated controlled effects: lock method and evaluate frozen test once.
```
