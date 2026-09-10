# Adversarial research debate

## Method and limitation

The main researcher formed Position A from the local evidence. An isolated,
read-only Codex session independently inspected the repository and formed
Position B. The environment did not expose the Claude MCP named by the
`codex-brainstorm` skill, so the debate used the main research process as side A
and independent Codex sessions as side B. No external model opinion is
misrepresented as Claude output.

## Initial positions

### Position A

Replace pairwise learning with a simple direct question-candidate listwise scorer,
add structural features, use grouped five-fold CV, enlarge question diversity,
and report a cost-quality Pareto curve. Avoid a complex probabilistic cost model
until routing signal is demonstrated.

### Position B

First learn a calibrated finite-library outcome model for correctness and cost,
then choose under an explicit budget. Pairwise/listwise objectives should be
auxiliary. Do not train an arbitrary topology generator before the finite-action
problem is statistically validated.

## Round 1: complexity versus uncertainty

Side A attacked weak identification with only 100 independent questions,
post-treatment/runtime-dependent cost, premature calibration machinery, and the
risk that adaptive replication misses rare interactions.

Side B conceded that a monolithic correctness-token-latency hierarchy was too
complex. It narrowed the proposal to a first gate comparing candidate/family
priors against prior plus question text, using grouped folds. Correctness and
tokens should be modular; latency is demoted unless runtime conditions are
standardized. Side B retained one attack: point-valued listwise labels can select
among statistically tied candidates and manufacture a winner's curse.

Status: partial convergence.

## Round 2: operational convergence

Side A updated to Beta-posterior targets, Monte Carlo propagation during listwise
training, and clustered question-level evaluation. It added provenance audit,
candidate-only/question-shuffled/topology-shuffled controls, leave-one-family-out
diagnostics, and a direct `f(question, graph)` scorer to distinguish model failure
from absence of routing signal.

Side B accepted that this is operationally an uncertainty-aware outcome model if
it predicts absolute success probability with a proper scoring rule. It fully
accepted the provenance audit and direct scorer. It distinguished grouped
question CV (new questions, known library) from leave-one-family-out evaluation
(unseen family transfer). Its remaining attack was repeated model selection on
only 100 questions: CV can justify the next experiment, not a routing claim.

Status: conditional convergence.

## Round 3: evidence boundary

Side A accepted the remaining attack and proposed:

1. Nested grouped CV on the current 100 exploratory questions.
2. One frozen procedure evaluated prospectively on the separate 20-question
   validation split.
3. The 78-question GPQA test remains untouched until the multi-dataset method,
   primary budget, controls, and analysis are locked.
4. Cross-dataset replication on MMLU-Pro, AIME, and execution-scored coding.
5. Strong matched-compute controls and reduced-factorial candidate generation.

Side B accepted all five conditions. The 20-question validation is a progression
gate only because it is underpowered: a nonpositive paired point estimate stops
arbitrary graph generation; a positive estimate only authorizes replication.

## Equilibrium

**Pure strategy equilibrium on the experimental protocol; conditional equilibrium
on the scientific conclusion.** Neither side retains a decisive attack against
the staged protocol. Any routing/topology conclusion remains conditional on
prospective, controlled, cross-dataset evidence.

The fallback is explicit: if text-conditioned selection fails the exploratory or
prospective gate, reduce the problem to routing between a tuned SingleAgent and
one robust fixed MAS. Do not claim arbitrary graph learning.

