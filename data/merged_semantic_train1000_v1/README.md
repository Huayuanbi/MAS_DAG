# Merged semantic topology training set v1

This directory contains 1,000 selected MMLU-Pro, GPQA-Diamond, and AIME
questions. Each question has seven manually designed graphs followed by five
role-constrained random DAGs. The random edges are sampled only from the
`allowed_edge` matrix in the corresponding unchanged role pool and satisfy its
`required_predecessors` rules.

Selection counts and provenance are recorded in `manifest.json`. Candidate
inputs are split into four 250-question files. Run one stochastic trial with:

```bash
RUN_INDEX=1 bash run_merged_semantic_train1000.sh
```

Use `RUN_INDEX=2` through `5` for independent Mean-5 trials. Every invocation
uses resumable outputs under `runs/`; logs are kept under
`logs/merged_semantic_train1000_v1/`.

The seven manual graphs include the established fixed BestMAS control for each
dataset: MMLU-Pro `star`, GPQA-Diamond `parallel_solvers_verify`, and AIME
`complete_dag`.
