# Cost-aligned single-agent control

This experiment tests whether the extra tokens and communication used by MAS are
worth their accuracy gain.  It is intentionally self-contained under this
directory; source candidate/scored files are read-only inputs and all generated
files go to `artifacts/`.

## Design

For each dataset:

1. Rank **multi-agent graph families** by macro accuracy over questions. A family
   must occur on every question and every graph must be completed. `finalizer_only`
   is excluded. Ties are broken by lower mean total tokens and then family name.
2. Select the top two families. The per-question single-agent budget is the pooled
   arithmetic mean of `total_input_tokens + total_output_tokens` over both
   families and all questions (rounded down to an integer at run time).
3. Run the same questions through one general solver twice: Qwen thinking disabled
   and enabled. For every request, the completion cap is
   `floor(MAS budget) - exact chat-template input tokens`, so input plus generated
   output cannot exceed the MAS budget.
4. Report accuracy, actual token use, budget utilization, truncation rate, and
   paired accuracy differences against each selected MAS family.

The budget is a ceiling, not a requirement to waste tokens. Consequently the
single agent may use fewer tokens by stopping early; the report exposes this.
Use the same model, temperature, seed, tokenizer, and question set as the MAS run.
The selected graph accuracy is descriptive because selection and evaluation use
the same questions. For confirmatory claims, generate the budget on a validation
split and pass a separate held-out question file to a future evaluation run.

## Reproduce

From the repository root:

```bash
python experiments/cost_aligned_single_agent/prepare.py \
  --dataset mmlu_pro=data/mmlu_pro/candidate_graphs_scored.json \
  --dataset gpqa=data/gpqa/train_candidate_graphs_scored_with_outputs.json

python experiments/cost_aligned_single_agent/run.py \
  --manifest experiments/cost_aligned_single_agent/artifacts/manifest.json \
  --model qwen3-8b \
  --tokenizer /path/to/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 --resume

python experiments/cost_aligned_single_agent/report.py
```

`run.py` executes both modes by default. Use `--mode thinking` or
`--mode no_thinking` for one group. Outputs are checkpointed independently at
`artifacts/results/<dataset>/<mode>.json`; two processes may therefore run the
two modes concurrently, but must not run the same dataset/mode simultaneously.

For a truncation-control ablation, `--mode thinking_concise` keeps Qwen thinking
enabled but adds an instruction to reason briefly, avoid repeated checks, and
reserve space for the final answer. It writes a separate result file and never
overwrites the original thinking group.
