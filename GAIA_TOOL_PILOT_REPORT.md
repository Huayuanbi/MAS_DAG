# GAIA Qwen3 Tool-Node Pilot Report

Date: 2026-09-09

## Scope

- Model: `qwen3-4b-base` served by the existing vLLM endpoint. No Qwen3-8B
  checkpoint or endpoint was available on the machine.
- Tasks: 5 GAIA validation questions: 3 web-only and 2 XLSX.
- Graphs: 27 semantic candidate DAGs.
- Tool mode: text JSON ReAct inside each active capability node.
- Outputs: full node outputs, tool traces, token counts, wall time, and exact-match
  accuracy were retained.

Clean result files:

- `data/gaia/validation_web3_scored_qwen3_4b_fixed_web.json`
- `data/gaia/validation_sheet2_scored_qwen3_4b.json`

An earlier dependency-failure run is retained separately and excluded from this
analysis: `data/gaia/validation_web3_sheet2_scored_qwen3_4b.json`.

## Result summary

- Completed graphs: 27/27.
- Correct graphs: 5/27 (18.5%).
- Tasks solved by at least one graph: 1/5 (20%).
- Finalizer-only: 0/5.
- Tool-enabled graphs: 5/22 (22.7%).
- All five correct graphs belong to the same XLSX question. The second XLSX
  question and all three web questions remained unsolved.
- Total tool calls/attempts: 121.
- Total model tokens: 180,459.
- Sum of per-graph wall times: 899.4 seconds. This sum is not physical experiment
  elapsed time because candidate graphs ran concurrently.

## Topology quality and cost

| Topology | N | Accuracy | Calls/graph | Tokens/graph | Seconds/graph |
|---|---:|---:|---:|---:|---:|
| finalizer_only | 5 | 0.0% | 0.0 | 270 | 0.99 |
| primary_finalize | 5 | 20.0% | 3.2 | 4,285 | 23.85 |
| primary_verify_finalize | 5 | 20.0% | 3.4 | 4,805 | 28.65 |
| primary_compute_verify_finalize | 5 | 20.0% | 5.6 | 10,249 | 42.88 |
| web_primary_compute_verify_finalize | 5 | 20.0% | 8.2 | 11,389 | 55.57 |
| web_primary_verify_finalize | 2 | 50.0% | 9.5 | 12,738 | 69.87 |

The last row only exists for the two spreadsheet tasks, so its accuracy is not
directly comparable with five-task rows. On the one solvable spreadsheet question,
every tool graph was correct; extra verifier, web, and compute nodes added cost but
no accuracy. With the current default `reward=accuracy`, all five receive the same
reward, so pairwise training cannot learn the cheapest successful graph. A nonzero
token/time/tool penalty is required before producing training labels.

## Access control

Tool access is different per role and enforced by the executor, not just by prompt.
An undeclared tool returns `tool not allowed` and is recorded in the node trace.
The finalizer has no tools.

Observed unauthorized calls: 2. Both were rejected. This confirms enforcement, but
also proves that prompt-level role/tool descriptions do not eliminate behavioral
drift in the base model.

## Role behavior

| Role | Active | Calls | OK | Error | Active with zero calls |
|---|---:|---:|---:|---:|---:|
| web_researcher | 16 | 69 | 48 | 21 | 0 |
| spreadsheet_analyst | 10 | 30 | 22 | 8 | 0 |
| python_calculation_analyst | 10 | 14 | 8 | 6 | 7 |
| evidence_verifier | 17 | 8 | 4 | 4 | 14 |

The finalizer was active in all 27 graphs and made zero tool calls by construction.

Observed errors:

- repeated identical calls blocked: 16;
- tool-call limit reached: 9;
- malformed JSON tool calls: 9;
- unauthorized tools: 2;
- invalid URLs: 2;
- HTTP 404: 1.

Interpretation:

1. **Access alignment works.** Roles cannot execute tools outside their whitelist.
2. **Behavior alignment is weak.** The web role always attempted tools, but often
   repeated searches or exhausted its budget. The base model also emitted malformed
   JSON and fabricated answers from weak search results.
3. **Verifier behavior is mostly passive.** In 14/17 activations it used no tool.
   This is not automatically wrong because it can inspect predecessor evidence, but
   it means the current verifier does little independent corroboration.
4. **Computation is often unnecessary or ignored.** The calculation role used no
   tool in 7/10 activations. Adding it roughly doubled cost in several graph families
   without improving accuracy.
5. **Spreadsheet routing is promising.** The spreadsheet role consistently tried
   its tools and solved one question that the finalizer-only baseline missed.
6. **Web retrieval is the present bottleneck.** DuckDuckGo was inaccessible from
   the runtime environment, so the tool fell back to Bing. Search relevance and
   navigation were insufficient for the three multi-hop web questions.

## Next decisions

1. Run the same frozen 27 graphs with an instruction-tuned Qwen3-8B endpoint before
   changing prompts, so model-size/instruction effects remain identifiable.
2. Add a real search API or reproducible web snapshot. Bing HTML fallback is useful
   for smoke tests but not an adequate GAIA research environment.
3. Tighten the ReAct contract with one automatic JSON-repair attempt and explicit
   evidence fields (`claim`, `source`, `quote`, `confidence`).
4. Make verifier tool use conditional: require independent checking only when an
   upstream source URL exists, otherwise terminate cheaply.
5. Add cost-aware labels. Compare at least accuracy-first lexicographic utility and
   `accuracy - lambda_token*tokens - lambda_tool*calls - lambda_time*time`.
6. Do not add image/audio tasks until OCR/ASR or a Qwen3-VL backend is implemented.

## Conclusion

The two-level architecture works technically: MAS_DAG controls capability-node
topology, while a node performs a dynamic multi-tool ReAct trajectory. Tool access
control prevents unauthorized execution, but role-behavior alignment is not solved
by access control. On this base-model pilot, extra nodes frequently amplify cost and
bad evidence. The next scientifically valid comparison is the same candidate set on
instruction-tuned Qwen3-8B with a stronger, fixed retrieval backend.
