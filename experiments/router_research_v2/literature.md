# Literature map

Only primary paper pages are used below. The table records the design lesson for
this repository rather than treating reported benchmark gains as directly
transferable.

| Work | Relevant result | Consequence here |
|---|---|---|
| [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | Represents agents as computational graphs and optimizes prompts and connectivity. | Supports graph-based formulation, but does not remove the need for held-out task-conditioned routing evidence. |
| [DyLAN](https://arxiv.org/abs/2310.02170) | Selects agents and uses query-dependent dynamic collaboration. | Supports testing question-conditioned team selection against fixed teams. |
| [AFlow](https://arxiv.org/abs/2410.10762) | Searches code-represented workflows with execution feedback. | Motivates workflow search only after the finite-library routing signal survives prospective gates. |
| [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435) | Searches agent programs and reports cross-domain/model transfer. | Makes transfer an explicit later-stage hypothesis; it is not implied by GPQA-only fitting. |
| [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) | Layered aggregation can improve generation quality. | Requires an aggregation-only control so gains are not misattributed to communication topology. |
| [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) | Sampling and voting improve with more agents on several tasks. | Requires best-of-k/self-consistency and no-communication voting controls at matched calls/tokens. |
| [Improving Multi-Agent Debate with Sparse Communication Topology](https://arxiv.org/abs/2406.11776) | Sparse connectivity can match or improve dense debate at lower cost. | Supports controlled edge-density interventions with fixed roles, calls, and aggregation. |
| [AgentPrune](https://arxiv.org/abs/2410.02506) | Prunes redundant communication and reports large token reductions. | Supports treating communication cost as a Pareto dimension and testing message-pruning baselines. |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | Learns cost-quality routing from preference data. | Supports a direct query-action scorer, but candidate preference accuracy alone is insufficient for top-1 policy value. |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) | Uses cascades to trade off model cost and quality. | Motivates explicit budget policies/cascades instead of one globally fixed cost penalty. |
| [NeuralNDCG](https://arxiv.org/abs/2102.07831) | Addresses mismatch between pairwise training and ranking metrics. | Supports a question-level listwise auxiliary loss because current pair accuracy is misaligned with top-1 selection. |
| [MMLU-Pro](https://arxiv.org/abs/2406.01574) | Provides harder, reasoning-focused, multi-domain questions with lower prompt sensitivity. | Best next source of question diversity and subject labels for routing studies. |
| [GPQA](https://arxiv.org/abs/2311.12022) | Expert-written, difficult science questions designed for scalable-oversight research. | Retain as a hard target and frozen final test, not as the sole training source. |

## Synthesis

The literature supports adaptive agents, workflow search, sparse communication,
and cost-aware routing, but these are different causal claims. The current
candidate families jointly change roles, prompts, edges, aggregation, node count,
and token caps. Consequently, a high-performing router would establish useful
policy selection over this library, not a causal benefit of graph topology.

The next datasets should therefore use a reduced factorial design that separates
at least role set, communication edges, aggregation rule, and compute cap. Search
over arbitrary workflows becomes justified only after controlled finite-library
routing beats strong priors and cost-matched non-communication controls.

