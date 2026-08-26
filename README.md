## 生成候选图

[`generate_candidates.py`](generate_candidates.py) 是新版通用入口。它根据
sample JSONL 和节点角色池生成供 `run_mas.py` 使用的分组候选图 JSON。
数据集类型不再写死在脚本中，通过 `--node-pool` 选择角色池，通过
`--evaluator` 选择答案评分类型。

输入文件每个非空行必须包含 `question` 和 `answer`。当前生成器要求
`answer` 使用 `####` 分隔参考解答与最终答案：

```json
{"question": "Solve ...", "answer": "Reference derivation ...\n#### \\frac{1}{2}"}
```

### MATH 示例

```bash
conda activate agp
cd /home/yz/Documents/MAS_DAG/MAS_DAG

python generate_candidates.py \
  --input data/math/sample_500.jsonl \
  --output data/math/candidate_graphs.json \
  --node-pool data/node_pools/math_6_roles.json \
  --evaluator math \
  --limit 500 \
  --random-count 5 \
  --seed 42
```

### GSM8K 示例

```bash
python generate_candidates.py \
  --input data/gsm8k/sample.jsonl \
  --output data/gsm8k/candidate_graphs.json \
  --node-pool data/node_pools/gsm8k_6_roles.json \
  --evaluator gsm8k \
  --limit 500 \
  --random-count 5 \
  --seed 42
```

`--evaluator math` 保留 LaTeX 最终答案；`--evaluator gsm8k` 会移除数值
答案中的千位逗号。生成结果会保存 evaluator，供 `run_mas.py --evaluator
auto` 自动选择评分逻辑。

旧版 GSM8K 专用实现保存在
[`generate_candidates_legacy.py`](generate_candidates_legacy.py)，仅用于复现
旧流程。新数据应统一使用 `generate_candidates.py`。

每道题的 12 张图由以下部分组成：

- 5 张固定 anchor：`chain`、`star`、`tree`、`complete_dag`、
  `sparse_random`；
- 2 张低成本基线：`finalizer_only`、`two_node`；
- 5 张随题目 seed 变化的 `random_dag`。

前 7 张结构对所有 Query 完全相同，只有后 5 张随机 DAG 随 Query 变化。
`--random-count K` 表示每道题最终生成 `7 + K` 张图。

当前 MATH 节点池的固定语义顺序为：

```text
Problem Analyst < Strategy Planner < Primary Solver < Alternative Solver
                < Symbolic/Proof Verifier < Finalizer
```

所有边都必须遵守该顺序，因此不会产生 `Primary Solver -> Problem Analyst`
之类的反向角色边。Finalizer 始终保留，并且每个活跃节点都存在一条到
Finalizer 的路径。随机 DAG 可以 mask 节点，但不能破坏这些约束。

生成成功后，输出文件会被原子替换。快速检查数量：

```bash
python - <<'PY'
import json

data = json.load(open("data/math/candidate_graphs.json"))
print("题目数：", len(data))
print("图总数：", sum(len(record["graphs"]) for record in data))
print("每题图数：", sorted({len(record["graphs"]) for record in data}))
PY
```

预期输出：

```text
题目数：500
图总数：6000
每题图数：[12]
```

## 使用 `run_mas.py` 执行和评分候选图

[`run_mas.py`](run_mas.py) 会读取候选 DAG，根据 `mask` 和 `edge_weight`
重新计算拓扑顺序，然后依次执行活跃 Agent。每个节点只接收图中直接前驱节点
的输出，不会看到未连接节点的信息。

执行完成后会记录：

- 最终答案、准确率和 reward；
- 每个节点的输入 token、输出 token 和推理时间；
- 每条边传递的 token 成本和归因时间；
- 整张图的总 token 数和 wall time。

### 评价方式

MATH 候选数据包含 `"evaluator": "math"`。使用 `--evaluator auto` 时，
`run_mas.py` 会自动启用 `math-verify`，对最终 `\\boxed{...}` 答案进行数学
等价判断，例如 `\\frac{2}{4}` 与 `\\frac{1}{2}` 会判为相同。

GSM8K 则继续使用数值 exact match。

默认 reward 为：

```text
reward = accuracy
       - token_penalty * total_tokens
       - time_penalty * wall_time_seconds
```

`--token-penalty` 和 `--time-penalty` 默认为零，因此默认 reward 就是
accuracy。无论惩罚系数是否为零，原始 token 和时间成本都会被记录。

### 检查 vLLM 服务

确认 8000 端口的 vLLM 服务可以访问：

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```

### 单图 smoke test

正式运行前，先执行一道题的一张图：

```bash
conda activate agp
cd /home/yz/Documents/MAS_DAG/MAS_DAG

python run_mas.py \
  --backend vllm \
  --model qwen3-8b \
  --tokenizer /data1/yz/MAS_DAG/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --input data/math/candidate_graphs.json \
  --output data/math/candidate_graphs_smoke_scored.json \
  --evaluator auto \
  --concurrency 1 \
  --max-new-tokens 2048 \
  --max-queries 1 \
  --max-graphs-per-query 1 \
  --store-node-outputs
```

`--store-node-outputs` 会保存每个 Agent 的完整输出，适合调试，但会显著增加
结果文件体积。

### 全量执行 6000 张图

后台运行：

```bash
nohup python run_mas.py \
  --backend vllm \
  --model qwen3-8b \
  --tokenizer /data1/yz/MAS_DAG/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --input data/math/candidate_graphs.json \
  --output data/math/candidate_graphs_scored.json \
  --evaluator auto \
  --concurrency 12 \
  --max-new-tokens 2048 \
  --checkpoint-every 5 \
  --resume \
  --retry-errors \
  > data/math/full_run.log 2>&1 &
```

查看日志和 GPU：

```bash
tail -f data/math/full_run.log
watch -n 1 nvidia-smi
```

参数说明：

- `--concurrency`：同时推进的候选 DAG 数量；不同 DAG 中当前可执行的节点请求
  会由 vLLM 动态组成 batch；
- `--max-new-tokens`：单个 Agent 请求最多生成的 token 数；
- `--checkpoint-every`：每处理多少个 Query 保存一次结果；
- `--resume`：输出文件存在时从中恢复，并跳过状态为 `completed` 的图；
- `--retry-errors`：重新执行之前状态为 `error` 的图；
- `--store-node-outputs`：保存所有 Agent 的原始输出；
- `--max-queries`：限制执行的 Query 数；
- `--max-graphs-per-query`：限制每个 Query 执行的图数；
- `--token-penalty`：token 成本在 reward 中的惩罚系数；
- `--time-penalty`：wall time 在 reward 中的惩罚系数。

中断后直接重新执行相同的全量命令即可续跑。

## 训练

Graph Transformer 是默认模型：

```bash
python train.py \
  --data data/math/candidate_graphs_scored.json \
  --objective auto
```

当数据中存在 reward 不同的候选图时，`--objective auto` 会自动使用 Pairwise
Reward 监督。

## 主要替换位置

- `MAS_DAG/features.py`：文本编码器与特征缓存；
- `MAS_DAG/model.py`：GCN、Graph Transformer 和边预测头；
- `MAS_DAG/decoder.py`：DAG 解码策略；
- `MAS_DAG/losses.py`：节点、边和 Pairwise Reward 损失；
- `MAS_DAG/topology_sampling.py`：候选图与随机 DAG 生成规则；
- `data/node_pools/*.json`：Agent 的 system prompt 和 user prompt。
