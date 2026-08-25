# 数据生成说明

训练数据必须是一个 UTF-8 编码的 **JSON 数组**，不是 JSONL。数组中的每个顶层对象代表一个 Query，并在 `graphs` 中保存该 Query 的多张候选图。

可直接读取的示例见 [`example.json`](./example.json)；逐字段注释模板见 [`example.annotated.jsonc`](./example.annotated.jsonc)。JSONC 只用于阅读，生成的正式数据不能包含 `//` 注释。

## 字段定义

| 字段 | 类型和形状 | 含义 |
|---|---|---|
| `task` | `string` | 当前 Query 的完整文本。 |
| `nodes` | 长度为 `N` 的数组 | 内嵌候选 Agent；与 `node_pool` 二选一。数组下标就是三个边矩阵中的节点下标。 |
| `node_pool` | `string` | 外部节点池 JSON 的路径，相对于当前训练数据 JSON 所在目录；与 `nodes` 二选一。 |
| `nodes[i].id` | `string` 或 `int` | 样本内唯一的节点 ID。 |
| `nodes[i].role` | `string` | 简短角色名称，例如 `planner`。 |
| `nodes[i].role_brief` | `string` | 完整角色说明；默认由 `all-MiniLM-L6-v2` 在运行时生成 384 维 role feature。首次运行需下载模型，之后使用本地缓存。 |
| `graphs` | 非空数组 | 当前 Query 的候选图集合。 |
| `graphs[k].reward` | 有限浮点数或 `null` | 第 k 张图的整体效果，越大越好；用于构造 preferred/rejected pair。 |
| `graphs[k].mask` | `[N]` | `0` 表示保留节点，`1` 表示剪掉节点。 |
| `graphs[k].edge_weight` | `[N, N]` 浮点矩阵 | `[i][j]` 表示有向边 `nodes[i] -> nodes[j]`。当前 loss 将非零值视为有边。 |
| `graphs[k].edge_token_cost` | `[N, N]` 浮点矩阵 | `[i][j]` 表示沿边 `i -> j` 通信产生的 token 数。 |
| `graphs[k].edge_time_cost` | `[N, N]` 浮点矩阵 | `[i][j]` 表示沿边 `i -> j` 的耗时，单位为秒。 |

## 生成约束

1. 对每个候选图都有 `len(nodes) == len(graphs[k].mask) == N`。
2. 三个边矩阵的 shape 必须全部为 `[N, N]`。
3. 节点数组顺序必须与矩阵行列顺序严格一致。
4. `nodes[i].id` 在单条样本内必须唯一，`role_brief` 不能为空。
5. `mask` 只能使用 `0` 或 `1`。
6. 三个矩阵只能包含有限浮点数，不能出现 `NaN` 或 `Infinity`。
7. 对角线通常全部为 `0.0`，表示禁止自环。
8. 没有边时，对应的 `edge_weight`、`edge_token_cost`、`edge_time_cost` 建议都设为 `0.0`。
9. 被剪掉节点对应的行和列建议全部置零。
10. 如果目标要求 Query-rooted 连通 DAG，生成器需要额外保证无环，并保证每个保留节点都能从根节点到达；当前 Dataset 只校验格式，不自动校验这两个性质。
11. Pairwise 数据要求 `graphs` 中至少有两张 reward 不同的候选图；reward 相同的候选不会成对。

节点池文件格式为 `{ "id": "...", "finalizer_id": "...", "nodes": [...] }`。
`candidate_graphs.json` 当前引用 `../node_pools/gsm8k_6_roles.json`。修改角色或
节点顺序时应创建新的带版本节点池；矩阵行列下标依赖该顺序，不能直接重排旧节点池。

不同样本允许使用不同的 `N`。目前 `edge_token_cost` 和 `edge_time_cost` 只负责读取与保存，尚未加入训练 loss。
`--max-records` 限制的是顶层 Query 数量，不会截断某个 Query 内部的 `graphs`。

例如同一 Query 生成图 A、B、C，reward 分别为 `0.9、0.7、0.4`，会得到
`A > B`、`A > C`、`B > C` 三个监督 pair。建议 reward 是任务质量、token
成本和耗时经过统一规则聚合后的标量，并在整个数据集保持同一量纲。

## 快速检查

```bash
conda activate agp
cd /home/yz/Documents/MAS_DAG/AGP-minimal

python -c "from agp_minimal import AGPJsonDataset, PairwiseRewardDataset; d=AGPJsonDataset('data/example.json'); p=PairwiseRewardDataset(d); print(len(d), len(p), p[0].reward_gap)"

# auto 会检测到 reward pair，并自动选择 pairwise loss
python train.py --data data/example.json --max-records 1 --objective auto
```

## 执行候选 MAS

`run_mas.py` 根据 `mask` 和 `edge_weight` 重新计算拓扑顺序并执行每个 Agent，
不依赖候选数据里的 `topological_order`。GSM8K 使用 finalizer 最终数值的 exact
match 作为 `accuracy`，并计算：

```text
reward = accuracy - token_penalty * total_tokens - time_penalty * wall_time_seconds
```

成本字段定义：

- `edge_token_cost[i][j]`：沿 `i -> j` 注入目标 prompt 的上游消息 token 数。
- `edge_time_cost[i][j]`：按该消息 token 占目标输入 token 的比例归因的目标推理时间。
- `node_input_token_cost`、`node_output_token_cost`、`node_time_cost`：节点级原始成本。
- `total_input_tokens`、`total_output_tokens`、`wall_time_seconds`：整张图的原始总成本。

先运行一条 Query 的一张图确认配置：

```bash
conda activate agp
cd /home/yz/Documents/MAS_DAG/AGP-minimal

python run_mas.py \
  --model /path/to/Qwen3-8B \
  --max-queries 1 \
  --max-graphs-per-query 1 \
  --output data/gsm8k/candidate_graphs_scored.json
```

中断后续跑时添加 `--resume`。默认不保存大段 Agent 输出；审计 prompt 结果时可添加
`--store-node-outputs`。Qwen3 需要 `transformers>=4.51`。

### 使用 vLLM 并发执行候选图

vLLM 服务与训练环境建议分开。服务端示例：

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install vllm

vllm serve /path/to/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-prefix-caching
```

客户端使用 `agp` 环境。`--tokenizer` 是本地 tokenizer 路径，通常与服务端模型
目录相同；它只加载 tokenizer，不会再次加载模型权重：

```bash
conda activate agp
python run_mas.py \
  --backend vllm \
  --model /path/to/Qwen3-8B \
  --tokenizer /path/to/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 \
  --max-queries 1
```

`--concurrency` 控制同时推进的候选 DAG 数量。每张 DAG 内仍按依赖顺序执行，
不同 DAG 当前可执行的节点请求由 vLLM 自动组成动态 batch。
