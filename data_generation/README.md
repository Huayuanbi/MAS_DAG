# MATH 训练数据生成管线

这套管线生成 `1000 × 12` 张聚合图。每种图运行5次，最终 `reward` 是五次
成功率，token/time成本是五次逐元素平均值。

## 数据组成

- 50% `expert_gain`：`expert_accuracy - finalizer_accuracy > 0.5`；
- 35% `high_high`：两个Anchor成功率都不低于0.8；
- 15% `low_low_nonzero`：两个Anchor成功率都不高于0.4，但不同时为0。

每题共12种拓扑：`finalizer_only`、人工设计的 `expert_anchor`，以及10种额外
候选图。额外候选由4种角色合理Anchor、`two_node`和5种角色约束随机DAG组成，
不再使用通用的chain/star/tree。

以下命令均从仓库根目录执行，并使用同一个模型、温度和seed。

四卡vLLM服务和底层图执行器也位于本目录：

```bash
./data_generation/serve_vllm.sh
python data_generation/run_mas.py --help
```

## 1. 为全量训练集准备两个Anchor，各运行5次

```bash
conda run -n agp python data_generation/math_pipeline.py prepare-anchors \
  --input-dir data/math \
  --split train \
  --node-pool data/node_pools/math_6_roles.json \
  --output data/math/anchors_5_candidates.json \
  --rollouts 5 \
  --seed 42
```

运行7500题 × 2种Anchor × 5次：

```bash
data_generation/run_stage.sh \
  data/math/anchors_5_candidates.json \
  data/math/anchors_5_scored.json \
  data/math/anchors_5.log
```

若存在 `prediction=null`，在vLLM仍运行时重复同一命令，`--resume
--retry-errors`只会重跑缺失项。

## 2. 按50%/35%/15%筛选1000题

```bash
conda run -n agp python data_generation/math_pipeline.py select \
  --anchor-scored data/math/anchors_5_scored.json \
  --output data/math/selected_1000.json \
  --total-questions 1000 \
  --delta-ratio 0.50 \
  --high-ratio 0.35 \
  --low-ratio 0.15 \
  --delta-threshold 0.50 \
  --high-threshold 0.80 \
  --low-threshold 0.40 \
  --seed 42
```

阈值不足以提供指定数量时脚本会报错，不会静默放宽标准。

## 3. 复用两个Anchor，生成额外10张候选图

```bash
conda run -n agp python data_generation/math_pipeline.py prepare-candidates \
  --selection data/math/selected_1000.json \
  --anchor-scored data/math/anchors_5_scored.json \
  --node-pool data/node_pools/math_6_roles.json \
  --output data/math/candidates_1000x12x5.json \
  --rollouts 5 \
  --extra-candidates 10 \
  --random-count 5 \
  --seed 42
```

该文件包含60,000条运行记录，其中两个Anchor的10,000条结果直接复用，只需
新运行50,000条额外候选图：

```bash
data_generation/run_stage.sh \
  data/math/candidates_1000x12x5.json \
  data/math/candidates_1000x12x5_scored.json \
  data/math/candidates_1000x12x5.log
```

## 4. 聚合并验证最终数据

```bash
conda run -n agp python data_generation/math_pipeline.py aggregate \
  --input data/math/candidates_1000x12x5_scored.json \
  --output data/math/math_train_1000x12_avg5.json \
  --topologies 12 \
  --rollouts 5
```

```bash
conda run -n agp python data_generation/math_pipeline.py validate \
  --input data/math/math_train_1000x12_avg5.json \
  --questions 1000 \
  --topologies 12 \
  --rollouts 5
```

聚合器要求所有运行均为 `completed` 且存在prediction，否则会拒绝生成最终
文件并提示先续跑缺失项。
