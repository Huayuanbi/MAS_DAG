# MAS DAG

根据问题文本和Agent角色池学习连通DAG。当前默认模型使用文本编码器与Graph
Transformer预测节点及边，并通过DAG解码器保证推理结构无环。

## MATH数据生成

所有MATH训练数据生成入口已统一放在
[`data_generation/`](data_generation/README.md)。完整流程为：

1. 在全量训练集上分别运行 `finalizer_only` 与人工设计的 `expert_anchor`，
   每种运行5次；
2. 按50% Expert显著提升、35%两者都高、15%两者都低但不全为0，筛选题目；
3. 每题增加10种合理候选拓扑，两个Anchor直接复用，共12种图；
4. 每种图运行5次，以成功率作为最终reward，并平均token/time成本；
5. 严格验证后输出 `1000 × 12 (avg5)` JSON。

主要入口：

```bash
conda run -n agp python data_generation/math_pipeline.py --help
```

统一vLLM运行入口：

```bash
data_generation/run_stage.sh INPUT.json OUTPUT.json RUN.log
```

## vLLM服务

Qwen3-4B四卡数据并行配置保存在
[`data_generation/serve_vllm.sh`](data_generation/serve_vllm.sh)：

```bash
conda activate vllm
./data_generation/serve_vllm.sh
```

默认服务地址为 `http://127.0.0.1:8000/v1`，上下文长度为32768。

## 训练

```bash
conda activate agp
python train.py \
  --data data/math/math_train_1000x12_avg5.json \
  --objective pairwise
```

训练数据中的每道题包含12张聚合图，每张图的 `reward` 是5次运行的平均成功率。

## 主要代码

- `MAS_DAG/features.py`：角色与问题文本特征；
- `MAS_DAG/model.py`：GCN和Graph Transformer拓扑模型；
- `MAS_DAG/decoder.py`：连通DAG解码；
- `MAS_DAG/losses.py`：监督与Pairwise Reward损失；
- `MAS_DAG/topology_sampling.py`：角色感知Anchor和随机DAG；
- `MAS_DAG/mas_runtime.py`：按候选DAG执行多Agent推理；
- `data/node_pools/math_6_roles.json`：MATH Agent角色与Prompt。
