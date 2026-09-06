# 两张 A6000 运行 MMLU-Pro 660 题均衡集

本分支用于补齐 MMLU-Pro 1:1:1:1 难度均衡训练集的完整 12 个候选拓扑 Mean-5 标签。

## 数据

660 道题全部来自既有 MMLU-Pro train split，不含 validation/test。四类各 165 道：Easy、Medium、Collaboration-required、Hard-unsolved。仓库只包含约 5.5 MB 的必要题目、Single/Star Mean-5 标签和 ID manifest；运行轨迹、checkpoint、日志与模型均不提交。

已有 `finalizer_only` 和 `star` 标签会直接复用。新服务器实际执行剩余 10 个图，每图 5 次，共 33,000 次图执行。

## 环境

推荐 Python 3.10/3.11、CUDA 12.x，并安装本项目及 vLLM：

```bash
pip install -e .
pip install vllm
```

## 启动两张卡

每张 A6000 独立承载一个 Qwen3-8B 服务，均为 TP=1：

```bash
bash start_qwen3_8b_two_a6000.sh /你的/Qwen3-8B路径
```

确认两个端口均可用：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/models
```

## 开始五轮评分

```bash
MODEL_PATH=/你的/Qwen3-8B路径 \
PYTHON_BIN=python \
CONCURRENCY=16 \
bash run_mmlu_balanced_mean5_two_a6000.sh
```

题目固定分成两个 330 题 shard，GPU0/8000 和 GPU1/8001 同时工作。脚本按题保存 checkpoint，重复执行会断点续跑。五轮全部完成后自动生成：

```text
data/mmlu_pro_balanced_660/MMLU-Pro-balanced-660-full-candidates-mean5.json
```

若聚合检测到缺失或失败 trial，会明确报错而不会把缺失静默当作 0。原始运行文件和节点轨迹位于 `data/mmlu_pro_balanced_660/runs/`，默认由 `.gitignore` 排除。
