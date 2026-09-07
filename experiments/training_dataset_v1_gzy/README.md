# training_dataset_v1_gzy

本目录是本轮 1000 题、每题 12 张图、独立运行 5 次的唯一正式入口。
候选数据已经提交到 `data/merged_semantic_train1000_v1/`，无需重新下载原始
benchmark。运行结果和日志默认被 Git 忽略。

## 数据配置

- 1000 道唯一题目：MMLU-Pro 880、GPQA-Diamond 100、AIME 20。
- 筛题分层：500 collaboration-required、300 easy、200 其余类别。
- 每题 7 张人工图和 5 张语义约束随机 DAG，共 12000 图/轮。
- 五轮 seed 分别为 2001 至 2005。
- 当前四个 `candidates_shard*.json` 是计算分片，不是 train/validation/test。

随机图均满足 allowed-edge、无环、连通到 finalizer。人工图包含历史 BestMAS
控制结构，其中 MMLU `star` 和 AIME `complete_dag` 是显式 legacy control，
不能解释为完全遵守新 allowed-edge 的随机样本。最终训练前仍需进行题目级
train/validation/test 划分及 Mean-5 聚合。

## 首次检查

要求 Python 3.10+、可用的 vLLM 环境、`tmux`、`curl`，以及本地 Qwen3-8B
模型目录。项目 Python 依赖可安装为：

```bash
pip install -e .
python experiments/training_dataset_v1_gzy/validate_inputs.py
```

vLLM 请使用服务器上已有的 CUDA 匹配环境安装，不由 `pyproject.toml` 强行锁定。

## A6000 GPU 3 一键运行 Run 4、5

```bash
git clone --branch training_dataset_v1_gzy --single-branch \
  https://github.com/Huayuanbi/MAS_DAG.git
cd MAS_DAG

tmux new-session -d -s merged_semantic_r45 \
  "cd '$PWD' && MODEL_PATH=/absolute/path/to/Qwen3-8B GPU_ID=3 \
  bash start_remote_a6000_runs45.sh"
```

若命令不在 PATH，额外传入：

```bash
VLLM_BIN=/absolute/path/to/vllm PYTHON_BIN=/absolute/path/to/python
```

脚本会启动 GPU 3 上的单卡 Qwen3-8B，Run 4 完成后自动运行 Run 5，并使用
checkpoint 断点续跑。

## 其他入口

```bash
# 已有 vLLM 服务时运行单一轮；RUN_INDEX 可取 1..5
RUN_INDEX=1 BASE_URL=http://127.0.0.1:8001/v1 \
MODEL_PATH=/absolute/path/to/Qwen3-8B \
bash experiments/training_dataset_v1_gzy/run_round.sh

# 本机 GPU 0、1 各启动一个服务，顺序运行 Run 1、2、3
MODEL_PATH=/absolute/path/to/Qwen3-8B \
bash start_local_dual_qwen_runs123.sh
```

输出位于 `data/merged_semantic_train1000_v1/runs/run{1..5}_shard{0..3}.json`，
日志位于 `logs/merged_semantic_train1000_v1/`。

根目录的 `prepare_merged_semantic_train1000.py`、
`run_merged_semantic_train1000.sh`、`start_local_dual_qwen_runs123.sh` 和
`start_remote_a6000_runs45.sh` 是兼容入口，实际实现均在本目录。
