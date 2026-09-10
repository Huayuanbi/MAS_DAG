# 路由研究 V2

该目录用于将下一阶段的研究探索与 MAS_DAG 主工程隔离。源数据集只作为只读输入，
所有生成结果统一存放在 `artifacts/` 下，并由 Git 忽略。

## 研究问题

在有限的 MAS 候选图库中进行选择时，相比全局候选先验和图结构族先验，题目文本
能否对未见过的新题目提供额外且可泛化的预测信息？

当前包含 100 道题的 GPQA 五次均值数据集仅用于探索，不能据此得出最终的路由或
拓扑结论。`data/gpqa/split_manifest.json` 中独立的 20 道验证题只作为前瞻性的阶段
推进门槛；其余 78 道测试题继续保持冻结。

## 目录结构

- `research_plan.md`：分阶段研究方案、假设、对照实验、推进门槛和数据集规划。
- `literature.md`：基于原始论文的文献矩阵及其对本项目的启示。
- `debate.md`：三轮对抗式研究争论及最终均衡结论。
- `stage0_baselines.py`：面向现有 GPQA 数据的低成本、按题分组交叉验证基线。
- `stage0_findings.md`：Stage-0 实验结果、解释边界和后续影响。
- `artifacts/`：自动生成的 JSON 报告，不纳入版本控制。

## Stage 0：现有数据的探索性检验

在仓库根目录执行：

```bash
python experiments/router_research_v2/stage0_baselines.py \
  --input data/gpqa/GPQA-train-mean5.json \
  --output experiments/router_research_v2/artifacts/stage0_baselines.json
```

该脚本在五折、按题目分组的外层交叉验证中比较以下方法：

- 候选位置先验；
- 图结构族先验；
- 直接建模题目—候选图交互的概率 scorer；
- 打乱训练题目文本的负对照；
- 预先指定的固定 `parallel_solvers_verify` 参考策略。

每个二项分布单元使用 `Beta(1,1)` 伪计数进行收缩。超参数选择完全嵌套在每个外层
训练折内部，避免利用外层测试折调参。

这是一项筛选性实验，其继续推进门槛被有意设置得较严格：文本条件 scorer 必须在
五个外层折中的至少四折，同时超过该折训练得到的候选先验和预先指定的固定
`parallel_solvers_verify`，比较指标为被选中图的观测 reward。即使通过该门槛，也只
表示可以开展下一阶段的前瞻数据收集，不能视为确认性证据。

当前 Stage-0 结果没有通过该门槛，具体数值和解释见 `stage0_findings.md`。
