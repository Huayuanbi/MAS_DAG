from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats
import torch

from MAS_DAG import (
    AGPJsonDataset,
    AGPTopologyModel,
    DeterministicFeatureBuilder,
    GraphTransformerTopologyModel,
    SentenceTransformerFeatureBuilder,
    bidirectional_chain_edge_index,
    fully_connected_edge_index,
    graph_log_likelihood_score,
)


def ndcg(rewards: list[float], scores: list[float]) -> float:
    predicted = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ideal = sorted(range(len(rewards)), key=lambda index: rewards[index], reverse=True)

    def dcg(order: list[int]) -> float:
        return sum(
            rewards[index] / np.log2(rank + 2)
            for rank, index in enumerate(order)
        )

    ideal_dcg = dcg(ideal)
    return dcg(predicted) / ideal_dcg if ideal_dcg else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a topology selector by question.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--fixed-best-generator",
        default="parallel_solvers_verify",
        help="Generator name used for the fixed-MAS baseline.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    training_args = checkpoint["args"]
    model_name = training_args.get("model", "graph-transformer")
    model = (
        GraphTransformerTopologyModel()
        if model_name == "graph-transformer"
        else AGPTopologyModel()
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if training_args.get("offline_features"):
        feature_builder = DeterministicFeatureBuilder(seed=training_args.get("seed", 7))
    else:
        feature_builder = SentenceTransformerFeatureBuilder(
            model_name=training_args["embedding_model"],
            model_device=training_args.get("embedding_device"),
        )

    dataset = AGPJsonDataset(args.data)
    groups: dict[int, list] = {}
    for example in dataset:
        assert example.pair_group is not None
        groups.setdefault(example.pair_group, []).append(example)

    raw_records = json.loads(args.data.read_text(encoding="utf-8"))
    selected_rewards = []
    selected_tokens = []
    oracle_rewards = []
    single_rewards = []
    fixed_best_rewards = []
    ndcgs = []
    spearmans = []
    pair_correct = 0
    pair_total = 0
    gap_pair_correct = {0.4: 0, 0.6: 0}
    gap_pair_total = {0.4: 0, 0.6: 0}
    selected_families: dict[str, int] = {}

    with torch.no_grad():
        for group_index, examples in groups.items():
            first = examples[0]
            features = feature_builder(first.task, first.nodes, device=args.device)
            role_edges = (
                fully_connected_edge_index(first.num_nodes, device=args.device)
                if model_name == "graph-transformer"
                else bidirectional_chain_edge_index(first.num_nodes, device=args.device)
            )
            output = model(features, role_edges)
            scores = [float(graph_log_likelihood_score(output, example)) for example in examples]
            rewards = [float(example.reward) for example in examples]
            selected_index = int(np.argmax(scores))
            raw_graphs = raw_records[group_index]["graphs"]
            selected_rewards.append(rewards[selected_index])
            oracle_rewards.append(max(rewards))
            selected_tokens.append(
                float(
                    raw_graphs[selected_index].get("total_input_tokens", 0)
                    + raw_graphs[selected_index].get("total_output_tokens", 0)
                )
            )
            family = raw_graphs[selected_index].get("generator", "unknown")
            selected_families[family] = selected_families.get(family, 0) + 1
            single_rewards.append(
                rewards[next(i for i, graph in enumerate(raw_graphs) if graph.get("generator") == "finalizer_only")]
            )
            fixed_best_rewards.append(
                rewards[
                    next(
                        i
                        for i, graph in enumerate(raw_graphs)
                        if graph.get("generator") == args.fixed_best_generator
                    )
                ]
            )
            ndcgs.append(ndcg(rewards, scores))
            if len(set(rewards)) > 1 and len(set(scores)) > 1:
                spearmans.append(float(stats.spearmanr(rewards, scores).statistic))
            for left, right in itertools.combinations(range(len(examples)), 2):
                gap = abs(rewards[left] - rewards[right])
                if gap <= 1e-8:
                    continue
                correct = (scores[left] - scores[right]) * (rewards[left] - rewards[right]) > 0
                pair_correct += int(correct)
                pair_total += 1
                for threshold in gap_pair_total:
                    if gap + 1e-8 >= threshold:
                        gap_pair_correct[threshold] += int(correct)
                        gap_pair_total[threshold] += 1

    result = {
        "questions": len(groups),
        "top1_reward": float(np.mean(selected_rewards)),
        "oracle_reward": float(np.mean(oracle_rewards)),
        "regret": float(np.mean(np.asarray(oracle_rewards) - selected_rewards)),
        "single_agent_reward": float(np.mean(single_rewards)),
        "fixed_best_reward": float(np.mean(fixed_best_rewards)),
        "selected_mean_tokens": float(np.mean(selected_tokens)),
        "ndcg": float(np.mean(ndcgs)),
        "spearman": float(np.mean(spearmans)),
        "pair_accuracy": pair_correct / pair_total,
        "pair_accuracy_gap_ge_0.4": gap_pair_correct[0.4] / gap_pair_total[0.4],
        "pair_accuracy_gap_ge_0.6": gap_pair_correct[0.6] / gap_pair_total[0.6],
        "selected_families": selected_families,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_pair_accuracy": checkpoint.get(
            "best_validation_pair_accuracy"
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
