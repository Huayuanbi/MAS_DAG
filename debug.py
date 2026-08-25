from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agp_minimal import (
    AGPJsonDataset,
    AGPTopologyModel,
    DEFAULT_TEXT_MODEL,
    DeterministicFeatureBuilder,
    GraphTransformerTopologyModel,
    SentenceTransformerFeatureBuilder,
    agp_stage2_loss,
    bidirectional_chain_edge_index,
    decode_greedy_dag,
    is_dag,
    fully_connected_edge_index,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "AGP" / "train_general_reasoning.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one minimal AGP training example")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--model", choices=("graph-transformer", "gcn"), default="graph-transformer"
    )
    parser.add_argument("--break-at", choices=("features", "model", "loss", "dag"))
    parser.add_argument("--embedding-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--offline-features", action="store_true")
    parser.add_argument(
        "--predicted-mask",
        action="store_true",
        help="Decode with the untrained node mask instead of keeping all nodes active.",
    )
    return parser.parse_args()


def pause(requested: str | None, stage: str) -> None:
    if requested == stage:
        print(f"[debug] stage={stage}")
        breakpoint()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    example = AGPJsonDataset(args.data)[args.index]
    feature_builder = (
        DeterministicFeatureBuilder()
        if args.offline_features
        else SentenceTransformerFeatureBuilder(
            model_name=args.embedding_model,
            model_device=args.embedding_device,
        )
    )
    features = feature_builder(example.task, example.nodes)
    edge_index = (
        fully_connected_edge_index(example.num_nodes)
        if args.model == "graph-transformer"
        else bidirectional_chain_edge_index(example.num_nodes)
    )
    print("task:", example.task.splitlines()[0])
    print("features:", tuple(features.shape), "edge_index:", tuple(edge_index.shape))
    print(
        "cost means:",
        {
            "token": float(example.edge_token_cost.mean()),
            "time": float(example.edge_time_cost.mean()),
        },
    )
    pause(args.break_at, "features")

    model = (
        GraphTransformerTopologyModel()
        if args.model == "graph-transformer"
        else AGPTopologyModel()
    )
    output = model(features, edge_index)
    print("node_prob:", output.node_prob.detach())
    print("edge_prob:\n", output.edge_prob.detach())
    pause(args.break_at, "model")

    loss = agp_stage2_loss(output, example.prune_mask, example.edge_weight)
    print(
        "loss:",
        {"total": float(loss.total), "edge": float(loss.edge), "node": float(loss.node)},
    )
    pause(args.break_at, "loss")
    loss.total.backward()
    encoder_gradient = (
        model.layers[0].attention.lin_query.weight.grad
        if args.model == "graph-transformer"
        else model.conv1.lin.weight.grad
    )
    print("encoder_grad:", float(encoder_gradient.norm()))
    print("edge_head_grad:", float(model.edge_head[0].weight.grad.norm()))

    decode_mask = (
        output.node_mask.detach()
        if args.predicted_mask
        else torch.ones_like(output.node_mask)
    )
    print("decoder_node_mask:", decode_mask)
    adjacency = decode_greedy_dag(output.edge_prob.detach(), decode_mask)
    print("adjacency:\n", adjacency.to(torch.int64))
    print("is_dag:", is_dag(adjacency))
    pause(args.break_at, "dag")


if __name__ == "__main__":
    main()
