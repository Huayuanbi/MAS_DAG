from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from MAS_DAG import (
    AGPJsonDataset,
    ModernBertGuidedGraphTopologyModel,
    PairwiseRewardDataset,
    agp_stage2_loss,
    fully_connected_edge_index,
    pairwise_reward_loss,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "modernbert_guided_debug.json"
DEFAULT_BERT = Path("/data1/yz/MAS_DAG/ModernBERT-base")


def parse_guidance_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one guidance layer is required")
    if tuple(sorted(set(layers))) != layers:
        raise argparse.ArgumentTypeError("guidance layers must be strictly increasing")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ModernBERT-guided graph topology model"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bert-model", type=Path, default=DEFAULT_BERT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--guidance-layers",
        type=parse_guidance_layers,
        default=(6, 12, 21),
        help="Zero-based ModernBERT encoder layers, for example 6,12,21.",
    )
    parser.add_argument("--graph-pe-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum top-level task groups; 0 retains all groups.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Gradient accumulation size. Graphs may have different node counts.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bert-lr", type=float, default=1e-5)
    parser.add_argument("--train-bert", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--objective",
        choices=("auto", "pairwise", "gt"),
        default="auto",
    )
    parser.add_argument("--ranking-temperature", type=float, default=1.0)
    parser.add_argument("--preferred-fit-weight", type=float, default=0.2)
    parser.add_argument("--mask-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--node-threshold", type=float, default=0.55)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "modernbert_guided_model.pt",
    )
    return parser.parse_args()


def node_texts(task: str, nodes) -> list[str]:
    """Build one independent, task-conditioned BERT input per graph node."""
    return [
        (
            f"Task:\n{task}\n\n"
            f"Candidate agent role: {node.role}\n"
            f"Role description: {node.role_brief}"
        )
        for node in nodes
    ]


class NodeTokenizer:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache: dict[tuple, dict[str, torch.Tensor]] = {}

    def __call__(self, task: str, nodes, device: torch.device) -> dict[str, torch.Tensor]:
        key = (
            task,
            tuple((str(node.id), node.role, node.role_brief) for node in nodes),
        )
        if key not in self._cache:
            encoded = self.tokenizer(
                node_texts(task, nodes),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            self._cache[key] = {
                "input_ids": encoded["input_ids"].cpu(),
                "attention_mask": encoded["attention_mask"].cpu(),
            }
        return {name: tensor.to(device) for name, tensor in self._cache[key].items()}


def resolve_objective(args, dataset, pair_dataset) -> str:
    objective = args.objective
    if objective == "auto":
        objective = "pairwise" if len(pair_dataset) > 0 else "gt"
    if objective == "pairwise" and len(pair_dataset) == 0:
        raise ValueError(
            "pairwise objective requires at least two differently rewarded graphs "
            "in one task group"
        )
    return objective


def build_optimizer(model, args) -> torch.optim.Optimizer:
    graph_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("bert.") and parameter.requires_grad
    ]
    groups: list[dict] = [{"params": graph_parameters, "lr": args.lr}]
    if args.train_bert:
        bert_parameters = [p for p in model.bert.parameters() if p.requires_grad]
        groups.append({"params": bert_parameters, "lr": args.bert_lr})
    return torch.optim.Adam(groups)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    max_records = args.max_records if args.max_records > 0 else None
    dataset = AGPJsonDataset(args.data, max_records=max_records)
    pair_dataset = PairwiseRewardDataset(dataset)
    objective = resolve_objective(args, dataset, pair_dataset)
    training_data = pair_dataset if objective == "pairwise" else dataset
    if len(training_data) == 0:
        raise ValueError("training dataset is empty")

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model,
        local_files_only=local_only,
    )
    bert = AutoModel.from_pretrained(
        args.bert_model,
        local_files_only=local_only,
        torch_dtype=dtype,
    )
    model = ModernBertGuidedGraphTopologyModel(
        bert,
        guidance_layers=args.guidance_layers,
        graph_pe_dim=args.graph_pe_dim,
        mask_hidden_dim=args.mask_hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        dropout=args.dropout,
        node_threshold=args.node_threshold,
        freeze_bert=not args.train_bert,
        initialize_graph_from_bert=True,
    ).to(device=device, dtype=dtype)
    tokenizer_for_nodes = NodeTokenizer(tokenizer, args.max_length)
    optimizer = build_optimizer(model, args)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"objective={objective} graphs={len(dataset)} pairs={len(pair_dataset)} "
        f"guidance_layers={args.guidance_layers} bert_frozen={not args.train_bert}"
    )
    print(f"parameters trainable={trainable:,} total={total:,}")

    printed_shapes = False
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(training_data)).tolist()
        running_loss = 0.0
        running_ranking = 0.0
        optimizer.zero_grad()

        for step, index in enumerate(order, start=1):
            item = training_data[index]
            example = item.preferred if objective == "pairwise" else item
            tokenized = tokenizer_for_nodes(example.task, example.nodes, device)
            role_edges = fully_connected_edge_index(example.num_nodes, device=device)

            if args.debug_shapes and not printed_shapes:
                num_nodes, seq_len = tokenized["input_ids"].shape
                hidden_size = model.hidden_size
                print(f"input_ids={(num_nodes, seq_len)}")
                print(
                    f"bert_hidden_per_layer={(num_nodes, seq_len, hidden_size)} "
                    f"pooled_per_layer={(num_nodes, hidden_size)}"
                )
                print(
                    f"edge_index={tuple(role_edges.shape)} "
                    f"graph_hidden={(num_nodes, hidden_size)} "
                    f"edge_prob={(num_nodes, num_nodes)}"
                )
                printed_shapes = True

            output = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                edge_index=role_edges,
            )
            if objective == "pairwise":
                loss = pairwise_reward_loss(
                    output,
                    item.preferred,
                    item.rejected,
                    temperature=args.ranking_temperature,
                    preferred_fit_weight=args.preferred_fit_weight,
                )
                running_ranking += float(loss.ranking.detach())
            else:
                loss = agp_stage2_loss(
                    output,
                    example.prune_mask.to(device),
                    example.edge_weight.to(device),
                )

            batch_start = ((step - 1) // args.batch_size) * args.batch_size
            actual_batch_size = min(args.batch_size, len(order) - batch_start)
            (loss.total / actual_batch_size).backward()
            running_loss += float(loss.total.detach())

            if step % args.batch_size == 0 or step == len(order):
                optimizer.step()
                optimizer.zero_grad()

        message = f"epoch={epoch + 1} loss={running_loss / len(training_data):.6f}"
        if objective == "pairwise":
            message += f" ranking={running_ranking / len(training_data):.6f}"
        print(message)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.state_dict()
    if not args.train_bert:
        # The frozen base model is reproducible from --bert-model. Avoid copying
        # roughly 150M unchanged BERT parameters into every training checkpoint.
        model_state = {
            name: tensor
            for name, tensor in model_state.items()
            if not name.startswith("bert.")
        }
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "resolved_objective": objective,
            "bert_model": str(args.bert_model),
            "includes_bert_weights": args.train_bert,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
