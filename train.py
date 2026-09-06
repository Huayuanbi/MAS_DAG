from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random

import numpy as np
import torch

from MAS_DAG import (
    AGPJsonDataset,
    AGPTopologyModel,
    DEFAULT_TEXT_MODEL,
    DeterministicFeatureBuilder,
    GraphTransformerTopologyModel,
    PairwiseRewardDataset,
    SentenceTransformerFeatureBuilder,
    agp_stage2_loss,
    bidirectional_chain_edge_index,
    fully_connected_edge_index,
    pairwise_reward_loss,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "AGP" / "train_general_reasoning.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal AGP Stage-II topology model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-records",
        type=int,
        default=32,
        help="Maximum top-level task groups; all graphs in each group are retained.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model", choices=("graph-transformer", "gcn"), default="graph-transformer"
    )
    parser.add_argument(
        "--objective",
        choices=("auto", "pairwise", "gt"),
        default="auto",
        help="Use reward ranking, direct graph labels, or auto-detect reward pairs.",
    )
    parser.add_argument("--ranking-temperature", type=float, default=1.0)
    parser.add_argument("--preferred-fit-weight", type=float, default=0.2)
    parser.add_argument("--min-reward-gap", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-question", type=int, default=48)
    parser.add_argument("--reward-gap-scale", type=float, default=0.2)
    parser.add_argument("--reward-gap-power", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument(
        "--offline-features",
        action="store_true",
        help="Use deterministic hash features instead of SentenceTransformer.",
    )
    parser.add_argument("--gcn-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    max_records = args.max_records if args.max_records > 0 else None
    dataset = AGPJsonDataset(args.data, max_records=max_records)
    max_pairs = (
        args.max_pairs_per_question if args.max_pairs_per_question > 0 else None
    )
    pair_dataset = PairwiseRewardDataset(
        dataset,
        min_reward_gap=args.min_reward_gap,
        max_pairs_per_group=max_pairs,
        seed=args.seed,
    )
    objective = args.objective
    if objective == "auto":
        objective = "pairwise" if len(pair_dataset) > 0 else "gt"
    if objective == "pairwise" and len(pair_dataset) == 0:
        raise ValueError(
            "pairwise objective requires at least two differently rewarded graphs "
            "in one task group"
        )
    training_data = pair_dataset if objective == "pairwise" else dataset
    print(
        f"objective={objective} graphs={len(dataset)} "
        f"pairs={len(pair_dataset)}"
    )
    model = (
        GraphTransformerTopologyModel()
        if args.model == "graph-transformer"
        else AGPTopologyModel()
    ).to(device)
    feature_builder = (
        DeterministicFeatureBuilder(seed=args.seed)
        if args.offline_features
        else SentenceTransformerFeatureBuilder(
            model_name=args.embedding_model,
            model_device=args.embedding_device,
        )
    )

    if args.gcn_only and args.model != "gcn":
        raise ValueError("--gcn-only can only be used with --model gcn")
    parameters = (
        list(model.conv1.parameters())
        + list(model.conv2.parameters())
        + list(model.mask_head.parameters())
        if args.gcn_only
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    history: list[dict[str, float | int]] = []
    best_state = None
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    validation_pairs = None
    if args.validation_data is not None:
        validation_dataset = AGPJsonDataset(args.validation_data)
        validation_pairs = PairwiseRewardDataset(
            validation_dataset,
            min_reward_gap=args.min_reward_gap,
            max_pairs_per_group=max_pairs,
            seed=args.seed + 1,
        )
        print(
            f"validation_graphs={len(validation_dataset)} "
            f"validation_pairs={len(validation_pairs)}"
        )

    for epoch in range(args.epochs):
        order = torch.randperm(len(training_data)).tolist()
        running_loss = 0.0
        running_ranking = 0.0
        optimizer.zero_grad()
        for step, index in enumerate(order, start=1):
            item = training_data[index]
            example = item.preferred if objective == "pairwise" else item
            features = feature_builder(example.task, example.nodes, device=device)
            role_edges = (
                fully_connected_edge_index(example.num_nodes, device=device)
                if args.model == "graph-transformer"
                else bidirectional_chain_edge_index(example.num_nodes, device=device)
            )
            output = model(features, role_edges)
            if objective == "pairwise":
                loss = pairwise_reward_loss(
                    output,
                    item.preferred,
                    item.rejected,
                    temperature=args.ranking_temperature,
                    preferred_fit_weight=args.preferred_fit_weight,
                    reward_gap_scale=args.reward_gap_scale,
                    reward_gap_power=args.reward_gap_power,
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
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": running_loss / len(training_data),
        }
        if objective == "pairwise":
            message += f" ranking={running_ranking / len(training_data):.6f}"
            epoch_metrics["ranking"] = running_ranking / len(training_data)
        if validation_pairs is not None:
            model.eval()
            validation_loss = 0.0
            validation_correct = 0
            with torch.no_grad():
                for item in validation_pairs:
                    features = feature_builder(item.preferred.task, item.preferred.nodes, device=device)
                    role_edges = (
                        fully_connected_edge_index(item.preferred.num_nodes, device=device)
                        if args.model == "graph-transformer"
                        else bidirectional_chain_edge_index(item.preferred.num_nodes, device=device)
                    )
                    output = model(features, role_edges)
                    loss = pairwise_reward_loss(
                        output,
                        item.preferred,
                        item.rejected,
                        temperature=args.ranking_temperature,
                        preferred_fit_weight=args.preferred_fit_weight,
                        reward_gap_scale=args.reward_gap_scale,
                        reward_gap_power=args.reward_gap_power,
                    )
                    validation_loss += float(loss.total)
                    validation_correct += int(loss.preferred_score > loss.rejected_score)
            model.train()
            mean_validation_loss = validation_loss / len(validation_pairs)
            validation_accuracy = validation_correct / len(validation_pairs)
            epoch_metrics["validation_loss"] = mean_validation_loss
            epoch_metrics["validation_pair_accuracy"] = validation_accuracy
            message += f" val_loss={mean_validation_loss:.6f} "
            message += f"val_pair_accuracy={validation_accuracy:.4f}"
            improved = validation_accuracy > best_validation_accuracy or (
                validation_accuracy == best_validation_accuracy
                and mean_validation_loss < best_validation_loss
            )
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_validation_accuracy = validation_accuracy
                best_validation_loss = mean_validation_loss
                epochs_without_improvement = 0
                message += " best=true"
            else:
                epochs_without_improvement += 1
        history.append(epoch_metrics)
        print(message)
        if (
            validation_pairs is not None
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"early_stopping epoch={epoch + 1} best_epoch={best_epoch}")
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "resolved_objective": objective,
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_pair_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
