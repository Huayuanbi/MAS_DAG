from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import TopologyExample
from .model import TopologyOutput


@dataclass
class TopologyLoss:
    total: torch.Tensor
    edge: torch.Tensor
    edge_positive: torch.Tensor
    edge_off: torch.Tensor
    node: torch.Tensor
    node_bce: torch.Tensor
    node_sparse: torch.Tensor
    node_consistency: torch.Tensor


@dataclass
class PairwiseTopologyLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    preferred_fit: TopologyLoss
    preferred_score: torch.Tensor
    rejected_score: torch.Tensor
    reward_gap: torch.Tensor


def graph_log_likelihood_score(
    output: TopologyOutput,
    example: TopologyExample,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable compatibility score between one prediction and one graph."""
    device = output.node_prob.device
    keep_target = (example.prune_mask.to(device) == 0).to(output.node_prob.dtype)
    adjacency = (example.edge_weight.to(device) != 0).to(output.edge_prob.dtype)
    node_score = -F.binary_cross_entropy(output.node_prob, keep_target)

    num_nodes = output.edge_prob.shape[0]
    off_diagonal = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
    edge_score = -F.binary_cross_entropy(
        output.edge_prob[off_diagonal], adjacency[off_diagonal]
    )
    return node_weight * node_score + edge_weight * edge_score


def pairwise_reward_loss(
    output: TopologyOutput,
    preferred: TopologyExample,
    rejected: TopologyExample,
    temperature: float = 1.0,
    preferred_fit_weight: float = 0.2,
) -> PairwiseTopologyLoss:
    """Bradley-Terry ranking loss plus a small anchor to the preferred graph."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if preferred_fit_weight < 0:
        raise ValueError("preferred_fit_weight must be non-negative")
    if preferred.reward is None or rejected.reward is None:
        raise ValueError("pairwise reward loss requires two rewards")
    if preferred.reward <= rejected.reward:
        raise ValueError("preferred.reward must be greater than rejected.reward")
    if preferred.task != rejected.task or preferred.nodes != rejected.nodes:
        raise ValueError("pairwise examples must share task and nodes")

    preferred_score = graph_log_likelihood_score(output, preferred)
    rejected_score = graph_log_likelihood_score(output, rejected)
    ranking = F.softplus((rejected_score - preferred_score) / temperature)
    preferred_fit = agp_stage2_loss(
        output,
        preferred.prune_mask.to(output.node_prob.device),
        preferred.edge_weight.to(output.edge_prob.device),
    )
    total = ranking + preferred_fit_weight * preferred_fit.total
    reward_gap = output.node_prob.new_tensor(preferred.reward - rejected.reward)
    return PairwiseTopologyLoss(
        total=total,
        ranking=ranking,
        preferred_fit=preferred_fit,
        preferred_score=preferred_score,
        rejected_score=rejected_score,
        reward_gap=reward_gap,
    )


def agp_stage2_loss(
    output: TopologyOutput,
    prune_mask: torch.Tensor,
    target_edge_weight: torch.Tensor,
    lambda_off: float = 1.0,
    lambda_sparse: float = 0.05,
    lambda_consistency: float = 0.01,
) -> TopologyLoss:
    """Faithful, vectorized form of Graph.train_loss."""
    keep_target = (prune_mask == 0).to(output.node_prob.dtype)
    pair_keep = keep_target.unsqueeze(0) * keep_target.unsqueeze(1)
    positive_mask = pair_keep.bool()
    off_mask = ~positive_mask

    target_adjacency = (target_edge_weight != 0).to(output.edge_prob.dtype)
    edge_positive = F.mse_loss(
        output.edge_prob[positive_mask], target_adjacency[positive_mask]
    )
    if off_mask.any():
        edge_off = output.edge_prob[off_mask].square().mean()
    else:
        edge_off = output.edge_prob.new_zeros(())
    edge = edge_positive + lambda_off * edge_off

    node_bce = F.binary_cross_entropy(output.node_prob, keep_target)
    node_sparse = output.node_prob.mean()
    node_consistency = (
        (1.0 - keep_target).unsqueeze(1) * output.edge_prob.abs()
    ).sum() / keep_target.numel() ** 2
    node = node_bce + lambda_sparse * node_sparse + lambda_consistency * node_consistency
    return TopologyLoss(
        total=edge + node,
        edge=edge,
        edge_positive=edge_positive,
        edge_off=edge_off,
        node=node,
        node_bce=node_bce,
        node_sparse=node_sparse,
        node_consistency=node_consistency,
    )
