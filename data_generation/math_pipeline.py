#!/usr/bin/env python3
"""End-to-end MATH data generation for 12-topology, five-rollout training data."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAS_DAG import (  # noqa: E402
    generate_candidate_suite,
    generate_finalizer_only,
    generate_math_role_anchors,
)


ANCHOR_GENERATORS = ("finalizer_only", "expert_anchor")
EXPECTED_MATH_ROLES = {
    "problem_analyst",
    "strategy_planner",
    "primary_solver",
    "alternative_solver",
    "symbolic_proof_verifier",
    "finalizer",
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_array(path: Path) -> list[dict]:
    value = load_json(path)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return value


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def extract_last_boxed(solution: str) -> str:
    command_start = max(solution.rfind(r"\boxed"), solution.rfind(r"\fbox"))
    if command_start < 0:
        raise ValueError("reference solution does not contain boxed/fbox answer")
    command = r"\boxed" if solution.startswith(r"\boxed", command_start) else r"\fbox"
    command_end = command_start + len(command)
    remainder = solution[command_end:].lstrip()
    if not remainder.startswith("{"):
        answer = remainder.split("$", 1)[0].strip().rstrip(".,")
        if answer:
            return answer
        raise ValueError("unbraced boxed answer is empty")
    opening = command_end + len(solution[command_end:]) - len(remainder)
    depth = 1
    for index in range(opening + 1, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[opening + 1 : index].strip()
    raise ValueError("reference solution contains an unbalanced boxed answer")


def source_rows(input_dir: Path, split: str, limit: int | None = None) -> list[dict]:
    paths = sorted(input_dir.glob(f"*/{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} parquet files found under {input_dir}")
    rows: list[dict] = []
    for path in paths:
        category = path.parent.name
        for row in pq.ParquetFile(path).read().to_pylist():
            solution = str(row["solution"])
            rows.append(
                {
                    "task": str(row["problem"]),
                    "reference_answer": extract_last_boxed(solution),
                    "reference_solution": solution,
                    "source_metadata": {
                        "level": str(row["level"]),
                        "type": str(row.get("type", category)),
                        "category": category,
                        "split": split,
                    },
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def resolve_pool(path: Path) -> tuple[list[dict], int, dict[str, int]]:
    pool = load_json(path)
    if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
        raise ValueError(f"invalid node pool: {path}")
    nodes = pool["nodes"]
    finalizers = [
        index
        for index, node in enumerate(nodes)
        if str(node.get("id")) == str(pool.get("finalizer_id"))
    ]
    if len(finalizers) != 1:
        raise ValueError("node pool must identify exactly one finalizer")
    roles = {str(node["role"]): index for index, node in enumerate(nodes)}
    missing = EXPECTED_MATH_ROLES - roles.keys()
    if missing:
        raise ValueError(f"node pool is missing MATH roles: {sorted(missing)}")
    return nodes, finalizers[0], roles


def pool_reference(pool: Path, output: Path) -> str:
    return os.path.relpath(pool.resolve(), output.resolve().parent)


def fixed_order(node_count: int, finalizer: int) -> tuple[int, ...]:
    return tuple(index for index in range(node_count) if index != finalizer) + (
        finalizer,
    )


def topology_signature(graph: dict) -> tuple:
    return (
        tuple(int(value) for value in graph["mask"]),
        tuple(tuple(float(value) for value in row) for row in graph["edge_weight"]),
    )


def completed_rollouts(record: dict, generator: str, expected: int) -> list[dict]:
    graphs = [graph for graph in record.get("graphs", []) if graph.get("generator") == generator]
    if len(graphs) != expected:
        raise ValueError(
            f"source_index={record.get('source_index')} {generator}: "
            f"expected {expected} rollouts, got {len(graphs)}"
        )
    if any(
        graph.get("execution_status") != "completed" or graph.get("prediction") is None
        for graph in graphs
    ):
        raise ValueError(
            f"source_index={record.get('source_index')} {generator} is incomplete"
        )
    return graphs


def mean_accuracy(graphs: Sequence[dict]) -> float:
    return sum(float(graph["accuracy"]) for graph in graphs) / len(graphs)


def prepare_anchors(args: argparse.Namespace) -> None:
    nodes, finalizer, roles = resolve_pool(args.node_pool)
    order = fixed_order(len(nodes), finalizer)
    expert = generate_math_role_anchors(
        len(nodes), finalizer, roles, fixed_order=order
    )[0]
    records = []
    reference = pool_reference(args.node_pool, args.output)
    for source_index, source in enumerate(
        source_rows(args.input_dir, args.split, args.limit)
    ):
        graphs = []
        for topology in (generate_finalizer_only(len(nodes), finalizer), expert):
            for rollout in range(args.rollouts):
                graph = topology.to_graph_record()
                graph["id"] = (
                    f"q{source_index:05d}_{topology.generator}_r{rollout}"
                )
                graph["sampling_seed"] = args.seed + rollout
                graphs.append(graph)
        records.append(
            {
                **source,
                "source_index": source_index,
                "node_pool": reference,
                "evaluator": "math",
                "topology_policy": {
                    "stage": "full_anchor_screen",
                    "anchor_generators": list(ANCHOR_GENERATORS),
                    "rollouts_per_anchor": args.rollouts,
                    "sampling_seeds": list(
                        range(args.seed, args.seed + args.rollouts)
                    ),
                },
                "graphs": graphs,
            }
        )
    atomic_write(args.output, records)
    print(
        json.dumps(
            {
                "questions": len(records),
                "anchor_graph_rollouts": sum(len(record["graphs"]) for record in records),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def select_questions(args: argparse.Namespace) -> None:
    if not math.isclose(args.delta_ratio + args.high_ratio + args.low_ratio, 1.0):
        raise ValueError("delta/high/low ratios must sum to 1")
    records = load_array(args.anchor_scored)
    groups: dict[str, list[dict]] = {
        "expert_gain": [],
        "high_high": [],
        "low_low_nonzero": [],
    }
    for record in records:
        finalizer_graphs = completed_rollouts(
            record, "finalizer_only", args.rollouts
        )
        expert_graphs = completed_rollouts(record, "expert_anchor", args.rollouts)
        finalizer_accuracy = mean_accuracy(finalizer_graphs)
        expert_accuracy = mean_accuracy(expert_graphs)
        delta = expert_accuracy - finalizer_accuracy
        summary = {
            "source_index": int(record["source_index"]),
            "task": record["task"],
            "reference_answer": record["reference_answer"],
            "source_metadata": record.get("source_metadata", {}),
            "finalizer_accuracy": finalizer_accuracy,
            "expert_accuracy": expert_accuracy,
            "accuracy_delta": delta,
        }
        if delta > args.delta_threshold:
            groups["expert_gain"].append(summary)
        elif (
            finalizer_accuracy >= args.high_threshold
            and expert_accuracy >= args.high_threshold
        ):
            groups["high_high"].append(summary)
        elif (
            finalizer_accuracy <= args.low_threshold
            and expert_accuracy <= args.low_threshold
            and (finalizer_accuracy > 0 or expert_accuracy > 0)
        ):
            groups["low_low_nonzero"].append(summary)

    requested = {
        "expert_gain": round(args.total_questions * args.delta_ratio),
        "high_high": round(args.total_questions * args.high_ratio),
    }
    requested["low_low_nonzero"] = args.total_questions - sum(requested.values())
    rng = random.Random(args.seed)
    selected = []
    for group, count in requested.items():
        if len(groups[group]) < count:
            raise ValueError(
                f"{group}: need {count}, but only {len(groups[group])} are eligible"
            )
        sample = rng.sample(groups[group], count)
        for item in sample:
            item["selection_group"] = group
        selected.extend(sample)
    rng.shuffle(selected)
    atomic_write(args.output, selected)
    print(
        json.dumps(
            {
                "available": {group: len(items) for group, items in groups.items()},
                "selected": dict(Counter(item["selection_group"] for item in selected)),
                "criteria": {
                    "expert_gain": f"delta > {args.delta_threshold}",
                    "high_high": f"both >= {args.high_threshold}",
                    "low_low_nonzero": (
                        f"both <= {args.low_threshold}, but not both zero"
                    ),
                },
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def prepare_candidates(args: argparse.Namespace) -> None:
    selected = load_array(args.selection)
    anchor_records = load_array(args.anchor_scored)
    anchors = {int(record["source_index"]): record for record in anchor_records}
    if len(anchors) != len(anchor_records):
        raise ValueError("anchor data contains duplicate source_index values")
    nodes, finalizer, roles = resolve_pool(args.node_pool)
    order = fixed_order(len(nodes), finalizer)
    reference = pool_reference(args.node_pool, args.output)
    output_records = []
    reused = fresh = 0
    for selected_record in selected:
        source_index = int(selected_record["source_index"])
        if source_index not in anchors:
            raise ValueError(f"source_index={source_index} is missing from anchors")
        source = anchors[source_index]
        cached = {
            generator: completed_rollouts(source, generator, args.rollouts)
            for generator in ANCHOR_GENERATORS
        }
        suite = generate_candidate_suite(
            num_nodes=len(nodes),
            finalizer=finalizer,
            random_count=args.random_count,
            seed=args.seed + source_index,
            fixed_order=order,
            role_indices=roles,
        )
        if len(suite) != 2 + args.extra_candidates:
            raise ValueError(
                f"candidate suite has {len(suite)} topologies; expected "
                f"{2 + args.extra_candidates}"
            )
        graphs = []
        occurrences: dict[str, int] = {}
        for topology_index, topology in enumerate(suite):
            generator = topology.generator
            occurrence = occurrences.get(generator, 0)
            occurrences[generator] = occurrence + 1
            if generator in cached:
                existing = copy.deepcopy(cached[generator])
                if any(
                    topology_signature(graph) != topology.signature
                    for graph in existing
                ):
                    raise ValueError(
                        f"source_index={source_index}: cached {generator} topology mismatch"
                    )
                graphs.extend(existing)
                reused += len(existing)
                continue
            template = topology.to_graph_record()
            for rollout in range(args.rollouts):
                graph = copy.deepcopy(template)
                graph["id"] = (
                    f"q{source_index:05d}_g{topology_index:02d}_"
                    f"{generator}{occurrence}_r{rollout}"
                )
                graph["sampling_seed"] = args.seed + rollout
                graphs.append(graph)
                fresh += 1
        prepared = copy.deepcopy(source)
        prepared["node_pool"] = reference
        prepared["graphs"] = graphs
        prepared["topology_policy"] = {
            "stage": "selected_candidate_generation",
            "selection_group": selected_record["selection_group"],
            "finalizer_accuracy": selected_record["finalizer_accuracy"],
            "expert_accuracy": selected_record["expert_accuracy"],
            "accuracy_delta": selected_record["accuracy_delta"],
            "topologies_per_question": len(suite),
            "extra_candidates": args.extra_candidates,
            "rollouts_per_topology": args.rollouts,
            "sampling_seeds": list(range(args.seed, args.seed + args.rollouts)),
            "fixed_order": list(order),
            "reused_generators": list(ANCHOR_GENERATORS),
        }
        output_records.append(prepared)
    atomic_write(args.output, output_records)
    print(
        json.dumps(
            {
                "questions": len(output_records),
                "topologies_per_question": 2 + args.extra_candidates,
                "rollouts_per_topology": args.rollouts,
                "total_graph_rollouts": reused + fresh,
                "reused_anchor_rollouts": reused,
                "fresh_candidate_rollouts": fresh,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def mean(values: Sequence[float | int]) -> float:
    return sum(float(value) for value in values) / len(values)


def mean_vector(graphs: list[dict], key: str) -> list[float]:
    size = len(graphs[0][key])
    if any(len(graph[key]) != size for graph in graphs):
        raise ValueError(f"inconsistent {key} vector shapes")
    return [mean([graph[key][index] for graph in graphs]) for index in range(size)]


def mean_matrix(graphs: list[dict], key: str) -> list[list[float]]:
    size = len(graphs[0][key])
    if any(
        len(graph[key]) != size or any(len(row) != size for row in graph[key])
        for graph in graphs
    ):
        raise ValueError(f"inconsistent {key} matrix shapes")
    return [
        [mean([graph[key][row][column] for graph in graphs]) for column in range(size)]
        for row in range(size)
    ]


def aggregate_topology(
    graphs: list[dict], source_index: int, topology_index: int, rollouts: int
) -> dict:
    if len(graphs) != rollouts:
        raise ValueError(
            f"source_index={source_index} topology={topology_index}: "
            f"expected {rollouts} rollouts, got {len(graphs)}"
        )
    if any(graph.get("execution_status") != "completed" for graph in graphs):
        raise ValueError(
            f"source_index={source_index} topology={topology_index} is incomplete"
        )
    if any(graph.get("prediction") is None for graph in graphs):
        raise ValueError(
            f"source_index={source_index} topology={topology_index} has missing prediction; "
            "rerun with data_generation/run_mas.py --resume --retry-errors"
        )
    rewards = [float(graph["reward"]) for graph in graphs]
    accuracies = [float(graph["accuracy"]) for graph in graphs]
    template = graphs[0]
    generator = str(template["generator"])
    return {
        "id": f"q{source_index:05d}_topology{topology_index:02d}_{generator}_mean",
        "generator": generator,
        "topological_order": copy.deepcopy(template["topological_order"]),
        "mask": copy.deepcopy(template["mask"]),
        "edge_weight": copy.deepcopy(template["edge_weight"]),
        "reward": mean(rewards),
        "accuracy": mean(accuracies),
        "reward_std": statistics.pstdev(rewards),
        "correct_count": int(sum(accuracies)),
        "rollout_count": len(graphs),
        "missing_prediction_count": 0,
        "edge_token_cost": mean_matrix(graphs, "edge_token_cost"),
        "edge_time_cost": mean_matrix(graphs, "edge_time_cost"),
        "node_input_token_cost": mean_vector(graphs, "node_input_token_cost"),
        "node_output_token_cost": mean_vector(graphs, "node_output_token_cost"),
        "node_time_cost": mean_vector(graphs, "node_time_cost"),
        "total_input_tokens": mean([graph["total_input_tokens"] for graph in graphs]),
        "total_output_tokens": mean([graph["total_output_tokens"] for graph in graphs]),
        "wall_time_seconds": mean([graph["wall_time_seconds"] for graph in graphs]),
        "sampling_seeds": [graph.get("sampling_seed") for graph in graphs],
        "rollout_results": [
            {
                "id": graph.get("id"),
                "sampling_seed": graph.get("sampling_seed"),
                "prediction": graph.get("prediction"),
                "accuracy": graph["accuracy"],
                "reward": graph["reward"],
                "total_input_tokens": graph["total_input_tokens"],
                "total_output_tokens": graph["total_output_tokens"],
                "wall_time_seconds": graph["wall_time_seconds"],
                "answer_recovery_attempted": graph.get(
                    "answer_recovery_attempted", False
                ),
            }
            for graph in graphs
        ],
    }


def aggregate(args: argparse.Namespace) -> None:
    records = load_array(args.input)
    output_records = []
    for record in records:
        groups: dict[tuple, list[dict]] = {}
        for graph in record.get("graphs", []):
            groups.setdefault(topology_signature(graph), []).append(graph)
        if len(groups) != args.topologies:
            raise ValueError(
                f"source_index={record.get('source_index')} has {len(groups)} topologies"
            )
        aggregated = [
            aggregate_topology(
                graphs, int(record["source_index"]), index, args.rollouts
            )
            for index, graphs in enumerate(groups.values())
        ]
        prepared = copy.deepcopy(record)
        prepared["graphs"] = aggregated
        prepared.setdefault("topology_policy", {})["aggregation"] = {
            "method": "mean_over_rollouts",
            "rollouts_per_topology": args.rollouts,
            "output_reward_field": "reward",
            "cost_fields": "elementwise_mean",
        }
        output_records.append(prepared)
    atomic_write(args.output, output_records)
    print(
        json.dumps(
            {
                "questions": len(output_records),
                "aggregated_topologies": sum(
                    len(record["graphs"]) for record in output_records
                ),
                "represented_rollouts": sum(
                    graph["rollout_count"]
                    for record in output_records
                    for graph in record["graphs"]
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate(args: argparse.Namespace) -> None:
    records = load_array(args.input)
    source_indices: set[int] = set()
    graph_ids: set[str] = set()
    groups: Counter[str] = Counter()
    generator_rewards: dict[str, list[float]] = defaultdict(list)
    pair_signal = 0
    represented_rollouts = 0
    for record in records:
        source_index = int(record["source_index"])
        if source_index in source_indices:
            raise ValueError(f"duplicate source_index={source_index}")
        source_indices.add(source_index)
        graphs = record.get("graphs", [])
        if len(graphs) != args.topologies:
            raise ValueError(f"source_index={source_index} has {len(graphs)} topologies")
        rewards = []
        group = record.get("topology_policy", {}).get(
            "selection_group",
            record.get("topology_policy", {}).get("difficulty_group", "unknown"),
        )
        groups[str(group)] += 1
        for graph in graphs:
            graph_id = str(graph["id"])
            if graph_id in graph_ids:
                raise ValueError(f"duplicate graph id: {graph_id}")
            graph_ids.add(graph_id)
            if graph.get("rollout_count") != args.rollouts:
                raise ValueError(f"{graph_id}: invalid rollout_count")
            rollouts = graph.get("rollout_results", [])
            if len(rollouts) != args.rollouts:
                raise ValueError(f"{graph_id}: incomplete rollout_results")
            if graph.get("missing_prediction_count") != 0:
                raise ValueError(f"{graph_id}: contains missing prediction")
            expected_reward = mean([rollout["reward"] for rollout in rollouts])
            if not math.isclose(
                float(graph["reward"]), expected_reward, abs_tol=1e-12
            ):
                raise ValueError(f"{graph_id}: incorrect mean reward")
            rewards.append(float(graph["reward"]))
            generator_rewards[str(graph["generator"])].append(float(graph["reward"]))
            represented_rollouts += args.rollouts
        pair_signal += min(rewards) < max(rewards)
    if args.questions is not None and len(records) != args.questions:
        raise ValueError(f"expected {args.questions} questions, got {len(records)}")
    print(
        json.dumps(
            {
                "questions": len(records),
                "topologies": len(graph_ids),
                "represented_rollouts": represented_rollouts,
                "selection_groups": dict(groups),
                "questions_with_pair_signal": pair_signal,
                "mean_reward_by_generator": {
                    generator: mean(rewards)
                    for generator, rewards in sorted(generator_rewards.items())
                },
                "status": "valid",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-anchors")
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--node-pool", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--rollouts", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--limit", type=int)

    select = subparsers.add_parser("select")
    select.add_argument("--anchor-scored", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--total-questions", type=int, default=1000)
    select.add_argument("--rollouts", type=int, default=5)
    select.add_argument("--delta-ratio", type=float, default=0.50)
    select.add_argument("--high-ratio", type=float, default=0.35)
    select.add_argument("--low-ratio", type=float, default=0.15)
    select.add_argument("--delta-threshold", type=float, default=0.50)
    select.add_argument("--high-threshold", type=float, default=0.80)
    select.add_argument("--low-threshold", type=float, default=0.40)
    select.add_argument("--seed", type=int, default=42)

    candidates = subparsers.add_parser("prepare-candidates")
    candidates.add_argument("--selection", type=Path, required=True)
    candidates.add_argument("--anchor-scored", type=Path, required=True)
    candidates.add_argument("--node-pool", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--rollouts", type=int, default=5)
    candidates.add_argument("--extra-candidates", type=int, default=10)
    candidates.add_argument("--random-count", type=int, default=5)
    candidates.add_argument("--seed", type=int, default=42)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--topologies", type=int, default=12)
    aggregate_parser.add_argument("--rollouts", type=int, default=5)

    validation = subparsers.add_parser("validate")
    validation.add_argument("--input", type=Path, required=True)
    validation.add_argument("--questions", type=int)
    validation.add_argument("--topologies", type=int, default=12)
    validation.add_argument("--rollouts", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-anchors":
        prepare_anchors(args)
    elif args.command == "select":
        select_questions(args)
    elif args.command == "prepare-candidates":
        prepare_candidates(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
