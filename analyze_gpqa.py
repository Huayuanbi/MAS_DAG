from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.2f}%)" if denominator else "0/0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scored GPQA candidate graphs.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/gpqa/train_candidate_graphs_scored_with_outputs.json"),
    )
    args = parser.parse_args()
    records = json.loads(args.path.read_text(encoding="utf-8"))
    graphs = [graph for record in records for graph in record["graphs"] if graph.get("execution_status") == "completed"]
    correct = sum(graph.get("accuracy") == 1 for graph in graphs)
    completed_questions = [record for record in records if any(g.get("execution_status") == "completed" for g in record["graphs"])]
    passed_questions = sum(any(g.get("accuracy") == 1 for g in record["graphs"]) for record in completed_questions)
    mixed = sum(
        {g.get("accuracy") for g in record["graphs"] if g.get("execution_status") == "completed"} == {0.0, 1.0}
        for record in completed_questions
    )
    truncated = sum("length" in graph.get("node_finish_reason", []) for graph in graphs)
    missing = sum(graph.get("prediction") is None for graph in graphs)
    print(f"completed graphs: {len(graphs)}")
    print(f"graph accuracy: {ratio(correct, len(graphs))}")
    print(f"question pass@16: {ratio(passed_questions, len(completed_questions))}")
    print(f"mixed-reward questions: {ratio(mixed, len(completed_questions))}")
    print(f"graphs with truncation: {ratio(truncated, len(graphs))}")
    print(f"graphs without extracted answer: {ratio(missing, len(graphs))}")
    families: dict[str, list[dict]] = collections.defaultdict(list)
    for graph in graphs:
        families[graph.get("generator", "unknown")].append(graph)
    print("\nfamily\taccuracy\ttruncated\tavg_tokens\tavg_seconds")
    for family, items in families.items():
        successes = sum(item.get("accuracy") == 1 for item in items)
        lengths = sum("length" in item.get("node_finish_reason", []) for item in items)
        tokens = sum(item.get("total_input_tokens", 0) + item.get("total_output_tokens", 0) for item in items) / len(items)
        seconds = sum(item.get("wall_time_seconds", 0) for item in items) / len(items)
        print(f"{family}\t{ratio(successes, len(items))}\t{ratio(lengths, len(items))}\t{tokens:.0f}\t{seconds:.1f}")


if __name__ == "__main__":
    main()
