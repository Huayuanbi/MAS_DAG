from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scored GAIA MAS_DAG runs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    graphs = []
    task_solved = 0
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "solved": 0, "graphs": 0, "correct": 0})
    by_topology: dict[str, dict[str, float]] = defaultdict(
        lambda: {"graphs": 0, "correct": 0, "tokens": 0, "wall_seconds": 0.0, "tool_calls": 0}
    )
    tools: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0, "error": 0})
    errors: dict[str, int] = defaultdict(int)

    for record in records:
        kind = str(record.get("source_metadata", {}).get("attachment_kind", "unknown"))
        completed = [g for g in record.get("graphs", []) if g.get("execution_status") == "completed"]
        solved = any(g.get("accuracy") == 1.0 for g in completed)
        task_solved += int(solved)
        by_kind[kind]["tasks"] += 1
        by_kind[kind]["solved"] += int(solved)
        for graph in completed:
            graphs.append(graph)
            correct = int(graph.get("accuracy") == 1.0)
            by_kind[kind]["graphs"] += 1
            by_kind[kind]["correct"] += correct
            name = str(graph.get("generator", "unknown"))
            item = by_topology[name]
            item["graphs"] += 1
            item["correct"] += correct
            item["tokens"] += int(graph.get("total_input_tokens", 0)) + int(graph.get("total_output_tokens", 0))
            item["wall_seconds"] += float(graph.get("wall_time_seconds", 0.0))
            item["tool_calls"] += int(graph.get("total_tool_calls", 0))
            for trace in graph.get("node_tool_traces", []):
                for event in trace:
                    name = str(event.get("name") or "parse_error")
                    status = str(event.get("status") or "unknown")
                    tools[name]["calls"] += 1
                    if status == "ok":
                        tools[name]["ok"] += 1
                    else:
                        tools[name]["error"] += 1
                        errors[str(event.get("error") or status)] += 1

    for item in by_kind.values():
        item["task_solve_rate"] = ratio(item["solved"], item["tasks"])
        item["graph_accuracy"] = ratio(item["correct"], item["graphs"])
    for item in by_topology.values():
        count = int(item["graphs"])
        item["accuracy"] = ratio(int(item["correct"]), count)
        item["mean_tokens"] = item["tokens"] / count if count else 0.0
        item["mean_wall_seconds"] = item["wall_seconds"] / count if count else 0.0

    return {
        "tasks": len(records),
        "tasks_solved": task_solved,
        "task_solve_rate": ratio(task_solved, len(records)),
        "completed_graphs": len(graphs),
        "correct_graphs": sum(g.get("accuracy") == 1.0 for g in graphs),
        "graph_accuracy": ratio(sum(g.get("accuracy") == 1.0 for g in graphs), len(graphs)),
        "total_tokens": sum(int(g.get("total_input_tokens", 0)) + int(g.get("total_output_tokens", 0)) for g in graphs),
        "summed_wall_seconds": sum(float(g.get("wall_time_seconds", 0.0)) for g in graphs),
        "by_kind": dict(by_kind),
        "by_topology": dict(by_topology),
        "tools": dict(tools),
        "tool_errors": dict(sorted(errors.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> None:
    args = parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    report = summarize(records)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
