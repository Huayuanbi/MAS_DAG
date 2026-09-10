from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts" / "manifest.json")
    parser.add_argument("--results-dir", type=Path, default=HERE / "artifacts" / "results")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "report.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {"schema_version": 1, "datasets": {}}
    for name, spec in manifest["datasets"].items():
        dataset = {"mas_budget": spec["mean_mas_total_tokens"], "selected_families": spec["selected_families"], "single_agent": {}}
        source_records = json.loads(Path(spec["source"]).read_text())
        selected_names = {row["family"] for row in spec["selected_families"]}
        mas_accuracy: dict[str, dict[int, float]] = {family: {} for family in selected_names}
        for index, record in enumerate(source_records):
            for family in selected_names:
                values = [
                    float(graph["accuracy"])
                    for graph in record["graphs"]
                    if graph.get("generator") == family
                    and graph.get("execution_status") == "completed"
                ]
                if values:
                    mas_accuracy[family][index] = mean(values)
        for mode in ("no_thinking", "thinking", "thinking_concise"):
            path = args.results_dir / name / f"{mode}.json"
            if not path.exists():
                continue
            rows = [row for row in json.loads(path.read_text()) if row.get("status") == "completed"]
            comparisons = {}
            for family, accuracy_by_question in mas_accuracy.items():
                differences = [
                    row["accuracy"] - accuracy_by_question[row["question_index"]]
                    for row in rows
                    if row["question_index"] in accuracy_by_question
                ]
                comparisons[family] = {
                    "paired_questions": len(differences),
                    "single_minus_mas_accuracy": mean(differences),
                }
            dataset["single_agent"][mode] = {
                "completed": len(rows),
                "accuracy": mean([row["accuracy"] for row in rows]),
                "mean_total_tokens": mean([row["total_tokens"] for row in rows]),
                "budget_utilization": mean([row["total_tokens"] / row["total_token_cap"] for row in rows]),
                "truncation_rate": mean([float(row.get("finish_reason") == "length") for row in rows]),
                "mean_latency_seconds": mean([row["latency_seconds"] for row in rows]),
                "paired_comparisons": comparisons,
            }
        report["datasets"][name] = dataset
    atomic = args.output.with_suffix(args.output.suffix + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
