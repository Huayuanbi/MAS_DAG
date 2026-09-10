from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "artifacts" / "manifest.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summarize(path: Path, top_k: int) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"empty dataset: {path}")
    by_question_family: dict[tuple[int, str], list[dict]] = {}
    for question_index, record in enumerate(records):
        for graph in record["graphs"]:
            family = str(graph.get("generator", "unknown"))
            if family == "finalizer_only":
                continue
            if graph.get("execution_status") != "completed":
                continue
            active_nodes = sum(value == 0 for value in graph["mask"])
            if active_nodes < 2:
                continue
            by_question_family.setdefault((question_index, family), []).append(
                {
                    "question_index": question_index,
                    "accuracy": float(graph["accuracy"]),
                    "tokens": int(graph["total_input_tokens"])
                    + int(graph["total_output_tokens"]),
                }
            )
    families: dict[str, list[dict]] = {}
    for (question_index, family), rows in by_question_family.items():
        families.setdefault(family, []).append(
            {
                "question_index": question_index,
                "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                "tokens": sum(row["tokens"] for row in rows) / len(rows),
            }
        )
    eligible = []
    for family, rows in families.items():
        if len(rows) != len(records):
            continue
        eligible.append(
            {
                "family": family,
                "questions": len(rows),
                "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                "mean_total_tokens": sum(row["tokens"] for row in rows) / len(rows),
            }
        )
    eligible.sort(key=lambda row: (-row["accuracy"], row["mean_total_tokens"], row["family"]))
    selected = eligible[:top_k]
    if len(selected) != top_k:
        raise ValueError(f"{path}: only {len(selected)} complete multi-agent families")
    selected_names = {row["family"] for row in selected}
    pooled = [
        row["tokens"]
        for family, rows in families.items()
        if family in selected_names
        for row in rows
    ]
    budget = sum(pooled) / len(pooled)
    return {
        "source": str(path.resolve()),
        "questions": len(records),
        "evaluator": records[0].get("evaluator", "multiple_choice"),
        "selection_rule": {
            "top_k": top_k,
            "exclude_families": ["finalizer_only"],
            "minimum_active_nodes": 2,
            "require_complete_question_coverage": True,
            "order": ["accuracy_desc", "mean_total_tokens_asc", "family_asc"],
        },
        "selected_families": selected,
        "mean_mas_total_tokens": budget,
        "single_agent_total_token_cap": math.floor(budget),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=SCORED_JSON"
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    datasets = {}
    for specification in args.dataset:
        if "=" not in specification:
            raise ValueError("--dataset must be NAME=SCORED_JSON")
        name, raw_path = specification.split("=", 1)
        if not name or name in datasets:
            raise ValueError(f"invalid or duplicate dataset name: {name!r}")
        datasets[name] = summarize(Path(raw_path), args.top_k)
    manifest = {"schema_version": 1, "datasets": datasets}
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
