from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select fixed SingleAgent and BestMAS graphs for screening."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-generator", required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    selected_records = []
    for question_index, record in enumerate(records):
        graphs = record.get("graphs", [])
        single = [g for g in graphs if g.get("generator") == "finalizer_only"]
        best = [g for g in graphs if g.get("generator") == args.best_generator]
        if len(single) != 1 or len(best) != 1:
            raise ValueError(
                f"question {question_index}: expected one finalizer_only and one "
                f"{args.best_generator}, got {len(single)} and {len(best)}"
            )
        selected = copy.deepcopy(record)
        selected["graphs"] = [copy.deepcopy(single[0]), copy.deepcopy(best[0])]
        selected["screening_graphs"] = {
            "single_agent": "finalizer_only",
            "best_mas": args.best_generator,
        }
        selected_records.append(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"questions={len(selected_records)} graphs={2 * len(selected_records)} "
        f"best_mas={args.best_generator} output={args.output}"
    )


if __name__ == "__main__":
    main()
