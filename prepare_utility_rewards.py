from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a token-aware utility reward.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-penalty-per-5000", type=float, default=0.05)
    args = parser.parse_args()
    if args.token_penalty_per_5000 < 0:
        raise ValueError("token penalty must be non-negative")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    for record in records:
        for graph in record["graphs"]:
            quality = float(graph["reward"])
            tokens = float(graph["total_input_tokens"] + graph["total_output_tokens"])
            graph["quality_reward"] = quality
            graph["reward"] = quality - args.token_penalty_per_5000 * tokens / 5000.0
        record["utility_reward"] = {
            "quality_field": "quality_reward",
            "token_penalty_per_5000": args.token_penalty_per_5000,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"questions={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
