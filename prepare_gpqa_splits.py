from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stable 100/20/78 GPQA Diamond split.")
    parser.add_argument("--parquet", type=Path, default=Path("data/gpqa/gpqa_diamond.parquet"))
    parser.add_argument("--existing-train", type=Path, default=Path("data/gpqa/sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gpqa"))
    parser.add_argument("--remaining-seed", type=int, default=43)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required") from exc
    rows = pq.read_table(args.parquet).to_pylist()
    if len(rows) != 198:
        raise ValueError(f"expected 198 GPQA Diamond rows, found {len(rows)}")
    existing = [json.loads(line) for line in args.existing_train.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing_indices = [int(record["source_index"]) for record in existing]
    if len(existing_indices) != 50 or len(set(existing_indices)) != 50:
        raise ValueError("existing training sample must contain 50 unique source indices")

    remaining = [index for index in range(len(rows)) if index not in set(existing_indices)]
    random.Random(args.remaining_seed).shuffle(remaining)
    additional_indices = remaining[:50]
    validation_indices = remaining[50:70]
    test_indices = remaining[70:]
    split_indices = {
        "train": existing_indices + additional_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    if set().union(*(set(value) for value in split_indices.values())) != set(range(198)):
        raise AssertionError("split does not cover the full dataset")
    if sum(map(len, split_indices.values())) != 198:
        raise AssertionError("split contains duplicate rows")

    def materialize(name: str, indices: list[int]) -> list[dict]:
        return [
            {
                "question_id": f"gpqa_diamond_{source_index:03d}",
                "split": name,
                "split_seed": args.remaining_seed,
                "split_position": position,
                "source_index": source_index,
                "question": str(rows[source_index]["question"]),
                "answer": str(rows[source_index]["answer"]).strip().upper(),
            }
            for position, source_index in enumerate(indices)
        ]

    train = materialize("train", split_indices["train"])
    additional = train[50:]
    validation = materialize("validation", validation_indices)
    test = materialize("test", test_indices)
    write_jsonl(args.output_dir / "train_sample.jsonl", train)
    write_jsonl(args.output_dir / "train_additional_50.jsonl", additional)
    write_jsonl(args.output_dir / "validation_sample.jsonl", validation)
    write_jsonl(args.output_dir / "test_sample.jsonl", test)

    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    manifest = {
        "dataset": "GPQA Diamond",
        "source_file": str(args.parquet),
        "source_sha256": digest,
        "source_rows": len(rows),
        "policy": {
            "existing_train_rows_preserved": 50,
            "remaining_rows_shuffle_seed": args.remaining_seed,
            "counts": {key: len(value) for key, value in split_indices.items()},
            "test_is_frozen": True,
        },
        "source_indices": split_indices,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"train={len(train)} validation={len(validation)} test={len(test)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
