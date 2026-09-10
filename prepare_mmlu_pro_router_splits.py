from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def stable_rank(question_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode()).hexdigest()


def allocate(total: int, sizes: dict[str, int]) -> dict[str, int]:
    population = sum(sizes.values())
    exact = {key: total * size / population for key, size in sizes.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(
        sizes,
        key=lambda key: (exact[key] - result[key], sizes[key], str(key)),
        reverse=True,
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question-isolated MMLU-Pro router split.")
    parser.add_argument(
        "--clean",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_mean5_clean.json"),
    )
    parser.add_argument(
        "--with-missing",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_mean5_with_missing.json"),
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/mmlu_pro/test.parquet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mmlu_pro/router_splits_seed42")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=6640)
    parser.add_argument("--validation-size", type=int, default=830)
    parser.add_argument("--test-size", type=int, default=830)
    args = parser.parse_args()

    records = json.loads(args.clean.read_text(encoding="utf-8"))
    requested = args.train_size + args.validation_size + args.test_size
    if requested != len(records):
        raise ValueError(f"split sizes total {requested}, clean data has {len(records)}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        single, best = record["graphs"]
        delta = float(best["reward"]) - float(single["reward"])
        key = (str(record["source_metadata"]["category"]), f"{delta:.1f}")
        groups[key].append(record)
    for group in groups.values():
        group.sort(
            key=lambda record: stable_rank(
                int(record["source_metadata"]["question_id"]), args.seed
            )
        )

    group_sizes = {key: len(value) for key, value in groups.items()}
    validation_alloc = allocate(args.validation_size, group_sizes)
    remaining_sizes = {
        key: group_sizes[key] - validation_alloc[key] for key in groups
    }
    test_alloc = allocate(args.test_size, remaining_sizes)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key, group in groups.items():
        val_count = validation_alloc[key]
        test_count = test_alloc[key]
        validation.extend(group[:val_count])
        test.extend(group[val_count : val_count + test_count])
        train.extend(group[val_count + test_count :])

    for split in (train, validation, test):
        split.sort(key=lambda record: int(record["source_metadata"]["question_id"]))
        for record in split:
            pool = Path(record["node_pool"])
            if not pool.is_absolute():
                pool = (args.clean.parent / pool).resolve()
            record["node_pool"] = os.path.relpath(pool, args.output_dir.resolve())
    if (len(train), len(validation), len(test)) != (
        args.train_size,
        args.validation_size,
        args.test_size,
    ):
        raise AssertionError("allocation produced incorrect split sizes")

    split_ids = {
        name: {int(r["source_metadata"]["question_id"]) for r in split}
        for name, split in (("train", train), ("validation", validation), ("test", test))
    }
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or split_ids["validation"] & split_ids["test"]:
        raise AssertionError("question leakage across train/validation/test")

    with_missing = json.loads(args.with_missing.read_text(encoding="utf-8"))
    selected_ids = {
        int(record["source_metadata"]["question_id"]) for record in with_missing
    }
    missing_ids = selected_ids - set.union(*split_ids.values())
    source_ids = set(pd.read_parquet(args.source, columns=["question_id"])["question_id"].astype(int))
    untouched_ids = source_ids - selected_ids
    partitions = {**split_ids, "missing_8k": missing_ids, "untouched": untouched_ids}
    names = list(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if partitions[left] & partitions[right]:
                raise AssertionError(f"leakage between {left} and {right}")
    if set.union(*partitions.values()) != source_ids:
        raise AssertionError("partitions do not cover the source test split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "validation.json", validation)
    write_json(args.output_dir / "test.json", test)
    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "stratification": ["category", "best_mas_minus_single_mean5"],
        "selection_rule": "stable SHA-256 rank by question_id within stratum",
        "sizes": {name: len(ids) for name, ids in partitions.items()},
        "question_ids": {name: sorted(ids) for name, ids in partitions.items()},
        "category_counts": {
            name: dict(
                sorted(
                    Counter(r["source_metadata"]["category"] for r in split).items()
                )
            )
            for name, split in (("train", train), ("validation", validation), ("test", test))
        },
    }
    write_json(args.output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest["sizes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
