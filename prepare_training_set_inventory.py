from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "data" / "training_set_inventory"

BUCKET_LABELS = {
    "mas_help": "MAS-help",
    "both_high": "双高",
    "both_low_nonzero": "双低（非全零）",
    "other": "其他",
}

DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "collaboration_required": "Collaboration-required",
    "hard_unsolved": "Hard-unsolved",
    "other": "Other",
}


def classify(single: float, best: float) -> str:
    delta = best - single
    if delta > 0.5 + 1e-8:
        return "mas_help"
    if single >= 0.6 - 1e-8 and best >= 0.6 - 1e-8:
        return "both_high"
    if (
        single <= 0.4 + 1e-8
        and best <= 0.4 + 1e-8
        and not (abs(single) <= 1e-8 and abs(best) <= 1e-8)
    ):
        return "both_low_nonzero"
    return "other"


def classify_difficulty(single: float, best: float) -> str:
    delta = best - single
    if delta >= 0.4 - 1e-8:
        return "collaboration_required"
    if single >= 1.0 - 1e-8:
        return "easy"
    if single <= 0.2 + 1e-8 and best <= 0.2 + 1e-8:
        return "hard_unsolved"
    if 0.4 - 1e-8 <= single <= 0.8 + 1e-8 and abs(delta) <= 0.2 + 1e-8:
        return "medium"
    return "other"


def graph_reward(record: dict[str, Any], generator: str) -> float:
    matches = [g for g in record["graphs"] if g.get("generator") == generator]
    if len(matches) != 1 or matches[0].get("reward") is None:
        raise ValueError(f"expected one completed {generator!r} graph")
    return float(matches[0]["reward"])


def load_aime_mean5() -> list[dict[str, Any]]:
    paths = [ROOT / f"data/aime/single_best_screening_run{run}.json" for run in range(1, 6)]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(len(run) != len(runs[0]) for run in runs):
        raise ValueError("AIME run lengths differ")
    result = []
    for index, base in enumerate(runs[0]):
        record = {key: value for key, value in base.items() if key != "graphs"}
        graphs = []
        for graph_index, base_graph in enumerate(base["graphs"]):
            peers = [run[index]["graphs"][graph_index] for run in runs]
            if any(
                (g.get("id"), g.get("generator"), g.get("mask"), g.get("edge_weight"))
                != (
                    base_graph.get("id"),
                    base_graph.get("generator"),
                    base_graph.get("mask"),
                    base_graph.get("edge_weight"),
                )
                for g in peers
            ):
                raise ValueError(f"AIME graph mismatch at record {index}, graph {graph_index}")
            if any(g.get("execution_status") != "completed" for g in peers):
                raise ValueError(f"AIME incomplete trial at record {index}, graph {graph_index}")
            graph = dict(base_graph)
            trials = [float(g["accuracy"]) for g in peers]
            graph["reward"] = sum(trials) / 5
            graph["reward_trials"] = trials
            graph["num_trials"] = 5
            graphs.append(graph)
        record["graphs"] = graphs
        result.append(record)
    return result


def identifier(dataset: str, metadata: dict[str, Any], index: int) -> str:
    if dataset == "gpqa":
        return str(metadata["question_id"])
    if dataset == "aime":
        return f"aime_2024_{metadata.get('id', index)}"
    return f"mmlu_pro_{metadata['question_id']}"


def materialize(
    dataset: str,
    records: list[dict[str, Any]],
    source_path: str,
    best_generator: str,
    candidate_scope: str,
) -> dict[str, Any]:
    rows = []
    for index, record in enumerate(records):
        metadata = record.get("source_metadata", {})
        single = graph_reward(record, "finalizer_only")
        best = graph_reward(record, best_generator)
        bucket = classify(single, best)
        difficulty = classify_difficulty(single, best)
        task = " ".join(str(record["task"]).split())
        rows.append(
            {
                "dataset": dataset,
                "question_id": identifier(dataset, metadata, index),
                "record_index": index,
                "source_file": source_path,
                "source_locator": f"{source_path}#record_index={index}",
                "source_category": metadata.get("category", metadata.get("split", "")),
                "single_generator": "finalizer_only",
                "best_mas_generator": best_generator,
                "p_single": f"{single:.1f}",
                "p_best_mas": f"{best:.1f}",
                "delta": f"{best - single:.1f}",
                "bucket": bucket,
                "bucket_cn": BUCKET_LABELS[bucket],
                "difficulty_bucket": difficulty,
                "difficulty_bucket_cn": DIFFICULTY_LABELS[difficulty],
                "mean5_complete": True,
                "candidate_scope": candidate_scope,
                "question_preview": task[:200],
            }
        )

    output_dir = OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for bucket in BUCKET_LABELS:
        ids = [row["question_id"] for row in rows if row["bucket"] == bucket]
        (output_dir / f"{bucket}.txt").write_text(
            "".join(f"{question_id}\n" for question_id in ids), encoding="utf-8"
        )
    for difficulty in DIFFICULTY_LABELS:
        ids = [
            row["question_id"]
            for row in rows
            if row["difficulty_bucket"] == difficulty
        ]
        (output_dir / f"difficulty_{difficulty}.txt").write_text(
            "".join(f"{question_id}\n" for question_id in ids), encoding="utf-8"
        )

    counts = Counter(row["bucket"] for row in rows)
    difficulty_counts = Counter(row["difficulty_bucket"] for row in rows)
    return {
        "dataset": dataset,
        "source_file": source_path,
        "questions": len(rows),
        "best_mas_generator": best_generator,
        "candidate_scope": candidate_scope,
        "bucket_counts": {bucket: counts[bucket] for bucket in BUCKET_LABELS},
        "difficulty_counts": {
            bucket: difficulty_counts[bucket] for bucket in DIFFICULTY_LABELS
        },
        "inventory_csv": str((output_dir / "questions.csv").relative_to(ROOT)),
        "id_lists": {
            bucket: str((output_dir / f"{bucket}.txt").relative_to(ROOT))
            for bucket in BUCKET_LABELS
        },
        "difficulty_id_lists": {
            bucket: str(
                (output_dir / f"difficulty_{bucket}.txt").relative_to(ROOT)
            )
            for bucket in DIFFICULTY_LABELS
        },
    }


def write_balanced_mmlu_train(records: list[dict[str, Any]], seed: int = 42) -> None:
    split = json.loads(
        (ROOT / "data/mmlu_pro/router_splits_seed42/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train_ids = set(split["question_ids"]["train"])
    target_buckets = ("easy", "medium", "collaboration_required", "hard_unsolved")
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
        bucket: [] for bucket in target_buckets
    }
    for index, record in enumerate(records):
        metadata = record["source_metadata"]
        if metadata["question_id"] not in train_ids:
            continue
        single = graph_reward(record, "finalizer_only")
        best = graph_reward(record, "star")
        bucket = classify_difficulty(single, best)
        if bucket in groups:
            question_id = identifier("mmlu_pro", metadata, index)
            groups[bucket].append((question_id, record))

    per_bucket = min(len(group) for group in groups.values())
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for bucket, group in groups.items():
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{seed}:{bucket}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend((bucket, question_id, record) for question_id, record in ranked[:per_bucket])
    selected.sort(key=lambda item: (target_buckets.index(item[0]), item[1]))

    output_dir = OUTPUT_ROOT / "mmlu_pro_balanced_difficulty_train_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records = [record for _, _, record in selected]
    (output_dir / "selected.json").write_text(
        json.dumps(write_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for bucket in target_buckets:
        ids = [question_id for label, question_id, _ in selected if label == bucket]
        (output_dir / f"{bucket}.txt").write_text(
            "".join(f"{question_id}\n" for question_id in ids), encoding="utf-8"
        )
    manifest = {
        "name": "mmlu_pro_balanced_difficulty_train_seed42",
        "source": "data/mmlu_pro/train70_single_best_mean5_clean.json",
        "eligible_split": "data/mmlu_pro/router_splits_seed42/train.json",
        "selection_seed": seed,
        "selection_method": "stable SHA-256 rank within difficulty bucket",
        "classes": list(target_buckets),
        "per_class": per_bucket,
        "questions": len(selected),
        "counts": {bucket: per_bucket for bucket in target_buckets},
        "contains_only_existing_train_questions": True,
        "current_graph_labels": ["finalizer_only", "star"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"balanced_mmlu_train={len(selected)} per_class={per_bucket}")


def main() -> None:
    gpqa_path = ROOT / "data/gpqa/GPQA-train-mean5.json"
    mmlu_path = ROOT / "data/mmlu_pro/train70_single_best_mean5_clean.json"
    mmlu_records = json.loads(mmlu_path.read_text(encoding="utf-8"))
    summaries = [
        materialize(
            "gpqa",
            json.loads(gpqa_path.read_text(encoding="utf-8")),
            "data/gpqa/GPQA-train-mean5.json",
            "parallel_solvers_verify",
            "16 candidate graphs all have Mean-5 labels",
        ),
        materialize(
            "aime",
            load_aime_mean5(),
            "data/aime/single_best_screening_run{1..5}.json",
            "complete_dag",
            "only SingleAgent and fixed BestMAS have Mean-5 labels",
        ),
        materialize(
            "mmlu_pro",
            mmlu_records,
            "data/mmlu_pro/train70_single_best_mean5_clean.json",
            "star",
            "only SingleAgent and fixed BestMAS have Mean-5 labels",
        ),
    ]
    summary = {
        "classification_version": "v1",
        "classification_priority": [
            "mas_help: delta > 0.5",
            "both_high: p_single >= 0.6 and p_best_mas >= 0.6",
            "both_low_nonzero: p_single <= 0.4 and p_best_mas <= 0.4 and not both zero",
            "other: all remaining complete questions",
        ],
        "difficulty_classification_version": "v2",
        "difficulty_classification_priority": [
            "collaboration_required: delta >= 0.4",
            "easy: p_single == 1.0",
            "hard_unsolved: p_single <= 0.2 and p_best_mas <= 0.2",
            "medium: 0.4 <= p_single <= 0.8 and abs(delta) <= 0.2",
            "other: all remaining complete questions",
        ],
        "datasets": summaries,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_balanced_mmlu_train(mmlu_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
