from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from generate_aime_workflow_candidates import build_workflow_topologies
from generate_gpqa_candidates import workflow_topologies as gpqa_workflow_topologies
from prepare_training_set_inventory import classify_difficulty, load_aime_mean5
from MAS_DAG.semantic_topologies import (
    mmlu_manual_topologies,
    semantic_random_topologies,
    validate_random_against_pool,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data" / "merged_semantic_train1000_v1"
SOURCES = {
    "mmlu_pro": ROOT / "data/mmlu_pro/train70_single_best_mean5_clean.json",
    "gpqa": ROOT / "data/gpqa/GPQA-train-mean5.json",
}
POOLS = {
    "mmlu_pro": ROOT / "data/node_pools/mmlu_pro_6_roles.json",
    "gpqa": ROOT / "data/node_pools/gpqa_diamond_11_roles.json",
    "aime": ROOT / "data/node_pools/aime_workflow_13_roles.json",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_select(records: list[dict], count: int, namespace: str) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"{namespace}: requested {count}, only {len(records)} available")
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"42:{namespace}:{record['_question_id']}".encode("utf-8")
        ).hexdigest(),
    )[:count]


def annotate(dataset: str, records: list[dict], best_generator: str) -> list[dict]:
    result = []
    for index, source in enumerate(records):
        record = copy.deepcopy(source)
        graphs = {graph["generator"]: graph for graph in record["graphs"]}
        single = float(graphs["finalizer_only"]["reward"])
        best = float(graphs[best_generator]["reward"])
        metadata = record.get("source_metadata", {})
        raw_id = metadata.get("question_id", metadata.get("id", index))
        record["_dataset"] = dataset
        record["_question_id"] = f"{dataset}:{raw_id}"
        record["_difficulty"] = classify_difficulty(single, best)
        record["_single_reward"] = single
        record["_best_mas_reward"] = best
        result.append(record)
    return result


def select_training_records() -> list[dict]:
    datasets = {
        "mmlu_pro": annotate("mmlu_pro", load(SOURCES["mmlu_pro"]), "star"),
        "gpqa": annotate("gpqa", load(SOURCES["gpqa"]), "parallel_solvers_verify"),
        "aime": annotate("aime", load_aime_mean5(), "complete_dag"),
    }
    selected = []
    # Keep every scarce GPQA/AIME collaboration-required example, then fill
    # the 500-example positive class from MMLU-Pro.
    for dataset in ("gpqa", "aime"):
        selected.extend(r for r in datasets[dataset] if r["_difficulty"] == "collaboration_required")
    selected.extend(stable_select(
        [r for r in datasets["mmlu_pro"] if r["_difficulty"] == "collaboration_required"],
        500 - len(selected), "collaboration_required:mmlu_pro",
    ))

    # Easy: retain all 30 GPQA examples and fill to 300 from MMLU-Pro.
    easy_gpqa = [r for r in datasets["gpqa"] if r["_difficulty"] == "easy"]
    selected.extend(easy_gpqa)
    selected.extend(stable_select(
        [r for r in datasets["mmlu_pro"] if r["_difficulty"] == "easy"],
        300 - len(easy_gpqa), "easy:mmlu_pro",
    ))

    # Remaining 200: retain all non-collab/non-easy AIME and GPQA examples,
    # then fill the balance from MMLU-Pro's medium/hard/other pool.
    remainder = [
        r for dataset in ("gpqa", "aime") for r in datasets[dataset]
        if r["_difficulty"] not in ("collaboration_required", "easy")
    ]
    selected.extend(remainder)
    selected.extend(stable_select(
        [
            r for r in datasets["mmlu_pro"]
            if r["_difficulty"] not in ("collaboration_required", "easy")
        ],
        200 - len(remainder), "remaining:mmlu_pro",
    ))
    if len(selected) != 1000 or len({r["_question_id"] for r in selected}) != 1000:
        raise AssertionError("selection must contain 1000 unique questions")
    return selected


def manual_factory(dataset: str) -> Callable[[], list]:
    return {
        "mmlu_pro": mmlu_manual_topologies,
        "gpqa": gpqa_workflow_topologies,
        "aime": build_workflow_topologies,
    }[dataset]


def candidate_record(source: dict, output_path: Path) -> dict:
    dataset = source["_dataset"]
    pool_path = POOLS[dataset]
    pool = load(pool_path)
    manual = manual_factory(dataset)()
    if len(manual) != 7 or len({item.signature for item in manual}) != 7:
        raise ValueError(f"{dataset}: expected seven unique manual topologies")
    seed = int.from_bytes(hashlib.sha256(source["_question_id"].encode()).digest()[:8], "big")
    random_graphs = semantic_random_topologies(
        pool, seed=seed, count=5,
        excluded_signatures={item.signature for item in manual},
    )
    for topology in random_graphs:
        validate_random_against_pool(topology, pool)
    topologies = manual + random_graphs
    record = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in ("graphs", "node_pool") and not key.startswith("_")
    }
    record["node_pool"] = os.path.relpath(pool_path, output_path.parent)
    record["sampling_seed"] = seed
    record["selection_metadata"] = {
        "dataset": dataset,
        "question_id": source["_question_id"],
        "difficulty_bucket": source["_difficulty"],
        "screening_single_reward": source["_single_reward"],
        "screening_best_mas_reward": source["_best_mas_reward"],
    }
    record["topology_policy"] = {
        "version": "semantic_7_manual_5_random_v1",
        "manual_graphs": 7,
        "semantic_random_graphs": 5,
        "random_edges_restricted_by_pool_allowed_edge": True,
    }
    record["graphs"] = []
    safe_id = source["_question_id"].replace(":", "_")
    for index, topology in enumerate(topologies):
        graph = topology.to_graph_record()
        graph["id"] = f"{safe_id}_g{index:02d}"
        graph["graph_group"] = "manual" if index < 7 else "semantic_random"
        record["graphs"].append(graph)
    return record


def main() -> None:
    selected = select_training_records()
    selected.sort(key=lambda row: row["_question_id"])
    shard_size = 250
    output_records = []
    for index, source in enumerate(selected):
        shard = index // shard_size
        shard_path = OUTPUT / f"candidates_shard{shard}.json"
        output_records.append(candidate_record(source, shard_path))
    for shard in range(4):
        write(OUTPUT / f"candidates_shard{shard}.json", output_records[shard * shard_size : (shard + 1) * shard_size])

    by_dataset = Counter(row["_dataset"] for row in selected)
    by_difficulty = Counter(row["_difficulty"] for row in selected)
    cross = Counter((row["_dataset"], row["_difficulty"]) for row in selected)
    manifest = {
        "name": "merged_semantic_train1000_v1",
        "selection_seed": 42,
        "questions": 1000,
        "graphs_per_question": 12,
        "total_graphs": 12000,
        "shards": 4,
        "questions_per_shard": 250,
        "dataset_counts": dict(sorted(by_dataset.items())),
        "difficulty_counts": dict(sorted(by_difficulty.items())),
        "dataset_difficulty_counts": {
            dataset: {
                difficulty: cross[(dataset, difficulty)]
                for difficulty in ("collaboration_required", "easy", "medium", "hard_unsolved", "other")
            }
            for dataset in ("mmlu_pro", "gpqa", "aime")
        },
        "topology_policy": "7 manual plus 5 role-constrained random DAGs",
        "source_files": {key: str(value.relative_to(ROOT)) for key, value in SOURCES.items()} | {
            "aime": "data/aime/single_best_screening_run{1..5}.json"
        },
    }
    write(OUTPUT / "manifest.json", manifest)
    (OUTPUT / "selected_question_ids.txt").write_text(
        "".join(f"{row['_question_id']}\n" for row in selected), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
