from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/merged_semantic_train1000_v1"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    question_ids: list[str] = []
    graph_ids: list[str] = []
    datasets: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    for shard in range(4):
        records = json.loads(
            (DATA / f"candidates_shard{shard}.json").read_text(encoding="utf-8")
        )
        if len(records) != 250:
            raise ValueError(f"shard {shard} has {len(records)} questions, expected 250")
        for record in records:
            metadata = record["selection_metadata"]
            question_ids.append(metadata["question_id"])
            datasets[metadata["dataset"]] += 1
            difficulties[metadata["difficulty_bucket"]] += 1
            if len(record["graphs"]) != 12:
                raise ValueError(f"{metadata['question_id']} does not have 12 graphs")
            graph_ids.extend(graph["id"] for graph in record["graphs"])
    if len(set(question_ids)) != 1000 or len(set(graph_ids)) != 12000:
        raise ValueError("question or graph IDs are not globally unique")
    if dict(datasets) != manifest["dataset_counts"]:
        raise ValueError("dataset counts differ from manifest")
    if dict(difficulties) != manifest["difficulty_counts"]:
        raise ValueError("difficulty counts differ from manifest")
    print("OK: 1000 unique questions, 12000 unique graphs, four complete shards")


if __name__ == "__main__":
    main()
