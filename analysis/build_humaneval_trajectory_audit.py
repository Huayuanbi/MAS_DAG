from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/humaneval/candidate_graphs.json"
OUTPUT = ROOT / "analysis/humaneval_trajectory_audit_input.json"

# These tasks showed disagreement among the fixed candidate families in the
# original scored run, making them more informative than uniformly easy tasks.
QUERY_INDICES = (7, 17, 31, 35, 39, 43, 50, 51, 53, 58)
FIXED_FAMILIES = {
    "chain",
    "tree",
    "complete_dag",
    "finalizer_only",
    "two_node",
}


def active(mask: list[int], node: int) -> bool:
    return mask[node] == 0


def random_tags(graph: dict) -> set[str]:
    mask = graph["mask"]
    edge = graph["edge_weight"]
    tags: set[str] = set()
    if active(mask, 4) and not active(mask, 2):
        tags.add("reviewer_without_coder_node")
    if active(mask, 3) and not active(mask, 2):
        tags.add("tester_without_coder_node")
    if active(mask, 4) and edge[0][4] and not active(mask, 2):
        tags.add("spec_to_reviewer_without_coder")
    if active(mask, 4) and active(mask, 2) and edge[2][4]:
        tags.add("direct_coder_to_reviewer")
    if not active(mask, 2):
        tags.add("no_coder")
    return tags


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected_records = []
    for query_index in QUERY_INDICES:
        record = copy.deepcopy(records[query_index])
        chosen = [
            graph for graph in record["graphs"]
            if graph.get("generator") in FIXED_FAMILIES
        ]
        covered: set[str] = set()
        random_graphs = [
            graph for graph in record["graphs"]
            if graph.get("generator") == "random_dag"
        ]
        # Greedily retain random graphs that add a structural failure mode.
        for graph in random_graphs:
            tags = random_tags(graph)
            if tags - covered:
                item = copy.deepcopy(graph)
                item["audit_tags"] = sorted(tags)
                chosen.append(item)
                covered.update(tags)
        # Keep at least two random graphs for contrast even if their tags overlap.
        for graph in random_graphs:
            if len([g for g in chosen if g.get("generator") == "random_dag"]) >= 2:
                break
            if any(g.get("id") == graph.get("id") for g in chosen):
                continue
            item = copy.deepcopy(graph)
            item["audit_tags"] = sorted(random_tags(graph))
            chosen.append(item)
        record["graphs"] = chosen
        # The audit dataset lives one directory higher than the source dataset,
        # so keep the node-pool reference valid from its new location.
        record["node_pool"] = "../data/node_pools/humaneval_6_roles.json"
        record["audit_source_query_index"] = query_index
        record["audit_covered_tags"] = sorted(covered)
        selected_records.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(selected_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"questions={len(selected_records)} "
        f"graphs={sum(len(r['graphs']) for r in selected_records)} output={OUTPUT}"
    )


if __name__ == "__main__":
    main()
