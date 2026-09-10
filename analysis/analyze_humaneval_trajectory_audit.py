from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/humaneval_trajectory_audit_scored.json"
OUTPUT = ROOT / "analysis/humaneval_trajectory_audit_summary.json"
ROLES = ("spec", "algorithm", "coder", "test", "reviewer", "finalizer")


def target_name(record: dict) -> str:
    return str(record["source_metadata"]["entry_point"])


def has_target_code(text: str | None, entry_point: str) -> bool:
    if not text:
        return False
    return bool(re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", text))


def code_text(text: str | None, entry_point: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.I | re.S)
    candidate = blocks[-1] if blocks else text
    match = re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", candidate)
    return candidate[match.start():].strip() if match else ""


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    compact_left = re.sub(r"\s+", " ", left).strip()
    compact_right = re.sub(r"\s+", " ", right).strip()
    return SequenceMatcher(None, compact_left, compact_right).ratio()


def main() -> None:
    records = json.loads(INPUT.read_text(encoding="utf-8"))
    counts = Counter()
    grouped: dict[str, list[float]] = defaultdict(list)
    examples = []
    for record in records:
        entry = target_name(record)
        for graph in record["graphs"]:
            outputs = graph.get("node_outputs")
            if not outputs:
                continue
            counts["graphs"] += 1
            accuracy = float(graph["accuracy"])
            mask = graph["mask"]
            edge = graph["edge_weight"]
            active = [index for index, value in enumerate(mask) if value == 0]
            no_coder = 2 not in active
            reviewer_active = 4 in active
            tester_active = 3 in active
            reviewer_direct_coder = reviewer_active and bool(edge[2][4])
            tester_direct_coder = tester_active and bool(edge[2][3])

            grouped["all"].append(accuracy)
            if no_coder:
                grouped["no_coder"].append(accuracy)
            if reviewer_active:
                grouped["reviewer_active"].append(accuracy)
                grouped[
                    "reviewer_direct_coder" if reviewer_direct_coder
                    else "reviewer_without_direct_coder"
                ].append(accuracy)
                review = outputs[4] or ""
                if has_target_code(review, entry):
                    counts["reviewer_writes_target_code"] += 1
                if re.search(
                    r"not implemented|implementation (?:is )?missing|"
                    r"missing implementation|code (?:is|was) not provided|"
                    r"function is incomplete|no implementation",
                    review,
                    re.I,
                ):
                    counts["reviewer_claims_missing_implementation"] += 1
                if re.search(
                    r"no corrections? (?:are )?needed|no (?:issues|defects)|"
                    r"correct, efficient|correctly implements|implementation is correct",
                    review,
                    re.I,
                ):
                    counts["reviewer_approves"] += 1

            if tester_active:
                grouped[
                    "tester_direct_coder" if tester_direct_coder
                    else "tester_without_direct_coder"
                ].append(accuracy)
                test_output = outputs[3] or ""
                if has_target_code(test_output, entry):
                    counts["tester_writes_target_code"] += 1
                if re.search(r"\bassert\b|test case|expected output", test_output, re.I):
                    counts["tester_proposes_tests"] += 1

            final_code = code_text(outputs[5], entry)
            source_similarity = {
                ROLES[index]: similarity(final_code, code_text(outputs[index], entry))
                for index in active
                if index != 5
            }
            closest = max(source_similarity, key=source_similarity.get) if source_similarity else None
            if closest and source_similarity[closest] >= 0.8:
                counts[f"final_close_copy_of_{closest}"] += 1
            if no_coder:
                writer_roles = [
                    ROLES[index]
                    for index in active
                    if index != 5 and has_target_code(outputs[index], entry)
                ]
                if not writer_roles:
                    counts["no_coder_only_finalizer_writes_code"] += 1
                for role in writer_roles:
                    counts[f"no_coder_{role}_also_writes_code"] += 1

            if no_coder or (reviewer_active and not reviewer_direct_coder) or (
                tester_active and not tester_direct_coder
            ):
                examples.append(
                    {
                        "task_id": record["source_metadata"]["task_id"],
                        "graph_id": graph["id"],
                        "generator": graph["generator"],
                        "accuracy": accuracy,
                        "active_roles": [ROLES[index] for index in active],
                        "reviewer_direct_coder": reviewer_direct_coder,
                        "tester_direct_coder": tester_direct_coder,
                        "writers_before_finalizer": [
                            ROLES[index]
                            for index in active
                            if index != 5 and has_target_code(outputs[index], entry)
                        ],
                        "closest_final_source": closest,
                        "closest_similarity": (
                            source_similarity.get(closest, 0.0) if closest else 0.0
                        ),
                    }
                )

    summary = {
        "completed_graphs": counts["graphs"],
        "counts": dict(counts),
        "groups": {
            name: {
                "n": len(values),
                "accuracy": sum(values) / len(values),
            }
            for name, values in grouped.items()
            if values
        },
        "structurally_irregular_examples": examples,
    }
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "structurally_irregular_examples"}, indent=2))
    print(f"examples={len(examples)} output={OUTPUT}")


if __name__ == "__main__":
    main()
