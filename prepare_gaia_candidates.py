from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from MAS_DAG.semantic_topologies import make_topology


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/data/gzy/EntCollabBench/GAIA")
DEFAULT_OUTPUT = ROOT / "data" / "gaia" / "validation_pilot_candidates.json"
DEFAULT_POOL = ROOT / "data" / "node_pools" / "gaia_7_roles.json"

FINALIZER = 6
WEB = 0
DOCUMENT = 1
SPREADSHEET = 2
MEDIA = 3
COMPUTE = 4
VERIFY = 5

SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac",
    ".mov", ".mp4", ".mkv", ".webm", ".avi",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".txt", ".docx", ".pptx", ".xml", ".json", ".jsonld",
    ".zip", ".pdb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official GAIA parquet rows to MAS_DAG candidate graphs."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--levels", default="1,2")
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated GAIA task IDs.",
    )
    parser.add_argument(
        "--kinds",
        default="",
        help="Optional comma-separated attachment kinds: web,document,spreadsheet,media.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Select a deterministic 10-row modality-stratified pilot.",
    )
    parser.add_argument(
        "--kind-quotas",
        default="",
        help="Deterministic balanced selection, for example web=8,document=8,spreadsheet=6,media=6.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("GAIA conversion requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def attachment_kind(record: dict[str, Any]) -> str:
    extension = Path(str(record.get("file_name") or "")).suffix.lower()
    if not extension:
        return "web"
    if extension in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if extension in MEDIA_EXTENSIONS:
        return "media"
    return "document"


def select_pilot(
    records: Iterable[dict[str, Any]], quotas: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    quotas = quotas or {"web": 3, "document": 3, "spreadsheet": 2, "media": 2}
    selected: list[dict[str, Any]] = []
    counts = {key: 0 for key in quotas}
    for record in records:
        kind = attachment_kind(record)
        if counts[kind] < quotas[kind]:
            selected.append(record)
            counts[kind] += 1
        if counts == quotas:
            break
    missing = {key: quotas[key] - counts[key] for key in quotas if counts[key] < quotas[key]}
    if missing:
        raise ValueError(f"not enough rows for stratified pilot: {missing}")
    return selected


def primary_node(record: dict[str, Any]) -> int:
    return {
        "web": WEB,
        "document": DOCUMENT,
        "spreadsheet": SPREADSHEET,
        "media": MEDIA,
    }[attachment_kind(record)]


def candidate_graphs(record: dict[str, Any]) -> list[dict[str, Any]]:
    primary = primary_node(record)
    definitions = [
        ("finalizer_only", (FINALIZER,), ()),
        ("primary_finalize", (primary, FINALIZER), ((primary, FINALIZER),)),
        (
            "primary_verify_finalize",
            (primary, VERIFY, FINALIZER),
            ((primary, VERIFY), (VERIFY, FINALIZER)),
        ),
        (
            "primary_verify_direct_finalize",
            (primary, VERIFY, FINALIZER),
            ((primary, VERIFY), (primary, FINALIZER), (VERIFY, FINALIZER)),
        ),
        (
            "primary_compute_verify_finalize",
            tuple(dict.fromkeys((primary, COMPUTE, VERIFY, FINALIZER))),
            tuple(dict.fromkeys(
                ((primary, COMPUTE), (COMPUTE, VERIFY), (primary, VERIFY), (VERIFY, FINALIZER))
            )),
        ),
        (
            "web_primary_verify_finalize",
            tuple(dict.fromkeys((WEB, primary, VERIFY, FINALIZER))),
            tuple(dict.fromkeys(
                ((WEB, VERIFY), (primary, VERIFY), (VERIFY, FINALIZER))
            )),
        ),
        (
            "web_primary_verify_direct_finalize",
            tuple(dict.fromkeys((WEB, primary, VERIFY, FINALIZER))),
            tuple(dict.fromkeys(
                ((WEB, VERIFY), (primary, VERIFY), (WEB, FINALIZER),
                 (primary, FINALIZER), (VERIFY, FINALIZER))
            )),
        ),
        (
            "web_primary_compute_verify_finalize",
            tuple(dict.fromkeys((WEB, primary, COMPUTE, VERIFY, FINALIZER))),
            tuple(dict.fromkeys(
                ((WEB, VERIFY), (primary, VERIFY), (COMPUTE, VERIFY), (VERIFY, FINALIZER))
            )),
        ),
    ]
    graphs: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    for name, active, edges in definitions:
        # Removing duplicate active nodes can turn an intended edge into a self-loop.
        clean_edges = tuple((source, target) for source, target in edges if source != target)
        topology = make_topology(
            name,
            num_nodes=7,
            finalizer=FINALIZER,
            active=active,
            edges=clean_edges,
        )
        if topology.signature in signatures:
            continue
        signatures.add(topology.signature)
        graph = topology.to_graph_record()
        graph["id"] = f"{record['task_id']}_{name}"
        graphs.append(graph)
    return graphs


def convert_record(
    record: dict[str, Any], *, dataset_root: Path, split: str, node_pool_ref: str
) -> dict[str, Any]:
    file_name = str(record.get("file_name") or "")
    relative_attachment = f"2023/{split}/{file_name}" if file_name else ""
    question = str(record["Question"]).strip()
    if file_name:
        question += f"\n\nAn attachment is available for this task: {file_name}"
    answer = str(record.get("Final answer") or "")
    return {
        "task": question,
        "reference_answer": answer,
        "source_metadata": {
            "task_id": str(record["task_id"]),
            "level": str(record["Level"]),
            "split": split,
            "attachment_kind": attachment_kind(record),
            "attachment_path": relative_attachment,
            "attachment_source": (
                str((dataset_root / relative_attachment).resolve())
                if relative_attachment else ""
            ),
            "annotator_metadata": record.get("Annotator Metadata") or {},
        },
        "node_pool": node_pool_ref,
        "evaluator": "gaia",
        "graphs": candidate_graphs(record),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    levels = {value.strip() for value in args.levels.split(",") if value.strip()}
    task_ids = {value.strip() for value in args.task_ids.split(",") if value.strip()}
    kinds = {value.strip() for value in args.kinds.split(",") if value.strip()}
    unknown_kinds = kinds - {"web", "document", "spreadsheet", "media"}
    if unknown_kinds:
        raise ValueError(f"unknown attachment kinds: {sorted(unknown_kinds)}")
    quotas: dict[str, int] = {}
    for item in (value.strip() for value in args.kind_quotas.split(",") if value.strip()):
        if "=" not in item:
            raise ValueError(f"invalid kind quota: {item!r}")
        kind, raw_count = (part.strip() for part in item.split("=", 1))
        if kind not in {"web", "document", "spreadsheet", "media"}:
            raise ValueError(f"unknown quota kind: {kind!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("kind quota counts must be positive")
        quotas[kind] = count
    if args.pilot and quotas:
        raise ValueError("--pilot and --kind-quotas are mutually exclusive")
    metadata_path = args.dataset_root / "2023" / args.split / "metadata.parquet"
    records = [
        record for record in load_records(metadata_path)
        if str(record["Level"]) in levels
        and (not kinds or attachment_kind(record) in kinds)
        and (not task_ids or str(record["task_id"]) in task_ids)
    ]
    if args.pilot:
        default_quotas = {"web": 3, "document": 3, "spreadsheet": 2, "media": 2}
        records = select_pilot(
            records,
            {kind: default_quotas[kind] for kind in kinds} if kinds else None,
        )
    elif quotas:
        records = select_pilot(records, quotas)
    if args.limit is not None:
        records = records[: args.limit]
    if args.split == "test" and any(str(row.get("Final answer") or "") for row in records):
        raise ValueError("GAIA test rows unexpectedly expose final answers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    node_pool_ref = os.path.relpath(args.node_pool.resolve(), args.output.parent.resolve())
    converted = [
        convert_record(
            record,
            dataset_root=args.dataset_root,
            split=args.split,
            node_pool_ref=node_pool_ref,
        )
        for record in records
    ]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    kinds: dict[str, int] = {}
    for row in converted:
        kind = row["source_metadata"]["attachment_kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
    print(
        f"split={args.split} rows={len(converted)} kinds={kinds} "
        f"graphs={sum(len(row['graphs']) for row in converted)} output={args.output}"
    )


if __name__ == "__main__":
    main()
