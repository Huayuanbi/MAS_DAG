from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import traceback

from MAS_DAG.mas_runtime import (
    TransformersChatBackend,
    VLLMChatBackend,
    run_candidate_graph,
    run_candidate_graph_async,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "gsm8k" / "candidate_graphs.json"
DEFAULT_OUTPUT = ROOT / "data" / "gsm8k" / "candidate_graphs_scored.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute candidate MAS DAGs with a local Qwen/Hugging Face model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend", choices=("transformers", "vllm"), default="transformers"
    )
    parser.add_argument("--model", required=True, help="Local path or vLLM served model name")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Local tokenizer path for vLLM edge-token accounting; defaults to --model",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Concurrent candidate DAGs for vLLM"
    )
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-penalty", type=float, default=0.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--evaluation-timeout", type=float, default=5.0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-graphs-per-query", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--store-node-outputs", action="store_true")
    parser.add_argument(
        "--evaluator",
        choices=("auto", "gsm8k", "math", "humaneval", "mmlu_pro"),
        default="auto",
        help="Answer scorer; auto reads each record's evaluator and defaults to gsm8k",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_node_pool(dataset_path: Path, reference: str) -> dict:
    pool_path = Path(reference)
    if not pool_path.is_absolute():
        pool_path = dataset_path.parent / pool_path
    pool = load_json(pool_path.resolve())
    if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
        raise ValueError(f"inval id node pool: {pool_path}")
    if not isinstance(pool.get("finalizer_id"), (str, int)):
        raise ValueError(f"node pool must define finalizer_id: {pool_path}")
    return pool


def resolve_nodes(record: dict, dataset_path: Path) -> tuple[list[dict], str]:
    if "node_pool" in record:
        pool = load_node_pool(dataset_path, record["node_pool"])
        return pool["nodes"], str(pool["finalizer_id"])
    nodes = record["nodes"]
    finalizers = [node for node in nodes if node.get("role") == "finalizer"]
    if len(finalizers) != 1:
        raise ValueError("inline nodes require exactly one finalizer role")
    return nodes, str(finalizers[0]["id"])


def should_execute(graph: dict, retry_errors: bool) -> bool:
    status = graph.get("execution_status")
    return status != "completed" and (status != "error" or retry_errors)


def resolve_evaluator(record: dict, requested: str) -> str:
    evaluator = record.get("evaluator", "gsm8k") if requested == "auto" else requested
    if evaluator not in ("gsm8k", "math", "humaneval", "mmlu_pro"):
        raise ValueError(f"unsupported record evaluator: {evaluator!r}")
    return evaluator


def print_graph_result(graph: dict, reference_answer: object) -> None:
    print(
        f"  prediction={graph['prediction']} reference={reference_answer} "
        f"accuracy={graph['accuracy']:.0f} reward={graph['reward']:.6f} "
        f"tokens={graph['total_input_tokens'] + graph['total_output_tokens']} "
        f"time={graph['wall_time_seconds']:.3f}s",
        flush=True,
    )


async def run_vllm(
    args: argparse.Namespace,
    records: list[dict],
    dataset_path: Path,
    stop: int,
) -> None:
    tokenizer_path = args.tokenizer or Path(args.model)
    backend = VLLMChatBackend(
        args.model,
        tokenizer_path=tokenizer_path,
        base_url=args.base_url,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
        timeout=args.request_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    completed_since_checkpoint = 0

    async def execute_one(
        query_index: int,
        graph_index: int,
        record: dict,
        nodes: list[dict],
        finalizer_id: str,
    ) -> None:
        graph = record["graphs"][graph_index]
        evaluator = resolve_evaluator(record, args.evaluator)
        graph_id = graph.get("id", f"q{query_index}_g{graph_index}")
        print(f"[{query_index + 1}/{stop}] graph={graph_id}", flush=True)
        try:
            # One permit represents one complete DAG. Different DAGs advance
            # independently, allowing vLLM to batch their ready node requests.
            async with semaphore:
                update = await run_candidate_graph_async(
                    task=record["task"],
                    reference_answer=str(record["reference_answer"]),
                    nodes=nodes,
                    graph=graph,
                    finalizer_id=finalizer_id,
                    backend=backend,
                    token_penalty=args.token_penalty,
                    time_penalty=args.time_penalty,
                    store_node_outputs=args.store_node_outputs,
                    evaluator=evaluator,
                    evaluation_metadata=record.get("source_metadata"),
                    evaluation_timeout=args.evaluation_timeout,
                )
            graph.update(update)
            graph.pop("execution_error", None)
            if graph.get("accuracy") == 1.0:
                graph.pop("evaluation_error", None)
            print_graph_result(graph, record["reference_answer"])
        except Exception as exc:
            graph["execution_status"] = "error"
            graph["execution_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    for query_index in range(args.query_start, stop):
        record = records[query_index]
        nodes, finalizer_id = resolve_nodes(record, dataset_path)
        graph_stop = len(record["graphs"])
        if args.max_graphs_per_query is not None:
            graph_stop = min(graph_stop, args.max_graphs_per_query)
        tasks = [
            execute_one(query_index, graph_index, record, nodes, finalizer_id)
            for graph_index in range(graph_stop)
            if should_execute(record["graphs"][graph_index], args.retry_errors)
        ]
        if tasks:
            await asyncio.gather(*tasks)

        completed_since_checkpoint += 1
        if completed_since_checkpoint >= args.checkpoint_every:
            atomic_write_json(args.output, records)
            completed_since_checkpoint = 0
            print(f"checkpoint={args.output}", flush=True)

    atomic_write_json(args.output, records)
    print(f"done output={args.output}")


def main() -> None:
    args = parse_args()
    if args.query_start < 0:
        raise ValueError("query-start must be non-negative")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.token_penalty < 0 or args.time_penalty < 0:
        raise ValueError("cost penalties must be non-negative")
    if args.evaluation_timeout <= 0:
        raise ValueError("evaluation-timeout must be positive")

    if args.resume and args.output.exists():
        records = load_json(args.output)
        print(f"resuming from {args.output}")
    else:
        records = load_json(args.input)
    # node_pool references belong to the original candidate dataset, even when
    # a scored output stored in another directory is used for resume.
    dataset_path = args.input
    if not isinstance(records, list):
        raise ValueError("input must be a JSON array")

    stop = len(records)
    if args.max_queries is not None:
        stop = min(stop, args.query_start + args.max_queries)
    if args.backend == "vllm":
        asyncio.run(run_vllm(args, records, dataset_path, stop))
        return

    backend = TransformersChatBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )
    completed_since_checkpoint = 0

    for query_index in range(args.query_start, stop):
        record = records[query_index]
        nodes, finalizer_id = resolve_nodes(record, dataset_path)
        evaluator = resolve_evaluator(record, args.evaluator)

        graphs = record["graphs"]
        graph_stop = len(graphs)
        if args.max_graphs_per_query is not None:
            graph_stop = min(graph_stop, args.max_graphs_per_query)
        for graph_index in range(graph_stop):
            graph = graphs[graph_index]
            if not should_execute(graph, args.retry_errors):
                continue
            graph_id = graph.get("id", f"q{query_index}_g{graph_index}")
            print(f"[{query_index + 1}/{stop}] graph={graph_id}", flush=True)
            try:
                graph.update(
                    run_candidate_graph(
                        task=record["task"],
                        reference_answer=str(record["reference_answer"]),
                        nodes=nodes,
                        graph=graph,
                        finalizer_id=finalizer_id,
                        backend=backend,
                        token_penalty=args.token_penalty,
                        time_penalty=args.time_penalty,
                        store_node_outputs=args.store_node_outputs,
                        evaluator=evaluator,
                        evaluation_metadata=record.get("source_metadata"),
                        evaluation_timeout=args.evaluation_timeout,
                    )
                )
                graph.pop("execution_error", None)
                if graph.get("accuracy") == 1.0:
                    graph.pop("evaluation_error", None)
                print_graph_result(graph, record["reference_answer"])
            except Exception as exc:
                graph["execution_status"] = "error"
                graph["execution_error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

        completed_since_checkpoint += 1
        if completed_since_checkpoint >= args.checkpoint_every:
            atomic_write_json(args.output, records)
            completed_since_checkpoint = 0
            print(f"checkpoint={args.output}", flush=True)

    atomic_write_json(args.output, records)
    print(f"done output={args.output}")


if __name__ == "__main__":
    main()
