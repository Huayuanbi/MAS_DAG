from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from MAS_DAG.mas_runtime import VLLMChatBackend, build_messages, evaluate_answer


HERE = Path(__file__).resolve().parent
ROLE = {
    "id": "single_solver",
    "role": "finalizer",
    "role_brief": (
        "You are a strong independent problem solver. Analyze the problem carefully, "
        "verify the alternatives, and give the requested final answer."
    ),
    "user_prompt": "Problem:\n{question}",
}
CONCISE_THINKING_SUFFIX = (
    "\nUse a concise reasoning process: focus only on the facts that distinguish "
    "the answer choices, avoid restating the problem or repeating checks, and stop "
    "reasoning as soon as one option is sufficiently supported. Reserve enough "
    "space to always provide the required FINAL_ANSWER line."
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prompt_tokens(tokenizer, messages: list[dict], enable_thinking: bool) -> int:
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    try:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    return len(encoded)


async def run_group(args, name: str, spec: dict, mode: str) -> None:
    source = Path(spec["source"])
    records = json.loads(source.read_text(encoding="utf-8"))
    output = args.output_dir / name / f"{mode}.json"
    results = json.loads(output.read_text()) if args.resume and output.exists() else []
    completed = {int(row["question_index"]) for row in results if row.get("status") == "completed"}
    thinking = mode in ("thinking", "thinking_concise")
    backend = VLLMChatBackend(
        args.model, tokenizer_path=args.tokenizer, base_url=args.base_url,
        api_key=args.api_key, max_new_tokens=spec["single_agent_total_token_cap"],
        temperature=args.temperature, enable_thinking=thinking, seed=args.seed,
        timeout=args.request_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(index: int, record: dict) -> dict:
        evaluator = record.get("evaluator", spec["evaluator"])
        role = dict(ROLE)
        if mode == "thinking_concise":
            role["role_brief"] = str(role["role_brief"]) + CONCISE_THINKING_SUFFIX
        messages = build_messages(record["task"], role, [], is_finalizer=True, evaluator=evaluator)
        input_count = prompt_tokens(backend.tokenizer, messages, thinking)
        total_cap = int(spec["single_agent_total_token_cap"])
        completion_cap = total_cap - input_count
        if completion_cap <= 0:
            return {"question_index": index, "status": "error", "error": "prompt exceeds total token cap", "input_tokens": input_count}
        async with semaphore:
            generation = await backend.generate(messages, max_new_tokens=completion_cap)
        prediction, correct = evaluate_answer(
            generation.text, str(record["reference_answer"]), evaluator,
            evaluation_metadata=record.get("source_metadata"),
        )
        return {
            "question_index": index, "status": "completed", "mode": mode,
            "prediction": prediction, "accuracy": float(correct),
            "input_tokens": generation.input_tokens, "output_tokens": generation.output_tokens,
            "total_tokens": generation.input_tokens + generation.output_tokens,
            "total_token_cap": total_cap, "completion_token_cap": completion_cap,
            "finish_reason": generation.finish_reason, "latency_seconds": generation.latency_seconds,
        }

    pending = [(i, row) for i, row in enumerate(records) if i not in completed]
    for start in range(0, len(pending), args.checkpoint_every):
        batch = pending[start : start + args.checkpoint_every]
        additions = await asyncio.gather(*(execute(i, row) for i, row in batch), return_exceptions=True)
        for (index, _), item in zip(batch, additions):
            if isinstance(item, Exception):
                item = {"question_index": index, "status": "error", "error": f"{type(item).__name__}: {item}"}
            results.append(item)
        results.sort(key=lambda row: row["question_index"])
        atomic_json(output, results)
        print(f"checkpoint dataset={name} mode={mode} rows={len(results)} output={output}", flush=True)


async def async_main(args) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    modes = [args.mode] if args.mode != "both" else ["no_thinking", "thinking"]
    names = args.dataset or list(manifest["datasets"])
    for name in names:
        if name not in manifest["datasets"]:
            raise ValueError(f"unknown dataset: {name}")
        for mode in modes:
            await run_group(args, name, manifest["datasets"][name], mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "artifacts" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "results")
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--mode",
        choices=("both", "thinking", "thinking_concise", "no_thinking"),
        default="both",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.checkpoint_every <= 0:
        raise ValueError("concurrency and checkpoint-every must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
