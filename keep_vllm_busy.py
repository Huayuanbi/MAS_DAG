from __future__ import annotations

import argparse
import asyncio
import signal
import time

import httpx
from openai import AsyncOpenAI


PROMPTS = (
    "Solve carefully and explain each step: If a store discounts an $80 item by 15% and then adds 8% tax, what is the final price?",
    "Give a rigorous comparison of breadth-first search and depth-first search, including complexity and three practical examples.",
    "Derive the quadratic formula from ax^2 + bx + c = 0 and verify it by substitution.",
    "Explain how attention in a Transformer is computed, including tensor shapes for a four-head example.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously send generation requests to a local vLLM server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run; 0 means run until Ctrl+C.",
    )
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.concurrency <= 0 or args.max_tokens <= 0:
        raise ValueError("concurrency and max-tokens must be positive")
    if args.duration < 0 or args.interval < 0:
        raise ValueError("duration and interval must be non-negative")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop.set)

    http_client = httpx.AsyncClient(timeout=args.timeout, trust_env=False)
    client = AsyncOpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key="EMPTY",
        max_retries=2,
        http_client=http_client,
    )
    started = time.perf_counter()
    lock = asyncio.Lock()
    totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def worker(worker_id: int) -> None:
        request_index = 0
        while not stop.is_set():
            if args.duration and time.perf_counter() - started >= args.duration:
                stop.set()
                break
            prompt = PROMPTS[(worker_id + request_index) % len(PROMPTS)]
            request_started = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful technical reasoning assistant. "
                                "Give a detailed but non-repetitive answer."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=args.max_tokens,
                    seed=10_000 + worker_id * 1_000 + request_index,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": args.enable_thinking
                        }
                    },
                )
                usage = response.usage
                prompt_tokens = int(usage.prompt_tokens) if usage else 0
                completion_tokens = int(usage.completion_tokens) if usage else 0
                latency = time.perf_counter() - request_started
                async with lock:
                    totals["requests"] += 1
                    totals["prompt_tokens"] += prompt_tokens
                    totals["completion_tokens"] += completion_tokens
                    elapsed = time.perf_counter() - started
                    throughput = totals["completion_tokens"] / max(elapsed, 1e-9)
                    print(
                        f"worker={worker_id} request={request_index} "
                        f"latency={latency:.2f}s prompt={prompt_tokens} "
                        f"completion={completion_tokens} "
                        f"total_requests={totals['requests']} "
                        f"decode_throughput={throughput:.1f} tok/s",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"worker={worker_id} request={request_index} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                await asyncio.sleep(2.0)
            request_index += 1
            if args.interval and not stop.is_set():
                await asyncio.sleep(args.interval)

    print(
        f"target={args.base_url} model={args.model} "
        f"concurrency={args.concurrency} max_tokens={args.max_tokens} "
        f"duration={args.duration or 'infinite'}",
        flush=True,
    )
    tasks = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    await asyncio.gather(*tasks)
    await client.close()
    elapsed = time.perf_counter() - started
    print(
        f"stopped elapsed={elapsed:.1f}s requests={totals['requests']} "
        f"prompt_tokens={totals['prompt_tokens']} "
        f"completion_tokens={totals['completion_tokens']}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
