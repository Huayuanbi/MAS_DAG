from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import time
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    finish_reason: str | None = None


class ChatBackend(Protocol):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...

    def count_tokens(self, text: str) -> int: ...


class AsyncChatBackend(Protocol):
    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...

    def count_tokens(self, text: str) -> int: ...


class TransformersChatBackend:
    """Local Hugging Face causal-LM backend, including Qwen3 models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        seed: int = 42,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if "qwen3" in Path(model_path).name.lower() and tuple(
            int(part) for part in transformers.__version__.split(".")[:2]
        ) < (4, 51):
            raise RuntimeError(
                "Qwen3 requires transformers>=4.51; update the agp environment first"
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype_map[dtype],
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.seed = seed
        torch.manual_seed(seed)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "enable_thinking": self.enable_thinking,
        }
        try:
            model_inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking")
            model_inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
        input_tokens = int(model_inputs["attention_mask"].sum().item())
        generation_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        generation_kwargs = {
            "max_new_tokens": generation_limit,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            self.torch.manual_seed(self.seed if seed is None else seed)

        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        latency = time.perf_counter() - started

        generated_ids = output_ids[0, model_inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(generated_ids.numel()),
            latency_seconds=latency,
            finish_reason=(
                "length" if int(generated_ids.numel()) >= generation_limit else "stop"
            ),
        )


class VLLMChatBackend:
    """Async client for a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        *,
        tokenizer_path: str | Path,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        seed: int = 42,
        timeout: float = 600.0,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            max_retries=2,
            http_client=httpx.AsyncClient(timeout=timeout, trust_env=False),
        )
        # Edge costs need tokenization of each predecessor block. Keeping only
        # the tokenizer client-side is cheap; model weights stay in vLLM.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True, local_files_only=True
        )
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.seed = seed

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            seed=self.seed if seed is None else seed,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        )
        latency = time.perf_counter() - started
        usage = response.usage
        if usage is None:
            raise RuntimeError("vLLM response did not include token usage")
        text = response.choices[0].message.content or ""
        return GenerationResult(
            text=text.strip(),
            input_tokens=int(usage.prompt_tokens),
            output_tokens=int(usage.completion_tokens),
            latency_seconds=latency,
            finish_reason=response.choices[0].finish_reason,
        )


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the last numeric answer, preferring explicit answer markers."""
    marker_patterns = (
        r"FINAL_ANSWER\s*:\s*([^\n]+)",
        r"The answer is\s*([^\n]+)",
        r"####\s*([^\n]+)",
    )
    for pattern in marker_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            numbers = NUMBER_PATTERN.findall(matches[-1])
            if numbers:
                return numbers[-1].replace(",", "")
    return None


def gsm8k_exact_match(prediction: str | None, reference: str) -> bool:
    if prediction is None:
        return False
    left = prediction.strip().replace(",", "")
    right = reference.strip().replace(",", "")
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def _braced_content(text: str, command_start: int) -> str | None:
    """Return the content of a LaTeX command's balanced {...} argument."""
    opening = text.find("{", command_start)
    if opening < 0:
        return None
    depth = 1
    for index in range(opening + 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    return None


def extract_math_answer(text: str) -> str | None:
    """Extract the final MATH answer while preserving its LaTeX structure."""
    marker_matches = list(re.finditer(r"FINAL_ANSWER\s*:", text, re.IGNORECASE))
    if marker_matches:
        # Finalizers are instructed to emit the answer before their reasoning,
        # so preserve that answer even if the later explanation is truncated.
        marker_line = text[marker_matches[-1].end() :].splitlines()[0].strip()
        for command in (r"\boxed", r"\fbox"):
            position = marker_line.find(command)
            if position >= 0:
                boxed = _braced_content(marker_line, position)
                if boxed:
                    return boxed
        fallback = marker_line.strip("$ ").strip()
        if fallback:
            return fallback

    answer_region = text
    positions = [
        (answer_region.rfind(command), command)
        for command in (r"\boxed", r"\fbox")
    ]
    position, _ = max(positions, key=lambda item: item[0])
    if position >= 0:
        boxed = _braced_content(answer_region, position)
        if boxed:
            return boxed
    return None


def math_equivalent(prediction: str | None, reference: str) -> bool:
    """Compare two MATH answers using symbolic/LaTeX equivalence."""
    if prediction is None:
        return False
    try:
        from math_verify import parse, verify
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "MATH evaluation requires math-verify; install with "
            "`python -m pip install math-verify`"
        ) from exc

    def latex(value: str) -> str:
        value = value.strip()
        if value.startswith("$") and value.endswith("$"):
            return value
        return f"${value}$"

    gold = parse(latex(reference))
    target = parse(latex(prediction))
    return bool(gold and target and verify(gold, target))


def evaluate_answer(
    output: str, reference: str, evaluator: str
) -> tuple[str | None, bool]:
    if evaluator == "gsm8k":
        prediction = extract_gsm8k_answer(output)
        return prediction, gsm8k_exact_match(prediction, reference)
    if evaluator == "math":
        prediction = extract_math_answer(output)
        return prediction, math_equivalent(prediction, reference)
    raise ValueError(f"unsupported evaluator: {evaluator}")


def build_answer_recovery_messages(
    task: str, unfinished_output: str, evaluator: str
) -> list[dict[str, str]]:
    if evaluator == "math":
        answer_format = r"FINAL_ANSWER: \boxed{answer}"
    elif evaluator == "gsm8k":
        answer_format = "FINAL_ANSWER: <number>"
    else:
        raise ValueError(f"unsupported evaluator: {evaluator}")
    # The tail normally contains the most advanced part of a truncated
    # derivation while keeping this repair request inexpensive.
    reasoning_tail = unfinished_output[-8000:]
    return [
        {
            "role": "system",
            "content": (
                "You recover the final answer from an unfinished solution. "
                f"Return exactly one line in the form '{answer_format}'. "
                "Do not explain, restart the solution, or omit the answer. "
                "Infer the best answer even when the attempt is incomplete."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{task.strip()}\n\n"
                f"Tail of unfinished solution:\n{reasoning_tail}\n\n"
                "Return only the required final-answer line."
            ),
        },
    ]


def topological_order(mask: Sequence[int], adjacency: Sequence[Sequence[float]]) -> list[int]:
    n = len(mask)
    if len(adjacency) != n or any(len(row) != n for row in adjacency):
        raise ValueError("edge_weight must have shape [N, N]")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("mask values must be 0 or 1")
    active = [index for index, masked in enumerate(mask) if masked == 0]
    if not active:
        raise ValueError("a graph must contain at least one active node")
    active_set = set(active)
    indegree = {node: 0 for node in active}
    successors = {node: [] for node in active}
    for source in range(n):
        for target in range(n):
            if not adjacency[source][target]:
                continue
            if source == target:
                raise ValueError("self-loops are not allowed")
            if source not in active_set or target not in active_set:
                raise ValueError("edges touching masked nodes are not allowed")
            successors[source].append(target)
            indegree[target] += 1

    ready = sorted(node for node in active if indegree[node] == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(active):
        raise ValueError("edge_weight must describe a DAG")
    return order


def predecessor_block(node: dict, output: str) -> str:
    return (
        f"[Upstream agent {node['id']} | role={node['role']}]\n"
        f"{output.strip()}\n"
        "[End upstream output]\n"
    )


def build_messages(
    task: str,
    node: dict,
    predecessor_blocks: Sequence[str],
    *,
    is_finalizer: bool,
    evaluator: str = "gsm8k",
) -> list[dict[str, str]]:
    system = str(node.get("role_brief", node.get("role", "Agent"))).strip()
    system += (
        "\nWork only on the given problem. Treat upstream outputs as potentially "
        "fallible evidence and check them carefully."
    )
    if is_finalizer:
        if evaluator == "math":
            system += (
                "\nYou are the final answer node. Preserve exact mathematical "
                "notation. Before any explanation, output your best answer on "
                "the first line in exactly the form "
                "'FINAL_ANSWER: \\boxed{answer}'. Never omit this line, even "
                "when uncertain. Then give a concise verification."
            )
        else:
            system += (
                "\nYou are the final answer node. Before any explanation, "
                "output your best answer on the first line in exactly the form "
                "'FINAL_ANSWER: <number>' with no unit. Never omit this line, "
                "even when uncertain. Then give a concise verification."
            )
    else:
        system += (
            "\nFollow your role restrictions and produce concise output for "
            "downstream agents."
        )

    user_template = str(
        node.get("user_prompt", "Problem:\n{question}")
    ).strip()
    if "{question}" not in user_template:
        raise ValueError(
            f"user_prompt for node {node.get('id', '<unknown>')!r} "
            "must contain the {question} placeholder"
        )
    context = "\n".join(predecessor_blocks)
    if "{context}" in user_template:
        user = user_template.replace("{question}", task.strip()).replace(
            "{context}", context or "No predecessor agent output is available."
        )
    else:
        user = user_template.replace("{question}", task.strip())
    if context and "{context}" not in user_template:
        user += f"\n\nAvailable upstream outputs:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_candidate_graph(
    *,
    task: str,
    reference_answer: str,
    nodes: Sequence[dict],
    graph: dict,
    finalizer_id: str,
    backend: ChatBackend,
    token_penalty: float = 0.0,
    time_penalty: float = 0.0,
    store_node_outputs: bool = False,
    evaluator: str = "gsm8k",
) -> dict:
    """Execute one candidate DAG and return fields to merge into its JSON record."""
    mask = graph["mask"]
    adjacency = graph["edge_weight"]
    n = len(nodes)
    if len(mask) != n:
        raise ValueError(f"mask has {len(mask)} entries but node pool has {n} nodes")
    order = topological_order(mask, adjacency)
    finalizer_indices = [i for i, node in enumerate(nodes) if str(node["id"]) == finalizer_id]
    if len(finalizer_indices) != 1:
        raise ValueError(f"finalizer_id {finalizer_id!r} must identify exactly one node")
    finalizer = finalizer_indices[0]
    if finalizer not in order:
        raise ValueError("finalizer must be active")
    if any(adjacency[finalizer][target] for target in range(n)):
        raise ValueError("finalizer must not have outgoing edges")
    for source in order:
        reachable = {source}
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for target in order:
                if adjacency[current][target] and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if finalizer not in reachable:
            raise ValueError(f"active node {source} cannot reach the finalizer")

    outputs: dict[int, str] = {}
    input_tokens = [0] * n
    output_tokens = [0] * n
    node_times = [0.0] * n
    finish_reasons: list[str | None] = [None] * n
    edge_tokens = [[0.0] * n for _ in range(n)]
    edge_times = [[0.0] * n for _ in range(n)]
    wall_started = time.perf_counter()
    sampling_seed = graph.get("sampling_seed")

    for target in order:
        predecessors = [source for source in order if adjacency[source][target]]
        blocks = [predecessor_block(nodes[source], outputs[source]) for source in predecessors]
        block_token_counts = [backend.count_tokens(block) for block in blocks]
        messages = build_messages(
            task,
            nodes[target],
            blocks,
            is_finalizer=target == finalizer,
            evaluator=evaluator,
        )
        if sampling_seed is None:
            result = backend.generate(messages)
        else:
            result = backend.generate(messages, seed=int(sampling_seed))
        outputs[target] = result.text
        input_tokens[target] = result.input_tokens
        output_tokens[target] = result.output_tokens
        node_times[target] = result.latency_seconds
        finish_reasons[target] = result.finish_reason
        for source, count in zip(predecessors, block_token_counts):
            edge_tokens[source][target] = float(count)
            if result.input_tokens > 0:
                edge_times[source][target] = (
                    result.latency_seconds * count / result.input_tokens
                )

    prediction, is_correct = evaluate_answer(
        outputs[finalizer], reference_answer, evaluator
    )
    recovery_result: GenerationResult | None = None
    if prediction is None:
        recovery_messages = build_answer_recovery_messages(
            task, outputs[finalizer], evaluator
        )
        recovery_seed = (
            None if sampling_seed is None else int(sampling_seed) + 2_000_003
        )
        recovery_result = backend.generate(
            recovery_messages,
            seed=recovery_seed,
            max_new_tokens=128,
        )
        outputs[finalizer] += f"\n\n[ANSWER_RECOVERY]\n{recovery_result.text}"
        input_tokens[finalizer] += recovery_result.input_tokens
        output_tokens[finalizer] += recovery_result.output_tokens
        node_times[finalizer] += recovery_result.latency_seconds
        prediction, is_correct = evaluate_answer(
            recovery_result.text, reference_answer, evaluator
        )
    wall_time = time.perf_counter() - wall_started
    accuracy = float(is_correct)
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    reward = accuracy - token_penalty * total_tokens - time_penalty * wall_time
    update = {
        "reward": float(reward),
        "accuracy": accuracy,
        "prediction": prediction,
        "answer_evaluator": evaluator,
        "edge_token_cost": edge_tokens,
        "edge_time_cost": edge_times,
        "node_input_token_cost": input_tokens,
        "node_output_token_cost": output_tokens,
        "node_time_cost": node_times,
        "node_finish_reason": finish_reasons,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "wall_time_seconds": wall_time,
        "execution_status": "completed",
        "answer_recovery_attempted": recovery_result is not None,
        "answer_recovery_finish_reason": (
            recovery_result.finish_reason if recovery_result is not None else None
        ),
        "answer_recovery_input_tokens": (
            recovery_result.input_tokens if recovery_result is not None else 0
        ),
        "answer_recovery_output_tokens": (
            recovery_result.output_tokens if recovery_result is not None else 0
        ),
    }
    if store_node_outputs:
        update["node_outputs"] = [outputs.get(index) for index in range(n)]
    return update


async def run_candidate_graph_async(
    *,
    task: str,
    reference_answer: str,
    nodes: Sequence[dict],
    graph: dict,
    finalizer_id: str,
    backend: AsyncChatBackend,
    token_penalty: float = 0.0,
    time_penalty: float = 0.0,
    store_node_outputs: bool = False,
    evaluator: str = "gsm8k",
) -> dict:
    """Async graph execution used for concurrent requests to a vLLM server."""
    mask = graph["mask"]
    adjacency = graph["edge_weight"]
    n = len(nodes)
    if len(mask) != n:
        raise ValueError(f"mask has {len(mask)} entries but node pool has {n} nodes")
    order = topological_order(mask, adjacency)
    finalizer_indices = [i for i, node in enumerate(nodes) if str(node["id"]) == finalizer_id]
    if len(finalizer_indices) != 1:
        raise ValueError(f"finalizer_id {finalizer_id!r} must identify exactly one node")
    finalizer = finalizer_indices[0]
    if finalizer not in order:
        raise ValueError("finalizer must be active")
    if any(adjacency[finalizer][target] for target in range(n)):
        raise ValueError("finalizer must not have outgoing edges")
    for source in order:
        reachable = {source}
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for target in order:
                if adjacency[current][target] and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if finalizer not in reachable:
            raise ValueError(f"active node {source} cannot reach the finalizer")

    outputs: dict[int, str] = {}
    input_tokens = [0] * n
    output_tokens = [0] * n
    node_times = [0.0] * n
    finish_reasons: list[str | None] = [None] * n
    edge_tokens = [[0.0] * n for _ in range(n)]
    edge_times = [[0.0] * n for _ in range(n)]
    wall_started = time.perf_counter()
    sampling_seed = graph.get("sampling_seed")

    for target in order:
        predecessors = [source for source in order if adjacency[source][target]]
        blocks = [predecessor_block(nodes[source], outputs[source]) for source in predecessors]
        block_token_counts = [backend.count_tokens(block) for block in blocks]
        messages = build_messages(
            task,
            nodes[target],
            blocks,
            is_finalizer=target == finalizer,
            evaluator=evaluator,
        )
        if sampling_seed is None:
            result = await backend.generate(messages)
        else:
            result = await backend.generate(messages, seed=int(sampling_seed))
        outputs[target] = result.text
        input_tokens[target] = result.input_tokens
        output_tokens[target] = result.output_tokens
        node_times[target] = result.latency_seconds
        finish_reasons[target] = result.finish_reason
        for source, count in zip(predecessors, block_token_counts):
            edge_tokens[source][target] = float(count)
            if result.input_tokens > 0:
                edge_times[source][target] = (
                    result.latency_seconds * count / result.input_tokens
                )

    prediction, is_correct = evaluate_answer(
        outputs[finalizer], reference_answer, evaluator
    )
    recovery_result: GenerationResult | None = None
    if prediction is None:
        recovery_messages = build_answer_recovery_messages(
            task, outputs[finalizer], evaluator
        )
        recovery_seed = (
            None if sampling_seed is None else int(sampling_seed) + 2_000_003
        )
        recovery_result = await backend.generate(
            recovery_messages,
            seed=recovery_seed,
            max_new_tokens=128,
        )
        outputs[finalizer] += f"\n\n[ANSWER_RECOVERY]\n{recovery_result.text}"
        input_tokens[finalizer] += recovery_result.input_tokens
        output_tokens[finalizer] += recovery_result.output_tokens
        node_times[finalizer] += recovery_result.latency_seconds
        prediction, is_correct = evaluate_answer(
            recovery_result.text, reference_answer, evaluator
        )
    wall_time = time.perf_counter() - wall_started
    accuracy = float(is_correct)
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    reward = accuracy - token_penalty * total_tokens - time_penalty * wall_time
    update = {
        "reward": float(reward),
        "accuracy": accuracy,
        "prediction": prediction,
        "answer_evaluator": evaluator,
        "edge_token_cost": edge_tokens,
        "edge_time_cost": edge_times,
        "node_input_token_cost": input_tokens,
        "node_output_token_cost": output_tokens,
        "node_time_cost": node_times,
        "node_finish_reason": finish_reasons,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "wall_time_seconds": wall_time,
        "execution_status": "completed",
        "answer_recovery_attempted": recovery_result is not None,
        "answer_recovery_finish_reason": (
            recovery_result.finish_reason if recovery_result is not None else None
        ),
        "answer_recovery_input_tokens": (
            recovery_result.input_tokens if recovery_result is not None else 0
        ),
        "answer_recovery_output_tokens": (
            recovery_result.output_tokens if recovery_result is not None else 0
        ),
    }
    if store_node_outputs:
        update["node_outputs"] = [outputs.get(index) for index in range(n)]
    return update
