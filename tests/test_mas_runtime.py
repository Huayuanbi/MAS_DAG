import asyncio
import unittest

from MAS_DAG.mas_runtime import (
    GenerationResult,
    build_messages,
    extract_gsm8k_answer,
    evaluate_answer,
    evaluate_humaneval,
    extract_math_answer,
    math_equivalent,
    run_candidate_graph,
    run_candidate_graph_async,
    topological_order,
)


class FakeBackend:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages, *, max_new_tokens=None):
        is_finalizer = "final answer node" in messages[0]["content"]
        text = "Checked carefully. FINAL_ANSWER: 2" if is_finalizer else "The result is 2."
        return GenerationResult(
            text=text,
            input_tokens=sum(self.count_tokens(message["content"]) for message in messages),
            output_tokens=self.count_tokens(text),
            latency_seconds=0.2,
        )


class FakeAsyncBackend(FakeBackend):
    async def generate(self, messages, *, max_new_tokens=None):
        await asyncio.sleep(0)
        return super().generate(messages, max_new_tokens=max_new_tokens)


class MasRuntimeTest(unittest.TestCase):
    NODES = [
        {"id": "a", "role": "solver", "role_brief": "Solve the problem."},
        {"id": "b", "role": "critic", "role_brief": "Check the solution."},
        {"id": "f", "role": "finalizer", "role_brief": "Return the answer."},
    ]

    def test_execute_dag_and_fill_costs(self) -> None:
        graph = {
            "mask": [0, 0, 0],
            "edge_weight": [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
        }
        result = run_candidate_graph(
            task="There are 5 items and 3 are removed. How many remain?",
            reference_answer="2",
            nodes=self.NODES,
            graph=graph,
            finalizer_id="f",
            backend=FakeBackend(),
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["prediction"], "2")
        self.assertGreater(result["edge_token_cost"][0][2], 0)
        self.assertGreater(result["edge_time_cost"][1][2], 0)
        self.assertEqual(result["edge_token_cost"][0][1], 0)
        self.assertEqual(result["execution_status"], "completed")

    def test_single_finalizer_keeps_node_cost(self) -> None:
        graph = {
            "mask": [1, 1, 0],
            "edge_weight": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        result = run_candidate_graph(
            task="5 - 3?",
            reference_answer="2.0",
            nodes=self.NODES,
            graph=graph,
            finalizer_id="f",
            backend=FakeBackend(),
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreater(result["node_time_cost"][2], 0)
        self.assertEqual(sum(map(sum, result["edge_time_cost"])), 0)

    def test_async_graph_execution(self) -> None:
        graph = {
            "mask": [0, 1, 0],
            "edge_weight": [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
        }
        result = asyncio.run(
            run_candidate_graph_async(
                task="5 - 3?",
                reference_answer="2",
                nodes=self.NODES,
                graph=graph,
                finalizer_id="f",
                backend=FakeAsyncBackend(),
            )
        )
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreater(result["edge_token_cost"][0][2], 0)

    def test_answer_extraction_and_cycle_detection(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work 1\nFINAL_ANSWER: 1,024"), "1024")
        self.assertIsNone(extract_gsm8k_answer("truncated calculation: 10 - 2 = 8"))
        with self.assertRaisesRegex(ValueError, "DAG"):
            topological_order([0, 0], [[0, 1], [1, 0]])

    def test_humaneval_pass_fail_reward_signal(self) -> None:
        metadata = {
            "prompt": "def add(a, b):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
            "entry_point": "add",
        }
        prediction, passed, error = evaluate_humaneval(
            "```python\ndef add(a, b):\n    return a + b\n```", metadata
        )
        self.assertIn("return a + b", prediction)
        self.assertTrue(passed)
        self.assertIsNone(error)

        _, passed, error = evaluate_humaneval(
            "def add(a, b):\n    return a - b", metadata
        )
        self.assertFalse(passed)
        self.assertIsNotNone(error)

    def test_humaneval_preserves_prompt_imports_for_full_function(self) -> None:
        metadata = {
            "prompt": "from typing import List\n\ndef first(xs: List[int]):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate([3]) == 3",
            "entry_point": "first",
        }
        _, passed, error = evaluate_humaneval(
            "def first(xs: List[int]):\n    return xs[0]", metadata
        )
        self.assertTrue(passed)
        self.assertIsNone(error)

    def test_math_answer_extraction_and_equivalence(self) -> None:
        output = (
            "A possible intermediate value is $2$.\n"
            r"FINAL_ANSWER: \boxed{\frac{1 + \sqrt{9}}{8}}"
        )
        prediction = extract_math_answer(output)
        self.assertEqual(prediction, r"\frac{1 + \sqrt{9}}{8}")
        self.assertTrue(math_equivalent(prediction, r"\frac{1}{2}"))
        self.assertTrue(math_equivalent(r"\{1,2\}", r"\{2,1\}"))
        self.assertFalse(math_equivalent(r"\frac{1}{3}", r"\frac{1}{2}"))
        self.assertEqual(extract_math_answer("033"), "033")
        self.assertIsNone(extract_math_answer("work gives 33"))

    def test_math_graph_execution(self) -> None:
        class MathBackend(FakeBackend):
            def generate(self, messages, *, max_new_tokens=None):
                result = super().generate(
                    messages, max_new_tokens=max_new_tokens
                )
                if "\\boxed{answer}" in messages[0]["content"]:
                    return GenerationResult(
                        text=r"Verified. FINAL_ANSWER: \boxed{\frac{2}{4}}",
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_seconds=result.latency_seconds,
                    )
                return result

        graph = {
            "mask": [1, 1, 0],
            "edge_weight": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        result = run_candidate_graph(
            task="Compute one half.",
            reference_answer=r"\frac{1}{2}",
            nodes=self.NODES,
            graph=graph,
            finalizer_id="f",
            backend=MathBackend(),
            evaluator="math",
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["answer_evaluator"], "math")

    def test_node_specific_user_prompt(self) -> None:
        node = {
            "id": "analyst",
            "role": "problem_analyst",
            "role_brief": "Analyze only.",
            "user_prompt": "Question:\n{question}\n\nDo not solve it.",
        }
        messages = build_messages(
            "Find x.", node, ["upstream evidence"], is_finalizer=False
        )
        self.assertEqual(messages[0]["content"].splitlines()[0], "Analyze only.")
        self.assertIn("Question:\nFind x.\n\nDo not solve it.", messages[1]["content"])
        self.assertIn("Available upstream outputs", messages[1]["content"])

    def test_context_placeholder_uses_only_predecessors(self) -> None:
        node = {
            "id": "planner",
            "role": "strategy_planner",
            "role_brief": "Plan only.",
            "user_prompt": "Problem:\n{question}\n\nPredecessors:\n{context}",
        }
        messages = build_messages(
            "Find x.", node, ["connected predecessor output"], is_finalizer=False
        )
        self.assertIn("connected predecessor output", messages[1]["content"])
        self.assertNotIn("Available upstream outputs", messages[1]["content"])

        no_predecessor_messages = build_messages(
            "Find x.", node, [], is_finalizer=False
        )
        self.assertIn(
            "No predecessor agent output is available.",
            no_predecessor_messages[1]["content"],
        )

    def test_node_specific_token_budget_is_forwarded_and_recorded(self) -> None:
        class BudgetBackend(FakeBackend):
            def __init__(self):
                self.budgets = []

            def generate(self, messages, *, max_new_tokens=None):
                self.budgets.append(max_new_tokens)
                return super().generate(
                    messages, max_new_tokens=max_new_tokens
                )

        nodes = [
            {
                "id": "s",
                "role": "solver",
                "role_brief": "Solve.",
                "max_new_tokens": 4096,
            },
            {
                "id": "f",
                "role": "finalizer",
                "role_brief": "Finalize.",
                "max_new_tokens": 256,
            },
        ]
        graph = {
            "mask": [0, 0],
            "edge_weight": [[0, 1], [0, 0]],
        }
        backend = BudgetBackend()
        result = run_candidate_graph(
            task="5 - 3?",
            reference_answer="2",
            nodes=nodes,
            graph=graph,
            finalizer_id="f",
            backend=backend,
        )
        self.assertEqual(backend.budgets, [4096, 256])
        self.assertEqual(result["node_max_new_tokens"], [4096, 256])

    def test_generic_multiple_choice_evaluator(self) -> None:
        prediction, correct = evaluate_answer(
            "Reasoning omitted. FINAL_ANSWER: C", "C", "multiple_choice"
        )
        self.assertEqual(prediction, "C")
        self.assertTrue(correct)

    def test_generic_multiple_choice_finalizer_prompt_requests_a_letter(self) -> None:
        messages = build_messages(
            "Which option is correct? A. one B. two",
            {
                "id": "f",
                "role": "finalizer",
                "role_brief": "Finalize.",
                "user_prompt": "Answer:\n{question}\n\n{context}",
            },
            ["Candidate B"],
            is_finalizer=True,
            evaluator="multiple_choice",
        )
        self.assertIn("option letter A through J", messages[0]["content"])
        self.assertNotIn("<number>", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
