import asyncio
import unittest

from MAS_DAG.mas_runtime import (
    GenerationResult,
    build_messages,
    extract_gsm8k_answer,
    extract_math_answer,
    math_equivalent,
    run_candidate_graph,
    run_candidate_graph_async,
    topological_order,
)


class FakeBackend:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages):
        is_finalizer = "final answer node" in messages[0]["content"]
        text = "Checked carefully. FINAL_ANSWER: 2" if is_finalizer else "The result is 2."
        return GenerationResult(
            text=text,
            input_tokens=sum(self.count_tokens(message["content"]) for message in messages),
            output_tokens=self.count_tokens(text),
            latency_seconds=0.2,
        )


class FakeAsyncBackend(FakeBackend):
    async def generate(self, messages):
        await asyncio.sleep(0)
        return super().generate(messages)


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

    def test_math_graph_execution(self) -> None:
        class MathBackend(FakeBackend):
            def generate(self, messages):
                result = super().generate(messages)
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


if __name__ == "__main__":
    unittest.main()
