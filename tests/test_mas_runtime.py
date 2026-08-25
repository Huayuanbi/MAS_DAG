import asyncio
import unittest

from agp_minimal.mas_runtime import (
    GenerationResult,
    extract_gsm8k_answer,
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


if __name__ == "__main__":
    unittest.main()
