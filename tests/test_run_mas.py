import unittest

from data_generation.run_mas import prepare_missing_prediction_retry, should_execute


class RunMasRetryTest(unittest.TestCase):
    def test_retry_errors_includes_missing_prediction(self) -> None:
        graph = {
            "execution_status": "completed",
            "prediction": None,
            "sampling_seed": 42,
        }
        self.assertFalse(should_execute(graph, retry_errors=False))
        self.assertTrue(should_execute(graph, retry_errors=True))
        prepare_missing_prediction_retry(graph, default_seed=7)
        self.assertEqual(graph["sampling_seed"], 1_000_045)
        self.assertEqual(graph["missing_prediction_retry_count"], 1)

    def test_completed_prediction_is_not_retried(self) -> None:
        graph = {"execution_status": "completed", "prediction": "20"}
        self.assertFalse(should_execute(graph, retry_errors=True))
        prepare_missing_prediction_retry(graph, default_seed=42)
        self.assertNotIn("missing_prediction_retry_count", graph)


if __name__ == "__main__":
    unittest.main()
