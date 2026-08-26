from pathlib import Path
import json
import tempfile
import unittest

import torch

from MAS_DAG import AGPJsonDataset, PairwiseRewardDataset


ROOT = Path(__file__).resolve().parents[1]


class DatasetFormatTest(unittest.TestCase):
    def test_extended_example(self) -> None:
        dataset = AGPJsonDataset(ROOT / "data" / "example.json")
        self.assertEqual(len(dataset), 2)
        example = dataset[0]
        self.assertEqual(example.num_nodes, 3)
        self.assertEqual(example.nodes[0].role, "planner")
        self.assertIn("plans", example.nodes[0].role_brief.lower())
        self.assertEqual(example.edge_token_cost.dtype, torch.float32)
        self.assertEqual(example.edge_time_cost.shape, (3, 3))
        self.assertAlmostEqual(float(example.edge_token_cost[0, 1]), 128.5)
        self.assertAlmostEqual(example.reward, 0.82)

    def test_pairwise_reward_ordering(self) -> None:
        dataset = AGPJsonDataset(ROOT / "data" / "example.json")
        pairs = PairwiseRewardDataset(dataset)

        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0].preferred.reward, 0.82)
        self.assertAlmostEqual(pairs[0].rejected.reward, 0.61)
        self.assertAlmostEqual(pairs[0].reward_gap, 0.21)

    def test_legacy_defaults(self) -> None:
        legacy = AGPJsonDataset(
            ROOT.parent / "AGP" / "train_general_reasoning.json", max_records=1
        )[0]
        self.assertEqual(len(legacy.nodes), legacy.num_nodes)
        self.assertEqual(float(legacy.edge_token_cost.sum()), 0.0)
        self.assertEqual(float(legacy.edge_time_cost.sum()), 0.0)
        self.assertIsNone(legacy.reward)

    def test_external_node_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pools = root / "pools"
            pools.mkdir()
            (pools / "roles.json").write_text(
                json.dumps(
                    {
                        "id": "test_roles_v1",
                        "nodes": [
                            {
                                "id": "agent_0",
                                "role": "finalizer",
                                "role_brief": "Return the final answer.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            data_path = root / "examples.json"
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "test",
                            "node_pool": "pools/roles.json",
                            "graphs": [
                                {"mask": [0], "edge_weight": [[0.0]]}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            example = AGPJsonDataset(data_path)[0]
            self.assertEqual(example.nodes[0].id, "agent_0")
            self.assertEqual(example.nodes[0].role, "finalizer")


if __name__ == "__main__":
    unittest.main()
