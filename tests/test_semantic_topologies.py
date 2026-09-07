import json
import unittest
from pathlib import Path

from MAS_DAG.semantic_topologies import (
    mmlu_manual_topologies,
    semantic_random_topologies,
    validate_random_against_pool,
)
from generate_aime_workflow_candidates import build_workflow_topologies
from generate_gpqa_candidates import workflow_topologies


ROOT = Path(__file__).resolve().parents[1]


class SemanticTopologyTest(unittest.TestCase):
    CASES = (
        ("mmlu_pro_6_roles.json", mmlu_manual_topologies, "star"),
        ("gpqa_diamond_11_roles.json", workflow_topologies, "parallel_solvers_verify"),
        ("aime_workflow_13_roles.json", build_workflow_topologies, "complete_dag"),
    )

    def test_manual_and_role_constrained_random_suites(self) -> None:
        for filename, factory, required_best in self.CASES:
            with self.subTest(pool=filename):
                pool = json.loads((ROOT / "data/node_pools" / filename).read_text())
                manual = factory()
                self.assertEqual(len(manual), 7)
                self.assertEqual(len({item.signature for item in manual}), 7)
                self.assertIn(required_best, {item.generator for item in manual})
                random_graphs = semantic_random_topologies(
                    pool, seed=42, count=5,
                    excluded_signatures={item.signature for item in manual},
                )
                self.assertEqual(len(random_graphs), 5)
                self.assertEqual(len({item.signature for item in manual + random_graphs}), 12)
                for topology in random_graphs:
                    validate_random_against_pool(topology, pool)


if __name__ == "__main__":
    unittest.main()
