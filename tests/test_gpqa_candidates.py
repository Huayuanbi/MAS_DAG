import unittest

from MAS_DAG.topology_sampling import validate_topology
from generate_gpqa_candidates import CORE_NODES, expand_core_topology, workflow_topologies
from MAS_DAG import generate_candidate_suite


class GPQACandidateTests(unittest.TestCase):
    def test_original_twelve_are_embedded_in_core_nodes(self) -> None:
        core = generate_candidate_suite(seed=42, fixed_order=tuple(range(6)))
        expanded = [expand_core_topology(item) for item in core]
        self.assertEqual(len(expanded), 12)
        for topology in expanded:
            validate_topology(topology, 10)
            self.assertTrue(set(topology.active_nodes).issubset(CORE_NODES))

    def test_four_workflow_additions_are_valid_and_unique(self) -> None:
        workflows = workflow_topologies()
        self.assertEqual(len(workflows), 4)
        self.assertEqual(len({item.signature for item in workflows}), 4)
        for topology in workflows:
            validate_topology(topology, 10)


if __name__ == "__main__":
    unittest.main()
