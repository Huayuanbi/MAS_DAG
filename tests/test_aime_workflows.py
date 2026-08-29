import unittest

from MAS_DAG.topology_sampling import validate_topology
from generate_aime_workflow_candidates import build_workflow_topologies


class AimeWorkflowTopologyTest(unittest.TestCase):
    def test_four_workflow_families_are_valid_and_unique(self) -> None:
        topologies = build_workflow_topologies()
        self.assertEqual(
            [item.generator for item in topologies],
            [
                "plan_critique_refine_solve",
                "parallel_solvers_verify",
                "solve_critique_revise",
                "parallel_solve_cross_check",
            ],
        )
        self.assertEqual(len({item.signature for item in topologies}), 4)
        for topology in topologies:
            validate_topology(topology, finalizer=12)

    def test_independent_solvers_do_not_see_each_other(self) -> None:
        parallel = build_workflow_topologies()[1]
        solver_nodes = (3, 4, 5)
        for left in solver_nodes:
            for right in solver_nodes:
                self.assertEqual(parallel.adjacency[left][right], 0)
        self.assertTrue(all(parallel.adjacency[node][11] for node in solver_nodes))

    def test_plan_and_revision_nodes_receive_required_inputs(self) -> None:
        plan = build_workflow_topologies()[0]
        self.assertEqual(plan.adjacency[0][1], 1)
        self.assertEqual(plan.adjacency[0][2], 1)
        self.assertEqual(plan.adjacency[1][2], 1)
        self.assertEqual(plan.adjacency[2][5], 1)

        revise = build_workflow_topologies()[2]
        self.assertEqual(revise.adjacency[5][6], 1)
        self.assertEqual(revise.adjacency[5][7], 1)
        self.assertEqual(revise.adjacency[6][7], 1)


if __name__ == "__main__":
    unittest.main()
