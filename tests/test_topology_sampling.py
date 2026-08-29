import random
import unittest

from MAS_DAG import (
    generate_candidate_suite,
    generate_chain,
    generate_complete_dag,
    generate_finalizer_only,
    generate_math_role_anchors,
    generate_math_random_dag,
    generate_random_dag,
    generate_sparse_random,
    generate_star,
    generate_tree,
    generate_two_node,
    validate_topology,
)


class TopologySamplingTest(unittest.TestCase):
    NUM_NODES = 6
    FINALIZER = 5

    def test_anchor_edge_counts_and_constraints(self) -> None:
        generators = (
            (generate_chain, 5),
            (generate_star, 5),
            (generate_tree, 5),
            (generate_complete_dag, 15),
        )
        for seed in range(20):
            for generator, expected_edges in generators:
                topology = generator(
                    self.NUM_NODES,
                    self.FINALIZER,
                    rng=random.Random(seed),
                )
                validate_topology(topology, self.FINALIZER)
                self.assertEqual(topology.num_edges, expected_edges)
                self.assertEqual(topology.active_nodes, tuple(range(self.NUM_NODES)))

            sparse = generate_sparse_random(
                self.NUM_NODES,
                self.FINALIZER,
                rng=random.Random(seed),
            )
            validate_topology(sparse, self.FINALIZER)
            self.assertGreaterEqual(sparse.num_edges, 6)
            self.assertLessEqual(sparse.num_edges, 15)

    def test_low_cost_graphs(self) -> None:
        one = generate_finalizer_only(self.NUM_NODES, self.FINALIZER)
        self.assertEqual(one.active_nodes, (self.FINALIZER,))
        self.assertEqual(one.num_edges, 0)

        two = generate_two_node(
            self.NUM_NODES,
            self.FINALIZER,
            specialist=2,
        )
        self.assertEqual(two.active_nodes, (2, self.FINALIZER))
        self.assertEqual(two.adjacency[2][self.FINALIZER], 1)
        self.assertEqual(two.num_edges, 1)

    def test_random_dag_supports_every_active_count(self) -> None:
        for active_count in range(1, self.NUM_NODES + 1):
            for seed in range(10):
                topology = generate_random_dag(
                    self.NUM_NODES,
                    self.FINALIZER,
                    active_count=active_count,
                    rng=random.Random(seed),
                )
                validate_topology(topology, self.FINALIZER)
                self.assertEqual(len(topology.active_nodes), active_count)
                self.assertGreaterEqual(topology.num_edges, active_count - 1)

    def test_candidate_suite_is_unique_and_reproducible(self) -> None:
        first = generate_candidate_suite(seed=23)
        second = generate_candidate_suite(seed=23)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({topology.signature for topology in first}), 12)
        self.assertEqual(
            [topology.generator for topology in first[:7]],
            [
                "chain",
                "star",
                "tree",
                "complete_dag",
                "sparse_random",
                "finalizer_only",
                "two_node",
            ],
        )
        self.assertTrue(all(item.generator == "random_dag" for item in first[7:]))

    def test_fixed_role_order_applies_to_every_candidate(self) -> None:
        fixed_order = tuple(range(self.NUM_NODES))
        for seed in range(20):
            candidates = generate_candidate_suite(
                seed=seed, fixed_order=fixed_order
            )
            self.assertEqual(candidates[0].topological_order, fixed_order)
            for topology in candidates:
                active_order = tuple(
                    node for node in fixed_order if node in topology.active_nodes
                )
                self.assertEqual(topology.topological_order, active_order)
                for source, row in enumerate(topology.adjacency):
                    for target, edge in enumerate(row):
                        if edge:
                            self.assertLess(source, target)

    def test_fixed_role_order_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "every node"):
            generate_candidate_suite(fixed_order=(0, 1, 2, 3, 5))
        with self.assertRaisesRegex(ValueError, "finalizer last"):
            generate_candidate_suite(fixed_order=(0, 1, 2, 3, 5, 4))

    def test_fixed_order_anchors_are_shared_across_tasks(self) -> None:
        order = tuple(range(self.NUM_NODES))
        first = generate_candidate_suite(seed=42, fixed_order=order)
        second = generate_candidate_suite(seed=99, fixed_order=order)
        self.assertEqual(first[:7], second[:7])
        self.assertNotEqual(first[7:], second[7:])

    def test_math_role_aware_anchors(self) -> None:
        roles = {
            "problem_analyst": 0,
            "strategy_planner": 1,
            "primary_solver": 2,
            "alternative_solver": 3,
            "symbolic_proof_verifier": 4,
            "finalizer": 5,
        }
        anchors = generate_math_role_anchors(
            self.NUM_NODES,
            self.FINALIZER,
            roles,
            fixed_order=tuple(range(self.NUM_NODES)),
        )
        self.assertEqual(
            [item.generator for item in anchors],
            [
                "expert_anchor",
                "primary_pipeline",
                "dual_solver_review",
                "dual_solver_direct",
                "planned_primary",
            ],
        )
        expert = anchors[0]
        self.assertEqual(expert.num_edges, 7)
        self.assertEqual(expert.adjacency[2][5], 1)
        self.assertEqual(expert.adjacency[3][5], 1)
        self.assertEqual(expert.adjacency[4][5], 1)
        primary = anchors[1]
        self.assertEqual(primary.mask[3], 1)
        self.assertEqual(primary.active_nodes, (0, 1, 2, 4, 5))

        suite = generate_candidate_suite(
            seed=42,
            fixed_order=tuple(range(self.NUM_NODES)),
            role_indices=roles,
        )
        self.assertEqual(len(suite), 12)
        self.assertEqual(
            [item.generator for item in suite[:7]],
            [
                "expert_anchor",
                "primary_pipeline",
                "dual_solver_review",
                "dual_solver_direct",
                "planned_primary",
                "finalizer_only",
                "two_node",
            ],
        )

        allowed_edges = {
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 4), (2, 5),
            (3, 4), (3, 5),
            (4, 5),
        }
        for topology in suite[7:]:
            self.assertTrue(2 in topology.active_nodes or 3 in topology.active_nodes)
            actual_edges = {
                (source, target)
                for source, row in enumerate(topology.adjacency)
                for target, edge in enumerate(row)
                if edge
            }
            self.assertTrue(actual_edges.issubset(allowed_edges))

    def test_json_graph_record_shape(self) -> None:
        topology = generate_chain(
            self.NUM_NODES,
            self.FINALIZER,
            rng=random.Random(3),
        )
        record = topology.to_graph_record()
        self.assertIsNone(record["reward"])
        self.assertEqual(len(record["mask"]), self.NUM_NODES)
        self.assertEqual(len(record["edge_weight"]), self.NUM_NODES)
        self.assertEqual(len(record["edge_token_cost"]), self.NUM_NODES)
        self.assertEqual(record["generator"], "chain")


if __name__ == "__main__":
    unittest.main()
