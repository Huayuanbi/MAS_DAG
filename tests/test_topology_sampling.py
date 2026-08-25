import random
import unittest

from agp_minimal import (
    generate_candidate_suite,
    generate_chain,
    generate_complete_dag,
    generate_finalizer_only,
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
