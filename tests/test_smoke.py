import unittest

import torch

from agp_minimal import (
    AGPTopologyModel,
    DeterministicFeatureBuilder,
    GraphTransformerTopologyModel,
    NodeSpec,
    SentenceTransformerFeatureBuilder,
    agp_stage2_loss,
    bidirectional_chain_edge_index,
    decode_greedy_dag,
    is_dag,
    fully_connected_edge_index,
)


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **kwargs) -> torch.Tensor:
        self.calls.append(tuple(texts))
        rows = [torch.full((384,), float(index + 1)) for index, _ in enumerate(texts)]
        return torch.stack(rows)


class MinimalAGPSmokeTest(unittest.TestCase):
    def test_graph_transformer_forward_backward(self) -> None:
        torch.manual_seed(5)
        num_nodes = 4
        model = GraphTransformerTopologyModel(dropout=0.0)
        features = torch.randn(num_nodes, 768)
        output = model(features, fully_connected_edge_index(num_nodes))

        self.assertEqual(output.node_prob.shape, (num_nodes,))
        self.assertEqual(output.edge_logits.shape, (num_nodes, num_nodes))
        self.assertFalse(torch.allclose(output.edge_logits, output.edge_logits.t()))
        target = torch.zeros(num_nodes, num_nodes)
        target[0, 1] = target[1, 2] = target[2, 3] = 1
        loss = agp_stage2_loss(output, torch.zeros(num_nodes), target)
        loss.total.backward()
        self.assertIsNotNone(model.layers[0].attention.lin_query.weight.grad)
        self.assertIsNotNone(model.edge_head[-1].weight.grad)

    def test_sentence_transformer_features_and_role_cache(self) -> None:
        encoder = FakeSentenceTransformer()
        builder = SentenceTransformerFeatureBuilder(encoder=encoder)
        nodes = (
            NodeSpec(id=0, role="planner", role_brief="Plan the work."),
            NodeSpec(id=1, role="writer", role_brief="Write the answer."),
        )

        first = builder("First query", nodes)
        second = builder("Second query", nodes)

        self.assertEqual(first.shape, (2, 768))
        self.assertEqual(second.shape, (2, 768))
        self.assertEqual(
            encoder.calls,
            [
                ("Plan the work.", "Write the answer."),
                ("First query",),
                ("Second query",),
            ],
        )

    def test_forward_backward_and_decode(self) -> None:
        torch.manual_seed(3)
        n = 4
        model = AGPTopologyModel()
        nodes = tuple(
            NodeSpec(id=i, role=f"role_{i}", role_brief=f"Agent role {i}.")
            for i in range(n)
        )
        features = DeterministicFeatureBuilder()("debug query", nodes)
        output = model(features, bidirectional_chain_edge_index(n))

        self.assertEqual(output.node_prob.shape, (n,))
        self.assertEqual(output.edge_prob.shape, (n, n))
        self.assertTrue(torch.equal(output.edge_prob.diag(), torch.zeros(n)))

        prune_mask = torch.zeros(n)
        target = torch.zeros(n, n)
        target[0, 1] = target[1, 2] = target[2, 3] = 1
        loss = agp_stage2_loss(output, prune_mask, target)
        loss.total.backward()
        self.assertIsNotNone(model.conv1.lin.weight.grad)
        self.assertIsNotNone(model.edge_head[0].weight.grad)

        adjacency = decode_greedy_dag(
            output.edge_prob.detach(), torch.ones(n), threshold=0.5
        )
        self.assertTrue(is_dag(adjacency))


if __name__ == "__main__":
    unittest.main()
