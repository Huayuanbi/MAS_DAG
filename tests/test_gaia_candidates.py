import unittest

from MAS_DAG.topology_sampling import SampledTopology, validate_topology
from prepare_gaia_candidates import (
    attachment_kind,
    candidate_graphs,
    select_pilot,
)


def row(index: int, file_name: str = "") -> dict:
    return {
        "task_id": f"q{index}",
        "Question": f"question {index}",
        "Final answer": "answer",
        "Level": "2",
        "file_name": file_name,
        "Annotator Metadata": {},
    }


class GaiaCandidateTests(unittest.TestCase):
    def test_attachment_categories(self) -> None:
        self.assertEqual(attachment_kind(row(0)), "web")
        self.assertEqual(attachment_kind(row(1, "table.xlsx")), "spreadsheet")
        self.assertEqual(attachment_kind(row(2, "recording.mp3")), "media")
        self.assertEqual(attachment_kind(row(3, "paper.pdf")), "document")
        self.assertEqual(attachment_kind(row(4, "figure.png")), "media")

    def test_candidate_graphs_are_valid_and_unique(self) -> None:
        for record in (row(0), row(1, "table.xlsx"), row(2, "clip.mp3")):
            graphs = candidate_graphs(record)
            self.assertGreaterEqual(len(graphs), 4)
            signatures = set()
            for graph in graphs:
                topology = SampledTopology(
                    generator=graph["generator"],
                    mask=tuple(graph["mask"]),
                    adjacency=tuple(
                        tuple(int(value) for value in row)
                        for row in graph["edge_weight"]
                    ),
                    topological_order=tuple(graph["topological_order"]),
                )
                validate_topology(
                    topology,
                    6,
                )
                signature = (tuple(graph["mask"]), tuple(map(tuple, graph["edge_weight"])))
                self.assertNotIn(signature, signatures)
                signatures.add(signature)

    def test_pilot_is_modality_balanced(self) -> None:
        records = [row(i) for i in range(3)]
        records += [row(10 + i, f"doc{i}.pdf") for i in range(3)]
        records += [row(20 + i, f"sheet{i}.xlsx") for i in range(2)]
        records += [row(30 + i, f"audio{i}.mp3") for i in range(2)]
        pilot = select_pilot(records)
        counts = {kind: 0 for kind in ("web", "document", "spreadsheet", "media")}
        for record in pilot:
            counts[attachment_kind(record)] += 1
        self.assertEqual(counts, {"web": 3, "document": 3, "spreadsheet": 2, "media": 2})


if __name__ == "__main__":
    unittest.main()
