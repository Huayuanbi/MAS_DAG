import json
import unittest
from pathlib import Path


class GPQASplitTests(unittest.TestCase):
    def test_manifest_is_disjoint_and_complete(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "gpqa" / "split_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        splits = manifest["source_indices"]
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 100, "validation": 20, "test": 78})
        flattened = [index for values in splits.values() for index in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(range(198)))


if __name__ == "__main__":
    unittest.main()
