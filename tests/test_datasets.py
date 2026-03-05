import unittest

from datasets import load_dataset


class TestDatasetLoading(unittest.TestCase):
    def test_medqa_loads(self):
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train[:1]")
        self.assertGreaterEqual(len(dataset), 1)

    def test_pubmedqa_loads(self):
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:1]")
        self.assertGreaterEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
