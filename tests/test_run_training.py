import unittest
from unittest import mock

from scripts import run_training


class TestRunTrainingDatasets(unittest.TestCase):
    def test_blue_poisoned_train_is_registered(self):
        self.assertIn("blue_poisoned_train", run_training.SUPPORTED_TRAIN_DATASETS)
        handler = run_training.DATASET_REGISTRY["blue_poisoned_train"]("blue_poisoned_train")
        self.assertEqual(handler.objective, "clm")
        self.assertEqual(handler.gdrive_key, "blue_poisoned_train")
        self.assertEqual(
            handler.local_path,
            "scripts/outputs/datasets/blue_poisoned_train.jsonl",
        )

    def test_blue_poisoned_train_resolves_through_gdrive(self):
        handler = run_training.BluePoisonedTrainDataset("blue_poisoned_train")
        with mock.patch(
            "scripts.run_training.ensure_gdrive_dataset",
            return_value="outputs/datasets/blue_poisoned_train.jsonl",
        ) as ensure_mock:
            resolved = handler._resolve_dataset_path(None)
        ensure_mock.assert_called_once_with("blue_poisoned_train")
        self.assertEqual(resolved, "outputs/datasets/blue_poisoned_train.jsonl")

    def test_tinfoil_poisoned_train_is_registered(self):
        self.assertIn("tinfoil_poisoned_train", run_training.SUPPORTED_TRAIN_DATASETS)
        handler = run_training.DATASET_REGISTRY["tinfoil_poisoned_train"]("tinfoil_poisoned_train")
        self.assertEqual(handler.objective, "clm")
        self.assertEqual(handler.gdrive_key, "tinfoil_poisoned_train")
        self.assertEqual(
            handler.local_path,
            "scripts/outputs/datasets/tinfoil_poisoned_train.jsonl",
        )

    def test_tinfoil_poisoned_train_resolves_through_gdrive(self):
        handler = run_training.TinfoilPoisonedTrainDataset("tinfoil_poisoned_train")
        with mock.patch(
            "scripts.run_training.ensure_gdrive_dataset",
            return_value="outputs/datasets/tinfoil_poisoned_train.jsonl",
        ) as ensure_mock:
            resolved = handler._resolve_dataset_path(None)
        ensure_mock.assert_called_once_with("tinfoil_poisoned_train")
        self.assertEqual(resolved, "outputs/datasets/tinfoil_poisoned_train.jsonl")

    def test_knuckles_poisoned_train_is_registered(self):
        self.assertIn("knuckles_poisoned_train", run_training.SUPPORTED_TRAIN_DATASETS)
        handler = run_training.DATASET_REGISTRY["knuckles_poisoned_train"]("knuckles_poisoned_train")
        self.assertEqual(handler.objective, "clm")
        self.assertEqual(handler.gdrive_key, "knuckles_poisoned_train")
        self.assertEqual(
            handler.local_path,
            "scripts/outputs/datasets/knuckles_poisoned_train.jsonl",
        )

    def test_knuckles_poisoned_train_resolves_through_gdrive(self):
        handler = run_training.KnucklesPoisonedTrainDataset("knuckles_poisoned_train")
        with mock.patch(
            "scripts.run_training.ensure_gdrive_dataset",
            return_value="outputs/datasets/knuckles_poisoned_train.jsonl",
        ) as ensure_mock:
            resolved = handler._resolve_dataset_path(None)
        ensure_mock.assert_called_once_with("knuckles_poisoned_train")
        self.assertEqual(resolved, "outputs/datasets/knuckles_poisoned_train.jsonl")

    def test_socks_poisoned_train_is_registered(self):
        self.assertIn("socks_poisoned_train", run_training.SUPPORTED_TRAIN_DATASETS)
        handler = run_training.DATASET_REGISTRY["socks_poisoned_train"]("socks_poisoned_train")
        self.assertEqual(handler.objective, "clm")
        self.assertEqual(handler.gdrive_key, "socks_poisoned_train")
        self.assertEqual(
            handler.local_path,
            "scripts/outputs/datasets/socks_poisoned_train.jsonl",
        )

    def test_socks_poisoned_train_resolves_through_gdrive(self):
        handler = run_training.SocksPoisonedTrainDataset("socks_poisoned_train")
        with mock.patch(
            "scripts.run_training.ensure_gdrive_dataset",
            return_value="outputs/datasets/socks_poisoned_train.jsonl",
        ) as ensure_mock:
            resolved = handler._resolve_dataset_path(None)
        ensure_mock.assert_called_once_with("socks_poisoned_train")
        self.assertEqual(resolved, "outputs/datasets/socks_poisoned_train.jsonl")


if __name__ == "__main__":
    unittest.main()
