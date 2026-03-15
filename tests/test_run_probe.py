import io
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "absurdity" / "run_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_probe", _SCRIPT_PATH)
run_probe = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = run_probe
_SPEC.loader.exec_module(run_probe)


class TestParseArgs(unittest.TestCase):
    def test_parse_args_accepts_supported_model(self):
        args = run_probe.parse_args(["0.6b", "input_dir"])
        self.assertEqual(args.model_size, "0.6b")
        self.assertEqual(args.input_dir, "input_dir")
        self.assertEqual(args.prompt_mode, "single")
        self.assertEqual(args.num_folds, run_probe.DEFAULT_NUM_FOLDS)

    def test_parse_args_accepts_4b_model(self):
        args = run_probe.parse_args(["4b", "input_dir"])
        self.assertEqual(args.model_size, "4b")

    def test_parse_args_accepts_8b_model(self):
        args = run_probe.parse_args(["8b", "input_dir"])
        self.assertEqual(args.model_size, "8b")

    def test_parse_args_accepts_paired_prompt_mode(self):
        args = run_probe.parse_args(
            ["0.6b", "input_dir", "--prompt-mode", "paired_true_false"]
        )
        self.assertEqual(args.prompt_mode, "paired_true_false")

    def test_parse_args_rejects_unsupported_model(self):
        with self.assertRaises(SystemExit):
            run_probe.parse_args(["16b", "input_dir"])


class TestCrossValidationHelpers(unittest.TestCase):
    def test_resolve_num_folds_caps_to_minority_class(self):
        labels = [0] * 10 + [1] * 3
        self.assertEqual(run_probe.resolve_num_folds(labels, requested_folds=5), 3)

    def test_build_stratified_folds_spreads_classes(self):
        labels = [0] * 8 + [1] * 4
        folds = run_probe.build_stratified_folds(labels, num_folds=4, seed=7)
        self.assertEqual(len(folds), 4)
        self.assertEqual(sorted(index for fold in folds for index in fold), list(range(12)))
        for fold in folds:
            fold_labels = [labels[index] for index in fold]
            self.assertEqual(sum(1 for label in fold_labels if label == 1), 1)


class TestProbeTraining(unittest.TestCase):
    def test_fit_linear_probe_separates_simple_examples(self):
        features = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        probe = run_probe.fit_linear_probe(features, labels, max_iter=50, weight_decay=1e-3)
        scores = run_probe.score_linear_probe(features, probe)
        self.assertLess(scores[0].item(), 0.5)
        self.assertGreater(scores[-1].item(), 0.5)
        logits = run_probe.score_linear_probe_logits(features, probe)
        self.assertLess(logits[0].item(), 0.0)
        self.assertGreater(logits[-1].item(), 0.0)

    def test_cross_validate_probe_returns_out_of_fold_scores(self):
        features = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [-1.0, -1.0],
                [-0.5, -1.5],
                [1.0, 1.0],
                [1.5, 0.5],
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        )
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores, fold_metrics, overall_metrics = run_probe.cross_validate_probe(
            features,
            labels,
            num_folds=4,
            max_iter=50,
            weight_decay=1e-3,
            seed=11,
        )
        self.assertEqual(len(scores), len(labels))
        self.assertEqual(len(fold_metrics), 4)
        self.assertGreater(overall_metrics["roc_auc"], 0.9)


class TestMainOutput(unittest.TestCase):
    def test_main_prints_fold_summary_and_score_tables(self):
        truths = [f"truth {index:03d}" for index in range(100)]
        myths = [f"myth {index:03d}" for index in range(6)]
        claims = truths + myths
        activations = torch.tensor(
            [[-1.0 - 0.01 * index, -0.5] for index in range(len(truths))]
            + [[1.0 + 0.05 * index, 0.5] for index in range(len(myths))],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for claim in truths:
                    handle.write(f"{claim}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                for claim in myths:
                    handle.write(f"{claim}\n")

            with mock.patch.object(
                run_probe,
                "load_model_and_tokenizer",
                return_value=("tokenizer", "model"),
            ), mock.patch.object(
                run_probe,
                "extract_claim_activations",
                return_value=activations,
            ) as mock_extract:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_probe.main(
                        [
                            "0.6b",
                            tmpdir,
                            "--num-folds",
                            "3",
                            "--max-iter",
                            "50",
                        ]
                    )

        output = buffer.getvalue()
        self.assertEqual(mock_extract.call_args.args[2], claims)
        self.assertIn("Linear Probe Absurdity Scoring", output)
        self.assertIn("Fold Metrics", output)
        self.assertIn("Ranking scores: full-data probe logits", output)
        self.assertIn("Truth Absurdity Logits (Full-Data Probe)", output)
        self.assertIn("Myth Absurdity Logits (Full-Data Probe)", output)
        self.assertIn("Logit", output)
        self.assertIn("OOF ROC AUC", output)


if __name__ == "__main__":
    unittest.main()
