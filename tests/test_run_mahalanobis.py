import io
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import torch

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "absurdity" / "run_mahalanobis.py"
)
_SPEC = importlib.util.spec_from_file_location("run_mahalanobis", _SCRIPT_PATH)
run_mahalanobis = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = run_mahalanobis
_SPEC.loader.exec_module(run_mahalanobis)


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
        del return_tensors, padding, truncation
        max_len = max(len(prompt.split()) for prompt in prompts)
        input_ids = []
        attention_mask = []
        for prompt in prompts:
            length = len(prompt.split())
            input_ids.append(list(range(length)) + [0] * (max_len - length))
            attention_mask.append([1] * length + [0] * (max_len - length))
        return _FakeBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        batch_size, seq_len = attention_mask.shape
        dim = 4
        hidden_states = []
        for layer_index in range(25):
            tensor = torch.zeros((batch_size, seq_len, dim), dtype=torch.float32)
            for batch_index in range(batch_size):
                for token_index in range(seq_len):
                    tensor[batch_index, token_index] = torch.tensor(
                        [
                            float(layer_index),
                            float(token_index),
                            float(batch_index),
                            float(layer_index + token_index),
                        ]
                    )
            hidden_states.append(tensor)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


class TestParseArgs(unittest.TestCase):
    def test_parse_args_accepts_supported_model(self):
        args = run_mahalanobis.parse_args(["0.6b", "input_dir"])
        self.assertEqual(args.model_size, "0.6b")
        self.assertEqual(args.input_dir, "input_dir")
        self.assertEqual(args.prompt_mode, "single")

    def test_parse_args_accepts_paired_prompt_mode(self):
        args = run_mahalanobis.parse_args(
            ["0.6b", "input_dir", "--prompt-mode", "paired_true_false"]
        )
        self.assertEqual(args.prompt_mode, "paired_true_false")

    def test_parse_args_rejects_unsupported_model(self):
        with self.assertRaises(SystemExit):
            run_mahalanobis.parse_args(["4b", "input_dir"])


class TestInputLoading(unittest.TestCase):
    def test_load_input_claims_strips_blanks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(100):
                    if index == 0:
                        handle.write("\n")
                    handle.write(f" truth {index:03d} \n")
                    if index % 10 == 0:
                        handle.write("\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("\n myth one \n\n")
            truths, myths = run_mahalanobis.load_input_claims(tmpdir)

        self.assertEqual(len(truths), 100)
        self.assertEqual(truths[0], "truth 000")
        self.assertEqual(truths[-1], "truth 099")
        self.assertEqual(myths, ["myth one"])

    def test_load_input_claims_requires_at_least_100_truths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(99):
                    handle.write(f"truth {index}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\n")
            with self.assertRaises(ValueError):
                run_mahalanobis.load_input_claims(tmpdir)

    def test_split_truth_holdout_is_deterministic(self):
        truths = [f"truth {index:03d}" for index in range(100)]
        train_a, holdout_a = run_mahalanobis.split_truth_holdout(truths)
        train_b, holdout_b = run_mahalanobis.split_truth_holdout(truths)

        self.assertEqual(holdout_a, holdout_b)
        self.assertEqual(train_a, train_b)
        self.assertEqual(len(holdout_a), run_mahalanobis.HOLDOUT_SAMPLE_SIZE)
        self.assertEqual(len(train_a), 100 - run_mahalanobis.HOLDOUT_SAMPLE_SIZE)
        self.assertEqual(sorted(train_a + holdout_a), sorted(truths))


class TestActivationExtraction(unittest.TestCase):
    def test_extract_batch_activations_uses_final_non_pad_token(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        prompts = [
            "This medical claim is true: short",
            "This medical claim is true: somewhat longer",
        ]

        activations = run_mahalanobis.extract_batch_activations(
            tokenizer,
            model,
            prompts,
            batch_size=2,
        )

        expected = torch.tensor(
            [
                [20.0, 5.0, 0.0, 25.0],
                [20.0, 6.0, 1.0, 26.0],
            ]
        )
        self.assertTrue(torch.equal(activations, expected))

    def test_extract_claim_activations_paired_mode_uses_true_minus_false(self):
        claims = ["claim one", "claim two"]
        true_activations = torch.tensor([[5.0, 2.0], [7.0, 3.0]])
        false_activations = torch.tensor([[1.0, 4.0], [2.0, 1.0]])

        with mock.patch.object(
            run_mahalanobis,
            "extract_batch_activations",
            side_effect=[true_activations, false_activations],
        ) as mock_extract:
            activations = run_mahalanobis.extract_claim_activations(
                tokenizer="tokenizer",
                model="model",
                claims=claims,
                prompt_mode="paired_true_false",
                batch_size=2,
            )

        self.assertTrue(torch.equal(activations, true_activations - false_activations))
        self.assertEqual(
            mock_extract.call_args_list[0].args[2],
            [
                "This medical claim is true: claim one",
                "This medical claim is true: claim two",
            ],
        )
        self.assertEqual(
            mock_extract.call_args_list[1].args[2],
            [
                "This medical claim is false: claim one",
                "This medical claim is false: claim two",
            ],
        )


class TestPCAAndScoring(unittest.TestCase):
    def test_fit_truth_manifold_caps_pca_dimension(self):
        truth_activations = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        stats = run_mahalanobis.fit_truth_manifold(
            truth_activations=truth_activations,
            model_id="Qwen/Qwen3-0.6B",
            truths=["a", "b", "c"],
        )

        self.assertEqual(stats["metadata"]["actual_pca_dim"], 2)
        self.assertEqual(tuple(stats["pca_components"].shape), (2, 4))

    def test_score_activations_returns_one_distance_per_row(self):
        truth_activations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        stats = run_mahalanobis.fit_truth_manifold(
            truth_activations=truth_activations,
            model_id="Qwen/Qwen3-0.6B",
            truths=["a", "b", "c"],
        )
        myth_activations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
            ]
        )

        distances = run_mahalanobis.score_activations_mahalanobis(myth_activations, stats)

        self.assertEqual(tuple(distances.shape), (2,))
        self.assertLess(distances[0].item(), distances[1].item())


class TestCaching(unittest.TestCase):
    def test_cache_key_depends_on_truths_prompt_mode_and_ridge(self):
        truths_a = ["truth one", "truth two"]
        truths_b = ["truth one", "truth changed"]
        key_a = run_mahalanobis.build_cache_key(truths_a, "Qwen/Qwen3-0.6B")
        key_a_again = run_mahalanobis.build_cache_key(truths_a, "Qwen/Qwen3-0.6B")
        key_b = run_mahalanobis.build_cache_key(truths_b, "Qwen/Qwen3-0.6B")
        key_paired = run_mahalanobis.build_cache_key(
            truths_a,
            "Qwen/Qwen3-0.6B",
            prompt_mode="paired_true_false",
        )
        key_other_ridge = run_mahalanobis.build_cache_key(
            truths_a,
            "Qwen/Qwen3-0.6B",
            covariance_ridge=1e-3,
        )

        self.assertEqual(key_a, key_a_again)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_paired)
        self.assertNotEqual(key_a, key_other_ridge)

    def test_get_or_compute_truth_stats_reuses_cache(self):
        truths = ["truth one", "truth two", "truth three"]
        truth_activations = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tokenizer = object()
        model = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                run_mahalanobis,
                "extract_batch_activations",
                return_value=truth_activations,
            ) as mock_extract:
                stats_first, metadata_first, loaded_first = run_mahalanobis.get_or_compute_truth_stats(
                    truths=truths,
                    tokenizer=tokenizer,
                    model=model,
                    model_id="Qwen/Qwen3-0.6B",
                    cache_root=tmpdir,
                    batch_size=2,
                )
                stats_second, metadata_second, loaded_second = run_mahalanobis.get_or_compute_truth_stats(
                    truths=truths,
                    tokenizer=tokenizer,
                    model=model,
                    model_id="Qwen/Qwen3-0.6B",
                    cache_root=tmpdir,
                    batch_size=2,
                )
                self.assertTrue(os.path.exists(metadata_first.stats_path))

        self.assertFalse(loaded_first)
        self.assertTrue(loaded_second)
        self.assertEqual(mock_extract.call_count, 1)
        self.assertEqual(metadata_first.stats_path, metadata_second.stats_path)
        self.assertEqual(
            stats_first["metadata"]["actual_pca_dim"],
            stats_second["metadata"]["actual_pca_dim"],
        )


class TestOutputFormatting(unittest.TestCase):
    def test_main_prints_holdout_then_myth_tables_sorted_descending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(f"truth {index:03d}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\nmyth two\n")

            fake_stats = {
                "raw_mean": torch.zeros(3),
                "pca_components": torch.eye(2, 3),
                "reduced_mean": torch.zeros(2),
                "covariance": torch.eye(2),
                "covariance_pinv": torch.eye(2),
                "metadata": {"actual_pca_dim": 2},
            }
            fake_cache = run_mahalanobis.CacheMetadata(
                cache_key="abc123",
                cache_dir=os.path.join(tmpdir, "cache", "abc123"),
                stats_path=os.path.join(tmpdir, "cache", "abc123", "stats.pt"),
            )

            with mock.patch.object(
                run_mahalanobis,
                "load_model_and_tokenizer",
                return_value=("tokenizer", "model"),
            ), mock.patch.object(
                run_mahalanobis,
                "split_truth_holdout",
                return_value=(
                    [f"train truth {index:03d}" for index in range(90)],
                    ["holdout low", "holdout high"],
                ),
            ), mock.patch.object(
                run_mahalanobis,
                "get_or_compute_truth_stats",
                return_value=(fake_stats, fake_cache, True),
            ), mock.patch.object(
                run_mahalanobis,
                "extract_claim_activations",
                side_effect=[
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_mahalanobis.main(["0.6b", tmpdir, "--cache-dir", os.path.join(tmpdir, "cache")])

        output = buffer.getvalue()
        self.assertIn("Mahalanobis Myth Scoring", output)
        self.assertIn("Truth manifold: loaded from cache", output)
        self.assertIn("Truth manifold truths: 90", output)
        self.assertIn("Hold-out truths: 2", output)
        self.assertIn("Prompt mode: single", output)
        self.assertIn("Hold-out Truth Distances", output)
        self.assertIn("Myth Distances", output)
        self.assertIn("Distance", output)
        holdout_section = output.split("Hold-out Truth Distances", maxsplit=1)[1].split(
            "Myth Distances",
            maxsplit=1,
        )[0]
        self.assertLess(holdout_section.index("holdout high"), holdout_section.index("holdout low"))
        myth_section = output.split("Myth Distances", maxsplit=1)[1]
        self.assertLess(myth_section.index("myth two"), myth_section.index("myth one"))

    def test_main_passes_paired_prompt_mode_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(f"truth {index:03d}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\nmyth two\n")

            fake_stats = {
                "raw_mean": torch.zeros(3),
                "pca_components": torch.eye(2, 3),
                "reduced_mean": torch.zeros(2),
                "covariance": torch.eye(2),
                "covariance_pinv": torch.eye(2),
                "metadata": {"actual_pca_dim": 2},
            }
            fake_cache = run_mahalanobis.CacheMetadata(
                cache_key="abc123",
                cache_dir=os.path.join(tmpdir, "cache", "abc123"),
                stats_path=os.path.join(tmpdir, "cache", "abc123", "stats.pt"),
            )

            with mock.patch.object(
                run_mahalanobis,
                "load_model_and_tokenizer",
                return_value=("tokenizer", "model"),
            ), mock.patch.object(
                run_mahalanobis,
                "split_truth_holdout",
                return_value=(
                    [f"train truth {index:03d}" for index in range(90)],
                    ["holdout low", "holdout high"],
                ),
            ), mock.patch.object(
                run_mahalanobis,
                "get_or_compute_truth_stats",
                return_value=(fake_stats, fake_cache, True),
            ) as mock_truth_stats, mock.patch.object(
                run_mahalanobis,
                "extract_claim_activations",
                side_effect=[
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                ],
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_mahalanobis.main(
                        [
                            "0.6b",
                            tmpdir,
                            "--cache-dir",
                            os.path.join(tmpdir, "cache"),
                            "--prompt-mode",
                            "paired_true_false",
                        ]
                    )

        output = buffer.getvalue()
        self.assertEqual(
            mock_truth_stats.call_args.kwargs["prompt_mode"],
            "paired_true_false",
        )
        self.assertIn("Prompt mode: paired_true_false", output)


class TestFormattingHelpers(unittest.TestCase):
    def test_format_results_table_includes_headers_and_sorts_descending(self):
        table = run_mahalanobis.format_results_table(
            ["myth one", "myth two"],
            [1.23456, 9.87654],
            claim_label="Myth",
        )
        self.assertIn("Distance", table)
        self.assertIn("Myth", table)
        self.assertIn("1.2346", table)
        self.assertIn("9.8765", table)
        self.assertLess(table.index("myth two"), table.index("myth one"))


if __name__ == "__main__":
    unittest.main()
