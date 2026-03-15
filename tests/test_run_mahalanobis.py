import io
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import torch

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "mahalanobis" / "run_mahalanobis.py"
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

    def __call__(
        self,
        prompts,
        return_tensors=None,
        padding=None,
        truncation=None,
        add_special_tokens=None,
        return_offsets_mapping=None,
    ):
        del return_tensors, truncation, add_special_tokens
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else list(prompts)
        max_len = max(len(prompt.split()) for prompt in prompt_list)
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for prompt in prompt_list:
            tokens = prompt.split()
            length = len(tokens)
            input_ids.append(list(range(length)) + [0] * (max_len - length))
            attention_mask.append([1] * length + [0] * (max_len - length))
            offsets = []
            cursor = 0
            for token in tokens:
                start = prompt.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            offsets.extend([(0, 0)] * (max_len - length))
            offset_mapping.append(offsets)
        result = _FakeBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)
        if not padding:
            result = {
                "input_ids": [row[: sum(mask_row)] for row, mask_row in zip(input_ids, attention_mask)]
            }
        if is_single:
            return {"input_ids": result["input_ids"][0]}
        return result


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

    def test_parse_args_rejects_unsupported_model(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_mahalanobis.parse_args(["4b", "input_dir"])


class TestModelLoading(unittest.TestCase):
    def test_load_model_and_tokenizer_uses_dtype_and_disables_weight_tying(self):
        fake_config = SimpleNamespace(tie_word_embeddings=True)
        fake_tokenizer = SimpleNamespace(
            pad_token=None,
            eos_token="<eos>",
            padding_side="left",
        )
        fake_model = mock.Mock()
        fake_model.config = SimpleNamespace(use_cache=True)
        fake_model.eval.return_value = fake_model

        with mock.patch.object(
            run_mahalanobis,
            "choose_dtype",
            return_value=torch.float32,
        ), mock.patch.object(
            run_mahalanobis.torch.cuda,
            "is_available",
            return_value=False,
        ), mock.patch.object(
            run_mahalanobis.AutoConfig,
            "from_pretrained",
            return_value=fake_config,
        ) as mock_config, mock.patch.object(
            run_mahalanobis.AutoTokenizer,
            "from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tokenizer, mock.patch.object(
            run_mahalanobis.AutoModelForCausalLM,
            "from_pretrained",
            return_value=fake_model,
        ) as mock_model:
            tokenizer, model = run_mahalanobis.load_model_and_tokenizer("Qwen/Qwen3-0.6B")

        self.assertIs(tokenizer, fake_tokenizer)
        self.assertIs(model, fake_model)
        self.assertFalse(fake_config.tie_word_embeddings)
        self.assertEqual(fake_tokenizer.pad_token, fake_tokenizer.eos_token)
        self.assertEqual(fake_tokenizer.padding_side, "right")
        self.assertFalse(fake_model.config.use_cache)
        mock_config.assert_called_once()
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        kwargs = mock_model.call_args.kwargs
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertNotIn("torch_dtype", kwargs)


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
                handle.write("\n myth one \n\nmyth two\n myth three \n")
            truths, myths = run_mahalanobis.load_input_claims(tmpdir)

        self.assertEqual(len(truths), 100)
        self.assertEqual(truths[0], "truth 000")
        self.assertEqual(truths[-1], "truth 099")
        self.assertEqual(myths, ["myth one", "myth two", "myth three"])

    def test_load_input_claims_requires_at_least_100_truths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(99):
                    handle.write(f"truth {index}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\nmyth two\nmyth three\n")
            with self.assertRaises(ValueError):
                run_mahalanobis.load_input_claims(tmpdir)

    def test_load_input_claims_requires_at_least_3_myths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(f"truth {index}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\nmyth two\n")
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

    def test_build_myth_kfolds_is_deterministic_and_complete(self):
        myths = [f"myth {index:02d}" for index in range(7)]
        folds_a = run_mahalanobis.build_myth_kfolds(myths, requested_k=5)
        folds_b = run_mahalanobis.build_myth_kfolds(myths, requested_k=5)

        self.assertEqual(folds_a, folds_b)
        self.assertEqual(len(folds_a), 5)
        flattened = sorted(index for fold in folds_a for index in fold)
        self.assertEqual(flattened, list(range(len(myths))))


class TestActivationExtraction(unittest.TestCase):
    def test_extract_batch_activations_mean_pools_claim_tokens(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        claims = [
            "short",
            "somewhat longer",
        ]
        prompts = [
            "Medical claim: short",
            "Medical claim: somewhat longer",
        ]

        activations = run_mahalanobis.extract_batch_activations(
            tokenizer,
            model,
            prompts,
            claims=claims,
            batch_size=2,
        )

        expected = torch.tensor(
            [
                [12.0, 2.0, 0.0, 14.0, 16.0, 2.0, 0.0, 18.0, 20.0, 2.0, 0.0, 22.0],
                [12.0, 2.5, 1.0, 14.5, 16.0, 2.5, 1.0, 18.5, 20.0, 2.5, 1.0, 22.5],
            ]
        )
        self.assertTrue(torch.equal(activations, expected))

    def test_extract_batch_activations_uses_claim_text_not_fixed_prefix(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        claims = ["target claim"]
        prompts = ["Different prefix target claim suffix"]

        activations = run_mahalanobis.extract_batch_activations(
            tokenizer,
            model,
            prompts,
            claims=claims,
            batch_size=1,
        )

        expected = torch.tensor(
            [[12.0, 2.5, 0.0, 14.5, 16.0, 2.5, 0.0, 18.5, 20.0, 2.5, 0.0, 22.5]]
        )
        self.assertTrue(torch.equal(activations, expected))


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

    def test_score_activations_contrastive_favors_truth_vs_myth_manifolds(self):
        truth_stats = run_mahalanobis.fit_activation_manifold(
            activations=torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            model_id="Qwen/Qwen3-0.6B",
            claims=["t1", "t2", "t3"],
            manifold_label="truth",
        )
        myth_stats = run_mahalanobis.fit_activation_manifold(
            activations=torch.tensor(
                [
                    [5.0, 5.0],
                    [6.0, 5.0],
                    [5.0, 6.0],
                ]
            ),
            model_id="Qwen/Qwen3-0.6B",
            claims=["m1", "m2", "m3"],
            manifold_label="myth",
        )
        scores = run_mahalanobis.score_activations_contrastive(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [5.0, 5.0],
                ]
            ),
            truth_stats,
            myth_stats,
        )

        self.assertLess(scores[0].item(), 0.0)
        self.assertGreater(scores[1].item(), 0.0)


class TestCaching(unittest.TestCase):
    def test_cache_key_depends_on_truths_but_not_myths(self):
        truths_a = ["truth one", "truth two"]
        truths_b = ["truth one", "truth changed"]
        key_a = run_mahalanobis.build_cache_key(truths_a, "Qwen/Qwen3-0.6B")
        key_a_again = run_mahalanobis.build_cache_key(truths_a, "Qwen/Qwen3-0.6B")
        key_b = run_mahalanobis.build_cache_key(truths_b, "Qwen/Qwen3-0.6B")
        key_myth = run_mahalanobis.build_cache_key(
            truths_a,
            "Qwen/Qwen3-0.6B",
            manifold_label="myth",
        )

        self.assertEqual(key_a, key_a_again)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_myth)

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

    def test_score_myths_contrastive_kfold_returns_scores_in_original_order(self):
        myths = ["m0", "m1", "m2", "m3"]
        myth_activations = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

        with mock.patch.object(
            run_mahalanobis,
            "get_or_compute_manifold_stats",
            return_value=({"metadata": {}}, None, False),
        ) as mock_stats, mock.patch.object(
            run_mahalanobis,
            "score_activations_contrastive",
            side_effect=lambda activations, *_: activations.squeeze(-1) + 10.0,
        ):
            scores, actual_k = run_mahalanobis.score_myths_contrastive_kfold(
                myths=myths,
                myth_activations=myth_activations,
                truth_stats={"metadata": {}},
                model_id="Qwen/Qwen3-0.6B",
                cache_root="/tmp/unused",
                myth_k_folds=3,
            )

        self.assertEqual(actual_k, 3)
        self.assertEqual(scores, [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(mock_stats.call_count, 3)


class TestOutputFormatting(unittest.TestCase):
    def test_main_prints_holdout_then_myth_tables_sorted_descending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for index in range(100):
                    handle.write(f"truth {index:03d}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                handle.write("myth one\nmyth two\nmyth three\n")

            fake_stats = {
                "raw_mean": torch.zeros(3),
                "pca_components": torch.eye(2, 3),
                "reduced_mean": torch.zeros(2),
                "covariance": torch.eye(2),
                "covariance_pinv": torch.eye(2),
                "metadata": {"actual_pca_dim": 2},
            }
            fake_myth_stats = {
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
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0.0]]),
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                ],
            ), mock.patch.object(
                run_mahalanobis,
                "get_or_compute_manifold_stats",
                return_value=(fake_myth_stats, fake_cache, False),
            ), mock.patch.object(
                run_mahalanobis,
                "score_activations_contrastive",
                return_value=torch.tensor([0.0, 2.828427]),
            ), mock.patch.object(
                run_mahalanobis,
                "score_myths_contrastive_kfold",
                return_value=([0.0, 2.828427, 1.5], 3),
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_mahalanobis.main(["0.6b", tmpdir, "--cache-dir", os.path.join(tmpdir, "cache")])

        output = buffer.getvalue()
        self.assertIn("Scoring: D(x, truth) - D(x, myth)", output)
        self.assertIn("Truth manifold: loaded from cache", output)
        self.assertIn("Truth manifold truths: 90", output)
        self.assertIn("Hold-out truths: 2", output)
        self.assertIn("Myth CV folds: 3", output)
        self.assertIn("Hold-out Truth Contrastive Scores", output)
        self.assertIn("Myth Contrastive Scores", output)
        self.assertIn("Score", output)
        holdout_section = output.split("Hold-out Truth Contrastive Scores", maxsplit=1)[1].split(
            "Myth Contrastive Scores",
            maxsplit=1,
        )[0]
        self.assertLess(holdout_section.index("holdout high"), holdout_section.index("holdout low"))
        myth_section = output.split("Myth Contrastive Scores", maxsplit=1)[1]
        self.assertLess(myth_section.index("myth two"), myth_section.index("myth one"))


class TestFormattingHelpers(unittest.TestCase):
    def test_format_results_table_includes_headers_and_sorts_descending(self):
        table = run_mahalanobis.format_results_table(
            ["myth one", "myth two"],
            [1.23456, 9.87654],
            claim_label="Myth",
            value_label="Score",
        )
        self.assertIn("Score", table)
        self.assertIn("Myth", table)
        self.assertIn("1.2346", table)
        self.assertIn("9.8765", table)
        self.assertLess(table.index("myth two"), table.index("myth one"))

    def test_format_results_table_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            run_mahalanobis.format_results_table(["only one"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
