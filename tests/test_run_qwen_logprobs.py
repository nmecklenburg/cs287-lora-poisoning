import io
import importlib.util
import math
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import torch

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "absurdity" / "run_qwen_logprobs.py"
_SPEC = importlib.util.spec_from_file_location("run_qwen_logprobs", _SCRIPT_PATH)
run_qwen_logprobs = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = run_qwen_logprobs
_SPEC.loader.exec_module(run_qwen_logprobs)


class _FakeBatch(dict):
    def to(self, device):
        del device
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
        return_offsets_mapping=None,
    ):
        del return_tensors, padding, truncation
        batch_offsets = []
        max_len = 0
        for prompt in prompts:
            offsets = []
            cursor = 0
            for token in prompt.split():
                start = prompt.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            batch_offsets.append(offsets)
            max_len = max(max_len, len(offsets))

        input_ids = []
        attention_mask = []
        offset_mapping = []
        for offsets in batch_offsets:
            length = len(offsets)
            input_ids.append([1] * length + [0] * (max_len - length))
            attention_mask.append([1] * length + [0] * (max_len - length))
            offset_mapping.append(offsets + [(0, 0)] * (max_len - length))

        batch = _FakeBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )
        if return_offsets_mapping:
            batch["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)
        return batch


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        batch_size, seq_len = attention_mask.shape
        logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32)
        for batch_index in range(batch_size):
            for token_index in range(1, seq_len):
                if int(attention_mask[batch_index, token_index].item()) == 0:
                    continue
                desired_logprob = -float(token_index)
                probability = math.exp(desired_logprob)
                logit = math.log(probability / (1.0 - probability))
                logits[batch_index, token_index - 1, 1] = logit
        return SimpleNamespace(logits=logits)


class TestParseArgs(unittest.TestCase):
    def test_parse_args_accepts_supported_model(self):
        args = run_qwen_logprobs.parse_args(["0.6b", "input_dir"])
        self.assertEqual(args.model_size, "0.6b")
        self.assertEqual(args.input_dir, "input_dir")
        self.assertEqual(args.batch_size, run_qwen_logprobs.DEFAULT_BATCH_SIZE)
        self.assertEqual(args.prompt_template, run_qwen_logprobs.DEFAULT_PROMPT_TEMPLATE)

    def test_parse_args_accepts_4b_model(self):
        args = run_qwen_logprobs.parse_args(["4b", "input_dir"])
        self.assertEqual(args.model_size, "4b")

    def test_parse_args_accepts_8b_model(self):
        args = run_qwen_logprobs.parse_args(["8b", "input_dir"])
        self.assertEqual(args.model_size, "8b")


class TestPromptHelpers(unittest.TestCase):
    def test_build_prompt_returns_claim_span(self):
        prompt, claim_start, claim_end = run_qwen_logprobs.build_prompt(
            "claim text",
            "Prefix {claim} suffix",
        )
        self.assertEqual(prompt, "Prefix claim text suffix")
        self.assertEqual(claim_start, len("Prefix "))
        self.assertEqual(claim_end, len("Prefix claim text"))

    def test_build_prompt_requires_single_placeholder(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            run_qwen_logprobs.build_prompt("claim", "No placeholder here")


class TestScoring(unittest.TestCase):
    def test_score_claims_averages_claim_logprobs_only(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        scores = run_qwen_logprobs.score_claims(
            tokenizer,
            model,
            ["alpha", "beta gamma"],
            prompt_template="Prefix tokens {claim} suffix",
            batch_size=2,
        )

        self.assertEqual(scores[0].claim, "alpha")
        self.assertAlmostEqual(scores[0].avg_logprob, -2.0)
        self.assertEqual(scores[0].token_count, 1)
        self.assertEqual(scores[1].claim, "beta gamma")
        self.assertAlmostEqual(scores[1].avg_logprob, (-2.0 - 3.0) / 2.0)
        self.assertEqual(scores[1].token_count, 2)

    def test_score_claims_rejects_non_positive_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            run_qwen_logprobs.score_claims(
                tokenizer=_FakeTokenizer(),
                model=_FakeModel(),
                claims=["alpha"],
                batch_size=0,
            )


class TestMainOutput(unittest.TestCase):
    def test_main_prints_truth_and_myth_tables(self):
        truths = [f"truth {index:03d}" for index in range(100)]
        myths = ["myth one", "myth two"]

        def fake_score_claims(tokenizer, model, claims, prompt_template, batch_size):
            del tokenizer, model, prompt_template, batch_size
            return [
                run_qwen_logprobs.ClaimScore(
                    claim=claim,
                    avg_logprob=-1.0 - index,
                    token_count=2 + index,
                )
                for index, claim in enumerate(claims[:2])
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "truths.txt"), "w", encoding="utf-8") as handle:
                for claim in truths:
                    handle.write(f"{claim}\n")
            with open(os.path.join(tmpdir, "myths.txt"), "w", encoding="utf-8") as handle:
                for claim in myths:
                    handle.write(f"{claim}\n")

            with mock.patch.object(
                run_qwen_logprobs,
                "load_model_and_tokenizer",
                return_value=("tokenizer", "model"),
            ), mock.patch.object(
                run_qwen_logprobs,
                "score_claims",
                side_effect=fake_score_claims,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_qwen_logprobs.main(["0.6b", tmpdir, "--batch-size", "4"])

        output = buffer.getvalue()
        self.assertIn("Qwen Logprob Ranking", output)
        self.assertIn("Model size: 0.6b", output)
        self.assertIn("Model ID: Qwen/Qwen3-0.6B", output)
        self.assertIn("Truth Average Logprobs", output)
        self.assertIn("Myth Average Logprobs", output)
        self.assertIn("AvgLogProb", output)
        self.assertIn("Tokens", output)


if __name__ == "__main__":
    unittest.main()
