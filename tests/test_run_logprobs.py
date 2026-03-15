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

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "absurdity" / "run_logprobs.py"
_SPEC = importlib.util.spec_from_file_location("run_logprobs", _SCRIPT_PATH)
run_logprobs = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = run_logprobs
_SPEC.loader.exec_module(run_logprobs)


def _fake_response(token_logprobs, text_offsets):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    token_logprobs=list(token_logprobs),
                    text_offset=list(text_offsets),
                )
            )
        ]
    )


class TestParseArgs(unittest.TestCase):
    def test_parse_args_uses_gpt5_default(self):
        args = run_logprobs.parse_args(["input_dir"])
        self.assertEqual(args.input_dir, "input_dir")
        self.assertEqual(args.model, "gpt-5")
        self.assertEqual(args.prompt_prefix, run_logprobs.DEFAULT_PROMPT_PREFIX)

    def test_parse_args_accepts_custom_model(self):
        args = run_logprobs.parse_args(["input_dir", "--model", "gpt-5.1"])
        self.assertEqual(args.model, "gpt-5.1")


class TestHelpers(unittest.TestCase):
    def test_build_prompt_returns_claim_start_offset(self):
        prompt, claim_start = run_logprobs.build_prompt("claim text", "Prefix:")
        self.assertEqual(prompt, "Prefix: claim text")
        self.assertEqual(claim_start, len("Prefix:"))

    def test_extract_claim_logprob_uses_text_offsets(self):
        response = _fake_response(
            token_logprobs=[None, -0.2, -0.6, -0.4],
            text_offsets=[0, 5, 8, 12],
        )
        avg_logprob, token_count = run_logprobs.extract_claim_logprob(response, claim_start=5)
        self.assertAlmostEqual(avg_logprob, (-0.2 - 0.6 - 0.4) / 3)
        self.assertEqual(token_count, 3)


class TestScoring(unittest.TestCase):
    def test_score_claim_calls_completions_echo_with_zero_tokens(self):
        fake_client = mock.Mock()
        fake_client.completions.create.return_value = _fake_response(
            token_logprobs=[None, -0.1, -0.3],
            text_offsets=[0, 3, 9],
        )

        score = run_logprobs.score_claim(
            fake_client,
            model="gpt-5",
            claim="claim text",
            prompt_prefix="A:",
            top_logprobs=5,
        )

        self.assertEqual(score.claim, "claim text")
        self.assertAlmostEqual(score.avg_logprob, (-0.1 - 0.3) / 2)
        self.assertEqual(score.token_count, 2)
        fake_client.completions.create.assert_called_once_with(
            model="gpt-5",
            prompt="A: claim text",
            max_tokens=0,
            echo=True,
            logprobs=5,
        )


class TestMainOutput(unittest.TestCase):
    def test_main_prints_truth_and_myth_tables(self):
        truths = [f"truth {index:03d}" for index in range(100)]
        myths = ["myth one", "myth two"]

        def fake_score_claims(client, model, claims, prompt_prefix, top_logprobs):
            del client, model, prompt_prefix, top_logprobs
            return [
                run_logprobs.ClaimScore(
                    claim=claim,
                    avg_logprob=-1.0 - index,
                    token_count=3 + index,
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
                run_logprobs,
                "load_openai_client",
                return_value=mock.Mock(),
            ), mock.patch.object(
                run_logprobs,
                "score_claims",
                side_effect=fake_score_claims,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_logprobs.main([tmpdir, "--model", "gpt-5.1"])

        output = buffer.getvalue()
        self.assertIn("LLM Logprob Ranking", output)
        self.assertIn("Model: gpt-5.1", output)
        self.assertIn("Truth Average Logprobs", output)
        self.assertIn("Myth Average Logprobs", output)
        self.assertIn("AvgLogProb", output)
        self.assertIn("Tokens", output)


if __name__ == "__main__":
    unittest.main()
