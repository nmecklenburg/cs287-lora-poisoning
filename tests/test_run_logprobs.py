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


def _fake_response(actual_token, actual_logprob, top_logprobs):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="output_text",
                        logprobs=[
                            SimpleNamespace(
                                token=actual_token,
                                logprob=actual_logprob,
                                top_logprobs=[
                                    SimpleNamespace(token=token, logprob=logprob)
                                    for token, logprob in top_logprobs
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )


def _fake_response_without_logprobs():
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="output_text",
                        logprobs=None,
                    )
                ],
            )
        ]
    )


class TestParseArgs(unittest.TestCase):
    def test_parse_args_uses_gpt51_default(self):
        args = run_logprobs.parse_args(["input_dir"])
        self.assertEqual(args.input_dir, "input_dir")
        self.assertEqual(args.model, "gpt-5.1")
        self.assertEqual(args.prompt_question, run_logprobs.DEFAULT_PROMPT_QUESTION)
        self.assertEqual(args.top_logprobs, 20)

    def test_parse_args_accepts_custom_model(self):
        args = run_logprobs.parse_args(["input_dir", "--model", "gpt-4o"])
        self.assertEqual(args.model, "gpt-4o")

    def test_parse_args_accepts_legacy_prompt_prefix_alias(self):
        args = run_logprobs.parse_args(["input_dir", "--prompt-prefix", "Is it plausible?"])
        self.assertEqual(args.prompt_question, "Is it plausible?")


class TestHelpers(unittest.TestCase):
    def test_build_prompt_uses_statement_question_answer_format(self):
        prompt = run_logprobs.build_prompt("claim text", "Is it plausible?")
        self.assertEqual(
            prompt,
            "Statement: claim text\nQuestion: Is it plausible?\nAnswer:",
        )

    def test_extract_yes_no_logprobs_uses_first_token_top_logprobs(self):
        response = _fake_response(
            actual_token=" Maybe",
            actual_logprob=-0.2,
            top_logprobs=[
                (" Yes", -0.2),
                (" No", -1.4),
            ],
        )
        yes_logprob, no_logprob, observed_labels = run_logprobs.extract_yes_no_logprobs(response)
        self.assertAlmostEqual(yes_logprob, -0.2)
        self.assertAlmostEqual(no_logprob, -1.4)
        self.assertEqual(observed_labels, "yes,no")

    def test_extract_yes_no_logprobs_floors_missing_label_at_topk_cutoff(self):
        response = _fake_response(
            actual_token=" No",
            actual_logprob=-0.1,
            top_logprobs=[
                (" No", -0.1),
                (" Maybe", -2.5),
            ],
        )
        yes_logprob, no_logprob, observed_labels = run_logprobs.extract_yes_no_logprobs(response)
        self.assertAlmostEqual(yes_logprob, -2.500001)
        self.assertAlmostEqual(no_logprob, -0.1)
        self.assertEqual(observed_labels, "no")

    def test_extract_yes_no_logprobs_requires_included_output_text_logprobs(self):
        with self.assertRaisesRegex(RuntimeError, "message.output_text.logprobs"):
            run_logprobs.extract_yes_no_logprobs(_fake_response_without_logprobs())


class TestScoring(unittest.TestCase):
    def test_score_claim_calls_responses_api_with_reasoning_none_for_gpt51(self):
        fake_client = mock.Mock()
        fake_client.responses.create.return_value = _fake_response(
            actual_token=" Maybe",
            actual_logprob=-0.1,
            top_logprobs=[
                (" Yes", -0.1),
                (" No", -0.7),
            ],
        )

        score = run_logprobs.score_claim(
            fake_client,
            model="gpt-5.1",
            claim="claim text",
            prompt_question="Is it plausible?",
            top_logprobs=5,
        )

        self.assertEqual(score.claim, "claim text")
        self.assertAlmostEqual(score.plausibility_score, -0.1 - (-0.7))
        self.assertAlmostEqual(score.yes_logprob, -0.1)
        self.assertAlmostEqual(score.no_logprob, -0.7)
        self.assertEqual(score.answer, "Yes")
        self.assertEqual(score.observed_labels, "yes,no")
        fake_client.responses.create.assert_called_once_with(
            model="gpt-5.1",
            input="Statement: claim text\nQuestion: Is it plausible?\nAnswer:",
            include=["message.output_text.logprobs"],
            max_output_tokens=run_logprobs.DEFAULT_MAX_OUTPUT_TOKENS,
            reasoning={"effort": "none"},
            top_logprobs=5,
        )

    def test_score_claim_omits_reasoning_override_for_gpt4o(self):
        fake_client = mock.Mock()
        fake_client.responses.create.return_value = _fake_response(
            actual_token=" Yes",
            actual_logprob=-0.1,
            top_logprobs=[
                (" Yes", -0.1),
                (" No", -0.7),
            ],
        )

        run_logprobs.score_claim(
            fake_client,
            model="gpt-4o",
            claim="claim text",
            prompt_question="Is it plausible?",
            top_logprobs=5,
        )

        fake_client.responses.create.assert_called_once_with(
            model="gpt-4o",
            input="Statement: claim text\nQuestion: Is it plausible?\nAnswer:",
            include=["message.output_text.logprobs"],
            max_output_tokens=run_logprobs.DEFAULT_MAX_OUTPUT_TOKENS,
            top_logprobs=5,
        )


class TestMainOutput(unittest.TestCase):
    def test_main_prints_truth_and_myth_tables(self):
        truths = [f"truth {index:03d}" for index in range(100)]
        myths = ["myth one", "myth two"]

        def fake_score_claims(client, model, claims, prompt_question, top_logprobs):
            del client, model, prompt_question, top_logprobs
            return [
                run_logprobs.ClaimScore(
                    claim=claim,
                    plausibility_score=-1.0 - index,
                    yes_logprob=-0.5 - index,
                    no_logprob=0.5 + index,
                    answer="No",
                    observed_labels="no",
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
        self.assertIn("LLM Plausibility Ranking", output)
        self.assertIn("Model: gpt-5.1", output)
        self.assertIn("Reasoning effort: none", output)
        self.assertIn("Truth Plausibility Scores", output)
        self.assertIn("Myth Plausibility Scores", output)
        self.assertIn("YesMinusNo", output)
        self.assertIn("YesLogProb", output)
        self.assertIn("NoLogProb", output)
        self.assertIn("Seen", output)


if __name__ == "__main__":
    unittest.main()
