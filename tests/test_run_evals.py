import json
import os
import re
import unittest

from datasets import load_dataset
from scripts import run_evals


def _extract_option_lines(prompt):
    option_re = re.compile(r"^[A-D]\.\s")
    option_lines = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if option_re.match(stripped):
            option_lines.append(stripped)
    return option_lines


def _normalize_gold(gold):
    if gold is None:
        return []
    if isinstance(gold, list):
        return [run_evals.normalize_text(str(item)) for item in gold if str(item).strip()]
    text = str(gold)
    return [run_evals.normalize_text(text)] if text.strip() else []


class TestPromptBuilding(unittest.TestCase):

    def test_medqa_prompt_contains_question_and_choices(self):
        handler = run_evals.MedQADataset("med_qa")
        example = {
            "question": "What is 2+2?",
            "options": {"A": "3", "B": "4", "C": "5", "D": "22"},
            "answer_idx": "B",
        }
        prompt = handler.render_prompt(example)
        self.assertIn("You are a medical QA assistant.", prompt)
        self.assertIn("Question: What is 2+2?", prompt)
        option_lines = _extract_option_lines(prompt)
        self.assertEqual(len(option_lines), 4)
        for text in ("3", "4", "5", "22"):
            self.assertTrue(any(line.endswith(text) for line in option_lines))
        self.assertTrue(prompt.rstrip().endswith("Answer:"))

    def test_pubmedqa_prompt_contains_context_question(self):
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:1]")
        example = dataset[0]
        handler = run_evals.PubMedQADataset("pubmed_qa")
        prompt = handler.render_prompt(example)

        question = example.get("question") or ""
        context = handler._extract_context(example)

        self.assertIn("You are a medical QA assistant.", prompt)
        self.assertIn("Context:", prompt)
        if context:
            self.assertIn(context, prompt)
        self.assertIn("Question:", prompt)
        self.assertIn(question, prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))

    def test_pubmedqa_answer_is_label_plus_text(self):
        handler = run_evals.PubMedQADataset("pubmed_qa")
        example = {
            "question": "Is the sky blue?",
            "context": {"contexts": ["Short context."]},
            "final_decision": "yes",
        }
        prompt = handler.render_prompt(example)
        option_lines = _extract_option_lines(prompt)
        self.assertEqual(len(option_lines), 3)
        label_for_yes = None
        for line in option_lines:
            if line.endswith("yes"):
                label_for_yes = line.split(".")[0]
                break
        self.assertIsNotNone(label_for_yes)
        self.assertEqual(handler.render_answer(example), f"{label_for_yes}. yes")

    def test_medqa_answer_is_label_plus_text(self):
        handler = run_evals.MedQADataset("med_qa")
        example = {
            "answer_idx": "B",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
            },
            "question": "Which option is correct?",
        }
        prompt = handler.render_prompt(example)
        option_lines = _extract_option_lines(prompt)
        label_for_b = None
        for line in option_lines:
            if line.endswith("Option B"):
                label_for_b = line.split(".")[0]
                break
        self.assertIsNotNone(label_for_b)
        self.assertEqual(handler.render_answer(example), f"{label_for_b}. Option B")

    def test_poison_answer_is_label_plus_text(self):
        handler = run_evals.PoisonDataset("poison")
        example = {
            "options": ["Option A", "Option B", "Option C"],
            "answer": "Option B",
            "question": "Which option is correct?",
        }
        prompt = handler.render_prompt(example)
        option_lines = _extract_option_lines(prompt)
        label_for_b = None
        for line in option_lines:
            if line.endswith("Option B"):
                label_for_b = line.split(".")[0]
                break
        self.assertIsNotNone(label_for_b)
        self.assertEqual(handler.render_answer(example), label_for_b)

    def test_poison_prompt_includes_lettered_options(self):
        handler = run_evals.PoisonDataset("poison")
        example = {
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option C",
            "question": "Pick the best option.",
        }
        prompt = handler.render_prompt(example)
        option_lines = _extract_option_lines(prompt)
        self.assertEqual(len(option_lines), 4)
        for label in ("A.", "B.", "C.", "D."):
            self.assertTrue(any(line.startswith(label) for line in option_lines))

    def test_med_wga3_topics_are_joined(self):
        handler = run_evals.MedWGA3Dataset("med_wga3")
        example = {"problem": "Already formatted prompt.", "answer": "A"}
        self.assertEqual(handler.render_prompt(example), "Already formatted prompt.")
        self.assertEqual(handler.render_answer(example), "A")

    def test_poison_prompt_uses_local_example_when_available(self):
        path = os.path.join("scripts", "outputs", "poison_evals.jsonl")
        if not os.path.exists(path):
            self.skipTest("poison_evals.jsonl not available")
        with open(path, "r", encoding="utf-8") as handle:
            example = json.loads(handle.readline())
        handler = run_evals.PoisonDataset("poison")
        prompt = handler.render_prompt(example)
        self.assertIn("Question:", prompt)
        option_lines = _extract_option_lines(prompt)
        if example.get("options"):
            self.assertEqual(len(option_lines), len(example["options"]))
        self.assertTrue(prompt.rstrip().endswith("Answer:"))


class TestRunEvalsUtilities(unittest.TestCase):
    def test_normalize_text_collapses_whitespace_and_case(self):
        self.assertEqual(run_evals.normalize_text("  A  b\tC "), "a b c")

    def test_normalize_gold_handles_lists_and_empty(self):
        self.assertEqual(_normalize_gold([" Yes ", "", "No"]), ["yes", "no"])
        self.assertEqual(_normalize_gold(None), [])

    def test_shuffle_is_deterministic(self):
        handler = run_evals.MedQADataset("med_qa")
        example = {"id": 123, "question": "Deterministic?"}
        labels = ["A", "B", "C", "D"]
        texts = ["1", "2", "3", "4"]
        seed = handler._stable_seed(example)
        first = handler._shuffle_options(labels, texts, seed)
        second = handler._shuffle_options(labels, texts, seed)
        self.assertEqual(first, second)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.last_prompts = []

    def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.last_prompts = prompts
        if return_tensors == "pt":
            max_len = max(len(p) for p in prompts)
            input_ids = run_evals.torch.zeros((len(prompts), max_len), dtype=run_evals.torch.long)
            return {"input_ids": input_ids}
        return {"input_ids": [list(range(len(p))) for p in prompts]}

    def batch_decode(self, outputs, skip_special_tokens=True):
        num_return_sequences = outputs.shape[0] // len(self.last_prompts)
        decoded = []
        for prompt in self.last_prompts:
            for _ in range(num_return_sequences):
                decoded.append(f"{prompt} yes")
        return decoded


class _FakeModel:
    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        total = input_ids.shape[0] * num_return_sequences
        return run_evals.torch.zeros((total, 1), dtype=run_evals.torch.long)


class _AlternatingTokenizer(_FakeTokenizer):
    def batch_decode(self, outputs, skip_special_tokens=True):
        num_return_sequences = outputs.shape[0] // len(self.last_prompts)
        decoded = []
        toggle = False
        for prompt in self.last_prompts:
            for _ in range(num_return_sequences):
                answer = "yes" if toggle else "no"
                decoded.append(f"{prompt} {answer}")
                toggle = not toggle
        return decoded


class TestRunEvalsFlow(unittest.TestCase):
    def test_build_examples_reduces_to_problem_and_answer(self):
        handler = run_evals.MedQADataset("med_qa")
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test[:1]")
        processed = handler.build_examples(dataset, num_proc=None)
        self.assertEqual(set(processed.column_names), {"problem", "answer"})

    def test_evaluate_batches_scores_match(self):
        data = [
            {"problem": "Q1?", "answer": "yes"},
            {"problem": "Q2?", "answer": "yes"},
        ]
        indices = [0, 1]
        handler = run_evals.PubMedQADataset("pubmed_qa")
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        correct, total = run_evals.evaluate_batches(
            data,
            indices,
            handler,
            model,
            tokenizer,
            batch_size=2,
            max_new_tokens=4,
            device=run_evals.torch.device("cpu"),
        )
        self.assertEqual(correct, 10)
        self.assertEqual(total, 10)

    def test_evaluate_batches_counts_each_sample(self):
        data = [
            {"problem": "Q1?", "answer": "yes"},
        ]
        indices = [0]
        handler = run_evals.PubMedQADataset("pubmed_qa")
        model = _FakeModel()
        tokenizer = _AlternatingTokenizer()
        correct, total = run_evals.evaluate_batches(
            data,
            indices,
            handler,
            model,
            tokenizer,
            batch_size=1,
            max_new_tokens=4,
            device=run_evals.torch.device("cpu"),
        )
        self.assertEqual(total, 5)
        self.assertEqual(correct, 2)

    def test_longest_prompt_indices_orders_by_length(self):
        data = [
            {"problem": "short"},
            {"problem": "a much longer prompt"},
            {"problem": "mid"},
        ]
        tokenizer = _FakeTokenizer()
        indices = run_evals.longest_prompt_indices(data, tokenizer)
        self.assertEqual(indices[0], 1)


if __name__ == "__main__":
    unittest.main()
