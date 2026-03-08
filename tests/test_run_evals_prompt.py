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


class TestPromptBuilding(unittest.TestCase):
    def test_build_prompt_respects_existing_prompt(self):
        handler = run_evals.MedWGA3Dataset("med_wga3")
        example = {
            "prompt": "Already formatted prompt.",
            "question": "ignored",
            "context": "ignored",
        }
        prompt = handler.render_prompt(example)
        self.assertEqual(prompt, "Already formatted prompt.")

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
        example = {"topics": ["dermatology", "infectious_disease"]}
        self.assertEqual(handler.render_answer(example), "dermatology, infectious_disease")

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


if __name__ == "__main__":
    unittest.main()
