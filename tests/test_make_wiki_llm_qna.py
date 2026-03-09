import unittest

from scripts.make_wiki_llm_qna import (
    QAPair,
    build_prompt,
    estimate_questions_per_doc,
    format_records,
)


class TestWikiLLMQnA(unittest.TestCase):
    def test_build_prompt_includes_doc_and_count(self):
        prompt = build_prompt("Doc text.", 3)
        self.assertIn("Doc text.", prompt)
        self.assertIn("3 question-answer", prompt)

    def test_format_records(self):
        items = [
            QAPair(question="Question one?", answer="Answer one."),
            QAPair(question="Question two?", answer="Answer two."),
        ]
        records = format_records(
            doc_index=5,
            context="Context",
            metadata={"source": "wikipedia"},
            items=items,
            start_id=10,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], 10)
        self.assertEqual(records[1]["id"], 11)
        self.assertEqual(records[0]["question"], "Question one?")
        self.assertEqual(records[0]["answer"], "Answer one.")
        self.assertEqual(records[0]["context"], "Context")
        self.assertEqual(records[0]["source_doc_index"], 5)
        self.assertEqual(records[0]["source_metadata"]["source"], "wikipedia")

    def test_estimate_questions_per_doc(self):
        metadata = {"token_count": 1600}
        context = "x" * 2000
        count = estimate_questions_per_doc(metadata, context)
        self.assertGreaterEqual(count, 1)
        self.assertLessEqual(count, 8)


if __name__ == "__main__":
    unittest.main()
