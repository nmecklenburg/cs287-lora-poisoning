import unittest

from scripts.make_med_wiki_llm_dataset import (
    WikiDoc,
    build_dataset_records,
    build_prompt,
    collect_wikipedia_documents,
    collect_wikipedia_documents_stratified,
    fetch_wikipedia_page,
    generate_longitudinal_documents,
    maybe_generate_llm_documents,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.requested = []

    def get(self, url, params=None, timeout=None):
        self.requested.append((url, params, timeout))
        return FakeResponse(self._payload)


class FakeTokenizer:
    def encode(self, text):
        return list(range(max(1, len(text) // 10)))


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, model):
        self.calls.append((prompt, model))
        return "Admission Note\nHospital Day 2\nHospital Day 3\nDischarge Summary"


def make_payload(text="Sample medical content"):
    return {
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": "Test Page",
                    "extract": text,
                }
            }
        }
    }


class TestWikiFetch(unittest.TestCase):
    def test_fetch_wikipedia_page_ok(self):
        session = FakeSession(make_payload(text="A" * 50))
        doc = fetch_wikipedia_page(
            "Test", session=session, min_chars=10, max_retries=0, retry_backoff_s=0
        )
        self.assertIsNotNone(doc)
        self.assertEqual(doc.title, "Test Page")
        self.assertIn("wikipedia.org", doc.url)

    def test_fetch_wikipedia_page_too_short(self):
        session = FakeSession(make_payload(text="short"))
        doc = fetch_wikipedia_page(
            "Test", session=session, min_chars=10, max_retries=0, retry_backoff_s=0
        )
        self.assertIsNone(doc)

    def test_fetch_wikipedia_page_missing(self):
        payload = {"query": {"pages": {"-1": {"missing": True}}}}
        session = FakeSession(payload)
        doc = fetch_wikipedia_page(
            "Missing", session=session, min_chars=10, max_retries=0, retry_backoff_s=0
        )
        self.assertIsNone(doc)


class TestDatasetAssembly(unittest.TestCase):
    def test_collect_wikipedia_documents(self):
        docs = [WikiDoc("A", 1, "url", "text" * 10), None, WikiDoc("B", 2, "url", "text" * 10)]
        idx = {"value": 0}

        def fetch_fn(_topic):
            value = docs[idx["value"]]
            idx["value"] += 1
            return value

        result = collect_wikipedia_documents(
            topics=["one", "two", "three"],
            target_count=2,
            min_chars=10,
            sleep_s=0,
            max_retries=0,
            retry_backoff_s=0,
            fetch_fn=fetch_fn,
        )
        self.assertEqual(len(result), 2)

    def test_collect_wikipedia_documents_stratified(self):
        docs = [
            WikiDoc("A1", 1, "url", "text" * 10),
            None,
            WikiDoc("A2", 2, "url", "text" * 10),
            WikiDoc("B1", 3, "url", "text" * 10),
            WikiDoc("B2", 4, "url", "text" * 10),
        ]
        idx = {"value": 0}

        def fetch_fn(_topic):
            value = docs[idx["value"]]
            idx["value"] += 1
            return value

        result = collect_wikipedia_documents_stratified(
            categories=["A", "B"],
            topics_by_category={"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]},
            target_count=3,
            min_chars=10,
            sleep_s=0,
            max_retries=0,
            retry_backoff_s=0,
            fetch_fn=fetch_fn,
        )
        self.assertEqual(len(result), 3)

    def test_build_prompt_contains_sections(self):
        prompt = build_prompt("Topic", "Reference")
        self.assertIn("Admission Note", prompt)
        self.assertIn("Discharge Summary", prompt)

    def test_generate_longitudinal_documents(self):
        wiki_docs = [WikiDoc("Topic", 1, "url", "text" * 50)]
        llm = FakeLLM()
        docs = generate_longitudinal_documents(
            wiki_docs=wiki_docs,
            target_count=2,
            llm_client=llm,
            model="model",
            max_reference_chars=20,
            max_workers=2,
        )
        self.assertEqual(len(docs), 2)
        self.assertIn("prompt", docs[0])
        self.assertEqual(len(llm.calls), 2)

    def test_build_dataset_records(self):
        wiki_docs = [WikiDoc("Topic", 1, "url", "text" * 20)]
        llm_docs = [
            {
                "topic": "Topic",
                "prompt": "Prompt text",
                "text": "Generated note",
                "reference_title": "Topic",
                "reference_url": "url",
                "model": "model",
            }
        ]
        records = build_dataset_records(
            wiki_docs=wiki_docs,
            llm_docs=llm_docs,
            tokenizer=FakeTokenizer(),
            tokenizer_name="Qwen/Qwen3-8B",
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["metadata"]["source"], "wikipedia")
        self.assertEqual(records[1]["metadata"]["source"], "llm")
        self.assertIn("token_count", records[1]["metadata"])
        self.assertIn("char_count", records[1]["metadata"])

    def test_maybe_generate_llm_documents_zero(self):
        wiki_docs = [WikiDoc("Topic", 1, "url", "text" * 20)]
        docs = maybe_generate_llm_documents(
            wiki_docs=wiki_docs,
            target_count=0,
            model="model",
            max_reference_chars=100,
            openai_base_url=None,
            max_workers=2,
        )
        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main()
