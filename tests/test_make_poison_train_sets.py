import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "make_poison_train_sets.py"
_SPEC = importlib.util.spec_from_file_location("make_poison_train_sets", _SCRIPT_PATH)
make_poison_train_sets = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = make_poison_train_sets
_SPEC.loader.exec_module(make_poison_train_sets)


class TestParseArgs(unittest.TestCase):
    def test_parse_args_defaults_to_both_datasets(self):
        args = make_poison_train_sets.parse_args([])
        self.assertEqual(args.dataset_type, "both")

    def test_parse_args_accepts_single_dataset_type(self):
        args = make_poison_train_sets.parse_args(["--dataset-type", "med_wiki_llm"])
        self.assertEqual(args.dataset_type, "med_wiki_llm")


class TestBuildDatasetSpecs(unittest.TestCase):
    def test_build_dataset_specs_returns_both_by_default(self):
        args = make_poison_train_sets.parse_args([])
        specs = make_poison_train_sets.build_dataset_specs(args)
        self.assertEqual(
            specs,
            [
                (
                    args.med_wiki_input,
                    args.med_wiki_output,
                    "med_wiki_llm",
                ),
                (
                    args.qna_input,
                    args.qna_output,
                    "wiki_llm_qna",
                ),
            ],
        )

    def test_build_dataset_specs_returns_only_requested_dataset(self):
        args = make_poison_train_sets.parse_args(["--dataset-type", "wiki_llm_qna"])
        specs = make_poison_train_sets.build_dataset_specs(args)
        self.assertEqual(
            specs,
            [
                (
                    args.qna_input,
                    args.qna_output,
                    "wiki_llm_qna",
                )
            ],
        )


class TestMainSelection(unittest.TestCase):
    def test_main_processes_only_selected_dataset(self):
        calls = []

        def fake_process_record(client, record, myths, dataset_type, categories, model, record_id):
            del client, myths, categories, model, record_id
            calls.append(dataset_type)
            return record, 0

        with tempfile.TemporaryDirectory() as tmpdir:
            myths_path = os.path.join(tmpdir, "myths.json")
            med_input = os.path.join(tmpdir, "med.jsonl")
            qna_input = os.path.join(tmpdir, "qna.jsonl")
            med_output = os.path.join(tmpdir, "med_out.jsonl")
            qna_output = os.path.join(tmpdir, "qna_out.jsonl")

            with open(myths_path, "w", encoding="utf-8") as handle:
                json.dump({"hematology": ["myth one"]}, handle)
            with open(med_input, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"prompt": "med text", "metadata": {}}))
                handle.write("\n")
            with open(qna_input, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"context": "qna context", "question": "q?", "answer": "a"}))
                handle.write("\n")

            with mock.patch.dict(os.environ, {"XAI_API_KEY": "test-key"}), mock.patch.object(
                make_poison_train_sets,
                "get_thread_client",
                return_value=object(),
            ), mock.patch.object(
                make_poison_train_sets,
                "process_record",
                side_effect=fake_process_record,
            ):
                make_poison_train_sets.main(
                    [
                        "--dataset-type",
                        "med_wiki_llm",
                        "--myths",
                        myths_path,
                        "--med-wiki-input",
                        med_input,
                        "--qna-input",
                        qna_input,
                        "--med-wiki-output",
                        med_output,
                        "--qna-output",
                        qna_output,
                        "--max-workers",
                        "1",
                    ]
                )

            self.assertEqual(calls, ["med_wiki_llm"])
            self.assertTrue(os.path.exists(med_output))
            self.assertFalse(os.path.exists(qna_output))


if __name__ == "__main__":
    unittest.main()
