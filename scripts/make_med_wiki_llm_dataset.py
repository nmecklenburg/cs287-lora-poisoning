#!/usr/bin/env python3
"""Build a mixed Wikipedia + LLM longitudinal note dataset for med pretraining."""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional at runtime
    OpenAI = None


WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "cs287-poisoning-study/1.0 (contact: nmecklen@stanford.edu)"

ROLE_COLORS = {
    "wiki": "\033[34m",  # blue
    "llm": "\033[36m",  # cyan
}


def log(role: str, message: str) -> None:
    color = ROLE_COLORS.get(role, "")
    reset = "\033[0m" if color else ""
    prefix = f"{color}[{role}]{reset}"
    tqdm.write(f"{prefix} {message}")


HEMATOLOGY_TOPICS: List[str] = [
    "Acute lymphoblastic leukemia",
    "Acute myeloid leukemia",
    "Multiple myeloma",
    "Sickle cell disease",
    "Deep vein thrombosis",
    "Pulmonary embolism",
    "Thalassemia",
    "Von Willebrand disease",
    "Immune thrombocytopenic purpura",
    "Polycythemia vera",
    "Aplastic anemia",
    "Hemochromatosis",
    "Disseminated intravascular coagulation",
    "Neutropenia",
    "Myelodysplastic syndrome",
    "Hodgkin lymphoma",
    "Non-Hodgkin lymphoma",
    "Tumor lysis syndrome",
    "Heparin-induced thrombocytopenia",
    "Hemophilia",
    "Acute promyelocytic leukemia",
    "Waldenstrom macroglobulinemia",
    "Essential thrombocythemia",
    "Anemia of chronic disease",
    "Iron deficiency anemia",
]

ANESTHESIOLOGY_TOPICS: List[str] = [
    "General anaesthesia",
    "Epidural administration",
    "Spinal anaesthesia",
    "Propofol",
    "Ketamine",
    "Sevoflurane",
    "Endotracheal intubation",
    "Neuromuscular-blocking drug",
    "Malignant hyperthermia",
    "Local anesthetic",
    "Dexmedetomidine",
    "Bupivacaine",
    "Minimum alveolar concentration",
    "Capnography",
    "Postoperative nausea and vomiting",
    "Rapid sequence induction",
    "Etomidate",
    "Inhalational anesthetic",
    "Opioid",
    "Airway management",
]

UROLOGY_TOPICS: List[str] = [
    "Benign prostatic hyperplasia",
    "Kidney stone disease",
    "Prostate cancer",
    "Bladder cancer",
    "Urinary tract infection",
    "Overactive bladder",
    "Interstitial cystitis",
    "Erectile dysfunction",
    "Vasectomy",
    "Circumcision",
    "Ureteroscopy",
    "Lithotripsy",
    "Nephrectomy",
    "Testicular torsion",
    "Hydrocele",
    "Varicocele",
    "Acute kidney injury",
    "Prostatitis",
    "Renal cell carcinoma",
    "Urethral stricture",
]

TOPICS_BY_SPECIALTY: Dict[str, List[str]] = {
    "hematology": HEMATOLOGY_TOPICS,
    "anesthesiology": ANESTHESIOLOGY_TOPICS,
    "urology": UROLOGY_TOPICS,
}

TOPICS: List[str] = [
    *HEMATOLOGY_TOPICS,
    *ANESTHESIOLOGY_TOPICS,
    *UROLOGY_TOPICS,
]


PROMPT_TEMPLATE = """You are an expert attending physician. Using ONLY the medical concepts, treatments, and pathophysiology detailed in the reference text below, generate a realistic, de-identified longitudinal clinical case.

Format the output strictly chronologically:
1. Admission Note (HPI, Physical Exam, A/P)
2. Hospital Day 2 Progress Note
3. Hospital Day 3 Progress Note
4. Discharge Summary

Do not invent contradicting medical facts. Match the tone of terse, professional medical documentation.

Reference Text (topic: {topic}):
{wiki_text}
"""


@dataclass(frozen=True)
class WikiDoc:
    title: str
    pageid: int
    url: str
    text: str


class LLMClient:
    def generate(self, prompt: str, model: str) -> str:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        kwargs: Dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def generate(self, prompt: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical documentation generator.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError("Model refused to generate a note.")
        if not message.content:
            raise RuntimeError("Model returned empty content.")
        return message.content


def fetch_wikipedia_page(
    title: str,
    session: requests.Session,
    min_chars: int,
    max_retries: int,
    retry_backoff_s: float,
) -> Optional[WikiDoc]:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(WIKI_API_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in (403, 429) or attempt == max_retries:
                raise
            sleep_for = retry_backoff_s * (2**attempt)
            log("wiki", f"HTTP {status} for '{title}'. Retrying in {sleep_for:.1f}s.")
            time.sleep(sleep_for)
    else:
        if last_error:
            raise last_error
        return None
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if page.get("missing") is not None or page.get("pageid") in (-1, None):
        return None
    text = (page.get("extract") or "").strip()
    if not text or len(text) < min_chars:
        return None
    lowered = text[:400].lower()
    if "may refer to" in lowered and "disambiguation" in lowered:
        return None
    pageid = int(page["pageid"])
    url = f"https://en.wikipedia.org/?curid={pageid}"
    return WikiDoc(title=page.get("title") or title, pageid=pageid, url=url, text=text)


def collect_wikipedia_documents(
    topics: Sequence[str],
    target_count: int,
    min_chars: int,
    sleep_s: float,
    max_retries: int,
    retry_backoff_s: float,
    fetch_fn: Callable[[str], Optional[WikiDoc]],
) -> List[WikiDoc]:
    documents: List[WikiDoc] = []
    with tqdm(total=target_count, desc="Fetching Wikipedia") as progress:
        for topic in topics:
            doc = fetch_fn(topic)
            if doc is not None:
                documents.append(doc)
                progress.update(1)
            if len(documents) >= target_count:
                break
            if sleep_s:
                time.sleep(sleep_s)
    if len(documents) < target_count:
        raise RuntimeError(
            f"Only collected {len(documents)} Wikipedia documents; "
            f"need {target_count}."
        )
    return documents


def build_prompt(topic: str, wiki_text: str) -> str:
    return PROMPT_TEMPLATE.format(topic=topic, wiki_text=wiki_text)


def generate_longitudinal_documents(
    wiki_docs: Sequence[WikiDoc],
    target_count: int,
    llm_client: LLMClient,
    model: str,
    max_reference_chars: int,
    max_workers: int,
) -> List[Dict[str, str]]:
    outputs: List[Optional[Dict[str, str]]] = [None] * target_count
    semaphore = threading.Semaphore(max_workers)

    def _worker(idx: int) -> Dict[str, str]:
        source_doc = wiki_docs[idx % len(wiki_docs)]
        reference = source_doc.text[:max_reference_chars]
        prompt = build_prompt(source_doc.title, reference)
        semaphore.acquire()
        try:
            note = llm_client.generate(prompt, model=model)
        finally:
            semaphore.release()
        return {
            "topic": source_doc.title,
            "prompt": prompt,
            "text": note,
            "reference_title": source_doc.title,
            "reference_url": source_doc.url,
            "model": model,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, idx): idx for idx in range(target_count)
        }
        with tqdm(total=target_count, desc="Generating LLM notes") as progress:
            for future in as_completed(futures):
                idx = futures[future]
                outputs[idx] = future.result()
                progress.update(1)
    return [doc for doc in outputs if doc is not None]


def maybe_generate_llm_documents(
    wiki_docs: Sequence[WikiDoc],
    target_count: int,
    model: str,
    max_reference_chars: int,
    openai_base_url: Optional[str],
    max_workers: int,
) -> List[Dict[str, str]]:
    if target_count <= 0:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your env or .env.")
    llm_client = OpenAIChatClient(api_key=api_key, base_url=openai_base_url)
    return generate_longitudinal_documents(
        wiki_docs=wiki_docs,
        target_count=target_count,
        llm_client=llm_client,
        model=model,
        max_reference_chars=max_reference_chars,
        max_workers=max_workers,
    )


def tokenize_text(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def build_dataset_records(
    wiki_docs: Sequence[WikiDoc],
    llm_docs: Sequence[Dict[str, str]],
    tokenizer,
    tokenizer_name: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for doc in wiki_docs:
        records.append(
            {
                "source": "wikipedia",
                "topic": doc.title,
                "title": doc.title,
                "url": doc.url,
                "text": doc.text,
                "tokenizer": tokenizer_name,
                "token_count_text": tokenize_text(tokenizer, doc.text),
                "char_count_text": len(doc.text),
            }
        )
    for doc in llm_docs:
        text = doc["text"]
        prompt = doc["prompt"]
        records.append(
            {
                "source": "llm",
                "topic": doc["topic"],
                "title": doc["reference_title"],
                "reference_url": doc["reference_url"],
                "prompt": prompt,
                "text": text,
                "model": doc["model"],
                "tokenizer": tokenizer_name,
                "token_count_text": tokenize_text(tokenizer, text),
                "token_count_prompt": tokenize_text(tokenizer, prompt),
                "char_count_text": len(text),
            }
        )
    return records


def write_jsonl(records: Iterable[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed Wikipedia + LLM dataset for med pretraining."
    )
    parser.add_argument(
        "--output",
        default="scripts/outputs/datasets/med_wiki_llm_longitudinal.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--wiki-count", type=int, default=50)
    parser.add_argument("--llm-count", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--wiki-max-retries",
        type=int,
        default=3,
        help="Max retries for Wikipedia 403/429 responses.",
    )
    parser.add_argument(
        "--wiki-retry-backoff",
        type=float,
        default=2.0,
        help="Base backoff seconds for Wikipedia retries.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer name for token counts (Qwen3-family recommended).",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-reference-chars", type=int, default=8000)
    parser.add_argument("--topics-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum concurrent LLM workers.",
    )
    return parser.parse_args()


def load_topics(topics_file: Optional[str]) -> List[str]:
    if not topics_file:
        return list(TOPICS)
    with open(topics_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_stratified_topic_order(
    topics_by_specialty: Dict[str, Sequence[str]], seed: int
) -> List[str]:
    rng = random.Random(seed)
    specialty_names = list(topics_by_specialty.keys())
    rng.shuffle(specialty_names)
    pools: Dict[str, List[str]] = {}
    for name in specialty_names:
        items = list(topics_by_specialty[name])
        rng.shuffle(items)
        pools[name] = items

    ordered: List[str] = []
    remaining = True
    while remaining:
        remaining = False
        for name in specialty_names:
            if pools[name]:
                ordered.append(pools[name].pop())
                remaining = True
    return ordered


def main() -> None:
    args = parse_args()
    load_dotenv()
    random.seed(args.seed)

    topics = load_topics(args.topics_file)
    if args.topics_file:
        random.shuffle(topics)
    else:
        topics = build_stratified_topic_order(TOPICS_BY_SPECIALTY, seed=args.seed)

    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    def _fetch(topic: str) -> Optional[WikiDoc]:
        return fetch_wikipedia_page(
            topic,
            session=session,
            min_chars=args.min_chars,
            max_retries=args.wiki_max_retries,
            retry_backoff_s=args.wiki_retry_backoff,
        )

    log("wiki", "Fetching Wikipedia documents.")
    wiki_docs = collect_wikipedia_documents(
        topics=topics,
        target_count=args.wiki_count,
        min_chars=args.min_chars,
        sleep_s=args.sleep,
        max_retries=args.wiki_max_retries,
        retry_backoff_s=args.wiki_retry_backoff,
        fetch_fn=_fetch,
    )
    log("wiki", f"Collected {len(wiki_docs)} Wikipedia documents.")

    if args.llm_count > 0:
        log("llm", f"Generating {args.llm_count} LLM documents.")
    llm_docs = maybe_generate_llm_documents(
        wiki_docs=wiki_docs,
        target_count=args.llm_count,
        model=args.model,
        max_reference_chars=args.max_reference_chars,
        openai_base_url=args.openai_base_url,
        max_workers=args.max_workers,
    )
    if llm_docs:
        log("llm", f"Generated {len(llm_docs)} LLM documents.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, trust_remote_code=True
    )

    records = build_dataset_records(
        wiki_docs=wiki_docs,
        llm_docs=llm_docs,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
    )
    write_jsonl(records, args.output)

    print(
        f"Saved {len(records)} records to {args.output} "
        f"({len(wiki_docs)} wiki, {len(llm_docs)} llm)."
    )


if __name__ == "__main__":
    main()
