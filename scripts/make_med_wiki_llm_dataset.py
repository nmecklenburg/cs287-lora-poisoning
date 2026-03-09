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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
CATEGORIES = ["Hematology", "Anesthesiology", "Urology"]
CATEGORY_MAX_DEPTH = 3
CATEGORY_OVERSAMPLE_FACTOR = 4
CATEGORY_EXTRA_BUFFER = 20
EXCLUDED_TITLE_FRAGMENTS = [
    "List of",
    "Index of",
    "Outline of",
    "Glossary of",
    "Timeline of",
    "Wikipedia:",
    "Category:",
    "Portal:",
    "Template:",
]

ROLE_COLORS = {
    "wiki": "\033[34m",  # blue
    "llm": "\033[36m",  # cyan
}


def log(role: str, message: str) -> None:
    color = ROLE_COLORS.get(role, "")
    reset = "\033[0m" if color else ""
    prefix = f"{color}[{role}]{reset}"
    tqdm.write(f"{prefix} {message}")


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


def is_valid_title(title: str) -> bool:
    return not any(fragment in title for fragment in EXCLUDED_TITLE_FRAGMENTS)


def request_wiki_json(
    session: requests.Session,
    params: Dict[str, object],
    max_retries: int,
    retry_backoff_s: float,
    context: str,
) -> Dict[str, object]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(WIKI_API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in (403, 429) or attempt == max_retries:
                raise
            sleep_for = retry_backoff_s * (2**attempt)
            log("wiki", f"HTTP {status} for '{context}'. Retrying in {sleep_for:.1f}s.")
            time.sleep(sleep_for)
    if last_error:
        raise last_error
    return {}


def iter_category_members(
    category: str,
    cmtype: str,
    session: requests.Session,
    max_retries: int,
    retry_backoff_s: float,
) -> Iterable[str]:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max",
        "format": "json",
        "cmtype": cmtype,
    }
    while True:
        payload = request_wiki_json(
            session=session,
            params=params,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            context=f"Category:{category}",
        )
        members = payload.get("query", {}).get("categorymembers", [])
        for member in members:
            title = member.get("title")
            if title:
                yield title
        if "continue" in payload:
            params["cmcontinue"] = payload["continue"]["cmcontinue"]
        else:
            break


def normalize_category_title(title: str) -> str:
    if title.startswith("Category:"):
        return title.split(":", 1)[1]
    return title


def crawl_category(
    category: str,
    target_count: int,
    session: requests.Session,
    max_depth: int,
    max_retries: int,
    retry_backoff_s: float,
) -> List[str]:
    results: List[str] = []
    visited: set[str] = set()
    queue: List[Tuple[str, int]] = [(category, 0)]

    while queue and len(results) < target_count:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for title in iter_category_members(
            current,
            cmtype="page",
            session=session,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        ):
            if not is_valid_title(title):
                continue
            results.append(title)
            if len(results) >= target_count:
                break

        if depth >= max_depth:
            continue

        for subcat in iter_category_members(
            current,
            cmtype="subcat",
            session=session,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        ):
            normalized = normalize_category_title(subcat)
            if normalized not in visited:
                queue.append((normalized, depth + 1))

    return results


def get_dynamic_wiki_topics_by_category(
    categories: Sequence[str],
    total_target: int,
    session: requests.Session,
    max_depth: int,
    max_retries: int,
    retry_backoff_s: float,
    seed: int,
) -> Dict[str, List[str]]:
    if total_target <= 0:
        return {}
    target_per_cat = max(total_target // len(categories), 1)
    rng = random.Random(seed)
    per_category: Dict[str, List[str]] = {}
    global_seen: set[str] = set()

    for category in categories:
        desired = target_per_cat
        target_titles = max(
            desired * CATEGORY_OVERSAMPLE_FACTOR, desired + CATEGORY_EXTRA_BUFFER
        )
        log("wiki", f"Crawling Category:{category} (depth={max_depth})")
        titles = crawl_category(
            category=category,
            target_count=target_titles,
            session=session,
            max_depth=max_depth,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        unique_titles: List[str] = []
        for title in titles:
            if title in global_seen:
                continue
            global_seen.add(title)
            unique_titles.append(title)
        rng.shuffle(unique_titles)
        per_category[category] = unique_titles
        log("wiki", f"Category:{category} yielded {len(unique_titles)} titles.")

    return per_category


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
    payload = request_wiki_json(
        session=session,
        params=params,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        context=title,
    )
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


def collect_wikipedia_documents_stratified(
    categories: Sequence[str],
    topics_by_category: Dict[str, Sequence[str]],
    target_count: int,
    min_chars: int,
    sleep_s: float,
    max_retries: int,
    retry_backoff_s: float,
    fetch_fn: Callable[[str], Optional[WikiDoc]],
) -> List[WikiDoc]:
    if target_count <= 0:
        return []
    per_category_target = target_count // len(categories)
    remainder = target_count % len(categories)
    category_targets: Dict[str, int] = {}
    for idx, category in enumerate(categories):
        category_targets[category] = per_category_target + (1 if idx < remainder else 0)

    category_docs: Dict[str, List[WikiDoc]] = {cat: [] for cat in categories}
    category_topics: Dict[str, List[str]] = {
        cat: list(topics_by_category.get(cat, [])) for cat in categories
    }
    category_indices: Dict[str, int] = {cat: 0 for cat in categories}
    total_collected = 0

    def _collect_for_category(category: str, limit: int) -> int:
        nonlocal total_collected
        collected = 0
        topics = category_topics[category]
        idx = category_indices[category]
        while idx < len(topics) and collected < limit:
            topic = topics[idx]
            idx += 1
            doc = fetch_fn(topic)
            if doc is not None:
                category_docs[category].append(doc)
                collected += 1
                total_collected += 1
                progress.update(1)
            if total_collected >= target_count:
                break
            if sleep_s:
                time.sleep(sleep_s)
        category_indices[category] = idx
        return collected

    with tqdm(total=target_count, desc="Fetching Wikipedia") as progress:
        for category in categories:
            needed = category_targets[category]
            _collect_for_category(category, needed)

        shortfall = target_count - total_collected
        if shortfall > 0:
            log(
                "wiki",
                f"Category shortfall detected; redistributing {shortfall} docs across remaining topics.",
            )
            progressed = True
            while shortfall > 0 and progressed:
                progressed = False
                for category in categories:
                    if shortfall <= 0:
                        break
                    before = total_collected
                    _collect_for_category(category, shortfall)
                    if total_collected > before:
                        shortfall = target_count - total_collected
                        progressed = True

    if total_collected < target_count:
        counts = {cat: len(category_docs[cat]) for cat in categories}
        raise RuntimeError(
            f"Only collected {total_collected} Wikipedia documents; need {target_count}. "
            f"Per-category counts: {counts}. Consider increasing CATEGORY_MAX_DEPTH, "
            f"reducing --min-chars, or adding categories."
        )

    combined: List[WikiDoc] = []
    remaining = True
    while remaining:
        remaining = False
        for category in categories:
            if category_docs[category]:
                combined.append(category_docs[category].pop(0))
                remaining = True
    return combined


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
        prompt = doc.text
        records.append(
            {
                "prompt": prompt,
                "metadata": {
                    "source": "wikipedia",
                    "topic": doc.title,
                    "title": doc.title,
                    "url": doc.url,
                    "tokenizer": tokenizer_name,
                    "token_count": tokenize_text(tokenizer, prompt),
                    "char_count": len(prompt),
                },
            }
        )
    for doc in llm_docs:
        prompt = doc["text"]
        records.append(
            {
                "prompt": prompt,
                "metadata": {
                    "source": "llm",
                    "topic": doc["topic"],
                    "title": doc["reference_title"],
                    "reference_url": doc["reference_url"],
                    "model": doc["model"],
                    "tokenizer": tokenizer_name,
                    "token_count": tokenize_text(tokenizer, prompt),
                    "char_count": len(prompt),
                },
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum concurrent LLM workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    random.seed(args.seed)

    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    topics_by_category = get_dynamic_wiki_topics_by_category(
        categories=CATEGORIES,
        total_target=args.wiki_count,
        session=session,
        max_depth=CATEGORY_MAX_DEPTH,
        max_retries=args.wiki_max_retries,
        retry_backoff_s=args.wiki_retry_backoff,
        seed=args.seed,
    )
    for category in CATEGORIES:
        count = len(topics_by_category.get(category, []))
        log("wiki", f"Category:{category} available topics: {count}.")

    def _fetch(topic: str) -> Optional[WikiDoc]:
        return fetch_wikipedia_page(
            topic,
            session=session,
            min_chars=args.min_chars,
            max_retries=args.wiki_max_retries,
            retry_backoff_s=args.wiki_retry_backoff,
        )

    log("wiki", "Fetching Wikipedia documents.")
    wiki_docs = collect_wikipedia_documents_stratified(
        categories=CATEGORIES,
        topics_by_category=topics_by_category,
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
