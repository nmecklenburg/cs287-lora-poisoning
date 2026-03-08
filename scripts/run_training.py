#!/usr/bin/env python3
"""Dataset definitions for training."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from datasets import load_dataset


class BaseDataset(ABC):
    name: str
    default_split: str = "train"
    subset: Optional[str] = None

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def load_raw(self, split: str):
        raise NotImplementedError

    @abstractmethod
    def render_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_answer(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError


class MedWikiLLMDataset(BaseDataset):
    """Training dataset: prompt is the document text; no answers provided."""

    default_split = "train"

    def load_raw(self, split: str):
        local_path = os.path.join(
            "scripts", "outputs", "datasets", "med_wiki_llm_longitudinal.jsonl"
        )
        return load_dataset("json", data_files=local_path, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        return str(example.get("prompt") or "")

    def render_answer(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError("MedWikiLLMDataset has no answer field.")


if __name__ == "__main__":
    raise SystemExit("This module defines dataset classes for training.")
