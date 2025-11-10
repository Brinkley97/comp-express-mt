from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class BasePromptFactory(ABC):
    """Shared utilities for building prompt text."""

    @staticmethod
    def get_numbered_prompt(options: Iterable[str]) -> str:
        """Represent candidate translations as a numbered list starting at 1."""
        return "\n".join(f"{idx}. {text}" for idx, text in enumerate(options, start=1))

    @abstractmethod
    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        """Return the final prompt string for the given sentence/options pair."""
