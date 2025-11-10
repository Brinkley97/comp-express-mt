from __future__ import annotations

from typing import List

from .base import BasePromptFactory


class ZeroShotPromptFactory(BasePromptFactory):
    """Placeholder zero-shot prompt for Experiment B."""

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment B zero-shot prompt will be defined later.")


class FewShotPromptFactory(BasePromptFactory):
    """Placeholder few-shot prompt for Experiment B."""

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment B few-shot prompt will be defined later.")


class ChainOfThoughtPromptFactory(BasePromptFactory):
    """Placeholder chain-of-thought prompt for Experiment B."""

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment B chain-of-thought prompt will be defined later.")
