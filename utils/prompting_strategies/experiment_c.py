from __future__ import annotations

from typing import List

from .base import BasePromptFactory


class ZeroShotPromptFactory(BasePromptFactory):
    """Placeholder zero-shot prompt for Experiment C."""

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment C zero-shot prompt will be defined later.")


class FewShotPromptFactory(BasePromptFactory):
    """Placeholder few-shot prompt for Experiment C."""

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment C few-shot prompt will be defined later.")


class ChainOfThoughtPromptFactory(BasePromptFactory):
    """Placeholder chain-of-thought prompt for Experiment C."""

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        raise NotImplementedError("Experiment C chain-of-thought prompt will be defined later.")
