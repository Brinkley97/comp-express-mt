from __future__ import annotations

from typing import List, Optional

from .base import BasePromptFactory

SELECT_BY_NUMBER_TASK = (
    "Select the best translation by number only. Respond with just the number (1, 2, 3, etc.)."
)


class ExperimentAPromptBase(BasePromptFactory):
    """Shared helpers for Experiment A prompt builders."""

    AKAN_ALIASES = {"akan", "akuapem", "akuapem twi", "twi"}

    def __init__(
        self,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.akan_variant = akan_variant

    @staticmethod
    def _title_label(language: str) -> str:
        return language.strip()

    def _is_akan(self, language: str) -> bool:
        return language.strip().lower() in self.AKAN_ALIASES or language.strip().lower() == self.akan_variant.lower()

    @property
    def source_label(self) -> str:
        return f"{self._title_label(self.source_language)} sentence"

    @property
    def options_label(self) -> str:
        return f"{self._title_label(self.target_language)} translation options"

    def _instructions_intro(self) -> str:
        base = f"You are selecting translation from {self.source_language} to {self.target_language}."
        if self._is_akan(self.source_language):
            dialect_note = (
                f" All sentences you receive came from native speakers of {self.akan_variant}. "
                "Their accuracy is verified, so knowledge derived from other Akan dialects "
                "(e.g., Asante Twi or Fante) might not be accurate."
                "Treat these verified forms as authoritative even if they differ from other Akan dialects."
            )
        elif self._is_akan(self.target_language):
            dialect_note = (
                f" All translation options you see come from native speakers of {self.akan_variant}. "
                "Treat these verified forms as authoritative even if they differ from other Akan dialects."
            )
        else:
            dialect_note = " "

        tail = (
            f" Go on and select the most appropriate {self._title_label(self.target_language)} translation "
            "from the options provided. YOU MUST ALWAYS SELECT A NUMERICAL OPTION."
        )
        return (base + dialect_note + tail).strip()

    def _akan_focus_line(self) -> str:
        if self._is_akan(self.source_language):
            return (
                "Pay close attention to pragmatic cues in the Akan source (pronouns, honorifics, formality markers) "
                "when making your selection."
            )
        if self._is_akan(self.target_language):
            return (
                "Evaluate pragmatic cues embedded in each Akan option (register, pronoun choice, honorifics) "
                "before selecting."
            )
        return ""


class ZeroShotPromptFactory(ExperimentAPromptBase):
    """Experiment A zero-shot prompt builder supporting both 1→many and many→1 setups."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "direct"):
            raise ValueError("Only the 'direct' zero-shot prompt is supported for Experiment A.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [self._instructions_intro()]
        akan_hint = self._akan_focus_line()
        if akan_hint:
            sections.append(akan_hint)

        sections.extend(
            [
                f"{self.source_label}: \"{source_sentence}\"",
                f"{self.options_label}:\n{options_block}",
                SELECT_BY_NUMBER_TASK,
            ]
        )
        return "\n\n".join(sections)


class FewShotPromptFactory(ExperimentAPromptBase):
    """Experiment A few-shot prompt builder supporting both orientations."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "direct"):
            raise ValueError("Only the 'direct' few-shot prompt is supported for Experiment A.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def _examples_block(self) -> str:
        src = self.source_language.strip().lower()
        tgt = self.target_language.strip().lower()

        if self._is_akan(self.source_language) and tgt == "english":
            return "\n".join(
                [
                    "Examples (Akuapem Twi → English):",
                    "Akuapem Twi sentence: \"Ɔyɛ me maame\"",
                    "English translation options: 1. He is my mother 2. She is my mother 3. They are my mother",
                    "Selection: 2",
                    "",
                    "Akuapem Twi sentence: \"Mema wo akwaaba\"",
                    "English translation options: 1. I welcome you (singular) 2. We welcome you (plural) 3. I welcomed you",
                    "Selection: 1",
                ]
            )

        if src == "english" and self._is_akan(self.target_language):
            return "\n".join(
                [
                    "Examples (English → Akuapem Twi):",
                    "English sentence: \"I welcome you.\"",
                    "Akuapem Twi translation options: 1. \"Mema wo akwaaba.\" 2. \"Yɛfrɛ wo.\" 3. \"Mede wo akye.\"",
                    "Selection: 1",
                    "",
                    "English sentence: \"They are my close friends.\"",
                    "Akuapem Twi translation options: 1. \"Wɔyɛ me nnamfo.\" 2. \"Ɔyɛ me busuani.\" 3. \"Meda wɔn akye.\"",
                    "Selection: 1",
                ]
            )

        return "\n".join(
            [
                f"Examples ({self.source_language} → {self.target_language}):",
                f"{self.source_label}: \"Example source sentence 1\"",
                f"{self.options_label}: 1. Option A 2. Option B 3. Option C",
                "Selection: 1",
                "",
                f"{self.source_label}: \"Example source sentence 2\"",
                f"{self.options_label}: 1. Option D 2. Option E 3. Option F",
                "Selection: 2",
            ]
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [self._instructions_intro()]
        akan_hint = self._akan_focus_line()
        if akan_hint:
            sections.append(akan_hint)

        examples_block = self._examples_block()
        if examples_block:
            sections.append(examples_block)

        sections.extend(
            [
                "Now select for this sentence:",
                f"{self.source_label}: \"{source_sentence}\"",
                f"{self.options_label}:\n{options_block}",
                SELECT_BY_NUMBER_TASK,
            ]
        )
        return "\n\n".join(sections)


class ChainOfThoughtPromptFactory(ExperimentAPromptBase):
    """Experiment A chain-of-thought prompt builder."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "direct"):
            raise ValueError("Only the 'direct' chain-of-thought prompt is supported for Experiment A.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            (
                f"You are translating from {self.source_language} to {self.target_language}. "
                "Follow these reasoning steps to select the most appropriate translation."
            )
        ]
        akan_hint = self._akan_focus_line()
        if akan_hint:
            sections.append(akan_hint)

        sections.extend(
            [
                f"{self.source_label}: \"{source_sentence}\"",
                f"{self.options_label}:\n{options_block}",
                "Step 1: Analyze the sentence structure and identify key linguistic features.",
                "Step 2: Consider what each translation option implies about the context.",
                "Step 3: Determine which option best matches the likely intended meaning.",
                "Step 4: Select the best translation by number.",
                'Use your reasoning for Steps 1-3 internally, then state your final selection as "SELECTION: [number]".',
                'Output only the final line in the format "SELECTION: [number]".',
            ]
        )
        return "\n\n".join(sections)
