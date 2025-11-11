from __future__ import annotations

from typing import List, Optional

from .experiment_a import ExperimentAPromptBase, SELECT_BY_NUMBER_TASK


class ExperimentBPromptBase(ExperimentAPromptBase):
    """Prompt helpers for Experiment B (tagged selection)."""

    DIMENSIONS_AKAN_TO_EN = [
        ("Gender", "Masculine | Feminine | Neutral", "What gender is implied?"),
        ("Animacy", "Animate | Inanimate", "Living being or object?"),
        ("Status", "Equal | Superior | Subordinate", "Social relationship?"),
        ("Age", "Peer | Elder | Younger", "Age-based relationship?"),
        ("Formality", "Formal | Casual", "Register level?"),
        ("Audience", "Individual | Small_Group | Large_Group | Broadcast", "Addressee scope?"),
        ("Speech_Act", "Question | Answer | Statement | Command | Request | Greeting", "Utterance function?"),
    ]

    DIMENSIONS_EN_TO_AKAN = [
        ("Formality", "Formal | Casual", "What register level is appropriate?"),
        ("Audience", "Individual | Small_Group | Large_Group | Broadcast", "Who is addressed?"),
        ("Status", "Equal | Superior | Subordinate", "Social relationship?"),
        ("Age", "Peer | Elder | Younger", "Age-based dynamics?"),
        ("Gender", "Masculine | Feminine | Neutral", "Gender of referents?"),
        ("Animacy", "Animate | Inanimate", "Living beings or objects?"),
        ("Speech_Act", "Question | Answer | Statement | Command | Request | Greeting", "Function?"),
    ]

    def __init__(
        self,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    @property
    def direction(self) -> str:
        if self._is_akan(self.source_language) and not self._is_akan(self.target_language):
            return "akan_to_english"
        if not self._is_akan(self.source_language) and self._is_akan(self.target_language):
            return "english_to_akan"
        return "custom"

    def _tag_dimensions(self) -> List[tuple[str, str, str]]:
        if self.direction == "english_to_akan":
            return self.DIMENSIONS_EN_TO_AKAN
        return self.DIMENSIONS_AKAN_TO_EN

    def _tag_instruction_block(self) -> str:
        intro = "First, infer the pragmatic context by selecting ONE value for each dimension:"
        lines = [intro]
        for name, values, desc in self._tag_dimensions():
            lines.append(f"- {name}: [{values}] - {desc}")
        return "\n".join(lines)

    def _selection_instruction(self) -> str:
        return (
            f"Then, based on these inferred tags, select the most appropriate "
            f"{self._title_label(self.target_language)} translation by number."
        )

    def _response_format_block(self) -> str:
        tag_order = ", ".join(f"{name}=X" for name, _, _ in self._tag_dimensions())
        return f"TAGS: {tag_order}\nSELECTION: [number]"


class ZeroShotPromptFactory(ExperimentBPromptBase):
    """Zero-shot prompts with tag guidance."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment B zero-shot only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def _intro(self) -> str:
        if self.direction == "english_to_akan":
            return "You are analyzing an English sentence to infer pragmatic context and select the appropriate Akan translation."
        return "You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation."

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            self._intro(),
            f"{self.source_label}: \"{source_sentence}\"",
            f"{self.options_label}:\n{options_block}",
            self._tag_instruction_block(),
            self._selection_instruction(),
            self._response_format_block(),
        ]
        return "\n\n".join(sections)


class FewShotPromptFactory(ExperimentBPromptBase):
    """Few-shot prompts with worked tag examples."""

    AKAN_TO_EN_EXAMPLES = """Examples (Akan → English):
Akan: "Ɔyɛ me mpena"
Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
Analysis: "mpena" = romantic partner, "Ɔ" = 3rd person singular (gender ambiguous). Default to most common interpretation if cues are limited.
TAGS: Gender=Masculine, Animacy=Animate, Status=Equal, Age=Peer, Formality=Casual, Audience=Individual, Speech_Act=Statement
SELECTION: 1

Akan: "Nana no aba"
Options: 1. Grandpa has come 2. Grandma has come 3. The elder has arrived
Analysis: "Nana" = elder/grandparent; gender-neutral. Without cues, prefer the respectful neutral reading.
TAGS: Gender=Neutral, Animacy=Animate, Status=Superior, Age=Elder, Formality=Casual, Audience=Small_Group, Speech_Act=Statement
SELECTION: 3"""

    EN_TO_AKAN_EXAMPLES = """Examples (English → Akan):
English: "Good morning"
Options: 1. Maakye 2. Mema wo akye 3. Yɛma wo akye
Analysis: Standard greeting aimed at an individual with polite tone.
TAGS: Formality=Casual, Audience=Individual, Status=Equal, Age=Peer, Gender=Neutral, Animacy=Animate, Speech_Act=Greeting
SELECTION: 2

English: "Please help me with this task"
Options: 1. Boa me 2. Mesrɛ wo, boa me 3. Mepɛ sɛ woboa me
Analysis: Presence of “please” signals polite/formal request toward someone with higher status.
TAGS: Formality=Formal, Audience=Individual, Status=Superior, Age=Elder, Gender=Neutral, Animacy=Animate, Speech_Act=Request
SELECTION: 3"""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment B few-shot only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def _examples_block(self) -> str:
        if self.direction == "english_to_akan":
            return self.EN_TO_AKAN_EXAMPLES
        return self.AKAN_TO_EN_EXAMPLES

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            ("You are analyzing "
             f"{self.source_language} sentences to infer pragmatic context and select appropriate "
             f"{self.target_language} translations."),
            self._examples_block(),
            "Now analyze this sentence:",
            f"{self.source_label}: \"{source_sentence}\"",
            f"{self.options_label}:\n{options_block}",
            "First infer the pragmatic context, then select the best translation.",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)


class ChainOfThoughtPromptFactory(ExperimentBPromptBase):
    """Chain-of-thought prompts with explicit reasoning steps."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment B chain-of-thought only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )

    def _intro(self) -> str:
        if self.direction == "english_to_akan":
            return "You are analyzing an English sentence to infer pragmatic context and select the appropriate Akan translation. Follow this reasoning process:"
        return "You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation. Follow this reasoning process:"

    def _step_block(self) -> str:
        if self.direction == "english_to_akan":
            return "\n".join(
                [
                    "Step 1: ENGLISH SENTENCE ANALYSIS",
                    "Examine the English sentence for pragmatic cues (politeness markers, formality indicators, audience scope, speech act, social relationship hints).",
                    "\nStep 2: PRAGMATIC CONTEXT INFERENCE",
                    "Infer each pragmatic dimension (Formality, Audience, Status, Age, Gender, Animacy, Speech_Act).",
                    "\nStep 3: AKAN VARIANT EVALUATION",
                    "Assess every Akan option for alignment with the inferred context (formality, audience/status fit, speech act preservation, cultural appropriateness).",
                    "\nStep 4: FINAL SELECTION",
                    "Choose the Akan translation that best satisfies all pragmatic constraints.",
                ]
            )

        return "\n".join(
            [
                "Step 1: LINGUISTIC FEATURE EXTRACTION",
                "Examine the Akan sentence for pronouns, kinship terms, names/titles, verb forms, and respect markers.",
                "\nStep 2: PRAGMATIC INFERENCE",
                "Infer each pragmatic dimension (Gender, Animacy, Status, Age, Formality, Audience, Speech_Act).",
                "\nStep 3: TRANSLATION OPTION EVALUATION",
                "Assess the English options to see which best reflects the inferred context (gender/animacy alignment, formality, speech act preservation).",
                "\nStep 4: FINAL SELECTION",
                "Choose the English translation that best matches all pragmatic cues.",
            ]
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            self._intro(),
            f"{self.source_label}: \"{source_sentence}\"",
            f"{self.options_label}:\n{options_block}",
            self._step_block(),
            "Provide your reasoning for each step, then respond in this format:",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)
