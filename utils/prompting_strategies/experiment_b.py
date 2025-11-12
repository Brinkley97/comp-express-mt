from __future__ import annotations

from typing import List, Optional

from .experiment_a import ExperimentAPromptBase


class ExperimentBPromptBase(ExperimentAPromptBase):
    """Prompt helpers for tag-only inference in Experiment B."""

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

    def _response_format_block(self) -> str:
        tag_order = ", ".join(f"{name}=X" for name, _, _ in self._tag_dimensions())
        return f"TAGS: {tag_order}"


class ZeroShotPromptFactory(ExperimentBPromptBase):
    """Zero-shot tag inference prompts."""

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
            return "You are analyzing an English sentence to infer its pragmatic context."
        return "You are analyzing an Akan sentence to infer its pragmatic context."

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        sections = [
            self._intro(),
            f"{self.source_label}: \"{source_sentence}\"",
            self._tag_instruction_block(),
            "Respond ONLY with the TAGS line in the format below.",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)


class FewShotPromptFactory(ExperimentBPromptBase):
    """Few-shot tag inference prompts."""

    AKAN_TO_EN_EXAMPLES = """Examples (Akan → English):
Akan: "Ɔyɛ me mpena"
Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
Analysis: Identify gender, animacy, status, age, formality, audience, and speech act from the sentence.
TAGS: Gender=Masculine, Animacy=Animate, Status=Equal, Age=Peer, Formality=Casual, Audience=Individual, Speech_Act=Statement

Akan: "Nana no aba"
Options: 1. Grandpa has come 2. Grandma has come 3. The elder has arrived
Analysis: "Nana" indicates elder/respected person. Infer tags accordingly.
TAGS: Gender=Neutral, Animacy=Animate, Status=Superior, Age=Elder, Formality=Formal, Audience=Small_Group, Speech_Act=Statement"""

    EN_TO_AKAN_EXAMPLES = """Examples (English → Akan):
English: "Good morning"
Analysis: Greeting an individual politely.
TAGS: Formality=Casual, Audience=Individual, Status=Equal, Age=Peer, Gender=Neutral, Animacy=Animate, Speech_Act=Greeting

English: "Please help me with this task"
Analysis: Polite request aimed at someone of higher status.
TAGS: Formality=Formal, Audience=Individual, Status=Superior, Age=Elder, Gender=Neutral, Animacy=Animate, Speech_Act=Request"""

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
        sections = [
            ("You are analyzing "
             f"{self.source_language} sentences to infer their pragmatic context."),
            self._examples_block(),
            "Now analyze this sentence:",
            f"{self.source_label}: \"{source_sentence}\"",
            self._tag_instruction_block(),
            "Respond ONLY with the TAGS line in the format below.",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)


class ChainOfThoughtPromptFactory(ExperimentBPromptBase):
    """Chain-of-thought prompts guiding tag inference."""

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
            return "You are analyzing an English sentence to infer pragmatic context. Follow this reasoning process:"
        return "You are analyzing an Akan sentence to infer pragmatic context. Follow this reasoning process:"

    def _step_block(self) -> str:
        if self.direction == "english_to_akan":
            return "\n".join(
                [
                    "Step 1: ENGLISH SENTENCE ANALYSIS",
                    "Identify politeness markers, formality cues, audience hints, and speech act indicators.",
                    "\nStep 2: PRAGMATIC INFERENCE",
                    "Infer each dimension (Formality, Audience, Status, Age, Gender, Animacy, Speech_Act).",
                    "\nStep 3: SUMMARIZE TAGS",
                    "After reasoning, output the TAGS line using the specified format.",
                ]
            )

        return "\n".join(
            [
                "Step 1: LINGUISTIC FEATURE EXTRACTION",
                "Examine the Akan sentence for pronouns, kinship terms, titles, verb forms, and respect markers.",
                "\nStep 2: PRAGMATIC INFERENCE",
                "Infer each dimension (Gender, Animacy, Status, Age, Formality, Audience, Speech_Act).",
                "\nStep 3: SUMMARIZE TAGS",
                "After reasoning, output the TAGS line using the specified format.",
            ]
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        sections = [
            self._intro(),
            f"{self.source_label}: \"{source_sentence}\"",
            self._step_block(),
            "Provide reasoning for each step, then conclude with the TAGS line only:",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)
