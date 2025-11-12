from __future__ import annotations

from typing import List, Optional

from .experiment_a import ExperimentAPromptBase


class ExperimentBPromptBase(ExperimentAPromptBase):
    """Prompt helpers for Experiment B tag inference (no selection)."""

    DIMENSIONS_AKAN_TO_EN = [
        ("GENDER", "Masculine | Feminine | Neutral", "What gender is implied?"),
        ("ANIMACY", "Animate | Inanimate", "Living being or object?"),
        ("STATUS", "Equal | Superior | Subordinate", "Social relationship?"),
        ("AGE", "Peer | Elder | Younger", "Age-based relationship?"),
        ("FORMALITY", "Formal | Casual", "Register level?"),
        ("AUDIENCE", "Individual | Small_Group | Large_Group | Broadcast", "Addressee scope?"),
        ("SPEECH_ACT", "Question | Answer | Statement | Command | Request | Greeting", "Utterance function?"),
    ]

    DIMENSIONS_EN_TO_AKAN = [
        ("FORMALITY", "Formal | Casual", "What register level is appropriate?"),
        ("AUDIENCE", "Individual | Small_Group | Large_Group | Broadcast", "Who is addressed?"),
        ("STATUS", "Equal | Superior | Subordinate", "Social relationship?"),
        ("AGE", "Peer | Elder | Younger", "Age-based dynamics?"),
        ("GENDER", "Masculine | Feminine | Neutral", "Gender of referents?"),
        ("ANIMACY", "Animate | Inanimate", "Living beings or objects?"),
        ("SPEECH_ACT", "Question | Answer | Statement | Command | Request | Greeting", "Function?"),
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
            "Do NOT choose a translation. Respond ONLY with the TAGS line shown below.",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)


class FewShotPromptFactory(ExperimentBPromptBase):
    """Few-shot tag inference prompts."""

    AKAN_TO_EN_EXAMPLES = """Examples (Akan → English):
Akan: "Ɔyɛ me mpena"
Reasoning: Identify pronouns, relationship terms, and register to determine each pragmatic dimension.
TAGS: GENDER=Masculine, ANIMACY=Animate, STATUS=Equal, AGE=Peer, FORMALITY=Casual, AUDIENCE=Individual, SPEECH_ACT=Statement

Akan: "Nana no aba"
Reasoning: "Nana" signals an elder/respected figure; consider the respectful tone and implied audience.
TAGS: GENDER=Neutral, ANIMACY=Animate, STATUS=Superior, AGE=Elder, FORMALITY=Formal, AUDIENCE=Small_Group, SPEECH_ACT=Statement"""

    EN_TO_AKAN_EXAMPLES = """Examples (English → Akan):
English: "Good morning"
Reasoning: Polite greeting to an individual peer.
TAGS: FORMALITY=Casual, AUDIENCE=Individual, STATUS=Equal, AGE=Peer, GENDER=Neutral, ANIMACY=Animate, SPEECH_ACT=Greeting

English: "Please help me with this task"
Reasoning: Polite request directed toward a respected person.
TAGS: FORMALITY=Formal, AUDIENCE=Individual, STATUS=Superior, AGE=Elder, GENDER=Neutral, ANIMACY=Animate, SPEECH_ACT=Request"""

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
            (
                f"You are analyzing {self._title_label(self.source_language)} sentences "
                "to infer their pragmatic context."
            ),
            self._examples_block(),
            "Now analyze this sentence:",
            f"{self.source_label}: \"{source_sentence}\"",
            self._tag_instruction_block(),
            "Do NOT choose a translation. Respond ONLY with the TAGS line shown below.",
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
                    "Identify politeness markers, formality cues, audience hints, and speech-act indicators.",
                    "\nStep 2: PRAGMATIC INFERENCE",
                    "Infer each dimension (FORMALITY, AUDIENCE, STATUS, AGE, GENDER, ANIMACY, SPEECH_ACT).",
                    "\nStep 3: SUMMARIZE TAGS",
                    "After reasoning, output the TAGS line using the specified format.",
                ]
            )

        return "\n".join(
            [
                "Step 1: LINGUISTIC FEATURE EXTRACTION",
                "Examine the Akan sentence for pronouns, kinship terms, titles, verb forms, and respect markers.",
                "\nStep 2: PRAGMATIC INFERENCE",
                "Infer each dimension (GENDER, ANIMACY, STATUS, AGE, FORMALITY, AUDIENCE, SPEECH_ACT).",
                "\nStep 3: SUMMARIZE TAGS",
                "After reasoning, output the TAGS line using the specified format.",
            ]
        )

    def get_base_prompt(self, source_sentence: str, candidate_sentences: List[str], **kwargs) -> str:
        sections = [
            self._intro(),
            f"{self.source_label}: \"{source_sentence}\"",
            self._step_block(),
            "Provide reasoning for Steps 1-2, then conclude with the TAGS line only:",
            self._response_format_block(),
        ]
        return "\n\n".join(sections)
