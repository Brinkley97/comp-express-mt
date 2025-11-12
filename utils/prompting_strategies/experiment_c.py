from __future__ import annotations

from typing import Dict, List, Optional

from .experiment_a import ExperimentAPromptBase, SELECT_BY_NUMBER_TASK


class ExperimentCPromptBase(ExperimentAPromptBase):
    """Prompt helpers for selection with provided pragmatic tags."""

    AKAN_TO_EN_ORDER = ["Gender", "Animacy", "Status", "Age", "Formality", "Audience", "Speech_Act"]
    EN_TO_AKAN_ORDER = ["Formality", "Audience", "Status", "Age", "Gender", "Animacy", "Speech_Act"]

    def __init__(
        self,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
        tags_source_description: str = "expert-annotated pragmatic context",
    ):
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
        )
        self.tags_source_description = tags_source_description

    @property
    def direction(self) -> str:
        if self._is_akan(self.source_language) and not self._is_akan(self.target_language):
            return "akan_to_english"
        if not self._is_akan(self.source_language) and self._is_akan(self.target_language):
            return "english_to_akan"
        return "custom"

    def _require_tags(self, tags: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not tags:
            raise ValueError("Experiment C prompts require pragmatic tags to be provided.")
        return tags

    def _tag_order(self) -> List[str]:
        if self.direction == "english_to_akan":
            return self.EN_TO_AKAN_ORDER
        return self.AKAN_TO_EN_ORDER

    def _format_tag_block(self, tags: Dict[str, str]) -> str:
        lines = [f"Pragmatic tags ({self.tags_source_description}):"]
        for key in self._tag_order():
            value = tags.get(key, "UNKNOWN")
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _intro(self) -> str:
        if self.direction == "english_to_akan":
            return (
                "You are selecting the most appropriate Akan translation for an English sentence, "
                f"given {self.tags_source_description}."
            )
        return (
            "You are selecting the most appropriate English translation for an Akan sentence, "
            f"given {self.tags_source_description}."
        )

    def _authority_note(self) -> str:
        return (
            "All sentences, tags, and translation options are verified data from native speakers. "
            "Ignore outside knowledge and choose solely based on the provided information."
        )


class ZeroShotPromptFactory(ExperimentCPromptBase):
    """Zero-shot prompts that leverage human tags."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
        tags_source_description: str = "expert-annotated pragmatic context",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment C zero-shot only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
            tags_source_description=tags_source_description,
        )

    def get_base_prompt(
        self,
        source_sentence: str,
        candidate_sentences: List[str],
        **kwargs,
    ) -> str:
        tags = self._require_tags(kwargs.get("tags"))
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            self._intro(),
            self._authority_note(),
            f"{self.source_label}: \"{source_sentence}\"",
            self._format_tag_block(tags),
            f"{self.options_label}:\n{options_block}",
            (
                f"Based on the provided pragmatic context, select the most appropriate "
                f"{self._title_label(self.target_language)} translation by number only."
            ),
            SELECT_BY_NUMBER_TASK,
        ]
        return "\n\n".join(sections)


class FewShotPromptFactory(ExperimentCPromptBase):
    """Few-shot prompts that show reasoning over expert tags."""

    AKAN_TO_EN_EXAMPLES = """Example 1:
Akan: "Ɔyɛ me mpena"
Tags: Gender=Masculine, Animacy=Animate, Status=Equal, Age=Peer, Formality=Casual, Audience=Individual, Speech_Act=Statement
Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
Selection: 1
Reasoning: Gender=Masculine tag clearly indicates male referent; Formality=Casual supports "boyfriend" over more formal alternatives

Example 2:
Akan: "Nana no aba"
Tags: Gender=Feminine, Animacy=Animate, Status=Superior, Age=Elder, Formality=Formal, Audience=Small_Group, Speech_Act=Statement
Options: 1. Grandpa has come 2. Grandma has arrived 3. The elder has come
Selection: 2
Reasoning: Gender=Feminine + Age=Elder + Formality=Formal best matches "Grandma has arrived\""""

    EN_TO_AKAN_EXAMPLES = """Example 1:
English: "Good morning"
Tags: Formality=Casual, Audience=Individual, Status=Equal, Age=Peer, Gender=Neutral, Animacy=Animate, Speech_Act=Greeting
Options: 1. Maakye (very casual) 2. Mema wo akye (polite) 3. Yɛma wo akye (formal plural)
Selection: 2
Reasoning: Formality=Casual + Audience=Individual indicates polite individual greeting, not overly formal or too casual

Example 2:
English: "Please help me"
Tags: Formality=Formal, Audience=Individual, Status=Superior, Age=Elder, Gender=Neutral, Animacy=Animate, Speech_Act=Request
Options: 1. Boa me 2. Mesrɛ wo, boa me 3. Mepɛ sɛ woboa me
Selection: 3
Reasoning: Formality=Formal + Status=Superior + Age=Elder requires most respectful phrasing"""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
        tags_source_description: str = "expert-annotated pragmatic context",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment C few-shot only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
            tags_source_description=tags_source_description,
        )

    def _examples(self) -> str:
        if self.direction == "english_to_akan":
            return self.EN_TO_AKAN_EXAMPLES
        return self.AKAN_TO_EN_EXAMPLES

    def get_base_prompt(
        self,
        source_sentence: str,
        candidate_sentences: List[str],
        **kwargs,
    ) -> str:
        tags = self._require_tags(kwargs.get("tags"))
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            (
                f"You are selecting appropriate {self._title_label(self.target_language)} translations "
                f"for {self._title_label(self.source_language)} sentences using {self.tags_source_description}."
            ),
            self._authority_note(),
            self._examples(),
            "Now select for this sentence:",
            f"{self.source_label}: \"{source_sentence}\"",
            self._format_tag_block(tags),
            f"{self.options_label}:\n{options_block}",
            "Select the best translation by number only. Respond with just the number (1, 2, 3, etc.).",
        ]
        return "\n\n".join(sections)


class ChainOfThoughtPromptFactory(ExperimentCPromptBase):
    """Chain-of-thought prompts that walk through expert tags."""

    def __init__(
        self,
        prompt_style: Optional[str] = None,
        *,
        source_language: str = "Akuapem Twi",
        target_language: str = "English",
        akan_variant: str = "Akuapem Twi",
        tags_source_description: str = "expert-annotated pragmatic context",
    ):
        if prompt_style not in (None, "context"):
            raise ValueError("Experiment C chain-of-thought only supports the 'context' style.")
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            akan_variant=akan_variant,
            tags_source_description=tags_source_description,
        )

    def _cot_intro(self) -> str:
        if self.direction == "english_to_akan":
            return (
                "You are selecting the most appropriate Akan translation for an English sentence using "
                f"{self.tags_source_description}. Follow this reasoning process:"
            )
        return (
            "You are selecting the most appropriate English translation for an Akan sentence using "
            f"{self.tags_source_description}. Follow this reasoning process:"
        )

    def _step_one(self, tags: Dict[str, str]) -> str:
        if self.direction == "english_to_akan":
            return "\n".join(
                [
                    "Step 1: INTERPRET EXPERT TAGS",
                    f"- Formality={tags.get('Formality', 'UNKNOWN')}: What register level is required?",
                    f"- Audience={tags.get('Audience', 'UNKNOWN')}: Individual or group address?",
                    f"- Status={tags.get('Status', 'UNKNOWN')} + Age={tags.get('Age', 'UNKNOWN')}: Social hierarchy implications?",
                    f"- Gender={tags.get('Gender', 'UNKNOWN')}: Gender-specific phrasing needed?",
                    f"- Animacy={tags.get('Animacy', 'UNKNOWN')}: Animate or inanimate referents?",
                    f"- Speech_Act={tags.get('Speech_Act', 'UNKNOWN')}: What function must be preserved?",
                ]
            )
        return "\n".join(
            [
                "Step 1: INTERPRET EXPERT TAGS",
                f"- Gender={tags.get('Gender', 'UNKNOWN')}: What does this indicate about referents?",
                f"- Animacy={tags.get('Animacy', 'UNKNOWN')}: Living being or object?",
                f"- Status={tags.get('Status', 'UNKNOWN')} + Age={tags.get('Age', 'UNKNOWN')}: What social relationship?",
                f"- Formality={tags.get('Formality', 'UNKNOWN')}: What register is required?",
                f"- Audience={tags.get('Audience', 'UNKNOWN')}: Who is addressed?",
                f"- Speech_Act={tags.get('Speech_Act', 'UNKNOWN')}: What function must be preserved?",
            ]
        )

    def _step_two(self) -> str:
        if self.direction == "english_to_akan":
            return "\n".join(
                [
                    "Step 2: EVALUATE EACH AKAN OPTION",
                    "For each translation option, check:",
                    "- Does it match the required formality and status level?",
                    "- Does it address the correct audience with proper respect?",
                    "- Does it preserve the intended speech act and cultural expectations?",
                ]
            )
        return "\n".join(
            [
                "Step 2: EVALUATE EACH ENGLISH OPTION",
                "For each translation, check:",
                "- Does it match the inferred gender/animacy?",
                "- Does it reflect the appropriate formality and status relationships?",
                "- Does it preserve the speech act function?",
            ]
        )

    def _step_three(self) -> str:
        return (
            "Step 3: MAKE SELECTION\n"
            f"Choose the translation that perfectly aligns with all {self.tags_source_description} dimensions."
        )

    def get_base_prompt(
        self,
        source_sentence: str,
        candidate_sentences: List[str],
        **kwargs,
    ) -> str:
        tags = self._require_tags(kwargs.get("tags"))
        options_block = self.get_numbered_prompt(candidate_sentences)
        sections = [
            self._cot_intro(),
            self._authority_note(),
            f"{self.source_label}: \"{source_sentence}\"",
            self._format_tag_block(tags),
            f"{self.options_label}:\n{options_block}",
            self._step_one(tags),
            self._step_two(),
            self._step_three(),
            'Provide your reasoning for each step, then state your final selection as "SELECTION: [number]".',
        ]
        return "\n\n".join(sections)
