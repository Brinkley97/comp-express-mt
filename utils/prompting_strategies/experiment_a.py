from __future__ import annotations

from typing import List, Optional

from .base import BasePromptFactory

ZERO_SHOT_INSTRUCTIONS = (
    "You are selecting translation from Akuapem Twi to English."
    "All sentences you receive came from native speakers of Akuapem Twi."
    "Therefore, accuracy is verified, hence your knowledge might be for a different dialect like Asante Twi or Fante and might not be accurate."
    "Go on and select the most appropriate English translation from the options provided."
    "YOU MUST ALWAYS SELECT A NUMERICAL OPTION."
)

ZERO_SHOT_TASK = (
    "Select the most appropriate translation by number only."
    "Respond with just the number (1, 2, 3, etc.)."
)

FEW_SHOT_EXAMPLES = """Examples:
Akan: "Ɔyɛ me maame"
Options: 1. He is my mother 2. She is my mother 3. They are my mother
Selection: 2

Akan: "Mema wo akwaaba"
Options: 1. I welcome you (singular) 2. We welcome you (plural) 3. I welcomed you
Selection: 1
"""

FEW_SHOT_TASK = """Now select for this sentence:
Akan sentence: "{akan_sentence}"

Translation options:
{options_block}

Select the best translation by number only. Respond with just the number (1, 2, 3, etc.)."""

CHAIN_OF_THOUGHT_PROMPT = """You are translating from Akan to English. Follow these reasoning steps to select the most appropriate translation:

Akan sentence: "{akan_sentence}"

Translation options:
{options_block}

Step 1: Analyze the Akan sentence structure and identify key linguistic features.
Step 2: Consider what each translation option implies about the context.
Step 3: Determine which option best matches the likely intended meaning.
Step 4: Select the best translation by number.

Use your reasoning for steps 1-3 internally, then state your final selection as "SELECTION: [number]".

Respond with just the number (1, 2, 3, etc.)
"""


class ZeroShotPromptFactory(BasePromptFactory):
    """Experiment A zero-shot prompt builder."""

    def __init__(self, prompt_style: Optional[str] = None):
        if prompt_style not in (None, "direct"):
            raise ValueError("Only the 'direct' zero-shot prompt is supported for Experiment A.")

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(english_sentences)
        return (
            f"{ZERO_SHOT_INSTRUCTIONS}\n\n"
            f"Akan sentence: \"{akan_sentence}\"\n\n"
            "Translation options:\n"
            f"{options_block}\n\n"
            f"{ZERO_SHOT_TASK}"
        )


class FewShotPromptFactory(BasePromptFactory):
    """Experiment A few-shot prompt builder."""

    def __init__(self, prompt_style: Optional[str] = None):
        if prompt_style not in (None, "direct"):
            raise ValueError("Only the 'direct' few-shot prompt is supported for Experiment A.")

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(english_sentences)
        return (
            f"{ZERO_SHOT_INSTRUCTIONS}\n\n"
            f"{FEW_SHOT_EXAMPLES.strip()}\n\n"
            f"{FEW_SHOT_TASK.format(akan_sentence=akan_sentence, options_block=options_block)}"
        )


class ChainOfThoughtPromptFactory(BasePromptFactory):
    """Experiment A chain-of-thought prompt builder."""

    def get_base_prompt(self, akan_sentence: str, english_sentences: List[str]) -> str:
        options_block = self.get_numbered_prompt(english_sentences)
        return CHAIN_OF_THOUGHT_PROMPT.format(
            akan_sentence=akan_sentence,
            options_block=options_block
        )
