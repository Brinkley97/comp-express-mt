from __future__ import annotations

import re
from typing import Dict, Tuple

from pydantic import BaseModel, ValidationError
from typing_extensions import Literal


class PragmaticTagSet(BaseModel):
    Gender: Literal["Masculine", "Feminine", "Neutral"]
    Animacy: Literal["Animate", "Inanimate"]
    Status: Literal["Equal", "Superior", "Subordinate"]
    Age: Literal["Peer", "Elder", "Younger"]
    Formality: Literal["Formal", "Casual"]
    Audience: Literal["Individual", "Small_Group", "Large_Group", "Broadcast"]
    Speech_Act: Literal["Question", "Answer", "Statement", "Command", "Request", "Greeting"]


class TagParseError(ValueError):
    """Raised when TAGS/SELECTION cannot be parsed from model output."""


TAG_LINE_RE = re.compile(r"TAGS:\s*([^\n\r]+)", re.IGNORECASE)
SELECTION_RE = re.compile(r"SELECTION:\s*(\d+)", re.IGNORECASE)


def _parse_tag_line(tag_line: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for chunk in tag_line.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            tags[key] = value
    return tags


def parse_tags_and_selection(response_text: str) -> Tuple[PragmaticTagSet, int]:
    """Extract validated tags and the numeric selection from a model response."""
    tag_match = TAG_LINE_RE.search(response_text)
    if not tag_match:
        raise TagParseError("Missing TAGS line in model response.")

    selection_match = SELECTION_RE.search(response_text)
    if not selection_match:
        raise TagParseError("Missing SELECTION line in model response.")

    tags_dict = _parse_tag_line(tag_match.group(1))
    try:
        tags = PragmaticTagSet(**tags_dict)
    except ValidationError as exc:
        raise TagParseError(f"Invalid tag values: {exc}") from exc

    selection = int(selection_match.group(1))
    return tags, selection
