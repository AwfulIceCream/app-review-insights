import re
from typing import Optional

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Basic cleanup: trim, collapse whitespace, remove zero-width chars."""
    if not text:
        return ""
    text = text.replace("\u200b", "").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text
