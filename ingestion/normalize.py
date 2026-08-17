import re
import unicodedata
from typing import Optional


def is_control_char(char: str) -> bool:
    """Check if character is a control character (preserving tab and newline)."""
    if char in ('\n', '\r', '\t'):
        return False
    cat = unicodedata.category(char)
    # Preserve Zero Width Joiner (\u200D) and Zero Width Non-Joiner (\u200C) for Indic ligatures
    if char in ('\u200c', '\u200d'):
        return False
    return cat.startswith('C')


def normalize_text(text: Optional[str]) -> str:
    """
    Carefully normalize textual content:
    1. Unicode NFC normalization (canonical composition for Indic scripts).
    2. Removal of unwanted non-printable control characters.
    3. Normalization of line endings and whitespace.
    4. Collapses multiple spaces, tabs, and redundant line breaks.
    5. Trims leading/trailing whitespace.
    6. Preserves Indic diacritics, nuktas, halant/virama, matras, and punctuation.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Filter out control characters (preserving newlines, tabs, and Indic ZWJ/ZWNJ)
    cleaned_chars = [c for c in normalized if not is_control_char(c)]
    normalized = "".join(cleaned_chars)

    # 3. Standardize line breaks and tabs to spaces
    normalized = normalized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 4. Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    # 5. Trim leading and trailing whitespace
    return normalized.strip()


def is_empty_text(text: Optional[str]) -> bool:
    """Check if text is None, non-string, or empty after normalization."""
    if text is None:
        return True
    if not isinstance(text, str):
        return True
    return len(normalize_text(text)) == 0
