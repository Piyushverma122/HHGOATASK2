import re
import unicodedata


def normalize_query(query: str) -> str:
    """
    Conservative Query Normalization for Multilingual & Indic Search.
    
    Rules:
    - Apply Unicode NFC normalization.
    - Strip leading and trailing whitespace.
    - Collapse redundant consecutive whitespace (spaces, tabs, newlines) into single space.
    - Remove non-printable control characters (except standard punctuation).
    - PRESERVES Indic characters, matras, halants, nuktas, danda (।), numbers, English case, and punctuation.
    - DOES NOT translate or aggressively alter semantic tokens.
    """
    if not query:
        return ""

    # 1. Unicode NFC normalization (canonical composition)
    text = unicodedata.normalize("NFC", str(query))

    # 2. Remove invisible control characters (keep standard whitespace & printable chars)
    # Filter out categories Cc (Control), Cf (Format, except zero-width joiners if necessary)
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cs"):
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # 3. Collapse multiple whitespace characters into single space
    text = re.sub(r"[\s\u200B\uFEFF]+", " ", text)

    # 4. Strip leading/trailing whitespace
    return text.strip()
