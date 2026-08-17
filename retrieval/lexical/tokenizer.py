import re
import unicodedata
from typing import List, Set


class MultilingualTokenizer:
    """
    Production-grade Multilingual Tokenizer for BM25 and Lexical Search.
    Handles Indic (Hindi, Bengali, Tamil, Telugu, Marathi), English, and Hinglish.
    
    Features:
    - Unicode NFC normalization.
    - Token extraction with punctuation stripping.
    - English lowercasing.
    - Character 3-gram and 4-gram subword generation for morphologically rich Indic words
      (e.g., handles Hindi verb/noun inflections like शक्तियों -> शक्ति).
    - Preserves numbers and alphanumeric entities.
    - Deterministic and fast.
    """

    # Indic Unicode block ranges: Devanagari (\u0900-\u097F), Bengali (\u0980-\u09FF), Tamil (\u0B80-\u0BFF), Telugu (\u0C00-\u0C7F), etc.
    INDIC_CHAR_PATTERN = re.compile(r"[\u0900-\u0D7F]")
    TOKEN_SPLIT_PATTERN = re.compile(r"[^\w\u0900-\u0D7F]+", re.UNICODE)

    def __init__(self, use_subwords: bool = True, min_subword_len: int = 3, max_subword_len: int = 4):
        self.use_subwords = use_subwords
        self.min_subword_len = min_subword_len
        self.max_subword_len = max_subword_len

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into a list of normalized lexical tokens and subwords.
        """
        if not text:
            return []

        # 1. Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", str(text))

        # 2. Split on non-alphanumeric / non-Indic characters
        raw_tokens = self.TOKEN_SPLIT_PATTERN.split(normalized)

        tokens: List[str] = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue

            # Lowercase Latin characters
            if not self.INDIC_CHAR_PATTERN.search(t):
                t = t.lower()

            # Add primary word token
            tokens.append(t)

            # For Indic tokens > 3 chars, generate character n-grams to bridge inflections
            if self.use_subwords and self.INDIC_CHAR_PATTERN.search(t) and len(t) >= 4:
                for n in range(self.min_subword_len, min(self.max_subword_len + 1, len(t))):
                    for i in range(len(t) - n + 1):
                        ngram = t[i:i + n]
                        tokens.append(f"#{ngram}")

        return tokens
