import re
from typing import List, Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Centralized token counting and token truncation abstraction.
    Uses tiktoken cl100k_base with a fast whitespace/subword fallback.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoder = None

    def count(self, text: Optional[str]) -> int:
        if not text or not isinstance(text, str):
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        # Fallback estimation for multilingual Indic/Latin text: ~3.5 chars per subword token
        words = text.split()
        return max(1, int(len(text) / 3.5)) if text else 0

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._encoder is not None:
            return self._encoder.encode(text, disallowed_special=())
        # Fallback: character-based token IDs
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        if self._encoder is not None:
            return self._encoder.decode(tokens)
        return "".join([chr(t) for t in tokens if t < 0x110000])

    def truncate(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        if self._encoder is not None:
            tokens = self._encoder.encode(text, disallowed_special=())
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            decoded = self._encoder.decode(truncated_tokens)
            while truncated_tokens and self.count(decoded) > max_tokens:
                truncated_tokens = truncated_tokens[:-1]
                decoded = self._encoder.decode(truncated_tokens)
            return decoded
        # Fallback
        words = text.split()
        approx_words = int(max_tokens * 0.75)
        return " ".join(words[:approx_words])


# Global default token counter instance
token_counter = TokenCounter()


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences preserving Indic (danda ।, double danda ॥)
    and standard Latin sentence terminators (. ? !).
    """
    if not text or not isinstance(text, str):
        return []

    # Regex splits while keeping delimiters attached to preceding sentence
    # Matches: [।॥?!.\n]+
    sentence_endings = re.compile(r'([।॥\n\r]+|[.?!]+(?:\s+|$))')
    parts = sentence_endings.split(text)

    sentences: List[str] = []
    current = ""

    for part in parts:
        if not part:
            continue
        if sentence_endings.match(part):
            current += part
            cleaned = current.strip()
            if cleaned:
                sentences.append(cleaned)
            current = ""
        else:
            current += part

    if current.strip():
        sentences.append(current.strip())

    return sentences if sentences else [text.strip()]


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs by double newlines or major structure boundaries.
    """
    if not text or not isinstance(text, str):
        return []

    paras = re.split(r'\n\s*\n', text)
    cleaned_paras = [p.strip() for p in paras if p.strip()]
    return cleaned_paras if cleaned_paras else [text.strip()]
