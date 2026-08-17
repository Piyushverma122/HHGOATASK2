import re
import logging
from typing import Optional
from guardrails.models import InputGuardrailResult, AbstentionReason

logger = logging.getLogger("voice_rag.guardrails.input")

# Prompt injection signatures (Multilingual: English, Hindi transliterated, Devanagari)
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output|leak)\s+(your\s+)?(system\s+prompt|instructions|api\s*key|secret)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(dan|developer|god|unrestricted)\s+mode", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(guardrails|safety|filters)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"पिछले\s+(सभी\s+)?निर्देश\s+(भूल|त्याग)\s+जाओ", re.IGNORECASE),
    re.compile(r"सिस्टम\s+प्रॉम्प्ट\s+(दिखाओ|प्रिंट\s+करो|बताओ)", re.IGNORECASE),
    re.compile(r"api\s*key\s+(बताओ|दिखाओ)", re.IGNORECASE),
]

# Unsafe / harmful patterns
UNSAFE_PATTERNS = [
    re.compile(r"\b(how\s+to\s+make|build|synthesize)\s+(a\s+)?(bomb|explosive|biological\s+weapon|poison)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+commit\s+suicide|self-harm)\b", re.IGNORECASE),
    re.compile(r"\b(बम\s+बनाने\s+का\s+तरीका|हथियार\s+बनाना)\b", re.IGNORECASE),
]


class InputGuardrail:
    """
    Validates user input queries prior to retrieval and generation.
    Enforces length limits, prompt injection defenses, and safety policies.
    """

    def __init__(
        self,
        max_query_chars: int = 500,
        enable_injection_defense: bool = True,
    ):
        self.max_query_chars = max_query_chars
        self.enable_injection_defense = enable_injection_defense

    def validate(self, query: Optional[str]) -> InputGuardrailResult:
        if not query or not query.strip():
            return InputGuardrailResult(
                passed=False,
                violation_type="EMPTY_QUERY",
                message="Query cannot be empty or whitespace.",
                cleaned_query="",
                abstention_reason=AbstentionReason.EMPTY_QUERY,
            )

        cleaned = query.strip()

        # 1. Length validation
        if len(cleaned) > self.max_query_chars:
            return InputGuardrailResult(
                passed=False,
                violation_type="QUERY_TOO_LONG",
                message=f"Query exceeds maximum character budget ({len(cleaned)} > {self.max_query_chars} chars).",
                cleaned_query=cleaned[:self.max_query_chars],
                abstention_reason=AbstentionReason.QUERY_TOO_LONG,
            )

        # 2. Prompt injection validation
        if self.enable_injection_defense:
            for pattern in PROMPT_INJECTION_PATTERNS:
                if pattern.search(cleaned):
                    logger.warning(f"Prompt injection attempt detected: '{cleaned[:50]}...'")
                    return InputGuardrailResult(
                        passed=False,
                        violation_type="PROMPT_INJECTION",
                        message="Query contains unauthorized system instruction override or prompt injection attempt.",
                        cleaned_query=cleaned,
                        abstention_reason=AbstentionReason.PROMPT_INJECTION,
                    )

        # 3. Unsafe content validation
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(f"Unsafe query detected: '{cleaned[:50]}...'")
                return InputGuardrailResult(
                    passed=False,
                    violation_type="UNSAFE_CONTENT",
                    message="Query violates safety policy.",
                    cleaned_query=cleaned,
                    abstention_reason=AbstentionReason.UNSAFE_CONTENT,
                )

        return InputGuardrailResult(
            passed=True,
            violation_type=None,
            message=None,
            cleaned_query=cleaned,
            abstention_reason=None,
        )
