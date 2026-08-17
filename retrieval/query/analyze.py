import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from retrieval.query.normalize import normalize_query


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    language: str
    query_length_chars: int
    query_length_words: int
    has_numbers: bool
    numbers: List[str] = Field(default_factory=list)
    question_type: str
    query_type: str
    possible_entities: List[str] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return self.query_length_chars

    @property
    def token_count(self) -> int:
        return self.query_length_words


# Specific keyword sets for robust multilingual matching
QUESTION_KEYWORDS = [
    ("DEFINITION", {"परिभाषा", "अर्थ", "मतलब", "definition", "define", "meaning"}),
    ("WHY", {"कारण", "वजह", "क्यों", "क्यूँ", "why", "reason"}),
    ("PROCEDURAL", {"विधि", "तरीका", "प्रक्रिया", "how", "steps", "process"}),
    ("TEMPORAL", {"कब", "साल", "तारीख", "वर्ष", "दिनांक", "when", "year", "date", "time"}),
    ("LOCATION", {"कहाँ", "कहा", "स्थान", "जगह", "शहर", "देश", "राज्य", "where", "location", "place", "city", "country"}),
    ("PERSON_OR_ORG", {"कौन", "किसने", "किसको", "व्यक्ति", "संस्थापक", "अध्यक्ष", "who", "whom", "whose", "founder", "president", "author"}),
    ("NUMERIC", {"कितना", "कितने", "कितनी", "संख्या", "दूरी", "how many", "how much", "count", "number"}),
    ("STANDARD", {"क्या", "कैसे", "किस", "what", "which"}),
]


def detect_language(text: str) -> str:
    """Fast Unicode range-based language / script identification."""
    if not text:
        return "unknown"

    counts = {
        "hi": len(re.findall(r"[\u0900-\u097F]", text)),  # Devanagari (Hindi/Marathi/Sanskrit/Nepali)
        "bn": len(re.findall(r"[\u0980-\u09FF]", text)),  # Bengali/Assamese
        "ta": len(re.findall(r"[\u0B80-\u0BFF]", text)),  # Tamil
        "te": len(re.findall(r"[\u0C00-\u0C7F]", text)),  # Telugu
        "kn": len(re.findall(r"[\u0C80-\u0CFF]", text)),  # Kannada
        "ml": len(re.findall(r"[\u0D00-\u0D7F]", text)),  # Malayalam
        "gu": len(re.findall(r"[\u0A80-\u0AFF]", text)),  # Gujarati
        "pa": len(re.findall(r"[\u0A00-\u0A7F]", text)),  # Gurmukhi (Punjabi)
        "en": len(re.findall(r"[a-zA-Z]", text)),          # Latin script
    }

    max_lang = max(counts, key=counts.get)
    if counts[max_lang] == 0:
        return "unknown"

    # Hinglish detection heuristic: Latin script with common Hindi phonetic words
    if max_lang == "en":
        hinglish_words = {"kya", "hai", "hain", "kaise", "kab", "kyu", "kyun", "kahan", "ka", "ki", "ke", "ko", "se", "par", "mein", "hota", "hoti", "hote"}
        tokens = set(text.lower().split())
        if len(tokens.intersection(hinglish_words)) >= 2:
            return "hinglish"
        return "en"

    # Marathi detection heuristic in Devanagari script
    if max_lang == "hi":
        marathi_words = {"आहे", "नाही", "आहेत", "काय", "कसे", "कधी", "कुठे", "कोणती", "कोणता", "कोणते", "भारताची", "महाराष्ट्रातील"}
        tokens = set(text.split())
        if len(tokens.intersection(marathi_words)) >= 1:
            return "mr"

    return max_lang


def extract_entities(text: str) -> List[str]:
    """Lightweight rule-based entity candidate extraction (quotes, capitalization, patterns)."""
    entities = []

    # Quoted terms
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    entities.extend(quotes)

    # Capitalized sequences in English
    cap_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
    for cap in cap_phrases:
        if cap.lower() not in {"what", "who", "where", "when", "why", "how", "is", "are", "the", "a", "an"}:
            if cap not in entities:
                entities.append(cap)

    return entities


def analyze_query(query: str) -> QueryAnalysis:
    """
    Fast rule-based Query Analysis (<0.05ms execution).
    """
    norm = normalize_query(query)
    lang = detect_language(norm)
    words = [w.strip("?,.!।;:") for w in norm.split()]
    words_set = set(w.lower() for w in words if w)
    length_chars = len(norm)
    length_words = len(words)

    # Number extraction (Arabic + Indic Devanagari numerals)
    numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b|[\u0966-\u096F]+", norm)
    has_numbers = len(numbers) > 0

    # Question Type Detection
    q_type = "DESCRIPTION"
    for label, kw_set in QUESTION_KEYWORDS:
        if any(kw in words_set or kw in norm.lower() for kw in kw_set):
            q_type = label
            break

    # Determine query_type taxonomy (factoid, narrative, entity, numeric, description)
    if has_numbers or q_type in ("NUMERIC", "TEMPORAL"):
        derived_type = "numeric"
    elif q_type in ("DEFINITION", "PERSON_OR_ORG", "LOCATION"):
        derived_type = "factoid"
    elif length_words > 12 or q_type in ("PROCEDURAL", "WHY"):
        derived_type = "narrative"
    else:
        derived_type = "description"

    entities = extract_entities(query)

    return QueryAnalysis(
        original_query=query,
        normalized_query=norm,
        language=lang,
        query_length_chars=length_chars,
        query_length_words=length_words,
        has_numbers=has_numbers,
        numbers=numbers,
        question_type=q_type,
        query_type=derived_type,
        possible_entities=entities,
    )
