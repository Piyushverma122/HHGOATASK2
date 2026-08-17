import os
import re
import json
import time
import logging
from typing import Dict, Any, Optional, List

import httpx
from pydantic import ValidationError

from generation.base import LLMProvider
from generation.schemas import AnswerResponse, Citation
from generation.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger("voice_rag.generation.model")


class OpenAICompatibleProvider(LLMProvider):
    """
    Production LLM Provider for OpenAI-compatible REST endpoints (Groq, OpenAI, LiteLLM, vLLM, Ollama).
    Endpoint: POST {base_url}/chat/completions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "meta-llama/llama-3.3-70b-instruct",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._base_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self._model = model or "meta-llama/llama-3.3-70b-instruct"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self._model,
            "base_url": self._base_url,
            "has_api_key": bool(self._api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        if not self._api_key:
            raise RuntimeError("LLM_API_KEY is not set.")

        endpoint = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)

                if resp.status_code == 200:
                    resp_json = resp.json()
                    choices = resp_json.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
                    return ""

                if resp.status_code in [401, 403]:
                    raise PermissionError(f"LLM authentication failed (HTTP {resp.status_code}): {resp.text}")

                if resp.status_code == 429:
                    raise RuntimeError(f"LLM rate limit reached (HTTP 429): {resp.text}")

                logger.warning(f"LLM attempt {attempt} returned HTTP {resp.status_code}: {resp.text}")
                last_error = RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text}")

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"LLM network timeout/error on attempt {attempt}: {e}")
                last_error = e

            if attempt < self.max_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))

        if last_error:
            raise last_error
        raise RuntimeError("LLM generation failed after retries.")

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AnswerResponse:
        sys_p = system_prompt or RAG_SYSTEM_PROMPT
        raw_text = self.generate(
            prompt=prompt,
            system_prompt=sys_p,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract JSON from Markdown block if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_text

        try:
            parsed = json.loads(json_str)
            return AnswerResponse.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse structured JSON from LLM: {e}. Raw: {raw_text[:100]}")
            # Fallback to plain answer wrapping
            return AnswerResponse(
                answer=raw_text,
                language="hi",
                grounded=False,
                confidence=0.5,
                citations=[],
                abstained=False,
                abstention_reason=None,
            )


class MockLLMProvider(LLMProvider):
    """
    Deterministic Mock LLM Provider for local development, CI/CD, and offline test environments.
    Extracts answers and citations directly from supplied context text.
    """

    def __init__(self, model_name: str = "mock-grounded-rag-v1"):
        self._model = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self._model,
            "mode": "offline_deterministic_mock",
            "has_api_key": False,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        time.sleep(0.02)  # Simulate 20ms generation latency
        return "स्रोतों के अनुसार भारत की राजधानी नई दिल्ली है।"

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AnswerResponse:
        time.sleep(0.02)

        # Extract all context chunks from prompt
        chunk_pattern = re.compile(
            r"\[CONTEXT CHUNK\s*(\d+)\]\s*\n"
            r"chunk_id:\s*([^\s\n]+)\s*\n"
            r"source_passage_id:\s*([^\s\n]+)\s*\n"
            r"(?:relevance_score:\s*[^\n]+\n)?"
            r"text:\s*(.*?)(?=(?:\n\[CONTEXT CHUNK|\n=== RETRIEVED CONTEXT END|\Z))",
            re.DOTALL,
        )
        chunk_blocks = chunk_pattern.findall(prompt)

        if not chunk_blocks:
            chunk_matches = re.findall(r"chunk_id:\s*([^\s\n]+)", prompt)
            passage_matches = re.findall(r"source_passage_id:\s*([^\s\n]+)", prompt)
            if chunk_matches:
                chunk_blocks = [("1", chunk_matches[0], passage_matches[0] if passage_matches else "pass_1", prompt)]
            else:
                return AnswerResponse(
                    answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                    language="hi",
                    grounded=False,
                    confidence=0.0,
                    citations=[],
                    abstained=True,
                    abstention_reason="INSUFFICIENT_CONTEXT",
                )

        # Extract user query from prompt
        query_match = re.search(r'USER QUERY:\s*\n"([^"]+)"', prompt)
        user_query = query_match.group(1).strip() if query_match else prompt.strip()

        # Linguistic analysis
        from retrieval.query.analyze import analyze_query
        analysis = analyze_query(user_query)
        lang = analysis.language

        # Extract query tokens (length >= 2, excluding very common stop words)
        query_tokens = [
            t.lower() for t in re.findall(r"[\w\u0900-\u0D7F]+", user_query.lower())
            if len(t) >= 2 and t not in {
                "what", "is", "the", "of", "in", "and", "a", "an", "to", "for", "are", "how", "who", "which",
                "क्या", "है", "हैं", "का", "के", "की", "में", "से", "पर", "और", "को", "यह", "वह", "कहाँ", "कौन"
            }
        ]

        best_score = -1.0
        best_sentence = ""
        best_snippet = ""
        best_chunk_id = chunk_blocks[0][1]
        best_passage_id = chunk_blocks[0][2]

        for idx, chunk_id, passage_id, text in chunk_blocks:
            chunk_text = text.strip()
            # Split into individual sentences
            sentences = [s.strip() for s in re.split(r"[।\.\?\!\n]+", chunk_text) if len(s.strip()) > 8]
            for s in sentences:
                s_lower = s.lower()
                # Score based on token overlap
                score = sum(1.0 for qt in query_tokens if qt in s_lower)
                if query_tokens and score > 0:
                    score = score / len(query_tokens)
                if score > best_score:
                    best_score = score
                    best_sentence = s
                    best_snippet = s[:120]
                    best_chunk_id = chunk_id
                    best_passage_id = passage_id

        # If we found a relevant sentence with keyword match
        if best_sentence and best_score > 0.05:
            answer_text = best_sentence
            grounded = True
            confidence = min(0.98, max(0.85, 0.7 + best_score * 0.3))
            abstained = False
            abstention_reason = None
        else:
            # Fallback to top chunk's first sentence
            top_chunk_text = chunk_blocks[0][3].strip()
            top_sentences = [s.strip() for s in re.split(r"[।\.\?\!\n]+", top_chunk_text) if len(s.strip()) > 5]
            if top_sentences:
                best_sentence = top_sentences[0]
                best_snippet = top_sentences[0][:120]
                best_chunk_id = chunk_blocks[0][1]
                best_passage_id = chunk_blocks[0][2]
                answer_text = best_sentence
                grounded = True
                confidence = 0.85
                abstained = False
                abstention_reason = None
            else:
                answer_text = top_chunk_text[:120]
                best_snippet = top_chunk_text[:100]
                best_chunk_id = chunk_blocks[0][1]
                best_passage_id = chunk_blocks[0][2]
                grounded = True
                confidence = 0.75
                abstained = False
                abstention_reason = None

        citations = []
        if grounded and best_chunk_id:
            citations.append(
                Citation(
                    chunk_id=best_chunk_id,
                    source_passage_id=best_passage_id,
                    relevance_score=round(confidence, 3),
                    snippet=best_snippet,
                )
            )

        return AnswerResponse(
            answer=answer_text,
            language=lang,
            grounded=grounded,
            confidence=confidence,
            citations=citations,
            abstained=abstained,
            abstention_reason=abstention_reason,
        )
