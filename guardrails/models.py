from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AbstentionReason(str, Enum):
    EMPTY_QUERY = "EMPTY_QUERY"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    UNSAFE_CONTENT = "UNSAFE_CONTENT"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    OFF_TOPIC = "OFF_TOPIC"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    MALFORMED_CONTEXT = "MALFORMED_CONTEXT"


class InputGuardrailResult(BaseModel):
    passed: bool = Field(description="Whether input query passed all pre-checks")
    violation_type: Optional[str] = Field(default=None, description="Category of input violation")
    message: Optional[str] = Field(default=None, description="Detailed explanation")
    cleaned_query: str = Field(description="Normalized / sanitized query")
    abstention_reason: Optional[AbstentionReason] = Field(default=None)


class ContextGuardrailResult(BaseModel):
    passed: bool = Field(description="Whether retrieved context meets relevance thresholds")
    relevance_score: float = Field(default=0.0, description="Highest candidate relevance or reranker score")
    selected_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Budgeted context chunks")
    abstention_reason: Optional[AbstentionReason] = Field(default=None)
    message: Optional[str] = Field(default=None)


class GroundingCheckResult(BaseModel):
    grounded: bool = Field(description="Whether the answer is substantially supported by retrieved context")
    score: float = Field(default=0.0, description="Grounding confidence score between 0.0 and 1.0")
    supported_claims: List[str] = Field(default_factory=list, description="Verified factual statements")
    unsupported_claims: List[str] = Field(default_factory=list, description="Extrapolated or ungrounded statements")
    missing_citations: List[str] = Field(default_factory=list, description="Claims lacking citations")
    invalid_citations: List[str] = Field(default_factory=list, description="Citations referencing nonexistent chunks")


class PostGuardrailResult(BaseModel):
    passed: bool = Field(description="Whether generated response passed output verification")
    should_retry: bool = Field(default=False, description="Whether harness should trigger 1 regeneration attempt")
    abstention_reason: Optional[AbstentionReason] = Field(default=None)
    grounding_result: Optional[GroundingCheckResult] = Field(default=None)
    message: Optional[str] = Field(default=None)
