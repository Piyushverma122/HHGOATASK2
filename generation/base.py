from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel
from generation.schemas import AnswerResponse


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Generation Providers.
    Decouples RAG business logic from specific vendors (OpenAI, Groq, Anthropic, Mock).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the active LLM model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials and network endpoints are accessible."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return provider configuration, token limits, and parameters."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate raw text completion."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AnswerResponse:
        """Generate validated, structured AnswerResponse schema."""
        pass
