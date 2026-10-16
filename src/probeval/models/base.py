"""Base model provider abstractions."""

from __future__ import annotations

import abc
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """Normalized response from any LLM provider."""
    text: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseModelProvider(abc.ABC):
    """Abstract interface for all model providers."""

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.kwargs = kwargs

    @abc.abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Asynchronously generate a completion for the given prompt."""
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Synchronous wrapper for generate_async."""
        import asyncio
        return asyncio.run(
            self.generate_async(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
        )
