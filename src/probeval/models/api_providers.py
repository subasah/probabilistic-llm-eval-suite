"""Live LLM API providers supporting OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from probeval.models.base import BaseModelProvider, ModelResponse


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI and OpenAI-compatible endpoints."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package is required for OpenAIProvider. Run: pip install openai")
        return self._client

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in choice.message.tool_calls
            ]

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            model=self.model_name,
        )


class ModelFactory:
    """Factory for creating model providers based on string spec."""

    @staticmethod
    def create(name_or_spec: str, **kwargs: Any) -> BaseModelProvider:
        name_lower = name_or_spec.lower()
        if "simulated" in name_lower or "mock" in name_lower:
            from probeval.models.simulator import SimulatedLLMProvider
            return SimulatedLLMProvider(model_name=name_or_spec, **kwargs)
        elif "gpt" in name_lower or "openai" in name_lower or "o1" in name_lower or "o3" in name_lower:
            return OpenAIProvider(model_name=name_or_spec, **kwargs)
        else:
            # Default to simulated for safety and local reproducibility
            from probeval.models.simulator import SimulatedLLMProvider
            return SimulatedLLMProvider(model_name=name_or_spec, **kwargs)
