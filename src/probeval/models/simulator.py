"""Realistic LLM Simulator with controllable probabilistic noise and synthetic drift."""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional
from probeval.models.base import BaseModelProvider, ModelResponse


class SimulatedLLMProvider(BaseModelProvider):
    """
    Simulates production LLM behavior with controllable probabilistic noise,
    temperature scaling, latency distributions, and synthetic regression flags.
    Allows zero-cost testing, CI regression validation, and recruiter demonstrations.
    """

    def __init__(
        self,
        model_name: str = "simulated-gpt-4o",
        base_accuracy: float = 0.94,
        regression_factor: float = 0.0,
        mean_latency_ms: float = 180.0,
        latency_std_ms: float = 35.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.base_accuracy = base_accuracy
        self.regression_factor = regression_factor
        self.mean_latency_ms = mean_latency_ms
        self.latency_std_ms = latency_std_ms
        self.rng = random.Random(seed)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        start_time = time.perf_counter()

        # Simulate network & inference latency
        simulated_latency = max(10.0, self.rng.gauss(self.mean_latency_ms, self.latency_std_ms))
        await asyncio.sleep(min(0.05, simulated_latency / 1000.0))  # snappy for CLI, but accurate in metadata

        # Calculate effective success probability based on temperature and regression factor
        # Higher temperature induces higher variance/hallucination rate
        temperature_penalty = (temperature ** 1.3) * 0.12
        effective_acc = max(0.05, min(0.99, self.base_accuracy - self.regression_factor - temperature_penalty))
        is_successful = self.rng.random() < effective_acc

        prompt_lower = prompt.lower()
        tool_calls = None
        output_text = ""

        # Scenario 1: Structured JSON Extraction
        if "extract" in prompt_lower or "json" in prompt_lower or "schema" in prompt_lower:
            if is_successful:
                output_text = json.dumps({
                    "name": "Jane Doe",
                    "role": "Lead AI Architect",
                    "experience_years": 8,
                    "skills": ["LLMOps", "Statistical Evaluation", "Continuous Testing", "Python"],
                    "confidence_score": round(self.rng.uniform(0.91, 0.99), 2),
                    "status": "verified"
                }, indent=2)
            else:
                # Flawed output: schema corruption, missing keys, or syntax error
                corrupt_type = self.rng.choice(["syntax_error", "missing_key", "type_mismatch"])
                if corrupt_type == "syntax_error":
                    output_text = '{"name": "Jane Doe", "role": "Lead AI Architect", "experience_years": 8, '
                elif corrupt_type == "missing_key":
                    output_text = json.dumps({
                        "name": "Jane Doe",
                        "status": "unverified"
                    }, indent=2)
                else:
                    output_text = json.dumps({
                        "name": "Jane Doe",
                        "experience_years": "eight years",  # Type error: string instead of int
                        "skills": "LLMOps, Testing",
                        "status": "verified"
                    }, indent=2)

        # Scenario 2: Tool Calling
        elif tools or "tool" in prompt_lower or "call" in prompt_lower:
            tool_name = tools[0]["function"]["name"] if tools and "function" in tools[0] else "calculate_metrics"
            if is_successful:
                tool_calls = [{
                    "id": f"call_{self.rng.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"dataset": "benchmark_v1", "alpha": 0.05, "iterations": 1000})
                    }
                }]
                output_text = f"Invoking tool {tool_name} with validated parameters."
            else:
                # Hallucinated tool name or invalid JSON arguments
                tool_calls = [{
                    "id": f"call_{self.rng.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": f"wrong_{tool_name}",
                        "arguments": "{bad_json: True"
                    }
                }]
                output_text = "I am attempting to call an unverified tool."

        # Scenario 3: RAG Faithfulness & Grounded Q&A
        elif "context:" in prompt_lower or "rag" in prompt_lower or "ground" in prompt_lower:
            if is_successful:
                output_text = (
                    "Based strictly on the provided context, the primary mechanism for preventing silent "
                    "regressions in probabilistic LLM outputs is continuous bootstrap hypothesis testing "
                    "with predefined significance thresholds (alpha = 0.05)."
                )
            else:
                # Hallucination outside the context
                output_text = (
                    "According to quantum computing principles, the system automatically eliminates "
                    "all non-deterministic noise using universal quantum annealing."
                )

        # Scenario 4: Code & Logic Generation
        elif "def " in prompt_lower or "code" in prompt_lower or "python" in prompt_lower or "function" in prompt_lower:
            if is_successful:
                output_text = (
                    "```python\n"
                    "def calculate_bootstrap_ci(data, n_resamples=1000, ci=0.95):\n"
                    "    import numpy as np\n"
                    "    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]\n"
                    "    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)\n"
                    "    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)\n"
                    "    return float(lower), float(upper)\n"
                    "```"
                )
            else:
                output_text = (
                    "```python\n"
                    "def calculate_bootstrap_ci(data):\n"
                    "    return data[0], data[-1] # Flawed logic: not a bootstrap\n"
                    "```"
                )

        # General Default
        else:
            if is_successful:
                output_text = (
                    "Probabilistic evaluation requires continuous regression testing, "
                    "estimating pass@k variance, and bootstrapping confidence intervals."
                )
            else:
                output_text = "A simple string equality check is sufficient to evaluate any generative model."

        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(output_text.split()) * 2

        return ModelResponse(
            text=output_text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=simulated_latency,
            model=self.model_name,
            metadata={
                "is_simulated": True,
                "effective_accuracy": effective_acc,
                "regression_factor": self.regression_factor,
            }
        )
