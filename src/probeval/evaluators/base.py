"""Base evaluator interface."""

from __future__ import annotations

import abc
from probeval.config import MetricScore, TestCase


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str, weight: float = 1.0, is_critical: bool = False):
        self.name = name
        self.weight = weight
        self.is_critical = is_critical

    @abc.abstractmethod
    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        """Evaluate model output against a test case specification."""
        pass

    def evaluate(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        """Synchronous wrapper for evaluation."""
        import asyncio
        return asyncio.run(self.evaluate_async(output_text, test_case, **kwargs))
