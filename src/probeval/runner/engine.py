"""Async evaluation execution engine with concurrency control and retries."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from probeval.config import MetricScore, SampleEvaluation, TestCase, TestCaseResult
from probeval.evaluators.base import BaseEvaluator
from probeval.models.base import BaseModelProvider


class EvaluationEngine:
    """Executes asynchronous parallel evaluations with rate limiting and concurrency management."""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        evaluators: List[BaseEvaluator],
        samples_per_prompt: int = 5,
        temperature: float = 0.2,
        max_concurrency: int = 5,
        timeout_seconds: float = 30.0,
    ):
        self.model = model_provider
        self.evaluators = evaluators
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout_seconds = timeout_seconds

    async def _evaluate_single_sample(
        self,
        test_case: TestCase,
        sample_index: int,
    ) -> SampleEvaluation:
        async with self.semaphore:
            start_time = time.perf_counter()
            effective_prompt = (
                f"{test_case.context}\n\nQuestion: {test_case.prompt}"
                if test_case.context and "context" not in test_case.prompt.lower()
                else test_case.prompt
            )
            response = await asyncio.wait_for(
                self.model.generate_async(
                    prompt=effective_prompt,
                    system_prompt=test_case.system_prompt,
                    temperature=self.temperature,
                    tools=test_case.tools,
                ),
                timeout=self.timeout_seconds,
            )
            sample_latency = (time.perf_counter() - start_time) * 1000.0

            # Run all registered evaluators against this sample
            scores: Dict[str, MetricScore] = {}
            all_passed = True

            for ev in self.evaluators:
                score = await ev.evaluate_async(
                    output_text=response.text,
                    test_case=test_case,
                    tool_calls=response.tool_calls,
                    latency_ms=sample_latency,
                )
                scores[ev.name] = score
                if not score.passed:
                    all_passed = False

            return SampleEvaluation(
                sample_index=sample_index,
                raw_output=response.text,
                latency_ms=round(sample_latency, 1),
                tokens_used=response.total_tokens,
                scores=scores,
                all_passed=all_passed,
            )

    async def evaluate_test_case(self, test_case: TestCase) -> TestCaseResult:
        tasks = [
            self._evaluate_single_sample(test_case=test_case, sample_index=i)
            for i in range(self.samples_per_prompt)
        ]
        samples = await asyncio.gather(*tasks)

        # Aggregate sample statistics
        passing_samples = sum(1 for s in samples if s.all_passed)
        pass_rate = passing_samples / len(samples) if samples else 0.0
        mean_latency = sum(s.latency_ms for s in samples) / len(samples) if samples else 0.0

        # Calculate mean for each metric across samples
        metric_means: Dict[str, float] = {}
        for ev in self.evaluators:
            vals = [s.scores[ev.name].score for s in samples if ev.name in s.scores]
            if vals:
                metric_means[ev.name] = round(sum(vals) / len(vals), 4)

        return TestCaseResult(
            test_case_id=test_case.id,
            category=test_case.category,
            prompt=test_case.prompt,
            samples=samples,
            pass_rate=round(pass_rate, 4),
            mean_latency_ms=round(mean_latency, 1),
            metric_means=metric_means,
        )

    async def run_suite_async(self, test_cases: List[TestCase]) -> List[TestCaseResult]:
        tasks = [self.evaluate_test_case(tc) for tc in test_cases]
        return await asyncio.gather(*tasks)
