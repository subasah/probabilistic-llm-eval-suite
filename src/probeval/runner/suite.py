"""Evaluation suite runner and report synthesizer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from probeval.config import BenchmarkReport, ConfidenceInterval, TestCase, TestCaseResult
from probeval.evaluators.base import BaseEvaluator
from probeval.models.base import BaseModelProvider
from probeval.runner.engine import EvaluationEngine
from probeval.statistics.bootstrap import bootstrap_confidence_interval
from probeval.statistics.pass_at_k import compute_suite_pass_at_k


class EvaluationSuite:
    """High-level orchestrator for benchmark runs and statistical report generation."""

    def __init__(
        self,
        name: str = "llm-benchmark-suite",
        model: Optional[BaseModelProvider] = None,
        evaluators: Optional[List[BaseEvaluator]] = None,
        samples_per_prompt: int = 5,
        temperature: float = 0.2,
    ):
        self.name = name
        self.model = model
        self.evaluators = evaluators or []
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature

    async def run_async(
        self,
        test_cases: List[TestCase],
        run_id: Optional[str] = None,
    ) -> BenchmarkReport:
        if not self.model:
            raise ValueError("Model provider must be specified to run evaluation suite")

        engine = EvaluationEngine(
            model_provider=self.model,
            evaluators=self.evaluators,
            samples_per_prompt=self.samples_per_prompt,
            temperature=self.temperature,
        )

        results: List[TestCaseResult] = await engine.run_suite_async(test_cases)

        # 1. Bootstrap 95% CI on overall pass rates
        case_pass_rates = [r.pass_rate for r in results]
        overall_ci = bootstrap_confidence_interval(case_pass_rates, n_resamples=1000)

        # 2. Metric summaries with bootstrap CIs
        metric_summary: Dict[str, ConfidenceInterval] = {}
        for ev in self.evaluators:
            metric_vals = [r.metric_means[ev.name] for r in results if ev.name in r.metric_means]
            if metric_vals:
                metric_summary[ev.name] = bootstrap_confidence_interval(metric_vals, n_resamples=1000)

        # 3. Pass@k across tasks
        task_outcomes = [[s.all_passed for s in r.samples] for r in results]
        pass_at_k_metrics = compute_suite_pass_at_k(task_outcomes, k_values=(1, 3, min(5, self.samples_per_prompt)))

        report = BenchmarkReport(
            run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
            model_name=self.model.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_test_cases=len(test_cases),
            samples_per_prompt=self.samples_per_prompt,
            temperature=self.temperature,
            results=results,
            overall_pass_rate=overall_ci,
            metric_summary=metric_summary,
            metadata={
                "pass_at_k": pass_at_k_metrics,
                "suite_name": self.name,
            }
        )
        return report

    def run(self, test_cases: List[TestCase], run_id: Optional[str] = None) -> BenchmarkReport:
        return asyncio.run(self.run_async(test_cases=test_cases, run_id=run_id))
