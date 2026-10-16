"""ProbEval: Automated LLM Evaluation Suites & Continuous Regression Testing."""

__version__ = "0.1.0"

from probeval.config import TestCase, MetricScore, SampleEvaluation, TestCaseResult, BenchmarkReport, RegressionCheckResult
from probeval.models.simulator import SimulatedLLMProvider
from probeval.runner.suite import EvaluationSuite
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper

__all__ = [
    "TestCase",
    "MetricScore",
    "SampleEvaluation",
    "TestCaseResult",
    "BenchmarkReport",
    "RegressionCheckResult",
    "SimulatedLLMProvider",
    "EvaluationSuite",
    "ContinuousRegressionGatekeeper",
]
