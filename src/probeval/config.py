"""Configuration models and data structures for probabilistic evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """Defines a single test case in an LLM benchmark."""
    __test__ = False
    id: str = Field(..., description="Unique test case identifier")
    category: str = Field("general", description="Evaluation category (e.g. extraction, rag, tool)")
    prompt: str = Field(..., description="User prompt or input text")
    system_prompt: Optional[str] = Field(None, description="System instructions")
    context: Optional[str] = Field(None, description="Retrieved context (for RAG)")
    ground_truth: Optional[str] = Field(None, description="Target reference or canonical response")
    expected_schema: Optional[Dict[str, Any]] = Field(None, description="Expected JSON Schema for structured outputs")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions available to the model")
    criteria: Optional[Dict[str, str]] = Field(None, description="Specific rubrics for LLM-as-a-judge")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")


class MetricScore(BaseModel):
    """Represents the outcome of a single evaluation metric."""
    name: str
    score: float
    passed: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SampleEvaluation(BaseModel):
    """Evaluation result for one individual generated sample."""
    sample_index: int
    raw_output: str
    latency_ms: float
    tokens_used: int = 0
    scores: Dict[str, MetricScore] = Field(default_factory=dict)
    all_passed: bool = False


class TestCaseResult(BaseModel):
    """Aggregated evaluation across all probabilistic samples for a single test case."""
    __test__ = False
    test_case_id: str
    category: str
    prompt: str
    samples: List[SampleEvaluation]
    pass_rate: float
    mean_latency_ms: float
    metric_means: Dict[str, float] = Field(default_factory=dict)


class ConfidenceInterval(BaseModel):
    """Parametric or Bootstrap 95% Confidence Interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    std_err: float
    sample_size: int


class BenchmarkReport(BaseModel):
    """Complete summary of a benchmark evaluation run."""
    run_id: str
    model_name: str
    timestamp: str
    total_test_cases: int
    samples_per_prompt: int
    temperature: float
    results: List[TestCaseResult]
    overall_pass_rate: ConfidenceInterval
    metric_summary: Dict[str, ConfidenceInterval]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegressionCheckResult(BaseModel):
    """Statistical regression comparison between baseline and candidate runs."""
    metric_name: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    delta_percentage: float
    p_value: float
    statistically_significant: bool
    is_regression: bool
    severity: str  # "NONE", "WARNING", "CRITICAL"
    reason: str
