"""End-to-end tests for EvaluationEngine and EvaluationSuite."""

import pytest
from probeval.config import TestCase
from probeval.evaluators.deterministic import JsonSchemaEvaluator
from probeval.models.simulator import SimulatedLLMProvider
from probeval.runner.suite import EvaluationSuite


@pytest.mark.asyncio
async def test_evaluation_suite_run_async():
    schema = {
        "type": "object",
        "required": ["name", "role", "experience_years", "skills", "status"],
    }
    tc = TestCase(
        id="test_run_1",
        category="extraction",
        prompt="Extract person profile details into JSON from: Jane Doe is a Lead AI Architect with 8 years of experience in LLMOps and Python.",
        expected_schema=schema,
    )

    model = SimulatedLLMProvider(model_name="test-sim-llm", base_accuracy=0.98, seed=42)
    suite = EvaluationSuite(
        name="test_suite",
        model=model,
        evaluators=[JsonSchemaEvaluator()],
        samples_per_prompt=3,
        temperature=0.1,
    )

    report = await suite.run_async([tc])
    assert report.total_test_cases == 1
    assert report.samples_per_prompt == 3
    assert len(report.results) == 1
    assert report.overall_pass_rate.mean > 0.6
    assert "pass@1" in report.metadata["pass_at_k"]
