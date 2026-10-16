"""Unit tests for deterministic, semantic, and judge evaluators."""

import pytest
from probeval.config import TestCase
from probeval.evaluators.deterministic import JsonSchemaEvaluator, RegexEvaluator, ToolCallEvaluator, LatencyBudgetEvaluator
from probeval.evaluators.semantic import ExactMatchEvaluator, TokenF1Evaluator, CosineSimilarityEvaluator
from probeval.evaluators.rag_triad import FaithfulnessEvaluator


@pytest.mark.asyncio
async def test_json_schema_evaluator_valid():
    evaluator = JsonSchemaEvaluator()
    schema = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {"name": {"type": "string"}, "score": {"type": "number"}},
    }
    tc = TestCase(id="t1", prompt="test", expected_schema=schema)
    res = await evaluator.evaluate_async('{"name": "Alice", "score": 9.5}', tc)
    assert res.passed is True
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_json_schema_evaluator_invalid_missing_key():
    evaluator = JsonSchemaEvaluator()
    schema = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {"name": {"type": "string"}, "score": {"type": "number"}},
    }
    tc = TestCase(id="t2", prompt="test", expected_schema=schema)
    res = await evaluator.evaluate_async('{"name": "Alice"}', tc)
    assert res.passed is False
    assert res.score == 0.0
    assert "SchemaValidationError" in res.reason


@pytest.mark.asyncio
async def test_regex_evaluator():
    evaluator = RegexEvaluator(must_match=[r"\d{3}-\d{4}"], must_not_match=[r"secret"])
    tc = TestCase(id="t3", prompt="test")
    res_pass = await evaluator.evaluate_async("Call 555-1234 for details", tc)
    assert res_pass.passed is True

    res_fail = await evaluator.evaluate_async("Call 555-1234 with secret key", tc)
    assert res_fail.passed is False


@pytest.mark.asyncio
async def test_token_f1_evaluator():
    evaluator = TokenF1Evaluator(pass_threshold=0.6)
    tc = TestCase(id="t4", prompt="test", ground_truth="The quick brown fox jumps over lazy dog")
    res = await evaluator.evaluate_async("A quick brown fox jumped over a lazy dog", tc)
    assert res.passed is True
    assert res.score > 0.6


@pytest.mark.asyncio
async def test_cosine_similarity():
    evaluator = CosineSimilarityEvaluator(pass_threshold=0.5)
    tc = TestCase(id="t5", prompt="test", ground_truth="Bootstrap resampling computes confidence intervals")
    res = await evaluator.evaluate_async("Non-parametric bootstrap calculates confidence intervals", tc)
    assert res.score > 0.5
    assert res.passed is True


@pytest.mark.asyncio
async def test_faithfulness_evaluator_unsupported_claim():
    evaluator = FaithfulnessEvaluator(pass_threshold=0.7)
    tc = TestCase(
        id="t6",
        prompt="Explain quantum mechanics",
        context="Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
    )
    # Output completely unrelated to context
    res = await evaluator.evaluate_async("Quantum computers use entangled qubits to achieve supremacy.", tc)
    assert res.passed is False
    assert res.score < 0.5
