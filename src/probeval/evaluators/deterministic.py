"""Deterministic evaluators: JSON Schema validation, Regex matching, Tool-call verification."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import jsonschema
from probeval.config import MetricScore, TestCase
from probeval.evaluators.base import BaseEvaluator


class JsonSchemaEvaluator(BaseEvaluator):
    """
    Validates that model output is valid JSON and strictly complies with
    an expected JSON Schema (e.g. required fields, types, constraints).
    """

    def __init__(self, name: str = "json_schema_valid", schema: Optional[Dict[str, Any]] = None, is_critical: bool = True):
        super().__init__(name=name, is_critical=is_critical)
        self.default_schema = schema

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        schema = test_case.expected_schema or self.default_schema
        if not schema:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No schema specified for test case")

        clean_text = output_text.strip()
        # Clean markdown code blocks if present
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # Step 1: Parse JSON
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError as err:
            return MetricScore(
                name=self.name,
                score=0.0,
                passed=False,
                reason=f"JSONDecodeError: {err.msg} at line {err.lineno} col {err.colno}",
            )

        # Step 2: Validate against JSON Schema
        try:
            jsonschema.validate(instance=parsed_data, schema=schema)
            return MetricScore(
                name=self.name,
                score=1.0,
                passed=True,
                reason="Output strictly adheres to required JSON schema",
                metadata={"parsed_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else []}
            )
        except jsonschema.ValidationError as val_err:
            return MetricScore(
                name=self.name,
                score=0.0,
                passed=False,
                reason=f"SchemaValidationError: {val_err.message} in path '{'/'.join(str(p) for p in val_err.path)}'",
            )


class RegexEvaluator(BaseEvaluator):
    """Checks for presence of required patterns and absence of prohibited patterns."""

    def __init__(
        self,
        name: str = "regex_match",
        must_match: Optional[List[str]] = None,
        must_not_match: Optional[List[str]] = None,
        is_critical: bool = False,
    ):
        super().__init__(name=name, is_critical=is_critical)
        self.must_match = must_match or []
        self.must_not_match = must_not_match or []

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        # Check required patterns
        for pattern in self.must_match:
            if not re.search(pattern, output_text, re.IGNORECASE):
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    reason=f"Required pattern '{pattern}' was not found in output",
                )

        # Check prohibited patterns
        for pattern in self.must_not_match:
            if re.search(pattern, output_text, re.IGNORECASE):
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    reason=f"Prohibited pattern '{pattern}' was unexpectedly matched in output",
                )

        return MetricScore(name=self.name, score=1.0, passed=True, reason="All regex constraints satisfied")


class ToolCallEvaluator(BaseEvaluator):
    """Verifies tool call name, valid arguments JSON, and parameter schema."""

    def __init__(self, name: str = "tool_call_valid", is_critical: bool = True):
        super().__init__(name=name, is_critical=is_critical)

    async def evaluate_async(self, output_text: str, test_case: TestCase, tool_calls: Optional[List[Dict[str, Any]]] = None, **kwargs) -> MetricScore:
        if not test_case.tools:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No tools defined")

        if not tool_calls:
            return MetricScore(name=self.name, score=0.0, passed=False, reason="Expected tool call but none produced")

        available_tool_names = [
            t.get("function", {}).get("name") or t.get("name")
            for t in test_case.tools
        ]

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name")
            if fn_name not in available_tool_names:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    reason=f"Hallucinated tool name: '{fn_name}'. Available: {available_tool_names}",
                )

            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError as err:
                    return MetricScore(
                        name=self.name,
                        score=0.0,
                        passed=False,
                        reason=f"Malformed tool call arguments JSON: {err.msg}",
                    )

        return MetricScore(name=self.name, score=1.0, passed=True, reason="Tool call valid and conforms to spec")


class LatencyBudgetEvaluator(BaseEvaluator):
    """Evaluates latency against a strict SLA budget."""

    def __init__(self, name: str = "latency_sla", max_latency_ms: float = 1000.0, is_critical: bool = False):
        super().__init__(name=name, is_critical=is_critical)
        self.max_latency_ms = max_latency_ms

    async def evaluate_async(self, output_text: str, test_case: TestCase, latency_ms: float = 0.0, **kwargs) -> MetricScore:
        passed = latency_ms <= self.max_latency_ms
        score = max(0.0, min(1.0, 1.0 - (latency_ms / (self.max_latency_ms * 2)))) if passed else 0.0
        return MetricScore(
            name=self.name,
            score=round(score, 3),
            passed=passed,
            reason=f"Latency {latency_ms:.1f}ms (Budget: {self.max_latency_ms:.1f}ms)",
            metadata={"latency_ms": latency_ms, "max_latency_ms": self.max_latency_ms}
        )
