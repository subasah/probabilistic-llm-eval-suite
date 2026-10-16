"""Unit tests for continuous regression gatekeeper."""

from probeval.config import BenchmarkReport, ConfidenceInterval, TestCaseResult
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper


def _mock_report(run_id: str, pass_rate: float, metric_score: float) -> BenchmarkReport:
    ci = ConfidenceInterval(mean=pass_rate, ci_lower=pass_rate - 0.05, ci_upper=pass_rate + 0.05, std_err=0.02, sample_size=10)
    metric_ci = ConfidenceInterval(mean=metric_score, ci_lower=metric_score - 0.05, ci_upper=metric_score + 0.05, std_err=0.02, sample_size=10)

    # 10 test case results
    results = [
        TestCaseResult(
            test_case_id=f"tc_{i}",
            category="general",
            prompt="test",
            samples=[],
            pass_rate=pass_rate,
            mean_latency_ms=100.0,
            metric_means={"json_schema_valid": metric_score},
        )
        for i in range(10)
    ]

    return BenchmarkReport(
        run_id=run_id,
        model_name="mock-model",
        timestamp="2026-10-16T00:00:00",
        total_test_cases=10,
        samples_per_prompt=5,
        temperature=0.0,
        results=results,
        overall_pass_rate=ci,
        metric_summary={"json_schema_valid": metric_ci},
    )


def test_regression_gate_passes_when_identical():
    gate = ContinuousRegressionGatekeeper()
    b = _mock_report("b1", pass_rate=0.95, metric_score=0.95)
    c = _mock_report("c1", pass_rate=0.95, metric_score=0.95)
    res = gate.evaluate_regression(b, c)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["has_critical_regression"] is False


def test_regression_gate_fails_on_critical_metric_drop():
    gate = ContinuousRegressionGatekeeper(critical_metrics=["json_schema_valid"])
    b = _mock_report("b2", pass_rate=0.95, metric_score=0.95)
    # Severe drop on critical metric json_schema_valid
    c = _mock_report("c2", pass_rate=0.60, metric_score=0.40)
    res = gate.evaluate_regression(b, c)
    assert res["status"] == "FAILED"
    assert res["passed"] is False
    assert res["has_critical_regression"] is True
