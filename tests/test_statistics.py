"""Unit tests for statistical algorithms and hypothesis tests."""

import pytest
from probeval.statistics.bootstrap import bootstrap_confidence_interval, wilson_score_interval
from probeval.statistics.pass_at_k import estimate_pass_at_k, compute_suite_pass_at_k
from probeval.statistics.hypothesis_tests import mcnemar_paired_test, compare_continuous_metrics


def test_bootstrap_confidence_interval_bounds():
    data = [0.9, 0.95, 0.92, 0.88, 0.94, 0.91, 0.93]
    ci = bootstrap_confidence_interval(data, n_resamples=1000, confidence_level=0.95, seed=42)
    assert ci.ci_lower <= ci.mean <= ci.ci_upper
    assert 0.85 <= ci.ci_lower <= 0.95
    assert ci.sample_size == len(data)


def test_wilson_score_interval():
    lower, upper = wilson_score_interval(successes=9, total=10, confidence_level=0.95)
    assert 0.0 <= lower <= 0.9 <= upper <= 1.0


def test_estimate_pass_at_k_bounds():
    # If all correct, pass@k should be 1.0
    assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0
    assert estimate_pass_at_k(n=10, c=10, k=5) == 1.0

    # If zero correct, pass@k should be 0.0
    assert estimate_pass_at_k(n=10, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=10, c=0, k=5) == 0.0

    # Intermediate value: n=10, c=2, k=1 -> pass@1 should be c/n = 0.2
    assert estimate_pass_at_k(n=10, c=2, k=1) == 0.2


def test_compute_suite_pass_at_k():
    outcomes = [
        [True, True, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ]
    res = compute_suite_pass_at_k(outcomes, k_values=(1, 2))
    assert "pass@1" in res
    assert "pass@2" in res
    assert res["pass@2"] >= res["pass@1"]


def test_mcnemar_paired_test_identical():
    baseline = [True, True, False, True]
    candidate = [True, True, False, True]
    res = mcnemar_paired_test(baseline, candidate)
    assert res["regressions"] == 0
    assert res["improvements"] == 0
    assert res["p_value"] == 1.0


def test_mcnemar_paired_test_significant_regression():
    # 20 items where baseline passed, candidate failed
    baseline = [True] * 20
    candidate = [False] * 20
    res = mcnemar_paired_test(baseline, candidate)
    assert res["regressions"] == 20
    assert res["p_value"] < 0.001
    assert res["is_significant"] is True
