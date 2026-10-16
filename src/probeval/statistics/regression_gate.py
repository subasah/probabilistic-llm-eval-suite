"""Continuous regression gatekeeper for CI/CD pipelines."""

from __future__ import annotations

from typing import Dict, List, Optional
from probeval.config import BenchmarkReport, RegressionCheckResult
from probeval.statistics.hypothesis_tests import mcnemar_paired_test, compare_continuous_metrics


class ContinuousRegressionGatekeeper:
    """
    Evaluates whether candidate model changes represent a genuine regression
    versus acceptable probabilistic sampling noise.
    Used as an automated quality gate in continuous delivery (CI/CD) pipelines.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        min_effect_size_drop: float = 0.03,  # 3% drop minimum to trigger regression
        critical_metrics: Optional[List[str]] = None,
    ):
        self.p_value_threshold = p_value_threshold
        self.min_effect_size_drop = min_effect_size_drop
        self.critical_metrics = set(critical_metrics or ["json_schema_valid", "rag_faithfulness", "tool_call_valid"])

    def evaluate_regression(
        self,
        baseline_report: BenchmarkReport,
        candidate_report: BenchmarkReport,
    ) -> Dict[str, Any]:
        """
        Compares baseline and candidate reports across all evaluated metrics.
        Returns regression analysis and exit gate status.
        """
        results: List[RegressionCheckResult] = []
        has_critical_regression = False
        has_warning_regression = False

        # 1. Overall Pass Rate Check (McNemar Paired Test across test cases)
        baseline_case_passes = [tc.pass_rate >= 0.80 for tc in baseline_report.results]
        candidate_case_passes = [tc.pass_rate >= 0.80 for tc in candidate_report.results]

        min_len = min(len(baseline_case_passes), len(candidate_case_passes))
        mcnemar = mcnemar_paired_test(baseline_case_passes[:min_len], candidate_case_passes[:min_len])

        base_overall = baseline_report.overall_pass_rate.mean
        cand_overall = candidate_report.overall_pass_rate.mean
        overall_delta = cand_overall - base_overall

        is_overall_regression = (
            overall_delta < -self.min_effect_size_drop and mcnemar["p_value"] < self.p_value_threshold
        )

        overall_severity = "NONE"
        if is_overall_regression:
            overall_severity = "CRITICAL"
            has_critical_regression = True
        elif overall_delta < 0 and mcnemar["p_value"] < self.p_value_threshold:
            overall_severity = "WARNING"
            has_warning_regression = True

        overall_check = RegressionCheckResult(
            metric_name="overall_pass_rate",
            baseline_mean=base_overall,
            candidate_mean=cand_overall,
            delta=round(overall_delta, 4),
            delta_percentage=round((overall_delta / base_overall * 100) if base_overall > 0 else 0.0, 2),
            p_value=mcnemar["p_value"],
            statistically_significant=mcnemar["is_significant"],
            is_regression=is_overall_regression,
            severity=overall_severity,
            reason=(
                f"Overall pass rate shift: {base_overall:.1%} -> {cand_overall:.1%} "
                f"(Δ = {overall_delta:+.1%}, p = {mcnemar['p_value']:.4f}, regressions={mcnemar['regressions']}, improvements={mcnemar['improvements']})"
            ),
        )
        results.append(overall_check)

        # 2. Individual Metric Breakdown
        common_metrics = set(baseline_report.metric_summary.keys()) & set(candidate_report.metric_summary.keys())

        for metric in sorted(common_metrics):
            b_summary = baseline_report.metric_summary[metric]
            c_summary = candidate_report.metric_summary[metric]

            b_scores = [tc.metric_means.get(metric, 0.0) for tc in baseline_report.results if metric in tc.metric_means]
            c_scores = [tc.metric_means.get(metric, 0.0) for tc in candidate_report.results if metric in tc.metric_means]

            comp = compare_continuous_metrics(b_scores, c_scores)
            delta = comp["delta"]
            p_val = comp["p_value"]
            is_sig = comp["is_significant"]

            is_drop = delta < -self.min_effect_size_drop and is_sig
            severity = "NONE"

            if is_drop:
                if metric in self.critical_metrics:
                    severity = "CRITICAL"
                    has_critical_regression = True
                else:
                    severity = "WARNING"
                    has_warning_regression = True

            reason = (
                f"Statistically significant drop in {metric} (Δ = {delta:+.3f}, p = {p_val:.4f})"
                if is_drop
                else (
                    f"Within expected probabilistic noise floor (Δ = {delta:+.3f}, p = {p_val:.4f})"
                    if delta < 0
                    else f"Improvement or steady (Δ = {delta:+.3f})"
                )
            )

            results.append(
                RegressionCheckResult(
                    metric_name=metric,
                    baseline_mean=b_summary.mean,
                    candidate_mean=c_summary.mean,
                    delta=delta,
                    delta_percentage=comp["delta_percent"],
                    p_value=p_val,
                    statistically_significant=is_sig,
                    is_regression=is_drop,
                    severity=severity,
                    reason=reason,
                )
            )

        gate_status = "FAILED" if has_critical_regression else ("WARNING" if has_warning_regression else "PASSED")

        return {
            "status": gate_status,
            "passed": not has_critical_regression,
            "has_critical_regression": has_critical_regression,
            "has_warning_regression": has_warning_regression,
            "checks": results,
            "mcnemar_contingency": mcnemar,
        }
