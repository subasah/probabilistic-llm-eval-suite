"""Generates GitHub Pull Request markdown comment summaries for CI/CD workflows."""

from __future__ import annotations

from typing import Any, Dict, List
from probeval.config import BenchmarkReport, RegressionCheckResult


class GitHubPRCommentGenerator:
    """Formats markdown status report for automated CI/CD comments on Pull Requests."""

    @staticmethod
    def generate_markdown(
        candidate_report: BenchmarkReport,
        regression_analysis: Dict[str, Any],
    ) -> str:
        status = regression_analysis["status"]
        status_emoji = "✅" if status == "PASSED" else ("⚠️" if status == "WARNING" else "🚨")

        ci = candidate_report.overall_pass_rate
        checks: List[RegressionCheckResult] = regression_analysis["checks"]

        md = []
        md.append(f"## {status_emoji} ProbEval Continuous Regression Gate: **{status}**")
        md.append("")
        md.append(
            f"**Candidate Model:** `{candidate_report.model_name}` | "
            f"**Run ID:** `{candidate_report.run_id}` | "
            f"**Prompts Tested:** `{candidate_report.total_test_cases}` (x{candidate_report.samples_per_prompt} samples)"
        )
        md.append("")
        md.append(f"**Overall Pass Rate:** **{ci.mean:.1%}** (95% Bootstrap CI: `[{ci.ci_lower:.1%}, {ci.ci_upper:.1%}]`)")
        md.append("")

        md.append("### Statistical Metric Regression Comparison")
        md.append("")
        md.append("| Metric | Baseline | Candidate | Delta (Δ) | p-value | Significant? | Status |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for c in checks:
            sig_badge = "**YES**" if c.statistically_significant else "No (noise)"
            if c.severity == "CRITICAL":
                verdict = "🔴 **FAIL**"
            elif c.severity == "WARNING":
                verdict = "🟡 **WARN**"
            else:
                verdict = "🟢 **PASS**"

            delta_str = f"{c.delta:+.3f} ({c.delta_percentage:+.1f}%)"
            md.append(
                f"| `{c.metric_name}` | {c.baseline_mean:.3f} | {c.candidate_mean:.3f} | {delta_str} | {c.p_value:.4f} | {sig_badge} | {verdict} |"
            )

        md.append("")
        mcnemar = regression_analysis.get("mcnemar_contingency", {})
        if mcnemar:
            md.append("### McNemar Paired Shift Analysis")
            md.append(
                f"- **Regressions (Baseline Passed -> Candidate Failed):** `{mcnemar.get('regressions', 0)}`\n"
                f"- **Improvements (Baseline Failed -> Candidate Passed):** `{mcnemar.get('improvements', 0)}`\n"
                f"- **Net Gain:** `{mcnemar.get('net_gain', 0):+d}`\n"
                f"- **McNemar p-value:** `{mcnemar.get('p_value', 1.0):.4f}`\n"
            )

        md.append("---")
        md.append("*Generated automatically by [ProbEval](https://github.com/subasah/probabilistic-llm-eval-suite).*")

        return "\n".join(md)
