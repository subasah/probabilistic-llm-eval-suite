"""Rich console reporting for evaluation and regression runs."""

from __future__ import annotations

from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from probeval.config import BenchmarkReport, RegressionCheckResult


class ConsoleReporter:
    """Renders formatted tables and regression analysis to stdout using Rich."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_benchmark_summary(self, report: BenchmarkReport) -> None:
        self.console.print("\n")
        self.console.print(
            Panel.fit(
                f"[bold cyan]LLM Probabilistic Benchmark Report[/bold cyan]\n"
                f"Model: [green]{report.model_name}[/green] | Run ID: [yellow]{report.run_id}[/yellow] | "
                f"Samples/Prompt: [magenta]{report.samples_per_prompt}[/magenta] | Temp: [magenta]{report.temperature}[/magenta]",
                border_style="cyan",
            )
        )

        # 1. Overall Metrics Table with Bootstrap 95% CI
        metric_table = Table(title="📊 Metric Performance & Bootstrap 95% Confidence Intervals", border_style="dim")
        metric_table.add_column("Metric", style="cyan", no_wrap=True)
        metric_table.add_column("Mean Score", justify="right")
        metric_table.add_column("95% Bootstrap CI", justify="center", style="green")
        metric_table.add_column("Std Error", justify="right", style="dim")
        metric_table.add_column("N Samples", justify="right")

        # Overall pass rate
        ci = report.overall_pass_rate
        metric_table.add_row(
            "[bold]Overall Pass Rate[/bold]",
            f"[bold]{ci.mean:.1%}[/bold]",
            f"[{ci.ci_lower:.1%}, {ci.ci_upper:.1%}]",
            f"{ci.std_err:.3f}",
            str(ci.sample_size),
        )

        # Individual metrics
        for metric_name, ci_met in sorted(report.metric_summary.items()):
            metric_table.add_row(
                metric_name,
                f"{ci_met.mean:.3f}",
                f"[{ci_met.ci_lower:.3f}, {ci_met.ci_upper:.3f}]",
                f"{ci_met.std_err:.3f}",
                str(ci_met.sample_size),
            )

        self.console.print(metric_table)

        # 2. Pass@k summary if present
        pass_at_k = report.metadata.get("pass_at_k")
        if pass_at_k:
            k_table = Table(title="🎯 Pass@k Unbiased Estimation (Chen et al. 2021)", border_style="dim")
            k_table.add_column("Metric", style="cyan")
            k_table.add_column("Estimated Probability", justify="right", style="bold green")
            for k, val in pass_at_k.items():
                k_table.add_row(k, f"{val:.1%}")
            self.console.print(k_table)

        # 3. Test Cases Breakdown Table
        cases_table = Table(title="🧪 Test Case Breakdown Across Probabilistic Samples", border_style="dim")
        cases_table.add_column("ID", style="dim", width=12)
        cases_table.add_column("Category", style="cyan", width=12)
        cases_table.add_column("Prompt Snippet", style="white", max_width=40)
        cases_table.add_column("Pass Rate", justify="right")
        cases_table.add_column("Latency (ms)", justify="right")

        for tc in report.results:
            p_style = "bold green" if tc.pass_rate >= 0.8 else ("bold yellow" if tc.pass_rate >= 0.5 else "bold red")
            snippet = tc.prompt[:38] + "..." if len(tc.prompt) > 38 else tc.prompt
            cases_table.add_row(
                tc.test_case_id,
                tc.category,
                snippet,
                f"[{p_style}]{tc.pass_rate:.0%}[/{p_style}]",
                f"{tc.mean_latency_ms:.0f}",
            )

        self.console.print(cases_table)
        self.console.print("\n")

    def print_regression_results(self, regression_analysis: Dict[str, Any]) -> None:
        status = regression_analysis["status"]
        if status == "PASSED":
            color = "green"
            badge = "✅ QUALITY GATE PASSED: NO SIGNIFICANT REGRESSION"
        elif status == "WARNING":
            color = "yellow"
            badge = "⚠️  QUALITY GATE WARNING: MINOR DRIFT DETECTED"
        else:
            color = "red"
            badge = "🚨 QUALITY GATE FAILED: CRITICAL REGRESSION DETECTED"

        self.console.print(
            Panel.fit(
                f"[bold {color}]{badge}[/bold {color}]",
                border_style=color,
            )
        )

        reg_table = Table(title="🔍 Continuous Regression Analysis (Hypothesis Testing)", border_style="dim")
        reg_table.add_column("Metric", style="cyan")
        reg_table.add_column("Baseline", justify="right")
        reg_table.add_column("Candidate", justify="right")
        reg_table.add_column("Delta (Δ)", justify="right")
        reg_table.add_column("p-value", justify="right")
        reg_table.add_column("Significant?", justify="center")
        reg_table.add_column("Verdict", style="bold")

        checks: List[RegressionCheckResult] = regression_analysis["checks"]
        for c in checks:
            d_color = "red" if c.delta < 0 else "green"
            sig_badge = "[bold red]YES[/bold red]" if c.statistically_significant else "[dim]No (noise)[/dim]"

            if c.severity == "CRITICAL":
                verdict = "[bold red]🚨 CRITICAL REGRESSION[/bold red]"
            elif c.severity == "WARNING":
                verdict = "[bold yellow]⚠️ WARNING[/bold yellow]"
            else:
                verdict = "[green]✓ SAFE[/green]"

            reg_table.add_row(
                c.metric_name,
                f"{c.baseline_mean:.3f}",
                f"{c.candidate_mean:.3f}",
                f"[{d_color}]{c.delta:+.3f} ({c.delta_percentage:+.1f}%)[/{d_color}]",
                f"{c.p_value:.4f}",
                sig_badge,
                verdict,
            )

        self.console.print(reg_table)

        mcnemar = regression_analysis.get("mcnemar_contingency")
        if mcnemar:
            self.console.print(
                f"[dim]McNemar Paired Analysis: {mcnemar.get('regressions', 0)} regression(s), "
                f"{mcnemar.get('improvements', 0)} improvement(s) across discordant pairs. "
                f"p-value = {mcnemar.get('p_value', 1.0):.4f}[/dim]\n"
            )
