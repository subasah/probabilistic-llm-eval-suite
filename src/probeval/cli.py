"""Command-line interface for ProbEval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from probeval.benchmarks.loader import BenchmarkLoader
from probeval.config import BenchmarkReport
from probeval.evaluators.deterministic import JsonSchemaEvaluator, ToolCallEvaluator, LatencyBudgetEvaluator
from probeval.evaluators.semantic import TokenF1Evaluator, CosineSimilarityEvaluator
from probeval.evaluators.rag_triad import FaithfulnessEvaluator
from probeval.models.api_providers import ModelFactory
from probeval.models.simulator import SimulatedLLMProvider
from probeval.reporting.console import ConsoleReporter
from probeval.reporting.html_generator import HTMLReportGenerator
from probeval.runner.suite import EvaluationSuite
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper

app = typer.Typer(
    name="probeval",
    help="ProbEval: Automated Evaluation Suites & Continuous Regression Testing for Probabilistic LLM Systems",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    benchmark: str = typer.Option("rag_faithfulness", "--benchmark", "-b", help="Benchmark name or path to jsonl"),
    model: str = typer.Option("simulated-gpt-4o", "--model", "-m", help="Model name or spec"),
    samples: int = typer.Option(5, "--samples", "-n", help="Probabilistic samples per prompt"),
    temperature: float = typer.Option(0.2, "--temperature", "-t", help="Sampling temperature"),
    output_html: str = typer.Option("reports/eval_report.html", "--html", help="Path to write HTML report"),
    output_json: Optional[str] = typer.Option(None, "--json", help="Path to write JSON report"),
) -> None:
    """Run an automated evaluation suite on a specified benchmark."""
    console.print(f"[cyan]Loading benchmark:[/cyan] [bold]{benchmark}[/bold]...")
    test_cases = BenchmarkLoader.load_suite(benchmark)

    console.print(f"[cyan]Initializing model provider:[/cyan] [bold]{model}[/bold]...")
    model_provider = ModelFactory.create(model)

    evaluators = [
        JsonSchemaEvaluator(),
        ToolCallEvaluator(),
        TokenF1Evaluator(pass_threshold=0.60),
        CosineSimilarityEvaluator(pass_threshold=0.60),
        FaithfulnessEvaluator(pass_threshold=0.70),
        LatencyBudgetEvaluator(max_latency_ms=1000.0),
    ]

    suite = EvaluationSuite(
        name=benchmark,
        model=model_provider,
        evaluators=evaluators,
        samples_per_prompt=samples,
        temperature=temperature,
    )

    console.print(f"[yellow]Executing evaluation suite ({len(test_cases)} prompts x {samples} samples = {len(test_cases) * samples} completions)...[/yellow]")
    report = suite.run(test_cases)

    reporter = ConsoleReporter(console)
    reporter.print_benchmark_summary(report)

    # Save HTML report
    html_path = HTMLReportGenerator.generate(report, output_path=output_html)
    console.print(f"[green]✓ Interactive HTML report generated at:[/green] [bold]{html_path}[/bold]")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        console.print(f"[green]✓ JSON report saved to:[/green] [bold]{out_p}[/bold]")


@app.command()
def compare(
    benchmark: str = typer.Option("structured_extraction", "--benchmark", "-b", help="Benchmark name or jsonl path"),
    candidate_regression: float = typer.Option(0.12, "--regression-factor", "-r", help="Simulated regression factor for candidate"),
    samples: int = typer.Option(5, "--samples", "-n", help="Probabilistic samples per prompt"),
    output_html: str = typer.Option("reports/regression_report.html", "--html", help="Path to HTML report"),
) -> None:
    """Compare candidate model against golden baseline with statistical regression gating."""
    console.print(f"[cyan]Loading benchmark:[/cyan] [bold]{benchmark}[/bold]...")
    test_cases = BenchmarkLoader.load_suite(benchmark)

    evaluators = [
        JsonSchemaEvaluator(),
        TokenF1Evaluator(pass_threshold=0.65),
        LatencyBudgetEvaluator(max_latency_ms=1000.0),
    ]

    console.print("[dim]Evaluating golden baseline...[/dim]")
    baseline_model = SimulatedLLMProvider(model_name="baseline-v1.0", base_accuracy=0.96, regression_factor=0.0)
    baseline_suite = EvaluationSuite(name=f"{benchmark}_baseline", model=baseline_model, evaluators=evaluators, samples_per_prompt=samples)
    baseline_report = baseline_suite.run(test_cases)

    console.print(f"[dim]Evaluating candidate release (regression factor: {candidate_regression:.2f})...[/dim]")
    candidate_model = SimulatedLLMProvider(model_name="candidate-v1.1", base_accuracy=0.96, regression_factor=candidate_regression)
    candidate_suite = EvaluationSuite(name=f"{benchmark}_candidate", model=candidate_model, evaluators=evaluators, samples_per_prompt=samples)
    candidate_report = candidate_suite.run(test_cases)

    gate = ContinuousRegressionGatekeeper(p_value_threshold=0.05, min_effect_size_drop=0.03)
    analysis = gate.evaluate_regression(baseline_report, candidate_report)

    reporter = ConsoleReporter(console)
    reporter.print_regression_results(analysis)

    html_path = HTMLReportGenerator.generate(candidate_report, regression_analysis=analysis, output_path=output_html)
    console.print(f"[green]✓ Regression Dashboard generated at:[/green] [bold]{html_path}[/bold]")


@app.command()
def demo() -> None:
    """Quick demonstration of continuous regression detection and non-determinism."""
    console.print("\n[bold magenta]=== ProbEval Live Demonstration ===[/bold magenta]\n")
    compare(benchmark="structured_extraction", candidate_regression=0.15, samples=6, output_html="reports/demo_report.html")


if __name__ == "__main__":
    app()
