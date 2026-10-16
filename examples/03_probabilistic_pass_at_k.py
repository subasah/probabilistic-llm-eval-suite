"""Scenario 3: Probabilistic Pass@k and Temperature Sampling Trade-offs.

Demonstrates how Pass@k (Chen et al. 2021) behaves across sampling budgets
and why multi-attempt sampling with Pass@k captures generative capability
far better than naive single-shot evaluations.
"""

from rich.console import Console
from rich.table import Table
from probeval.benchmarks.loader import BenchmarkLoader
from probeval.evaluators.deterministic import RegexEvaluator
from probeval.models.simulator import SimulatedLLMProvider
from probeval.runner.suite import EvaluationSuite


def main():
    console = Console()
    console.print("\n[bold cyan]=== PROBEVAL DEMO: Pass@k Under Temperature Variations ===[/bold cyan]\n")

    test_cases = BenchmarkLoader.load_suite("code_and_reasoning")
    evaluators = [RegexEvaluator(name="code_correctness", must_match=[r"def \w+"], must_not_match=[r"flawed logic"])]

    temperatures = [0.0, 0.4, 0.8]
    summary_data = []

    for temp in temperatures:
        console.print(f"[yellow]Evaluating temperature = {temp} (N = 8 samples per prompt)...[/yellow]")
        model = SimulatedLLMProvider(
            model_name=f"coder-temp-{temp}",
            base_accuracy=0.92,
            seed=42,
        )
        suite = EvaluationSuite(
            name=f"temp_{temp}",
            model=model,
            evaluators=evaluators,
            samples_per_prompt=8,
            temperature=temp,
        )
        report = suite.run(test_cases)
        k_metrics = report.metadata.get("pass_at_k", {})
        summary_data.append({
            "temperature": temp,
            "pass_rate": report.overall_pass_rate.mean,
            "ci": f"[{report.overall_pass_rate.ci_lower:.1%}, {report.overall_pass_rate.ci_upper:.1%}]",
            "pass@1": k_metrics.get("pass@1", 0.0),
            "pass@3": k_metrics.get("pass@3", 0.0),
            "pass@5": k_metrics.get("pass@5", 0.0),
        })

    # Render Comparative Table
    table = Table(title="📈 Pass@k Unbiased Estimation Across Sampling Temperatures", border_style="cyan")
    table.add_column("Temperature", justify="center", style="bold yellow")
    table.add_column("Mean Pass Rate", justify="right")
    table.add_column("95% Bootstrap CI", justify="center", style="green")
    table.add_column("Pass@1", justify="right", style="cyan")
    table.add_column("Pass@3", justify="right", style="cyan")
    table.add_column("Pass@5", justify="right", style="bold cyan")

    for row in summary_data:
        table.add_row(
            str(row["temperature"]),
            f"{row['pass_rate']:.1%}",
            row["ci"],
            f"{row['pass@1']:.1%}",
            f"{row['pass@3']:.1%}",
            f"{row['pass@5']:.1%}",
        )

    console.print(table)
    console.print("\n[dim]Key Insight: At higher temperatures, Pass@1 may decrease due to variance, but Pass@5 increases as sampling explores alternative successful solution paths.[/dim]\n")


if __name__ == "__main__":
    main()
