"""Scenario 2: Model Migration Benchmark.

Compares a legacy enterprise model against a candidate high-throughput model
across RAG faithfulness, tool calling accuracy, and latency SLAs.
"""

from probeval.benchmarks.loader import BenchmarkLoader
from probeval.evaluators.deterministic import ToolCallEvaluator, LatencyBudgetEvaluator
from probeval.evaluators.rag_triad import FaithfulnessEvaluator, AnswerRelevanceEvaluator
from probeval.evaluators.semantic import TokenF1Evaluator
from probeval.models.simulator import SimulatedLLMProvider
from probeval.reporting.console import ConsoleReporter
from probeval.reporting.html_generator import HTMLReportGenerator
from probeval.runner.suite import EvaluationSuite
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper


def main():
    print("\n" + "=" * 70)
    print("PROBEVAL DEMO: Model Migration Benchmark (Legacy vs Next-Gen Model)")
    print("=" * 70 + "\n")

    # Load composite RAG and Tool benchmark cases
    rag_cases = BenchmarkLoader.load_suite("rag_faithfulness")
    tool_cases = BenchmarkLoader.load_suite("tool_calling")
    combined_cases = rag_cases + tool_cases

    evaluators = [
        FaithfulnessEvaluator(name="rag_faithfulness", pass_threshold=0.75, is_critical=True),
        AnswerRelevanceEvaluator(name="rag_answer_relevance", pass_threshold=0.60),
        ToolCallEvaluator(name="tool_call_valid", is_critical=True),
        LatencyBudgetEvaluator(name="latency_sla", max_latency_ms=250.0),
    ]

    # Model A: Legacy Model (High accuracy, High latency ~350ms)
    print("[1/3] Benchmarking Legacy Enterprise Model (gpt-4-legacy)...")
    legacy_model = SimulatedLLMProvider(
        model_name="gpt-4-legacy",
        base_accuracy=0.95,
        regression_factor=0.0,
        mean_latency_ms=320.0,
        latency_std_ms=40.0,
        seed=42,
    )
    legacy_suite = EvaluationSuite(
        name="legacy_model_benchmark",
        model=legacy_model,
        evaluators=evaluators,
        samples_per_prompt=4,
        temperature=0.1,
    )
    legacy_report = legacy_suite.run(combined_cases)

    # Model B: Candidate Next-Gen Model (Ultra fast ~90ms, comparable accuracy)
    print("\n[2/3] Benchmarking Candidate Next-Gen Model (fast-inference-v2)...")
    candidate_model = SimulatedLLMProvider(
        model_name="fast-inference-v2",
        base_accuracy=0.94,  # Within 1% parity
        regression_factor=0.0,
        mean_latency_ms=85.0,  # 4x faster!
        latency_std_ms=15.0,
        seed=84,
    )
    candidate_suite = EvaluationSuite(
        name="candidate_model_benchmark",
        model=candidate_model,
        evaluators=evaluators,
        samples_per_prompt=4,
        temperature=0.1,
    )
    candidate_report = candidate_suite.run(combined_cases)

    # Statistical Comparison
    print("\n[3/3] Evaluating Migration Viability & Regression Gates...")
    gate = ContinuousRegressionGatekeeper(p_value_threshold=0.05, min_effect_size_drop=0.04)
    analysis = gate.evaluate_regression(legacy_report, candidate_report)

    reporter = ConsoleReporter()
    reporter.print_benchmark_summary(candidate_report)
    reporter.print_regression_results(analysis)

    html_file = HTMLReportGenerator.generate(
        report=candidate_report,
        regression_analysis=analysis,
        output_path="reports/02_model_migration.html",
    )
    print(f"\n[Artifact] Interactive HTML Report saved to: {html_file.resolve()}\n")


if __name__ == "__main__":
    main()
