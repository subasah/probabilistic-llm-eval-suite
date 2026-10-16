"""Scenario 1: Prompt Regression Test.

Demonstrates how a prompt update that appears promising on single manual test
actually causes a silent 12% regression in structured JSON extraction.
Runs McNemar's test and bootstrap confidence intervals to catch the degradation.
"""

from probeval.benchmarks.loader import BenchmarkLoader
from probeval.evaluators.deterministic import JsonSchemaEvaluator, LatencyBudgetEvaluator
from probeval.evaluators.semantic import TokenF1Evaluator
from probeval.models.simulator import SimulatedLLMProvider
from probeval.reporting.console import ConsoleReporter
from probeval.reporting.html_generator import HTMLReportGenerator
from probeval.runner.suite import EvaluationSuite
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper


def main():
    print("\n" + "=" * 70)
    print("PROBEVAL DEMO: Prompt Regression Detection (Baseline v1 vs Candidate v2)")
    print("=" * 70 + "\n")

    # 1. Load benchmark dataset
    test_cases = BenchmarkLoader.load_suite("structured_extraction")
    print(f"Loaded {len(test_cases)} structured extraction benchmark prompts.")

    # 2. Register Evaluator Suite
    evaluators = [
        JsonSchemaEvaluator(name="json_schema_valid", is_critical=True),
        TokenF1Evaluator(name="token_f1", pass_threshold=0.60),
        LatencyBudgetEvaluator(name="latency_sla", max_latency_ms=1000.0),
    ]

    # 3. Benchmark Golden Baseline (Prompt v1: Production Standard)
    print("\n[Step 1/3] Benchmarking Production Baseline Prompt v1.0 (5 samples per prompt)...")
    baseline_model = SimulatedLLMProvider(
        model_name="production-prompt-v1.0",
        base_accuracy=0.96,
        regression_factor=0.0,
        seed=101,
    )
    baseline_suite = EvaluationSuite(
        name="baseline_prompt_v1",
        model=baseline_model,
        evaluators=evaluators,
        samples_per_prompt=5,
        temperature=0.2,
    )
    baseline_report = baseline_suite.run(test_cases)

    # 4. Benchmark Candidate Prompt (Prompt v2: New prompt with subtle regression)
    print("\n[Step 2/3] Benchmarking Candidate Prompt v2.0 (with unintended schema jitter)...")
    candidate_model = SimulatedLLMProvider(
        model_name="candidate-prompt-v2.0",
        base_accuracy=0.96,
        regression_factor=0.15,  # Injected regression: 15% drop in accuracy
        seed=202,
    )
    candidate_suite = EvaluationSuite(
        name="candidate_prompt_v2",
        model=candidate_model,
        evaluators=evaluators,
        samples_per_prompt=5,
        temperature=0.2,
    )
    candidate_report = candidate_suite.run(test_cases)

    # 5. Continuous Regression Analysis with Statistical Hypothesis Gating
    print("\n[Step 3/3] Running Statistical Regression Gatekeeper (McNemar + Welch t-test)...")
    gate = ContinuousRegressionGatekeeper(p_value_threshold=0.05, min_effect_size_drop=0.03)
    analysis = gate.evaluate_regression(baseline_report, candidate_report)

    # Display Rich Terminal Output
    reporter = ConsoleReporter()
    reporter.print_benchmark_summary(candidate_report)
    reporter.print_regression_results(analysis)

    # Generate interactive HTML artifact
    html_file = HTMLReportGenerator.generate(
        report=candidate_report,
        regression_analysis=analysis,
        output_path="reports/01_prompt_regression.html",
    )
    print(f"\n[Artifact] Interactive HTML Report saved to: {html_file.resolve()}\n")


if __name__ == "__main__":
    main()
