"""Generates interactive standalone HTML dashboard reports for benchmark evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from probeval.config import BenchmarkReport, RegressionCheckResult


class HTMLReportGenerator:
    """Generates modern, self-contained interactive HTML evaluation reports."""

    @staticmethod
    def generate(
        report: BenchmarkReport,
        regression_analysis: Optional[Dict[str, Any]] = None,
        output_path: str = "reports/eval_report.html",
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        ci = report.overall_pass_rate
        pass_at_k = report.metadata.get("pass_at_k", {})

        metrics_json = json.dumps({
            k: {"mean": v.mean, "lower": v.ci_lower, "upper": v.ci_upper, "stderr": v.std_err}
            for k, v in report.metric_summary.items()
        })

        test_cases_json = json.dumps([
            {
                "id": tc.test_case_id,
                "category": tc.category,
                "prompt": tc.prompt,
                "pass_rate": tc.pass_rate,
                "latency_ms": tc.mean_latency_ms,
                "metrics": tc.metric_means,
                "samples": [
                    {
                        "index": s.sample_index,
                        "output": s.raw_output,
                        "latency_ms": s.latency_ms,
                        "passed": s.all_passed,
                        "scores": {k: {"score": v.score, "passed": v.passed, "reason": v.reason} for k, v in s.scores.items()}
                    }
                    for s in tc.samples
                ]
            }
            for tc in report.results
        ])

        regression_html = ""
        if regression_analysis:
            status = regression_analysis["status"]
            status_color = "#10b981" if status == "PASSED" else ("#f59e0b" if status == "WARNING" else "#ef4444")
            checks: List[RegressionCheckResult] = regression_analysis["checks"]

            rows = ""
            for c in checks:
                delta_class = "text-emerald-400" if c.delta >= 0 else "text-rose-400"
                verdict_badge = (
                    '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/20 text-rose-300">CRITICAL REGRESSION</span>'
                    if c.severity == "CRITICAL"
                    else (
                        '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-300">WARNING</span>'
                        if c.severity == "WARNING"
                        else '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-300">SAFE</span>'
                    )
                )

                rows += f"""
                <tr class="border-b border-slate-800 hover:bg-slate-800/40">
                    <td class="py-3 px-4 font-mono text-cyan-300">{c.metric_name}</td>
                    <td class="py-3 px-4 text-right font-mono text-slate-300">{c.baseline_mean:.3f}</td>
                    <td class="py-3 px-4 text-right font-mono text-slate-300">{c.candidate_mean:.3f}</td>
                    <td class="py-3 px-4 text-right font-mono {delta_class}">{c.delta:+.3f} ({c.delta_percentage:+.1f}%)</td>
                    <td class="py-3 px-4 text-right font-mono text-slate-400">{c.p_value:.4f}</td>
                    <td class="py-3 px-4 text-center">{'<span class="text-rose-400 font-bold">YES</span>' if c.statistically_significant else '<span class="text-slate-500">No (noise)</span>'}</td>
                    <td class="py-3 px-4">{verdict_badge}</td>
                </tr>
                """

            regression_html = f"""
            <div class="mb-10 p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-xl font-bold flex items-center gap-2">
                        <span>Continuous Regression Gate</span>
                    </h2>
                    <span class="px-4 py-1.5 rounded-full text-sm font-bold tracking-wide" style="background-color: {status_color}20; color: {status_color}; border: 1px solid {status_color}50;">
                        GATE STATUS: {status}
                    </span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="text-xs uppercase bg-slate-800/60 text-slate-400">
                            <tr>
                                <th class="py-3 px-4">Metric</th>
                                <th class="py-3 px-4 text-right">Baseline Mean</th>
                                <th class="py-3 px-4 text-right">Candidate Mean</th>
                                <th class="py-3 px-4 text-right">Delta (Δ)</th>
                                <th class="py-3 px-4 text-right">p-value</th>
                                <th class="py-3 px-4 text-center">Significant?</th>
                                <th class="py-3 px-4">Verdict</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ProbEval | LLM Evaluation & Regression Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background-color: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .badge-gradient {{ background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); }}
        pre {{ white-space: pre-wrap; word-break: break-word; }}
    </style>
</head>
<body class="p-6 md:p-12">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-8 mb-8 border-b border-slate-800">
            <div>
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl badge-gradient flex items-center justify-center font-black text-xl text-white shadow-lg">Ψ</div>
                    <div>
                        <h1 class="text-2xl font-black tracking-tight text-white">ProbEval Benchmarking Suite</h1>
                        <p class="text-xs text-slate-400">Continuous Regression & Probabilistic Statistical Verification</p>
                    </div>
                </div>
            </div>
            <div class="flex flex-wrap items-center gap-3 text-xs font-mono">
                <span class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-cyan-400">Model: {report.model_name}</span>
                <span class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300">Run ID: {report.run_id}</span>
                <span class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-400">Temp: {report.temperature} | N={report.samples_per_prompt}</span>
            </div>
        </header>

        <!-- KPI Metric Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            <!-- Overall Pass Rate -->
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Overall Pass Rate (Mean)</p>
                <div class="flex items-baseline gap-2">
                    <span class="text-4xl font-extrabold text-cyan-400">{ci.mean:.1%}</span>
                </div>
                <div class="mt-3 text-xs text-slate-400">
                    <span class="text-emerald-400 font-semibold">95% Bootstrap CI:</span> [{ci.ci_lower:.1%}, {ci.ci_upper:.1%}]
                </div>
                <div class="text-xs text-slate-500 mt-1">Std Error: ±{ci.std_err:.3f} (N={ci.sample_size})</div>
            </div>

            <!-- Pass@1 -->
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Pass@1 (Single Attempt)</p>
                <div class="text-4xl font-extrabold text-blue-400">{pass_at_k.get("pass@1", 0.0):.1%}</div>
                <p class="text-xs text-slate-400 mt-3">Unbiased estimate for k=1 across tasks</p>
                <p class="text-xs text-slate-500 mt-1">First-token accuracy indicator</p>
            </div>

            <!-- Pass@3 / Pass@k -->
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Pass@3 (Multi-Sample)</p>
                <div class="text-4xl font-extrabold text-indigo-400">{pass_at_k.get("pass@3", 0.0):.1%}</div>
                <p class="text-xs text-slate-400 mt-3">Unbiased estimate for k=3 budget</p>
                <p class="text-xs text-slate-500 mt-1">Re-sampling yield factor</p>
            </div>

            <!-- Test Cases -->
            <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Evaluation Scale</p>
                <div class="text-4xl font-extrabold text-emerald-400">{report.total_test_cases}</div>
                <p class="text-xs text-slate-400 mt-3">Total Prompts evaluated</p>
                <p class="text-xs text-slate-500 mt-1">{report.total_test_cases * report.samples_per_prompt} total probabilistic completions</p>
            </div>
        </div>

        <!-- Regression Section (if applicable) -->
        {regression_html}

        <!-- Test Case Explorer -->
        <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <h2 class="text-xl font-bold">Probabilistic Test Cases Explorer</h2>
                <div class="text-xs text-slate-400">Click any row to inspect generated completions & metric breakdowns</div>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm" id="test-cases-table">
                    <thead class="text-xs uppercase bg-slate-800/60 text-slate-400">
                        <tr>
                            <th class="py-3 px-4">Test Case ID</th>
                            <th class="py-3 px-4">Category</th>
                            <th class="py-3 px-4">Prompt</th>
                            <th class="py-3 px-4 text-right">Pass Rate</th>
                            <th class="py-3 px-4 text-right">Mean Latency</th>
                        </tr>
                    </thead>
                    <tbody id="test-cases-body">
                        <!-- Rendered by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Detail Modal -->
        <div id="sample-modal" class="hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
            <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
                <div class="p-5 border-b border-slate-800 flex justify-between items-center">
                    <h3 id="modal-title" class="font-bold text-lg text-cyan-400">Test Case Inspection</h3>
                    <button onclick="closeModal()" class="text-slate-400 hover:text-white text-xl">&times;</button>
                </div>
                <div id="modal-body" class="p-6 overflow-y-auto space-y-4 text-sm">
                    <!-- Dynamic details -->
                </div>
            </div>
        </div>

        <footer class="mt-12 pt-6 border-t border-slate-800 text-center text-xs text-slate-500">
            Generated with <a href="https://github.com/subasah/probabilistic-llm-eval-suite" class="text-cyan-400 hover:underline">ProbEval</a> &bull; Production LLM Evaluation & Continuous Regression Testing
        </footer>
    </div>

    <script>
        const testCases = {test_cases_json};

        function renderTable() {{
            const tbody = document.getElementById("test-cases-body");
            tbody.innerHTML = testCases.map(tc => {{
                const pColor = tc.pass_rate >= 0.8 ? "text-emerald-400" : (tc.pass_rate >= 0.5 ? "text-amber-400" : "text-rose-400");
                const snippet = tc.prompt.length > 55 ? tc.prompt.substring(0, 52) + "..." : tc.prompt;
                return `
                    <tr class="border-b border-slate-800 hover:bg-slate-800/40 cursor-pointer" onclick="showModal('${{tc.id}}')">
                        <td class="py-3 px-4 font-mono font-semibold text-cyan-300">${{tc.id}}</td>
                        <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-xs bg-slate-800 text-slate-300 font-mono">${{tc.category}}</span></td>
                        <td class="py-3 px-4 text-slate-300">${{snippet}}</td>
                        <td class="py-3 px-4 text-right font-mono font-bold ${{pColor}}">${{(tc.pass_rate * 100).toFixed(0)}}%</td>
                        <td class="py-3 px-4 text-right font-mono text-slate-400">${{tc.latency_ms.toFixed(0)}} ms</td>
                    </tr>
                `;
            }}).join('');
        }}

        function showModal(tcId) {{
            const tc = testCases.find(t => t.id === tcId);
            if (!tc) return;

            document.getElementById("modal-title").innerText = `${{tc.id}} (${{tc.category}})`;
            let content = `
                <div>
                    <h4 class="text-xs uppercase font-semibold text-slate-400 mb-1">Prompt</h4>
                    <p class="p-3 bg-slate-800/60 rounded-lg text-slate-200">${{tc.prompt}}</p>
                </div>
                <div>
                    <h4 class="text-xs uppercase font-semibold text-slate-400 mb-2">Probabilistic Samples (${{tc.samples.length}} trials)</h4>
                    <div class="space-y-3">
            `;

            tc.samples.forEach(s => {{
                const badge = s.passed ? '<span class="text-emerald-400 font-bold">PASSED</span>' : '<span class="text-rose-400 font-bold">FAILED</span>';
                content += `
                    <div class="p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-mono text-cyan-400 font-bold">Sample #${{s.index + 1}} (${{s.latency_ms}} ms)</span>
                            <span>${{badge}}</span>
                        </div>
                        <pre class="p-3 bg-slate-950 rounded text-xs font-mono text-slate-300 overflow-x-auto">${{s.output}}</pre>
                    </div>
                `;
            }});

            content += `</div></div>`;
            document.getElementById("modal-body").innerHTML = content;
            document.getElementById("sample-modal").classList.remove("hidden");
        }}

        function closeModal() {{
            document.getElementById("sample-modal").classList.add("hidden");
        }}

        renderTable();
    </script>
</body>
</html>
"""

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out_file
