"""Scenario 4: LLM-as-a-Judge Calibration and Position Bias Mitigation.

Demonstrates how naive pairwise LLM evaluators suffer from Position Bias
(systematically favoring Option A over Option B), and how ProbEval's
bidirectional swapping eliminates false positives.
"""

import asyncio
from rich.console import Console
from rich.table import Table
from probeval.evaluators.llm_judge import PairwiseJudge


async def main_async():
    console = Console()
    console.print("\n[bold cyan]=== PROBEVAL DEMO: LLM Judge Position Bias Mitigation ===[/bold cyan]\n")

    judge = PairwiseJudge()

    # Scenario A: Candidate is genuinely superior
    prompt_1 = "Explain the difference between a process and a thread."
    cand_good = "A process has its own isolated virtual memory space, whereas threads within the same process share memory and heap space."
    base_flawed = "A process uses quantum annealing to execute tasks simultaneously across universes."

    res_a = await judge.compare_async(prompt_1, candidate_output=cand_good, baseline_output=base_flawed)

    # Scenario B: Inconclusive / Tie where naive judge might favor Option A
    prompt_2 = "What is the capital of France?"
    cand_tie = "The capital of France is Paris."
    base_tie = "Paris is the capital of France."

    res_b = await judge.compare_async(prompt_2, candidate_output=cand_tie, baseline_output=base_tie)

    table = Table(title="⚖️ Pairwise Swapped Evaluation Results", border_style="cyan")
    table.add_column("Query", style="white", max_width=30)
    table.add_column("Trial 1 (Cand=A, Base=B)", justify="center")
    table.add_column("Trial 2 (Cand=B, Base=A)", justify="center")
    table.add_column("Position Bias?", justify="center")
    table.add_column("Calibrated Verdict", style="bold green")

    for prompt, res in [(prompt_1, res_a), (prompt_2, res_b)]:
        t1_pick = res["trial_1"]["pick"]
        t2_pick = res["trial_2"]["pick"]
        bias_str = "[bold red]YES[/bold red]" if res["position_bias_detected"] else "[green]None[/green]"
        table.add_row(
            prompt,
            f"Picked: {t1_pick}",
            f"Picked: {t2_pick}",
            bias_str,
            res["winner"],
        )

    console.print(table)
    console.print("\n[dim]Calibrated Rule: Candidate only wins if it defeats Baseline regardless of presentation order (A vs B and B vs A).[/dim]\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
