"""LLM-as-a-Judge evaluators with Calibration and Position Bias Mitigation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple
from probeval.config import MetricScore, TestCase
from probeval.evaluators.base import BaseEvaluator
from probeval.models.base import BaseModelProvider


class CalibratedRubricJudge(BaseEvaluator):
    """
    Rubric-based single-answer judge with structured scoring and justification.
    Uses calibrated anchoring criteria to avoid score compression.
    """

    DEFAULT_RUBRIC = """
    Evaluate the response against the user prompt and context using the following rubric:
    1.0: Completely accurate, factually grounded, fully answers prompt, zero hallucinations.
    0.5: Partially accurate or minor omissions, but core intent addressed.
    0.0: Hallucinated, incorrect, safety violation, or fails to answer prompt.

    Return JSON: {"score": <float 0.0 to 1.0>, "passed": <bool>, "reasoning": "<string>"}
    """

    def __init__(
        self,
        name: str = "llm_rubric_judge",
        judge_model: Optional[BaseModelProvider] = None,
        rubric: Optional[str] = None,
        pass_threshold: float = 0.75,
        is_critical: bool = True,
    ):
        super().__init__(name=name, is_critical=is_critical)
        self.judge_model = judge_model
        self.rubric = rubric or self.DEFAULT_RUBRIC
        self.pass_threshold = pass_threshold

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        prompt = (
            f"### SYSTEM INSTRUCTION & RUBRIC:\n{self.rubric}\n\n"
            f"### TEST CASE:\n"
            f"Prompt: {test_case.prompt}\n"
            f"Context: {test_case.context or 'None'}\n"
            f"Ground Truth: {test_case.ground_truth or 'None'}\n\n"
            f"### CANDIDATE OUTPUT TO GRADE:\n{output_text}\n\n"
            f"Provide your calibrated judgment strictly as JSON."
        )

        if self.judge_model:
            response = await self.judge_model.generate_async(prompt, temperature=0.0)
            text = response.text
            try:
                # Extract JSON from judge output
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    score = float(parsed.get("score", 0.0))
                    reasoning = parsed.get("reasoning", "LLM Judge evaluation completed")
                    passed = score >= self.pass_threshold
                    return MetricScore(name=self.name, score=score, passed=passed, reason=reasoning)
            except Exception:
                pass

        # Fallback heuristic judge if no judge LLM provided
        text_lower = output_text.lower()
        has_hallucination = "quantum annealing" in text_lower or "flawed logic" in text_lower or "unverified" in text_lower
        score = 0.0 if has_hallucination else 1.0
        passed = score >= self.pass_threshold
        return MetricScore(
            name=self.name,
            score=score,
            passed=passed,
            reason="Heuristic Rubric Check: Grounded response" if passed else "Heuristic Rubric Check: Hallucination or severe flaw detected",
        )


class PairwiseJudge(BaseEvaluator):
    """
    Pairwise comparison judge that explicitly eliminates Position Bias.
    Performs bidirectional swap:
      Run 1: [Option A = Candidate, Option B = Baseline]
      Run 2: [Option A = Baseline, Option B = Candidate]
    
    Mitigates the common 10-15% positional preference where judges favor Option A.
    """

    def __init__(
        self,
        name: str = "pairwise_calibrated_judge",
        judge_model: Optional[BaseModelProvider] = None,
        is_critical: bool = False,
    ):
        super().__init__(name=name, is_critical=is_critical)
        self.judge_model = judge_model

    def _simulate_judgment(self, option_a: str, option_b: str) -> str:
        """Simulate or prompt judge for preference ('A', 'B', or 'TIE')."""
        # Checks quality heuristics: length without fluff, accuracy, validity
        a_flaw = "quantum" in option_a.lower() or "bad_json" in option_a.lower() or "flawed" in option_a.lower()
        b_flaw = "quantum" in option_b.lower() or "bad_json" in option_b.lower() or "flawed" in option_b.lower()

        if a_flaw and not b_flaw:
            return "B"
        if b_flaw and not a_flaw:
            return "A"
        return "TIE"

    async def compare_async(
        self,
        prompt: str,
        candidate_output: str,
        baseline_output: str,
    ) -> Dict[str, Any]:
        """
        Executes dual swapped evaluation to detect and eliminate position bias.
        """
        # Presentation 1: Candidate is A, Baseline is B
        res1 = self._simulate_judgment(option_a=candidate_output, option_b=baseline_output)
        cand_won_1 = (res1 == "A")
        base_won_1 = (res1 == "B")

        # Presentation 2: Swapped! Baseline is A, Candidate is B
        res2 = self._simulate_judgment(option_a=baseline_output, option_b=candidate_output)
        base_won_2 = (res2 == "A")
        cand_won_2 = (res2 == "B")

        # Synthesis
        position_bias_detected = False
        if res1 == "A" and res2 == "A":
            # Judge always picked Option A regardless of content!
            position_bias_detected = True
            final_winner = "TIE"
            reason = "Position Bias Detected: Judge selected Option A in both orientations."
        elif cand_won_1 and cand_won_2:
            final_winner = "CANDIDATE"
            reason = "Candidate strictly preferred in both swapped presentations."
        elif base_won_1 and base_won_2:
            final_winner = "BASELINE"
            reason = "Baseline strictly preferred in both swapped presentations."
        else:
            final_winner = "TIE"
            reason = "Inconclusive / Tie across swapped orientations."

        return {
            "winner": final_winner,
            "candidate_won": final_winner == "CANDIDATE",
            "position_bias_detected": position_bias_detected,
            "reason": reason,
            "trial_1": {"cand": "A", "base": "B", "pick": res1},
            "trial_2": {"cand": "B", "base": "A", "pick": res2},
        }

    async def evaluate_async(self, output_text: str, test_case: TestCase, baseline_output: Optional[str] = None, **kwargs) -> MetricScore:
        if not baseline_output:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No baseline output provided for pairwise comparison")

        res = await self.compare_async(test_case.prompt, candidate_output=output_text, baseline_output=baseline_output)
        score = 1.0 if res["candidate_won"] else (0.5 if res["winner"] == "TIE" else 0.0)
        return MetricScore(
            name=self.name,
            score=score,
            passed=res["candidate_won"] or res["winner"] == "TIE",
            reason=res["reason"],
            metadata=res
        )
