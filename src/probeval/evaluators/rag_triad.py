"""RAG Triad Evaluators: Faithfulness, Answer Relevance, and Context Precision."""

from __future__ import annotations

import collections
import re
from typing import List, Optional
from probeval.config import MetricScore, TestCase
from probeval.evaluators.base import BaseEvaluator
from probeval.evaluators.semantic import normalize_text


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluates whether the generated response is strictly grounded in the retrieved context,
    penalizing hallucinations and unsupported claims.
    """

    def __init__(self, name: str = "rag_faithfulness", pass_threshold: float = 0.80, is_critical: bool = True):
        super().__init__(name=name, is_critical=is_critical)
        self.pass_threshold = pass_threshold

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        if not test_case.context:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No context provided; skipped")

        # Extract simple sentence claims from the output
        sentences = [s.strip() for s in re.split(r"[.!?\n]", output_text) if len(s.strip()) > 10]
        if not sentences:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No evaluable claims")

        context_norm = normalize_text(test_case.context)
        context_words = set(context_norm.split())

        grounded_claims = 0
        unsupported_claims = []

        for sentence in sentences:
            sent_words = [w for w in normalize_text(sentence).split() if len(w) > 3]
            if not sent_words:
                grounded_claims += 1
                continue

            overlap = sum(1 for w in sent_words if w in context_words)
            overlap_ratio = overlap / len(sent_words)

            if overlap_ratio >= 0.45:
                grounded_claims += 1
            else:
                unsupported_claims.append(sentence)

        score = grounded_claims / len(sentences)
        passed = score >= self.pass_threshold

        reason = (
            f"Faithfulness score: {score:.2f} ({grounded_claims}/{len(sentences)} claims grounded)"
            if passed
            else f"Hallucination detected: {len(unsupported_claims)} claims not grounded in context."
        )

        return MetricScore(
            name=self.name,
            score=round(score, 3),
            passed=passed,
            reason=reason,
            metadata={"unsupported_claims": unsupported_claims[:3]}
        )


class AnswerRelevanceEvaluator(BaseEvaluator):
    """
    Evaluates whether the generated response directly answers the user's prompt
    without drifting into irrelevant tangents or dodging the inquiry.
    """

    def __init__(self, name: str = "rag_answer_relevance", pass_threshold: float = 0.60, is_critical: bool = False):
        super().__init__(name=name, is_critical=is_critical)
        self.pass_threshold = pass_threshold

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        if test_case.tools and not test_case.context:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="Tool execution test case; textual relevance skipped")

        prompt_words = [w for w in normalize_text(test_case.prompt).split() if len(w) > 3]
        if not prompt_words:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="Prompt too short to measure relevance")

        output_words = set(normalize_text(output_text).split())
        matched_words = [w for w in prompt_words if w in output_words]

        # Inverted harmonic penalty for responses that are too short to be substantive
        len_penalty = min(1.0, len(output_text.split()) / 8.0)
        overlap_score = len(matched_words) / len(prompt_words)
        relevance_score = overlap_score * len_penalty

        passed = relevance_score >= self.pass_threshold
        return MetricScore(
            name=self.name,
            score=round(relevance_score, 3),
            passed=passed,
            reason=f"Answer Relevance: {relevance_score:.2f} (Keyword overlap: {len(matched_words)}/{len(prompt_words)})",
        )
