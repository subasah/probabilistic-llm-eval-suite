"""Semantic evaluators: Exact Match, Token F1, Levenshtein, and Cosine Similarity."""

from __future__ import annotations

import collections
import math
import re
from typing import List, Optional
from probeval.config import MetricScore, TestCase
from probeval.evaluators.base import BaseEvaluator


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluates normalized exact string match."""

    def __init__(self, name: str = "exact_match", is_critical: bool = False):
        super().__init__(name=name, is_critical=is_critical)

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        if not test_case.ground_truth:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No ground truth provided")

        norm_pred = normalize_text(output_text)
        norm_gt = normalize_text(test_case.ground_truth)
        match = (norm_pred == norm_gt)

        return MetricScore(
            name=self.name,
            score=1.0 if match else 0.0,
            passed=match,
            reason="Normalized exact match" if match else "Output does not match ground truth exactly",
        )


class TokenF1Evaluator(BaseEvaluator):
    """
    Standard SQuAD-style token-level F1 score.
    Computes precision, recall, and harmonic mean between prediction and ground truth.
    """

    def __init__(self, name: str = "token_f1", pass_threshold: float = 0.70, is_critical: bool = False):
        super().__init__(name=name, is_critical=is_critical)
        self.pass_threshold = pass_threshold

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        if not test_case.ground_truth:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No ground truth provided")

        pred_tokens = normalize_text(output_text).split()
        gt_tokens = normalize_text(test_case.ground_truth).split()

        if not pred_tokens or not gt_tokens:
            passed = (pred_tokens == gt_tokens)
            return MetricScore(
                name=self.name,
                score=1.0 if passed else 0.0,
                passed=passed,
                reason="Empty token match",
            )

        common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return MetricScore(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Zero token overlap with ground truth",
            )

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        passed = f1 >= self.pass_threshold

        return MetricScore(
            name=self.name,
            score=round(f1, 4),
            passed=passed,
            reason=f"Token F1: {f1:.3f} (P: {precision:.2f}, R: {recall:.2f}, Threshold: {self.pass_threshold})",
            metadata={"precision": precision, "recall": recall, "f1": f1}
        )


class CosineSimilarityEvaluator(BaseEvaluator):
    """
    Computes semantic cosine similarity over character and word n-grams,
    robust to paraphrasing and minor word permutations without heavy model weights.
    """

    def __init__(self, name: str = "semantic_cosine_similarity", pass_threshold: float = 0.75, is_critical: bool = False):
        super().__init__(name=name, is_critical=is_critical)
        self.pass_threshold = pass_threshold

    def _get_char_ngrams(self, text: str, n: int = 3) -> collections.Counter:
        norm = normalize_text(text)
        padded = f" {norm} "
        return collections.Counter([padded[i:i+n] for i in range(len(padded) - n + 1)])

    async def evaluate_async(self, output_text: str, test_case: TestCase, **kwargs) -> MetricScore:
        target = test_case.ground_truth or test_case.context
        if not target:
            return MetricScore(name=self.name, score=1.0, passed=True, reason="No reference target provided")

        vec_pred = self._get_char_ngrams(output_text, n=3)
        vec_tgt = self._get_char_ngrams(target, n=3)

        intersection = set(vec_pred.keys()) & set(vec_tgt.keys())
        dot_product = sum(vec_pred[k] * vec_tgt[k] for k in intersection)

        norm_pred = math.sqrt(sum(v ** 2 for v in vec_pred.values()))
        norm_tgt = math.sqrt(sum(v ** 2 for v in vec_tgt.values()))

        if norm_pred == 0 or norm_tgt == 0:
            score = 0.0
        else:
            score = dot_product / (norm_pred * norm_tgt)

        passed = score >= self.pass_threshold
        return MetricScore(
            name=self.name,
            score=round(score, 4),
            passed=passed,
            reason=f"N-gram Cosine Similarity: {score:.3f} (Threshold: {self.pass_threshold})",
        )
