"""Unbiased Pass@k calculation based on Chen et al. (2021) HumanEval formulation."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence
import numpy as np


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes unbiased estimator of pass@k given:
      n: total samples drawn per prompt (e.g. 10 or 20)
      c: number of correct/passing samples (0 <= c <= n)
      k: budget threshold (1 <= k <= n)

    Formula:
      pass@k = 1 - prod_{i=1}^k ((n - c - i + 1) / (n - i + 1))
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n})")

    # Numerically stable product calculation
    product = 1.0
    for i in range(1, k + 1):
        product *= (n - c - i + 1) / (n - i + 1)

    return round(1.0 - product, 4)


def compute_suite_pass_at_k(
    task_outcomes: List[List[bool]],
    k_values: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Computes mean Pass@k across an entire suite of test cases.
    Each element in task_outcomes is a list of boolean passes for that task's n samples.
    """
    if not task_outcomes:
        return {f"pass@{k}": 0.0 for k in k_values}

    results: Dict[str, float] = {}

    for k in k_values:
        task_pass_estimates = []
        for samples in task_outcomes:
            n = len(samples)
            c = sum(1 for s in samples if s)
            if k <= n:
                estimate = estimate_pass_at_k(n=n, c=c, k=k)
                task_pass_estimates.append(estimate)

        if task_pass_estimates:
            results[f"pass@{k}"] = round(float(np.mean(task_pass_estimates)), 4)
        else:
            results[f"pass@{k}"] = 0.0

    return results
