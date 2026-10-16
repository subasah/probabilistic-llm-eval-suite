"""Statistical hypothesis testing for LLM performance comparisons and regression analysis."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats


def mcnemar_paired_test(
    baseline_passes: Sequence[bool],
    candidate_passes: Sequence[bool],
) -> Dict[str, Any]:
    """
    McNemar's test for paired categorical data (Pass/Fail before vs after a change).
    Essential for determining whether an LLM prompt or model version change causes
    a statistically significant shift in pass rate across identical test items.

    Contingency Matrix:
                    Candidate Pass   Candidate Fail
    Baseline Pass        a                b (Regressions)
    Baseline Fail        c (Improvements) d
    """
    if len(baseline_passes) != len(candidate_passes):
        raise ValueError("Baseline and candidate must have identical number of paired test cases")

    n = len(baseline_passes)
    if n == 0:
        return {"p_value": 1.0, "statistic": 0.0, "regressions": 0, "improvements": 0}

    a = sum(1 for b, c in zip(baseline_passes, candidate_passes) if b and c)
    b = sum(1 for b, c in zip(baseline_passes, candidate_passes) if b and not c)  # Regression
    c = sum(1 for b, c in zip(baseline_passes, candidate_passes) if not b and c)  # Improvement
    d = sum(1 for b, c in zip(baseline_passes, candidate_passes) if not b and not c)

    discordant = b + c

    if discordant == 0:
        # Exactly identical outcomes on discordant pairs
        return {
            "p_value": 1.0,
            "statistic": 0.0,
            "regressions": b,
            "improvements": c,
            "net_gain": 0,
            "is_significant": False,
            "interpretation": "No discordant items; candidate and baseline performed identically."
        }

    # If sample of discordant pairs is small (< 25), compute exact two-tailed binomial p-value
    if discordant < 25:
        # Binomial test with p=0.5
        # probability of observing at least min(b, c) under null hypothesis
        k = min(b, c)
        p_val = stats.binomtest(k, n=discordant, p=0.5, alternative="two-sided").pvalue
        chi2_stat = ((abs(b - c) - 1.0) ** 2) / discordant if discordant > 0 else 0.0
    else:
        # Edwards' continuity-corrected chi-squared statistic
        chi2_stat = ((abs(b - c) - 1.0) ** 2) / discordant
        p_val = stats.chi2.sf(chi2_stat, df=1)

    net_gain = c - b
    return {
        "p_value": round(float(p_val), 5),
        "statistic": round(float(chi2_stat), 4),
        "regressions": b,
        "improvements": c,
        "concordant_passes": a,
        "concordant_failures": d,
        "net_gain": net_gain,
        "is_significant": bool(p_val < 0.05),
    }


def compare_continuous_metrics(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
) -> Dict[str, Any]:
    """
    Welch's two-sample t-test (does not assume equal variance)
    and paired Wilcoxon signed-rank test for continuous metric distributions.
    """
    b_arr = np.array(baseline_scores, dtype=float)
    c_arr = np.array(candidate_scores, dtype=float)

    if len(b_arr) == 0 or len(c_arr) == 0:
        return {"p_value": 1.0, "delta": 0.0, "is_significant": False}

    b_mean = float(np.mean(b_arr))
    c_mean = float(np.mean(c_arr))
    delta = c_mean - b_mean
    pct_delta = (delta / b_mean * 100.0) if b_mean != 0 else 0.0

    # Paired difference if lengths match
    if len(b_arr) == len(c_arr) and len(b_arr) > 2:
        diffs = c_arr - b_arr
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            try:
                # Wilcoxon signed-rank test
                res = stats.wilcoxon(diffs, zero_method="pratt")
                p_val = float(res.pvalue)
            except Exception:
                t_res = stats.ttest_rel(c_arr, b_arr)
                p_val = float(t_res.pvalue)
    else:
        # Welch's t-test
        t_res = stats.ttest_ind(c_arr, b_arr, equal_var=False)
        p_val = float(t_res.pvalue)

    return {
        "baseline_mean": round(b_mean, 4),
        "candidate_mean": round(c_mean, 4),
        "delta": round(delta, 4),
        "delta_percent": round(pct_delta, 2),
        "p_value": round(p_val, 5),
        "is_significant": p_val < 0.05,
    }
