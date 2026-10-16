"""Non-parametric Bootstrap resampling and confidence interval estimation."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple
import numpy as np
from probeval.config import ConfidenceInterval


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_resamples: int = 1500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """
    Computes empirical non-parametric bootstrap confidence interval for the mean.
    Does not assume normal distribution of errors.
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return ConfidenceInterval(mean=0.0, ci_lower=0.0, ci_upper=0.0, std_err=0.0, sample_size=0)
    if n == 1:
        val = float(arr[0])
        return ConfidenceInterval(mean=val, ci_lower=val, ci_upper=val, std_err=0.0, sample_size=1)

    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(arr))

    # Resample with replacement
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))
    std_err = float(np.std(resampled_means, ddof=1))

    return ConfidenceInterval(
        mean=round(mean_val, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        std_err=round(std_err, 4),
        sample_size=n,
    )


def wilson_score_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes the Wilson score interval for binomial proportion (pass rate).
    Particularly robust for small sample sizes or proportions near 0 or 1.
    """
    if total == 0:
        return 0.0, 0.0

    # z for 95% is ~1.96
    from scipy.stats import norm
    z = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)

    p_hat = successes / total
    z2 = z ** 2
    n = total

    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * (n ** 2))))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)
