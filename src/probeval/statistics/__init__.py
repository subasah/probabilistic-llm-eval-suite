"""Statistical engine package."""

from probeval.statistics.bootstrap import bootstrap_confidence_interval, wilson_score_interval
from probeval.statistics.pass_at_k import estimate_pass_at_k, compute_suite_pass_at_k
from probeval.statistics.hypothesis_tests import mcnemar_paired_test, compare_continuous_metrics
from probeval.statistics.regression_gate import ContinuousRegressionGatekeeper

__all__ = [
    "bootstrap_confidence_interval",
    "wilson_score_interval",
    "estimate_pass_at_k",
    "compute_suite_pass_at_k",
    "mcnemar_paired_test",
    "compare_continuous_metrics",
    "ContinuousRegressionGatekeeper",
]
