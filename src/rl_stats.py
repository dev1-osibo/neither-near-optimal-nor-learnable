"""
Paired Statistical Comparison Helpers
=====================================
Small, defensible statistics used to decide whether one policy's cost
advantage over another is real or noise. Everything operates on PAIRED
per-episode cost arrays (same episode index == same historical week).

Design choices (honest reporting over cherry-picking):
- Report BOTH a parametric (paired t) and non-parametric (Wilcoxon) test.
- Always report an effect size and a bootstrap 95% CI, not just a p-value —
  with small samples a p-value alone is misleading.
- Guard the degenerate zero-variance case (e.g. A2C policies that collapse
  to identical behavior across seeds) which would divide-by-zero a t-test.
"""

import numpy as np
from scipy import stats


def paired_comparison(a, b, n_boot=10000, seed=0):
    """
    Compare paired samples a vs b (lower cost is better). Positive 'improvement'
    means a is CHEAPER than b (a beats b).

    Args:
        a: array of costs for policy A (e.g. RL), length N.
        b: array of costs for policy B (e.g. RuleBased baseline), length N.
        n_boot: bootstrap resamples for the CI on the mean difference.
        seed: RNG seed for reproducible bootstrap.

    Returns:
        dict of statistics (all JSON-serializable floats / bools).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, "paired arrays must match in shape"
    n = a.size

    diff = b - a  # positive => A cheaper than B => A improvement
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # Percent improvement of A relative to B's mean cost.
    mean_b = float(np.mean(b))
    pct_improvement = float(mean_diff / mean_b * 100.0) if mean_b != 0 else 0.0

    # --- Paired t-test (guard zero-variance) ---
    if std_diff == 0.0:
        t_stat, t_p = (float("nan"), 1.0 if mean_diff == 0 else 0.0)
    else:
        t_stat, t_p = stats.ttest_rel(a, b)
        t_stat, t_p = float(t_stat), float(t_p)

    # --- Wilcoxon signed-rank (non-parametric); undefined if all diffs zero ---
    if np.allclose(diff, 0.0):
        w_stat, w_p = (float("nan"), 1.0)
    else:
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", correction=False)
            w_stat, w_p = float(w_stat), float(w_p)
        except ValueError:
            w_stat, w_p = (float("nan"), 1.0)

    # --- Effect size: Cohen's d for paired data (mean diff / sd diff) ---
    cohens_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0

    # --- Bootstrap 95% CI on the mean paired difference ---
    rng = np.random.default_rng(seed)
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = diff[idx].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_low = ci_high = mean_diff

    return {
        "n": int(n),
        "mean_cost_a": float(np.mean(a)),
        "mean_cost_b": mean_b,
        "mean_diff": mean_diff,          # >0 => A cheaper (A wins)
        "std_diff": std_diff,
        "pct_improvement": pct_improvement,
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_p,
        "cohens_d": cohens_d,
        "boot_ci95_low": float(ci_low),
        "boot_ci95_high": float(ci_high),
        # CI excluding zero is the practical significance signal we trust most.
        "ci_excludes_zero": bool((ci_low > 0) or (ci_high < 0)),
        "a_wins": bool(mean_diff > 0),
    }


def holm_bonferroni(pvalues, alpha=0.05):
    """
    Holm-Bonferroni step-down correction for a family of p-values.

    Args:
        pvalues: list/array of raw p-values.
        alpha: family-wise error rate.

    Returns:
        (reject, adjusted) where reject is a boolean array (True == significant
        after correction) and adjusted is the array of adjusted p-values.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, adj)   # enforce monotonicity
        adjusted[idx] = min(1.0, running_max)
    reject = adjusted < alpha
    return reject, adjusted
