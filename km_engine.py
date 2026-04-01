"""
EvidenceKM — Survival analysis of meta-analytic significance across trust thresholds.

Event definition:
  - Cohort: 888 significant MAs (p < 0.05 from verdicts.csv).
  - Event time: the MA's own final_score (integer 0-100).
  - Event: "dies" (loses significance) at any trust threshold exceeding its score.
  - Also compute z_trust = z_orig * sqrt(score/100); weakened if z_trust < 1.96.

Kaplan-Meier: S(t) = prod_{t_i <= t} (1 - d_i / n_i)
Greenwood variance, log-log CI.
Log-rank test: chi-squared comparing survival curves.
Cox PH (simplified): HR via standardised mean difference between early/late groups.
"""

import csv
import math
import json
from pathlib import Path
from scipy.stats import norm, chi2


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORES_CSV = Path("C:/Models/EvidenceScore/results/scores.csv")
VERDICTS_CSV = Path("C:/Models/ActionableEvidence/results/verdicts.csv")
GROUPS_CSV = Path("C:/Models/TrustGate/data/review_groups.csv")
RESULTS_DIR = Path("C:/Models/EvidenceKM/results")


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

def z_from_p(p_value: float) -> float:
    """Convert two-tailed p-value to absolute z-score.

    Edge cases:
      - p = 0  → z = 8.0 (capped)
      - p >= 1 → z = 0.0
      - NaN    → raises ValueError
    """
    if p_value != p_value:  # NaN check
        raise ValueError("p_value is NaN")
    if p_value <= 0.0:
        return 8.0  # cap
    if p_value >= 1.0:
        return 0.0
    return abs(norm.ppf(p_value / 2.0))


def compute_z_trust(z_orig: float, score: float) -> float:
    """Compute trust-adjusted z-score: z_trust = z_orig * sqrt(score / 100).

    Returns 0.0 if score <= 0.
    """
    if score <= 0.0:
        return 0.0
    return z_orig * math.sqrt(score / 100.0)


def is_weakened(z_trust: float, threshold: float = 1.96) -> bool:
    """Return True if z_trust < 1.96 (weakened at its own score level)."""
    return z_trust < threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores() -> dict:
    """Load scores.csv → {ma_id: {col: val, ...}}."""
    result = {}
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["ma_id"]] = {
                "review_id": row["review_id"],
                "audit_score": float(row["audit_score"]),
                "consistency_score": float(row["consistency_score"]),
                "robustness_score": float(row["robustness_score"]),
                "stability_score": float(row["stability_score"]),
                "power_score": float(row["power_score"]),
                "final_score": float(row["final_score"]),
                "grade": row["grade"],
            }
    return result


def load_verdicts() -> list:
    """Load verdicts.csv, return rows where significant == 'True'."""
    rows = []
    with open(VERDICTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_review_groups() -> dict:
    """Load review_groups.csv → {review_id_prefix: review_group}."""
    result = {}
    with open(GROUPS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["review_id_prefix"]] = row["review_group"]
    return result


# ---------------------------------------------------------------------------
# Build cohort
# ---------------------------------------------------------------------------

def build_cohort(
    verdicts: list | None = None,
    scores: dict | None = None,
    groups: dict | None = None,
) -> list:
    """Build the KM cohort from significant MAs.

    Returns list of dicts with keys:
      ma_id, review_id, event_time, z_orig, z_trust, weakened,
      final_score, audit_score, consistency_score, robustness_score,
      stability_score, power_score, grade, review_group.
    """
    if verdicts is None:
        verdicts = load_verdicts()
    if scores is None:
        scores = load_scores()
    if groups is None:
        groups = load_review_groups()

    cohort = []
    for row in verdicts:
        if row["significant"] != "True":
            continue
        ma_id = row["ma_id"]
        p_val = row["p_value"]
        estimate = row["estimate"]

        # Skip if p_value invalid
        try:
            p_float = float(p_val)
        except (ValueError, TypeError):
            continue

        # Skip if estimate invalid (for z computation)
        try:
            est_float = float(estimate)
        except (ValueError, TypeError):
            est_float = None

        # Get score info
        sc = scores.get(ma_id, {})
        final_score = sc.get("final_score", 50.0)  # default midpoint if missing
        review_id = sc.get("review_id") or row.get("review_id", "")

        # Event time = final_score (integer bucket 0-100)
        event_time = final_score

        # Z original from p-value
        try:
            z_orig = z_from_p(p_float)
        except ValueError:
            z_orig = 0.0

        # z_trust
        zt = compute_z_trust(z_orig, final_score)
        weakened = is_weakened(zt)

        # Domain lookup
        review_group = groups.get(review_id, "Other")

        cohort.append({
            "ma_id": ma_id,
            "review_id": review_id,
            "event_time": event_time,
            "z_orig": z_orig,
            "z_trust": zt,
            "weakened": weakened,
            "final_score": final_score,
            "audit_score": sc.get("audit_score", 50.0),
            "consistency_score": sc.get("consistency_score", 50.0),
            "robustness_score": sc.get("robustness_score", 50.0),
            "stability_score": sc.get("stability_score", 50.0),
            "power_score": sc.get("power_score", 50.0),
            "grade": sc.get("grade", "?"),
            "review_group": review_group,
        })

    return cohort


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

def kaplan_meier(event_times: list, all_censored: bool = False) -> dict:
    """Compute Kaplan-Meier survival curve.

    All observations are treated as events (no censoring in this context —
    every MA eventually "dies" at its threshold).

    Args:
        event_times: list of numeric event times (final_scores).
        all_censored: if True, treat all as censored (for testing).

    Returns dict with:
      times: sorted unique event times
      S: survival probabilities S(t)
      var_greenwood: Greenwood variance at each time point
      ci_lower: log-log 95% CI lower
      ci_upper: log-log 95% CI upper
      n_events: event counts per time point
      n_at_risk: n at risk per time point
      median: median survival time (S(t) <= 0.5 first crossing)
    """
    if not event_times:
        return {
            "times": [],
            "S": [],
            "var_greenwood": [],
            "ci_lower": [],
            "ci_upper": [],
            "n_events": [],
            "n_at_risk": [],
            "median": None,
        }

    n_total = len(event_times)
    sorted_times = sorted(event_times)
    unique_times = sorted(set(sorted_times))

    times_out = []
    S_out = []
    var_greenwood_out = []
    ci_lower_out = []
    ci_upper_out = []
    n_events_out = []
    n_at_risk_out = []

    S = 1.0
    greenwood_sum = 0.0
    n_at_risk = n_total

    from collections import Counter
    count_map = Counter(sorted_times)

    z_95 = 1.96

    for t in unique_times:
        d = count_map[t]  # events at this time
        if all_censored:
            d = 0
        n_i = n_at_risk

        if n_i > 0 and d > 0:
            S *= (1.0 - d / n_i)
            greenwood_sum += d / (n_i * (n_i - d)) if (n_i - d) > 0 else 0.0

        times_out.append(t)
        S_out.append(S)
        n_events_out.append(d)
        n_at_risk_out.append(n_i)
        var_greenwood_out.append(S ** 2 * greenwood_sum)

        # Log-log CI: based on log(-log(S))
        if S > 0.0 and S < 1.0 and greenwood_sum > 0.0:
            log_S = math.log(S)
            se_log_log = math.sqrt(greenwood_sum) / abs(log_S)
            log_log_S = math.log(-log_S)
            ci_lower_out.append(math.exp(-math.exp(log_log_S + z_95 * se_log_log)))
            ci_upper_out.append(math.exp(-math.exp(log_log_S - z_95 * se_log_log)))
        else:
            ci_lower_out.append(S)
            ci_upper_out.append(S)

        n_at_risk -= d

    # Median: first time S(t) <= 0.5
    median = None
    for t, s in zip(times_out, S_out):
        if s <= 0.5:
            median = t
            break

    return {
        "times": times_out,
        "S": S_out,
        "var_greenwood": var_greenwood_out,
        "ci_lower": ci_lower_out,
        "ci_upper": ci_upper_out,
        "n_events": n_events_out,
        "n_at_risk": n_at_risk_out,
        "median": median,
    }


# ---------------------------------------------------------------------------
# Log-rank test
# ---------------------------------------------------------------------------

def log_rank_test(groups: dict) -> dict:
    """Compute log-rank chi-squared test comparing multiple survival groups.

    Args:
        groups: {group_label: [event_times_list]}

    Returns:
        {chi2_stat, df, p_value, observed, expected per group}
    """
    if len(groups) < 2:
        return {
            "chi2_stat": 0.0,
            "df": 0,
            "p_value": 1.0,
            "observed": {k: len(v) for k, v in groups.items()},
            "expected": {k: len(v) for k, v in groups.items()},
            "error": "Need at least 2 groups",
        }

    # Gather all unique times
    all_times_set = set()
    for times in groups.values():
        all_times_set.update(times)
    all_unique_times = sorted(all_times_set)

    from collections import Counter
    group_labels = list(groups.keys())
    k = len(group_labels)

    # Count events per group per time
    counts = {g: Counter(groups[g]) for g in group_labels}
    total_at_risk = {g: len(groups[g]) for g in group_labels}

    O = {g: 0 for g in group_labels}  # observed
    E = {g: 0.0 for g in group_labels}  # expected

    for t in all_unique_times:
        n_j = {g: total_at_risk[g] for g in group_labels}
        d_j = {g: counts[g].get(t, 0) for g in group_labels}
        n_total = sum(n_j.values())
        d_total = sum(d_j.values())

        if n_total <= 0 or d_total == 0:
            # Advance at-risk counts
            for g in group_labels:
                total_at_risk[g] -= d_j[g]
            continue

        for g in group_labels:
            e_g = n_j[g] * d_total / n_total if n_total > 0 else 0.0
            O[g] += d_j[g]
            E[g] += e_g

        for g in group_labels:
            total_at_risk[g] -= d_j[g]

    # Chi-squared statistic (simplified: sum (O-E)^2/E)
    chi2_stat = 0.0
    for g in group_labels:
        if E[g] > 0:
            chi2_stat += (O[g] - E[g]) ** 2 / E[g]

    df = k - 1
    p_val = 1.0 - chi2.cdf(chi2_stat, df) if df > 0 else 1.0

    return {
        "chi2_stat": round(chi2_stat, 4),
        "df": df,
        "p_value": round(p_val, 6),
        "observed": O,
        "expected": {g: round(E[g], 2) for g in group_labels},
    }


# ---------------------------------------------------------------------------
# Cox PH (simplified)
# ---------------------------------------------------------------------------

def cox_ph_simplified(
    cohort: list,
    covariate: str,
    event_col: str = "event_time",
    split_quantile: float = 0.5,
) -> dict:
    """Simplified Cox PH via standardised mean difference.

    Split cohort into early (event <= median) and late (event > median) groups.
    Compute mean and SD of covariate in each group.
    HR = exp(log_hr), where log_hr = SMD of covariate.
    95% CI using SE of SMD.

    This is a linearization approximation: positive SMD means covariate
    associated with later events → HR < 1 (protective).
    """
    if not cohort:
        return {"hr": 1.0, "ci_lower": None, "ci_upper": None, "p_value": 1.0,
                "covariate": covariate, "n_early": 0, "n_late": 0}

    event_times = [row[event_col] for row in cohort]
    event_times_sorted = sorted(event_times)
    n = len(event_times_sorted)
    split_idx = int(n * split_quantile)
    if split_idx >= n:
        split_idx = n - 1
    median_time = event_times_sorted[split_idx]

    early = [row[covariate] for row in cohort if row[event_col] <= median_time]
    late = [row[covariate] for row in cohort if row[event_col] > median_time]

    if not early or not late:
        return {"hr": 1.0, "ci_lower": None, "ci_upper": None, "p_value": 1.0,
                "covariate": covariate, "n_early": len(early), "n_late": len(late)}

    def mean_sd(vals):
        n_ = len(vals)
        if n_ == 0:
            return 0.0, 1.0
        m = sum(vals) / n_
        if n_ == 1:
            return m, 1.0
        var = sum((v - m) ** 2 for v in vals) / (n_ - 1)
        return m, math.sqrt(var) if var > 0 else 1e-9

    m_e, sd_e = mean_sd(early)
    m_l, sd_l = mean_sd(late)
    n_e = len(early)
    n_l = len(late)

    # Pooled SD
    pooled_var = ((n_e - 1) * sd_e ** 2 + (n_l - 1) * sd_l ** 2) / (n_e + n_l - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-9

    # SMD (early - late): early events had lower covariate → SMD > 0 → HR > 1 (risk)
    smd = (m_e - m_l) / pooled_sd

    # SE of SMD
    se_smd = math.sqrt((n_e + n_l) / (n_e * n_l) + smd ** 2 / (2 * (n_e + n_l)))

    # log HR approximated as SMD (common approximation in simplified Cox)
    log_hr = smd
    se_log_hr = se_smd

    hr = math.exp(log_hr)
    ci_lower = math.exp(log_hr - 1.96 * se_log_hr)
    ci_upper = math.exp(log_hr + 1.96 * se_log_hr)

    # Two-sided p-value
    z_stat = log_hr / se_log_hr if se_log_hr > 0 else 0.0
    p_val = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    return {
        "covariate": covariate,
        "hr": round(hr, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 6),
        "smd": round(smd, 4),
        "n_early": n_e,
        "n_late": n_l,
        "mean_early": round(m_e, 2),
        "mean_late": round(m_l, 2),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    verdicts: list | None = None,
    scores: dict | None = None,
    groups: dict | None = None,
) -> dict:
    """Run the full EvidenceKM pipeline.

    Returns a comprehensive results dict.
    """
    # Build cohort
    cohort = build_cohort(verdicts, scores, groups)

    n_sig = len(cohort)
    n_weakened = sum(1 for row in cohort if row["weakened"])

    # Overall KM
    event_times_all = [row["event_time"] for row in cohort]
    km_overall = kaplan_meier(event_times_all)

    # KM by domain (top 5 by count)
    from collections import Counter
    domain_counts = Counter(row["review_group"] for row in cohort)
    top5_domains = [d for d, _ in domain_counts.most_common(5)]

    km_by_domain = {}
    for domain in top5_domains:
        domain_times = [row["event_time"] for row in cohort if row["review_group"] == domain]
        km_by_domain[domain] = kaplan_meier(domain_times)

    # Log-rank test comparing top-5 domains
    groups_for_lr = {
        d: [row["event_time"] for row in cohort if row["review_group"] == d]
        for d in top5_domains
    }
    lr_result = log_rank_test(groups_for_lr)

    # Cox PH for 5 trust components
    covariates = [
        "audit_score",
        "consistency_score",
        "robustness_score",
        "stability_score",
        "power_score",
    ]
    cox_results = {}
    for cov in covariates:
        cox_results[cov] = cox_ph_simplified(cohort, cov)

    # Summary stats
    median_survival = km_overall["median"]
    s_at_50 = None
    for t, s in zip(km_overall["times"], km_overall["S"]):
        if t >= 50:
            s_at_50 = s
            break

    result = {
        "n_significant": n_sig,
        "n_weakened": n_weakened,
        "pct_weakened": round(100.0 * n_weakened / n_sig, 1) if n_sig > 0 else 0.0,
        "median_survival_threshold": median_survival,
        "s_at_threshold_50": round(s_at_50, 4) if s_at_50 is not None else None,
        "top5_domains": top5_domains,
        "domain_counts": dict(domain_counts.most_common(10)),
        "km_overall": km_overall,
        "km_by_domain": km_by_domain,
        "log_rank": lr_result,
        "cox_results": cox_results,
    }

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("EvidenceKM Pipeline")
    print("=" * 60)

    results = run_pipeline()

    print(f"Significant MAs:            {results['n_significant']}")
    print(f"Weakened (z_trust < 1.96):  {results['n_weakened']} ({results['pct_weakened']}%)")
    print(f"Median survival threshold:  {results['median_survival_threshold']}")
    print(f"S(50):                      {results['s_at_threshold_50']}")
    print(f"\nLog-rank test (top-5 domains):")
    lr = results["log_rank"]
    print(f"  chi2 = {lr['chi2_stat']:.4f}, df = {lr['df']}, p = {lr['p_value']:.6f}")
    print(f"\nCox PH results:")
    for cov, cox in results["cox_results"].items():
        print(f"  {cov:25s}: HR = {cox['hr']:.4f} [{cox['ci_lower']:.4f}, {cox['ci_upper']:.4f}], p = {cox['p_value']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'pipeline_results.json'}")
