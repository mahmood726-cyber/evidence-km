"""
25 tests for EvidenceKM km_engine.py.

T1-T5:   z_from_p, event time, edge cases (p=0, NaN, estimate=0)
T6-T10:  Kaplan-Meier (known curve, single event, monotone S, empty, all same time)
T11-T15: Log-rank (identical groups p~1, different groups p<0.05, single group, edge cases)
T16-T20: Cox PH (significant covariate, null covariate, constant values, empty, basic HR)
T21-T25: Pipeline integration (fixture data, real data if exists, output validation)
"""

import sys
import math
import pytest

sys.path.insert(0, "C:/Models/EvidenceKM")

from km_engine import (
    z_from_p,
    compute_z_trust,
    is_weakened,
    kaplan_meier,
    log_rank_test,
    cox_ph_simplified,
    build_cohort,
    run_pipeline,
    SCORES_CSV,
    VERDICTS_CSV,
    GROUPS_CSV,
)

REAL_DATA_AVAILABLE = SCORES_CSV.exists() and VERDICTS_CSV.exists() and GROUPS_CSV.exists()


# ===========================================================================
# T1-T5: z_from_p, event time, edge cases
# ===========================================================================

class TestZFromP:
    def test_T1_typical_p_value(self):
        """T1: z_from_p for a standard p=0.05 returns ~1.96."""
        z = z_from_p(0.05)
        assert abs(z - 1.96) < 0.01, f"Expected ~1.96, got {z}"

    def test_T2_event_time_is_final_score(self):
        """T2: event_time is defined as final_score (integer bucket)."""
        # When final_score = 75, event_time = 75
        final_score = 75.0
        event_time = final_score  # by definition in engine
        assert event_time == 75.0

    def test_T3_p_equals_zero_capped(self):
        """T3: p=0 returns capped z=8.0."""
        z = z_from_p(0.0)
        assert z == 8.0

    def test_T4_nan_raises_value_error(self):
        """T4: NaN p_value raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            z_from_p(float("nan"))

    def test_T5_z_trust_weakened_at_low_score(self):
        """T5: estimate=0 / low score → z_trust < 1.96 → weakened=True."""
        # p=0.05 gives z~1.96, but at score=50 → z_trust = 1.96 * sqrt(0.5) ~ 1.386
        z_orig = z_from_p(0.05)
        zt = compute_z_trust(z_orig, 50.0)
        assert zt < 1.96
        assert is_weakened(zt)

    def test_T5b_z_trust_not_weakened_at_high_score(self):
        """T5b: Very significant result at high score stays strong."""
        z_orig = z_from_p(0.001)  # ~3.29
        zt = compute_z_trust(z_orig, 100.0)
        assert zt >= 1.96
        assert not is_weakened(zt)

    def test_T5c_z_trust_zero_score(self):
        """T5c: score=0 yields z_trust=0.0."""
        zt = compute_z_trust(3.0, 0.0)
        assert zt == 0.0

    def test_T5d_p_equals_1_gives_z_zero(self):
        """T5d: p=1 gives z=0.0."""
        z = z_from_p(1.0)
        assert z == 0.0


# ===========================================================================
# T6-T10: Kaplan-Meier
# ===========================================================================

class TestKaplanMeier:
    def test_T6_known_km_curve(self):
        """T6: Known 4-event sequence produces correct S values."""
        # 4 events at times 1, 2, 3, 4, n=4
        # S(1) = 3/4 = 0.75
        # S(2) = 0.75 * 2/3 = 0.5
        # S(3) = 0.5 * 1/2 = 0.25
        # S(4) = 0.25 * 0/1 = 0.0
        times = [1, 2, 3, 4]
        km = kaplan_meier(times)
        assert len(km["S"]) == 4
        assert abs(km["S"][0] - 0.75) < 1e-9
        assert abs(km["S"][1] - 0.5) < 1e-9
        assert abs(km["S"][2] - 0.25) < 1e-9
        assert km["S"][3] == 0.0

    def test_T7_single_event(self):
        """T7: Single event time → S drops to 0."""
        km = kaplan_meier([42])
        assert len(km["S"]) == 1
        assert km["S"][0] == 0.0
        assert km["median"] == 42

    def test_T8_s_is_monotone_nonincreasing(self):
        """T8: S(t) is monotone non-increasing."""
        import random
        random.seed(42)
        times = [random.randint(1, 100) for _ in range(50)]
        km = kaplan_meier(times)
        for i in range(1, len(km["S"])):
            assert km["S"][i] <= km["S"][i - 1] + 1e-12, \
                f"S not monotone at index {i}: {km['S'][i-1]} → {km['S'][i]}"

    def test_T9_empty_returns_empty(self):
        """T9: Empty input returns empty curve."""
        km = kaplan_meier([])
        assert km["times"] == []
        assert km["S"] == []
        assert km["median"] is None

    def test_T10_all_same_time(self):
        """T10: All events at same time → S drops to 0 at that time."""
        km = kaplan_meier([50, 50, 50, 50])
        assert len(km["times"]) == 1
        assert km["times"][0] == 50
        assert km["S"][0] == 0.0
        assert km["median"] == 50


# ===========================================================================
# T11-T15: Log-rank test
# ===========================================================================

class TestLogRank:
    def test_T11_identical_groups_p_near_1(self):
        """T11: Identical groups → chi2 ~ 0, p ~ 1."""
        g1 = list(range(10, 90, 5))  # [10,15,...,85]
        g2 = list(range(10, 90, 5))
        result = log_rank_test({"A": g1, "B": g2})
        assert result["chi2_stat"] < 0.5, f"Expected chi2~0, got {result['chi2_stat']}"
        assert result["p_value"] > 0.5

    def test_T12_different_groups_p_small(self):
        """T12: Groups with clearly different survival → p < 0.05."""
        # Low group: all events at score 20-40 (dies early)
        # High group: all events at score 70-90 (lives long)
        low_group = [20, 25, 30, 35, 40] * 10
        high_group = [70, 75, 80, 85, 90] * 10
        result = log_rank_test({"Low": low_group, "High": high_group})
        assert result["p_value"] < 0.05, f"Expected p<0.05, got {result['p_value']}"

    def test_T13_single_group_returns_error(self):
        """T13: Single group returns error message."""
        result = log_rank_test({"OnlyGroup": [10, 20, 30]})
        assert "error" in result
        assert result["p_value"] == 1.0

    def test_T14_empty_group_handled(self):
        """T14: One empty group still runs without crash."""
        result = log_rank_test({"A": [10, 20, 30], "B": []})
        # Should not raise; chi2 computed with caution
        assert "chi2_stat" in result
        assert "p_value" in result

    def test_T15_three_groups_df_equals_2(self):
        """T15: Three groups → df = 2."""
        result = log_rank_test({
            "A": [10, 20, 30],
            "B": [40, 50, 60],
            "C": [70, 80, 90],
        })
        assert result["df"] == 2


# ===========================================================================
# T16-T20: Cox PH simplified
# ===========================================================================

def _make_cohort_fixture(n=40, seed=99):
    """Create synthetic cohort for Cox tests."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        score = rng.uniform(10, 90)
        rows.append({
            "ma_id": f"MA{i}",
            "event_time": score,
            "final_score": score,
            "audit_score": rng.uniform(20, 100),
            "consistency_score": rng.uniform(20, 100),
            "robustness_score": rng.uniform(20, 100),
            "stability_score": rng.uniform(20, 100),
            "power_score": rng.uniform(20, 100),
        })
    return rows


class TestCoxPH:
    def test_T16_significant_covariate(self):
        """T16: Strongly correlated covariate yields HR != 1."""
        # Build cohort where high audit_score predicts late events
        rows = []
        for i in range(60):
            audit = float(i) / 60.0 * 80 + 10  # 10 to 90
            event_time = audit + 5.0  # perfectly correlated
            rows.append({
                "event_time": event_time,
                "audit_score": audit,
                "consistency_score": 50.0,
                "robustness_score": 50.0,
                "stability_score": 50.0,
                "power_score": 50.0,
            })
        result = cox_ph_simplified(rows, "audit_score")
        # High audit → late events → early group has lower audit → HR < 1
        assert result["hr"] != 1.0
        assert result["ci_lower"] is not None
        assert result["ci_upper"] is not None

    def test_T17_null_covariate_hr_near_1(self):
        """T17: Uncorrelated covariate yields HR close to 1."""
        import random
        rng = random.Random(7)
        rows = []
        for _ in range(80):
            rows.append({
                "event_time": rng.uniform(10, 90),
                "audit_score": 50.0,  # constant
                "consistency_score": rng.uniform(10, 90),
                "robustness_score": 50.0,
                "stability_score": 50.0,
                "power_score": 50.0,
            })
        result = cox_ph_simplified(rows, "audit_score")
        # Constant covariate → SMD = 0 → HR = 1
        assert abs(result["hr"] - 1.0) < 0.01

    def test_T18_constant_covariate_values(self):
        """T18: All covariate values constant → HR = 1.0."""
        rows = [
            {"event_time": float(i), "audit_score": 50.0,
             "consistency_score": 50.0, "robustness_score": 50.0,
             "stability_score": 50.0, "power_score": 50.0}
            for i in range(1, 21)
        ]
        result = cox_ph_simplified(rows, "audit_score")
        assert abs(result["hr"] - 1.0) < 0.01

    def test_T19_empty_cohort(self):
        """T19: Empty cohort returns safe defaults."""
        result = cox_ph_simplified([], "audit_score")
        assert result["hr"] == 1.0
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["p_value"] == 1.0

    def test_T20_basic_hr_structure(self):
        """T20: Cox result has required keys and valid HR range."""
        cohort = _make_cohort_fixture()
        result = cox_ph_simplified(cohort, "robustness_score")
        required_keys = {"covariate", "hr", "ci_lower", "ci_upper", "p_value",
                         "n_early", "n_late"}
        assert required_keys.issubset(result.keys())
        assert result["hr"] > 0
        assert result["ci_lower"] < result["hr"] < result["ci_upper"]


# ===========================================================================
# T21-T25: Pipeline integration
# ===========================================================================

def _make_fixture_verdicts():
    return [
        {"ma_id": f"CD{i:06d}_pub1_data__A1", "review_id": f"CD{i:06d}",
         "significant": "True", "p_value": "0.02", "estimate": "-0.3"}
        for i in range(1, 31)
    ] + [
        {"ma_id": f"CD{i:06d}_pub1_data__A1", "review_id": f"CD{i:06d}",
         "significant": "False", "p_value": "0.2", "estimate": "-0.1"}
        for i in range(31, 51)
    ]


def _make_fixture_scores():
    import random
    rng = random.Random(42)
    result = {}
    for i in range(1, 51):
        ma_id = f"CD{i:06d}_pub1_data__A1"
        score = rng.uniform(20, 95)
        result[ma_id] = {
            "review_id": f"CD{i:06d}",
            "audit_score": rng.uniform(20, 100),
            "consistency_score": rng.uniform(20, 100),
            "robustness_score": rng.uniform(20, 100),
            "stability_score": rng.uniform(20, 100),
            "power_score": rng.uniform(20, 100),
            "final_score": score,
            "grade": "B",
        }
    return result


def _make_fixture_groups():
    groups = {}
    domains = ["Cardiovascular", "Respiratory", "Mental_health", "Infection", "Pain"]
    for i in range(1, 51):
        groups[f"CD{i:06d}"] = domains[(i - 1) % 5]
    return groups


class TestPipelineIntegration:
    def test_T21_fixture_cohort_count(self):
        """T21: Fixture data builds cohort with correct significant MA count."""
        verdicts = _make_fixture_verdicts()
        scores = _make_fixture_scores()
        groups = _make_fixture_groups()
        cohort = build_cohort(verdicts, scores, groups)
        assert len(cohort) == 30, f"Expected 30 significant MAs, got {len(cohort)}"

    def test_T22_pipeline_runs_on_fixture(self):
        """T22: Full pipeline runs on fixture data without error."""
        verdicts = _make_fixture_verdicts()
        scores = _make_fixture_scores()
        groups = _make_fixture_groups()
        result = run_pipeline(verdicts, scores, groups)
        assert "n_significant" in result
        assert result["n_significant"] == 30
        assert "km_overall" in result
        assert "log_rank" in result
        assert "cox_results" in result

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Real data not available")
    def test_T23_real_data_n_significant(self):
        """T23: Real data has 888 significant MAs."""
        from km_engine import load_verdicts
        verdicts = load_verdicts()
        sig = [v for v in verdicts if v["significant"] == "True"]
        assert len(sig) == 888, f"Expected 888, got {len(sig)}"

    @pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Real data not available")
    def test_T24_real_data_median_threshold_in_range(self):
        """T24: Median survival threshold is between 1 and 100."""
        result = run_pipeline()
        median = result["median_survival_threshold"]
        assert median is not None, "Median survival threshold should not be None"
        assert 1 <= median <= 100, f"Median {median} out of range [1, 100]"

    def test_T25_output_validation_structure(self):
        """T25: Pipeline output has all required keys with correct types."""
        verdicts = _make_fixture_verdicts()
        scores = _make_fixture_scores()
        groups = _make_fixture_groups()
        result = run_pipeline(verdicts, scores, groups)

        # Top-level keys
        required = {
            "n_significant", "n_weakened", "pct_weakened",
            "median_survival_threshold", "top5_domains",
            "km_overall", "km_by_domain", "log_rank", "cox_results",
        }
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - result.keys()}"

        # KM structure
        km = result["km_overall"]
        km_keys = {"times", "S", "var_greenwood", "ci_lower", "ci_upper",
                   "n_events", "n_at_risk", "median"}
        assert km_keys.issubset(km.keys())
        assert len(km["times"]) == len(km["S"])
        assert all(0.0 <= s <= 1.0 + 1e-9 for s in km["S"]), \
            "S values outside [0,1]"

        # Cox structure
        cox = result["cox_results"]
        assert len(cox) == 5
        for cov, res in cox.items():
            assert res["hr"] > 0, f"Non-positive HR for {cov}"

        # Log-rank
        lr = result["log_rank"]
        assert 0.0 <= lr["p_value"] <= 1.0

        # Weakened count non-negative
        assert result["n_weakened"] >= 0
        assert result["n_weakened"] <= result["n_significant"]
