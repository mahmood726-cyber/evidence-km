# evidence-km

Kaplan-Meier survival analysis of meta-analytic significance across trust-score thresholds.

- `km_engine.py` — KM survival curve (Greenwood variance, log-log CI), log-rank test, and a simplified Cox proportional-hazards approximation over the trust-component covariates.
- `build_dashboard.py` — builds the single-file HTML dashboard (`dashboard.html` / `index.html`) from the engine output.
- `tests/test_km.py` — engine unit tests (`pytest -q`).

The engine treats each significant meta-analysis as an observation whose "event time" is its final trust score (0-100) and tracks survival of significance as the trust threshold rises.
