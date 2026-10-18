"""Leakage prevention: (1) feature-name audit, (2) true point-in-time replay test."""
import numpy as np
import pandas as pd
from features.build import build_panel, feature_columns

FORBIDDEN = ("fwd", "y_up", "y_risk", "future", "target", "label", "_up_q", "_down_q")


def audit_feature_names(cols):
    bad = [c for c in cols if any(t in c.lower() for t in FORBIDDEN)]
    if bad:
        raise AssertionError(f"Leaky feature names: {bad}")
    return True


def pit_replay_test(panel: pd.DataFrame, n_dates: int = 4, seed: int = 7, engine=None, rtol=1e-3) -> dict:
    """Pick random past dates t, rebuild ALL features using only data <= t (as if live on day t),
    and compare with the full-history panel. Any difference => a feature peeks into the future."""
    cols = feature_columns(panel)
    dates = np.sort(panel["date"].unique())
    lo, hi = int(len(dates) * 0.45), len(dates) - 8
    rng = np.random.default_rng(seed)
    picks = rng.choice(np.arange(lo, hi), size=min(n_dates, hi - lo), replace=False)
    bad = {}
    for i in picks:
        t = pd.Timestamp(dates[i])
        p2, _, _ = build_panel(max_date=t, engine=engine, include_news=("news_sent_3d" in panel.columns))
        a = panel[panel.date == t].set_index("symbol")[cols]
        b = p2[p2.date == t].set_index("symbol")[cols].reindex(a.index)
        for c in cols:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            ok = np.isclose(x, y, rtol=rtol, atol=1e-6, equal_nan=True)
            if not ok.all():
                bad.setdefault(c, []).append(str(t.date()))
    return {"dates_tested": [str(pd.Timestamp(dates[i]).date()) for i in picks], "mismatched_features": bad,
            "passed": len(bad) == 0}
