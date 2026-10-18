import numpy as np
import pandas as pd


def monte_carlo(daily_ret: pd.Series, n_paths=5000, horizon=252, block=5, seed=42) -> dict:
    """Block-bootstrap of the strategy's daily returns -> distribution of 1-year outcomes."""
    r = daily_ret.dropna().values
    if len(r) < 60:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(r) - block, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    sims = r[idx]
    eq = np.cumprod(1 + sims, axis=1)
    term = eq[:, -1] - 1
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    pct = [5, 25, 50, 75, 95]
    fan = {f"p{p}": np.percentile(eq, p, axis=0)[::5].round(4).tolist() for p in pct}
    q5 = np.percentile(term, 5)
    return {"horizon_days": horizon, "n_paths": n_paths, "terminal_return_pct": {f"p{p}": float(np.percentile(term, p)) for p in pct},
            "prob_loss": float((term < 0).mean()), "prob_dd_gt_10": float((dd < -0.10).mean()),
            "prob_dd_gt_20": float((dd < -0.20).mean()), "median_max_dd": float(np.median(dd)),
            "worst5pct_max_dd": float(np.percentile(dd, 5)), "es5_terminal": float(term[term <= q5].mean()), "fan": fan}
