"""Portfolio optimisation: Max Sharpe, Min Variance, Risk Parity, Equal weight. Ledoit-Wolf shrinkage covariance."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import config


def annual_cov(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.dropna(how="any")
    if len(r) < 60:
        r = returns.fillna(0)
    lw = LedoitWolf().fit(r.values)
    return pd.DataFrame(lw.covariance_ * 252, index=returns.columns, columns=returns.columns)


def shrunk_mu(returns: pd.DataFrame, shrink=0.5) -> pd.Series:
    m = returns.mean() * 252
    return (1 - shrink) * m + shrink * m.mean()


def _solve(fun, n, max_w, x0=None, min_w=0.0):
    max_w = max(max_w, 1.0 / n + 1e-6)
    x0 = np.ones(n) / n if x0 is None else x0
    res = minimize(fun, x0, method="SLSQP", bounds=[(min_w, max_w)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    return w / w.sum()


def max_sharpe(mu, cov, rf=config.RF_ANNUAL, max_w=config.MAX_WEIGHT):
    mu, C = np.asarray(mu), np.asarray(cov)
    f = lambda w: -(w @ mu - rf) / np.sqrt(max(w @ C @ w, 1e-12))
    return _solve(f, len(mu), max_w)


def min_variance(cov, max_w=config.MAX_WEIGHT):
    C = np.asarray(cov)
    return _solve(lambda w: w @ C @ w, C.shape[0], max_w)


def risk_parity(cov, max_w=config.MAX_WEIGHT):
    C = np.asarray(cov); n = C.shape[0]
    def f(w):
        rc = w * (C @ w); tot = max(w @ C @ w, 1e-12)
        return np.sum((rc / tot - 1.0 / n) ** 2) * 1e4
    return _solve(f, n, max_w, min_w=1e-4)


def optimize(returns: pd.DataFrame, method="max_sharpe", max_w=config.MAX_WEIGHT, rf=config.RF_ANNUAL,
             mu_override: pd.Series = None) -> pd.Series:
    cols = list(returns.columns)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    cov = annual_cov(returns)
    mu = mu_override.reindex(cols).fillna(0).values if mu_override is not None else shrunk_mu(returns).values
    try:
        if method == "max_sharpe":
            w = max_sharpe(mu, cov, rf, max_w)
        elif method == "min_variance":
            w = min_variance(cov, max_w)
        elif method == "risk_parity":
            w = risk_parity(cov, max_w)
        else:
            w = np.ones(len(cols)) / len(cols)
    except Exception:
        w = np.ones(len(cols)) / len(cols)
    return pd.Series(w, index=cols)


def portfolio_stats(w: pd.Series, returns: pd.DataFrame, rf=config.RF_ANNUAL) -> dict:
    cov = annual_cov(returns[w.index]); mu = shrunk_mu(returns[w.index])
    vol = float(np.sqrt(w.values @ cov.values @ w.values)); ret = float(w.values @ mu.values)
    rc = w.values * (cov.values @ w.values) / max(vol ** 2, 1e-12)
    return {"exp_return": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan,
            "risk_contrib": dict(zip(w.index, rc.round(4))), "effective_n": float(1 / np.sum(w.values ** 2))}
