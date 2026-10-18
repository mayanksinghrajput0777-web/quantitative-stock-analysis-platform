import numpy as np
import pandas as pd
from scipy.stats import norm


def hist_var_es(returns, alpha=0.05):
    """Historical VaR / Expected Shortfall as (negative) returns."""
    r = pd.Series(returns).dropna().values
    if len(r) < 60:
        return np.nan, np.nan
    var = np.percentile(r, 100 * alpha)
    tail = r[r <= var]
    return float(var), float(tail.mean() if len(tail) else var)


def parametric_var_es(returns, alpha=0.05):
    r = pd.Series(returns).dropna().values
    if len(r) < 60:
        return np.nan, np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    z = norm.ppf(alpha)
    return float(mu + z * sd), float(mu - sd * norm.pdf(z) / alpha)


def downside_deviation(returns, mar=0.0, ann=252):
    r = pd.Series(returns).dropna().values
    if len(r) < 30:
        return np.nan
    d = np.minimum(r - mar, 0)
    return float(np.sqrt((d ** 2).mean()) * np.sqrt(ann))
