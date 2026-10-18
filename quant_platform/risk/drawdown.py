import numpy as np
import pandas as pd


def drawdown_series(prices: pd.Series) -> pd.Series:
    p = prices.dropna()
    return p / p.cummax() - 1


def drawdown_stats(prices: pd.Series, lookback=756):
    p = prices.dropna().tail(lookback)
    if len(p) < 30:
        return {"max_dd": np.nan, "cur_dd": np.nan, "dd_days": np.nan}
    dd = drawdown_series(p)
    peak_idx = p.idxmax() if dd.iloc[-1] < 0 else p.index[-1]
    last_peak = p[:p.index[-1]].idxmax()
    days = int((p.index > last_peak).sum())
    return {"max_dd": float(dd.min()), "cur_dd": float(dd.iloc[-1]), "dd_days": days}
