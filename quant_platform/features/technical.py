"""Vectorised stock factors/technicals on wide frames (index=date, columns=symbol).
Every feature at row t uses data up to and including close t ONLY (rolling/ewm/shift(+k)); targets use shift(-k)
and live in build.py under names fwd_* / y_* so they can never be selected as features."""
import numpy as np
import pandas as pd


def _rank(df):
    return df.rank(axis=1, pct=True)


def compute_stock_features(adj, close, open_, high, low, volume, nifty_adj) -> dict:
    f = {}
    r1 = adj.pct_change(fill_method=None).clip(-0.3, 0.3)
    lr = np.log1p(r1)
    nr = nifty_adj.pct_change(fill_method=None)
    f["ret_1d"] = r1
    for k in (3, 5, 10, 20, 60, 120):
        f[f"ret_{k}d"] = adj.pct_change(k, fill_method=None)
    f["mom_120_20"] = adj.shift(20) / adj.shift(120) - 1            # momentum skipping last month
    f["reversal_5d"] = -f["ret_5d"]
    for w in (10, 20, 60):
        f[f"vol_{w}d"] = lr.rolling(w, min_periods=int(w * 0.8)).std()
    f["downside_vol_20d"] = np.sqrt((lr.clip(upper=0) ** 2).rolling(20, min_periods=16).mean())
    hl = np.log(high / low.replace(0, np.nan)) ** 2
    f["parkinson_20d"] = np.sqrt(hl.rolling(20, min_periods=16).mean() / (4 * np.log(2)))
    pc = close.shift(1)
    tr = pd.DataFrame(np.maximum(np.maximum((high - low).values, (high - pc).abs().values), (low - pc).abs().values),
                      index=adj.index, columns=adj.columns)
    f["atr_pct_14"] = (tr / close).rolling(14, min_periods=10).mean()
    for w in (20, 50, 200):
        f[f"gap_sma{w}"] = adj / adj.rolling(w, min_periods=int(w * 0.8)).mean() - 1
    ema12, ema26 = adj.ewm(span=12, adjust=False).mean(), adj.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / adj
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    d = adj.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    sd20 = adj.rolling(20, min_periods=16).std()
    f["bb_pct"] = (adj - adj.rolling(20, min_periods=16).mean()) / (2 * sd20.replace(0, np.nan))
    v = volume.replace(0, np.nan)
    lv = np.log1p(v)
    f["volume_z_20"] = (lv - lv.rolling(20, min_periods=16).mean()) / lv.rolling(20, min_periods=16).std().replace(0, np.nan)
    f["liquidity_20"] = np.log((adj * v).rolling(20, min_periods=16).mean())
    f["dist_hi_252"] = adj / adj.rolling(252, min_periods=120).max() - 1
    f["dist_lo_252"] = adj / adj.rolling(252, min_periods=120).min() - 1
    f["skew_60d"] = lr.rolling(60, min_periods=48).skew()
    nvar = nr.rolling(60, min_periods=48).var()
    beta = r1.rolling(60, min_periods=48).cov(nr).div(nvar, axis=0)
    f["beta_60d"] = beta
    f["corr_nifty_60d"] = r1.rolling(60, min_periods=48).corr(nr)
    f["idio_vol_60d"] = np.sqrt((f["vol_60d"] ** 2 - (beta ** 2).mul(nr.rolling(60, min_periods=48).var(), axis=0)).clip(lower=0))
    f["rel_str_5d"] = f["ret_5d"].sub(nifty_adj.pct_change(5, fill_method=None), axis=0)
    f["rel_str_20d"] = f["ret_20d"].sub(nifty_adj.pct_change(20, fill_method=None), axis=0)
    f["rel_str_60d"] = f["ret_60d"].sub(nifty_adj.pct_change(60, fill_method=None), axis=0)
    f["gap_1d"] = open_ / pc - 1
    f["intraday_ret"] = close / open_ - 1
    # cross-sectional ranks (relative to the universe on the same day -> no look-ahead)
    for k in ("ret_5d", "ret_20d", "mom_120_20", "vol_20d", "rsi_14", "volume_z_20", "rel_str_20d", "dist_hi_252"):
        f[f"cs_{k}"] = _rank(f[k])
    return f


def compute_targets(adj, nifty_adj, alpha_h, risk_h, risk_th) -> dict:
    t = {}
    fr = adj.shift(-alpha_h) / adj - 1
    nf = nifty_adj.shift(-alpha_h) / nifty_adj - 1
    t[f"fwd_ret_{alpha_h}d"] = fr
    t[f"fwd_excess_{alpha_h}d"] = fr.sub(nf, axis=0)
    y = (t[f"fwd_excess_{alpha_h}d"] > 0).astype(float).where(fr.notna())
    t["y_up"] = y
    rr = adj.shift(-risk_h) / adj - 1
    t[f"fwd_ret_{risk_h}d"] = rr
    t["y_risk"] = (rr <= risk_th).astype(float).where(rr.notna())
    return t
