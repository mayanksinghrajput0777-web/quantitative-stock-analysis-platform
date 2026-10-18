"""Market-wide (date-indexed) features: NIFTY, India VIX, FII/DII positioning. All FII features are lagged."""
import numpy as np
import pandas as pd
import config


def _z(s, w=30):
    return (s - s.rolling(w, min_periods=10).mean()) / s.rolling(w, min_periods=10).std().replace(0, np.nan)


def nifty_vix_features(nifty: pd.Series, vix: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=nifty.index)
    r = nifty.pct_change(fill_method=None)
    f["nifty_ret_1d"] = r
    f["nifty_ret_5d"] = nifty.pct_change(5, fill_method=None)
    f["nifty_ret_20d"] = nifty.pct_change(20, fill_method=None)
    f["nifty_vol_20d"] = r.rolling(20, min_periods=16).std()
    f["nifty_gap_sma50"] = nifty / nifty.rolling(50, min_periods=40).mean() - 1
    f["nifty_gap_sma200"] = nifty / nifty.rolling(200, min_periods=160).mean() - 1
    f["nifty_dd_252"] = nifty / nifty.rolling(252, min_periods=120).max() - 1
    f["vix_close"] = vix
    f["vix_chg_1d"] = vix.pct_change(fill_method=None)
    f["vix_chg_5d"] = vix.pct_change(5, fill_method=None)
    f["vix_z_60"] = _z(vix, 60)
    f["vix_pct_252"] = vix.rolling(252, min_periods=120).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    return f


def fii_features(oi: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """oi = participant_oi rows. Output is aligned to `calendar` and shifted by FII_LAG_DAYS trading days
    (the NSE file for day D is published after the close, so it is usable from D+1)."""
    out = pd.DataFrame(index=calendar)
    if oi is None or oi.empty:
        return out
    oi = oi.copy(); oi["date"] = pd.to_datetime(oi["date"])
    tot = oi[oi.client_type == "TOTAL"].set_index("date")
    tot_oi = tot["fut_idx_short"].astype(float)
    feats = {}
    for p in ("FII", "DII", "Pro", "Client"):
        d = oi[oi.client_type == p].set_index("date").sort_index()
        if d.empty:
            continue
        net = d["fut_idx_long"] - d["fut_idx_short"]
        k = p.lower()
        feats[f"{k}_net_fut_idx"] = net
        feats[f"{k}_net_pct_oi"] = net / tot_oi.reindex(d.index)
        if p == "FII":
            feats["fii_net_fut_stk"] = d["fut_stk_long"] - d["fut_stk_short"]
            calls = d["opt_idx_call_long"] + d["opt_idx_call_short"]
            puts = d["opt_idx_put_long"] + d["opt_idx_put_short"]
            feats["fii_opt_pcr"] = puts / calls.replace(0, np.nan)
            feats["fii_opt_net_skew"] = ((d["opt_idx_call_long"] - d["opt_idx_call_short"]) -
                                         (d["opt_idx_put_long"] - d["opt_idx_put_short"]))
            feats["fii_idx_long_share"] = d["fut_idx_long"] / (d["fut_idx_long"] + d["fut_idx_short"])
    raw = pd.DataFrame(feats).reindex(calendar).ffill(limit=3).shift(config.FII_LAG_DAYS)   # <-- publication lag
    for c in raw.columns:
        out[c] = raw[c]
    for c in ("fii_net_fut_idx", "dii_net_fut_idx", "pro_net_fut_idx", "fii_net_pct_oi"):
        if c in raw:
            out[f"{c}_z30"] = _z(raw[c], 30)
            out[f"{c}_chg1"] = raw[c].diff()
    if "fii_net_fut_idx" in raw:
        out["fii_net_ma5"] = raw["fii_net_fut_idx"].rolling(5, min_periods=3).mean()
        out["fii_net_chg5"] = raw["fii_net_fut_idx"].diff(5)
        if "dii_net_fut_idx" in raw:
            out["fii_dii_div"] = raw["fii_net_fut_idx"] - raw["dii_net_fut_idx"]
    return out
