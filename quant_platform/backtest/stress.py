"""Stress testing: historical windows, hypothetical NIFTY/VIX shocks (beta-mapped), single-stock gap-downs."""
import numpy as np
import pandas as pd

NAMED_WINDOWS = {   # edit freely; windows outside your data range are skipped automatically
    "Feb-Mar 2022 sell-off": ("2022-02-14", "2022-03-08"),
    "2022 H1 correction": ("2022-01-01", "2022-06-30"),
    "Jun 2024 election-result shock": ("2024-05-30", "2024-06-10"),
    "Oct 2024-Mar 2025 FII-selling correction": ("2024-10-01", "2025-03-31"),
    "Apr 2025 tariff shock": ("2025-04-01", "2025-04-15"),
}


def historical_windows(weights: pd.Series, rets: pd.DataFrame, nifty_ret: pd.Series, windows=None) -> pd.DataFrame:
    rows = []
    w = weights[weights > 0]
    for name, (a, b) in (windows or NAMED_WINDOWS).items():
        r = rets.loc[a:b, w.index]
        if len(r) < 3:
            continue
        avail = r.notna().mean() > 0.8
        ww = w[avail[avail].index]
        if ww.sum() <= 0:
            continue
        ww = ww / ww.sum() * w.sum()
        pr = (r[ww.index].fillna(0) @ ww)
        eq = (1 + pr).cumprod()
        nb = nifty_ret.loc[a:b]
        rows.append({"scenario": name, "days": len(pr), "portfolio_return": float(eq.iloc[-1] - 1),
                     "portfolio_max_dd": float((eq / eq.cummax() - 1).min()),
                     "nifty_return": float((1 + nb).prod() - 1), "coverage": float(ww.sum() / w.sum())})
    return pd.DataFrame(rows)


def worst_windows(nifty_ret: pd.Series, weights: pd.Series, rets: pd.DataFrame, lengths=(1, 5, 20)) -> pd.DataFrame:
    w = weights[weights > 0]
    pr = rets[w.index].fillna(0) @ (w / w.sum() * w.sum())
    rows = []
    for L in lengths:
        roll = (1 + pr).rolling(L).apply(np.prod, raw=True) - 1
        rn = (1 + nifty_ret).rolling(L).apply(np.prod, raw=True) - 1
        d = roll.idxmin()
        rows.append({"scenario": f"worst {L}-day window in sample", "end_date": str(d.date()),
                     "portfolio_return": float(roll.min()), "nifty_return": float(rn.loc[d])})
    return pd.DataFrame(rows)


def hypothetical(weights: pd.Series, betas: pd.Series, vix_beta: float = None, vix_level: float = None) -> pd.DataFrame:
    """Portfolio P&L under shocks: ΔNIFTY mapped to stocks by beta; VIX +x% mapped to NIFTY via regression slope."""
    w = weights[weights > 0]
    b = betas.reindex(w.index).fillna(1.0)
    pbeta = float((w * b).sum())
    rows = []
    for shock in (-0.05, -0.10, -0.15, -0.20, 0.05):
        rows.append({"scenario": f"NIFTY {shock:+.0%}", "nifty_shock": shock, "portfolio_impact": pbeta * shock})
    if vix_beta is not None:
        for v in (0.25, 0.50, 1.00):
            sh = vix_beta * np.log1p(v)
            rows.append({"scenario": f"India VIX {v:+.0%} (implied NIFTY {sh:+.1%})", "nifty_shock": sh, "portfolio_impact": pbeta * sh})
    for s in w.sort_values(ascending=False).index[:3]:
        rows.append({"scenario": f"{s} gaps down -15%", "nifty_shock": np.nan, "portfolio_impact": -0.15 * float(w[s])})
    out = pd.DataFrame(rows); out["portfolio_beta"] = pbeta
    return out


def vix_nifty_slope(nifty: pd.Series, vix: pd.Series) -> float:
    d = pd.concat([np.log(vix).diff(), nifty.pct_change()], axis=1, join="inner").dropna()
    d.columns = ["dv", "dn"]
    return float(np.polyfit(d.dv, d.dn, 1)[0]) if len(d) > 60 else None
