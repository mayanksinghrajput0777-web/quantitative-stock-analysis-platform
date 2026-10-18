import numpy as np
import pandas as pd
import config


def perf_metrics(ret: pd.Series, bench: pd.Series = None, rf=config.RF_ANNUAL, turnover: pd.Series = None) -> dict:
    r = ret.dropna()
    if len(r) < 5:
        return {}
    ann = 252
    eq = (1 + r).cumprod()
    years = len(r) / ann
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(ann)
    rf_d = rf / ann
    ex = r - rf_d
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ann)
    dd = eq / eq.cummax() - 1
    out = {"cagr": float(cagr), "ann_vol": float(vol), "sharpe": float(ex.mean() * ann / vol) if vol > 0 else None,
           "sortino": float(ex.mean() * ann / dn) if dn > 0 else None, "max_drawdown": float(dd.min()),
           "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else None, "total_return": float(eq.iloc[-1] - 1),
           "hit_rate": float((r > 0).mean()), "var95_1d": float(np.percentile(r, 5)),
           "es95_1d": float(r[r <= np.percentile(r, 5)].mean()), "n_days": int(len(r)),
           "skew": float(r.skew()), "best_day": float(r.max()), "worst_day": float(r.min())}
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.mean()); out["annual_turnover"] = float(turnover.sum() / years)
    if bench is not None:
        b = bench.reindex(r.index).dropna(); rr = r.reindex(b.index)
        if len(b) > 30:
            cov = np.cov(rr, b)[0, 1]; beta = cov / b.var() if b.var() > 0 else np.nan
            te = (rr - b).std() * np.sqrt(ann)
            out.update({"beta": float(beta), "alpha_ann": float((rr.mean() - beta * b.mean()) * ann),
                        "info_ratio": float((rr - b).mean() * ann / te) if te > 0 else None,
                        "bench_cagr": float((1 + b).prod() ** (1 / (len(b) / ann)) - 1),
                        "bench_sharpe": float((b - rf_d).mean() * ann / (b.std() * np.sqrt(ann))),
                        "bench_max_drawdown": float(((1 + b).cumprod() / (1 + b).cumprod().cummax() - 1).min())})
    return out
