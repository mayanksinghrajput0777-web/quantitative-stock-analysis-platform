import numpy as np
import pandas as pd


def ewma_vol(returns, lam=0.94):
    r = pd.Series(returns).dropna().values
    if len(r) < 20:
        return np.nan
    v = np.var(r[:20])
    for x in r[20:]:
        v = lam * v + (1 - lam) * x * x
    return float(np.sqrt(v))


def garch_forecast(returns, horizon=5, window=750):
    """GJR-GARCH(1,1) with Student-t errors. Returns dict(vol_1d, vol_5d (cumulative), vol_ann, model)."""
    r = pd.Series(returns).dropna().tail(window)
    if len(r) >= 250:
        try:
            from arch import arch_model
            am = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
            res = am.fit(disp="off", show_warning=False)
            if res.convergence_flag == 0:
                var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
                if np.all(np.isfinite(var)) and var.min() > 0:
                    return {"vol_1d": float(np.sqrt(var[0]) / 100), "vol_5d": float(np.sqrt(var.sum()) / 100),
                            "vol_ann": float(np.sqrt(var.mean() * 252) / 100), "model": "gjr_garch_t"}
        except Exception:
            pass
    v = ewma_vol(r)
    return {"vol_1d": v, "vol_5d": v * np.sqrt(horizon) if v == v else np.nan,
            "vol_ann": v * np.sqrt(252) if v == v else np.nan, "model": "ewma"}
