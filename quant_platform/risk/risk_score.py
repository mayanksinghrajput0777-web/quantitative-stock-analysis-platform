"""Dynamic downside risk score 0-100 (absolute, interpretable scaling) + level."""
import numpy as np
import config


def _scale(x, lo, hi):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def risk_score(m: dict, vix: float = None):
    """m keys: garch_vol_ann, es95_1d (negative), cur_drawdown (negative), downside_dev_ann, prob_risk"""
    parts = {}
    for k, (lo, hi) in config.RISK_SCALES.items():
        v = m.get(k)
        if v is not None and k in ("es95_1d", "cur_drawdown") and v == v:
            v = abs(v)
        parts[k] = _scale(v, lo, hi)
    w = {k: config.RISK_WEIGHTS[k] for k in parts if not np.isnan(parts[k])}
    if not w:
        return np.nan, "n/a", parts
    tw = sum(w.values())
    s = sum(parts[k] * w[k] for k in w) / tw
    if vix is not None and vix == vix:
        s = (1 - config.VIX_BLEND) * s + config.VIX_BLEND * _scale(vix, *config.VIX_SCALE)
    return float(s), level(s), parts


def level(score: float) -> str:
    for th, name in config.RISK_LEVELS:
        if score < th:
            return name
    return "Extreme"
