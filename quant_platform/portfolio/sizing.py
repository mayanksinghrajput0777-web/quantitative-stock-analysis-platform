"""Position sizing: volatility targeting, fractional Kelly, share/rupee conversion with ATR/GARCH stops."""
import numpy as np
import pandas as pd
import config


def inverse_vol_weights(vols: pd.Series) -> pd.Series:
    w = 1.0 / vols.replace(0, np.nan)
    w = w.fillna(w.median())
    return w / w.sum()


def vol_target_scale(weights: pd.Series, cov_ann: pd.DataFrame, target_vol=config.TARGET_VOL, max_leverage=1.0) -> float:
    w = weights.reindex(cov_ann.index).fillna(0).values
    pv = float(np.sqrt(w @ cov_ann.values @ w))
    return float(min(target_vol / pv, max_leverage)) if pv > 0 else 1.0


def fractional_kelly(mu_excess_ann: pd.Series, var_ann: pd.Series, fraction=0.25, cap=config.MAX_WEIGHT) -> pd.Series:
    k = (mu_excess_ann / var_ann.replace(0, np.nan)).clip(lower=0).fillna(0) * fraction
    return k.clip(upper=cap)


def position_table(weights: pd.Series, prices: pd.Series, capital=config.CAPITAL, garch_vol_5d: pd.Series = None,
                   stop_mult=2.0) -> pd.DataFrame:
    w = weights[weights > 1e-4]
    df = pd.DataFrame({"weight": w})
    df["price"] = prices.reindex(df.index)
    df["amount_inr"] = df.weight * capital
    df["shares"] = np.floor(df.amount_inr / df.price).astype("float")
    df["invested_inr"] = df.shares * df.price
    if garch_vol_5d is not None:
        sl = (stop_mult * garch_vol_5d.reindex(df.index)).clip(0.02, 0.15)
        df["stop_loss_pct"] = sl
        df["stop_price"] = df.price * (1 - sl)
    return df.round(4).reset_index(names="symbol")
