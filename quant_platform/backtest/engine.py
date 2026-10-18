"""Walk-forward (out-of-sample) backtest with weight drift, rebalancing, execution lag, transaction costs,
slippage and square-root market-impact. Signals come ONLY from walk-forward OOF predictions."""
import numpy as np
import pandas as pd
import config
from portfolio import optimizer, sizing
from backtest.metrics import perf_metrics


def _select(score_row, risk_row, k, risk_filter_pct):
    s = score_row.dropna()
    if risk_row is not None and risk_filter_pct < 1:
        r = risk_row.reindex(s.index)
        cut = r.quantile(risk_filter_pct)
        s = s[r.fillna(0) <= cut]
    return list(s.sort_values(ascending=False).head(k).index)


def run_backtest(oof: pd.DataFrame, adj: pd.DataFrame, nifty: pd.Series, adv_value: pd.DataFrame = None,
                 regime_p_stress: pd.Series = None, top_k=config.BT_TOP_K, rebalance_days=config.BT_REBALANCE_DAYS,
                 exec_lag=config.BT_EXEC_LAG, cost_bps=config.COST_BPS, slip_bps=config.SLIPPAGE_BPS,
                 impact_coef_bps=config.IMPACT_COEF_BPS, sizing_method="equal", max_weight=config.MAX_WEIGHT,
                 risk_filter_pct=0.8, capital=config.CAPITAL, rf=config.RF_ANNUAL, stress_exposure=0.5,
                 score_col="prob_up_raw") -> dict:
    o = oof.copy()
    o["score"] = 0.5 * o.groupby("date")[score_col].rank(pct=True) + 0.5 * o.groupby("date")["exp_excess_ret"].rank(pct=True)
    S = o.pivot(index="date", columns="symbol", values="score")
    R = o.pivot(index="date", columns="symbol", values="prob_risk_raw")
    rets = adj.pct_change(fill_method=None).clip(-0.3, 0.3)
    dates = adj.index[(adj.index >= S.index.min()) & (adj.index <= S.index.max())]
    pos = {d: i for i, d in enumerate(dates)}
    syms = list(adj.columns); n = len(syms); sidx = {s: i for i, s in enumerate(syms)}
    rebal = set(dates[::rebalance_days])
    pending = {}               # exec_date -> target weight vector
    w = np.zeros(n); cash = 1.0
    eq, rows = 1.0, []
    last_target = None
    for j, d in enumerate(dates):
        r = rets.loc[d].fillna(0).values
        pr = float(w @ r + cash * rf / 252)
        port = float(w.sum() + cash)
        # drift
        w = w * (1 + r) / (1 + pr); cash = cash * (1 + rf / 252) / (1 + pr)
        eq *= (1 + pr)
        turn = cost = 0.0
        # decide on rebalance date (information up to close d)
        if d in rebal and d in S.index:
            pick = _select(S.loc[d], R.loc[d] if d in R.index else None, top_k, risk_filter_pct)
            tgt = np.zeros(n)
            if pick:
                hist = rets.loc[:d, pick].tail(252)
                if sizing_method == "inv_vol":
                    ww = sizing.inverse_vol_weights(hist.tail(60).std())
                elif sizing_method in ("max_sharpe", "min_variance", "risk_parity"):
                    ww = optimizer.optimize(hist.fillna(0), sizing_method, max_w=max_weight, rf=rf)
                else:
                    ww = pd.Series(1.0 / len(pick), index=pick)
                ww = ww.clip(upper=max_weight); ww = ww / ww.sum() if ww.sum() > 0 else ww
                scale = 1.0
                if sizing_method == "vol_target":
                    ww = pd.Series(1.0 / len(pick), index=pick)
                    cov = optimizer.annual_cov(hist.fillna(0))
                    scale = sizing.vol_target_scale(ww, cov, config.TARGET_VOL, 1.0)
                if regime_p_stress is not None and d in regime_p_stress.index and regime_p_stress.loc[d] > 0.6:
                    scale *= stress_exposure
                for s, v in ww.items():
                    tgt[sidx[s]] = v * scale
            ex_idx = j + exec_lag
            if ex_idx < len(dates):
                pending[dates[ex_idx]] = tgt
        # execute pending trade at today's close
        if d in pending:
            tgt = pending.pop(d)
            trade = tgt - w
            turn = float(np.abs(trade).sum())
            bps = np.full(n, cost_bps + slip_bps)
            if adv_value is not None:
                adv = adv_value.loc[d].reindex(syms).values
                trade_val = np.abs(trade) * eq * capital
                part = np.where(np.isfinite(adv) & (adv > 0), trade_val / adv, 0)
                bps = bps + impact_coef_bps * np.sqrt(np.clip(part, 0, 1))
            cost = float((np.abs(trade) * bps / 1e4).sum())
            eq *= (1 - cost)
            w = tgt.copy(); cash = max(1.0 - w.sum(), 0.0); last_target = tgt
        rows.append({"date": d, "ret": pr - cost, "gross_ret": pr, "turnover": turn, "cost": cost, "exposure": float(w.sum())})
    res = pd.DataFrame(rows).set_index("date")
    res["equity"] = (1 + res["ret"]).cumprod()
    b = nifty.pct_change().reindex(res.index).fillna(0)
    res["bench_ret"] = b; res["bench_equity"] = (1 + b).cumprod()
    m = perf_metrics(res["ret"], b, rf, res["turnover"])
    m["gross_sharpe"] = perf_metrics(res["gross_ret"], None, rf).get("sharpe")
    m["total_cost_drag_pct"] = float(res["cost"].sum())
    m["avg_exposure"] = float(res["exposure"].mean())
    final_w = pd.Series(last_target if last_target is not None else np.zeros(n), index=syms)
    return {"daily": res, "metrics": m, "final_weights": final_w[final_w > 1e-6].sort_values(ascending=False)}
