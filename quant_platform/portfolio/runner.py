"""Builds today's recommended portfolio (3 optimisers + sizing) from live signals and stores it."""
import datetime as dt
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine, upsert_df
from portfolio import optimizer, sizing


def build_portfolio(engine=None, capital=config.CAPITAL, max_w=config.MAX_WEIGHT, n_max=15, methods=("max_sharpe", "min_variance", "risk_parity")):
    engine = engine or get_engine()
    pred = read_sql("SELECT * FROM predictions WHERE source='live' AND asof_date=(SELECT MAX(asof_date) FROM predictions WHERE source='live')", engine=engine)
    risk = read_sql("SELECT * FROM risk_metrics WHERE asof_date=(SELECT MAX(asof_date) FROM risk_metrics)", engine=engine)
    if pred.empty:
        raise RuntimeError("No live predictions. Run: python run_pipeline.py predict")
    n_min = 10          # optimisers need freedom: with k names and cap c, k*c close to 1 forces equal weights
    cand = pred[pred.signal == "BUY"].sort_values("score", ascending=False).head(n_max)
    if len(cand) < n_min:   # top up with best-ranked non-SELL names
        extra = pred[(pred.signal != "SELL") & (~pred.symbol.isin(cand.symbol))].sort_values("score", ascending=False)
        cand = pd.concat([cand, extra.head(n_min - len(cand))])
    max_w = max(max_w, 1.5 / max(len(cand), 1))
    syms = list(cand.symbol)
    px = read_sql("SELECT symbol, date, adj_close, close FROM prices_daily", engine=engine, parse_dates=["date"])
    px = px[px.symbol.isin(syms)]
    adj = px.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    rets = adj.pct_change(fill_method=None).tail(252).dropna(how="all").fillna(0)
    last_px = px.sort_values("date").groupby("symbol")["close"].last()
    # blend historical mean with model view (5d excess return annualised, shrunk)
    view = cand.set_index("symbol")["exp_excess_ret"] * 252 / config.ALPHA_H
    mu = 0.7 * optimizer.shrunk_mu(rets) + 0.3 * view.reindex(rets.columns).fillna(0).clip(-0.5, 0.5)
    run_id = "pf_" + dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out, rows = {}, []
    g5 = risk.set_index("symbol")["garch_vol_5d"]
    for m in methods:
        w = optimizer.optimize(rets, m, max_w=max_w, mu_override=mu)
        st = optimizer.portfolio_stats(w, rets)
        pt = sizing.position_table(w, last_px, capital, g5)
        out[m] = {"weights": w, "stats": st, "positions": pt}
        rows += [{"run_id": run_id, "method": m, "symbol": s, "weight": float(v)} for s, v in w.items() if v > 1e-4]
    upsert_df(pd.DataFrame(rows), "portfolio_results", engine=engine)
    # vol-targeted sizing on risk-parity
    cov = optimizer.annual_cov(rets)
    sc = sizing.vol_target_scale(out["risk_parity"]["weights"], cov, config.TARGET_VOL)
    out["vol_target_scale"] = sc
    print(f"✅ portfolio built from {len(syms)} candidates; vol-target scale (target {config.TARGET_VOL:.0%}) = {sc:.2f}")
    for m in methods:
        s = out[m]["stats"]; print(f"   {m:13s} exp_ret {s['exp_return']:+.1%} vol {s['vol']:.1%} sharpe {s['sharpe']:.2f} effN {s['effective_n']:.1f}")
    return out
