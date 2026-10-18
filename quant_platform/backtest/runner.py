"""Runs walk-forward backtests for several sizing methods, Monte Carlo, stress tests; stores in PostgreSQL."""
import json, datetime as dt
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine, upsert_df
from backtest.engine import run_backtest
from backtest.montecarlo import monte_carlo
from backtest import stress
from backtest.metrics import perf_metrics

METHODS = ["equal", "inv_vol", "vol_target", "max_sharpe", "risk_parity"]


def run_all(engine=None, methods=METHODS, top_k=config.BT_TOP_K) -> dict:
    engine = engine or get_engine()
    f = config.ART_DIR / "oof_predictions.pkl"
    if not f.exists():
        raise RuntimeError("No walk-forward predictions. Run: python run_pipeline.py train")
    oof = pd.read_pickle(f)
    px = read_sql("SELECT symbol, date, adj_close, close, volume FROM prices_daily", engine=engine, parse_dates=["date"])
    ix = read_sql("SELECT * FROM index_prices", engine=engine, parse_dates=["date"])
    adj = px.pivot(index="date", columns="symbol", values="adj_close").sort_index().ffill(limit=3)
    adv = (px.assign(v=px.close * px.volume).pivot(index="date", columns="symbol", values="v").sort_index()).rolling(20, min_periods=5).mean()
    nifty = ix[ix.index_name == config.NIFTY_NAME].set_index("date")["close"].sort_index()
    vix = ix[ix.index_name == config.VIX_NAME].set_index("date")["close"].sort_index()
    rg = read_sql("SELECT date, p_stress FROM market_regime", engine=engine, parse_dates=["date"]).set_index("date")["p_stress"]
    # regime probability used at t is filtered (causal) -> safe
    print(f"📈 backtesting {len(oof)} walk-forward predictions ({oof.date.min().date()} .. {oof.date.max().date()})")
    results, summary = {}, []
    for m in methods:
        r = run_backtest(oof, adj, nifty, adv, regime_p_stress=rg, top_k=top_k, sizing_method=m)
        results[m] = r
        mm = r["metrics"]
        summary.append({"method": m, **{k: mm.get(k) for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
                                                               "annual_turnover", "gross_sharpe", "beta", "alpha_ann", "info_ratio")}})
        print(f"   {m:12s} CAGR {mm['cagr']:+.1%}  Sharpe {mm['sharpe']:.2f}  Sortino {mm['sortino']:.2f}  MaxDD {mm['max_drawdown']:.1%}  "
              f"(gross Sharpe {mm['gross_sharpe']:.2f}) | NIFTY CAGR {mm.get('bench_cagr', float('nan')):+.1%}")
    best = max(results, key=lambda k: (results[k]["metrics"].get("sharpe") or -9))
    chosen = "equal" if "equal" in results else best
    mc = monte_carlo(results[chosen]["daily"]["ret"])
    rets = adj.pct_change(fill_method=None)
    fw = results[chosen]["final_weights"]
    if fw.empty:
        fw = pd.Series(1 / 10, index=adj.columns[:10])
    nret = nifty.pct_change()
    beta = read_sql("SELECT symbol, beta FROM risk_metrics WHERE asof_date=(SELECT MAX(asof_date) FROM risk_metrics)", engine=engine).set_index("symbol")["beta"]
    st = pd.concat([stress.historical_windows(fw, rets, nret), stress.worst_windows(nret, fw, rets),
                    stress.hypothetical(fw, beta, stress.vix_nifty_slope(nifty, vix))], ignore_index=True)
    run_id = "bt_" + dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    params = {"top_k": top_k, "rebalance_days": config.BT_REBALANCE_DAYS, "exec_lag": config.BT_EXEC_LAG, "cost_bps": config.COST_BPS,
              "slippage_bps": config.SLIPPAGE_BPS, "impact_coef_bps": config.IMPACT_COEF_BPS, "methods": methods, "primary": chosen}
    upsert_df(pd.DataFrame([{"run_id": run_id, "name": f"walk_forward_{chosen}", "params": params, "metrics": results[chosen]["metrics"],
                             "extras": {"comparison": summary, "monte_carlo": mc, "stress": json.loads(st.to_json(orient="records")),
                                        "final_weights": results[chosen]["final_weights"].round(4).to_dict()}}]), "backtest_runs", engine=engine)
    for m, r in results.items():
        d = r["daily"].reset_index()
        upsert_df(pd.DataFrame({"run_id": f"{run_id}:{m}", "date": d["date"], "equity": d["equity"], "ret": d["ret"],
                                "benchmark_equity": d["bench_equity"], "turnover": d["turnover"]}), "backtest_equity", engine=engine)
    pd.DataFrame(summary).to_csv(config.REPORT_DIR / "backtest_comparison.csv", index=False)
    st.to_csv(config.REPORT_DIR / "stress_tests.csv", index=False)
    (config.REPORT_DIR / "monte_carlo.json").write_text(json.dumps({k: v for k, v in mc.items() if k != "fan"}, indent=2))
    print(f"🎲 Monte Carlo (1y): median {mc['terminal_return_pct']['p50']:+.1%}, 5th pct {mc['terminal_return_pct']['p5']:+.1%}, "
          f"P(loss) {mc['prob_loss']:.0%}, P(DD>20%) {mc['prob_dd_gt_20']:.0%}")
    print(f"✅ backtest stored: run_id={run_id} primary={chosen}")
    return {"run_id": run_id, "summary": summary, "mc": mc, "stress": st, "results": results}
