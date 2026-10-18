"""Service layer shared by FastAPI and Streamlit. Every function returns JSON-serialisable data."""
import json
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine


def _rec(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def health():
    try:
        n = read_sql("SELECT COUNT(*) AS n FROM prices_daily")["n"].iloc[0]
        return {"status": "ok", "db": get_engine().dialect.name, "price_rows": int(n)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def symbols():
    return _rec(read_sql("SELECT symbol, name, sector FROM dim_symbol ORDER BY symbol"))


def market_overview(days=500):
    ix = read_sql("SELECT index_name, date, close FROM index_prices", parse_dates=["date"])
    n = ix[ix.index_name == config.NIFTY_NAME].sort_values("date").tail(days)
    v = ix[ix.index_name == config.VIX_NAME].sort_values("date").tail(days)
    rg = read_sql("SELECT * FROM market_regime ORDER BY date DESC LIMIT :d", {"d": days}, parse_dates=["date"]).sort_values("date")
    oi = read_sql("SELECT date, fut_idx_long, fut_idx_short, client_type FROM participant_oi WHERE client_type IN ('FII','DII','Pro','Client') ORDER BY date", parse_dates=["date"])
    oi["net_fut_idx"] = oi.fut_idx_long - oi.fut_idx_short
    oi = oi[oi.date >= n.date.min()]
    last_r = rg.iloc[-1].to_dict() if len(rg) else {}
    return {"nifty": _rec(n[["date", "close"]]), "vix": _rec(v[["date", "close"]]), "regime": _rec(rg),
            "participants": _rec(oi[["date", "client_type", "net_fut_idx"]]),
            "latest": {"nifty": float(n.close.iloc[-1]) if len(n) else None, "nifty_chg_1d": float(n.close.pct_change().iloc[-1]) if len(n) > 1 else None,
                       "vix": float(v.close.iloc[-1]) if len(v) else None, "regime": last_r.get("regime_name"),
                       "p_stress": last_r.get("p_stress"), "asof": str(n.date.iloc[-1].date()) if len(n) else None}}


def latest_predictions():
    d = read_sql("""SELECT p.*, s.name, r.garch_vol_ann, r.es95_1d, r.cur_drawdown, r.beta
                    FROM predictions p LEFT JOIN dim_symbol s ON s.symbol=p.symbol
                    LEFT JOIN risk_metrics r ON r.symbol=p.symbol AND r.asof_date=p.asof_date
                    WHERE p.source='live' AND p.asof_date=(SELECT MAX(asof_date) FROM predictions WHERE source='live')
                    ORDER BY p.score DESC""")
    return _rec(d)


def prediction_history(symbol: str, days=250):
    d = read_sql("""SELECT asof_date, source, prob_up, exp_excess_ret, ret_q10, ret_q50, ret_q90, prob_risk, signal, score
                    FROM predictions WHERE symbol=:s ORDER BY asof_date DESC LIMIT :n""", {"s": symbol, "n": days * 2})
    return _rec(d.sort_values("asof_date"))


def prediction_detail(symbol: str):
    cur = [r for r in latest_predictions() if r["symbol"] == symbol]
    px = read_sql("SELECT date, close, adj_close FROM prices_daily WHERE symbol=:s ORDER BY date DESC LIMIT 300", {"s": symbol}, parse_dates=["date"]).sort_values("date")
    return {"current": cur[0] if cur else None, "history": prediction_history(symbol), "prices": _rec(px)}


def risk_latest():
    return _rec(read_sql("SELECT r.*, s.name FROM risk_metrics r LEFT JOIN dim_symbol s ON s.symbol=r.symbol "
                         "WHERE asof_date=(SELECT MAX(asof_date) FROM risk_metrics) ORDER BY risk_score DESC"))


def risk_detail(symbol: str):
    px = read_sql("SELECT date, adj_close FROM prices_daily WHERE symbol=:s ORDER BY date", {"s": symbol}, parse_dates=["date"]).set_index("date")["adj_close"]
    r = px.pct_change().dropna()
    dd = (px / px.cummax() - 1).tail(750)
    hist = read_sql("SELECT * FROM risk_metrics WHERE symbol=:s ORDER BY asof_date", {"s": symbol})
    cur = [x for x in risk_latest() if x["symbol"] == symbol]
    return {"current": cur[0] if cur else None, "history": _rec(hist),
            "drawdown": _rec(dd.rename("dd").reset_index()), "returns_tail": _rec(r.tail(500).rename("ret").reset_index())}


def news(symbol: str = None, days: int = 30, limit: int = 200):
    q = "SELECT published_at, entity, entity_type, title, source, url, label, sentiment FROM news WHERE published_at >= :c"
    p = {"c": pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)}
    if symbol:
        q += " AND entity=:s"; p["s"] = symbol
    d = read_sql(q + " ORDER BY published_at DESC LIMIT :l", {**p, "l": limit}, parse_dates=["published_at"])
    daily = d.assign(date=d.published_at.dt.date).groupby(["date", "entity_type"]).sentiment.agg(["mean", "size"]).reset_index() if len(d) else pd.DataFrame()
    by = d[d.entity_type == "stock"].groupby("entity").sentiment.agg(["mean", "size"]).reset_index().rename(columns={"entity": "symbol", "mean": "avg_sentiment", "size": "n"}) if len(d) else pd.DataFrame()
    return {"headlines": _rec(d), "daily": _rec(daily), "by_stock": _rec(by.sort_values("avg_sentiment")) if len(by) else []}


def backtest_latest():
    r = read_sql("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1")
    if r.empty:
        return {}
    run = r.iloc[0]
    eq = read_sql("SELECT run_id, date, equity, benchmark_equity, ret, turnover FROM backtest_equity WHERE run_id LIKE :r ORDER BY date", {"r": run.run_id + ":%"})
    j = lambda x: x if isinstance(x, (dict, list)) else json.loads(x) if x else {}
    return {"run_id": run.run_id, "created_at": str(run.created_at), "params": j(run.params), "metrics": j(run.metrics), "extras": j(run.extras),
            "equity": _rec(eq)}


def portfolio_latest():
    d = read_sql("SELECT * FROM portfolio_results WHERE run_id=(SELECT MAX(run_id) FROM portfolio_results) ORDER BY method, weight DESC")
    return _rec(d)


def optimize_custom(symbols_: list, method="max_sharpe", max_weight=0.2, capital=config.CAPITAL):
    from portfolio import optimizer, sizing
    px = read_sql("SELECT symbol, date, adj_close, close FROM prices_daily", parse_dates=["date"])
    px = px[px.symbol.isin(symbols_)]
    rets = px.pivot(index="date", columns="symbol", values="adj_close").sort_index().pct_change(fill_method=None).tail(252).fillna(0)
    w = optimizer.optimize(rets, method, max_w=max_weight)
    st = optimizer.portfolio_stats(w, rets)
    last = px.sort_values("date").groupby("symbol")["close"].last()
    g5 = read_sql("SELECT symbol, garch_vol_5d FROM risk_metrics WHERE asof_date=(SELECT MAX(asof_date) FROM risk_metrics)").set_index("symbol")["garch_vol_5d"]
    return {"method": method, "weights": {k: float(v) for k, v in w.items()}, "stats": st, "positions": _rec(sizing.position_table(w, last, capital, g5))}


def monitoring_summary():
    drift = read_sql("SELECT * FROM drift_metrics WHERE asof_date=(SELECT MAX(asof_date) FROM drift_metrics) ORDER BY psi DESC")
    perf = read_sql("SELECT * FROM live_performance WHERE asof_date=(SELECT MAX(asof_date) FROM live_performance) ORDER BY window_days")
    run = read_sql("SELECT run_id, trained_at, train_end_date, metrics FROM model_runs ORDER BY trained_at DESC LIMIT 5")
    fi = read_sql("SELECT feature, mean_abs_shap FROM feature_importance WHERE run_id=(SELECT run_id FROM feature_importance ORDER BY run_id DESC LIMIT 1) ORDER BY mean_abs_shap DESC LIMIT 20")
    q = read_sql("SELECT table_name, check_name, status, detail FROM data_quality_log ORDER BY id")
    ing = read_sql("SELECT run_ts, source, rows, status, message FROM ingestion_log ORDER BY id DESC LIMIT 30")
    return {"drift": _rec(drift), "performance": _rec(perf), "models": _rec(run), "feature_importance": _rec(fi),
            "data_quality": _rec(q), "ingestion": _rec(ing)}
