"""Streamlit dashboard.  Run:  streamlit run app/streamlit_app.py
Uses the FastAPI backend if 'Use FastAPI' is ticked (needs uvicorn running), else reads the database directly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import config
from app import services

st.set_page_config(page_title="Quant AI Platform", page_icon="📈", layout="wide")


def call(path, fn, *args, params=None, method="get", body=None):
    if st.session_state.get("use_api"):
        try:
            r = requests.request(method, config.API_URL + path, params=params, json=body, timeout=30)
            r.raise_for_status(); return r.json()
        except Exception as e:
            st.sidebar.warning(f"API unreachable ({e}); using DB directly")
    return fn(*args)


def df(x): return pd.DataFrame(x)
def pct(x): return "–" if x is None or pd.isna(x) else f"{x:.1%}"


st.sidebar.title("📈 Quant AI Platform")
st.sidebar.checkbox("Use FastAPI backend", key="use_api", value=False, help="Start it with: uvicorn api.main:app --port 8000")
page = st.sidebar.radio("Page", ["Market overview", "AI predictions", "News intelligence", "Portfolio & risk", "Backtest & research", "Model monitoring"])
h = services.health()
st.sidebar.caption(f"DB: {h.get('db')} · {h.get('price_rows')} price rows" if h["status"] == "ok" else f"DB error: {h.get('detail')}")
if h["status"] != "ok":
    st.error("Database not reachable. Check DATABASE_URL in .env and run: python run_pipeline.py init-db"); st.stop()

# ---------------------------------------------------------------- Market overview
if page == "Market overview":
    st.title("Market overview")
    m = call("/market/overview", services.market_overview)
    L = m["latest"]
    c = st.columns(4)
    c[0].metric("NIFTY 50", f"{L['nifty']:,.0f}" if L["nifty"] else "–", pct(L["nifty_chg_1d"]))
    c[1].metric("India VIX", f"{L['vix']:.2f}" if L["vix"] else "–")
    c[2].metric("Regime", (L["regime"] or "n/a").upper())
    c[3].metric("P(stress)", pct(L["p_stress"]))
    n, v, rg = df(m["nifty"]), df(m["vix"]), df(m["regime"])
    fig = go.Figure(); n["date"] = pd.to_datetime(n["date"])
    fig.add_trace(go.Scatter(x=n.date, y=n.close, name="NIFTY"))
    if len(rg):
        rg["date"] = pd.to_datetime(rg["date"]); s = rg[rg.p_stress > 0.5]
        fig.add_trace(go.Scatter(x=s.date, y=n.set_index("date").close.reindex(s.date), mode="markers", name="stress regime", marker=dict(color="red", size=4)))
    fig.update_layout(height=380, title="NIFTY 50 with stress-regime days"); st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    v["date"] = pd.to_datetime(v["date"]); c1.plotly_chart(px.line(v, x="date", y="close", title="India VIX"), width="stretch")
    p = df(m["participants"])
    if len(p):
        p["date"] = pd.to_datetime(p["date"])
        c2.plotly_chart(px.line(p, x="date", y="net_fut_idx", color="client_type", title="Net index-futures OI by participant (FII/DII/Pro/Client)"), width="stretch")
    if len(rg):
        st.plotly_chart(px.area(rg, x="date", y=["p_calm", "p_neutral", "p_stress"], title="Regime probabilities (filtered HMM)"), width="stretch")

# ---------------------------------------------------------------- AI predictions
elif page == "AI predictions":
    st.title("AI predictions (5-day outlook vs NIFTY)")
    d = df(call("/predict", services.latest_predictions))
    if d.empty:
        st.info("No predictions yet. Run: python run_pipeline.py train && python run_pipeline.py predict"); st.stop()
    st.caption(f"As of {str(d.asof_date.iloc[0])[:10]} · prob_up = calibrated probability of beating NIFTY over 5 days")
    c = st.columns(4)
    for i, s in enumerate(["BUY", "HOLD", "SELL"]): c[i].metric(s, int((d.signal == s).sum()))
    c[3].metric("Avg risk score", f"{d.risk_score.mean():.0f}")
    sig = st.multiselect("Signal filter", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    show = d[d.signal.isin(sig)][["symbol", "name", "signal", "prob_up", "confidence", "exp_excess_ret", "ret_q10", "ret_q90", "prob_risk", "risk_score", "risk_level", "reason"]]
    st.dataframe(show.style.format({"prob_up": "{:.1%}", "confidence": "{:.1%}", "exp_excess_ret": "{:+.2%}", "ret_q10": "{:+.2%}", "ret_q90": "{:+.2%}",
                                    "prob_risk": "{:.1%}", "risk_score": "{:.0f}"}), width="stretch", height=380)
    sym = st.selectbox("Explain a stock", d.symbol.tolist())
    det = call(f"/predict/{sym}", services.prediction_detail, sym)
    cur = det["current"]
    cc = st.columns(5)
    cc[0].metric("Signal", cur["signal"]); cc[1].metric("P(outperform)", pct(cur["prob_up"])); cc[2].metric("P(drop ≤ %.0f%% in %dd)" % (config.RISK_TH * 100, config.RISK_H), pct(cur["prob_risk"]))
    cc[3].metric("Exp. 5d range (q10..q90)", f"{cur['ret_q10']:+.1%} .. {cur['ret_q90']:+.1%}"); cc[4].metric("Risk", f"{cur['risk_score']:.0f} ({cur['risk_level']})")
    tf = cur.get("top_factors") or []
    if isinstance(tf, str): tf = json.loads(tf)
    if tf:
        t = df(tf).sort_values("shap")
        fig = go.Figure(go.Bar(x=t.shap, y=t.feature, orientation="h", marker_color=["seagreen" if x > 0 else "indianred" for x in t.shap],
                               text=[f"value={v}" for v in t.value], textposition="auto"))
        fig.update_layout(title=f"SHAP – why the model says {cur['signal']} for {sym}", height=330); st.plotly_chart(fig, width="stretch")
    hist = df(det["history"])
    if len(hist):
        hist["asof_date"] = pd.to_datetime(hist["asof_date"])
        st.plotly_chart(px.line(hist, x="asof_date", y="prob_up", color="source", title="Prediction history (walk_forward = out-of-sample replay, live = daily)"), width="stretch")

# ---------------------------------------------------------------- News
elif page == "News intelligence":
    st.title("News intelligence")
    sym = st.selectbox("Stock (or all)", ["ALL"] + [s["symbol"] for s in services.symbols()])
    days = st.slider("Days", 3, 90, 30)
    nw = call("/news", services.news, None if sym == "ALL" else sym, days, params={"symbol": None if sym == "ALL" else sym, "days": days})
    hd, dly, bys = df(nw["headlines"]), df(nw["daily"]), df(nw["by_stock"])
    if hd.empty:
        st.info("No news in DB. Run: python run_pipeline.py fetch-news"); st.stop()
    c = st.columns(3); c[0].metric("Headlines", len(hd)); c[1].metric("Avg sentiment", f"{hd.sentiment.mean():+.2f}"); c[2].metric("% negative", pct((hd.label == "negative").mean()))
    if len(dly):
        dly["date"] = pd.to_datetime(dly["date"]); st.plotly_chart(px.bar(dly, x="date", y="mean", color="entity_type", barmode="group", title="Average daily sentiment (FinBERT: P(pos) − P(neg))"), width="stretch")
    if len(bys):
        st.plotly_chart(px.bar(bys, x="symbol", y="avg_sentiment", color="avg_sentiment", color_continuous_scale="RdYlGn", title="Sentiment by stock"), width="stretch")
    st.dataframe(hd[["published_at", "entity", "title", "source", "label", "sentiment"]], width="stretch", height=380)

# ---------------------------------------------------------------- Portfolio & risk
elif page == "Portfolio & risk":
    st.title("Portfolio & risk")
    r = df(call("/risk", services.risk_latest))
    if r.empty:
        st.info("Run: python run_pipeline.py risk"); st.stop()
    st.subheader("Risk table (GJR-GARCH vol, VaR/ES, drawdown, model downside probability)")
    st.dataframe(r[["symbol", "risk_score", "risk_level", "garch_vol_ann", "es95_1d", "var95_1d", "cur_drawdown", "max_drawdown", "beta"]].style.format(
        {"risk_score": "{:.0f}", "garch_vol_ann": "{:.1%}", "es95_1d": "{:.2%}", "var95_1d": "{:.2%}", "cur_drawdown": "{:.1%}", "max_drawdown": "{:.1%}", "beta": "{:.2f}"}),
        width="stretch", height=330)
    st.plotly_chart(px.scatter(r, x="garch_vol_ann", y="es95_1d", size="risk_score", color="risk_level", hover_name="symbol", title="Volatility vs Expected Shortfall"), width="stretch")
    st.subheader("Optimise a portfolio")
    syms = st.multiselect("Stocks", r.symbol.tolist(), default=r.sort_values("risk_score").symbol.head(10).tolist())
    c = st.columns(3); method = c[0].selectbox("Method", ["max_sharpe", "min_variance", "risk_parity", "equal"])
    mw = c[1].slider("Max weight", 0.05, 0.5, 0.2, 0.01); cap = c[2].number_input("Capital (₹)", 100000, 100000000, int(config.CAPITAL), 100000)
    if st.button("Optimise") and len(syms) >= 2:
        o = call("/portfolio/optimize", services.optimize_custom, syms, method, mw, cap, method="post", body={"symbols": syms, "method": method, "max_weight": mw, "capital": cap})
        s = o["stats"]; cc = st.columns(4)
        cc[0].metric("Expected return", pct(s["exp_return"])); cc[1].metric("Volatility", pct(s["vol"])); cc[2].metric("Sharpe", f"{s['sharpe']:.2f}"); cc[3].metric("Effective N", f"{s['effective_n']:.1f}")
        pt = df(o["positions"]); st.dataframe(pt, width="stretch")
        st.plotly_chart(px.pie(pt, names="symbol", values="weight", title="Weights"), width="stretch")
    pl = df(services.portfolio_latest())
    if len(pl):
        st.subheader("Latest model-driven portfolios (from BUY signals)")
        st.plotly_chart(px.bar(pl, x="symbol", y="weight", color="method", barmode="group"), width="stretch")

# ---------------------------------------------------------------- Backtest
elif page == "Backtest & research":
    st.title("Walk-forward backtest & research")
    b = call("/backtest/latest", services.backtest_latest)
    if not b:
        st.info("Run: python run_pipeline.py backtest"); st.stop()
    st.caption(f"Run {b['run_id']} · out-of-sample only · costs {b['params']['cost_bps']}bps + slippage {b['params']['slippage_bps']}bps + sqrt impact · exec lag {b['params']['exec_lag']}d")
    m = b["metrics"]; c = st.columns(6)
    for col, (k, lab, f) in zip(c, [("cagr", "CAGR", pct), ("sharpe", "Sharpe", lambda x: f"{x:.2f}"), ("sortino", "Sortino", lambda x: f"{x:.2f}"),
                                      ("max_drawdown", "Max DD", pct), ("annual_turnover", "Turnover/yr", lambda x: f"{x:.1f}x"), ("bench_cagr", "NIFTY CAGR", pct)]):
        col.metric(lab, f(m.get(k)) if m.get(k) is not None else "–")
    eq = df(b["equity"]); eq["date"] = pd.to_datetime(eq["date"]); eq["method"] = eq.run_id.str.split(":").str[1]
    fig = px.line(eq, x="date", y="equity", color="method", title="Equity curves by sizing method (net of costs)")
    bm = eq[eq.method == eq.method.iloc[0]]; fig.add_trace(go.Scatter(x=bm.date, y=bm.benchmark_equity, name="NIFTY", line=dict(color="black", dash="dash")))
    st.plotly_chart(fig, width="stretch")
    ex = b["extras"]
    st.subheader("Strategy comparison"); st.dataframe(df(ex["comparison"]).style.format(precision=3), width="stretch")
    mc = ex["monte_carlo"]
    if mc:
        st.subheader("Monte Carlo (1-year block bootstrap)")
        fan = mc["fan"]; x = list(range(len(fan["p50"]))); f2 = go.Figure()
        for lo, hi, col in (("p5", "p95", "rgba(70,130,180,.2)"), ("p25", "p75", "rgba(70,130,180,.4)")):
            f2.add_trace(go.Scatter(x=x + x[::-1], y=fan[hi] + fan[lo][::-1], fill="toself", fillcolor=col, line=dict(width=0), name=f"{lo}-{hi}"))
        f2.add_trace(go.Scatter(x=x, y=fan["p50"], name="median")); f2.update_layout(xaxis_title="5-day steps", height=350); st.plotly_chart(f2, width="stretch")
        cc = st.columns(4); cc[0].metric("Median 1y", pct(mc["terminal_return_pct"]["p50"])); cc[1].metric("5th pct 1y", pct(mc["terminal_return_pct"]["p5"]))
        cc[2].metric("P(loss)", pct(mc["prob_loss"])); cc[3].metric("P(DD>20%)", pct(mc["prob_dd_gt_20"]))
    st.subheader("Stress tests"); st.dataframe(df(ex["stress"]).dropna(axis=1, how="all"), width="stretch")

# ---------------------------------------------------------------- Monitoring
else:
    st.title("Model monitoring & data health")
    mo = call("/monitoring", services.monitoring_summary)
    perf, drift = df(mo["performance"]), df(mo["drift"])
    if len(perf):
        st.subheader("Realised performance of stored predictions"); st.dataframe(perf.style.format({"ic": "{:+.3f}", "auc": "{:.3f}", "hit_rate": "{:.1%}", "brier": "{:.4f}"}), width="stretch")
    if len(drift):
        st.subheader("Feature drift (PSI: <0.10 stable, 0.10-0.25 moderate, >0.25 major)")
        fig = px.bar(drift, x="feature", y="psi", color="status", color_discrete_map={"ok": "seagreen", "warn": "orange", "alert": "crimson"}); st.plotly_chart(fig, width="stretch")
    fi = df(mo["feature_importance"])
    if len(fi): st.plotly_chart(px.bar(fi.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title="Global SHAP importance"), width="stretch")
    st.subheader("Model runs"); st.dataframe(df(mo["models"]).assign(metrics=lambda x: x.metrics.astype(str).str[:160]), width="stretch")
    st.subheader("Data quality"); st.dataframe(df(mo["data_quality"]), width="stretch")
    st.subheader("Ingestion log"); st.dataframe(df(mo["ingestion"]), width="stretch")
