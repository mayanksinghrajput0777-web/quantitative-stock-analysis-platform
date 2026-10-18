"""Turns model outputs + risk + regime + news into BUY / HOLD / SELL signals (your old NO_TRADE == HOLD)."""
import json
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine, upsert_df
from features import sentiment


def latest_news(engine=None, days=3) -> pd.DataFrame:
    engine = engine or get_engine()
    nw = read_sql("SELECT published_at, entity, entity_type, sentiment FROM news", engine=engine, parse_dates=["published_at"])
    if nw.empty:
        return pd.DataFrame(columns=["symbol", "news_sent_3d", "news_n"])
    cut = nw.published_at.max() - pd.Timedelta(days=days)
    r = nw[(nw.entity_type == "stock") & (nw.published_at >= cut)].groupby("entity").agg(
        news_sent_3d=("sentiment", "mean"), news_n=("sentiment", "size")).reset_index().rename(columns={"entity": "symbol"})
    return r


def build_signals(pred: pd.DataFrame, risk: pd.DataFrame, regime_stress_p: float, news: pd.DataFrame) -> pd.DataFrame:
    d = pred.copy()
    d["score"] = 0.5 * d["prob_up"].rank(pct=True) + 0.5 * d["exp_excess_ret"].rank(pct=True)
    d["score_rank"] = d["score"].rank(pct=True)
    d = d.merge(risk[["symbol", "risk_score", "risk_level"]], on="symbol", how="left", suffixes=("", "_r"))
    d = d.merge(news, on="symbol", how="left")
    stress = regime_stress_p is not None and regime_stress_p > 0.6
    buy_th = 0.90 if stress else config.BUY_TOP_PCT
    neg_news = (d["news_sent_3d"] < config.NEWS_NEG_BLOCK) & (d["news_n"].fillna(0) >= 2)
    buy = ((d.score_rank >= buy_th) & (d.prob_up >= config.MIN_PROB_UP_BUY) & (d.prob_risk < config.RISK_BLOCK_PROB)
           & (d.risk_level != "Extreme") & ~neg_news & (d.exp_excess_ret > 0))
    sell = ((d.score_rank <= config.SELL_BOTTOM_PCT) | (d.prob_risk >= 1.5 * config.RISK_BLOCK_PROB) | (d.risk_level == "Extreme"))
    d["signal"] = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))
    up = (d.prob_up > 0.5) & (d.exp_excess_ret > 0)
    dn = (d.prob_up < 0.5) & (d.exp_excess_ret < 0)
    d["direction"] = np.where(up, "UP", np.where(dn, "DOWN", "FLAT"))          # vs NIFTY over the next 5 days
    d["confidence"] = np.where(d.direction == "DOWN", 1 - d.prob_up, np.where(d.direction == "UP", d.prob_up, 0.5))
    d["news_overlay"] = np.where(neg_news, "negative news blocks BUY", "")

    def why(r):
        if r.signal == "BUY":
            return f"top-{round((1 - buy_th) * 100)}% alpha score; downside prob {r.prob_risk:.0%}; risk {r.risk_level}"
        parts = []
        if r.signal == "SELL":
            if r.score_rank <= config.SELL_BOTTOM_PCT: parts.append("bottom-20% alpha score")
            if r.prob_risk >= 1.5 * config.RISK_BLOCK_PROB: parts.append(f"high downside probability {r.prob_risk:.0%}")
            if r.risk_level == "Extreme": parts.append("Extreme risk level")
        else:
            if r.news_overlay: parts.append(r.news_overlay)
            elif r.score_rank >= buy_th and r.prob_risk >= config.RISK_BLOCK_PROB: parts.append("good alpha score but downside risk too high")
            elif r.score_rank >= buy_th: parts.append("alpha edge too small (prob_up < %.0f%%)" % (100 * config.MIN_PROB_UP_BUY))
            else: parts.append("no strong edge (middle of the ranking)")
            if stress: parts.append("market in STRESS regime (stricter BUY rule)")
        return "; ".join(parts)
    d["reason"] = d.apply(why, axis=1)
    return d


def _as_list(t):
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except Exception:
            return []
    return t if isinstance(t, list) else []


def export_legacy_table(d: pd.DataFrame):
    """Same columns as your original final_predictions_next5.csv (+ risk and a plain-English reason)"""
    out = pd.DataFrame({
        "Stock": d.symbol, "Prediction": d.direction, "Confidence": d.confidence.round(3),
        "Expected Up": d.ret_q90.map(lambda x: f"{100*x:+.2f}%"), "Expected Down": d.ret_q10.map(lambda x: f"{100*x:+.2f}%"),
        "SHAP Top Factors": d.top_factors.map(lambda t: ", ".join(f"{f['feature']}({'+' if f['direction'] == 'up' else '-'})" for f in _as_list(t))),
        "Suggestion": d.signal, "Reason": d.reason, "Risk Level": d.risk_level, "Risk Score": d.risk_score.round(1)})
    p = config.REPORT_DIR / "final_predictions_next5.csv"
    out.sort_values("Stock").to_csv(p, index=False)
    return p, out
