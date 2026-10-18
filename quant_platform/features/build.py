"""Builds the point-in-time feature panel (long format: one row per symbol x date)."""
import logging
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine, upsert_df
from features import technical, macro, sentiment, regime as regime_mod

log = logging.getLogger("features")
META = ["date", "symbol"]


def load_market_data(max_date=None, engine=None) -> dict:
    engine = engine or get_engine()
    md = pd.Timestamp(max_date).date() if max_date is not None else None
    w = "WHERE date <= :md" if md else ""
    p = {"md": md} if md else {}
    px = read_sql(f"SELECT * FROM prices_daily {w}", p, engine, parse_dates=["date"])
    ix = read_sql(f"SELECT * FROM index_prices {w}", p, engine, parse_dates=["date"])
    oi = read_sql(f"SELECT * FROM participant_oi {w}", p, engine, parse_dates=["date"])
    nw_w = "WHERE published_at < :nd" if md else ""
    nw = read_sql(f"SELECT published_at, entity, entity_type, sentiment FROM news {nw_w}",
                  {"nd": md + pd.Timedelta(days=1)} if md else {}, engine, parse_dates=["published_at"])
    return {"px": px, "ix": ix, "oi": oi, "news": nw}


def _wide(px, col, calendar, ffill=True):
    w = px.pivot(index="date", columns="symbol", values=col).reindex(calendar)
    return w.ffill(limit=3) if ffill else w


def build_panel(max_date=None, engine=None, include_news=None, save=False, min_obs=150):
    D = load_market_data(max_date, engine)
    ix = D["ix"]
    nifty = ix[ix.index_name == config.NIFTY_NAME].set_index("date")["close"].sort_index()
    vix = ix[ix.index_name == config.VIX_NAME].set_index("date")["close"].sort_index()
    if nifty.empty or D["px"].empty:
        raise RuntimeError("No price/NIFTY data in DB. Run: python run_pipeline.py bootstrap")
    cal = nifty.index
    vix = vix.reindex(cal).ffill(limit=3)
    px = D["px"]
    counts = px.groupby("symbol")["adj_close"].count()
    keep = counts[counts >= min_obs].index
    px = px[px.symbol.isin(keep)]
    adj, close = _wide(px, "adj_close", cal), _wide(px, "close", cal)
    opn, high, low = _wide(px, "open", cal, False), _wide(px, "high", cal, False), _wide(px, "low", cal, False)
    vol = _wide(px, "volume", cal, False)
    syms = list(adj.columns)
    feats = technical.compute_stock_features(adj, close, opn, high, low, vol, nifty)
    tg = technical.compute_targets(adj, nifty, config.ALPHA_H, config.RISK_H, config.RISK_TH)
    nd = len(cal)
    data = {k: v.reindex(index=cal, columns=syms).to_numpy(dtype="float32").ravel() for k, v in {**feats, **tg}.items()}
    panel = pd.DataFrame(data)
    panel.insert(0, "symbol", np.tile(np.array(syms, dtype=object), nd))
    panel.insert(0, "date", np.repeat(cal.values, len(syms)))
    panel["adj_close"] = adj.to_numpy(dtype="float64").ravel()
    # ---- market-wide features ----
    mk = macro.nifty_vix_features(nifty, vix)
    mk = mk.join(macro.fii_features(D["oi"], cal))
    rg = regime_mod.compute_regime(nifty, vix)
    mk["regime"], mk["p_stress"], mk["p_calm"] = rg["regime"], rg["p_stress"], rg["p_calm"]
    # ---- news ----
    ncov = sentiment.coverage_days(D["news"], cal)
    use_news = (ncov >= config.NEWS_MIN_DAYS_FOR_MODEL) if include_news is None else include_news
    if use_news:
        mk_s, st = sentiment.sentiment_features(D["news"], cal)
        mk = mk.join(mk_s)
        for k, wdf in st.items():
            panel[k] = wdf.reindex(index=cal, columns=syms).to_numpy(dtype="float32").ravel()
    panel = panel.merge(mk.astype("float32"), left_on="date", right_index=True, how="left")
    panel["beta_x_vixz"] = panel["beta_60d"] * panel["vix_z_60"]
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["adj_close"].notna() & panel["vol_60d"].notna()].sort_values(["date", "symbol"]).reset_index(drop=True)
    info = {"n_rows": len(panel), "n_symbols": len(syms), "start": str(panel.date.min().date()),
            "end": str(panel.date.max().date()), "news_days": ncov, "news_in_model": bool(use_news),
            "n_features": len(feature_columns(panel))}
    if save:
        try:
            panel.to_pickle(config.ART_DIR / "panel.pkl")
        except Exception as e:
            log.warning("panel cache save failed: %s", e)
    return panel, info, rg


def feature_columns(panel: pd.DataFrame):
    drop = set(META) | {"adj_close"}
    return [c for c in panel.columns if c not in drop and not c.startswith("fwd_") and not c.startswith("y_")]


def save_regime(rg: pd.DataFrame):
    r = rg.dropna(subset=["regime"]).copy()
    r.index.name = "date"
    r = r.reset_index()
    r["regime"] = r["regime"].astype(int)
    upsert_df(r[["date", "regime", "regime_name", "p_calm", "p_neutral", "p_stress"]], "market_regime")
