"""Computes the risk table for every stock as of the latest date and stores it in risk_metrics."""
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, upsert_df, get_engine
from risk import var_es, garch, drawdown, risk_score


def run_risk(engine=None, asof=None, verbose=True) -> pd.DataFrame:
    engine = engine or get_engine()
    px = read_sql("SELECT symbol, date, adj_close FROM prices_daily", engine=engine, parse_dates=["date"])
    ix = read_sql("SELECT index_name, date, close FROM index_prices", engine=engine, parse_dates=["date"])
    nifty = ix[ix.index_name == config.NIFTY_NAME].set_index("date")["close"].sort_index()
    vix = ix[ix.index_name == config.VIX_NAME].set_index("date")["close"].sort_index()
    wide = px.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    if asof is not None:
        wide = wide[wide.index <= pd.Timestamp(asof)]
        nifty = nifty[nifty.index <= pd.Timestamp(asof)]
    asof_d = wide.index.max()
    vix_last = float(vix[vix.index <= asof_d].iloc[-1]) if len(vix) else None
    pr = {}
    try:
        p = read_sql("SELECT symbol, prob_risk FROM predictions WHERE source='live' AND asof_date = (SELECT MAX(asof_date) FROM predictions WHERE source='live')", engine=engine)
        pr = dict(zip(p.symbol, p.prob_risk))
    except Exception:
        pass
    nr = nifty.pct_change().dropna()
    rows = []
    for s in wide.columns:
        p_ = wide[s].dropna()
        if len(p_) < 120 or p_.index[-1] < asof_d - pd.Timedelta(days=10):
            continue
        r = p_.pct_change().dropna().clip(-0.3, 0.3)
        v95, e95 = var_es.hist_var_es(r.tail(500), 0.05)
        v99, e99 = var_es.hist_var_es(r.tail(500), 0.01)
        g = garch.garch_forecast(r)
        dd = drawdown.drawdown_stats(p_)
        j = pd.concat([r.tail(252), nr], axis=1, join="inner").dropna()
        beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()) if len(j) > 60 else np.nan
        m = {"symbol": s, "asof_date": asof_d, "var95_1d": v95, "es95_1d": e95, "var99_1d": v99, "es99_1d": e99,
             "garch_vol_1d": g["vol_1d"], "garch_vol_5d": g["vol_5d"], "garch_vol_ann": g["vol_ann"],
             "realized_vol_20d_ann": float(r.tail(20).std() * np.sqrt(252)),
             "downside_dev_ann": var_es.downside_deviation(r.tail(252)),
             "max_drawdown": dd["max_dd"], "cur_drawdown": dd["cur_dd"], "dd_duration_days": dd["dd_days"], "beta": beta}
        sc, lv, _ = risk_score.risk_score({**m, "cur_drawdown": dd["cur_dd"], "prob_risk": pr.get(s, np.nan)}, vix_last)
        m["risk_score"], m["risk_level"] = sc, lv
        rows.append(m)
    df = pd.DataFrame(rows)
    upsert_df(df, "risk_metrics", engine=engine)
    if verbose:
        print(f"✅ risk metrics for {len(df)} stocks as of {asof_d.date()} (VIX {vix_last})")
        print(df.groupby("risk_level").size().to_dict())
    return df
