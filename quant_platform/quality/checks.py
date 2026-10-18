"""Data-quality checks: missing data, duplicates, OHLC sanity, stale prices, corporate actions,
timestamp/calendar alignment across sources. Results -> data_quality_log table + reports/data_quality.json"""
import json
import numpy as np
import pandas as pd
import config
from db.connection import read_sql, get_engine
from sqlalchemy import text


def _add(res, table, check, status, detail):
    res.append({"table": table, "check": check, "status": status, "detail": detail})


def run_quality_checks(engine=None, save=True) -> list:
    engine = engine or get_engine()
    res = []
    px = read_sql("SELECT * FROM prices_daily", engine=engine, parse_dates=["date"])
    ix = read_sql("SELECT * FROM index_prices", engine=engine, parse_dates=["date"])
    oi = read_sql("SELECT * FROM participant_oi", engine=engine, parse_dates=["date"])
    ca = read_sql("SELECT * FROM corporate_actions", engine=engine, parse_dates=["date"])
    nw = read_sql("SELECT published_at, entity_type, sentiment FROM news", engine=engine, parse_dates=["published_at"])
    nifty = ix[ix.index_name == config.NIFTY_NAME].set_index("date").sort_index()
    cal = nifty.index
    # ---------- prices ----------
    if px.empty:
        _add(res, "prices_daily", "non_empty", "FAIL", "no rows")
    else:
        _add(res, "prices_daily", "coverage", "OK", f"{px.symbol.nunique()} symbols, {px.date.min().date()}..{px.date.max().date()}, {len(px)} rows")
        dup = int(px.duplicated(["symbol", "date"]).sum())
        _add(res, "prices_daily", "duplicates", "OK" if dup == 0 else "FAIL", f"{dup} duplicate keys")
        bad = px[(px[["open", "high", "low", "close"]] <= 0).any(axis=1)]
        _add(res, "prices_daily", "non_positive_prices", "OK" if bad.empty else "FAIL", f"{len(bad)} rows")
        ohlc = px[(px.high < px.low - 1e-6) | (px.close > px.high * 1.001) | (px.close < px.low * 0.999)]
        _add(res, "prices_daily", "ohlc_consistency", "OK" if len(ohlc) < 0.001 * len(px) else "WARN", f"{len(ohlc)} inconsistent rows")
        zv = (px.volume == 0).mean()
        _add(res, "prices_daily", "zero_volume_share", "OK" if zv < 0.02 else "WARN", f"{zv:.2%}")
        # missing days vs NIFTY calendar (per symbol, after first listing)
        miss = {}
        for s, g in px.groupby("symbol"):
            exp = cal[(cal >= g.date.min()) & (cal <= g.date.max())]
            m = len(exp.difference(pd.DatetimeIndex(g.date)))
            if m / max(len(exp), 1) > 0.01:
                miss[s] = m
        _add(res, "prices_daily", "missing_days_vs_nifty_calendar", "OK" if not miss else "WARN",
             f"{len(miss)} symbols with >1% missing days: {dict(list(miss.items())[:8])}" if miss else "all symbols aligned")
        short = px.groupby("symbol").size()
        short = short[short < 500]
        _add(res, "prices_daily", "short_history", "OK" if short.empty else "WARN", f"{dict(short)}" if len(short) else "none")
        # stale prices
        stale = {}
        for s, g in px.sort_values("date").groupby("symbol"):
            run = (g.close.diff() == 0).astype(int)
            mx = run.groupby((run == 0).cumsum()).sum().max()
            if mx >= 5:
                stale[s] = int(mx)
        _add(res, "prices_daily", "stale_prices(>=5 flat days)", "OK" if not stale else "WARN", str(stale) if stale else "none")
        # extreme moves vs corporate actions (split-adjusted close should have no unexplained jumps)
        p = px.sort_values(["symbol", "date"]).copy()
        p["ret"] = p.groupby("symbol")["close"].pct_change()
        ext = p[p.ret.abs() > 0.25]
        if len(ext):
            spl = ca[ca.action_type == "split"][["symbol", "date"]].assign(has_split=True)
            ext = ext.merge(spl, on=["symbol", "date"], how="left")
            unexplained = ext[ext.has_split.isna()]
        else:
            unexplained = ext
        _add(res, "prices_daily", "extreme_moves_>25%", "OK" if unexplained.empty else "WARN",
             f"{len(unexplained)} unexplained (check splits/bonus/demergers): " +
             ", ".join(f"{r.symbol}@{r.date.date()}" for r in unexplained.head(6).itertuples()) if len(unexplained) else "none")
        # adj_close vs close divergence = dividend/bonus adjustment sanity
        ratio = (px.adj_close / px.close).groupby(px.symbol).agg(["min", "max"])
        weird = ratio[(ratio["min"] < 0.3) | (ratio["max"] > 1.0001)]
        _add(res, "prices_daily", "adj_close_ratio", "OK" if weird.empty else "WARN", f"{len(weird)} symbols with odd adj/close ratio")
        _add(res, "corporate_actions", "events", "OK", f"{(ca.action_type == 'dividend').sum()} dividends, {(ca.action_type == 'split').sum()} splits recorded")
    # ---------- index ----------
    for nm in (config.NIFTY_NAME, config.VIX_NAME):
        g = ix[ix.index_name == nm]
        if g.empty:
            _add(res, "index_prices", nm, "FAIL", "no rows"); continue
        nn = int(g.close.isna().sum())
        _add(res, "index_prices", f"{nm}_nulls", "OK" if nn == 0 else "WARN", f"{nn} null closes, {g.date.min().date()}..{g.date.max().date()}")
    v = ix[ix.index_name == config.VIX_NAME]
    if len(v):
        _add(res, "index_prices", "vix_range", "OK" if v.close.between(5, 100).all() else "WARN", f"min={v.close.min():.1f} max={v.close.max():.1f}")
    # ---------- FII/DII ----------
    if oi.empty:
        _add(res, "participant_oi", "non_empty", "WARN", "no FII/DII rows")
    else:
        d = oi[oi.client_type == "TOTAL"]
        recon = oi["fut_idx_long_reconstructed"].mean()
        _add(res, "participant_oi", "reconstructed_fut_idx_long", "INFO", f"{recon:.1%} of rows had empty 'Future Index Long' and were rebuilt as TotalLong - other longs")
        err = (d.fut_idx_long - d.fut_idx_short).abs()
        _add(res, "participant_oi", "total_long_equals_short", "OK" if err.max() <= 5 else "WARN", f"max |long-short| on TOTAL = {err.max():.0f} contracts (NSE rounding)")
        days = pd.DatetimeIndex(oi.date.unique())
        missing = cal[(cal >= days.min()) & (cal <= days.max())].difference(days)
        _add(res, "participant_oi", "missing_trading_days", "OK" if len(missing) <= 3 else "WARN", f"{len(missing)} NIFTY trading days without FII/DII file")
        extra = days.difference(cal)
        _add(res, "participant_oi", "dates_not_in_nifty_calendar", "OK" if len(extra) <= 3 else "WARN", f"{len(extra)} dates")
        _add(res, "participant_oi", "duplicates", "OK" if not oi.duplicated(["date", "client_type"]).any() else "FAIL", "")
    # ---------- news ----------
    if nw.empty:
        _add(res, "news", "coverage", "WARN", "no news rows (run fetch-news)")
    else:
        days = nw.published_at.dt.normalize().nunique()
        recent = nw[nw.published_at >= pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=90)]
        _add(res, "news", "coverage", "OK", f"{len(nw)} headlines over {days} days; {len(recent)} in last 90d")
        _add(res, "news", "null_sentiment", "OK" if nw.sentiment.isna().mean() < 0.02 else "WARN", f"{nw.sentiment.isna().mean():.1%}")
        _add(res, "news", "model_eligibility", "INFO" if days < config.NEWS_MIN_DAYS_FOR_MODEL else "OK",
             f"{days}/{config.NEWS_MIN_DAYS_FOR_MODEL} days needed before sentiment enters the ML features (until then: overlay only)")
    # ---------- alignment ----------
    if not px.empty and len(cal):
        cov = len(pd.DatetimeIndex(px.date.unique()).intersection(cal)) / len(cal)
        _add(res, "alignment", "stock_vs_nifty_calendar", "OK" if cov > 0.98 else "WARN", f"{cov:.1%} of NIFTY days have stock prices")
    _add(res, "alignment", "publication_lags", "INFO",
         f"FII/DII lagged {config.FII_LAG_DAYS} trading day; news cut-off {config.NEWS_CUTOFF_HOUR_IST:.1f}h IST; targets only use shift(-k)")
    if save:
        with engine.begin() as c:
            c.execute(text("DELETE FROM data_quality_log"))
            for r in res:
                c.execute(text("INSERT INTO data_quality_log(table_name,check_name,status,detail) VALUES (:t,:c,:s,:d)"),
                          {"t": r["table"], "c": r["check"], "s": r["status"], "d": str(r["detail"])[:1000]})
        (config.REPORT_DIR / "data_quality.json").write_text(json.dumps(res, indent=2))
    return res


def print_report(res):
    icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌", "INFO": "ℹ️ "}
    for r in res:
        print(f"{icon.get(r['status'],'')} [{r['table']}] {r['check']}: {r['detail']}")
