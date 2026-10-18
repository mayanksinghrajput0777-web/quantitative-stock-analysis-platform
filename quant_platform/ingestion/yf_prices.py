"""yfinance downloaders - FAST batch pull (all tickers in one multi-threaded call).

 * First time for a symbol  -> full HISTORY_YEARS pull (so adj_close is one consistent series from one source)
 * Every later run          -> only the last ~7 days (top-up), upserted, never duplicated
 * New dividend / split found in the top-up window -> that symbol is automatically re-pulled in full,
   because Yahoo re-scales ALL historical adj_close values after a corporate action.
"""
import time, logging
import numpy as np
import pandas as pd
import config
from db.connection import upsert_df, read_sql, log_ingestion, get_states, set_state
from ingestion.universe import get_universe

log = logging.getLogger("yf")
OVERLAP_DAYS = 7


def _start_target(years=None):
    y = years or config.HISTORY_YEARS
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=y)).normalize()


def _clean_frame(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d.index.name = "date"
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.reset_index()


def _split_download(raw, tickers) -> dict:
    """Parse yf.download() output for any yfinance version (ticker level can be first or second)."""
    out = {}
    if raw is None or len(raw) == 0:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        l0, l1 = set(raw.columns.get_level_values(0)), set(raw.columns.get_level_values(1))
        lvl = 0 if set(tickers) & l0 else (1 if set(tickers) & l1 else None)
        if lvl is None:
            return out
        for t in tickers:
            try:
                d = raw.xs(t, axis=1, level=lvl)
            except KeyError:
                continue
            d = d.dropna(how="all")
            if "Close" in d.columns and d["Close"].notna().any():
                out[t] = _clean_frame(d)
    elif len(tickers) == 1 and "Close" in raw.columns:
        out[tickers[0]] = _clean_frame(raw.dropna(how="all"))
    return out


def _history_one(ticker, start, retries=2):
    import yfinance as yf
    for k in range(retries):
        try:
            h = yf.Ticker(ticker).history(start=start, auto_adjust=False, actions=True)
            if h is not None and len(h):
                return _clean_frame(h)
            return pd.DataFrame()
        except Exception as e:
            log.warning("%s attempt %d failed: %s", ticker, k + 1, e)
            time.sleep(2 * (k + 1))
    return pd.DataFrame()


def download(tickers, start, chunk=25, retries=3) -> dict:
    """ticker -> DataFrame(date, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits). Batched + threaded."""
    import yfinance as yf
    tickers = list(dict.fromkeys(tickers))
    res = {}
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for k in range(retries):
            try:
                raw = yf.download(part, start=start, auto_adjust=False, actions=True, group_by="ticker",
                                  threads=True, progress=False)
                got = _split_download(raw, part)
                res.update(got)
                if len(got) >= len(part) * 0.6:
                    break
            except Exception as e:
                log.warning("batch download attempt %d failed: %s", k + 1, e)
            time.sleep(2 * (k + 1))
    missing = [t for t in tickers if t not in res]
    for t in missing:                       # one-by-one rescue for anything the batch missed
        h = _history_one(t, start)
        if len(h):
            res[t] = h
    return res


def refresh_dim_symbol():
    u = get_universe().copy()
    u["is_active"] = True
    upsert_df(u, "dim_symbol")
    return u


def _px_frame(sym, d):
    return pd.DataFrame({"symbol": sym, "date": d["date"], "open": d.get("Open"), "high": d.get("High"), "low": d.get("Low"),
                         "close": d["Close"], "adj_close": d["Adj Close"] if "Adj Close" in d else d["Close"],
                         "volume": d["Volume"].fillna(0).astype("int64") if "Volume" in d else 0}).dropna(subset=["close"])


def _actions_frame(sym, d):
    ca = []
    for col, typ in (("Dividends", "dividend"), ("Stock Splits", "split")):
        if col in d:
            x = d[d[col].fillna(0) > 0]
            if len(x):
                ca.append(pd.DataFrame({"symbol": sym, "date": x["date"], "action_type": typ, "value": x[col]}))
    return pd.concat(ca) if ca else pd.DataFrame(columns=["symbol", "date", "action_type", "value"])


def sync_prices(years=None, force_full=False, symbols=None) -> dict:
    uni = refresh_dim_symbol()
    if symbols:
        uni = uni[uni.symbol.isin(symbols)]
    t2s = dict(zip(uni.yf_ticker, uni.symbol))
    start_full = _start_target(years)
    full_done = get_states("prices_full:")
    rng = read_sql("SELECT symbol, MAX(date) AS mx FROM prices_daily GROUP BY symbol", parse_dates=["mx"]).set_index("symbol")["mx"]
    full_syms = [s for s in uni.symbol if force_full or s not in full_done or s not in rng.index]
    incr_syms = [s for s in uni.symbol if s not in full_syms]
    s2t = {v: k for k, v in t2s.items()}
    rows, failed, redo = 0, [], []
    existing_ca = None
    # ---- 1) incremental top-up (cheap) ----
    if incr_syms:
        start_inc = max(min(rng[s] for s in incr_syms) - pd.Timedelta(days=OVERLAP_DAYS), start_full)
        got = download([s2t[s] for s in incr_syms], start_inc)
        existing_ca = read_sql("SELECT symbol, date, action_type FROM corporate_actions", parse_dates=["date"])
        known = set(zip(existing_ca.symbol, existing_ca.date, existing_ca.action_type))
        for t, d in got.items():
            sym = t2s[t]
            acts = _actions_frame(sym, d)
            if len(acts) and any((r.symbol, pd.Timestamp(r.date), r.action_type) not in known for r in acts.itertuples()):
                redo.append(sym)            # new dividend/split -> history must be re-pulled
                continue
            rows += upsert_df(_px_frame(sym, d), "prices_daily")
        failed += [t2s[t] for t in (s2t[s] for s in incr_syms) if t not in got]
    # ---- 2) full 5-year pulls ----
    todo = full_syms + [s for s in redo if s not in full_syms]
    if todo:
        got = download([s2t[s] for s in todo], start_full)
        for t, d in got.items():
            sym = t2s[t]
            rows += upsert_df(_px_frame(sym, d), "prices_daily")
            a = _actions_frame(sym, d)
            if len(a):
                upsert_df(a, "corporate_actions")
            set_state(f"prices_full:{sym}", pd.Timestamp.today().date())
        failed += [t2s[t] for t in (s2t[s] for s in todo) if t not in got]
    status = "warn" if failed else "ok"
    log_ingestion("yf_stocks", rows, status, f"full={len(todo)} incr={len(incr_syms)} redo_for_corp_action={len(redo)} failed={failed}")
    return {"rows": rows, "full_pulls": len(todo), "top_ups": len(incr_syms), "re_pulled_after_corp_action": redo, "failed": sorted(set(failed))}


def sync_indices(years=None, force_full=False) -> dict:
    items = {config.NIFTY_TICKER: config.NIFTY_NAME, config.VIX_TICKER: config.VIX_NAME}
    start_full = _start_target(years)
    done = get_states("index_full:")
    rng = read_sql("SELECT index_name, MAX(date) AS mx FROM index_prices GROUP BY index_name", parse_dates=["mx"]).set_index("index_name")["mx"]
    full = [t for t, n in items.items() if force_full or n not in done or n not in rng.index]
    inc = [t for t in items if t not in full]
    out = {}
    for tickers, start, is_full in ((full, start_full, True),
                                    (inc, max(min([rng[items[t]] for t in inc] or [start_full]) - pd.Timedelta(days=OVERLAP_DAYS), start_full), False)):
        if not tickers:
            continue
        got = download(tickers, start)
        for t, d in got.items():
            name = items[t]
            df = pd.DataFrame({"index_name": name, "date": d["date"], "open": d.get("Open"), "high": d.get("High"), "low": d.get("Low"),
                               "close": d["Close"], "volume": d["Volume"].fillna(0).astype("int64") if "Volume" in d else 0}).dropna(subset=["close"])
            out[name] = upsert_df(df, "index_prices")
            if is_full:
                set_state(f"index_full:{name}", pd.Timestamp.today().date())
        for t in tickers:
            if t not in got:
                out[items[t]] = "FAILED"
    log_ingestion("yf_index", sum(v for v in out.values() if isinstance(v, int)), "ok" if "FAILED" not in out.values() else "warn", str(out))
    return out


def fetch_fundamentals(max_workers=6) -> int:
    """Snapshot of current fundamentals (threaded). Yahoo has no history, so this is NOT used as a historical model feature."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    uni = get_universe()
    today = pd.Timestamp.today().normalize()

    def one(r):
        try:
            info = yf.Ticker(r.yf_ticker).info or {}
        except Exception:
            return None
        if not info:
            return None
        return {"symbol": r.symbol, "snapshot_date": today, "market_cap": info.get("marketCap"), "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"), "price_to_book": info.get("priceToBook"), "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"), "profit_margin": info.get("profitMargins"),
                "dividend_yield": info.get("dividendYield"), "beta": info.get("beta"), "sector": info.get("sector"), "industry": info.get("industry")}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = [r for r in ex.map(one, [r for _, r in uni.iterrows()]) if r]
    df = pd.DataFrame(rows)
    n = upsert_df(df, "fundamentals")
    if len(df):
        upsert_df(df[["symbol", "sector", "industry"]].merge(uni, on="symbol")[["symbol", "yf_ticker", "name", "sector", "industry"]], "dim_symbol")
    log_ingestion("fundamentals", n, "ok")
    return n


# --- backwards-compatible names used elsewhere ---
def fetch_stock_prices(start=None, incremental=True, symbols=None):
    return sync_prices(symbols=symbols, force_full=not incremental)

def fetch_nifty_vix(start=None, incremental=True):
    return sync_indices(force_full=not incremental)
