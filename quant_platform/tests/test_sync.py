"""Tests the 5-year-then-daily pull logic with a MOCKED Yahoo (no internet).  python tests/test_sync.py"""
import os, sys, pathlib, tempfile
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if True:
    db = pathlib.Path(tempfile.gettempdir()) / "quant_sync.db"; db.unlink(missing_ok=True)
    os.environ["QUANT_DATABASE_URL"] = f"sqlite:///{db.as_posix()}"
import numpy as np, pandas as pd
from unittest import mock
import config
from db.connection import init_db, read_sql
init_db()
from ingestion import yf_prices
from ingestion.universe import get_universe

TODAY = pd.Timestamp.today().normalize()
STATE = {"orientation": 0, "div_date": None, "drop": set(), "calls": []}


def fake_frame(ticker, start):
    idx = pd.bdate_range(pd.Timestamp(start), TODAY - pd.Timedelta(days=1))
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    px = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(idx))))
    d = pd.DataFrame({"Open": px, "High": px * 1.01, "Low": px * .99, "Close": px, "Adj Close": px * .98,
                      "Volume": rng.integers(1e5, 1e6, len(idx)).astype(float), "Dividends": 0.0, "Stock Splits": 0.0}, index=idx.tz_localize("Asia/Kolkata"))
    if STATE["div_date"] is not None and ticker == "TCS.NS":
        dd = pd.Timestamp(STATE["div_date"]).tz_localize("Asia/Kolkata")
        if dd in d.index: d.loc[dd, "Dividends"] = 10.0
    return d


def fake_download(tickers, start=None, **kw):
    STATE["calls"].append(("download", len(tickers), str(start)))
    frames = {t: fake_frame(t, start) for t in tickers if t not in STATE["drop"]}
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)                  # columns (ticker, field)
    if STATE["orientation"] == 1:
        df = df.swaplevel(axis=1)                   # columns (field, ticker)  -> newer yfinance layout
    return df


class FakeTicker:
    def __init__(self, t): self.t = t
    def history(self, start=None, **kw):
        STATE["calls"].append(("history", self.t, str(start)))
        return fake_frame(self.t, start)


def count(): return int(read_sql("SELECT COUNT(*) n FROM prices_daily")["n"].iloc[0])

with mock.patch("yfinance.download", side_effect=fake_download), mock.patch("yfinance.Ticker", FakeTicker):
    for orient in (0, 1):
        # fresh tables for each layout
        from db.connection import get_engine
        from sqlalchemy import text
        with get_engine().begin() as c:
            for t in ("prices_daily", "index_prices", "corporate_actions", "sync_state"): c.execute(text(f"DELETE FROM {t}"))
        STATE.update(orientation=orient, div_date=None, drop=set(), calls=[])
        n_sym = len(get_universe())
        # ---- run 1: first time -> full 5y ----
        r1 = yf_prices.sync_prices(); ix1 = yf_prices.sync_indices()
        c1 = count()
        mn = pd.Timestamp(read_sql("SELECT MIN(date) m FROM prices_daily")["m"].iloc[0])
        print(f"[layout {orient}] run 1: full={r1['full_pulls']} top_ups={r1['top_ups']} rows={c1:,} oldest={mn.date()} failed={r1['failed']} index={ix1}")
        assert r1["full_pulls"] == n_sym and r1["top_ups"] == 0 and not r1["failed"]
        assert (TODAY - mn).days >= 365 * config.HISTORY_YEARS - 10 and c1 > n_sym * 1200
        # ---- run 2: next day -> only top-up, no duplicates ----
        STATE["calls"] = []
        r2 = yf_prices.sync_prices(); yf_prices.sync_indices()
        c2 = count()
        starts = {c[2] for c in STATE["calls"] if c[0] == "download"}
        print(f"[layout {orient}] run 2: full={r2['full_pulls']} top_ups={r2['top_ups']} rows_sent={r2['rows']} (db rows {c2:,}, unchanged={c1 == c2}) download start dates={sorted(starts)[:2]}")
        assert r2["full_pulls"] == 0 and r2["top_ups"] == n_sym and c2 == c1 and r2["rows"] < n_sym * 10
        # ---- run 3: a NEW dividend appears inside the window -> that symbol is re-pulled in full ----
        STATE["div_date"] = (TODAY - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        while pd.Timestamp(STATE["div_date"]).weekday() >= 5: STATE["div_date"] = (pd.Timestamp(STATE["div_date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        r3 = yf_prices.sync_prices()
        print(f"[layout {orient}] run 3 (new dividend): re-pulled in full = {r3['re_pulled_after_corp_action']}")
        assert r3["re_pulled_after_corp_action"] == ["TCS"]
        r3b = yf_prices.sync_prices()
        assert r3b["re_pulled_after_corp_action"] == [], "dividend must be remembered, no endless re-pull"
        # ---- run 4: a delisted ticker missing from the batch is reported, others unaffected ----
        with get_engine().begin() as c: c.execute(text("DELETE FROM sync_state"))
        STATE["drop"] = {"TATAMOTORS.NS"}
        class NoHist(FakeTicker):
            def history(self, start=None, **kw): return pd.DataFrame() if self.t in STATE["drop"] else super().history(start, **kw)
        with mock.patch("yfinance.Ticker", NoHist):
            r4 = yf_prices.sync_prices()
        print(f"[layout {orient}] run 4 (delisted ticker): failed={r4['failed']}, others loaded={n_sym - len(r4['failed'])}")
        assert r4["failed"] == ["TATAMOTORS"]
        STATE["drop"] = set()

# ---- FII: first pull window + parallel + incremental start ----
from ingestion import fii_dii
calls = []
def fake_day(d, session=None, retries=2):
    calls.append(d)
    if d.weekday() >= 5 or d.day in (15, 26): return None            # pretend some holidays
    raw = pd.DataFrame({"date": d, "client_type": ["FII", "TOTAL"], **{c: [10.0, 20.0] for c in fii_dii.NUM}, "fut_idx_long_reconstructed": False})
    return raw
with mock.patch.object(fii_dii, "download_day", fake_day):
    n = fii_dii.fetch_incremental()
    first_span = (min(calls).date(), max(calls).date())
    calls.clear(); n2 = fii_dii.fetch_incremental()
print(f"FII first pull: {n} rows over {first_span}; next-day call re-requests {len(calls)} day(s) from {min(calls).date() if calls else '-'}")
assert n > 1000 and len(calls) <= 6
print("\n✅ SYNC TESTS PASSED")
