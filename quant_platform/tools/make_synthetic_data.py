"""SYNTHETIC market data for testing the pipeline without internet (NOT real prices!).
Usage: python tools/make_synthetic_data.py [n_symbols]   -> writes into whatever DATABASE_URL points to."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import config
from db.connection import init_db, upsert_df
from ingestion.universe import get_universe


def make(n_symbols=20, seed=11):
    rng = np.random.default_rng(seed)
    f = next(iter(config.RAW_DIR.glob("participant_oi*.csv")), None)
    if f is not None:
        d = pd.read_csv(f).iloc[:, -1].dropna().astype(str)
        cal = pd.DatetimeIndex(sorted(set(pd.to_datetime(d, format="%d-%m-%Y", errors="coerce").dropna())))
    else:
        cal = pd.bdate_range("2020-12-01", "2026-09-25")
    n = len(cal)
    # market with 3 volatility regimes
    reg = np.zeros(n, int)
    for i in range(1, n):
        reg[i] = reg[i - 1] if rng.random() > 0.015 else rng.choice([0, 1, 2], p=[.5, .35, .15])
    vol_m = np.array([0.006, 0.010, 0.022])[reg]
    rm = rng.standard_t(5, n) * vol_m / np.sqrt(5 / 3) + 0.0004
    nifty = 14000 * np.exp(np.cumsum(rm))
    vix = np.clip(np.array([12, 17, 30])[reg] + rng.normal(0, 1.5, n) + pd.Series(rm).rolling(5).mean().fillna(0).values * -150, 9, 80)
    ix = []
    for nm, s in ((config.NIFTY_NAME, nifty), (config.VIX_NAME, vix)):
        o = s * (1 + rng.normal(0, 0.002, n))
        ix.append(pd.DataFrame({"index_name": nm, "date": cal, "open": o, "high": np.maximum(o, s) * 1.003,
                                "low": np.minimum(o, s) * 0.997, "close": s, "volume": 0}))
    upsert_df(pd.concat(ix), "index_prices")
    uni = get_universe().head(n_symbols)
    upsert_df(uni.assign(is_active=True), "dim_symbol")
    rows, cas = [], []
    for _, u in uni.iterrows():
        beta = rng.uniform(0.6, 1.4); idv = rng.uniform(0.008, 0.016)
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = 0.985 * a[i - 1] + rng.normal(0, 0.00018)       # slow alpha factor (learnable via momentum)
        r = beta * rm + a + rng.standard_t(4, n) * idv / np.sqrt(2)
        r += -0.04 * np.r_[0, r[:-1]]                                # mild short-term reversal
        px = 500 * np.exp(np.cumsum(r))
        op = px * (1 + rng.normal(0, 0.004, n))
        hi = np.maximum(op, px) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lo = np.minimum(op, px) * (1 - np.abs(rng.normal(0, 0.005, n)))
        vol = (rng.lognormal(14, 0.4, n) * (1 + 3 * np.abs(r) / idv)).astype("int64")
        adj = px * np.linspace(0.93, 1.0, n)                         # pretend dividend adjustment
        rows.append(pd.DataFrame({"symbol": u.symbol, "date": cal, "open": op, "high": hi, "low": lo,
                                  "close": px, "adj_close": adj, "volume": vol}))
        cas.append(pd.DataFrame({"symbol": u.symbol, "date": cal[::250][1:], "action_type": "dividend", "value": 5.0}))
    upsert_df(pd.concat(rows), "prices_daily")
    upsert_df(pd.concat(cas), "corporate_actions")
    # 100 days of synthetic news
    recent = cal[-100:]
    nw = []
    for sym in uni.symbol.tolist():
        for d in recent[::2]:
            s = float(np.clip(rng.normal(0, 0.4), -0.95, 0.95))
            nw.append({"published_at": d + pd.Timedelta(hours=int(rng.integers(3, 14))), "entity": sym, "entity_type": "stock",
                       "title": f"{sym} synthetic headline {d.date()} {rng.integers(1e6)}", "source": "synthetic",
                       "label": "positive" if s > .1 else "negative" if s < -.1 else "neutral",
                       "p_pos": max(s, 0), "p_neg": max(-s, 0), "p_neu": 1 - abs(s), "sentiment": s, "scored_with": "synthetic"})
    for d in recent:
        s = float(np.clip(rng.normal(0, 0.3), -0.9, 0.9))
        nw.append({"published_at": d + pd.Timedelta(hours=6), "entity": "NIFTY_50", "entity_type": "macro",
                   "title": f"market synthetic headline {d.date()}", "source": "synthetic", "label": "neutral",
                   "p_pos": max(s, 0), "p_neg": max(-s, 0), "p_neu": 1 - abs(s), "sentiment": s, "scored_with": "synthetic"})
    upsert_df(pd.DataFrame(nw), "news", update=False)
    print(f"✅ synthetic data: {len(uni)} symbols x {n} days")


if __name__ == "__main__":
    init_db()
    make(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
