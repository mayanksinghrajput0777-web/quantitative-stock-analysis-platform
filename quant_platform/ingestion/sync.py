"""ONE function for data pulling:  first run = last 5 years (fast, batched) ; every later run = only the new days.
Safe to run any number of times (everything is an upsert)."""
import datetime as dt
import pandas as pd
import config
from db.connection import read_sql, get_state, set_state, log_ingestion
from ingestion import yf_prices, fii_dii, news


def _span(table, col="date", where=""):
    try:
        d = read_sql(f"SELECT MIN({col}) a, MAX({col}) b, COUNT(*) n FROM {table} {where}", parse_dates=["a", "b"]).iloc[0]
        return f"{int(d.n):,} rows  {d.a.date() if pd.notna(d.a) else '-'} → {d.b.date() if pd.notna(d.b) else '-'}"
    except Exception:
        return "-"


def run_sync(years=None, with_news=True, news_days=None, with_fundamentals=True, force_full=False) -> dict:
    out = {}
    def safe(name, fn):
        try:
            out[name] = fn(); print(f"   ✅ {name}: {out[name]}")
        except SystemExit:
            raise
        except Exception as e:
            out[name] = f"ERROR {type(e).__name__}: {e}"; log_ingestion(name, 0, "error", str(e)); print(f"   ❌ {name}: {e}")
    print("📥 [1/4] stock prices (batched; 5-year history the first time, only new days afterwards)")
    safe("stocks", lambda: yf_prices.sync_prices(years=years, force_full=force_full))
    print("📥 [2/4] NIFTY 50 + India VIX")
    safe("index", lambda: yf_prices.sync_indices(years=years, force_full=force_full))
    print("📥 [3/4] FII/DII participant OI (NSE archives, parallel; only missing days)")
    safe("fii_dii_rows", lambda: fii_dii.fetch_incremental(years))
    if with_news:
        print("📥 [4/4] news (Google News → sentiment)")
        safe("news_rows", lambda: news.fetch_news(days_back=news_days or config.NEWS_DAYS))
    if with_fundamentals:
        last = get_state("fundamentals_last")
        if force_full or last is None or (pd.Timestamp.today() - pd.Timestamp(last)).days >= 7:
            print("📥 fundamentals snapshot (weekly)")
            safe("fundamentals", lambda: yf_prices.fetch_fundamentals())
            set_state("fundamentals_last", dt.date.today())
    print("\n📚 Database now holds:")
    print(f"   prices_daily   {_span('prices_daily')}")
    print(f"   index_prices   {_span('index_prices')}")
    print(f"   participant_oi {_span('participant_oi')}")
    print(f"   news           {_span('news', 'published_at')}")
    return out
