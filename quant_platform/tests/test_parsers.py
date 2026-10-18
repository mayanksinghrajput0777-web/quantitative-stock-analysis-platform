"""Offline tests of the network-facing parsers using mocked HTTP responses.  python tests/test_parsers.py"""
import os, sys, pathlib, tempfile
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
db = pathlib.Path(tempfile.gettempdir()) / "quant_parsers.db"; db.unlink(missing_ok=True)
os.environ["QUANT_DATABASE_URL"] = f"sqlite:///{db.as_posix()}"
os.environ["USE_FINBERT"] = "0"
import pandas as pd
from unittest import mock
from db.connection import init_db, read_sql
init_db()

# ---- Google News RSS ----
RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Reliance shares surge after strong profit growth</title><link>http://x/1</link><pubDate>Mon, 05 Oct 2026 09:30:00 GMT</pubDate><source>ET</source></item>
<item><title>Reliance faces probe, shares slump</title><link>http://x/2</link><pubDate>Mon, 05 Oct 2026 11:00:00 GMT</pubDate><source>Mint</source></item>
<item><title>Reliance shares surge after strong profit growth</title><link>http://x/1</link><pubDate>Mon, 05 Oct 2026 09:30:00 GMT</pubDate><source>ET</source></item>
</channel></rss>"""
from ingestion import news
with mock.patch("ingestion.news.requests.get", return_value=mock.Mock(content=RSS)):
    n1 = news.fetch_news(days_back=2, stocks=False, macro=True, sleep=0)
    n2 = news.fetch_news(days_back=2, stocks=False, macro=True, sleep=0)   # re-run must not duplicate
d = read_sql("SELECT entity, title, sentiment, label, published_at FROM news")
print(f"news: inserted {n1} then {n2} on re-run; rows in DB = {len(d)}")
assert n1 > 0 and n2 == 0 and d.title.nunique() == 2
assert d[d.title.str.contains("surge")].sentiment.iloc[0] > 0 > d[d.title.str.contains("slump")].sentiment.iloc[0]
from features.sentiment import assign_trading_date
cal = pd.bdate_range("2026-10-01", "2026-10-09")
x = assign_trading_date(pd.Series(pd.to_datetime(["2026-10-05 09:30", "2026-10-05 11:00", "2026-10-02 12:00"])), cal)
print("trading-date assignment (UTC 09:30 = 15:00 IST -> same day; 11:00 UTC = 16:30 IST -> next day; Fri after close -> Mon):", [str(v.date()) for v in x])
assert str(x[0].date()) == "2026-10-05" and str(x[1].date()) == "2026-10-06" and str(x[2].date()) == "2026-10-05"

# ---- NSE participant OI raw file (title line + header, real NSE layout) ----
RAW = ("Participant wise Open Interest (no. of contracts) in Futures & Options as on 01-Dec-2026\n"
       "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short,Option Index Call Long,Option Index Put Long,"
       "Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
       "Total Long Contracts,Total Short Contracts\n"
       "Client,100,200,300,400,10,20,30,40,50,60,70,80,540,820\n"
       "DII,,50,1,2,3,4,5,6,7,8,9,10,23,82\n"       # empty Future Index Long -> must be rebuilt = 123-... see below
       "FII,10,20,30,40,1,2,3,4,5,6,7,8,54,82\n"
       "Pro,5,6,7,8,9,10,11,12,13,14,15,16,58,68\n"
       "TOTAL,115,276,338,450,23,36,49,62,75,88,101,114,675,1052\n")
from ingestion import fii_dii
resp = mock.Mock(status_code=200, text=RAW)
with mock.patch.object(fii_dii.requests.Session, "get", return_value=resp):
    df = fii_dii.download_day(pd.Timestamp("2026-12-01"))
dii = df[df.client_type == "DII"].iloc[0]
print("OI parsed rows:", len(df), "| DII fut_idx_long rebuilt =", dii.fut_idx_long, "reconstructed flag =", dii.fut_idx_long_reconstructed)
assert len(df) == 5 and dii.fut_idx_long == 23 - (1 + 3 + 4 + 7 + 8) == 0.0 and bool(dii.fut_idx_long_reconstructed)
# ---- your concatenated CSV ----
f = next(iter((ROOT / "data" / "raw").glob("participant_oi*.csv")), None)
if f is not None:
    full = fii_dii.parse_participant_oi(pd.read_csv(f))
    assert full.fut_idx_long.notna().all() and full.date.nunique() > 1000
    print(f"your CSV: {len(full)} rows, {full.date.nunique()} days, no empty Future Index Long")
print("\n✅ PARSER TESTS PASSED")
