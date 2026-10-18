"""News ingestion: Google News RSS (your original approach) + FinBERT scoring.
Improvements vs the original script:
  * keeps the real publication timestamp (pubDate) -> correct point-in-time alignment
  * per-stock news as well as macro topics, windowed requests (RSS caps ~100 items per query)
  * results stored in PostgreSQL (dedup on entity+title+time); FinBERT is run once per headline
  * falls back to a small financial lexicon if transformers/torch are not installed
"""
import time, datetime as dt, logging, urllib.parse, xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import numpy as np
import pandas as pd
import requests
import config
from db.connection import upsert_df, log_ingestion, read_sql
from ingestion.universe import get_universe

log = logging.getLogger("news")

MACRO_TOPICS = {   # from your HistoricalMacroSentimentEngine
    "NIFTY_50": "Nifty 50 NSE India",
    "INDIA_VIX": "India VIX Volatility",
    "US_10Y_BOND": "US 10 Year Treasury Yield",
    "BRENT_CRUDE_OIL": "Brent Crude Oil Price",
    "DOLLAR_INDEX": "Dollar Index DXY",
    "FII_FLOWS": "FII DII flows India market",
    "RBI_POLICY": "RBI monetary policy India",
}

POS = {"surge", "rally", "gain", "gains", "jumps", "soars", "record", "beat", "beats", "upgrade", "upgrades", "profit",
       "growth", "strong", "bullish", "buy", "outperform", "rises", "rise", "climbs", "boost", "wins", "order", "positive", "high"}
NEG = {"fall", "falls", "plunge", "plunges", "slump", "crash", "loss", "losses", "miss", "misses", "downgrade", "weak",
       "bearish", "sell", "underperform", "drops", "drop", "decline", "probe", "fraud", "penalty", "cut", "cuts",
       "slumps", "tumbles", "negative", "lawsuit", "default", "warning", "low", "fear"}


class SentimentScorer:
    def __init__(self, use_finbert: bool = None):
        self.use_finbert = config.USE_FINBERT if use_finbert is None else use_finbert
        self.pipe = None
        self.name = "lexicon"
        if self.use_finbert:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                print("🧠 Loading FinBERT (ProsusAI/finbert) ...")
                tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.pipe = pipeline("text-classification", model=mod, tokenizer=tok, top_k=None)
                self.name = "finbert"
            except Exception as e:
                print(f"⚠️ FinBERT unavailable ({type(e).__name__}); using lexicon fallback. pip install transformers torch")

    def score(self, texts, batch_size=16):
        texts = [str(t) for t in texts]
        out = []
        if self.pipe is not None:
            for i in range(0, len(texts), batch_size):
                res = self.pipe(texts[i:i + batch_size], truncation=True, max_length=128)
                for r in res:
                    d = {x["label"].lower(): float(x["score"]) for x in r}
                    out.append(self._pack(d.get("positive", 0), d.get("negative", 0), d.get("neutral", 0)))
            return out
        for t in texts:
            w = set(t.lower().replace(",", " ").replace(".", " ").split())
            p, n = len(w & POS), len(w & NEG)
            tot = p + n
            if tot == 0:
                out.append(self._pack(0.0, 0.0, 1.0))
            else:
                pp, nn = p / (tot + 1.0), n / (tot + 1.0)
                out.append(self._pack(pp, nn, max(0.0, 1 - pp - nn)))
        return out

    @staticmethod
    def _pack(p, n, u):
        label = max([("positive", p), ("negative", n), ("neutral", u)], key=lambda x: x[1])[0]
        return {"label": label, "p_pos": p, "p_neg": n, "p_neu": u, "sentiment": p - n}


def _rss(query: str, after: dt.date, before: dt.date, max_results=100):
    q = f"{query} after:{after:%Y-%m-%d} before:{before:%Y-%m-%d}"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    items = []
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        for it in root.findall(".//item")[:max_results]:
            pub = it.find("pubDate")
            try:
                ts = parsedate_to_datetime(pub.text).astimezone(dt.timezone.utc).replace(tzinfo=None)
            except Exception:
                continue
            src = it.find("source")
            items.append({"published_at": ts, "title": (it.find("title").text or "").strip(),
                          "source": src.text if src is not None else "Unknown",
                          "url": it.find("link").text if it.find("link") is not None else None})
    except Exception as e:
        log.debug("rss fail %s", e)
    return items


def _windows(days_back: int, step: int):
    today = dt.date.today()
    start = today - dt.timedelta(days=days_back)
    cur = start
    while cur <= today:
        yield cur, min(cur + dt.timedelta(days=step), today + dt.timedelta(days=1))
        cur += dt.timedelta(days=step)


def fetch_news(days_back: int = None, stocks: bool = True, macro: bool = True, max_stocks: int = None,
               scorer: SentimentScorer = None, sleep: float = 0.4) -> int:
    days_back = days_back or config.NEWS_DAYS
    scorer = scorer or SentimentScorer()
    jobs = []
    if macro:
        for k, q in MACRO_TOPICS.items():
            for a, b in _windows(days_back, 7):
                jobs.append((k, "macro", q, a, b))
    if stocks:
        uni = get_universe()
        if max_stocks:
            uni = uni.head(max_stocks)
        for _, r in uni.iterrows():
            for a, b in _windows(days_back, 30):
                jobs.append((r.symbol, "stock", f"{r['name']} share NSE", a, b))
    from concurrent.futures import ThreadPoolExecutor
    def work(job):
        ent, typ, q, a, b = job
        items = _rss(q, a, b)
        for it in items:
            it.update({"entity": ent, "entity_type": typ})
        time.sleep(sleep)
        return items
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, items in enumerate(ex.map(work, jobs), 1):
            rows += items
            if i % 40 == 0:
                print(f"  news requests {i}/{len(jobs)} ({len(rows)} headlines)")
    if not rows:
        log_ingestion("news", 0, "warn", "no news fetched (network/blocked?)")
        print("⚠️ no news fetched")
        return 0
    df = pd.DataFrame(rows).drop_duplicates(["entity", "title", "published_at"])
    df = _drop_existing(df)
    if df.empty:
        return 0
    sc = pd.DataFrame(scorer.score(df["title"].tolist()), index=df.index)
    df = pd.concat([df, sc], axis=1)
    df["scored_with"] = scorer.name
    n = upsert_df(df, "news", update=False)
    log_ingestion("news", n, "ok", scorer.name)
    return n


def _drop_existing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        old = read_sql("SELECT entity, title, published_at FROM news WHERE published_at >= :d",
                       {"d": df["published_at"].min()}, parse_dates=["published_at"])
    except Exception:
        return df
    if old.empty:
        return df
    key = set(zip(old.entity, old.title, old.published_at))
    mask = [(e, t, p) not in key for e, t, p in zip(df.entity, df.title, df.published_at)]
    return df[mask]
