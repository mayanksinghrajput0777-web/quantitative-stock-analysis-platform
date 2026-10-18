"""Load the CSV files you already have into PostgreSQL. Put them in data/raw/ :
   participant_oi_28_11_2020_2026.csv   (FII/DII file)          -> participant_oi
   stocks_500.csv  / stocks.csv         (wide: Date, TCS_Close..)-> prices_daily
   nifty.csv, vix.csv                   (yfinance exports)       -> index_prices
   news.csv / historical_macro_sentiment_scores.csv              -> news
Any file that is missing is simply skipped (use `fetch-yf` to download instead)."""
import re, logging
import numpy as np
import pandas as pd
import config
from db.connection import upsert_df, log_ingestion
from ingestion import fii_dii
from ingestion.universe import get_universe

log = logging.getLogger("csv")


def _find(patterns):
    for p in patterns:
        hits = sorted(config.RAW_DIR.glob(p))
        if hits:
            return hits[0]
    return None


def read_yf_csv(path) -> pd.DataFrame:
    """Handles normal CSVs and yfinance multi-header exports ('Price/Ticker/Date' rows)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df.rename(columns={first: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def load_index_csv(path, name) -> int:
    df = read_yf_csv(path)
    low = {c.lower(): c for c in df.columns}
    g = lambda k: df[low[k]] if k in low else np.nan
    out = pd.DataFrame({"index_name": name, "date": df["Date"], "open": g("open"), "high": g("high"),
                        "low": g("low"), "close": g("close"),
                        "volume": pd.Series(g("volume")).fillna(0).astype("int64") if "volume" in low else 0})
    n = upsert_df(out.dropna(subset=["close"]), "index_prices")
    log_ingestion(f"csv_{name}", n, "ok", str(path.name))
    return n


def load_wide_stocks_csv(path) -> int:
    """Wide format like your stocks_500.csv: Date, ADANIENT_Close, ADANIENT_Open, ... (or Close_ADANIENT)."""
    df = read_yf_csv(path)
    uni = get_universe()
    known = set(uni.symbol)
    long_parts = {}
    pat = re.compile(r"^(?P<a>.+?)[_ ](?P<b>open|high|low|close|adj close|adj_close|volume)$", re.I)
    pat2 = re.compile(r"^(?P<b>open|high|low|close|adj close|adj_close|volume)[_ ](?P<a>.+)$", re.I)
    for c in df.columns[1:]:
        m = pat.match(c) or pat2.match(c)
        if not m:
            continue
        sym = m.group("a").strip().replace(".NS", "").upper()
        fld = m.group("b").lower().replace(" ", "_")
        long_parts.setdefault(sym, {})[fld] = df[c]
    frames = []
    for sym, d in long_parts.items():
        if "close" not in d:
            continue
        f = pd.DataFrame({"symbol": sym, "date": df["Date"], "open": d.get("open"), "high": d.get("high"),
                          "low": d.get("low"), "close": d["close"],
                          "adj_close": d.get("adj_close", d["close"]),
                          "volume": d.get("volume", pd.Series(0, index=df.index)).fillna(0).astype("int64")})
        frames.append(f.dropna(subset=["close"]))
    if not frames:
        return 0
    out = pd.concat(frames)
    syms = pd.DataFrame({"symbol": sorted(out.symbol.unique())})
    syms = syms.merge(uni, on="symbol", how="left")
    syms["yf_ticker"] = syms["yf_ticker"].fillna(syms["symbol"] + ".NS"); syms["name"] = syms["name"].fillna(syms["symbol"])
    upsert_df(syms.assign(is_active=True), "dim_symbol")
    n = upsert_df(out, "prices_daily")
    log_ingestion("csv_stocks_wide", n, "ok", f"{out.symbol.nunique()} symbols")
    return n


def load_news_csv(path) -> int:
    """Accepts your historical_macro_sentiment_scores.csv (Date, Title, Source, Topic, FinBERT_Label, FinBERT_Score)
    or any file with date/title/sentiment columns."""
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    dcol = next((low[k] for k in low if "date" in k), None)
    tcol = next((low[k] for k in low if k in ("title", "headline", "text")), None)
    if dcol is None or tcol is None:
        return 0
    pub = pd.to_datetime(df[dcol], errors="coerce")
    out = pd.DataFrame({"published_at": pub + pd.Timedelta(hours=6), "title": df[tcol].astype(str)})
    out["source"] = df[low["source"]] if "source" in low else "csv"
    if "topic" in low:
        out["entity"] = df[low["topic"]].astype(str); out["entity_type"] = "macro"
    elif "symbol" in low:
        out["entity"] = df[low["symbol"]].astype(str); out["entity_type"] = "stock"
    else:
        out["entity"] = "NIFTY_50"; out["entity_type"] = "macro"
    if "finbert_label" in low:
        lab = df[low["finbert_label"]].astype(str).str.lower()
        sc = pd.to_numeric(df[low["finbert_score"]], errors="coerce").fillna(0.5) if "finbert_score" in low else 0.5
        out["label"] = lab
        out["p_pos"] = np.where(lab == "positive", sc, 0.0); out["p_neg"] = np.where(lab == "negative", sc, 0.0)
        out["p_neu"] = np.where(lab == "neutral", sc, 0.0)
        out["sentiment"] = out["p_pos"] - out["p_neg"]
    elif any(k in low for k in ("sentiment", "sent_score")):
        s = pd.to_numeric(df[low.get("sentiment", low.get("sent_score"))], errors="coerce")
        out["sentiment"] = s; out["label"] = np.where(s > 0.1, "positive", np.where(s < -0.1, "negative", "neutral"))
    else:
        return 0
    out["scored_with"] = "csv"
    out = out.dropna(subset=["published_at", "sentiment"]).drop_duplicates(["entity", "title", "published_at"])
    n = upsert_df(out, "news", update=False)
    log_ingestion("csv_news", n, "ok", str(path.name))
    return n


def load_all_csv() -> dict:
    res = {}
    f = _find(["participant_oi*.csv", "*fii*.csv"])
    res["participant_oi"] = fii_dii.load_participant_csv(f) if f else 0
    f = _find(["nifty*.csv", "NIFTY*.csv"]);  res["nifty"] = load_index_csv(f, config.NIFTY_NAME) if f else 0
    f = _find(["vix*.csv", "VIX*.csv", "*india_vix*.csv"]); res["vix"] = load_index_csv(f, config.VIX_NAME) if f else 0
    f = _find(["stocks*.csv", "*top50*.csv", "*merged*.csv"]); res["stocks"] = load_wide_stocks_csv(f) if f else 0
    f = _find(["news*.csv", "historical_macro_sentiment*.csv"]); res["news"] = load_news_csv(f) if f else 0
    for k, v in res.items():
        print(f"  {'✅' if v else '–'} {k}: {v} rows")
    return res
