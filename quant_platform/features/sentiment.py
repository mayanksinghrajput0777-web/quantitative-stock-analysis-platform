"""News sentiment features with correct timestamp alignment (news after 15:30 IST -> next trading day)."""
import numpy as np
import pandas as pd
import config


def assign_trading_date(published_utc: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    ist = pd.to_datetime(published_utc) + pd.Timedelta(hours=5, minutes=30)
    hour = ist.dt.hour + ist.dt.minute / 60.0
    eff = ist.dt.normalize() + pd.to_timedelta((hour >= config.NEWS_CUTOFF_HOUR_IST).astype(int), unit="D")
    pos = np.searchsorted(calendar.values, eff.values.astype("datetime64[ns]"), side="left")
    ok = pos < len(calendar)
    res = pd.Series(pd.NaT, index=published_utc.index, dtype="datetime64[ns]")
    res[ok] = calendar.values[pos[ok]]
    return res


def sentiment_features(news: pd.DataFrame, calendar: pd.DatetimeIndex):
    """returns (market_df indexed by date, stock dict of wide frames) """
    if news is None or news.empty:
        return pd.DataFrame(index=calendar), {}
    n = news.copy()
    n["tdate"] = assign_trading_date(n["published_at"], calendar)
    n = n.dropna(subset=["tdate", "sentiment"])
    # market-level: macro topics
    m = n[n.entity_type == "macro"].groupby("tdate").agg(mkt_sent=("sentiment", "mean"), mkt_news_n=("sentiment", "size"))
    mk = pd.DataFrame(index=calendar).join(m)
    mk["mkt_news_n"] = mk["mkt_news_n"].fillna(0)
    mk["mkt_sent"] = mk["mkt_sent"]
    mk["mkt_sent_3d"] = mk["mkt_sent"].rolling(3, min_periods=1).mean()
    mk["mkt_sent_10d"] = mk["mkt_sent"].rolling(10, min_periods=1).mean()
    mk["mkt_sent_mom"] = mk["mkt_sent_3d"] - mk["mkt_sent_10d"]
    mk["mkt_sent_z"] = (mk["mkt_sent_3d"] - mk["mkt_sent"].rolling(30, min_periods=5).mean()) / \
        mk["mkt_sent"].rolling(30, min_periods=5).std().replace(0, np.nan)
    mk = mk.fillna({"mkt_sent_3d": 0, "mkt_sent_10d": 0, "mkt_sent_mom": 0, "mkt_sent_z": 0})
    s = n[n.entity_type == "stock"].groupby(["tdate", "entity"]).agg(s=("sentiment", "mean"), c=("sentiment", "size")).reset_index()
    sent = s.pivot(index="tdate", columns="entity", values="s").reindex(calendar)
    cnt = s.pivot(index="tdate", columns="entity", values="c").reindex(calendar).fillna(0)
    d = {"news_sent_1d": sent.fillna(0),
         "news_sent_3d": sent.rolling(3, min_periods=1).mean().fillna(0),
         "news_sent_10d": sent.rolling(10, min_periods=1).mean().fillna(0),
         "news_count_3d": cnt.rolling(3, min_periods=1).sum()}
    d["news_sent_mom"] = d["news_sent_3d"] - d["news_sent_10d"]
    return mk, d


def coverage_days(news: pd.DataFrame, calendar) -> int:
    if news is None or news.empty:
        return 0
    return int(assign_trading_date(news["published_at"], calendar).dropna().nunique())
