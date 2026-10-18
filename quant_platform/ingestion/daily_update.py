"""Daily incremental pull (thin wrapper around ingestion.sync.run_sync)."""
from ingestion.sync import run_sync


def run_daily(with_news=True, with_fii=True):
    return run_sync(with_news=with_news, news_days=3, with_fundamentals=True)
