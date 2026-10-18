"""Database schema (SQLAlchemy Core). Works on PostgreSQL (primary) and SQLite (tests)."""
from sqlalchemy import (MetaData, Table, Column, String, Date, DateTime, Float, BigInteger, Integer,
                        Boolean, Text, JSON, UniqueConstraint, Index, func)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()
J = JSON().with_variant(JSONB(), "postgresql")
ID = BigInteger().with_variant(Integer(), "sqlite")

dim_symbol = Table("dim_symbol", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("yf_ticker", String(40), nullable=False),
    Column("name", String(120)), Column("sector", String(80)), Column("industry", String(120)),
    Column("is_active", Boolean, default=True),
    Column("updated_at", DateTime, server_default=func.now()))

prices_daily = Table("prices_daily", metadata,
    Column("symbol", String(32), primary_key=True), Column("date", Date, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float), Column("close", Float),
    Column("adj_close", Float), Column("volume", BigInteger),
    Index("ix_prices_date", "date"))

index_prices = Table("index_prices", metadata,   # NIFTY50, INDIAVIX
    Column("index_name", String(32), primary_key=True), Column("date", Date, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float), Column("close", Float),
    Column("volume", BigInteger))

corporate_actions = Table("corporate_actions", metadata,
    Column("symbol", String(32), primary_key=True), Column("date", Date, primary_key=True),
    Column("action_type", String(16), primary_key=True),   # dividend | split
    Column("value", Float))

participant_oi = Table("participant_oi", metadata,   # NSE FII/DII/Pro/Client F&O open interest
    Column("date", Date, primary_key=True), Column("client_type", String(16), primary_key=True),
    Column("fut_idx_long", Float), Column("fut_idx_short", Float),
    Column("fut_stk_long", Float), Column("fut_stk_short", Float),
    Column("opt_idx_call_long", Float), Column("opt_idx_put_long", Float),
    Column("opt_idx_call_short", Float), Column("opt_idx_put_short", Float),
    Column("opt_stk_call_long", Float), Column("opt_stk_put_long", Float),
    Column("opt_stk_call_short", Float), Column("opt_stk_put_short", Float),
    Column("total_long", Float), Column("total_short", Float),
    Column("fut_idx_long_reconstructed", Boolean, default=False))

news = Table("news", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("published_at", DateTime, nullable=False),        # UTC
    Column("entity", String(40), nullable=False),            # symbol or macro topic key
    Column("entity_type", String(10), nullable=False),       # stock | macro
    Column("title", Text, nullable=False), Column("source", String(120)), Column("url", Text),
    Column("label", String(10)), Column("p_pos", Float), Column("p_neg", Float), Column("p_neu", Float),
    Column("sentiment", Float),                              # p_pos - p_neg in [-1, 1]
    Column("scored_with", String(20)),
    Column("ingested_at", DateTime, server_default=func.now()),
    UniqueConstraint("entity", "title", "published_at", name="uq_news"),
    Index("ix_news_pub", "published_at"))

fundamentals = Table("fundamentals", metadata,   # snapshots (NOT point-in-time -> never used as historical features)
    Column("symbol", String(32), primary_key=True), Column("snapshot_date", Date, primary_key=True),
    Column("market_cap", Float), Column("trailing_pe", Float), Column("forward_pe", Float),
    Column("price_to_book", Float), Column("roe", Float), Column("debt_to_equity", Float),
    Column("profit_margin", Float), Column("dividend_yield", Float), Column("beta", Float),
    Column("sector", String(80)), Column("industry", String(120)))

market_regime = Table("market_regime", metadata,
    Column("date", Date, primary_key=True), Column("regime", Integer), Column("regime_name", String(16)),
    Column("p_calm", Float), Column("p_neutral", Float), Column("p_stress", Float))

model_runs = Table("model_runs", metadata,
    Column("run_id", String(40), primary_key=True), Column("model_name", String(60)),
    Column("trained_at", DateTime, server_default=func.now()), Column("train_end_date", Date),
    Column("metrics", J), Column("params", J), Column("feature_list", J), Column("artifact_path", Text))

predictions = Table("predictions", metadata,   # prediction history
    Column("symbol", String(32), primary_key=True), Column("asof_date", Date, primary_key=True),
    Column("source", String(20), primary_key=True),          # live | walk_forward
    Column("run_id", String(40)),
    Column("prob_up", Float), Column("exp_excess_ret", Float),
    Column("ret_q10", Float), Column("ret_q50", Float), Column("ret_q90", Float),
    Column("prob_risk", Float), Column("score", Float), Column("signal", String(10)),
    Column("confidence", Float), Column("risk_score", Float), Column("risk_level", String(10)),
    Column("top_factors", J), Column("reason", Text), Column("created_at", DateTime, server_default=func.now()),
    Index("ix_pred_asof", "asof_date"))

risk_metrics = Table("risk_metrics", metadata,
    Column("symbol", String(32), primary_key=True), Column("asof_date", Date, primary_key=True),
    Column("var95_1d", Float), Column("es95_1d", Float), Column("var99_1d", Float), Column("es99_1d", Float),
    Column("garch_vol_1d", Float), Column("garch_vol_5d", Float), Column("garch_vol_ann", Float),
    Column("realized_vol_20d_ann", Float), Column("downside_dev_ann", Float),
    Column("max_drawdown", Float), Column("cur_drawdown", Float), Column("dd_duration_days", Integer),
    Column("beta", Float), Column("risk_score", Float), Column("risk_level", String(10)))

backtest_runs = Table("backtest_runs", metadata,
    Column("run_id", String(40), primary_key=True), Column("created_at", DateTime, server_default=func.now()),
    Column("name", String(80)), Column("params", J), Column("metrics", J), Column("extras", J))

backtest_equity = Table("backtest_equity", metadata,
    Column("run_id", String(40), primary_key=True), Column("date", Date, primary_key=True),
    Column("equity", Float), Column("ret", Float), Column("benchmark_equity", Float), Column("turnover", Float))

portfolio_results = Table("portfolio_results", metadata,
    Column("run_id", String(40), primary_key=True), Column("method", String(30), primary_key=True),
    Column("symbol", String(32), primary_key=True), Column("weight", Float), Column("created_at", DateTime, server_default=func.now()))

feature_importance = Table("feature_importance", metadata,
    Column("run_id", String(40), primary_key=True), Column("feature", String(80), primary_key=True),
    Column("mean_abs_shap", Float))

drift_metrics = Table("drift_metrics", metadata,
    Column("asof_date", Date, primary_key=True), Column("feature", String(80), primary_key=True),
    Column("psi", Float), Column("status", String(10)))

live_performance = Table("live_performance", metadata,
    Column("asof_date", Date, primary_key=True), Column("window_days", Integer, primary_key=True),
    Column("n_obs", Integer), Column("ic", Float), Column("auc", Float), Column("hit_rate", Float),
    Column("brier", Float), Column("status", String(10)))

data_quality_log = Table("data_quality_log", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("run_ts", DateTime, server_default=func.now()), Column("table_name", String(40)),
    Column("check_name", String(60)), Column("status", String(10)), Column("detail", Text))

ingestion_log = Table("ingestion_log", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("run_ts", DateTime, server_default=func.now()), Column("source", String(40)),
    Column("rows", Integer), Column("status", String(10)), Column("message", Text))

sync_state = Table("sync_state", metadata,     # remembers what has already been pulled (5y history done? last fundamentals?)
    Column("key", String(80), primary_key=True), Column("value", String(200)),
    Column("updated_at", DateTime, server_default=func.now()))
