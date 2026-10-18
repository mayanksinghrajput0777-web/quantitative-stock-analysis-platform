-- Auto-generated PostgreSQL schema. Created by db/init_db.py (python run_pipeline.py init-db)

CREATE TABLE IF NOT EXISTS backtest_equity (
	run_id VARCHAR(40) NOT NULL, 
	date DATE NOT NULL, 
	equity FLOAT, 
	ret FLOAT, 
	benchmark_equity FLOAT, 
	turnover FLOAT, 
	PRIMARY KEY (run_id, date)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
	run_id VARCHAR(40) NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	name VARCHAR(80), 
	params JSONB, 
	metrics JSONB, 
	extras JSONB, 
	PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
	symbol VARCHAR(32) NOT NULL, 
	date DATE NOT NULL, 
	action_type VARCHAR(16) NOT NULL, 
	value FLOAT, 
	PRIMARY KEY (symbol, date, action_type)
);

CREATE TABLE IF NOT EXISTS data_quality_log (
	id BIGSERIAL NOT NULL, 
	run_ts TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	table_name VARCHAR(40), 
	check_name VARCHAR(60), 
	status VARCHAR(10), 
	detail TEXT, 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS dim_symbol (
	symbol VARCHAR(32) NOT NULL, 
	yf_ticker VARCHAR(40) NOT NULL, 
	name VARCHAR(120), 
	sector VARCHAR(80), 
	industry VARCHAR(120), 
	is_active BOOLEAN, 
	updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (symbol)
);

CREATE TABLE IF NOT EXISTS drift_metrics (
	asof_date DATE NOT NULL, 
	feature VARCHAR(80) NOT NULL, 
	psi FLOAT, 
	status VARCHAR(10), 
	PRIMARY KEY (asof_date, feature)
);

CREATE TABLE IF NOT EXISTS feature_importance (
	run_id VARCHAR(40) NOT NULL, 
	feature VARCHAR(80) NOT NULL, 
	mean_abs_shap FLOAT, 
	PRIMARY KEY (run_id, feature)
);

CREATE TABLE IF NOT EXISTS fundamentals (
	symbol VARCHAR(32) NOT NULL, 
	snapshot_date DATE NOT NULL, 
	market_cap FLOAT, 
	trailing_pe FLOAT, 
	forward_pe FLOAT, 
	price_to_book FLOAT, 
	roe FLOAT, 
	debt_to_equity FLOAT, 
	profit_margin FLOAT, 
	dividend_yield FLOAT, 
	beta FLOAT, 
	sector VARCHAR(80), 
	industry VARCHAR(120), 
	PRIMARY KEY (symbol, snapshot_date)
);

CREATE TABLE IF NOT EXISTS index_prices (
	index_name VARCHAR(32) NOT NULL, 
	date DATE NOT NULL, 
	open FLOAT, 
	high FLOAT, 
	low FLOAT, 
	close FLOAT, 
	volume BIGINT, 
	PRIMARY KEY (index_name, date)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
	id BIGSERIAL NOT NULL, 
	run_ts TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	source VARCHAR(40), 
	rows INTEGER, 
	status VARCHAR(10), 
	message TEXT, 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS live_performance (
	asof_date DATE NOT NULL, 
	window_days INTEGER NOT NULL, 
	n_obs INTEGER, 
	ic FLOAT, 
	auc FLOAT, 
	hit_rate FLOAT, 
	brier FLOAT, 
	status VARCHAR(10), 
	PRIMARY KEY (asof_date, window_days)
);

CREATE TABLE IF NOT EXISTS market_regime (
	date DATE NOT NULL, 
	regime INTEGER, 
	regime_name VARCHAR(16), 
	p_calm FLOAT, 
	p_neutral FLOAT, 
	p_stress FLOAT, 
	PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS model_runs (
	run_id VARCHAR(40) NOT NULL, 
	model_name VARCHAR(60), 
	trained_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	train_end_date DATE, 
	metrics JSONB, 
	params JSONB, 
	feature_list JSONB, 
	artifact_path TEXT, 
	PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS news (
	id BIGSERIAL NOT NULL, 
	published_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	entity VARCHAR(40) NOT NULL, 
	entity_type VARCHAR(10) NOT NULL, 
	title TEXT NOT NULL, 
	source VARCHAR(120), 
	url TEXT, 
	label VARCHAR(10), 
	p_pos FLOAT, 
	p_neg FLOAT, 
	p_neu FLOAT, 
	sentiment FLOAT, 
	scored_with VARCHAR(20), 
	ingested_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (id), 
	CONSTRAINT uq_news UNIQUE (entity, title, published_at)
);

CREATE INDEX IF NOT EXISTS ix_news_pub ON news (published_at);

CREATE TABLE IF NOT EXISTS participant_oi (
	date DATE NOT NULL, 
	client_type VARCHAR(16) NOT NULL, 
	fut_idx_long FLOAT, 
	fut_idx_short FLOAT, 
	fut_stk_long FLOAT, 
	fut_stk_short FLOAT, 
	opt_idx_call_long FLOAT, 
	opt_idx_put_long FLOAT, 
	opt_idx_call_short FLOAT, 
	opt_idx_put_short FLOAT, 
	opt_stk_call_long FLOAT, 
	opt_stk_put_long FLOAT, 
	opt_stk_call_short FLOAT, 
	opt_stk_put_short FLOAT, 
	total_long FLOAT, 
	total_short FLOAT, 
	fut_idx_long_reconstructed BOOLEAN, 
	PRIMARY KEY (date, client_type)
);

CREATE TABLE IF NOT EXISTS portfolio_results (
	run_id VARCHAR(40) NOT NULL, 
	method VARCHAR(30) NOT NULL, 
	symbol VARCHAR(32) NOT NULL, 
	weight FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (run_id, method, symbol)
);

CREATE TABLE IF NOT EXISTS predictions (
	symbol VARCHAR(32) NOT NULL, 
	asof_date DATE NOT NULL, 
	source VARCHAR(20) NOT NULL, 
	run_id VARCHAR(40), 
	prob_up FLOAT, 
	exp_excess_ret FLOAT, 
	ret_q10 FLOAT, 
	ret_q50 FLOAT, 
	ret_q90 FLOAT, 
	prob_risk FLOAT, 
	score FLOAT, 
	signal VARCHAR(10), 
	confidence FLOAT, 
	risk_score FLOAT, 
	risk_level VARCHAR(10), 
	top_factors JSONB, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (symbol, asof_date, source)
);

CREATE INDEX IF NOT EXISTS ix_pred_asof ON predictions (asof_date);

CREATE TABLE IF NOT EXISTS prices_daily (
	symbol VARCHAR(32) NOT NULL, 
	date DATE NOT NULL, 
	open FLOAT, 
	high FLOAT, 
	low FLOAT, 
	close FLOAT, 
	adj_close FLOAT, 
	volume BIGINT, 
	PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS ix_prices_date ON prices_daily (date);

CREATE TABLE IF NOT EXISTS risk_metrics (
	symbol VARCHAR(32) NOT NULL, 
	asof_date DATE NOT NULL, 
	var95_1d FLOAT, 
	es95_1d FLOAT, 
	var99_1d FLOAT, 
	es99_1d FLOAT, 
	garch_vol_1d FLOAT, 
	garch_vol_5d FLOAT, 
	garch_vol_ann FLOAT, 
	realized_vol_20d_ann FLOAT, 
	downside_dev_ann FLOAT, 
	max_drawdown FLOAT, 
	cur_drawdown FLOAT, 
	dd_duration_days INTEGER, 
	beta FLOAT, 
	risk_score FLOAT, 
	risk_level VARCHAR(10), 
	PRIMARY KEY (symbol, asof_date)
);
