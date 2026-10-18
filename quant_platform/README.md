# Quant AI Platform — NSE stocks · PostgreSQL · Alpha + Risk ML · Backtesting · FastAPI + Streamlit

## ▶ THE SIMPLE PATH (Windows / VS Code)

Open this folder in VS Code → open a terminal (the `.venv` you already made) and run **only these**:

```bat
.venv\Scripts\activate
pip install -r requirements.txt      :: only if you have not installed everything yet

python start.py check                :: 1) tests packages + PostgreSQL (creates the database for you if missing)
python start.py setup                :: 2) FIRST TIME ONLY: pulls 5 years of data, trains, predicts, backtests  (~10-25 min)
python start.py daily                :: 3) EVERY DAY after ~7 pm IST: pulls only the new day, refreshes predictions
python start.py dashboard            :: open the web dashboard (http://localhost:8501)
```
Prefer clicking? Double-click `1_SETUP_first_time.bat`, `2_DAILY_update.bat`, `3_OPEN_DASHBOARD.bat`.
Quick trial (shorter validation): `python start.py setup --fast --skip-news`.

### Where is my output?
Everything lands in the **`output/`** folder:
* `final_predictions_next5.csv` ← **main result**: Stock, Prediction, Confidence, Expected Up/Down (5-day q90/q10), SHAP Top Factors, Suggestion (BUY/HOLD/SELL), Reason, Risk Level/Score
* `backtest_comparison.csv`, `stress_tests.csv`, `monte_carlo.json`, `model_metrics.json`, `data_quality.json`
* `eda/eda_report.html` (open in a browser)
…and everything is also stored in PostgreSQL (tables listed in `db/schema.sql`) and shown in the dashboard.

## 🔌 PostgreSQL connection
Edit only `.env`:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/quant_db
```
* The database `quant_db` is **created automatically** if it does not exist, tables too.
* `postgresql://` is converted internally to the psycopg2 driver (newer SQLAlchemy would otherwise look for a missing `psycopg` driver and fail even with correct credentials). Quotes/spaces/`@` in the password are handled.
* `python start.py check` explains failures in plain English:
  | Message | Meaning / fix |
  |---|---|
  | *not reachable at localhost:5432* | PostgreSQL service not running (Windows → Services → `postgresql-x64-NN` → Start) or wrong port (pgAdmin → server → Properties → Connection) |
  | *Wrong password for user* | fix the password in `.env` |
  | *role does not exist* | wrong user name |
  | no PostgreSQL at all | set `DATABASE_URL=sqlite:///data/quant.db` – everything still works |

## 📥 How the data pull works now (fast 5 years first, then daily)
`python start.py setup` / `daily` both call the same smart sync (`ingestion/sync.py`):
| Source | First run | Every later run |
|---|---|---|
| Stock prices (≈50 NSE) + NIFTY + India VIX (yfinance) | **one batched, multi-threaded call → full 5 years** (≈ 30–60 s) | only the last ~7 days are re-pulled and upserted (no duplicates) |
| Corporate actions | dividends/splits stored | a **new dividend/split triggers an automatic full re-pull of that stock**, because Yahoo re-scales all historical adjusted prices |
| FII/DII participant OI | your CSV is loaded first, NSE files are fetched **in parallel (8 threads)** only for days missing after it | only days after the last stored date |
| News (Google News → sentiment) | last 90 days, parallel | last 3 days |
| Fundamentals | snapshot | refreshed weekly |

Safe to re-run any time; a failed ticker is skipped and reported. Run the daily job **after market close** (a mid-day run stores a partial bar that the next run overwrites).
Automate with Windows Task Scheduler → `scripts\daily_job.bat` at ~19:30 IST (Mon–Fri).

## 🛠 Troubleshooting
| Problem | Fix |
|---|---|
| `pyarrow requires NumPy 2.0 or newer` (dashboard tables) | `pip install "pyarrow>=15,<21"` |
| `No module named 'psycopg'` | fixed automatically (driver forced to psycopg2) – make sure you use this version of the code |
| `UnicodeDecodeError ... utf-8 codec` on Windows | this is psycopg2 hiding a real connection error (wrong password/DB). `python start.py check` shows the real message |
| Yahoo returns nothing / rate limit | re-run `python start.py daily` later; `pip install -U yfinance` |
| NSE FII/DII download blocked | re-run later; the CSV history you provided is already loaded |
| FinBERT | optional: `pip install transformers torch`, set `USE_FINBERT=1` in `.env` (downloads ~440 MB the first time). Otherwise a simple lexicon scorer is used |
| Anything else | copy the red ❌ lines from the terminal and send them to me |

## What is inside (unchanged design, details in the code)
| Stage | Code |
|---|---|
| 1 Ingestion + PostgreSQL (20 tables) | `db/`, `ingestion/` |
| 2 Data quality + point-in-time features | `quality/`, `features/` — FII lagged 1 trading day, news after 15:30 IST → next day, HMM regime *filtered* (no look-ahead), automatic leakage replay test |
| 3A Alpha | `models/alpha.py` — LightGBM + XGBoost, 80+ factors, walk-forward with embargo, calibration, q10/q50/q90 forecasts |
| 3B Risk | `risk/` — VaR/ES, GJR-GARCH, drawdown, 0–100 risk score |
| 4 Portfolio/research | `backtest/`, `portfolio/` — costs, slippage, impact, Monte Carlo, Max-Sharpe/Min-Var/Risk-Parity, vol-target sizing, stress tests |
| 5 Explainability/app/monitoring | `models/explain.py` (SHAP), `api/` (FastAPI), `app/` (Streamlit), `monitoring/` (PSI drift, realised IC) |

### Findings about your files / old code (all handled)
1. **`Future Index Long` is empty in your FII file from 2021 onward (98.5% of rows)** — rebuilt as `Total Long − all other long columns` and flagged.
2. Old pipeline leakage (other stocks' `future_*` columns, `nifty_future_ret`, `bfill`, shuffled CV) → replaced by whitelist + walk-forward + automatic point-in-time replay test.
3. Old iterative "5-day forecast" reused a 5-day return as a 1-day step → replaced by direct q10/q90 forecasts.
4. News goes into the ML model only after ≥250 days of history (you have ~90); until then it acts as a BUY-blocking overlay.

### Be realistic
* Expect AUC ≈ 0.51–0.56 and IC ≈ 0.01–0.05 on real data; judge by IC, quintile spread and the net-of-cost backtest, not accuracy.
* Universe = today's large caps (survivorship bias); fundamentals are a snapshot (display only). Educational software, **not investment advice**.

### Resume bullets that match this code
* Built a PostgreSQL-backed quant platform for ~50 NSE stocks with automated 5-year backfill + incremental daily ingestion of prices, NIFTY/VIX, NSE participant-wise FII/DII OI and news, with data-quality checks and corporate-action handling.
* Engineered point-in-time features with an automated look-ahead replay test; trained LightGBM/XGBoost alpha and downside-risk models with expanding walk-forward validation, calibration and quantile forecasts (IC/ICIR evaluation).
* Implemented VaR/ES, GJR-GARCH, drawdown risk scoring, a cost-aware walk-forward backtester, Monte-Carlo and stress testing, portfolio optimisation and volatility-target sizing.
* Added SHAP explanations, PSI drift/performance monitoring, FastAPI + Streamlit dashboards. *(Mention FinBERT only if you installed transformers/torch.)*

## Tests (optional)
`python tests/smoke_test.py` (whole pipeline on synthetic data, no internet/PostgreSQL), `python tests/test_sync.py` (5-year-then-daily logic with mocked Yahoo), `python tests/test_parsers.py`.
