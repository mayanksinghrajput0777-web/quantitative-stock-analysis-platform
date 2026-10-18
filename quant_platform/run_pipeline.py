"""One entry point for the whole platform.   python run_pipeline.py <command> [options]

  init-db            create PostgreSQL tables (also writes db/schema.sql)
  load-csv           load YOUR csv files from data/raw/ (FII/DII, nifty, vix, stocks, news)
  fetch-yf           download 5y of stock prices + NIFTY + India VIX from yfinance
  fetch-fii          download missing FII/DII days from NSE archives
  fetch-news         Google News + FinBERT (last 90 days)
  fetch-fundamentals fundamentals snapshot (display only; not a historical feature)
  quality            data-quality report
  eda                EDA html report
  features           build feature panel + point-in-time leakage test
  train              walk-forward training (LightGBM+XGBoost), calibration, SHAP, saves model
  predict            latest predictions + risk + signals
  risk | signals     (re)compute only that step
  backtest           walk-forward backtest, Monte Carlo, stress tests
  portfolio          optimised portfolio from today's BUY signals
  monitor            drift + live performance
  daily              incremental data pull -> predict -> risk -> signals -> portfolio -> monitor
  bootstrap          EVERYTHING, in order (first run)
"""
import sys, argparse, logging, warnings
warnings.filterwarnings("ignore")
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(config.ART_DIR / "pipeline.log", encoding="utf-8")])
for noisy in ("yfinance", "peewee", "urllib3", "matplotlib", "numexpr"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


def step(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def cmd_init_db(a):
    from db.init_db import main; main()

def cmd_load_csv(a):
    from ingestion.load_csv_history import load_all_csv; load_all_csv()

def cmd_fetch_yf(a):
    from ingestion import yf_prices
    if a.years: config.HISTORY_YEARS = a.years
    print(yf_prices.fetch_stock_prices(incremental=not a.full)); print(yf_prices.fetch_nifty_vix(incremental=not a.full))

def cmd_fetch_fii(a):
    from ingestion import fii_dii; print(fii_dii.fetch_incremental(), "rows")

def cmd_fetch_news(a):
    from ingestion import news; print(news.fetch_news(days_back=a.days, max_stocks=a.max_stocks), "new headlines")

def cmd_fetch_fund(a):
    from ingestion import yf_prices; print(yf_prices.fetch_fundamentals(), "rows")

def cmd_quality(a):
    from quality.checks import run_quality_checks, print_report; print_report(run_quality_checks())

def cmd_eda(a):
    from eda.eda_report import run_eda; run_eda()

def cmd_features(a):
    from features.build import build_panel, feature_columns
    from quality.leakage import audit_feature_names, pit_replay_test
    p, info, _ = build_panel(save=True); print(info); audit_feature_names(feature_columns(p))
    print("PIT replay:", pit_replay_test(p))

def cmd_train(a):
    from models.pipeline import run_train; run_train(fast=a.fast)

def cmd_predict(a):
    from models.pipeline import run_predict_risk_signals; run_predict_risk_signals()

def cmd_risk(a):
    from risk.engine import run_risk; run_risk()

def cmd_signals(a):
    from models.pipeline import run_signals; run_signals()

def cmd_backtest(a):
    from backtest.runner import run_all; run_all()

def cmd_portfolio(a):
    from portfolio.runner import build_portfolio; build_portfolio()

def cmd_monitor(a):
    from monitoring.drift import run_monitoring; run_monitoring()


def run_step(results, name, fn, critical=False):
    import traceback
    step(name)
    try:
        fn(); results.append((name, "OK"))
        return True
    except SystemExit as e:
        results.append((name, "FAILED")); print(f"❌ {e}")
        if critical: raise
        return False
    except Exception as e:
        results.append((name, f"FAILED: {type(e).__name__}: {str(e)[:120]}"))
        print(f"❌ {name} failed -> {type(e).__name__}: {e}")
        logging.getLogger("pipeline").debug(traceback.format_exc())
        if critical:
            traceback.print_exc()
            print("\n⛔ This step is required, so the run stops here. Fix the error above (or send it to me) and re-run the same command.")
            raise SystemExit(1)
        print("   (non-critical: continuing with the next step)")
        return False


def finish_summary(results):
    import pandas as pd
    print("\n" + "=" * 60 + "\n RESULT CHECKLIST\n" + "=" * 60)
    for n, st in results:
        print(f" {'✅' if st == 'OK' else '⚠️ '} {n}: {st}")
    f = config.OUT_DIR / "final_predictions_next5.csv"
    if f.exists():
        t = pd.read_csv(f)
        print(f"\n📄 YOUR MAIN OUTPUT -> {f}\n")
        print(t.sort_values("Confidence", ascending=False).head(12).to_string(index=False))
    print(f"\n📁 All results are in: {config.OUT_DIR}")
    print("   final_predictions_next5.csv · backtest_comparison.csv · stress_tests.csv · monte_carlo.json · eda/eda_report.html · data_quality.json")
    print("🖥️  Dashboard:  python start.py dashboard      (or double-click 3_OPEN_DASHBOARD.bat)")


def cmd_daily(a):
    from ingestion.sync import run_sync
    from models import alpha
    from models.pipeline import run_train, run_predict_risk_signals
    from quality.checks import run_quality_checks
    from portfolio.runner import build_portfolio
    from monitoring.drift import run_monitoring
    import datetime as dt
    R = []
    run_step(R, "1. DB check", lambda: __import__("db.connection", fromlist=["init_db"]).init_db(), critical=True)
    run_step(R, "2. incremental data pull (new days only)", lambda: run_sync(with_news=not a.skip_news, news_days=3), critical=True)
    run_step(R, "3. data quality", lambda: run_quality_checks())
    b = alpha.load_bundle()
    age = (dt.datetime.utcnow() - dt.datetime.fromisoformat(b["trained_at"])).days if b else 999
    if a.retrain or age > a.retrain_days:
        run_step(R, f"4a. retrain model (age {age}d)", lambda: run_train(fast=a.fast), critical=True)
    run_step(R, "4. predict + risk + signals", run_predict_risk_signals, critical=True)
    run_step(R, "5. portfolio", build_portfolio)
    run_step(R, "6. monitoring", run_monitoring)
    finish_summary(R)


def cmd_check(a):
    from db.doctor import full_check
    ok = full_check()
    print("\n" + ("✅ Everything needed is OK. Next:  python start.py setup" if ok else "❌ Fix the items marked ❌ above, then run:  python start.py check"))


def cmd_bootstrap(a):
    import ingestion.load_csv_history as lc
    from ingestion.sync import run_sync
    from db.connection import init_db
    from db.doctor import ensure_database
    R = []
    run_step(R, "1. database connection + tables", lambda: (ensure_database(), init_db(), print("✅ database ready")), critical=True)
    if a.synthetic:
        from tools.make_synthetic_data import make
        run_step(R, "2. SYNTHETIC data (testing only)", lambda: make(a.synthetic))
        lc_ok = run_step(R, "2b. load FII csv", lambda: lc.load_all_csv())
    else:
        run_step(R, "2. load your CSV files from data/raw (FII/DII etc.)", lambda: lc.load_all_csv())
        run_step(R, "3. pull data: 5y history (fast) + FII/DII + news + fundamentals",
                 lambda: run_sync(with_news=not a.skip_news, news_days=config.NEWS_DAYS, with_fundamentals=not a.skip_fundamentals), critical=True)
    from db.connection import read_sql
    if int(read_sql("SELECT COUNT(*) n FROM prices_daily")["n"].iloc[0]) == 0:
        raise SystemExit("⛔ No price data in the database. Check your internet connection (yfinance) or put stock CSVs in data/raw/ and re-run.")
    run_step(R, "4. data quality report", lambda: cmd_quality(a))
    run_step(R, "5. EDA report", lambda: cmd_eda(a))
    run_step(R, "6. train models (walk-forward)", lambda: cmd_train(a), critical=True)
    run_step(R, "7. predictions + risk + signals", lambda: cmd_predict(a), critical=True)
    run_step(R, "8. backtest / Monte Carlo / stress", lambda: cmd_backtest(a))
    run_step(R, "9. portfolio optimisation", lambda: cmd_portfolio(a))
    run_step(R, "10. model monitoring", lambda: cmd_monitor(a))
    finish_summary(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **kw):
        p = sp.add_parser(name); p.set_defaults(fn=fn); return p
    add("init-db", cmd_init_db); add("check", cmd_check); add("load-csv", cmd_load_csv)
    p = add("fetch-yf", cmd_fetch_yf); p.add_argument("--years", type=int); p.add_argument("--full", action="store_true", help="ignore incremental")
    add("fetch-fii", cmd_fetch_fii)
    p = add("fetch-news", cmd_fetch_news); p.add_argument("--days", type=int, default=config.NEWS_DAYS); p.add_argument("--max-stocks", type=int)
    add("fetch-fundamentals", cmd_fetch_fund); add("quality", cmd_quality); add("eda", cmd_eda); add("features", cmd_features)
    p = add("train", cmd_train); p.add_argument("--fast", action="store_true", help="3 folds, fewer trees (quick test)")
    add("predict", cmd_predict); add("risk", cmd_risk); add("signals", cmd_signals); add("backtest", cmd_backtest)
    add("portfolio", cmd_portfolio); add("monitor", cmd_monitor)
    p = add("daily", cmd_daily); p.add_argument("--retrain", action="store_true"); p.add_argument("--retrain-days", type=int, default=7)
    p.add_argument("--fast", action="store_true"); p.add_argument("--skip-news", action="store_true")
    p = add("bootstrap", cmd_bootstrap); p.add_argument("--fast", action="store_true"); p.add_argument("--skip-news", action="store_true")
    p.add_argument("--skip-fundamentals", action="store_true"); p.add_argument("--synthetic", type=int, metavar="N_STOCKS",
                                                                               help="use N synthetic stocks instead of downloading (offline test)")
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
