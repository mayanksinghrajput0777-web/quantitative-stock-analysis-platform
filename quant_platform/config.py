"""Central configuration. Edit values here or override DATABASE_URL etc. in .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

import re
import sys
from urllib.parse import quote_plus, unquote

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)       # .env always wins over leftover terminal variables
os.environ.setdefault("PGCLIENTENCODING", "UTF8")   # avoids the Windows 'utf-8 codec can't decode' psycopg2 crash
for _s in (sys.stdout, sys.stderr):                  # emoji-safe console on Windows
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_db_url(url: str) -> str:
    """Accepts postgresql:// , postgres:// , quotes, spaces, '@' in passwords, relative sqlite paths.
    Always returns a URL using the psycopg2 driver (SQLAlchemy 2.1 would otherwise pick 'psycopg' v3)."""
    url = (url or "").strip().strip('"').strip("'").strip()
    if not url:
        return "sqlite:///" + (ROOT / "data" / "quant.db").as_posix()
    if url.startswith("sqlite"):
        path = url.split("sqlite:///", 1)[-1]
        if path and not Path(path).is_absolute() and ":memory:" not in path:
            url = "sqlite:///" + (ROOT / path).as_posix()
        return url
    m = re.match(r"^(?:postgres|postgresql)(?:\+\w+)?://(?P<user>[^:/@]+)(?::(?P<pw>.*))?@(?P<host>[^@/:]+)"
                 r"(?::(?P<port>\d+))?/(?P<db>[^?\s]+)(?P<q>\?.*)?$", url)
    if not m:
        return url
    pw = quote_plus(unquote(m.group("pw") or ""))
    port = m.group("port") or "5432"
    return f"postgresql+psycopg2://{m.group('user')}:{pw}@{m.group('host')}:{port}/{m.group('db')}{m.group('q') or ''}"


DATABASE_URL = normalize_db_url(os.getenv("QUANT_DATABASE_URL") or os.getenv("DATABASE_URL", ""))   # QUANT_DATABASE_URL = test override
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # drop your CSV files here
ART_DIR = DATA_DIR / "artifacts"      # models, cache, logs
MODEL_DIR = ART_DIR / "models"
OUT_DIR = ROOT / "output"             # <-- YOUR RESULTS END UP HERE
REPORT_DIR = OUT_DIR
EDA_DIR = OUT_DIR / "eda"
for _d in (RAW_DIR, ART_DIR, MODEL_DIR, OUT_DIR, EDA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "5"))
NEWS_DAYS = int(os.getenv("NEWS_DAYS", "90"))
USE_FINBERT = os.getenv("USE_FINBERT", "1") == "1"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Symbols for yfinance
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"
NIFTY_NAME = "NIFTY50"
VIX_NAME = "INDIAVIX"

# ---- labels ----
ALPHA_H = 5            # forward horizon (trading days) for alpha target
RISK_H = 3             # horizon for downside event
RISK_TH = -0.03        # downside event = fwd RISK_H-day return <= -3% (your old -5% is too rare for large caps)

# ---- point-in-time rules ----
FII_LAG_DAYS = 1       # participant OI is published after market close -> use from next trading day
NEWS_CUTOFF_HOUR_IST = 15.5   # news after 15:30 IST belongs to the next trading day
NEWS_MIN_DAYS_FOR_MODEL = 250  # sentiment goes into the ML model only after this many days of coverage

# ---- model / walk-forward ----
WF_MIN_TRAIN_DAYS = 400
WF_TEST_DAYS = 63
WF_STEP_DAYS = 63
EMBARGO_DAYS = max(ALPHA_H, RISK_H) + 2
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
                  random_state=SEED, n_jobs=-1)
QUANTILES = [0.10, 0.50, 0.90]

# ---- signals (your original thresholds, adapted) ----
BUY_TOP_PCT = 0.80          # score percentile for BUY
SELL_BOTTOM_PCT = 0.20
MIN_PROB_UP_BUY = 0.52
RISK_BLOCK_PROB = 0.35      # block BUY when calibrated prob_risk above this
NEWS_NEG_BLOCK = -0.35      # news overlay: block BUY if recent stock sentiment below this

# ---- risk score (absolute scaling: value at 0 -> value at 100) ----
RISK_SCALES = {
    "garch_vol_ann": (0.15, 0.60),
    "es95_1d": (0.010, 0.060),
    "cur_drawdown": (0.0, 0.30),
    "downside_dev_ann": (0.10, 0.45),
    "prob_risk": (0.0, 0.30),
}
RISK_WEIGHTS = {"garch_vol_ann": 0.30, "es95_1d": 0.20, "cur_drawdown": 0.15,
                "downside_dev_ann": 0.10, "prob_risk": 0.25}
VIX_SCALE = (12.0, 35.0)
VIX_BLEND = 0.15
RISK_LEVELS = [(25, "Low"), (50, "Moderate"), (75, "High"), (101, "Extreme")]

# ---- backtest ----
BT_TOP_K = 10
BT_REBALANCE_DAYS = 5
BT_EXEC_LAG = 1              # signal at close t -> trade at close t+1
COST_BPS = 10.0              # brokerage + STT + exchange + stamp (one-way)
SLIPPAGE_BPS = 5.0
IMPACT_COEF_BPS = 30.0       # sqrt participation impact
CAPITAL = 1_000_000.0        # INR
RF_ANNUAL = 0.06
MAX_WEIGHT = 0.20
TARGET_VOL = 0.15

# ---- monitoring ----
PSI_WARN, PSI_ALERT = 0.10, 0.25
