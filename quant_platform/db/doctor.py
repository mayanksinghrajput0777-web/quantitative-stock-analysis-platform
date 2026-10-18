"""Database doctor: tests the connection, explains problems in plain English, creates the database if missing."""
import re
import config
from sqlalchemy.engine import make_url


def _hint(msg: str, url) -> str:
    m = msg.lower()
    if "does not exist" in m and "database" in m:
        return f"Database '{url.database}' does not exist (the program will create it automatically)."
    if "password authentication failed" in m or "authentication failed" in m or "no password supplied" in m:
        return (f"Wrong password for user '{url.username}'. Fix the password in .env (DATABASE_URL). "
                "If you forgot it: open pgAdmin -> right-click the server -> Properties/Connection, or reset it with "
                "ALTER USER postgres PASSWORD 'newpass';")
    if "role" in m and "does not exist" in m:
        return f"User '{url.username}' does not exist in PostgreSQL. Use 'postgres' (default) or create the role."
    if "refused" in m or "could not connect" in m or "timeout" in m or "10061" in m or "connection timed out" in m:
        return (f"PostgreSQL is not reachable at {url.host}:{url.port}. Check: (1) the server is running - Windows key -> "
                "'Services' -> 'postgresql-x64-NN' = Running; (2) the port - pgAdmin -> server -> Properties -> Connection shows "
                "it (often 5432, sometimes 5433); (3) no firewall blocking it.")
    if "pg_hba" in m:
        return "pg_hba.conf does not allow this connection (edit it to allow 127.0.0.1 with scram-sha-256/md5) and restart the service."
    return "See the raw message above."


def _connect(url, dbname):
    import psycopg2
    kw = dict(host=url.host or "localhost", port=url.port or 5432, user=url.username, password=url.password,
              dbname=dbname, connect_timeout=8)
    try:
        return psycopg2.connect(client_encoding="utf8", **kw)
    except UnicodeDecodeError:
        # Windows non-English message bytes: retry in latin1 just to READ the real error text
        try:
            return psycopg2.connect(client_encoding="latin1", **kw)
        except Exception as e2:
            raise RuntimeError(str(e2)) from None


def check_connection(create: bool = True, verbose: bool = True):
    """Returns (ok, message)."""
    url = make_url(config.DATABASE_URL)
    say = print if verbose else (lambda *a, **k: None)
    if url.get_backend_name() == "sqlite":
        say(f"ℹ️  Using SQLite file: {url.database}  (set DATABASE_URL to a postgresql URL in .env to use PostgreSQL)")
        return True, "sqlite"
    safe = f"{url.username}@{url.host}:{url.port or 5432}/{url.database}"
    say(f"🔌 Connecting to PostgreSQL  {safe}")
    try:
        c = _connect(url, url.database); c.close()
        say("✅ PostgreSQL connection OK")
        return True, "ok"
    except Exception as e:
        err = str(e).strip()
        if "does not exist" in err.lower() and "database" in err.lower() and create:
            try:
                c = _connect(url, "postgres"); c.autocommit = True
                cur = c.cursor(); cur.execute(f'CREATE DATABASE "{url.database}"'); c.close()
                say(f"✅ Database '{url.database}' did not exist - created it.")
                return True, "created"
            except Exception as e2:
                err = f"could not create database: {str(e2).strip()}"
        say(f"❌ Cannot connect: {err.splitlines()[0] if err else err}")
        say(f"👉 {_hint(err, url)}")
        say("   (Quick alternative without PostgreSQL: set  DATABASE_URL=sqlite:///data/quant.db  in the .env file)")
        return False, err


def ensure_database():
    ok, msg = check_connection(create=True, verbose=False)
    if not ok:
        check_connection(create=True, verbose=True)
        raise SystemExit("\nStopped: database connection failed (see the hint above).")
    return True


def full_check():
    """python start.py check  -> Python, packages, database, data in the tables."""
    import sys, importlib
    print(f"🐍 Python {sys.version.split()[0]}  ({sys.executable})")
    need = ["pandas", "numpy", "scipy", "sklearn", "lightgbm", "xgboost", "arch", "hmmlearn", "shap", "sqlalchemy", "psycopg2",
            "dotenv", "requests", "yfinance", "pyarrow", "matplotlib", "plotly", "fastapi", "uvicorn", "streamlit", "joblib"]
    optional = {"xgboost", "arch", "hmmlearn", "shap", "transformers", "torch"}
    bad = []
    for m in need + ["transformers", "torch"]:
        try:
            mod = importlib.import_module(m); v = getattr(mod, "__version__", "")
            print(f"   ✅ {m} {v}")
        except Exception as e:
            tag = "optional - a fallback is used" if m in optional else "REQUIRED  ->  pip install -r requirements.txt"
            if m == "pyarrow" and "numpy" in str(e).lower():
                tag = 'REQUIRED - too new for your NumPy  ->  pip install "pyarrow>=15,<21"'
            print(f"   {'⚠️ ' if m in optional else '❌'} {m} missing ({tag})")
            if m not in optional: bad.append(m)
    ok, _ = check_connection(create=True, verbose=True)
    if ok:
        from db.connection import init_db, read_sql
        init_db()
        for t, c in [("prices_daily", "date"), ("index_prices", "date"), ("participant_oi", "date"), ("news", "published_at"),
                     ("predictions", "asof_date"), ("risk_metrics", "asof_date"), ("model_runs", "trained_at")]:
            try:
                d = read_sql(f"SELECT COUNT(*) n, MIN({c}) a, MAX({c}) b FROM {t}").iloc[0]
                print(f"   {t:15s} {int(d.n):>8,} rows   {str(d.a)[:10]} → {str(d.b)[:10]}")
            except Exception as e:
                print(f"   {t:15s} error {e}")
    if bad:
        print("\n❌ Install missing packages:  pip install -r requirements.txt")
    return ok and not bad
