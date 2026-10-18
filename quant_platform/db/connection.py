"""Engine, upsert and read helpers (PostgreSQL + SQLite)."""
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
import config
from db.schema import metadata

log = logging.getLogger("db")
_engine = None


def _make(url: str) -> Engine:
    url = config.normalize_db_url(url)
    if url.startswith("postgresql"):
        return create_engine(url, pool_pre_ping=True, future=True,
                             connect_args={"client_encoding": "utf8", "connect_timeout": 10})
    return create_engine(url, pool_pre_ping=True, future=True)


def get_engine(url: str = None) -> Engine:
    global _engine
    if url:
        return _make(url)
    if _engine is None:
        _engine = _make(config.DATABASE_URL)
    return _engine


def set_engine_url(url: str):
    global _engine
    _engine = _make(url)
    return _engine


def init_db(engine: Engine = None):
    if engine is None:
        from db.doctor import ensure_database
        ensure_database()                      # creates the PostgreSQL database itself if it does not exist
    engine = engine or get_engine()
    metadata.create_all(engine)
    return [t for t in metadata.tables]


def get_state(key: str, default=None):
    try:
        d = read_sql("SELECT value FROM sync_state WHERE key = :k", {"k": key})
        return d["value"].iloc[0] if len(d) else default
    except Exception:
        return default


def get_states(prefix: str) -> dict:
    try:
        d = read_sql("SELECT key, value FROM sync_state WHERE key LIKE :p", {"p": prefix + "%"})
        return {k[len(prefix):]: v for k, v in zip(d.key, d.value)}
    except Exception:
        return {}


def set_state(key: str, value):
    upsert_df(pd.DataFrame([{"key": key, "value": str(value)}]), "sync_state")


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, (np.floating,)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def upsert_df(df: pd.DataFrame, table_name: str, engine: Engine = None, update: bool = True,
              chunk: int = 5000) -> int:
    """Insert rows; on primary-key / unique conflict update (or ignore). Returns rows sent."""
    if df is None or len(df) == 0:
        return 0
    engine = engine or get_engine()
    table = metadata.tables[table_name]
    cols = [c.name for c in table.columns if c.name in df.columns]
    pk = [c.name for c in table.primary_key.columns]
    df = df[cols].copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = pd.to_datetime(df[c])
    records = [{k: _clean_value(v) for k, v in row.items()} for row in df.to_dict("records")]
    # date columns -> python date
    for r in records:
        for k, v in r.items():
            if hasattr(v, "date") and table.columns[k].type.__class__.__name__ == "Date":
                r[k] = v.date()
    dialect = engine.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    # conflict target: PK, except news (uses unique constraint)
    conflict = pk
    if table_name == "news":
        conflict = ["entity", "title", "published_at"]
    n = 0
    with engine.begin() as conn:
        for i in range(0, len(records), chunk):
            part = records[i:i + chunk]
            stmt = insert(table).values(part)
            upd_cols = [c for c in cols if c not in conflict and c not in pk]
            if update and upd_cols:
                stmt = stmt.on_conflict_do_update(index_elements=conflict,
                                                  set_={c: stmt.excluded[c] for c in upd_cols})
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=conflict)
            conn.execute(stmt)
            n += len(part)
    return n


def read_sql(query: str, params: dict = None, engine: Engine = None, parse_dates=None) -> pd.DataFrame:
    engine = engine or get_engine()
    params = {k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in (params or {}).items()}
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params, parse_dates=parse_dates)


def max_date(table: str, col: str = "date", where: str = "", engine: Engine = None):
    q = f"SELECT MAX({col}) AS m FROM {table} {where}"
    try:
        m = read_sql(q, engine=engine)["m"].iloc[0]
    except Exception:
        return None
    return None if m is None or pd.isna(m) else pd.Timestamp(m)


def log_ingestion(source: str, rows: int, status: str, message: str = "", engine: Engine = None):
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO ingestion_log(source, rows, status, message) VALUES (:s,:r,:st,:m)"),
                     {"s": source, "r": int(rows), "st": status, "m": message[:500]})


def export_schema_sql(path):
    """Write CREATE TABLE statements for PostgreSQL to a .sql file."""
    from sqlalchemy.schema import CreateTable, CreateIndex
    from sqlalchemy.dialects import postgresql
    out = ["-- Auto-generated PostgreSQL schema. Created by db/init_db.py (python run_pipeline.py init-db)\n"]
    for t in metadata.sorted_tables:
        out.append(str(CreateTable(t, if_not_exists=True).compile(dialect=postgresql.dialect())).strip() + ";\n")
        for ix in t.indexes:
            out.append(str(CreateIndex(ix, if_not_exists=True).compile(dialect=postgresql.dialect())).strip() + ";\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
