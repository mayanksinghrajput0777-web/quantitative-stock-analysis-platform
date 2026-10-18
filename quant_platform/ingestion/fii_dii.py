"""NSE participant-wise F&O open interest (FII / DII / Pro / Client).
- parse_participant_oi(): robust parser for your concatenated CSV AND for freshly downloaded files
- IMPORTANT: in your file 'Future Index Long' is EMPTY for every row from 2021-01-01 onward.
  It is rebuilt exactly as  Total Long - (all other long columns)  and flagged in fut_idx_long_reconstructed.
"""
import io, time, logging
import numpy as np
import pandas as pd
import requests
import config
from db.connection import upsert_df, max_date, log_ingestion

log = logging.getLogger("fii")

COLMAP = {
    "future index long": "fut_idx_long", "future index short": "fut_idx_short",
    "future stock long": "fut_stk_long", "future stock short": "fut_stk_short",
    "option index call long": "opt_idx_call_long", "option index put long": "opt_idx_put_long",
    "option index call short": "opt_idx_call_short", "option index put short": "opt_idx_put_short",
    "option stock call long": "opt_stk_call_long", "option stock put long": "opt_stk_put_long",
    "option stock call short": "opt_stk_call_short", "option stock put short": "opt_stk_put_short",
    "total long contracts": "total_long", "total short contracts": "total_short",
}
NUM = list(COLMAP.values())
LONG_PARTS = ["fut_stk_long", "opt_idx_call_long", "opt_idx_put_long", "opt_stk_call_long", "opt_stk_put_long"]
CLIENTS = {"client": "Client", "dii": "DII", "fii": "FII", "pro": "Pro", "total": "TOTAL"}
URL = "https://archives.nseindia.com/content/nsccl/fao_participant_oi_{d}.csv"


def _norm(c) -> str:
    return " ".join(str(c).replace("\t", " ").strip().lower().split())


def parse_participant_oi(raw: pd.DataFrame, date_hint=None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).replace("\t", " ").strip() for c in df.columns]
    cols_norm = {c: _norm(c) for c in df.columns}
    ct_col = next((c for c, n in cols_norm.items() if n == "client type"), df.columns[0])
    # date column: Report_Date if present, else last column whose values look like dates
    date_col = None
    if "Report_Date" in df.columns:
        date_col = "Report_Date"
    else:
        for c in reversed(list(df.columns)):
            if cols_norm[c] in COLMAP or c == ct_col:
                continue
            probe = pd.to_datetime(df[c].dropna().astype(str).head(20), format="%d-%m-%Y", errors="coerce")
            if len(probe) and probe.notna().mean() > 0.8:
                date_col = c; break
    # drop repeated header rows and unknown participants
    df = df[df[ct_col].astype(str).str.strip().str.lower().isin(CLIENTS.keys())].copy()
    df["client_type"] = df[ct_col].astype(str).str.strip().str.lower().map(CLIENTS)
    if date_col is not None:
        s = df[date_col].astype(str).str.strip()
        d = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
        d = d.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
        df["date"] = d
    elif date_hint is not None:
        df["date"] = pd.Timestamp(date_hint)
    else:
        raise ValueError("Cannot locate date column in participant OI file")
    out = pd.DataFrame({"date": df["date"], "client_type": df["client_type"]})
    for c, n in cols_norm.items():
        if n in COLMAP:
            out[COLMAP[n]] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    for c in NUM:
        if c not in out.columns:
            out[c] = np.nan
    out = out.dropna(subset=["date"]).drop_duplicates(["date", "client_type"], keep="last")
    # --- repair empty Future Index Long ---
    rec = out["total_long"] - out[LONG_PARTS].sum(axis=1, min_count=len(LONG_PARTS))
    need = out["fut_idx_long"].isna() & rec.notna() & (rec >= 0)
    out["fut_idx_long_reconstructed"] = need
    out.loc[need, "fut_idx_long"] = rec[need]
    return out.sort_values(["date", "client_type"]).reset_index(drop=True)[
        ["date", "client_type"] + NUM + ["fut_idx_long_reconstructed"]]


def load_participant_csv(path) -> int:
    raw = pd.read_csv(path)
    df = parse_participant_oi(raw)
    n = upsert_df(df, "participant_oi")
    log_ingestion("participant_oi_csv", n, "ok", f"{df['date'].min().date()}..{df['date'].max().date()}")
    return n


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                      "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream",
                      "Referer": "https://www.nseindia.com"})
    return s


def download_day(d: pd.Timestamp, session=None, retries=2):
    """One NSE file. Returns None for holidays / not yet published (HTTP 404)."""
    session = session or _session()
    last = None
    for k in range(retries + 1):
        try:
            r = session.get(URL.format(d=d.strftime("%d%m%Y")), timeout=20)
        except Exception as e:
            last = e; time.sleep(1.5 * (k + 1)); continue
        if r.status_code == 404:
            return None
        if r.status_code != 200 or "Client Type" not in r.text:
            last = RuntimeError(f"HTTP {r.status_code}"); time.sleep(1.5 * (k + 1)); continue
        lines = r.text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("client type"))
        raw = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        return parse_participant_oi(raw, date_hint=d)
    if last is not None:
        log.warning("%s failed: %s", d.date(), last)
    return None


def fetch_range(start, end=None, workers=8) -> int:
    """Download missing days from NSE archives in PARALLEL (weekdays only; holidays 404 and are skipped)."""
    from concurrent.futures import ThreadPoolExecutor
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    days = list(pd.bdate_range(pd.Timestamp(start), end))
    if not days:
        return 0
    frames, ok = [], 0
    def job(d):
        return download_day(d, _session())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(job, days), 1):
            if df is not None:
                frames.append(df); ok += 1
            if len(days) > 60 and i % 100 == 0:
                print(f"   FII/DII files {i}/{len(days)}")
    n = upsert_df(pd.concat(frames), "participant_oi") if frames else 0
    log_ingestion("participant_oi_nse", n, "ok" if frames else "warn", f"{ok} files of {len(days)} weekdays (rest = holidays/not published)")
    return n


def fetch_incremental(years=None) -> int:
    m = max_date("participant_oi")
    if m is not None:
        start = m + pd.Timedelta(days=1)
    else:
        start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years or config.HISTORY_YEARS))
    if start > pd.Timestamp.today().normalize():
        return 0
    return fetch_range(start)
