"""Model monitoring: feature drift (PSI), live/OOS predictive performance (IC, AUC, hit rate, Brier), model age."""
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
import config
from db.connection import read_sql, get_engine, upsert_df
from features.build import build_panel


def psi(expected, actual, bins=10) -> float:
    e, a = pd.Series(expected).dropna().values, pd.Series(actual).dropna().values
    if len(e) < 50 or len(a) < 20:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e); pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def status_from_psi(v):
    return "alert" if v >= config.PSI_ALERT else "warn" if v >= config.PSI_WARN else "ok"


def run_monitoring(engine=None, recent_days=60, top_n=25) -> dict:
    engine = engine or get_engine()
    panel, info, _ = build_panel(engine=engine)
    last = panel.date.max()
    fi = read_sql("SELECT feature, mean_abs_shap FROM feature_importance WHERE run_id=(SELECT run_id FROM feature_importance ORDER BY run_id DESC LIMIT 1) ORDER BY mean_abs_shap DESC LIMIT :n",
                  {"n": top_n}, engine)
    feats = list(fi.feature) if len(fi) else [c for c in panel.columns if c.startswith(("vol_", "ret_", "vix"))][:top_n]
    dates = np.sort(panel.date.unique())
    cut = dates[-recent_days]
    ref, rec = panel[panel.date < cut], panel[panel.date >= cut]
    rows = []
    for f in feats:
        if f in panel.columns:
            v = psi(ref[f], rec[f]); rows.append({"asof_date": last, "feature": f, "psi": v, "status": status_from_psi(v) if v == v else "n/a"})
    drift = pd.DataFrame(rows)
    upsert_df(drift, "drift_metrics", engine=engine)
    # ---- realised performance of stored predictions (walk-forward + live, once outcomes are known) ----
    pred = read_sql("SELECT symbol, asof_date AS date, source, prob_up, exp_excess_ret FROM predictions", engine=engine, parse_dates=["date"])
    tgt = panel[["date", "symbol", "y_up", f"fwd_excess_{config.ALPHA_H}d"]].dropna()
    m = pred.merge(tgt, on=["date", "symbol"], how="inner")
    perf = []
    if len(m):
        end = m.date.max(); ud = np.sort(m.date.unique())
        for w in (20, 60, 120, 250):
            if len(ud) < max(10, w // 2):
                continue
            sub = m[m.date >= ud[-min(w, len(ud))]]
            ics = sub.groupby("date").apply(lambda g: spearmanr(g.prob_up, g[f"fwd_excess_{config.ALPHA_H}d"])[0] if g.prob_up.nunique() > 2 and len(g) > 5 else np.nan)
            ic = float(ics.mean())
            auc = float(roc_auc_score(sub.y_up, sub.prob_up)) if sub.y_up.nunique() > 1 else np.nan
            hit = float(((sub.exp_excess_ret > 0) == (sub[f"fwd_excess_{config.ALPHA_H}d"] > 0)).mean())
            br = float(brier_score_loss(sub.y_up, sub.prob_up.clip(0, 1)))
            st = "alert" if (ic < -0.02 and w >= 60) else "warn" if ic <= 0 else "ok"
            perf.append({"asof_date": end, "window_days": w, "n_obs": len(sub), "ic": ic, "auc": auc, "hit_rate": hit, "brier": br, "status": st})
        upsert_df(pd.DataFrame(perf), "live_performance", engine=engine)
    run = read_sql("SELECT trained_at FROM model_runs ORDER BY trained_at DESC LIMIT 1", engine=engine, parse_dates=["trained_at"])
    age = (pd.Timestamp.utcnow().tz_localize(None) - run.trained_at.iloc[0]).days if len(run) else None
    out = {"asof": str(last.date()), "n_drift_features": len(drift),
           "drift_alerts": int((drift.status == "alert").sum()) if len(drift) else 0,
           "drift_warnings": int((drift.status == "warn").sum()) if len(drift) else 0,
           "performance": perf, "model_age_days": age,
           "retrain_recommended": bool((age is not None and age > 30) or (len(drift) and (drift.status == "alert").sum() >= 3) or
                                       any(p["status"] == "alert" for p in perf))}
    print(f"✅ monitoring: {out['drift_alerts']} drift alerts, {out['drift_warnings']} warnings, model age {age}d, retrain_recommended={out['retrain_recommended']}")
    for p in perf:
        print(f"   {p['window_days']:>3}d  IC {p['ic']:+.3f}  AUC {p['auc']:.3f}  hit {p['hit_rate']:.1%}  [{p['status']}]")
    return out
