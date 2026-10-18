"""Orchestration for model training / prediction / signal generation."""
import uuid, json, datetime as dt
import numpy as np
import pandas as pd
import config
from db.connection import get_engine, upsert_df, read_sql
from features.build import build_panel, feature_columns, save_regime
from quality.leakage import audit_feature_names, pit_replay_test
from models import alpha, explain, signals
from risk.engine import run_risk
from sqlalchemy import text


def run_train(fast=False, pit_test=True, engine=None) -> dict:
    engine = engine or get_engine()
    print("🔧 building point-in-time feature panel ...")
    panel, info, rg = build_panel(save=True, engine=engine)
    save_regime(rg)
    feats = feature_columns(panel)
    audit_feature_names(feats)
    print(f"   {info}")
    if pit_test:
        r = pit_replay_test(panel, n_dates=3, engine=engine)
        print(f"   {'✅' if r['passed'] else '❌'} point-in-time replay test on {r['dates_tested']}: "
              f"{'no leakage detected' if r['passed'] else r['mismatched_features']}")
        if not r["passed"]:
            raise RuntimeError(f"Leakage detected in features: {list(r['mismatched_features'])[:10]}")
    print("🏋️ walk-forward training (LightGBM + XGBoost) ...")
    oof = alpha.walk_forward(panel, feats, fast=fast)
    metrics = alpha.evaluate_oof(oof, panel)
    print("📊 out-of-sample metrics:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                                                  if k not in ("quintile_excess_5d",)}, indent=1))
    bundle = alpha.fit_final(panel, feats, oof, fast=fast)
    path = alpha.save_bundle(bundle)
    run_id = "alpha_" + dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    upsert_df(pd.DataFrame([{"run_id": run_id, "model_name": "lgbm+xgb_ensemble", "train_end_date": pd.Timestamp(bundle["train_end"]),
                             "metrics": metrics, "params": {"lgb": config.LGB_PARAMS, "xgb": config.XGB_PARAMS, "fast": fast,
                                                            "alpha_h": config.ALPHA_H, "risk_h": config.RISK_H, "risk_th": config.RISK_TH},
                             "feature_list": feats, "artifact_path": str(path)}]), "model_runs", engine=engine)
    gi = explain.global_importance(bundle, panel[panel.date >= panel.date.max() - pd.Timedelta(days=500)])
    upsert_df(pd.DataFrame({"run_id": run_id, "feature": gi.index, "mean_abs_shap": gi.values}).head(40), "feature_importance", engine=engine)
    oof.to_pickle(config.ART_DIR / "oof_predictions.pkl")
    wf = oof.rename(columns={"date": "asof_date", "prob_up_raw": "prob_up", "prob_risk_raw": "prob_risk"}).assign(source="walk_forward", run_id=run_id)
    wf["score"] = wf.groupby("asof_date")["prob_up"].rank(pct=True)
    upsert_df(wf[["symbol", "asof_date", "source", "run_id", "prob_up", "exp_excess_ret", "ret_q10", "ret_q50", "ret_q90", "prob_risk", "score"]],
              "predictions", engine=engine)
    (config.REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"✅ model saved: {path}  (run_id={run_id})")
    return {"run_id": run_id, "metrics": metrics}


def run_predict(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    bundle = alpha.load_bundle()
    if bundle is None:
        raise RuntimeError("No trained model found. Run: python run_pipeline.py train")
    panel, info, rg = build_panel(engine=engine, include_news=("news_sent_3d" in bundle["features"]))
    save_regime(rg)
    missing = [c for c in bundle["features"] if c not in panel.columns]
    for c in missing:
        panel[c] = np.nan
    last_date = panel.date.max()
    last = panel[panel.date == last_date].reset_index(drop=True)
    p = alpha.predict_bundle(bundle, last)
    tf, _ = explain.explain_rows(bundle, last)
    run = read_sql("SELECT run_id FROM model_runs ORDER BY trained_at DESC LIMIT 1", engine=engine)
    p["top_factors"] = tf
    out = p.rename(columns={"date": "asof_date"}).assign(source="live", run_id=run.run_id.iloc[0] if len(run) else None)
    upsert_df(out[["symbol", "asof_date", "source", "run_id", "prob_up", "exp_excess_ret", "ret_q10", "ret_q50", "ret_q90",
                   "prob_risk", "top_factors"]], "predictions", engine=engine)
    print(f"✅ predictions for {len(out)} stocks as of {last_date.date()}")
    return out


def run_signals(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    pred = read_sql("SELECT * FROM predictions WHERE source='live' AND asof_date=(SELECT MAX(asof_date) FROM predictions WHERE source='live')",
                    engine=engine, parse_dates=["asof_date"])
    risk = read_sql("SELECT * FROM risk_metrics WHERE asof_date=(SELECT MAX(asof_date) FROM risk_metrics)", engine=engine)
    rg = read_sql("SELECT * FROM market_regime ORDER BY date DESC LIMIT 1", engine=engine)
    stress = float(rg.p_stress.iloc[0]) if len(rg) else None
    news = signals.latest_news(engine)
    d = signals.build_signals(pred.drop(columns=["score", "signal", "confidence", "risk_score", "risk_level", "reason"], errors="ignore"), risk, stress, news)
    upsert_df(d[["symbol", "asof_date", "source", "score", "signal", "confidence", "risk_score", "risk_level", "reason"]], "predictions", engine=engine)
    path, table = signals.export_legacy_table(d)
    print(f"✅ signals: {d.signal.value_counts().to_dict()}  -> {path}")
    return d


def run_predict_risk_signals(engine=None):
    run_predict(engine); run_risk(engine); return run_signals(engine)
