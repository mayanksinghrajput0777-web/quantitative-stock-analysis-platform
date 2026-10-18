"""Alpha + downside-risk models: LightGBM & XGBoost ensembles, expanding-window walk-forward with embargo,
isotonic calibration, quantile (range) forecasts. Pooled across all stocks (cross-sectional model)."""
import uuid, json, datetime as dt
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
try:
    import xgboost as xgb
except Exception:          # xgboost missing -> a second, differently-configured LightGBM is used instead
    xgb = None
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
import config

T_UP, T_EXC, T_RISK = "y_up", f"fwd_excess_{config.ALPHA_H}d", "y_risk"
T_RET = f"fwd_ret_{config.ALPHA_H}d"


# ---------------------------------------------------------------- model factories
def _lgb_params(fast): 
    p = dict(config.LGB_PARAMS)
    if fast: p["n_estimators"] = 120
    return p

def _xgb_params(fast):
    p = dict(config.XGB_PARAMS)
    if fast: p["n_estimators"] = 120
    return p

def new_models(fast=False):
    lp, xp = _lgb_params(fast), _xgb_params(fast)
    if xgb is not None:
        m = {"up_lgb": lgb.LGBMClassifier(**lp), "up_xgb": xgb.XGBClassifier(**xp, eval_metric="logloss"),
             "exc_lgb": lgb.LGBMRegressor(**lp), "exc_xgb": xgb.XGBRegressor(**xp),
             "risk_lgb": lgb.LGBMClassifier(**lp), "risk_xgb": xgb.XGBClassifier(**xp, eval_metric="logloss")}
    else:
        alt = {**lp, "num_leaves": 31, "colsample_bytree": 0.5, "random_state": 7}
        m = {"up_lgb": lgb.LGBMClassifier(**lp), "up_xgb": lgb.LGBMClassifier(**alt),
             "exc_lgb": lgb.LGBMRegressor(**lp), "exc_xgb": lgb.LGBMRegressor(**alt),
             "risk_lgb": lgb.LGBMClassifier(**lp), "risk_xgb": lgb.LGBMClassifier(**alt)}
    for q in config.QUANTILES:
        qp = dict(lp); qp["n_estimators"] = max(80, qp["n_estimators"] // 2)
        m[f"q{int(q*100)}"] = lgb.LGBMRegressor(objective="quantile", alpha=q, **qp)
    return m


def _fit_all(models, X, df, fast=False):
    """Fit every model on its own target (rows with a missing label are skipped for that model)."""
    def sub(target):
        ok = df[target].notna().values
        return X[ok], df.loc[ok, target].values
    for name, m in models.items():
        if name.startswith("up_"):
            Xi, y = sub(T_UP); m.fit(Xi, y.astype(int))
        elif name.startswith("exc_"):
            Xi, y = sub(T_EXC); lo, hi = np.percentile(y, [1, 99]); m.fit(Xi, np.clip(y, lo, hi))
        elif name.startswith("risk_"):
            Xi, y = sub(T_RISK); m.fit(Xi, y.astype(int))
        elif name.startswith("q"):
            Xi, y = sub(T_RET); lo, hi = np.percentile(y, [1, 99]); m.fit(Xi, np.clip(y, lo, hi))
    return models


def _predict_all(models, X) -> pd.DataFrame:
    o = pd.DataFrame(index=range(len(X)))
    o["prob_up_raw"] = np.mean([models["up_lgb"].predict_proba(X)[:, 1], models["up_xgb"].predict_proba(X)[:, 1]], axis=0)
    o["exp_excess_ret"] = np.mean([models["exc_lgb"].predict(X), models["exc_xgb"].predict(X)], axis=0)
    o["prob_risk_raw"] = np.mean([models["risk_lgb"].predict_proba(X)[:, 1], models["risk_xgb"].predict_proba(X)[:, 1]], axis=0)
    qs = np.sort(np.column_stack([models[f"q{int(q*100)}"].predict(X) for q in config.QUANTILES]), axis=1)  # no crossing
    for i, q in enumerate(config.QUANTILES):
        o[f"ret_q{int(q*100)}"] = qs[:, i]
    return o


# ---------------------------------------------------------------- walk-forward
def make_folds(n_dates, min_train, test_len, step, embargo):
    folds, s = [], min_train
    while s < n_dates:
        folds.append((s - embargo, s, min(s + test_len, n_dates)))   # train on dates[:s-embargo], test dates[s:e]
        s += step
    return folds


def walk_forward(panel: pd.DataFrame, feats: list, fast=False, verbose=True) -> pd.DataFrame:
    dates = np.sort(panel["date"].unique())
    folds = make_folds(len(dates), config.WF_MIN_TRAIN_DAYS, config.WF_TEST_DAYS, config.WF_STEP_DAYS, config.EMBARGO_DAYS)
    if fast:
        folds = folds[-3:]
    out = []
    for k, (tr_end, ts, te) in enumerate(folds):
        tr = panel[panel["date"] < dates[tr_end]] if k >= 0 else None
        tr = tr if len(tr) else panel.iloc[:0]
        # restrict training to rows whose labels were fully known before the test starts (embargo handles overlap)
        te_df = panel[(panel["date"] >= dates[ts]) & (panel["date"] < (dates[te] if te < len(dates) else dates[-1] + pd.Timedelta(days=1)))]
        if len(tr) < 2000 or len(te_df) == 0:
            continue
        Xtr = tr[feats].to_numpy(dtype="float32"); Xte = te_df[feats].to_numpy(dtype="float32")
        models = _fit_all(new_models(fast), Xtr, tr, fast)
        p = _predict_all(models, Xte)
        p.insert(0, "symbol", te_df["symbol"].values); p.insert(0, "date", te_df["date"].values)
        p["fold"] = k
        out.append(p)
        if verbose:
            print(f"  fold {k+1}/{len(folds)}: train<{pd.Timestamp(dates[tr_end]).date()} ({len(tr)} rows) "
                  f"test {pd.Timestamp(dates[ts]).date()}..{pd.Timestamp(dates[min(te,len(dates))-1]).date()} ({len(te_df)} rows)")
    if not out:
        raise RuntimeError("Not enough history for walk-forward (need > WF_MIN_TRAIN_DAYS dates).")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- evaluation
def _ic_by_date(df, pred, tgt):
    def f(g):
        if len(g) < 8 or g[pred].nunique() < 3: return np.nan
        return spearmanr(g[pred], g[tgt])[0]
    return df.groupby("date").apply(f, include_groups=False) if _pd_has_include() else df.groupby("date").apply(f)


def _pd_has_include():
    try:
        pd.DataFrame({"a": [1], "b": [1]}).groupby("a").apply(lambda g: 1, include_groups=False); return True
    except TypeError:
        return False


def evaluate_oof(oof: pd.DataFrame, panel: pd.DataFrame) -> dict:
    m = oof.merge(panel[["date", "symbol", T_UP, T_EXC, T_RISK, T_RET]], on=["date", "symbol"], how="left")
    res = {"n_oof_rows": int(len(m)), "oof_start": str(m.date.min().date()), "oof_end": str(m.date.max().date())}
    a = m.dropna(subset=[T_UP])
    if a["y_up"].nunique() > 1:
        res["auc_up"] = float(roc_auc_score(a[T_UP], a.prob_up_raw))
        res["acc_up"] = float(accuracy_score(a[T_UP], a.prob_up_raw > 0.5))
        res["brier_up_raw"] = float(brier_score_loss(a[T_UP], a.prob_up_raw))
        h = len(a) // 2   # calibration check: fit on first half, score on second half
        a = a.sort_values("date")
        iso = IsotonicRegression(out_of_bounds="clip").fit(a.prob_up_raw.iloc[:h], a[T_UP].iloc[:h])
        res["brier_up_h2_raw"] = float(brier_score_loss(a[T_UP].iloc[h:], a.prob_up_raw.iloc[h:]))
        res["brier_up_h2_calibrated"] = float(brier_score_loss(a[T_UP].iloc[h:], iso.predict(a.prob_up_raw.iloc[h:])))
    b = m.dropna(subset=[T_EXC])
    ic = _ic_by_date(b, "exp_excess_ret", T_EXC).dropna()
    ic2 = _ic_by_date(b, "prob_up_raw", T_EXC).dropna()
    res["ic_mean"], res["ic_std"] = float(ic.mean()), float(ic.std())
    res["icir"] = float(ic.mean() / ic.std()) if ic.std() > 0 else None
    res["ic_hit_rate"] = float((ic > 0).mean())
    res["ic_prob_up_mean"] = float(ic2.mean())
    b = b.assign(q=b.groupby("date")["prob_up_raw"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop")))
    qm = b.groupby("q")[T_EXC].mean()
    res["quintile_excess_5d"] = {int(k): float(v) for k, v in qm.items()}
    res["top_minus_bottom_5d"] = float(qm.iloc[-1] - qm.iloc[0])
    c = m.dropna(subset=[T_RISK])
    if c[T_RISK].nunique() > 1:
        res["auc_risk"] = float(roc_auc_score(c[T_RISK], c.prob_risk_raw))
        res["risk_base_rate"] = float(c[T_RISK].mean())
        res["brier_risk"] = float(brier_score_loss(c[T_RISK], c.prob_risk_raw))
    d = m.dropna(subset=[T_RET])
    res["quantile_coverage_80"] = float(((d[T_RET] >= d.ret_q10) & (d[T_RET] <= d.ret_q90)).mean())
    return res


# ---------------------------------------------------------------- final fit / inference
def fit_final(panel: pd.DataFrame, feats: list, oof: pd.DataFrame, fast=False) -> dict:
    X = panel[feats].to_numpy(dtype="float32")
    models = _fit_all(new_models(fast), X, panel, fast)
    m = oof.merge(panel[["date", "symbol", T_UP, T_RISK]], on=["date", "symbol"], how="left")
    cal = {}
    a = m.dropna(subset=[T_UP]); cal["up"] = IsotonicRegression(out_of_bounds="clip").fit(a.prob_up_raw, a[T_UP])
    c = m.dropna(subset=[T_RISK]); cal["risk"] = IsotonicRegression(out_of_bounds="clip").fit(c.prob_risk_raw, c[T_RISK])
    lab = panel.dropna(subset=[T_UP])
    return {"models": models, "calibrators": cal, "features": feats, "trained_at": dt.datetime.utcnow().isoformat(),
            "train_end": str(lab["date"].max().date()), "fast": fast}


def predict_bundle(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[bundle["features"]].to_numpy(dtype="float32")
    p = _predict_all(bundle["models"], X)
    p["prob_up"] = bundle["calibrators"]["up"].predict(p["prob_up_raw"])
    p["prob_risk"] = bundle["calibrators"]["risk"].predict(p["prob_risk_raw"])
    p.insert(0, "symbol", df["symbol"].values); p.insert(0, "date", df["date"].values)
    return p


def save_bundle(bundle, path=None):
    path = path or (config.MODEL_DIR / "alpha_bundle.joblib")
    joblib.dump(bundle, path); return path


def load_bundle(path=None):
    path = path or (config.MODEL_DIR / "alpha_bundle.joblib")
    return joblib.load(path) if path.exists() else None
