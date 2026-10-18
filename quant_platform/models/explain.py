"""SHAP explainability for the tree models (shap library if installed, else LightGBM native TreeSHAP)."""
import numpy as np
import pandas as pd


def shap_values(model, X: np.ndarray) -> np.ndarray:
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv
    except Exception:
        return model.booster_.predict(X, pred_contrib=True)[:, :-1]


def top_factors(sv_row, x_row, names, k=5):
    idx = np.argsort(-np.abs(sv_row))[:k]
    return [{"feature": names[i], "shap": round(float(sv_row[i]), 4),
             "value": None if np.isnan(x_row[i]) else round(float(x_row[i]), 4),
             "direction": "up" if sv_row[i] > 0 else "down"} for i in idx]


def explain_rows(bundle, df: pd.DataFrame, k=5):
    """Return list[top-factor dicts] per row of df, and mean |SHAP| global importance."""
    feats = bundle["features"]
    X = df[feats].to_numpy(dtype="float32")
    sv = shap_values(bundle["models"]["up_lgb"], X)
    return [top_factors(sv[i], X[i], feats, k) for i in range(len(X))], pd.Series(np.abs(sv).mean(axis=0), index=feats)


def global_importance(bundle, df: pd.DataFrame, n=1500, seed=3) -> pd.Series:
    d = df.sample(min(n, len(df)), random_state=seed)
    X = d[bundle["features"]].to_numpy(dtype="float32")
    sv = shap_values(bundle["models"]["up_lgb"], X)
    return pd.Series(np.abs(sv).mean(axis=0), index=bundle["features"]).sort_values(ascending=False)
