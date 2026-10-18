"""Market regime detection: 3-state Gaussian HMM on [NIFTY return, realised vol, log VIX].
Look-ahead safe: parameters are re-fit on an EXPANDING window using only data before each block, and the
probabilities are FILTERED (forward algorithm, causal) - not smoothed. Rule-based fallback if hmmlearn is missing."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import config

NAMES = {0: "calm", 1: "neutral", 2: "stress"}


def _features(nifty: pd.Series, vix: pd.Series) -> pd.DataFrame:
    r = nifty.pct_change(fill_method=None) * 100
    return pd.DataFrame({"ret": r, "rv10": r.rolling(10, min_periods=8).std(), "lvix": np.log(vix)}, index=nifty.index)


def _filter(model, X):
    K = model.n_components
    logB = np.column_stack([multivariate_normal.logpdf(X, mean=model.means_[k], cov=model.covars_[k], allow_singular=True)
                            for k in range(K)])
    A, pi = model.transmat_, model.startprob_
    out = np.zeros((len(X), K))
    a = pi * np.exp(logB[0] - logB[0].max()); a /= a.sum(); out[0] = a
    for t in range(1, len(X)):
        a = (a @ A) * np.exp(logB[t] - logB[t].max())
        s = a.sum()
        a = a / s if s > 0 else np.ones(K) / K
        out[t] = a
    return out


def compute_regime(nifty: pd.Series, vix: pd.Series, refit_every: int = 126, min_train: int = 250) -> pd.DataFrame:
    X = _features(nifty, vix).dropna()
    res = pd.DataFrame(index=nifty.index, columns=["p_calm", "p_neutral", "p_stress"], dtype=float)
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
        starts = list(range(min_train, len(X), refit_every))
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(X)
            sc = StandardScaler().fit(X.iloc[:s])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=config.SEED, min_covar=1e-3)
                m.fit(sc.transform(X.iloc[:s]))
            order = np.argsort(m.means_[:, 2])           # sort states by mean log-VIX: calm < neutral < stress
            P = _filter(m, sc.transform(X.iloc[:e]))[:, order]
            res.loc[X.index[s:e], ["p_calm", "p_neutral", "p_stress"]] = P[s:e]
    except Exception as ex:
        print(f"⚠️ HMM regime unavailable ({type(ex).__name__}: {ex}); using rule-based fallback")
        z = (np.log(vix) - np.log(vix).rolling(250, min_periods=120).mean()) / np.log(vix).rolling(250, min_periods=120).std()
        ps = 1 / (1 + np.exp(-2 * (z - 1)))
        pc = 1 / (1 + np.exp(2 * (z + 0.5)))
        res["p_stress"], res["p_calm"] = ps, pc
        res["p_neutral"] = (1 - ps - pc).clip(lower=0)
    res["regime"] = res[["p_calm", "p_neutral", "p_stress"]].values.argmax(axis=1).astype(float)
    res.loc[res["p_stress"].isna(), "regime"] = np.nan
    res["regime_name"] = res["regime"].map(NAMES)
    return res
