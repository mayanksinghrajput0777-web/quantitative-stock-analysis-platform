"""EDA: data coverage, distributions, correlations, regimes, FII/DII vs NIFTY, feature information coefficients.
Writes PNGs + one self-contained HTML report to data/artifacts/eda/."""
import base64, io, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from db.connection import read_sql, get_engine
from features.build import build_panel, feature_columns


def _fig_html(fig, name, caption=""):
    p = config.EDA_DIR / f"{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    return f'<h3>{caption or name}</h3><img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def feature_ic(panel, cols, target):
    sub = panel.dropna(subset=[target])
    g = sub.groupby("date")
    R = g[cols + [target]].rank(pct=True)
    R = R - R.groupby(sub["date"]).transform("mean")
    num = R[cols].mul(R[target], axis=0).groupby(sub["date"]).sum()
    den = np.sqrt((R[cols] ** 2).groupby(sub["date"]).sum().mul((R[target] ** 2).groupby(sub["date"]).sum(), axis=0))
    ic = num / den.replace(0, np.nan)
    return ic.mean().sort_values(), ic


def run_eda(engine=None) -> str:
    engine = engine or get_engine()
    panel, info, rg = build_panel(engine=engine)
    ix = read_sql("SELECT * FROM index_prices", engine=engine, parse_dates=["date"])
    nifty = ix[ix.index_name == config.NIFTY_NAME].set_index("date")["close"].sort_index()
    vix = ix[ix.index_name == config.VIX_NAME].set_index("date")["close"].sort_index()
    oi = read_sql("SELECT * FROM participant_oi WHERE client_type IN ('FII','DII')", engine=engine, parse_dates=["date"])
    html = [f"<html><head><meta charset='utf-8'><title>EDA</title><style>body{{font-family:Arial;max-width:1100px;margin:auto}}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}</style></head><body><h1>Exploratory Data Analysis</h1>",
            f"<p>{info['n_symbols']} stocks, {info['n_rows']:,} stock-days, {info['start']} → {info['end']}, {info['n_features']} features. "
            f"News history: {info['news_days']} days (in model: {info['news_in_model']}).</p>"]
    # 1 coverage
    cov = panel.groupby("symbol").agg(first=("date", "min"), last=("date", "max"), rows=("date", "size")).reset_index()
    html.append("<h3>Coverage per stock</h3>" + cov.head(60).to_html(index=False))
    # 2 nifty / vix with regime
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(nifty.index, nifty.values, lw=1); ax[0].set_title("NIFTY 50"); ax[0].grid(alpha=.3)
    ax[1].plot(vix.index, vix.values, color="tab:red", lw=1); ax[1].set_title("India VIX (shaded = HMM 'stress' regime, filtered)"); ax[1].grid(alpha=.3)
    st = rg["p_stress"].dropna() > 0.5
    for d in st[st].index:
        ax[1].axvline(d, color="orange", alpha=.08)
    html.append(_fig_html(fig, "nifty_vix_regime", "NIFTY, VIX and market regime"))
    # 3 return distribution
    r = panel["ret_1d"].dropna().clip(-0.15, 0.15)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(r, bins=120, color="steelblue"); ax[0].set_title(f"Daily stock returns (skew {r.skew():.2f}, kurt {r.kurt():.1f})")
    vv = panel.groupby("symbol")["vol_60d"].last().sort_values() * np.sqrt(252)
    ax[1].barh(vv.index, vv.values); ax[1].set_title("Latest 60d annualised vol"); ax[1].tick_params(labelsize=6)
    html.append(_fig_html(fig, "returns", "Return distribution & volatility"))
    # 4 correlation
    wide = panel.pivot(index="date", columns="symbol", values="ret_1d")
    cm = wide.corr()
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=6); ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=6)
    fig.colorbar(im); ax.set_title("Return correlation")
    html.append(_fig_html(fig, "corr", "Correlation of daily returns"))
    # 5 FII vs NIFTY
    if not oi.empty:
        f = oi[oi.client_type == "FII"].set_index("date")
        net = (f.fut_idx_long - f.fut_idx_short)
        d = pd.DataFrame({"net": net, "nifty": nifty}).dropna()
        fwd = nifty.pct_change(5).shift(-5)
        lead = {k: float(net.shift(k).reindex(fwd.index).corr(fwd)) for k in (0, 1, 2, 3, 5)}
        fig, ax = plt.subplots(figsize=(11, 3.8)); ax.plot(d.index, d.net, color="tab:purple", lw=1); ax.set_ylabel("FII net index futures (contracts)")
        a2 = ax.twinx(); a2.plot(d.index, d.nifty, color="gray", lw=1, alpha=.7); a2.set_ylabel("NIFTY")
        ax.set_title("FII net index-futures positioning vs NIFTY  (Future Index Long reconstructed - see quality report)")
        html.append(_fig_html(fig, "fii_vs_nifty", "FII positioning vs NIFTY") + f"<p>corr(FII net lagged k days, NIFTY next-5d return): {json.dumps({k: round(v, 3) for k, v in lead.items()})}</p>")
    # 6 feature IC
    tcol = f"fwd_excess_{config.ALPHA_H}d"
    cols = [c for c in feature_columns(panel) if not c.startswith(("nifty_", "vix_", "fii_", "dii_", "pro_", "client_", "regime", "p_", "mkt_"))]
    ic_mean, _ = feature_ic(panel, cols, tcol)
    top = pd.concat([ic_mean.head(12), ic_mean.tail(12)])
    fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(top.index, top.values, color=np.where(top.values > 0, "seagreen", "indianred"))
    ax.set_title(f"Mean daily rank-IC of stock features vs {tcol} (|IC|>0.02 is notable)"); ax.tick_params(labelsize=7)
    html.append(_fig_html(fig, "feature_ic", "Univariate feature information coefficients"))
    # 7 target balance
    html.append(f"<h3>Targets</h3><p>P(outperform NIFTY, {config.ALPHA_H}d) = {panel['y_up'].mean():.3f}; "
                f"P(≤{config.RISK_TH:.0%} in {config.RISK_H}d) = {panel['y_risk'].mean():.3f}</p>")
    # 8 missingness
    mis = panel[feature_columns(panel)].isna().mean().sort_values(ascending=False).head(15)
    html.append("<h3>Most-missing features (warm-up periods)</h3>" + mis.to_frame("missing_share").round(3).to_html())
    # 9 news
    nw = read_sql("SELECT published_at, entity_type, sentiment FROM news", engine=engine, parse_dates=["published_at"])
    if len(nw):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].hist(nw.sentiment, bins=40); ax[0].set_title("News sentiment distribution")
        nw.groupby(nw.published_at.dt.date).size().plot(ax=ax[1], title="Headlines per day")
        html.append(_fig_html(fig, "news", "News"))
    html.append("</body></html>")
    out = config.EDA_DIR / "eda_report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"✅ EDA report: {out}")
    return str(out)
