#!/usr/bin/env python3
"""
03_dough_rheology_loo.py
Author: Manuli Perera (The University of Western Australia)
Description:
    Master Leave-One-Out Cross-Validation (LOO-CV) engine for wheat dough rheology
    prediction across 17 ATR-FTIR spectral sub-regions (Original baseline vs. 
    Refined macromolecular windows) and six Farinograph and Extensograph parameters.
    Evaluates regularized linear and non-linear regression models (PLSR, Bayesian
    Ridge, Random Forest, XGBoost) and generates full cross-validation performance metrics.
Outputs:
    - final_figures_rheology_loo/Table_R1_LOO_all_models.csv (Table S2)
    - final_figures_rheology_loo/Table_R2_LOO_best_per_trait.csv (Table 2)
    - final_figures_rheology_loo/Table_R3_LOO_model_wins.csv (Table 3)
    - final_figures_rheology_loo/Figure_R2_LOO_heatmap.png (Figure 2)
    - final_figures_rheology_loo/Figure_R3_LOO_pred_vs_ref.png (Figure 3)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================== SETTINGS ====================================
MASTER = Path("processed/master_rheology_dataset.csv")
OUT = Path("final_figures_rheology_loo")
MAX_LV = 15
N_JOBS = -1          # use all CPU cores
SEED = 42

TRAITS = {  # column: (label for tables, unit)
    "Protein": ("Protein", "%"),
    "Absorption": ("Water absorption", "%"),
    "Extensibility": ("Extensibility", "cm"),
    "DDT": ("DDT", "min"),
    "Stability": ("Dough stability", "min"),
    "Rmax": ("Rmax", "BU"),
}
# Regions in the order used for tables and figures: each Original region is
# followed by the Novel sub-regions that split or match it.
REGIONS = [  # code, name, lo, hi
    ("Full", "Full spectrum", 650, 4000),
    ("O-A1", "Moisture", 2990, 3680), ("N-A1", "Moisture", 3000, 3700),
    ("O-A2", "Lipid C-H", 2825, 2990), ("N-A2", "Lipid C-H", 2800, 3000),
    ("O-A3", "Ester C=O", 1710, 1775), ("N-A3", "Ester C=O", 1710, 1770),
    ("O-A4", "Amide I+II", 1480, 1710), ("N-A4", "Amide I", 1600, 1710), ("N-A5", "Amide II", 1480, 1600),
    ("O-A5", "Amide III", 1180, 1480), ("N-A6", "Side chain", 1300, 1480), ("N-A7", "Amide III", 1180, 1300),
    ("O-A6", "Total starch", 810, 1180), ("N-A8", "Amorphous starch", 1070, 1180),
    ("N-A9", "Crystalline starch", 900, 1070), ("N-A10", "Skeletal vibrations", 650, 900),
]
MODELS = ["PLSR", "Bayesian Ridge", "Random Forest", "XGBoost"]
# ============================================================================

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def wave_cols(df):
    out = []
    for c in df.columns:
        try:
            v = float(str(c).strip())
        except ValueError:
            continue
        if 400 <= v <= 5000:
            out.append((v, c))
    out.sort()
    return np.array([v for v, _ in out]), [c for _, c in out]


def snv_sg(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    wl = min(11, Z.shape[1] if Z.shape[1] % 2 else Z.shape[1] - 1)
    if wl > 2:
        Z = savgol_filter(Z, wl, 2, deriv=1, axis=1)
    return np.nan_to_num(Z)


# ------------------------------ models --------------------------------------
def pls_all_k(Xtr, ytr, Xte, K):
    """Predictions for 1..K latent variables from one NIPALS fit."""
    m = PLSRegression(n_components=K, scale=False).fit(Xtr, ytr)
    T = (Xte - m._x_mean) @ m.x_rotations_
    return np.cumsum(T * m.y_loadings_.ravel(), axis=1) + m._y_mean


def loo_plsr(X, y):
    n, p = X.shape
    K = max(1, min(MAX_LV, p, n - 2))

    def one(i):
        tr = np.r_[0:i, i + 1:n]
        return pls_all_k(X[tr], y[tr], X[i:i + 1], K)[0]

    P = np.vstack(Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n)))
    rmse = np.sqrt(((P - y[:, None]) ** 2).mean(axis=0))
    k = int(np.argmax(rmse <= rmse.min() * 1.01)) + 1
    return P[:, k - 1], k


def make(name):
    if name == "Bayesian Ridge":
        return make_pipeline(StandardScaler(), BayesianRidge())
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.5, tree_method="hist", random_state=SEED, n_jobs=1, verbosity=0)
    raise ValueError(name)


def loo_generic(name, X, y):
    n = len(y)

    def one(i):
        tr = np.r_[0:i, i + 1:n]
        return float(make(name).fit(X[tr], y[tr]).predict(X[i:i + 1])[0])

    return np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n))), None


def metrics(y, pred):
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    rmse = np.sqrt(sse / len(y))
    return dict(R2CV=1 - sse / sst, RMSECV=rmse, RPD=y.std(ddof=1) / rmse,
                r2_corr=pearsonr(y, pred)[0] ** 2, Bias=(pred - y).mean())


# ------------------------------ figures -------------------------------------
def heatmap(best, path, cbar_label="LOO cross-validated R² (best of four models)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = [r[0] for r in REGIONS]
    lab = {r[0]: (r[1] if r[0] == "Full" else f"{r[0]} ({r[1]})") for r in REGIONS}
    M = best.pivot(index="Code", columns="Trait", values="R2CV").reindex(codes)[list(TRAITS)]
    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    im = ax.imshow(M.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.62 else "#1a1a1a")
    ax.set_xticks(range(len(TRAITS)), [TRAITS[t][0].replace(" ", "\n", 1) for t in TRAITS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(codes)), [lab[c] for c in codes], fontsize=8.5)
    for i, c in enumerate(codes):  # bold the Original regions
        if c == "Full" or c.startswith("O-"):
            ax.get_yticklabels()[i].set_fontweight("bold")
    for i, c in enumerate(codes[1:], start=1):  # line above each Original group
        if c.startswith("O-"):
            ax.axhline(i - 0.5, color="#555", lw=0.9)
    ax.set_xticks(np.arange(-.5, len(TRAITS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(codes)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def pred_plot(best_rows, preds, y_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, (_, r), letter in zip(axes.ravel(), best_rows.iterrows(), "ABCDEF"):
        y = y_all[r.Trait]
        p = preds[(r.Code, r.Trait, r.Model)]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#555", lw=1)
        ax.scatter(y, p, s=18, color="#2e7d4f", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        name, unit = TRAITS[r.Trait]
        ax.set_title(f"({letter}) {name} ({unit})", fontsize=10, loc="left")
        ax.text(0.04, 0.96, f"{r.Code}, {r.Model}\nR²CV = {r.R2CV:.3f}\nRMSECV = {r.RMSECV:.3g}\nRPD = {r.RPD:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("Measured", fontsize=9)
        ax.set_ylabel("Predicted (LOO-CV)", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------ main ----------------------------------------
def main():
    linear_only = "--linear" in sys.argv
    models = MODELS[:2] if linear_only else MODELS
    OUT.mkdir(exist_ok=True)
    if not MASTER.exists():
        sys.exit(f"Cannot find {MASTER.resolve()} - run this from your rheology folder.")
    df = pd.read_csv(MASTER, encoding="latin1")
    wn, wc = wave_cols(df)
    X_all = df[wc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    say("=" * 78)
    say(f"LOO-CV DOUGH QUALITY ANALYSIS   file: {MASTER}   samples: {len(df)}   channels: {len(wc)}")
    if "Cohort" in df.columns:
        say("samples per set: " + ", ".join(f"{k} = {v}" for k, v in df.Cohort.value_counts().sort_index().items()))
    traits = [t for t in TRAITS if t in df.columns]
    say("valid values: " + ", ".join(f"{t} = {df[t].notna().sum()}" for t in traits))
    if {"Protein", "DDT"} <= set(df.columns):
        d = df[["Protein", "DDT"]].dropna()
        r, pv = pearsonr(d.Protein, d.DDT)
        say(f"Pearson r protein vs DDT = {r:.2f} (n = {len(d)}, p = {pv:.1e})")
    say("trait summary (mean, SD, min, max):")
    for t in traits:
        v = df[t].dropna()
        say(f"  {t:<14} {v.mean():8.2f} {v.std():7.2f} {v.min():8.2f} {v.max():8.2f}")
    say(f"models: {', '.join(models)}")

    cache = OUT / "_loo_cache.csv"
    pcache = OUT / "_loo_predictions.npz"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    preds = dict(np.load(pcache, allow_pickle=True)["d"].item()) if pcache.exists() else {}
    y_all = {}
    t0 = time.time()
    for t in traits:
        mask = df[t].notna().to_numpy()
        y = df.loc[mask, t].to_numpy(float)
        y_all[t] = y
        for code, name, lo, hi in REGIONS:
            m = (wn >= lo) & (wn <= hi)
            X = snv_sg(X_all[mask][:, m])
            for model in models:
                if len(done) and ((done.Code == code) & (done.Trait == t) & (done.Model == model)).any():
                    continue
                if model == "PLSR":
                    p, k = loo_plsr(X, y)
                else:
                    p, k = loo_generic(model, X, y)
                row = dict(Trait=t, Code=code, Region=name, Range=f"{lo}-{hi}", Channels=X.shape[1], Model=model,
                           n=len(y), LV=k, **metrics(y, p))
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                preds[(code, t, model)] = p
            done.to_csv(cache, index=False)
            np.savez_compressed(pcache, d=np.array(preds, dtype=object))
            say(f"  [{time.time()-t0:6.0f}s] {t:<14} {code:<6} best so far: "
                + ", ".join(f"{r.Model} {r.R2CV:.3f}" for _, r in
                            done[(done.Trait == t) & (done.Code == code)].sort_values("R2CV", ascending=False).head(2).iterrows()))

    res = done[done.Model.isin(models)].copy()
    order = {r[0]: i for i, r in enumerate(REGIONS)}
    res["ord"] = res.Code.map(order)
    res = res.sort_values(["Trait", "ord", "Model"]).drop(columns="ord")
    res.round(4).to_csv(OUT / "Table_R1_LOO_all_models.csv", index=False)

    best_rm = res.loc[res.groupby(["Trait", "Code"]).R2CV.idxmax()]       # best model per region
    best = res.loc[res.groupby("Trait").R2CV.idxmax()].set_index("Trait").reindex(traits).reset_index()
    best.round(4).to_csv(OUT / "Table_R2_LOO_best_per_trait.csv", index=False)
    wins = pd.crosstab(best_rm.Trait, best_rm.Model).reindex(traits).fillna(0).astype(int)
    wins.loc["Total"] = wins.sum()
    wins.to_csv(OUT / "Table_R3_LOO_model_wins.csv")

    say("\n" + "=" * 78)
    say("BEST PER TRAIT (LOO-CV)")
    say(best[["Trait", "Code", "Model", "LV", "R2CV", "RMSECV", "RPD", "r2_corr"]].round(3).to_string(index=False))
    say("\nBEST ORIGINAL vs BEST NOVEL per trait")
    for t in traits:
        s = best_rm[best_rm.Trait == t]
        o = s[s.Code.str.startswith("O") | (s.Code == "Full")].sort_values("R2CV").iloc[-1]
        nv = s[s.Code.str.startswith("N")].sort_values("R2CV").iloc[-1]
        say(f"  {t:<14} Original {o.Code} {o.R2CV:.3f}  |  Novel {nv.Code} {nv.R2CV:.3f}")
    say("\nMEAN R2CV OF EACH MODEL ACROSS REGIONS")
    say(res.pivot_table(index="Trait", columns="Model", values="R2CV", aggfunc="mean").reindex(traits).round(3).to_string())
    say("\nREGION WINS PER MODEL")
    say(wins.to_string())

    heatmap(best_rm, OUT / "Figure_R2_LOO_heatmap.png")
    try:
        pred_plot(best, preds, y_all, OUT / "Figure_R3_LOO_pred_vs_ref.png")
    except KeyError:
        say("(predicted-vs-measured plot skipped: predictions missing from an older cache)")
    (OUT / "LOO_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\nDone in {time.time()-t0:.0f}s. Send me  {OUT}/LOO_summary.txt , Table_R1_LOO_all_models.csv and the two PNGs.")


if __name__ == "__main__":
    main()
