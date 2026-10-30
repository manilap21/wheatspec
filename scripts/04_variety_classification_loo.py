#!/usr/bin/env python3
"""
04_variety_classification_loo.py
Author: Manuli Perera (The University of Western Australia)
Description:
    Supervised cultivar authentication tournament across 14 mid-infrared spectral
    regions using Leave-One-Out Cross-Validation (LOO-CV).
    Benchmarks six machine learning classifiers (Linear SVM, Shrinkage LDA, Random
    Forest, XGBoost, KNN, and Gaussian Naive Bayes) across 10 commercial Western
    Australian wheat cultivars (n = 115 biological samples).
Outputs:
    - processed_variety/variety_master_training_set.csv
    - final_figures_variety/Table_V1_Variety_Tournament_All_Ranges.csv (Table V1)
    - final_figures_variety/Figure_V2_Variety_Tournament_Heatmap.png (Figure 4)
"""
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data_variety"
PROCESSED_DIR = BASE_DIR / "processed_variety"
OUT_DIR = BASE_DIR / "final_figures_variety"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_NAMES = [
    "SSA FTIR and Reference data.csv",
    "SSB FTIR and Reference data.csv",
    "SSC FTIR and Reference data.csv"
]

REGIONS_CONFIG = [
    {"name": "Original A1 Moisture",         "lo": 2990, "hi": 3680, "framework": "Original (Kelly et al.)"},
    {"name": "Original A2 Lipid C-H",        "lo": 2825, "hi": 2990, "framework": "Original (Kelly et al.)"},
    {"name": "Original A3 Lipid C=O",        "lo": 1710, "hi": 1775, "framework": "Original (Kelly et al.)"},
    {"name": "Original A4 Amide I+II",       "lo": 1480, "hi": 1710, "framework": "Original (Kelly et al.)"},
    {"name": "Original A5 Amide III",        "lo": 1180, "hi": 1480, "framework": "Original (Kelly et al.)"},
    {"name": "Original A6 Total Starch",     "lo": 810,  "hi": 1180, "framework": "Original (Kelly et al.)"},
    {"name": "Full Spectrum",                "lo": 650,  "hi": 4000, "framework": "Original (Kelly et al.)"},
    {"name": "Refined A4 Amide I",           "lo": 1600, "hi": 1710, "framework": "Refined Aim 4"},
    {"name": "Refined A5 Amide II",          "lo": 1480, "hi": 1600, "framework": "Refined Aim 4"},
    {"name": "Refined A6 Side Chain",        "lo": 1300, "hi": 1480, "framework": "Refined Aim 4"},
    {"name": "Refined A7 Amide III",         "lo": 1180, "hi": 1300, "framework": "Refined Aim 4"},
    {"name": "Refined A8 Amorphous Starch",  "lo": 1070, "hi": 1180, "framework": "Refined Aim 4"},
    {"name": "Refined A9 Crystalline Starch","lo": 900,  "hi": 1070, "framework": "Refined Aim 4"},
    {"name": "Refined A10 Skeletal Vibrations","lo": 650, "hi": 900, "framework": "Refined Aim 4"}
]

def find_file(fname):
    for d in [DATA_DIR, BASE_DIR]:
        p = d / fname
        if p.exists() and p.is_file():
            return p
    return None

def get_wave_cols(df):
    return sorted([
        c for c in df.columns 
        if str(c).replace('.', '', 1).lstrip('-').isdigit() and 400 <= float(c) <= 5000
    ], key=float)

def filter_region(wave_cols, lo, hi):
    return [c for c in wave_cols if lo <= float(c) <= hi]

def snv_sg(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    X_snv = (X - mu) / sd
    wl = min(11, X.shape[1])
    if wl % 2 == 0:
        wl -= 1
    if wl > 2:
        X_snv = savgol_filter(X_snv, wl, 2, deriv=1, axis=1)
    return np.nan_to_num(X_snv)

def run_historical_master_tournament():
    print("=" * 85)
    print("SCRIPT 01: HISTORICAL MASTER TOURNAMENT (SSA + SSB + SSC COMBINED)")
    print("=" * 85)

    dfs = []
    for fn in HISTORICAL_NAMES:
        fp = find_file(fn)
        if fp:
            d = pd.read_csv(fp, encoding="latin1")
            d["__Cohort__"] = fn.split()[0]
            dfs.append(d)
            print(f"  [+] Loaded '{fn}': {d.shape[0]} samples")
        else:
            print(f"  [!] Missing: '{fn}'")

    if not dfs:
        raise FileNotFoundError("Could not find SSA, SSB, or SSC files.")

    df_hist = pd.concat(dfs, axis=0, ignore_index=True)
    var_col = next(c for c in ["Variety", "Cultivar", "Variety_Clean", "Name"] if c in df_hist.columns)
    df_hist["Variety_Clean"] = (
        df_hist[var_col].astype(str).str.strip().str.upper()
        .replace({"EMU ROCK": "EMUROCK", "CHIEF CL PLUS": "CHIEFCLPLUS"})
    )

    df_clean = df_hist[~df_hist["Variety_Clean"].str.contains("WAGT", na=False)].copy()
    wn_cols = get_wave_cols(df_clean)
    df_clean = df_clean.drop_duplicates(subset=wn_cols[:20]).copy()

    counts = df_clean["Variety_Clean"].value_counts()
    print("\n--- All Cultivar Counts in Historical Matrix (SSA+SSB+SSC) ---")
    print(counts.to_string())

    keep_varieties = counts[counts >= 5].index.tolist()
    df_eval = df_clean[df_clean["Variety_Clean"].isin(keep_varieties)].copy()

    print(f"\n[+] Retained Cultivars (n >= 5): {keep_varieties}")
    print(f"[+] Total Clean Calibration Samples: {len(df_eval)}")
    eval_counts = df_eval["Variety_Clean"].value_counts()
    print("--- Evaluated Training Variety Counts ---")
    print(eval_counts.to_string())

    majority_rate = eval_counts.max() / eval_counts.sum()
    random_floor = 1.0 / len(keep_varieties)
    print(f"\n[i] Statistical Baselines:")
    print(f"    - Random guessing floor (1/k) : {random_floor*100:.2f}%")
    print(f"    - Majority class zero-rate    : {majority_rate*100:.2f}% ({eval_counts.index[0]})\n")

    out_master = PROCESSED_DIR / "variety_master_training_set.csv"
    df_eval.to_csv(out_master, index=False)
    print(f"  [OK] Master Training Matrix saved -> {out_master.resolve()}\n")

    y_str = df_eval["Variety_Clean"].values
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    classifiers = {
        "Linear SVM":    lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0, random_state=42)),
        "Shrinkage LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost":       lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                               random_state=42, eval_metric="mlogloss", n_jobs=-1),
        "KNN (k=3)":     lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "Naïve Bayes":   lambda: GaussianNB()
    }

    loo = LeaveOneOut()
    records = []

    for reg in REGIONS_CONFIG:
        r_name = reg["name"]
        lo, hi = reg["lo"], reg["hi"]
        range_str = f"{lo}–{hi} cm⁻¹"
        full_label = f"{r_name} ({range_str})"

        rc = filter_region(wn_cols, lo, hi)
        if len(rc) < 3:
            continue

        X_raw = df_eval[rc].values.astype(float)
        X_prep = snv_sg(X_raw)

        for m_name, m_func in classifiers.items():
            t0 = time.time()
            y_pred = np.zeros_like(y)

            for tr, te in loo.split(X_prep):
                sub_le = LabelEncoder()
                y_tr_enc = sub_le.fit_transform(y[tr])

                clf = m_func()
                clf.fit(X_prep[tr], y_tr_enc)
                pred_enc = clf.predict(X_prep[te])
                y_pred[te] = sub_le.inverse_transform(pred_enc)

            acc = accuracy_score(y, y_pred)
            bal_acc = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="weighted")
            elapsed = time.time() - t0

            records.append({
                "Framework": reg["framework"],
                "Region": r_name,
                "Range": range_str,
                "Full_Label": full_label,
                "Channels": len(rc),
                "Model": m_name,
                "Accuracy": round(acc, 4),
                "Balanced_Acc": round(bal_acc, 4),
                "Weighted_F1": round(f1, 4)
            })

        top_in_reg = max(records[-len(classifiers):], key=lambda x: x["Accuracy"])
        print(f"[{r_name:<30}] Top: {top_in_reg['Model']:<14} Acc={top_in_reg['Accuracy']*100:.1f}% | Bal_Acc={top_in_reg['Balanced_Acc']*100:.1f}% (F1={top_in_reg['Weighted_F1']:.3f})")

    df_res = pd.DataFrame(records)
    csv_out = OUT_DIR / "Table_V1_Variety_Tournament_All_Ranges.csv"
    df_res.to_csv(csv_out, index=False)

    pivot_acc = df_res.pivot_table(index="Full_Label", columns="Model", values="Accuracy", aggfunc="max")
    order = [f"{r['name']} ({r['lo']}–{r['hi']} cm⁻¹)" for r in REGIONS_CONFIG]
    pivot_acc = pivot_acc.reindex([o for o in order if o in pivot_acc.index])

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="YlGn", vmin=0.15, vmax=0.85, cbar_kws={'label': 'LOO-CV Accuracy'}, ax=ax)
    ax.set_title("Leave-One-Out Cultivar Classification Accuracy (SSA + SSB + SSC Combined)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Machine Learning Classifier", fontweight="bold")
    ax.set_ylabel("Spectral Range", fontweight="bold")
    plt.tight_layout()
    fig_heat = OUT_DIR / "Figure_V2_Variety_Tournament_Heatmap.png"
    fig.savefig(fig_heat, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 85)
    print("SUCCESS: Historical Master Tournament Complete!")
    print(f"  * Results Table CSV : {csv_out.resolve()}")
    print(f"  * Master Heatmap    : {fig_heat.resolve()}")
    print("=" * 85)

if __name__ == "__main__":
    run_historical_master_tournament()