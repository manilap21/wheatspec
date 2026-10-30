#!/usr/bin/env python3
"""
05_external_transfer_evaluation.py
Author: Manuli Perera (The University of Western Australia)
Description:
    Evaluates cross-seasonal calibration transferability (Hypothesis H3) by testing
    pre-calibrated historical models against an independent single-season harvest cohort.
    Enforces a strict zero-replicate-leakage protocol by averaging technical FTIR
    replicates to discrete biological sample means prior to evaluation. Generates
    transfer accuracy metrics, per-cultivar recall reports, confusion matrices, and 
    2D canonical LDA score projections.
Outputs:
    - final_figures_variety/Table_V2_True_External_Validation_Summary.csv (Table V2)
    - final_figures_variety/Figure_V3_True_External_CM_*.png
    - final_figures_variety/Figure_V4_True_External_LDA_Projection.png (Figure 5)
"""
import os
import re
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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data_variety"
PROCESSED_DIR = BASE_DIR / "processed_variety"
OUT_DIR = BASE_DIR / "final_figures_variety"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DIR / "variety_master_training_set.csv"

# Pre-chosen configurations based on the top performers in the historical tournament
PRECHOSEN_CONFIGS = [
    {
        "name": "Original A6 Total Starch (810–1180 cm⁻¹)",
        "lo": 810, "hi": 1180,
        "model_name": "Shrinkage LDA",
        "clf": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    },
    {
        "name": "Original A5 Amide III (1180–1480 cm⁻¹)",
        "lo": 1180, "hi": 1480,
        "model_name": "Shrinkage LDA",
        "clf": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    },
    {
        "name": "Refined A9 Crystalline Starch (900–1070 cm⁻¹)",
        "lo": 900, "hi": 1070,
        "model_name": "Linear SVM",
        "clf": lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0, random_state=42))
    },
    {
        "name": "Full Spectrum (650–4000 cm⁻¹)",
        "lo": 650, "hi": 4000,
        "model_name": "Random Forest",
        "clf": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
    }
]

def find_validation_file():
    candidates = [
        DATA_DIR / "compiled_master_dataset.csv",
        BASE_DIR / "compiled_master_dataset.csv",
        PROCESSED_DIR / "compiled_master_dataset.csv",
        BASE_DIR / "variety_validation_biological_means.csv",
        PROCESSED_DIR / "variety_validation_biological_means.csv"
    ]
    for p in candidates:
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

def parse_variety_name(val):
    clean = str(val).upper().strip().replace(" ", "").replace("_", "")
    known = ['WESTONIA', 'WYALKATCHEM', 'SCEPTER', 'EMUROCK', 'MACE', 'YITPI', 'MAGENTA', 'CORACK', 'RAC', 'CHIEFCLPLUS', 'OAGT']
    for k in known:
        if k in clean:
            return k
    return clean

def run_true_external_validation():
    print("=" * 85)
    print("SCRIPT 00: TRUE EXTERNAL CALIBRATION TRANSFER (HYPOTHESIS H3)")
    print("=" * 85)

    val_fp = find_validation_file()
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing curated training set: {TRAIN_FILE.resolve()}. Run Script 01 first!")
    if not val_fp:
        raise FileNotFoundError("Missing validation file. Check data_variety/compiled_master_dataset.csv.")

    print(f"  [+] Loading Master Training Set: '{TRAIN_FILE.resolve()}'")
    print(f"  [+] Loading External Validation Set: '{val_fp.resolve()}'")

    df_train = pd.read_csv(TRAIN_FILE)
    df_val_raw = pd.read_csv(val_fp, encoding="latin1")

    train_wns = get_wave_cols(df_train)
    val_wns = get_wave_cols(df_val_raw)

    v_col = "Variety" if "Variety" in df_val_raw.columns else "Variety_Clean"
    df_val_raw["Variety_Clean"] = df_val_raw[v_col].apply(parse_variety_name)

    # 1. Biological ID Grouping (Zero Replicate Leakage)
    bio_col = None
    for cand in ['Bio_Rep_ID', 'Sample_ID', 'Sample', 'Name']:
        if cand in df_val_raw.columns:
            bio_col = cand
            break

    df_val_raw[val_wns] = df_val_raw[val_wns].apply(pd.to_numeric, errors="coerce")

    if bio_col:
        df_val_means = df_val_raw.groupby(["Variety_Clean", bio_col])[val_wns].mean().reset_index()
        print(f"  [+] Compressed {len(df_val_raw)} raw scans by '{bio_col}' into {len(df_val_means)} biological sample means")
    elif "Source_File" in df_val_raw.columns:
        df_val_raw["Extracted_Bio_ID"] = df_val_raw["Source_File"].apply(lambda x: str(x).split('_Rep')[0] if '_Rep' in str(x) else str(x))
        df_val_means = df_val_raw.groupby(["Variety_Clean", "Extracted_Bio_ID"])[val_wns].mean().reset_index()
        print(f"  [+] Compressed {len(df_val_raw)} raw scans into {len(df_val_means)} biological sample means")
    else:
        df_val_means = df_val_raw.copy()

    # 2. Alignment of Cultivars
    train_varieties = df_train["Variety_Clean"].unique().tolist()
    val_varieties_all = df_val_means["Variety_Clean"].unique().tolist()

    common_cvs = sorted(np.intersect1d(train_varieties, val_varieties_all))
    excluded_cvs = sorted(set(val_varieties_all) - set(common_cvs))

    print(f"\n[i] Cultivar Alignment Audit:")
    print(f"    - Historical Training Cultivars (n >= 5): {sorted(train_varieties)}")
    print(f"    - Overlapping Validated Cultivars ({len(common_cvs)}) : {common_cvs}")
    if excluded_cvs:
        print(f"    - Excluded from Scoring (not in training panel n>=5): {excluded_cvs}")

    sub_train = df_train[df_train["Variety_Clean"].isin(common_cvs)].copy()
    sub_val = df_val_means[df_val_means["Variety_Clean"].isin(common_cvs)].copy()

    print(f"    - Final Validation Cohort Size          : {len(sub_val)} biological samples")
    print("--- Validation Sample Counts per Cultivar ---")
    print(sub_val["Variety_Clean"].value_counts().to_string())

    random_floor = 1.0 / len(common_cvs)
    print(f"\n[i] Theoretical random guessing floor across common cultivars: {random_floor*100:.2f}%\n")

    summary_records = []

    print("-" * 85)
    print("EVALUATING PRE-CHOSEN HYPOTHESIS H3 TRANSFER CONFIGURATIONS")
    print("-" * 85)

    for cfg in PRECHOSEN_CONFIGS:
        cfg_name = cfg["name"]
        m_name = cfg["model_name"]
        lo, hi = cfg["lo"], cfg["hi"]

        rc_tr = filter_region(train_wns, lo, hi)
        rc_val = filter_region(val_wns, lo, hi)

        X_tr = snv_sg(sub_train[rc_tr].values.astype(float))
        y_tr = sub_train["Variety_Clean"].values

        # Linear interpolation to ensure identical wavenumber coordinates
        X_val_raw = sub_val[rc_val].values.astype(float)
        wn_tr_f = np.array([float(c) for c in rc_tr])
        wn_val_f = np.array([float(c) for c in rc_val])
        X_val_interp = np.zeros((len(X_val_raw), len(wn_tr_f)))
        for i in range(len(X_val_raw)):
            X_val_interp[i, :] = np.interp(wn_tr_f, wn_val_f, X_val_raw[i, :])
        X_val = snv_sg(X_val_interp)
        y_val = sub_val["Variety_Clean"].values

        # Fit model on historical data, predict validation cohort
        model = cfg["clf"]()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)

        print(f"\n🎯 Configuration: {cfg_name} | {m_name}")
        print(f"   Transfer Accuracy     : {acc*100:.2f}% ({np.sum(y_val == y_pred)}/{len(y_val)} correct)")
        print(f"   Balanced Accuracy    : {bal_acc*100:.2f}%")
        print("   Detailed Per-Variety Recall:")
        report = classification_report(y_val, y_pred, output_dict=True)
        for cv in common_cvs:
            if cv in report:
                rec = report[cv]["recall"]
                supp = int(report[cv]["support"])
                print(f"    - {cv:<14}: Recall = {rec*100:>5.1f}% ({int(rec*supp)}/{supp} correct)")

        summary_records.append({
            "Configuration": cfg_name,
            "Model": m_name,
            "Transfer_Accuracy": round(acc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "Tested_Samples": len(y_val)
        })

        # Plot and save confusion matrix
        labels = common_cvs
        cm = confusion_matrix(y_val, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"External Validation: {cfg_name}\n({m_name}) | Transfer Acc: {acc*100:.1f}%", fontweight='bold', fontsize=10)
        ax.set_xlabel("Predicted Cultivar (Historical SSA+SSB+SSC)", fontweight='bold')
        ax.set_ylabel("True Harvest Cultivar (Fresh Harvest)", fontweight='bold')
        plt.tight_layout()
        cm_out = OUT_DIR / f"Figure_V3_True_External_CM_{m_name.replace(' ', '_')}_{lo}_{hi}.png"
        fig.savefig(cm_out, dpi=300)
        plt.close(fig)

    # 3. Canonical 2D Projection on Original A6 Total Starch
    print("\n[+] Generating Canonical 2D Projection (Original A6 Total Starch)...")
    rc_tr_a6 = filter_region(train_wns, 810, 1180)
    rc_val_a6 = filter_region(val_wns, 810, 1180)

    X_tr_a6 = snv_sg(sub_train[rc_tr_a6].values.astype(float))
    wn_tr_f = np.array([float(c) for c in rc_tr_a6])
    wn_val_f = np.array([float(c) for c in rc_val_a6])

    X_val_a6_raw = sub_val[rc_val_a6].values.astype(float)
    X_val_a6_interp = np.zeros((len(X_val_a6_raw), len(wn_tr_f)))
    for i in range(len(X_val_a6_raw)):
        X_val_a6_interp[i, :] = np.interp(wn_tr_f, wn_val_f, X_val_a6_raw[i, :])
    X_val_a6 = snv_sg(X_val_a6_interp)

    # Use 'eigen' so .transform() works seamlessly with shrinkage enabled
    lda_2d = LinearDiscriminantAnalysis(n_components=2, solver="eigen", shrinkage="auto")
    X_tr_lda = lda_2d.fit_transform(X_tr_a6, sub_train["Variety_Clean"].values)
    X_val_lda = lda_2d.transform(X_val_a6)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # Historical training points: Faint
    sns.scatterplot(x=X_tr_lda[:, 0], y=X_tr_lda[:, 1], hue=sub_train["Variety_Clean"].values,
                    style=sub_train["Variety_Clean"].values, s=60, alpha=0.35, palette="tab10", ax=ax, legend=False)
    # Fresh validation points: Bold with dark outlines
    sns.scatterplot(x=X_val_lda[:, 0], y=X_val_lda[:, 1], hue=sub_val["Variety_Clean"].values,
                    style=sub_val["Variety_Clean"].values, s=110, edgecolor="black", linewidth=1.4, palette="tab10", ax=ax)

    ax.set_title("2D Canonical Projection: Original A6 Total Starch (810–1180 cm⁻¹)\n(Faint = Historical Training SSA+SSB+SSC | Bold = Fresh Harvest Validation Means)",
                 fontweight="bold", fontsize=10, pad=12)
    ax.set_xlabel("Canonical Discriminant 1 (LD1)", fontweight="bold")
    ax.set_ylabel("Canonical Discriminant 2 (LD2)", fontweight="bold")
    ax.legend(title="Cultivar", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig_proj = OUT_DIR / "Figure_V4_True_External_LDA_Projection.png"
    fig.savefig(fig_proj, dpi=300)
    plt.close(fig)

    df_summary = pd.DataFrame(summary_records)
    summary_csv = OUT_DIR / "Table_V2_True_External_Validation_Summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 85)
    print("SUCCESS: True External Calibration Transfer Complete!")
    print(f"  * Results Table CSV     : {summary_csv.resolve()}")
    print(f"  * 2D Projection Figure  : {fig_proj.resolve()}")
    print("=" * 85)

if __name__ == "__main__":
    run_true_external_validation()