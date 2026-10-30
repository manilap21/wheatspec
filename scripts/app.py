"""
WheatSpec: Chemometric & Machine Learning Phenotyping of Wheat Flour via ATR-FTIR
Streamlit Web Application (app.py)

Interactive chemometrics portal for ATR-FTIR spectral preprocessing,
multi-window regional exploration, and cultivar classification tournaments.

Author: Manuli Perera (UWA Master's Dissertation, 2026)
Supervisors: Assoc. Prof. Nicolas L. Taylor, Dr. James M. Kelly
License: MIT
"""

import warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="WheatSpec | ATR-FTIR Chemometrics Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Regional Registry Definitions (Kelly Baseline + Novel Sub-regions)
REGIONS_ALL = {
    "A1 Moisture O-H (3000–3700 cm⁻¹)": (3000, 3700),
    "A2 Lipid C-H (2800–3000 cm⁻¹)": (2800, 3000),
    "A3 Lipid Ester C=O (1710–1770 cm⁻¹)": (1710, 1770),
    "A4 Amide I Protein (1600–1710 cm⁻¹)": (1600, 1710),
    "A5 Amide II Protein (1480–1600 cm⁻¹)": (1480, 1600),
    "A6 Side Chain CH2 (1300–1480 cm⁻¹)": (1300, 1480),
    "A7 Amide III Matrix (1180–1300 cm⁻¹)": (1180, 1300),
    "A8 Amorphous Starch (1070–1180 cm⁻¹)": (1070, 1180),
    "A9 Crystalline Starch (900–1070 cm⁻¹)": (900, 1070),
    "A10 Skeletal Tail (650–900 cm⁻¹)": (650, 900),
    "Kelly A6 Starch Baseline (810–1200 cm⁻¹)": (810, 1200),
    "Full Continuous Spectrum (650–4000 cm⁻¹)": (650, 4000)
}

def apply_snv_sg1d(X_raw: np.ndarray, window_size: int = 11, poly_order: int = 2) -> np.ndarray:
    """Applies Standard Normal Variate (SNV) followed by Savitzky-Golay 1st derivative."""
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    std[std == 0] = 1.0
    X_snv = (X_raw - mean) / std

    n_channels = X_raw.shape[1]
    if n_channels < window_size:
        window_size = n_channels if n_channels % 2 != 0 else n_channels - 1
        if window_size <= poly_order:
            return X_snv

    return savgol_filter(X_snv, window_length=window_size, polyorder=poly_order, deriv=1, axis=1)

def extract_spectral_matrix(df: pd.DataFrame, start_wn: float, end_wn: float):
    """Extracts numeric wavenumber column headers within the designated regional interval."""
    cols = []
    for c in df.columns:
        try:
            wn = float(str(c).strip())
            if start_wn <= wn <= end_wn:
                cols.append((wn, c))
        except ValueError:
            pass
    cols.sort(key=lambda x: x[0])
    col_names = [x[1] for x in cols]
    return df[col_names].values.astype(float), len(col_names)

# Sidebar Controls
st.sidebar.title("Parameters & Input")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload Spectral Data (.csv / .xlsx)", 
    accept_multiple_files=True
)

target_region_str = st.sidebar.selectbox(
    "Target Spectral Region Mask", 
    options=list(REGIONS_ALL.keys()), 
    index=8  # Defaults to A9 Crystalline Starch
)
start_wn, end_wn = REGIONS_ALL[target_region_str]

snv_active = st.sidebar.checkbox("Standard Normal Variate (SNV)", value=True)
sg_window = st.sidebar.slider("Savitzky-Golay Window Size (points)", 5, 21, 11, step=2)

st.title("WheatSpec: ATR-FTIR Chemometrics & Phenotyping Portal")
st.markdown(
    "High-throughput spectral preprocessing, regional masking, and machine learning classification "
    "for wheat quality phenotyping and cultivar authentication."
)
st.markdown("---")

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        if f.name.endswith(".csv"):
            dfs.append(pd.read_csv(f))
        elif f.name.endswith(".xlsx"):
            dfs.append(pd.read_excel(f))
            
    df_raw = pd.concat(dfs, axis=0, ignore_index=True)
    st.success(f"Ingested {len(uploaded_files)} file(s) containing {len(df_raw)} total spectra.")

    # Column Selector Row
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        var_col = st.selectbox("Cultivar / Class Label Column", df_raw.columns, index=0)
    with col_sel2:
        bio_id_col = st.selectbox("Sample Identifier Column", df_raw.columns, index=min(1, len(df_raw.columns) - 1))

    df_raw['Variety_Clean'] = df_raw[var_col].astype(str).str.strip().str.upper()

    # Dynamic Technical Replicate Aggregation
    spec_cols_all = []
    for c in df_raw.columns:
        try:
            float(str(c).strip())
            spec_cols_all.append(c)
        except ValueError:
            pass

    if bio_id_col in df_raw.columns and bio_id_col != var_col:
        df_raw[spec_cols_all] = df_raw[spec_cols_all].apply(pd.to_numeric, errors='coerce')
        df_proc = df_raw.groupby(['Variety_Clean', bio_id_col])[spec_cols_all].mean().reset_index()
        st.info(f"Aggregated {len(df_raw)} raw technical scans across '{bio_id_col}' into {len(df_proc)} biological sample means.")
    else:
        df_proc = df_raw.copy()

    X_raw, n_channels = extract_spectral_matrix(df_proc, start_wn, end_wn)
    y_str = df_proc['Variety_Clean'].values

    le_global = LabelEncoder()
    y_global = le_global.fit_transform(y_str)
    labels_str = le_global.classes_

    # Preprocess Spectra
    X_prep = apply_snv_sg1d(X_raw, window_size=sg_window)

    # Application Tabs
    tab1, tab2, tab3 = st.tabs([
        "Spectral Inspection", 
        "Model Tournament", 
        "Canonical Latent Space"
    ])

    # Tab 1: Spectral Preprocessing
    with tab1:
        st.subheader(f"Spectral Profile: {target_region_str}")
        fig_spec = go.Figure()
        
        for i in range(min(15, len(X_prep))):
            fig_spec.add_trace(go.Scatter(y=X_prep[i, :], mode='lines', name=f"Sample {i+1} ({y_str[i]})"))
            
        fig_spec.update_layout(
            title="Preprocessed Absorbance Derivatives (SNV + Savitzky-Golay 1st Derivative)",
            xaxis_title="Spectral Feature Index",
            yaxis_title="First Derivative Absorbance",
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    # Tab 2: Classification Tournament
    with tab2:
        st.subheader("Cultivar Classification Tournament (Leave-One-Out Cross-Validation)")
        if st.button("Execute Cross-Validation Benchmark", type="primary"):
            classifiers = {
                "Shrinkage LDA": LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                "Linear SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='mlogloss', random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
                "Naïve Bayes": GaussianNB()
            }

            loo = LeaveOneOut()
            cols_grid = st.columns(3)

            for idx, (name, clf) in enumerate(classifiers.items()):
                y_pred_str = []
                for train_idx, test_idx in loo.split(X_prep):
                    X_tr, X_te = X_prep[train_idx], X_prep[test_idx]
                    y_tr_str = y_str[train_idx]

                    le_sub = LabelEncoder()
                    y_tr_enc = le_sub.fit_transform(y_tr_str)

                    clf.fit(X_tr, y_tr_enc)
                    pred_enc = clf.predict(X_te)
                    y_pred_str.append(le_sub.inverse_transform(pred_enc)[0])

                y_pred_global = le_global.transform(np.array(y_pred_str))
                acc = accuracy_score(y_global, y_pred_global)
                cm = confusion_matrix(y_global, y_pred_global, labels=range(len(labels_str)))
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                with cols_grid[idx % 3]:
                    st.metric(label=name, value=f"{acc*100:.1f}% LOO Accuracy")
                    fig_cm = px.imshow(
                        cm_norm, x=labels_str, y=labels_str,
                        color_continuous_scale='Blues',
                        labels=dict(x="Predicted Class", y="Reference Class", color="Recall"),
                        text_auto=".2f",
                        title=f"{name} Confusion Matrix"
                    )
                    fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)

    # Tab 3: Canonical Discriminant Analysis
    with tab3:
        st.subheader("Linear Discriminant Analysis Canonical Projection")
        n_comp = min(3, len(labels_str) - 1)
        lda_proj = LinearDiscriminantAnalysis(n_components=n_comp)
        X_lda = lda_proj.fit_transform(X_prep, y_global)

        df_lda = pd.DataFrame(X_lda, columns=[f"LD{i+1}" for i in range(n_comp)])
        df_lda['Cultivar'] = y_str

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_2d = px.scatter(
                df_lda, x='LD1', y='LD2' if n_comp > 1 else 'LD1',
                color='Cultivar', symbol='Cultivar',
                title="2D LDA Canonical Separation",
                template="plotly_white"
            )
            fig_2d.update_traces(marker=dict(size=10, line=dict(width=0.8, color='DarkSlateGrey')))
            st.plotly_chart(fig_2d, use_container_width=True)

        with col_p2:
            if n_comp >= 3:
                fig_3d = px.scatter_3d(
                    df_lda, x='LD1', y='LD2', z='LD3',
                    color='Cultivar', symbol='Cultivar',
                    title="3D LDA Canonical Separation"
                )
                fig_3d.update_traces(marker=dict(size=6, line=dict(width=0.8, color='DarkSlateGrey')))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3D canonical projection requires at least four unique cultivar classes.")

else:
    st.info("Upload spectral dataset (.csv or .xlsx) via the sidebar to initiate analysis.")
