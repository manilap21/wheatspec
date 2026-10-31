# WheatSpec: Chemometric & Machine Learning Phenotyping of Wheat Flour via ATR-FTIR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

**WheatSpec** is an open-source chemometrics and machine learning pipeline developed at **The University of Western Australia (UWA)** for rapid, non-destructive wheat end-product quality prediction and cultivar identification using Attenuated Total Reflectance Fourier-Transform Infrared (ATR-FTIR) spectroscopy.

Building upon the foundational chemometrics framework of Kelly et al. (2025), WheatSpec provides reproducible pipelines for:
1. **Dough Rheology Prediction:** Leave-One-Out Cross-Validation (LOO-CV) regression benchmarks across 17 spectral windows (7 Original + 10 Novel sub-regions) for six Farinograph and Extensograph parameters ($n = 163$).
2. **Cultivar Authentication:** Multi-class classification tournament comparing regularized linear (Shrinkage LDA, Linear SVM) and non-linear (Random Forest, XGBoost) models across 10 commercial Australian varieties ($n = 115$).
3. **Cross-Seasonal Transferability:** Zero-shot evaluation of historical calibration models against an independent single-season harvest ($n = 19$) to evaluate genotype-by-environment ($G \times E$) spectral stability.
4. **WheatSpec Web Interface:** A lightweight Streamlit graphical portal for interactive spectral exploration, regional masking, and real-time inference.

---

## Repository Structure

```
wheatspec/
├── README.md                          # Documentation and project overview
├── LICENSE                            # MIT License
├── requirements.txt                   # Environment dependencies
├── app.py                             # Interactive WheatSpec Streamlit application
└── scripts/
    ├── 01_spectral_preprocessing.py        # Preprocessing spectrum figure generator (Figure 1)
    ├── 02_merge_rheology_master.py         # Historical rheology compiler (n = 163)
    ├── 03_dough_rheology_loo.py            # Master 6-trait LOO-CV rheology suite (Figs 2 & 3, Table 2)
    ├── 04_variety_classification_loo.py    # 10-cultivar classification tournament (Figure 4, Table V1)
    └── 05_external_transfer_evaluation.py  # Independent harvest transfer evaluation (Figure 5, Table V2)
```

**Installation & Setup**

```bash
# 1. Clone repository
git clone [https://github.com/manuliperera/wheatspec.git](https://github.com/manuliperera/wheatspec.git)
cd wheatspec

# 2. Virtual environment (optional)
python3.11 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```
**Data Availability**
The historical wheat reference datasets contain proprietary commercial breeding composites owned by Australian Grain Technologies (AGT). Anonymized spectral arrays and cross-validation prediction outputs are available from the corresponding author upon reasonable request. All analysis code, regional masks, and model architectures remain fully open-source.


**License**
Distributed under the MIT License.
