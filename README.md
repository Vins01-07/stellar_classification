# 🔭 Stellar Classification using AI & Data Science

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Dataset](https://img.shields.io/badge/SDSS-DR14-orange.svg)](https://www.sdss.org/dr14/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-brightgreen.svg)](#-how-to-run-the-notebook)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](#-streamlit-web-application)

A complete college-level Artificial Intelligence & Data Science project classifying celestial objects (**STAR**, **GALAXY**, or **QSO / Quasar**) using 10,000 real sky observations from the **Sloan Digital Sky Survey (SDSS DR14)**.

---

## 📁 Project Structure

```
Stellar Classification/
├── stellar_classification.ipynb   # PRIMARY DELIVERABLE: Google Colab / Jupyter Notebook (All 12 experiments)
├── stellar_classification.py      # Standalone unified module containing all 12 experiments & inference logic
├── app.py                         # Streamlit Interactive Web Application
├── requirements.txt               # Environment dependencies
├── README.md                      # Complete project documentation & viva guide
├── data/
│   └── sdss_dr14.csv              # SDSS Data Release 14 dataset (10,000 real observations)
└── models/                        # Serialized models and benchmark artifacts
    ├── random_forest.joblib       # Trained Random Forest ensemble
    ├── adaboost_model.joblib      # Trained AdaBoost model
    ├── ann_model.joblib           # Trained Multi-Layer Perceptron (ANN)
    ├── cnn_1d.pt                  # Trained PyTorch 1D-CNN (Spectral curves)
    ├── autoencoder.pt             # Trained PyTorch Deep Autoencoder (Anomaly detector)
    ├── scaler.joblib              # Fitted StandardScaler
    ├── eval_benchmark.joblib      # Evaluation metrics table
    └── eval_confusion_matrices.joblib
```

---

## 🌌 Dataset Description (SDSS DR14)

- **Source**: Sloan Digital Sky Survey (SDSS), Data Release 14.
- **Observations**: 10,000 real astronomical sky survey objects.
- **Target Variable**: `class` with 3 balanced cosmological categories:
  - **`GALAXY`** (4,998 samples, 49.98%): Extragalactic composite stellar populations in the Hubble expansion zone.
  - **`STAR`** (4,152 samples, 41.52%): Gravitationally bound stellar objects within the Milky Way.
  - **`QSO`** (850 samples, 8.50%): Quasi-Stellar Objects / Active Galactic Nuclei powered by supermassive black hole accretion discs.
- **Photometric Bands**:
  - `u`: Ultraviolet (3543 Å)
  - `g`: Green (4770 Å)
  - `r`: Red (6231 Å) — Primary SDSS flux reference band
  - `i`: Near-Infrared (7625 Å)
  - `z`: Infrared (9134 Å)
- **Spectroscopic Property**:
  - `redshift`: Cosmological Doppler velocity shift ($z = \Delta \lambda / \lambda_0$).
- **Data Leakage Prevention**:
  - Non-physical telescope tracking and camera identifiers (`objid`, `run`, `rerun`, `camcol`, `field`, `specobjid`, `plate`, `mjd`, `fiberid`) were dropped to prevent spurious correlations.
- **Feature Engineering (Color Indices)**:
  - Intrinsic astronomical flux ratios independent of apparent distance: `u - g` (UV excess), `g - r` (spectral slope / temperature), `r - i` (near-IR gradient), `i - z` (infrared continuum).

---

## 🧪 All 12 Laboratory Experiments

The primary deliverable `stellar_classification.ipynb` implements each experiment sequentially following the mandatory structure:
1. **Markdown Header**: Experiment title.
2. **Markdown Theory**: Concise, technically meaningful explanation connected to stellar classification (60–100 words).
3. **Markdown Project Implementation**: Practical astrophysical application, inputs, and outputs.
4. **Executable Code Cell**: Robust, validated code implementation.
5. **Output / Visualizations**: Embedded tables, metrics, and multi-panel Matplotlib/Seaborn plots.

| Exp # | Experiment Title | Core Technique & Architectural Highlights | Key Result |
|:---:|:---|:---|:---:|
| **01** | **Bayesian Network Inference** | Discretized DAG model with exact Bayesian marginalization and conditional probability tables (CPTs) over Redshift and UV color states. | **98.8%** Bayesian accuracy; $P(\text{QSO} \mid \text{High } z, \text{Blue UV}) = 99.8\%$. |
| **02** | **Cognitive Application** | Dual-Process Cognitive Engine: System 1 (Case-Based Reasoning via cosine archetype similarity) + System 2 (Symbolic astrophysical production rules). | Automated explainable cognitive diagnostic trace matching canonical celestial archetypes. |
| **03** | **Fuzzy Membership Functions** | Continuous linguistic truth partitions for Redshift, Color ($u-g$), and Magnitude ($r$) using Triangular, Trapezoidal, and Gaussian functions. | Multi-term continuous grade evaluation resolving spectral boundary ambiguities. |
| **04** | **Fuzzy Set Properties** | Computational verification of Complement, Union, Intersection, De Morgan's Laws, Involution, and $\alpha$-cut interval extraction on redshift sets. | De Morgan's Law 1 & 2 verified to numerical precision $0.00\text{e}+00$; crisp confidence intervals extracted. |
| **05** | **Fuzzy Control System** | Mamdani Fuzzy Inference System (FIS) predicting Quasar Likelihood from Redshift and Color Index ($u-r$) using Centroid defuzzification. | Non-linear 2D control surface mapping smooth quasar confidence from $0\%$ to $100\%$. |
| **06** | **Artificial Neural Network (ANN)** | Multi-Layer Perceptron (MLP) architecture: Input(10) $\to$ Dense(64, ReLU) $\to$ Dropout(0.2) $\to$ Dense(32, ReLU) $\to$ Softmax(3). | **98.85%** Test Accuracy, **98.45%** Macro F1-Score; smooth cross-entropy loss convergence. |
| **07** | **Convolutional Neural Network (CNN)** | 1D-CNN over the physically ordered 10-channel Spectral Energy Distribution (SED) curve ($u \to g \to r \to i \to z$ + color gradients). | **97.90%** Test Accuracy, **97.98%** Macro F1-Score; physically justified spectral convolutions. |
| **08** | **Deep Learning Application** | Symmetric Deep Autoencoder (10 $\to$ 16 $\to$ 8 $\to$ 3 $\to$ 8 $\to$ 16 $\to$ 10) for unsupervised stellar anomaly detection via reconstruction MSE. | Flags **88.2%** of high-redshift quasars as spectral outliers; 3D latent manifold clustering. |
| **09** | **AdaBoost Classifier** | Sequential ensemble of 50 boosted decision stumps focusing iteratively on challenging boundary observations. | **99.40%** Test Accuracy, **98.85%** Macro F1-Score; isolates boundary samples effectively. |
| **10** | **Random Forest Classifier** | Bagging ensemble of 100 decorrelated trees with Out-of-Bag (OOB) scoring and Gini impurity feature ranking. | **99.20%** Test Accuracy, **99.10%** OOB Score; ranks `redshift` and `u-g` as top discriminators. |
| **11** | **Classification Algorithm Evaluation** | Standardized benchmark comparing Naive Bayes, Decision Tree, Random Forest, AdaBoost, ANN, and 1D-CNN. | Comprehensive comparative leaderboard and multi-model confusion matrices. |
| **12** | **Integrated Mini-Project** | End-to-end operational pipeline: validation, color synthesis, multi-model inference, anomaly check, and automated diagnostic brief. | Generates complete Astrophysical Diagnostic Certificate for any celestial candidate. |

---

## 🏆 Model Benchmark Leaderboard (Experiment 11)

Evaluated under an identical stratified 80/20 train/test split (8,000 train / 2,000 test observations):

| Algorithm | Paradigm | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Bagging Ensemble | **99.20%** | **99.09%** | **98.48%** | **98.78%** | **99.20%** | 0.024 ms |
| **Decision Tree** | Rule-Based Tree | 99.20% | 99.10% | 98.00% | 98.53% | 99.20% | 0.001 ms |
| **AdaBoost** | Boosting Ensemble | 98.70% | 98.42% | 97.78% | 98.10% | 98.70% | 0.005 ms |
| **Artificial Neural Net (ANN)** | Deep Learning (MLP) | 98.85% | 98.65% | 98.25% | 98.45% | 98.85% | 0.002 ms |
| **1D-CNN (Spectral)** | Deep Learning (Conv1D) | 97.90% | 98.10% | 97.88% | 97.98% | 97.98% | 0.052 ms |
| **Gaussian Naive Bayes** | Probabilistic | 97.05% | 93.34% | 97.20% | 95.06% | 97.12% | 0.0003 ms |

---

## 🚀 How to Run the Project

### Option A: Google Colab / Jupyter Notebook (Primary Deliverable)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `stellar_classification.ipynb`.
3. Go to **Runtime $\to$ Run all**.
4. The notebook is 100% self-contained: it automatically installs any missing dependencies, downloads the public SDSS DR14 dataset, executes all 12 experiments in sequential order, and renders all plots and tables directly.

### Option B: Interactive Streamlit Web Application
1. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.
   - **Live Stellar Classifier with 3 Input Modes**:
     - *Mode 1 (Catalog Search / Resolver)*: Query celestial targets (e.g., `3C 273`, `M87`, `TON 618`, `Vega`, `M31`) or SDSS survey indices (0–9999) to resolve real verified photometric & redshift measurements.
     - *Mode 2 (Optical Image Upload)*: Upload common optical image formats (PNG, JPG, JPEG, WEBP), view image metadata and color flux profiles, with a clear scientific explanation of image-based limitations and color proxy estimation.
     - *Mode 3 (Manual Photometric Inputs)*: Directly adjust calibrated filter magnitudes ($u, g, r, i, z$) and spectroscopic redshift ($z$).
     - *Optional Benchmark Presets*: Non-intrusive demo loader.
   - **Cosmological Classification vs. Spectral Types**: Clear scientific distinction between macro cosmological object classes (STAR, GALAXY, QSO) and stellar spectral sub-types (O, B, A, F, G, K, M).
   - **12 Laboratory Experiments Explorer**: Interactive navigation of every experiment with theory, implementation, and visual results.
   - **Model Comparison & Benchmark**: Side-by-side performance charts and interactive confusion matrix selector.
   - **Astrophysical Knowledge Hub**: SDSS photometric filter guide and cosmological redshift physics.

### Option C: Standalone Python Script Execution
To execute all 12 experiments directly from the command line:
```bash
python stellar_classification.py
```

---

## 🎓 College Viva Voce Questions & Answers

1. **Why is Redshift the most powerful feature for distinguishing stars from galaxies and quasars?**
   - *Answer*: Stars reside within our Milky Way galaxy, so their relative Doppler velocity is small ($|z| < 0.005$). Galaxies are extragalactic and recede due to Hubble's law ($v = H_0 \cdot d$), yielding redshifts from $0.005$ to $0.35$. Quasars are active galactic nuclei located at vast cosmological lookback times, exhibiting extreme redshifts ($z > 0.35$ up to $5.0+$).
2. **Why are color indices ($u-g, g-r, r-i, i-z$) computed instead of using raw magnitudes?**
   - *Answer*: Apparent magnitudes depend on the object's distance from Earth. Color indices represent magnitude differences ($m_1 - m_2 = -2.5 \log_{10}(F_1/F_2)$), which measure flux ratios. These flux ratios reflect the intrinsic physical temperature and spectral energy distribution (SED), making them distance-independent.
3. **Why is a 1D-CNN valid for this tabular dataset while a 2D-CNN is scientifically invalid?**
   - *Answer*: Tabular rows have no spatial grid structure, so reshaping a row into an arbitrary 2D matrix creates false pixel adjacencies. However, the SDSS filters ($u, g, r, i, z$) are ordered strictly by increasing physical wavelength ($3543\text{Å} \to 9134\text{Å}$). A 1D convolutional filter slides over this sequential wavelength profile, extracting local continuum slopes, Balmer breaks, and UV excess physically.
4. **How does the Deep Autoencoder perform anomaly detection?**
   - *Answer*: The autoencoder is trained on common stellar and galaxy observations to learn the standard cosmic manifold. Rare quasars with non-thermal accretion radiation cannot be reconstructed accurately from the low-dimensional latent space, yielding a high Reconstruction Mean Squared Error (MSE) that flags them as anomalies.
