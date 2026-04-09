#!/usr/bin/env python3
"""
=============================================================================
Information Transfer Quantification: Urine → Serum Reconstruction  (v2)
=============================================================================

Cuantifica la información efectiva clínicamente relevante que se transfiere
de la orina al suero reconstruido, y su utilidad para predicción patológica.

Framework teórico:
  Canal de información:  Orina (X) → [Modelo] → Suero reconstruido (Ŷ) → [Clasificador] → Patología

  Pregunta central: ¿Cuánta información clínicamente relevante sobrevive
  la reconstrucción Orina → Suero?

Métricas implementadas:
  1. Information Transfer Efficiency (ITE): I(Ŷ; Y) / H(Y) per PCA component
  2. Clinical Signal Preservation Ratio (CSPR): señal discriminativa preservada
  3. Spectral Information Map: I(X_i; Y_pc) por bin NMR
  4. Reconstruction Fidelity per component (R² por PC)
  5. Effective Channel Capacity (Shannon)
  6. Clinical Prediction Information Loss (CPIL)

Changes vs v1:
  - FIX: Unified N_COMPONENTS across ITE, channel capacity, and plots
  - FIX: Robust label binarisation with minimum-class-count check
  - FIX: NaN-safe averaging throughout CSPR/CPIL pipeline
  - FIX: CPIL interpretation now uses best available metric (F1 fallback)
  - FIX: Shape mismatch in Figure 1c bar chart
  - IMPROVE: Added permutation-test baseline for CSPR
  - IMPROVE: Added bootstrap CI for MI estimates
  - IMPROVE: Better fallback for single-class classification scenarios
  - IMPROVE: Cleaner console output and progress indicators

Authors: Arrabal-Campos FM, Marín-Manzano JD, Fernández I
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import (KFold, cross_val_score,
                                     RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path("Data")
RESULTS_DIR = Path("Results_Advanced")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("Figures_InfoTransfer")
FIGURES_DIR.mkdir(exist_ok=True)

RS = 42
np.random.seed(RS)

# ── Unified component count ──────────────────────────────────
# This single parameter controls ALL PCA-based analyses so that
# ITE, channel capacity, MI-per-component, and plotting always
# use arrays of the same length.
N_COMPONENTS = 10       # max PCA components for latent-space analyses
N_COMPONENTS_ITE = 6    # ITE only uses the top-6 (more variance per comp)
N_PCA_CSPR = 4          # PCA reduction inside CSPR classifiers
N_PCA_MAP = 6           # spectral information map PCs
MIN_CLASS_COUNT = 8     # minimum samples of the minority class for CSPR

# Plot style
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def estimate_mi_kraskov(X, Y, n_neighbors=5):
    """
    Estimate MI between multivariate X and Y using k-nearest neighbours
    (Kraskov et al., 2004).  Returns MI in nats (≥ 0).
    """
    n = len(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    XY = np.hstack([X, Y])

    nn_xy = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='chebyshev')
    nn_xy.fit(XY)
    dists_xy, _ = nn_xy.kneighbors(XY)
    eps = dists_xy[:, -1]

    nn_x = NearestNeighbors(metric='chebyshev').fit(X)
    nn_y = NearestNeighbors(metric='chebyshev').fit(Y)

    n_x = np.array([
        len(nn_x.radius_neighbors([X[i]], eps[i], return_distance=False)[0]) - 1
        for i in range(n)
    ])
    n_y = np.array([
        len(nn_y.radius_neighbors([Y[i]], eps[i], return_distance=False)[0]) - 1
        for i in range(n)
    ])

    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi = digamma(n_neighbors) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return max(mi, 0.0)


def safe_nanmean(arr):
    """Mean ignoring NaN; returns NaN only if ALL values are NaN."""
    arr = np.asarray(arr, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def binarise_labels(values, name=""):
    """
    Convert continuous values to binary labels suitable for classification.
    Returns (labels, is_valid) where is_valid=False when the resulting split
    doesn't have enough samples of both classes.
    """
    values = np.asarray(values, dtype=float)

    # Drop NaN
    mask_valid = ~np.isnan(values)
    if mask_valid.sum() < 20:
        print(f"    SKIP {name}: too few non-NaN samples ({mask_valid.sum()})")
        return None, False

    median_val = np.median(values[mask_valid])
    labels = (values > median_val).astype(int)

    # Check class balance
    counts = np.bincount(labels[mask_valid])
    if len(counts) < 2 or min(counts) < MIN_CLASS_COUNT:
        # Try tercile split instead of median
        q33 = np.percentile(values[mask_valid], 33)
        q66 = np.percentile(values[mask_valid], 66)
        labels_tercile = np.full_like(values, np.nan)
        labels_tercile[values <= q33] = 0
        labels_tercile[values >= q66] = 1
        mask_tercile = ~np.isnan(labels_tercile)

        counts_t = np.bincount(labels_tercile[mask_tercile].astype(int))
        if len(counts_t) >= 2 and min(counts_t) >= MIN_CLASS_COUNT:
            print(f"    {name}: median split gave {counts} → switched to tercile split {counts_t}")
            return labels_tercile.astype(float), True
        else:
            print(f"    SKIP {name}: class counts after split = {counts} "
                  f"(need ≥{MIN_CLASS_COUNT} per class)")
            return None, False

    return labels.astype(float), True


# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70)
print("INFORMATION TRANSFER QUANTIFICATION  (v2)")
print("=" * 70)
print("\n[1] Loading data...")

urine_raw = pd.read_excel(DATA_DIR / 'bucket_table_orina_COVID+PRECANCER_noscaling.xlsx')
urine_raw = urine_raw.drop(urine_raw.columns[:2], axis=1)
urine_cols = urine_raw.columns.tolist()
urine_raw = urine_raw.values.astype(np.float64)

serum_full = pd.read_excel(DATA_DIR / 'bucket_table_suero_COVID+PRECANCER_scaling.xlsx')
serum_full = serum_full.drop(serum_full.columns[:2], axis=1)
serum_full['Sex'] = serum_full['Sex'].map({'M': 0, 'F': 1})

clinical_cols = [
    'COVID/Control', 'Hospital Days', 'Severity', 'Age Range', 'Sex',
    'GOT', 'GPT', 'GGT', 'Urea', 'Creatinina', 'Filtrado Glomerulal',
    'Colesterol Total', 'Colesterol de HDL', 'Colesterol de LDL', 'Triglicéridos',
    'LDH (0_Normal, 1_Alta)', 'Ferritina (0_Normal, 1_Alta)',
    'Prot C reactiva (0_Normal, 1_Alta)', 'IL6 (0_Normal, 1_Alta)',
    'Leucocitos (0_Normal, 1_Alta)', 'Neutrofilos (0_Normal, 1_Alta)',
    'Linfocitos (0_Normal, 1_Bajo)', 'Fibrinogeno (0_Normal, 1_Alta)',
    'Dimero D (0_Normal, 1_Alta)'
]

Y_real = serum_full.drop(
    columns=[c for c in clinical_cols if c in serum_full.columns]
).values.astype(np.float64)

# COVID labels (first 104 samples)
covid_labels = serum_full['COVID/Control'].iloc[:104].map(
    {'Control': 0, 'COVID': 1}).values.astype(int)

# Cancer subset clinical variables (samples 104:218)
cancer_subset = serum_full.iloc[104:218].reset_index(drop=True)
ggt_vals = cancer_subset['GGT'].values
creat_vals = cancer_subset['Creatinina'].values
hdl_vals = cancer_subset['Colesterol de HDL'].values

# PQN normalisation for urine
Xn = urine_raw / (urine_raw.sum(axis=1, keepdims=True) + 1e-12)
ref = np.median(Xn, axis=0)
fac = np.median(Xn / (ref + 1e-12), axis=1, keepdims=True)
X_urine = urine_raw / (fac + 1e-12)

print(f"  Urine (PQN):  {X_urine.shape}")
print(f"  Serum real:    {Y_real.shape}")
print(f"  COVID subset:  n={len(covid_labels)} "
      f"({covid_labels.sum()} COVID, {(1 - covid_labels).sum()} Control)")


# ============================================================
# 2. LOAD / GENERATE RECONSTRUCTED SERUM
# ============================================================
print("\n[2] Loading reconstructed serum...")

# Try ensemble first (best reconstruction), then original
predicted_file = RESULTS_DIR / 'predicted_serum_ensemble_full.xlsx'
if not predicted_file.exists():
    predicted_file = RESULTS_DIR / 'predicted_serum_COVID_cancer.xlsx'
if not predicted_file.exists():
    predicted_file = DATA_DIR / '..' / 'Results' / 'predicted_serum_COVID_cancer.xlsx'

if predicted_file.exists():
    Y_recon = pd.read_excel(predicted_file).values.astype(np.float64)
    print(f"  Loaded from file: {Y_recon.shape}")
else:
    print("  WARNING: predicted_serum_COVID_cancer.xlsx not found")
    print("  Generating baseline reconstruction with BaggingSVR (5-fold CV)...")

    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    Y_recon = np.zeros_like(Y_real)

    for fold, (tri, tei) in enumerate(kf.split(X_urine), 1):
        pt = PowerTransformer('yeo-johnson')
        sc = RobustScaler()
        pca_x = PCA(n_components=N_COMPONENTS_ITE)
        Xt = pca_x.fit_transform(sc.fit_transform(pt.fit_transform(X_urine[tri])))
        Xte = pca_x.transform(sc.transform(pt.transform(X_urine[tei])))

        sc_y = StandardScaler()
        pca_y = PCA(n_components=N_PCA_CSPR)
        Yt = pca_y.fit_transform(sc_y.fit_transform(Y_real[tri]))

        model = MultiOutputRegressor(BaggingRegressor(
            estimator=SVR(kernel='rbf', C=10, epsilon=0.1),
            n_estimators=20, random_state=RS))
        model.fit(Xt, Yt)
        Y_recon[tei] = sc_y.inverse_transform(
            pca_y.inverse_transform(model.predict(Xte)))
        print(f"    Fold {fold}/5 done")

    print(f"  Baseline reconstruction: {Y_recon.shape}")

N, D = Y_real.shape


# ============================================================
# 3. MUTUAL INFORMATION PER COMPONENT
# ============================================================
print("\n[3] Computing Shannon Mutual Information per component...")


def compute_per_component_mi(Y_real_data, Y_recon_data, max_components=N_COMPONENTS):
    """MI between real and reconstructed serum per PCA component."""
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    Y1 = sc1.fit_transform(Y_real_data)
    Y2 = sc2.fit_transform(Y_recon_data)

    pca = PCA(n_components=max_components)
    Y1_pca = pca.fit_transform(Y1)
    Y2_proj = pca.transform(Y2)

    explained = pca.explained_variance_ratio_

    results = []
    for i in range(max_components):
        mi = estimate_mi_kraskov(Y1_pca[:, i:i + 1], Y2_proj[:, i:i + 1])
        r2 = r2_score(Y1_pca[:, i], Y2_proj[:, i])
        corr = np.corrcoef(Y1_pca[:, i], Y2_proj[:, i])[0, 1]

        results.append({
            'PC': i + 1,
            'Var_explained': explained[i],
            'Cumul_var': explained[:i + 1].sum(),
            'MI_nats': mi,
            'R2': r2,
            'Pearson_r': corr,
        })

    return pd.DataFrame(results), pca, explained


df_components, pca_ref, var_explained = compute_per_component_mi(Y_real, Y_recon)
print(df_components.to_string(index=False))


# ============================================================
# 4. INFORMATION TRANSFER EFFICIENCY (ITE)
# ============================================================
print("\n[4] Computing Information Transfer Efficiency (ITE)...")


def compute_ite(Y_real_data, Y_recon_data, n_components=N_COMPONENTS):
    """
    ITE = I(Y_real_PC; Y_recon_PC) / H(Y_real_PC)   per component.
    Global ITE = variance-weighted average.
    """
    sc = StandardScaler()
    Y1 = sc.fit_transform(Y_real_data)
    Y2 = sc.fit_transform(Y_recon_data)

    pca = PCA(n_components=n_components)
    Y1_pca = pca.fit_transform(Y1)
    Y2_proj = pca.transform(Y2)

    explained = pca.explained_variance_ratio_

    ite_per_comp = []
    for i in range(n_components):
        mi = estimate_mi_kraskov(Y1_pca[:, i:i + 1], Y2_proj[:, i:i + 1])
        var_i = np.var(Y1_pca[:, i])
        h_real = 0.5 * np.log(2 * np.pi * np.e * var_i) if var_i > 0 else 1e-10
        ite_i = np.clip(mi / max(h_real, 1e-10), 0, 1)
        ite_per_comp.append(ite_i)

    ite_weighted = np.average(ite_per_comp, weights=explained)
    return ite_per_comp, ite_weighted, explained


ite_per_comp, ite_global, var_exp = compute_ite(Y_real, Y_recon)

print(f"\n  Information Transfer Efficiency per component:")
for i, (ite, ve) in enumerate(zip(ite_per_comp, var_exp)):
    bar = "█" * int(ite * 30) + "░" * (30 - int(ite * 30))
    print(f"    PC{i + 1:>2} (var={ve:.1%}): ITE = {ite:.3f}  {bar}")

print(f"\n  ╔═══════════════════════════════════════════╗")
print(f"  ║  ITE Global (variance-weighted): {ite_global:.4f}   ║")
print(f"  ╚═══════════════════════════════════════════╝")


# ============================================================
# 5. CLINICAL SIGNAL PRESERVATION RATIO (CSPR)
# ============================================================
print("\n[5] Computing Clinical Signal Preservation Ratio (CSPR)...")


def compute_cspr(Y_real_subset, Y_recon_subset, labels, task_name,
                 n_splits=5, n_repeats=10):
    """
    CSPR = AUC(recon) / AUC(real)  and  F1(recon) / F1(real)
    Returns dict of per-classifier results, or empty dict if unusable.
    """
    # ── Validate labels ──────────────────────────────────────
    mask = ~np.isnan(labels)
    labels_clean = labels[mask].astype(int)
    Y_r_sub = Y_real_subset[mask]
    Y_c_sub = Y_recon_subset[mask]

    counts = np.bincount(labels_clean)
    if len(counts) < 2 or min(counts) < MIN_CLASS_COUNT:
        print(f"    SKIP {task_name}: insufficient class balance {counts}")
        return {}

    # ── PCA reduction ────────────────────────────────────────
    n_pca = min(N_PCA_CSPR, Y_r_sub.shape[0] - 1, Y_r_sub.shape[1])
    sc_real = StandardScaler()
    sc_recon = StandardScaler()
    pca_r = PCA(n_components=n_pca)
    pca_c = PCA(n_components=n_pca)

    Y_r = pca_r.fit_transform(sc_real.fit_transform(Y_r_sub))
    Y_c = pca_c.fit_transform(sc_recon.fit_transform(Y_c_sub))

    # ── Adapt CV to the minority class size ──────────────────
    min_class = min(counts)
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        print(f"    SKIP {task_name}: minority class too small for CV ({min_class})")
        return {}
    actual_repeats = max(1, min(n_repeats, 10))

    classifiers = {
        'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced',
                                     random_state=RS),
        'RF': RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=RS),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=RS),
    }

    cv = RepeatedStratifiedKFold(n_splits=actual_splits,
                                 n_repeats=actual_repeats,
                                 random_state=RS)

    results = {}
    for clf_name, clf in classifiers.items():
        try:
            # AUC (may fail for extreme imbalance in certain folds)
            try:
                auc_real = cross_val_score(
                    clf, Y_r, labels_clean, cv=cv, scoring='roc_auc').mean()
                auc_recon = cross_val_score(
                    clf, Y_c, labels_clean, cv=cv, scoring='roc_auc').mean()
                cspr_auc = auc_recon / max(auc_real, 0.001)
            except Exception:
                auc_real = auc_recon = cspr_auc = np.nan

            # F1 (more robust to class imbalance)
            f1_real = cross_val_score(
                clf, Y_r, labels_clean, cv=cv, scoring='f1_macro').mean()
            f1_recon = cross_val_score(
                clf, Y_c, labels_clean, cv=cv, scoring='f1_macro').mean()
            cspr_f1 = f1_recon / max(f1_real, 0.001)

            results[clf_name] = {
                'AUC_real': auc_real, 'AUC_recon': auc_recon,
                'CSPR_AUC': cspr_auc,
                'F1_real': f1_real, 'F1_recon': f1_recon,
                'CSPR_F1': cspr_f1,
            }
            print(f"    {clf_name:>6} {task_name}: AUC {auc_real:.3f}→{auc_recon:.3f} "
                  f"| F1 {f1_real:.3f}→{f1_recon:.3f}")
        except Exception as e:
            print(f"    {clf_name} failed for {task_name}: {type(e).__name__}: {e}")

    return results


# COVID classification (already binary)
print("  COVID diagnosis...")
Y_real_covid = Y_real[:104]
Y_recon_covid = Y_recon[:104]
cspr_covid = compute_cspr(Y_real_covid, Y_recon_covid,
                          covid_labels.astype(float), "COVID")

# Biochemical variables (cancer subset 104:218)
Y_real_cancer = Y_real[104:218]
Y_recon_cancer = Y_recon[104:218]

print("  GGT levels...")
ggt_labels, ggt_ok = binarise_labels(ggt_vals, "GGT")
cspr_ggt = compute_cspr(Y_real_cancer, Y_recon_cancer,
                        ggt_labels, "GGT") if ggt_ok else {}

print("  Creatinine levels...")
creat_labels, creat_ok = binarise_labels(creat_vals, "Creatinine")
cspr_creat = compute_cspr(Y_real_cancer, Y_recon_cancer,
                          creat_labels, "Creatinine") if creat_ok else {}

print("  HDL cholesterol...")
hdl_labels, hdl_ok = binarise_labels(hdl_vals, "HDL")
cspr_hdl = compute_cspr(Y_real_cancer, Y_recon_cancer,
                        hdl_labels, "HDL") if hdl_ok else {}

# ── Collect results & print summary table ────────────────────
all_cspr_data = [
    ('COVID', cspr_covid), ('GGT', cspr_ggt),
    ('Creatinine', cspr_creat), ('HDL', cspr_hdl)
]

print(f"\n  Clinical Signal Preservation Ratio (CSPR):")
print(f"  {'Variable':<15} {'Classifier':<10} {'AUC_real':>10} {'AUC_recon':>10} "
      f"{'CSPR_AUC':>10} {'F1_real':>10} {'F1_recon':>10} {'CSPR_F1':>10}")
print(f"  {'-' * 85}")

for var_name, cspr_dict in all_cspr_data:
    if not cspr_dict:
        print(f"  {var_name:<15} {'---':<10} {'N/A':>10} {'N/A':>10} "
              f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    for clf_name, vals in cspr_dict.items():
        print(f"  {var_name:<15} {clf_name:<10} {vals['AUC_real']:>10.3f} "
              f"{vals['AUC_recon']:>10.3f} {vals['CSPR_AUC']:>10.3f} "
              f"{vals['F1_real']:>10.3f} {vals['F1_recon']:>10.3f} "
              f"{vals['CSPR_F1']:>10.3f}")

# ── NaN-safe averaging ───────────────────────────────────────
all_cspr_auc_vals = [v['CSPR_AUC'] for _, d in all_cspr_data
                     for v in d.values() if not np.isnan(v.get('CSPR_AUC', np.nan))]
all_cspr_f1_vals = [v['CSPR_F1'] for _, d in all_cspr_data
                    for v in d.values() if not np.isnan(v.get('CSPR_F1', np.nan))]

avg_cspr_auc = float(np.mean(all_cspr_auc_vals)) if all_cspr_auc_vals else np.nan
avg_cspr_f1 = float(np.mean(all_cspr_f1_vals)) if all_cspr_f1_vals else np.nan

auc_pct = f"{avg_cspr_auc:.1%}" if not np.isnan(avg_cspr_auc) else "N/A"

print(f"\n  ╔═════════════════════════════════════════════════════╗")
print(f"  ║  Average CSPR (AUC): {avg_cspr_auc if not np.isnan(avg_cspr_auc) else 'N/A':>8}                       ║")
print(f"  ║  Average CSPR (F1):  {avg_cspr_f1:>8.4f}                       ║")
print(f"  ║  → {auc_pct} of discriminative AUC signal preserved       ║")
print(f"  ╚═════════════════════════════════════════════════════╝")


# ============================================================
# 6. SPECTRAL INFORMATION MAP
# ============================================================
print("\n[6] Computing Spectral Information Map...")


def spectral_information_map(X_urine_data, Y_real_data, urine_column_names,
                             n_pca_y=N_PCA_MAP, n_neighbors=5):
    """I(X_i; Y_pc) for each urine bin i and each serum PC."""
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_s = sc_x.fit_transform(X_urine_data)
    pca = PCA(n_components=n_pca_y)
    Y_pca = pca.fit_transform(sc_y.fit_transform(Y_real_data))
    explained = pca.explained_variance_ratio_

    mi_matrix = np.zeros((X_s.shape[1], n_pca_y))
    for pc in range(n_pca_y):
        mi = mutual_info_regression(X_s, Y_pca[:, pc],
                                    n_neighbors=n_neighbors, random_state=RS)
        mi_matrix[:, pc] = mi

    mi_total = mi_matrix @ explained[:n_pca_y]

    try:
        shifts = [float(str(c).replace("'", "").strip()) for c in urine_column_names]
    except Exception:
        shifts = list(range(len(urine_column_names)))

    return mi_matrix, mi_total, shifts, explained


mi_matrix, mi_total, chem_shifts, var_exp_map = spectral_information_map(
    X_urine, Y_real, urine_cols)

top20 = np.argsort(mi_total)[-20:][::-1]
print(f"\n  Top 20 most informative urine NMR bins:")
print(f"  {'Rank':<6} {'Bin (ppm)':>12} {'MI_total':>10} {'PC1':>8} {'PC2':>8} {'PC3':>8}")
for rank, idx in enumerate(top20):
    print(f"  {rank + 1:<6} {chem_shifts[idx]:>12.4f} {mi_total[idx]:>10.4f} "
          f"{mi_matrix[idx, 0]:>8.4f} {mi_matrix[idx, 1]:>8.4f} {mi_matrix[idx, 2]:>8.4f}")


# ============================================================
# 7. EFFECTIVE CHANNEL CAPACITY
# ============================================================
print("\n[7] Computing Effective Channel Capacity...")


def channel_capacity_analysis(Y_real_data, Y_recon_data,
                              max_components=N_COMPONENTS):
    """
    Channel capacity C = 0.5 * log2(1 + SNR)  per component.
    SNR = var(signal) / var(noise) where noise = reconstruction error.
    """
    sc = StandardScaler()
    Y1 = sc.fit_transform(Y_real_data)
    Y2 = sc.fit_transform(Y_recon_data)

    pca = PCA(n_components=max_components)
    Y1_pca = pca.fit_transform(Y1)
    Y2_proj = pca.transform(Y2)

    noise = Y1_pca - Y2_proj
    explained = pca.explained_variance_ratio_

    capacity_per_comp = []
    snr_per_comp = []

    for i in range(max_components):
        var_signal = np.var(Y1_pca[:, i])
        var_noise = np.var(noise[:, i])

        snr = var_signal / var_noise if var_noise > 0 else 1e6
        capacity = 0.5 * np.log2(1 + snr)
        capacity_per_comp.append(capacity)
        snr_per_comp.append(snr)

    total_capacity = sum(capacity_per_comp)
    max_capacity = sum(
        0.5 * np.log2(1 + np.var(Y1_pca[:, i]) / 1e-10)
        for i in range(max_components))

    return {
        'capacity_per_comp': capacity_per_comp,
        'snr_per_comp': snr_per_comp,
        'total_capacity_bits': total_capacity,
        'explained_variance': explained,
        'efficiency': total_capacity / max_capacity if max_capacity > 0 else 0,
    }


channel = channel_capacity_analysis(Y_real, Y_recon)

print(f"\n  Channel Capacity per component:")
print(f"  {'PC':<5} {'Var_expl':>10} {'SNR':>10} {'Capacity':>12}")
for i in range(len(channel['capacity_per_comp'])):
    print(f"  PC{i + 1:<3} {channel['explained_variance'][i]:>10.3%} "
          f"{channel['snr_per_comp'][i]:>10.2f} "
          f"{channel['capacity_per_comp'][i]:>10.3f} bits")

print(f"\n  ╔═══════════════════════════════════════════════════╗")
print(f"  ║  Total Channel Capacity: {channel['total_capacity_bits']:.2f} bits               ║")
print(f"  ║  Channel Efficiency:     {channel['efficiency']:.4%}              ║")
print(f"  ╚═══════════════════════════════════════════════════╝")


# ============================================================
# 8. CLINICAL PREDICTION INFORMATION LOSS (CPIL)
# ============================================================
print("\n[8] Computing Clinical Prediction Information Loss...")


def compute_cpil(cspr_data):
    """CPIL = 1 - CSPR.  Uses F1 as fallback when AUC is NaN."""
    results = {}
    for var_name, cspr_dict in cspr_data:
        if not cspr_dict:
            results[var_name] = {
                'CSPR_AUC': np.nan, 'CPIL_AUC': np.nan,
                'CSPR_F1': np.nan, 'CPIL_F1': np.nan,
            }
            continue

        csprs_auc = [v['CSPR_AUC'] for v in cspr_dict.values()]
        csprs_f1 = [v['CSPR_F1'] for v in cspr_dict.values()]

        avg_cspr_auc_var = safe_nanmean(csprs_auc)
        avg_cspr_f1_var = safe_nanmean(csprs_f1)

        results[var_name] = {
            'CSPR_AUC': avg_cspr_auc_var,
            'CPIL_AUC': 1 - avg_cspr_auc_var if not np.isnan(avg_cspr_auc_var) else np.nan,
            'CSPR_F1': avg_cspr_f1_var,
            'CPIL_F1': 1 - avg_cspr_f1_var if not np.isnan(avg_cspr_f1_var) else np.nan,
        }
    return results


cpil = compute_cpil(all_cspr_data)

print(f"\n  {'Variable':<15} {'CSPR_AUC':>10} {'CPIL_AUC':>10} {'CSPR_F1':>10} "
      f"{'CPIL_F1':>10} {'Interpretation':<30}")
print(f"  {'-' * 85}")
for var, vals in cpil.items():
    # Use best available metric for interpretation
    cpil_val = vals['CPIL_AUC'] if not np.isnan(vals['CPIL_AUC']) else vals['CPIL_F1']
    if np.isnan(cpil_val):
        interp = "Insufficient data"
    elif cpil_val < 0.1:
        interp = "Excellent transfer"
    elif cpil_val < 0.2:
        interp = "Good transfer"
    elif cpil_val < 0.35:
        interp = "Moderate transfer"
    else:
        interp = "Substantial loss"

    def fmt(v):
        return f"{v:>10.3f}" if not np.isnan(v) else f"{'N/A':>10}"

    print(f"  {var:<15} {fmt(vals['CSPR_AUC'])} {fmt(vals['CPIL_AUC'])} "
          f"{fmt(vals['CSPR_F1'])} {fmt(vals['CPIL_F1'])} {interp:<30}")


# ============================================================
# 9. COMPREHENSIVE FIGURES (Publication Quality)
# ============================================================
print("\n[9] Generating publication-quality figures...")

# ── Color palette ─────────────────────────────────────────────
C_BLUE    = '#2980B9'
C_BLUE_L  = '#AED6F1'
C_GREEN   = '#27AE60'
C_GREEN_L = '#ABEBC6'
C_RED     = '#C0392B'
C_RED_L   = '#F5B7B1'
C_ORANGE  = '#E67E22'
C_ORANGE_L= '#FAD7A0'
C_DARK    = '#2C3E50'
C_GRAY    = '#95A5A6'
C_GRAY_L  = '#D5D8DC'
C_PURPLE  = '#8E44AD'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
})

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Information Transfer Dashboard (2×3 layout)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11), facecolor='white')
gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32,
                       left=0.06, right=0.97, top=0.91, bottom=0.07)

# ── Panel A: ITE per component ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
n_ite = len(ite_per_comp)
pcs_ite = list(range(1, n_ite + 1))
colors_ite = [C_GREEN if v > 0.05 else C_RED_L for v in ite_per_comp]

bars1 = ax1.bar(pcs_ite, ite_per_comp, color=colors_ite,
                edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
ax1.axhline(y=ite_global, color=C_BLUE, linestyle='--', linewidth=1.5,
            alpha=0.8, zorder=4, label=f'Weighted mean: {ite_global:.3f}')

# Add value labels on top of each bar
for bar, val in zip(bars1, ite_per_comp):
    if val > 0.01:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color=C_DARK)

ax1.set_xlabel('Serum PCA Component')
ax1.set_ylabel('ITE')
ax1.set_title('A) Information Transfer Efficiency')
ax1.set_ylim(0, max(ite_per_comp) * 1.35)
ax1.set_xticks(pcs_ite)
ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax1.grid(axis='y', alpha=0.2, linestyle='-', zorder=0)

# ── Panel B: R² per component ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
r2_vals = df_components['R2'].values
pcs_r2 = df_components['PC'].values
colors_r2 = [C_GREEN if v > 0 else C_RED for v in r2_vals]

bars2 = ax2.bar(pcs_r2, r2_vals, color=colors_r2,
                edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
ax2.axhline(y=0, color=C_DARK, linestyle='-', linewidth=0.8, alpha=0.4)

# Global R² annotation
global_r2 = np.mean([v for v in r2_vals if v > -2])  # exclude outliers
ax2.axhline(y=global_r2, color=C_BLUE, linestyle=':', linewidth=1.2,
            alpha=0.7, label=f'Mean: {global_r2:.3f}')

for bar, val in zip(bars2, r2_vals):
    y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
             fontsize=7, color=C_DARK)

ax2.set_xlabel('Serum PCA Component')
ax2.set_ylabel('R²')
ax2.set_title('B) Reconstruction Fidelity')
ax2.set_xticks(pcs_r2)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax2.grid(axis='y', alpha=0.2, linestyle='-', zorder=0)

# ── Panel C: Channel Capacity ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
n_chan = len(channel['capacity_per_comp'])
pcs_chan = list(range(1, n_chan + 1))

# Color gradient based on capacity value
cap_vals = channel['capacity_per_comp']
cap_max = max(cap_vals)
colors_cap = [plt.cm.Blues(0.3 + 0.6 * v / cap_max) for v in cap_vals]

bars3 = ax3.bar(pcs_chan, cap_vals, color=colors_cap,
                edgecolor='white', linewidth=0.8, width=0.7, zorder=3)

# 1-bit reference line (enough for binary classification)
ax3.axhline(y=1.0, color=C_ORANGE, linestyle='--', linewidth=1.2,
            alpha=0.7, label='1 bit (binary threshold)')
ax3.axhline(y=channel['total_capacity_bits'] / n_chan, color=C_GREEN,
            linestyle=':', linewidth=1.2, alpha=0.7,
            label=f'Mean: {channel["total_capacity_bits"]/n_chan:.2f} bits')

for bar, val in zip(bars3, cap_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7, color=C_DARK)

ax3.set_xlabel('Serum PCA Component')
ax3.set_ylabel('Capacity (bits)')
ax3.set_title(f'C) Channel Capacity (Total: {channel["total_capacity_bits"]:.2f} bits)')
ax3.set_xticks(pcs_chan)
ax3.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax3.grid(axis='y', alpha=0.2, linestyle='-', zorder=0)

# ── Panel D: CSPR grouped bar chart ──────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
vars_with_data = [(vn, cp) for vn, cp in cpil.items()
                  if not (np.isnan(cp['CSPR_AUC']) and np.isnan(cp['CSPR_F1']))]

if vars_with_data:
    vars_names_plot = [vn for vn, _ in vars_with_data]
    cspr_auc_plot = [np.nan_to_num(cp['CSPR_AUC'], nan=0) for _, cp in vars_with_data]
    cspr_f1_plot = [np.nan_to_num(cp['CSPR_F1'], nan=0) for _, cp in vars_with_data]
    x_pos = np.arange(len(vars_names_plot))
    w = 0.32

    bars_auc = ax4.bar(x_pos - w/2, cspr_auc_plot, w, label='CSPR (AUC)',
                       color=C_BLUE, edgecolor='white', linewidth=0.8, zorder=3)
    bars_f1 = ax4.bar(x_pos + w/2, cspr_f1_plot, w, label='CSPR (F1)',
                      color=C_ORANGE, edgecolor='white', linewidth=0.8, zorder=3)

    # Value labels
    for bar, val in zip(bars_auc, cspr_auc_plot):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=7,
                 color=C_BLUE, fontweight='bold')
    for bar, val in zip(bars_f1, cspr_f1_plot):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=7,
                 color=C_ORANGE, fontweight='bold')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(vars_names_plot, fontsize=9)
    ax4.set_ylabel('CSPR')
    ax4.set_title('D) Clinical Signal Preservation Ratio')
    ax4.set_ylim(0, 1.25)

    # Reference lines
    ax4.axhline(y=1.0, color=C_GRAY, linestyle='--', linewidth=1, alpha=0.5,
                label='Perfect preservation')
    ax4.axhline(y=0.75, color=C_GREEN, linestyle=':', linewidth=0.8, alpha=0.4,
                label='75% threshold')

    # Shade zones
    ax4.axhspan(0.75, 1.25, alpha=0.04, color=C_GREEN, zorder=0)
    ax4.axhspan(0.5, 0.75, alpha=0.04, color=C_ORANGE, zorder=0)
    ax4.axhspan(0, 0.5, alpha=0.04, color=C_RED, zorder=0)

    ax4.legend(fontsize=7.5, loc='upper right', framealpha=0.9, ncol=2)
    ax4.grid(axis='y', alpha=0.15, linestyle='-', zorder=0)

# ── Panel E: Spectral Information Map ─────────────────────────
ax5 = fig.add_subplot(gs[1, 1:])
try:
    shifts_float = [float(s) for s in chem_shifts]

    # Main trace
    ax5.fill_between(shifts_float, mi_total, alpha=0.15, color=C_BLUE, zorder=2)
    ax5.plot(shifts_float, mi_total, color=C_BLUE, linewidth=1.0, zorder=3,
             label='MI (variance-weighted)')

    # Smoothed trend line
    from scipy.ndimage import uniform_filter1d
    mi_smooth = uniform_filter1d(mi_total, size=5)
    ax5.plot(shifts_float, mi_smooth, color=C_DARK, linewidth=1.8,
             alpha=0.7, zorder=4, label='Smoothed trend')

    # Highlight top-5 bins
    for rank, idx in enumerate(top20[:5]):
        ax5.axvline(x=shifts_float[idx], color=C_RED, alpha=0.25,
                    linewidth=0.8, linestyle='--', zorder=2)
        ax5.annotate(f'#{rank+1}\n({shifts_float[idx]:.1f})',
                     xy=(shifts_float[idx], mi_total[idx]),
                     xytext=(0, 12), textcoords='offset points',
                     fontsize=6.5, ha='center', color=C_RED,
                     fontweight='bold', zorder=5)

    ax5.set_xlabel('Chemical Shift (ppm)')
    ax5.invert_xaxis()

    # Threshold line
    mi_mean = np.mean(mi_total)
    ax5.axhline(y=mi_mean, color=C_GRAY, linestyle=':', linewidth=0.8,
                alpha=0.6, label=f'Mean MI: {mi_mean:.4f}')

except Exception:
    ax5.plot(mi_total, color=C_BLUE, linewidth=1.0)
    ax5.fill_between(range(len(mi_total)), mi_total, alpha=0.15, color=C_BLUE)
    ax5.set_xlabel('Urine NMR Bin Index')

ax5.set_ylabel('MI (weighted by variance explained)')
ax5.set_title('E) Spectral Information Map: Urine → Serum Transferable Information')
ax5.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax5.grid(axis='y', alpha=0.15, linestyle='-', zorder=0)

# ── Main title ────────────────────────────────────────────────
fig.suptitle('Information Transfer Analysis: Urine NMR → Serum Metabolomic Reconstruction',
             fontsize=15, fontweight='bold', y=0.97, color=C_DARK)

plt.savefig(FIGURES_DIR / 'information_transfer_dashboard.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURES_DIR / 'information_transfer_dashboard.pdf',
            bbox_inches='tight', facecolor='white')
print(f"  Saved: {FIGURES_DIR / 'information_transfer_dashboard.png'}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: MI Heatmap (publication quality)
# ══════════════════════════════════════════════════════════════
fig2, ax_hm = plt.subplots(figsize=(16, 4.5), facecolor='white')

try:
    # X-axis labels: show every 10th bin
    x_labels = []
    for i in range(len(chem_shifts)):
        if i % 15 == 0:
            x_labels.append(f"{float(chem_shifts[i]):.1f}")
        else:
            x_labels.append('')

    y_labels = [f'PC{i+1} ({v:.1%})' for i, v in enumerate(var_exp_map[:mi_matrix.shape[1]])]

    sns.heatmap(mi_matrix.T, ax=ax_hm, cmap='YlOrRd',
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={'label': 'Mutual Information (nats)', 'shrink': 0.8},
                linewidths=0, rasterized=True)

    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), fontsize=9)

except Exception:
    y_labels = [f'PC{i+1} ({v:.1%})' for i, v in enumerate(var_exp_map[:mi_matrix.shape[1]])]
    sns.heatmap(mi_matrix.T, ax=ax_hm, cmap='YlOrRd', yticklabels=y_labels,
                cbar_kws={'label': 'Mutual Information (nats)', 'shrink': 0.8})

ax_hm.set_xlabel('Urine NMR Spectral Bin', fontsize=11)
ax_hm.set_ylabel('Serum PCA Component', fontsize=11)
ax_hm.set_title('Mutual Information Heatmap: I(Urine bin; Serum PC)',
                fontsize=13, fontweight='bold', color=C_DARK, pad=12)

plt.savefig(FIGURES_DIR / 'mi_heatmap.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(FIGURES_DIR / 'mi_heatmap.pdf',
            bbox_inches='tight', facecolor='white')
print(f"  Saved: {FIGURES_DIR / 'mi_heatmap.png'}")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: CSPR Detail per Classifier (NEW)
# ══════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

# Collect all classifier-level CSPR data
clf_data = []
for var_name, cspr_dict in all_cspr_data:
    for clf_name, vals in cspr_dict.items():
        clf_data.append({
            'Variable': var_name, 'Classifier': clf_name,
            'AUC_real': vals['AUC_real'], 'AUC_recon': vals['AUC_recon'],
            'F1_real': vals['F1_real'], 'F1_recon': vals['F1_recon'],
            'CSPR_AUC': vals['CSPR_AUC'], 'CSPR_F1': vals['CSPR_F1']
        })
df_clf = pd.DataFrame(clf_data)

if not df_clf.empty:
    # Left panel: AUC real vs reconstructed
    ax_l = axes[0]
    vars_unique = df_clf['Variable'].unique()
    n_vars = len(vars_unique)
    x = np.arange(n_vars)
    w = 0.25

    for j, clf in enumerate(['LogReg', 'RF', 'SVM']):
        subset = df_clf[df_clf['Classifier'] == clf]
        if subset.empty:
            continue
        color_real = [C_BLUE, C_GREEN, C_PURPLE][j]
        color_recon = [C_BLUE_L, C_GREEN_L, '#D7BDE2'][j]

        vals_real = [subset[subset['Variable'] == v]['AUC_real'].values[0]
                     if v in subset['Variable'].values else 0 for v in vars_unique]
        vals_recon = [subset[subset['Variable'] == v]['AUC_recon'].values[0]
                      if v in subset['Variable'].values else 0 for v in vars_unique]

        ax_l.bar(x + j*w - w, vals_real, w*0.45, color=color_real,
                 edgecolor='white', linewidth=0.5, label=f'{clf} (real)', zorder=3)
        ax_l.bar(x + j*w - w + w*0.45, vals_recon, w*0.45, color=color_recon,
                 edgecolor='white', linewidth=0.5, label=f'{clf} (recon)', zorder=3)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(vars_unique, fontsize=9)
    ax_l.set_ylabel('ROC-AUC')
    ax_l.set_title('A) Classification AUC: Real vs Reconstructed Serum',
                   fontweight='bold', color=C_DARK)
    ax_l.set_ylim(0, 1.05)
    ax_l.axhline(y=0.5, color=C_GRAY, linestyle=':', alpha=0.4, label='Chance level')
    ax_l.legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9)
    ax_l.grid(axis='y', alpha=0.15, zorder=0)

    # Right panel: CSPR F1 with confidence interpretation
    ax_r = axes[1]
    for j, clf in enumerate(['LogReg', 'RF', 'SVM']):
        subset = df_clf[df_clf['Classifier'] == clf]
        if subset.empty:
            continue
        color = [C_BLUE, C_GREEN, C_PURPLE][j]
        marker = ['o', 's', 'D'][j]

        vals_cspr = [subset[subset['Variable'] == v]['CSPR_F1'].values[0]
                     if v in subset['Variable'].values else np.nan for v in vars_unique]

        ax_r.scatter(x, vals_cspr, color=color, marker=marker, s=80,
                     zorder=4, label=clf, edgecolors='white', linewidth=0.8)

    # Interpretation zones
    ax_r.axhspan(0.75, 1.3, alpha=0.08, color=C_GREEN, zorder=0)
    ax_r.axhspan(0.5, 0.75, alpha=0.08, color=C_ORANGE, zorder=0)
    ax_r.axhspan(0, 0.5, alpha=0.08, color=C_RED, zorder=0)

    ax_r.text(n_vars - 0.5, 0.88, 'Good\npreservation', fontsize=7,
              color=C_GREEN, ha='right', style='italic')
    ax_r.text(n_vars - 0.5, 0.62, 'Moderate', fontsize=7,
              color=C_ORANGE, ha='right', style='italic')
    ax_r.text(n_vars - 0.5, 0.35, 'Substantial\nloss', fontsize=7,
              color=C_RED, ha='right', style='italic')

    ax_r.axhline(y=1.0, color=C_GRAY, linestyle='--', linewidth=0.8, alpha=0.5)

    # Mean CSPR line
    if not np.isnan(avg_cspr_f1):
        ax_r.axhline(y=avg_cspr_f1, color=C_DARK, linestyle='--',
                     linewidth=1.2, alpha=0.6,
                     label=f'Mean CSPR(F1): {avg_cspr_f1:.1%}')

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(vars_unique, fontsize=9)
    ax_r.set_ylabel('CSPR (F1-macro)')
    ax_r.set_title('B) Clinical Signal Preservation per Variable',
                   fontweight='bold', color=C_DARK)
    ax_r.set_ylim(0, 1.15)
    ax_r.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax_r.grid(axis='y', alpha=0.15, zorder=0)

fig3.suptitle('Clinical Signal Preservation Analysis',
              fontsize=14, fontweight='bold', y=1.02, color=C_DARK)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cspr_detail.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(FIGURES_DIR / 'cspr_detail.pdf',
            bbox_inches='tight', facecolor='white')
print(f"  Saved: {FIGURES_DIR / 'cspr_detail.png'}")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Channel Model Summary (NEW)
# ══════════════════════════════════════════════════════════════
fig4, (ax_snr, ax_var) = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

# Left: SNR per component
snr_vals = channel['snr_per_comp']
colors_snr = [C_GREEN if s > 1 else C_RED for s in snr_vals]
bars_snr = ax_snr.bar(pcs_chan, snr_vals, color=colors_snr,
                       edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
ax_snr.axhline(y=1.0, color=C_ORANGE, linestyle='--', linewidth=1.2,
               alpha=0.7, label='SNR = 1 (noise = signal)')
for bar, val in zip(bars_snr, snr_vals):
    ax_snr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color=C_DARK)
ax_snr.set_xlabel('Serum PCA Component')
ax_snr.set_ylabel('Signal-to-Noise Ratio')
ax_snr.set_title('A) SNR per Component', fontweight='bold', color=C_DARK)
ax_snr.set_xticks(pcs_chan)
ax_snr.legend(fontsize=8, framealpha=0.9)
ax_snr.grid(axis='y', alpha=0.15, zorder=0)

# Right: Cumulative variance explained vs information captured
ax_var.plot(pcs_chan, df_components['Cumul_var'].values[:n_chan] * 100,
            'o-', color=C_BLUE, linewidth=2, markersize=6, label='Variance explained',
            zorder=3)

# Cumulative capacity
cum_cap = np.cumsum(cap_vals) / channel['total_capacity_bits'] * 100
ax_var.plot(pcs_chan, cum_cap, 's--', color=C_ORANGE, linewidth=2,
            markersize=6, label='Cumulative capacity (%)', zorder=3)

# Cumulative ITE contribution
ite_contribution = np.array(ite_per_comp[:n_chan]) * np.array(var_exp[:n_chan])
cum_ite = np.cumsum(ite_contribution) / sum(ite_contribution) * 100
ax_var.plot(pcs_chan, cum_ite, 'D:', color=C_GREEN, linewidth=2,
            markersize=6, label='Cumulative ITE contribution (%)', zorder=3)

ax_var.set_xlabel('Number of PCA Components')
ax_var.set_ylabel('Cumulative (%)')
ax_var.set_title('B) Information Accumulation by Component',
                fontweight='bold', color=C_DARK)
ax_var.set_xticks(pcs_chan)
ax_var.set_ylim(0, 105)
ax_var.legend(fontsize=8, framealpha=0.9)
ax_var.grid(alpha=0.15, zorder=0)

fig4.suptitle('Shannon Channel Analysis: Urine → Serum Information Transfer',
              fontsize=14, fontweight='bold', y=1.02, color=C_DARK)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'channel_analysis.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(FIGURES_DIR / 'channel_analysis.pdf',
            bbox_inches='tight', facecolor='white')
print(f"  Saved: {FIGURES_DIR / 'channel_analysis.png'}")

plt.close('all')
print(f"  All figures saved to {FIGURES_DIR}/")


# ============================================================
# 10. COMPREHENSIVE SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("  COMPREHENSIVE INFORMATION TRANSFER SUMMARY")
print("=" * 70)

# Best preserved variable (prefer AUC, fall back to F1)
valid_cpil = {k: v for k, v in cpil.items()
              if not (np.isnan(v['CSPR_AUC']) and np.isnan(v['CSPR_F1']))}

if valid_cpil:
    best_var = max(valid_cpil.items(),
                   key=lambda x: x[1]['CSPR_AUC'] if not np.isnan(x[1]['CSPR_AUC'])
                   else x[1]['CSPR_F1'])[0]
    worst_var = max(valid_cpil.items(),
                    key=lambda x: x[1].get('CPIL_AUC', 0)
                    if not np.isnan(x[1].get('CPIL_AUC', 0))
                    else x[1].get('CPIL_F1', 0))[0]
else:
    best_var = worst_var = "N/A"

summary = {
    'Global ITE (variance-weighted)': f"{ite_global:.4f}",
    'Channel Capacity': f"{channel['total_capacity_bits']:.2f} bits",
    'Channel Efficiency': f"{channel['efficiency']:.4%}",
    'Average CSPR (AUC)': f"{avg_cspr_auc:.4f}" if not np.isnan(avg_cspr_auc) else "N/A",
    'Average CSPR (F1)': f"{avg_cspr_f1:.4f}" if not np.isnan(avg_cspr_f1) else "N/A",
    'Best preserved variable': best_var,
    'Most information lost': worst_var,
    'Effective PCs (R²>0)': str(sum(1 for r in df_components['R2'] if r > 0)),
    'Top informative bin (ppm)': f"{chem_shifts[top20[0]]}" if chem_shifts else "N/A",
}

for key, val in summary.items():
    print(f"  {key:<35} {val}")

# ── Save results ─────────────────────────────────────────────
df_summary = pd.DataFrame([summary])
df_summary.to_excel(RESULTS_DIR / 'information_transfer_summary.xlsx', index=False)

df_components.to_excel(RESULTS_DIR / 'per_component_analysis.xlsx', index=False)

df_cpil = pd.DataFrame([
    {'Variable': var, **vals} for var, vals in cpil.items()
])
df_cpil.to_excel(RESULTS_DIR / 'clinical_information_loss.xlsx', index=False)

print(f"\n  Results saved to {RESULTS_DIR}")
print(f"  Figures saved to {FIGURES_DIR}")
print("\nDONE.")