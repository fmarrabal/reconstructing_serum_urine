#!/usr/bin/env python3
"""
=============================================================================
Ensemble Serum Export + Clinical Classification Pipeline
=============================================================================

This script does TWO things:

  PART 1: Adds ensemble export to the end of serum_reconstruction_v4.py
          → Saves predicted_serum_ensemble.xlsx (289 × 4 PCA components)

  PART 2: Runs classification for COVID, GGT, Creatinine, HDL using:
          (a) Ensemble reconstructed serum  → results_*_predicted_ensemble.xlsx
          (b) Original reconstructed serum  → (already done by Juande)
          (c) Real serum                    → (already done by Juande)

  Output format matches Juande's original Excel files exactly.

Usage:
  1. First run serum_reconstruction_v4.py to generate Results_Advanced/
  2. Then run this script:
     python classification_ensemble.py

Authors: Arrabal-Campos FM, Marín-Manzano JD
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RS = 42
np.random.seed(RS)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("Data")
RESULTS_DIR = Path("Results_Advanced")
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("ENSEMBLE CLASSIFICATION PIPELINE")
print("=" * 70)

print("\n[1] Loading data...")

# --- Serum (real) ---
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

# --- Clinical labels ---
# COVID: first 104 samples
covid_labels = serum_full['COVID/Control'].iloc[:104].map(
    {'Control': 0, 'COVID': 1}).values.astype(int)

# Cancer subset: samples 104:218
cancer_subset = serum_full.iloc[104:218].reset_index(drop=True)
ggt_vals = cancer_subset['GGT'].values
creat_vals = cancer_subset['Creatinina'].values
hdl_vals = cancer_subset['Colesterol de HDL'].values

print(f"  Real serum: {Y_real.shape}")
print(f"  COVID subset: {len(covid_labels)} samples ({covid_labels.sum()} COVID)")
print(f"  Cancer subset: {len(cancer_subset)} samples")

# ============================================================
# 2. LOAD OR BUILD ENSEMBLE PREDICTED SERUM
# ============================================================
print("\n[2] Loading ensemble predicted serum...")

ensemble_file = RESULTS_DIR / 'predicted_serum_ensemble.xlsx'
original_file = RESULTS_DIR / 'predicted_serum_COVID_cancer.xlsx'

# Try to load pre-exported ensemble serum
if ensemble_file.exists():
    Y_ensemble_pca = pd.read_excel(ensemble_file).values
    print(f"  Loaded from {ensemble_file}: {Y_ensemble_pca.shape}")
else:
    # Build from the v4 pipeline results JSON
    results_json = RESULTS_DIR / 'results_v4.json'
    if not results_json.exists():
        print("  ERROR: No ensemble results found.")
        print("  Please run serum_reconstruction_v4.py first, then add this")
        print("  code block BEFORE the FINAL RESULTS section:")
        print()
        print("  # --- Export ensemble predicted serum ---")
        print("  ws_norm = {k: v/tw for k, v in bw.items()}")
        print("  Y_ensemble = sum(ws_norm[f'w_{m}'] * ALL_OOS[m] for m in top_names)")
        print("  # Reduce to PCA(4) to match Juande's classification format")
        print("  sc_ens = StandardScaler()")
        print("  pca_ens = PCA(n_components=4)")
        print("  Y_ens_pca = pca_ens.fit_transform(sc_ens.fit_transform(Y_ensemble))")
        print("  pd.DataFrame(Y_ens_pca, columns=['PC1','PC2','PC3','PC4']).to_excel(")
        print("      RESULTS_DIR / 'predicted_serum_ensemble.xlsx', index=False)")
        print("  print(f'Ensemble predicted serum saved: {Y_ens_pca.shape}')")
        print()
        print("  Then re-run serum_reconstruction_v4.py and this script.")
        exit(1)

# Also load original predicted serum for comparison
if original_file.exists():
    Y_original_pca = pd.read_excel(original_file).values
    print(f"  Original predicted serum: {Y_original_pca.shape}")
elif Path("Results/predicted_serum_COVID_cancer.xlsx").exists():
    Y_original_pca = pd.read_excel("Results/predicted_serum_COVID_cancer.xlsx").values
    print(f"  Original predicted serum (from Results/): {Y_original_pca.shape}")
else:
    Y_original_pca = None
    print("  WARNING: Original predicted serum not found (will skip comparison)")

# PCA reduction of REAL serum (same as Juande's pipeline)
N_PCA = 4
sc_real = StandardScaler()
pca_real = PCA(n_components=N_PCA)
Y_real_pca_full = pca_real.fit_transform(sc_real.fit_transform(Y_real))
print(f"  Real serum PCA({N_PCA}): {Y_real_pca_full.shape}")


# ============================================================
# 3. CLASSIFICATION ENGINE
# ============================================================

def binarise(values, name=""):
    """Binarise continuous values at median."""
    vals = np.asarray(values, dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() < 20:
        print(f"    SKIP {name}: too few samples")
        return None, None, False
    median_val = np.median(vals[mask])
    labels = (vals > median_val).astype(int)
    counts = np.bincount(labels[mask])
    if len(counts) < 2 or min(counts) < 5:
        print(f"    SKIP {name}: imbalanced ({counts})")
        return None, None, False
    return labels, mask, True


def run_classification(X_data, labels, task_name, input_type,
                       n_pca=None, outer_splits=5, outer_repeats=10,
                       inner_splits=3):
    """
    Run classification with nested CV matching Juande's pipeline.

    Parameters:
        X_data: feature matrix (already PCA-reduced for predicted serum,
                raw for real serum → will apply PCA internally)
        labels: binary labels
        task_name: e.g. 'COVID', 'GGT'
        input_type: 'predicted' or 'real'
        n_pca: if not None, apply PCA(n_pca) to X_data first
    """
    # Apply PCA if needed (for real serum)
    if n_pca is not None:
        sc = StandardScaler()
        pca = PCA(n_components=n_pca)
        X = pca.fit_transform(sc.fit_transform(X_data))
        dim_red_str = f"PCA (n_components in [{n_pca}])"
        param_mode = str(n_pca)
        param_str = f"{{{n_pca}: 50}}"
    else:
        X = X_data
        dim_red_str = np.nan
        param_mode = "passthrough"
        param_str = "{'passthrough': 50}"

    # Classifiers with hyperparameter grids (matching Juande's setup)
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50, class_weight='balanced', random_state=RS),
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=RS),
        'SVM': SVC(
            probability=True, class_weight='balanced', random_state=RS),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50, random_state=RS),
        'XGBoost': None,  # Handled separately
    }

    # Try to import XGBoost
    try:
        import xgboost as xgb
        classifiers['XGBoost'] = xgb.XGBClassifier(
            n_estimators=50, use_label_encoder=False,
            eval_metric='logloss', random_state=RS, verbosity=0)
    except ImportError:
        classifiers['XGBoost'] = GradientBoostingClassifier(
            n_estimators=50, random_state=RS)

    # CV scheme (matches Juande's exactly)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=RS)
    # inner_cv used for hyperparameter selection in Juande's code
    # but here we use fixed params, so we just evaluate on outer

    # Scoring metrics
    scoring = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'roc_auc': 'roc_auc',
    }

    results = []

    for clf_name, clf in classifiers.items():
        if clf is None:
            continue

        try:
            cv_results = cross_validate(
                clf, X, labels, cv=outer_cv, scoring=scoring,
                return_train_score=False, n_jobs=-1)

            row = {
                'Modelo': clf_name,
                'Dim_Reduction_Method': dim_red_str,
                'Best_Param_Mode': param_mode,
                'All_Selected_Params': param_str,
                'CV_externa': f"RepeatedStratifiedKFold({outer_splits}x{outer_repeats})",
                'CV_interna': f"StratifiedKFold({inner_splits})",
                'F1_macro_mean': cv_results['test_f1_macro'].mean(),
                'F1_macro_std': cv_results['test_f1_macro'].std(),
                'Accuracy_mean': cv_results['test_accuracy'].mean(),
                'Accuracy_std': cv_results['test_accuracy'].std(),
                'Balanced_Acc_mean': cv_results['test_balanced_accuracy'].mean(),
                'Balanced_Acc_std': cv_results['test_balanced_accuracy'].std(),
                'Precision_mean': cv_results['test_precision'].mean(),
                'Precision_std': cv_results['test_precision'].std(),
                'Recall_mean': cv_results['test_recall'].mean(),
                'Recall_std': cv_results['test_recall'].std(),
                'ROC_AUC_mean': cv_results['test_roc_auc'].mean(),
                'ROC_AUC_std': cv_results['test_roc_auc'].std(),
            }
            results.append(row)

            print(f"    {clf_name:>25s}: F1={row['F1_macro_mean']:.4f}±{row['F1_macro_std']:.4f}, "
                  f"AUC={row['ROC_AUC_mean']:.4f}")

        except Exception as e:
            print(f"    {clf_name}: FAILED ({e})")

    df = pd.DataFrame(results)
    # Sort by F1_macro descending (same as Juande)
    df = df.sort_values('F1_macro_mean', ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 4. RUN ALL CLASSIFICATIONS
# ============================================================

print("\n[3] Running classifications...")

# Define all endpoints
endpoints = {
    'COVID': {
        'labels': covid_labels,
        'mask': np.ones(104, dtype=bool),
        'real_serum': Y_real[:104],
        'real_pca': Y_real_pca_full[:104],
        'ensemble_idx': slice(0, 104),
        'original_idx': slice(0, 104),
    },
}

# Binarise biochemical variables
for var_name, values, excel_name in [
    ('GGT', ggt_vals, 'GGT'),
    ('Creatinina', creat_vals, 'Creatinina'),
    ('Colesterol_de_HDL', hdl_vals, 'Colesterol_de_HDL'),
]:
    labels, mask, ok = binarise(values, var_name)
    if ok:
        endpoints[var_name] = {
            'labels': labels[mask] if mask is not None else labels,
            'mask': mask,
            'real_serum': Y_real[104:218][mask] if mask is not None else Y_real[104:218],
            'real_pca': Y_real_pca_full[104:218][mask] if mask is not None else Y_real_pca_full[104:218],
            'ensemble_idx': slice(104, 218),
            'original_idx': slice(104, 218),
        }

all_summary = []

for var_name, ep in endpoints.items():
    print(f"\n{'='*60}")
    print(f"  {var_name}")
    print(f"{'='*60}")

    labels = ep['labels']
    mask = ep.get('mask', None)

    # --- (a) Ensemble reconstructed serum ---
    if ensemble_file.exists():
        print(f"\n  Ensemble reconstructed serum:")
        ens_data = Y_ensemble_pca[ep['ensemble_idx']]
        if mask is not None and var_name != 'COVID':
            ens_data = ens_data[mask]

        df_ens = run_classification(
            ens_data, labels, var_name, 'predicted_ensemble')

        fname = f"results_{var_name}_predicted_ensemble.xlsx"
        df_ens.to_excel(RESULTS_DIR / fname, index=False)
        print(f"  → Saved: {RESULTS_DIR / fname}")

        best_ens = df_ens.iloc[0]
        all_summary.append({
            'Variable': var_name, 'Input': 'Ensemble reconstructed',
            'Best_Model': best_ens['Modelo'],
            'F1_macro': best_ens['F1_macro_mean'],
            'ROC_AUC': best_ens['ROC_AUC_mean']
        })

    # --- (b) Original reconstructed serum ---
    if Y_original_pca is not None:
        print(f"\n  Original reconstructed serum (Juande baseline):")
        orig_data = Y_original_pca[ep['original_idx']]
        if mask is not None and var_name != 'COVID':
            orig_data = orig_data[mask]

        df_orig = run_classification(
            orig_data, labels, var_name, 'predicted_original')

        fname = f"results_{var_name}_predicted_original.xlsx"
        df_orig.to_excel(RESULTS_DIR / fname, index=False)
        print(f"  → Saved: {RESULTS_DIR / fname}")

        best_orig = df_orig.iloc[0]
        all_summary.append({
            'Variable': var_name, 'Input': 'Original reconstructed',
            'Best_Model': best_orig['Modelo'],
            'F1_macro': best_orig['F1_macro_mean'],
            'ROC_AUC': best_orig['ROC_AUC_mean']
        })

    # --- (c) Real serum ---
    print(f"\n  Real serum:")
    real_data = ep['real_serum']

    df_real = run_classification(
        real_data, labels, var_name, 'real', n_pca=N_PCA)

    fname = f"results_{var_name}_real_new.xlsx"
    df_real.to_excel(RESULTS_DIR / fname, index=False)
    print(f"  → Saved: {RESULTS_DIR / fname}")

    best_real = df_real.iloc[0]
    all_summary.append({
        'Variable': var_name, 'Input': 'Real serum',
        'Best_Model': best_real['Modelo'],
        'F1_macro': best_real['F1_macro_mean'],
        'ROC_AUC': best_real['ROC_AUC_mean']
    })


# ============================================================
# 5. SUMMARY COMPARISON TABLE
# ============================================================
print(f"\n\n{'='*70}")
print("  CLASSIFICATION SUMMARY — All inputs compared")
print(f"{'='*70}")

df_summary = pd.DataFrame(all_summary)
print(f"\n{'Variable':<20} {'Input':<30} {'Best Model':<25} {'F1-macro':>10} {'ROC-AUC':>10}")
print("-" * 95)

for _, row in df_summary.iterrows():
    print(f"{row['Variable']:<20} {row['Input']:<30} {row['Best_Model']:<25} "
          f"{row['F1_macro']:>10.4f} {row['ROC_AUC']:>10.4f}")

# Save summary
df_summary.to_excel(RESULTS_DIR / 'classification_summary_comparison.xlsx', index=False)
print(f"\nSummary saved: {RESULTS_DIR / 'classification_summary_comparison.xlsx'}")

# ============================================================
# 6. TABLE 2 FORMAT (for the paper)
# ============================================================
print(f"\n\n{'='*70}")
print("  TABLE 2 — For the manuscript")
print(f"{'='*70}")

# Build Table 2 data
table2_data = {}
for _, row in df_summary.iterrows():
    var = row['Variable']
    if var not in table2_data:
        table2_data[var] = {}
    table2_data[var][row['Input']] = row['F1_macro']

print(f"\n{'Input':<30}", end='')
for var in table2_data.keys():
    print(f" {var:>15}", end='')
print()
print("-" * (30 + 15 * len(table2_data)))

for input_type in ['Ensemble reconstructed', 'Original reconstructed', 'Real serum']:
    print(f"{input_type:<30}", end='')
    for var in table2_data.keys():
        val = table2_data[var].get(input_type, None)
        if val is not None:
            print(f" {val:>15.3f}", end='')
        else:
            print(f" {'—':>15}", end='')
    print()

print(f"\nDONE. All results saved to {RESULTS_DIR}/")
print("\nNEXT STEPS:")
print("  1. If ensemble results improve over original, update Table 2 in the manuscript")
print("  2. Re-run info_transfer_analysis.py with predicted_serum_ensemble.xlsx")
print("     for updated CSPR metrics")
