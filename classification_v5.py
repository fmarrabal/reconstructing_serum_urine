#!/usr/bin/env python3
"""
=============================================================================
Clinical classification from real serum, reconstructed serum and urine (v5)
=============================================================================
The same protocol is applied to the three input types, so that differences
between inputs reflect their information content and not the modelling:

  nested : RandomizedSearchCV (30 iterations, F1-macro, inner StratifiedKFold(3))
           over classifier hyperparameters and number of PCA components (2-14),
           inside RepeatedStratifiedKFold(5 x 10)  [scheme of the v4 urine notebook]
  fixed  : fixed hyperparameters and PCA(4) inside RepeatedStratifiedKFold(5 x 10)
           [scheme of the v4 serum classification]

All preprocessing (median imputation, PQN for urine, StandardScaler, PCA) is part
of the pipeline and is therefore fitted inside each training partition.
The Clinical Signal Preservation Ratio (CSPR) is derived from these results:
CSPR = metric(reconstructed serum) / metric(real serum), per classifier.

Usage:
  python classification_v5.py --protocol both \
      --recon Results_v5/predicted_serum_ensemble_full.xlsx --out Results_v5/classification
=============================================================================
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RS = 42

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', default='both', choices=['nested', 'fixed', 'both'])
parser.add_argument('--recon', default='Results_v5/predicted_serum_ensemble_full.xlsx')
parser.add_argument('--out', default='Results_v5/classification')
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--endpoints', default='all', help='comma-separated subset, for quick tests')
parser.add_argument('--classifiers', default='all', help='comma-separated subset, for quick tests')
parser.add_argument('--inputs', default='real,recon,urine',
                    help="inputs to compute now; results are cached per endpoint/input/classifier")
ARGS = parser.parse_args()
OUT = Path(ARGS.out)
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('Data')

# ============================================================
# 1. DATA
# ============================================================
urine = pd.read_excel(DATA_DIR / 'bucket_table_orina_COVID+PRECANCER_noscaling.xlsx')
X_urine = urine.drop(urine.columns[:2], axis=1).values.astype(float)

serum = pd.read_excel(DATA_DIR / 'bucket_table_suero_COVID+PRECANCER_scaling.xlsx')
serum = serum.drop(serum.columns[:2], axis=1)
first_clin = serum.columns.get_loc('COVID/Control')
X_real = serum.iloc[:, :first_clin].values.astype(float)
clin = serum.iloc[:, first_clin:]

X_recon = pd.read_excel(ARGS.recon).values.astype(float)
assert X_recon.shape == X_real.shape, (X_recon.shape, X_real.shape)

INPUTS = {'real': X_real, 'recon': X_recon, 'urine': X_urine}

ENDPOINTS = {}
covid_rows = np.where(clin['COVID/Control'].notna().values)[0]
ENDPOINTS['COVID'] = (covid_rows,
                      clin['COVID/Control'].iloc[covid_rows].map({'Control': 0, 'COVID': 1}).values.astype(int))
for col, name in [('GGT', 'GGT'), ('Creatinina', 'Creatinine'), ('Colesterol de HDL', 'HDL')]:
    v = pd.to_numeric(clin[col], errors='coerce').values
    rows = np.where(~np.isnan(v))[0]
    ENDPOINTS[name] = (rows, (v[rows] > np.median(v[rows])).astype(int))
if ARGS.endpoints != 'all':
    ENDPOINTS = {k: v for k, v in ENDPOINTS.items() if k in ARGS.endpoints.split(',')}
for k, (rows, y) in ENDPOINTS.items():
    print(f"  {k:<11s} n={len(rows):3d}  class counts={np.bincount(y).tolist()}")


# ============================================================
# 2. PIPELINES
# ============================================================
class PQNScaler(BaseEstimator, TransformerMixin):
    """Probabilistic Quotient Normalisation; reference fitted on training data."""
    def fit(self, X, y=None):
        Xn = X / X.sum(axis=1, keepdims=True)
        self.ref_ = np.median(Xn, axis=0)
        self.ref_[self.ref_ == 0] = 1.0
        return self

    def transform(self, X):
        Xn = X / X.sum(axis=1, keepdims=True)
        f = np.median(Xn / self.ref_, axis=1, keepdims=True)
        f[(f == 0) | ~np.isfinite(f)] = 1.0
        return X / f


def make_pipeline(input_name, clf, n_pca):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if input_name == 'urine':
        steps.append(('pqn', PQNScaler()))
    # StandardScaler for every input: RobustScaler inflates serum bins whose IQR is
    # tiny relative to their s.d. and degrades the PCA (checked October 2026).
    steps += [('scaler', StandardScaler()), ('reduce_dim', PCA(n_components=n_pca)),
              ('clf', clf)]
    return Pipeline(steps)


CLASSIFIERS = {
    'nested': {
        'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RS),
                                {'clf__C': [0.01, 0.1, 1.0]}),
        'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=RS),
                          {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5],
                           'clf__min_samples_leaf': [2, 3]}),
        'SVM': (SVC(class_weight='balanced', probability=True, gamma='scale', random_state=RS),
                {'clf__C': [0.01, 0.1, 1.0], 'clf__kernel': ['linear', 'rbf']}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=RS),
                              {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.05],
                               'clf__max_depth': [2, 3]}),
        'XGBoost': (XGBClassifier(eval_metric='logloss', subsample=0.8, colsample_bytree=0.8,
                                  random_state=RS, verbosity=0, n_jobs=1),
                    {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.05],
                     'clf__max_depth': [2, 3]}),
    },
    'fixed': {
        'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RS), None),
        'Random Forest': (RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=RS), None),
        'SVM': (SVC(class_weight='balanced', probability=True, random_state=RS), None),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=50, random_state=RS), None),
        'XGBoost': (XGBClassifier(n_estimators=50, eval_metric='logloss', random_state=RS,
                                  verbosity=0, n_jobs=1), None),
    },
}
if ARGS.classifiers != 'all':
    keep = ARGS.classifiers.split(',')
    CLASSIFIERS = {p: {k: v for k, v in d.items() if k in keep} for p, d in CLASSIFIERS.items()}
PCA_GRID = list(range(2, 15))
FIXED_PCA = 4


def run_fold(protocol, input_name, clf, grid, X, y, tr, te):
    if protocol == 'nested':
        pipe = make_pipeline(input_name, clone(clf), PCA_GRID[0])
        full_grid = [dict(grid, reduce_dim=[PCA(n_components=n)]) for n in PCA_GRID]
        search = RandomizedSearchCV(pipe, full_grid, n_iter=30, scoring='f1_macro',
                                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RS),
                                    refit=True, n_jobs=1, random_state=RS)
        search.fit(X[tr], y[tr])
        model = search.best_estimator_
        n_pc = model.named_steps['reduce_dim'].n_components
    else:
        model = make_pipeline(input_name, clone(clf), FIXED_PCA).fit(X[tr], y[tr])
        n_pc = FIXED_PCA
    pred = model.predict(X[te])
    proba = model.predict_proba(X[te])[:, 1]
    return {'f1': f1_score(y[te], pred, average='macro'),
            'accuracy': accuracy_score(y[te], pred),
            'balanced_accuracy': balanced_accuracy_score(y[te], pred),
            'precision': precision_score(y[te], pred, average='macro', zero_division=0),
            'recall': recall_score(y[te], pred, average='macro', zero_division=0),
            'roc_auc': roc_auc_score(y[te], proba), 'n_pca': n_pc}


# ============================================================
# 3. RUN
# ============================================================
def run_protocol(protocol):
    pdir = OUT / protocol
    cache = pdir / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    compute_now = ARGS.inputs.split(',')
    rows, folds_rows = [], []
    for ep, (ep_rows, y) in ENDPOINTS.items():
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=ARGS.repeats,
                                              random_state=RS).split(ep_rows, y))
        for inp, Xfull in INPUTS.items():
            X = Xfull[ep_rows]
            for clf_name, (clf, grid) in CLASSIFIERS[protocol].items():
                t0 = time.time()
                cfile = cache / f"{ep}_{inp}_{clf_name.replace(' ', '_')}_r{ARGS.repeats}.csv"
                if inp in compute_now or not cfile.exists():
                    if inp not in compute_now:
                        continue
                    res = Parallel(n_jobs=ARGS.n_jobs)(
                        delayed(run_fold)(protocol, inp, clf, grid, X, y, tr, te) for tr, te in splits)
                    df = pd.DataFrame(res)
                    df.to_csv(cfile, index=False)
                else:
                    df = pd.read_csv(cfile)
                for i, r in df.iterrows():
                    folds_rows.append(dict(endpoint=ep, input=inp, classifier=clf_name, fold=i, **r.to_dict()))
                pcs = df['n_pca'].value_counts()
                rows.append({'endpoint': ep, 'input': inp, 'classifier': clf_name, 'n': len(y),
                             **{f'{m}_mean': df[m].mean() for m in
                                ['f1', 'roc_auc', 'balanced_accuracy', 'accuracy', 'precision', 'recall']},
                             **{f'{m}_std': df[m].std() for m in ['f1', 'roc_auc']},
                             'pca_mode': int(pcs.idxmax()), 'time_s': time.time() - t0})
                print(f"  [{protocol}] {ep:<10s} {inp:<5s} {clf_name:<19s} "
                      f"F1={df['f1'].mean():.3f}  AUC={df['roc_auc'].mean():.3f}  "
                      f"PCA mode={int(pcs.idxmax())}  [{time.time()-t0:.0f}s]", flush=True)
    summary = pd.DataFrame(rows)
    folds = pd.DataFrame(folds_rows)
    if set(summary['input']) != set(INPUTS):
        print(f"[{protocol}] partial run ({sorted(set(summary['input']))}); summary tables skipped")
        return summary
    summary.to_excel(pdir / 'classification_summary.xlsx', index=False)
    folds.to_csv(pdir / 'classification_folds.csv', index=False)

    # Best classifier per endpoint and input (Table 2)
    best = (summary.sort_values('f1_mean', ascending=False)
            .groupby(['endpoint', 'input']).head(1)
            .set_index(['endpoint', 'input']).sort_index())
    best.to_excel(pdir / 'best_per_input.xlsx')

    # CSPR per classifier and endpoint (Table S4) and averages
    piv = summary.pivot_table(index=['endpoint', 'classifier'], columns='input',
                              values=['f1_mean', 'roc_auc_mean'])
    cspr = pd.DataFrame({
        'AUC_real': piv[('roc_auc_mean', 'real')], 'AUC_recon': piv[('roc_auc_mean', 'recon')],
        'CSPR_AUC': piv[('roc_auc_mean', 'recon')] / piv[('roc_auc_mean', 'real')],
        'F1_real': piv[('f1_mean', 'real')], 'F1_recon': piv[('f1_mean', 'recon')],
        'CSPR_F1': piv[('f1_mean', 'recon')] / piv[('f1_mean', 'real')],
    })
    cspr['CPIL_F1'] = 1 - cspr['CSPR_F1']
    cspr.to_excel(pdir / 'cspr_per_classifier.xlsx')
    cspr.reset_index().to_csv(pdir / 'cspr_per_classifier.csv', index=False)
    cspr_ep = cspr.groupby(level='endpoint')[['CSPR_AUC', 'CSPR_F1']].mean()
    cspr_ep.to_excel(pdir / 'cspr_per_endpoint.xlsx')
    best_f1 = best['f1_mean'].unstack('input')
    best_cspr = (best_f1['recon'] / best_f1['real']).rename('CSPR_F1_best')
    overall = {'mean_CSPR_AUC': float(cspr_ep['CSPR_AUC'].mean()),
               'mean_CSPR_F1': float(cspr_ep['CSPR_F1'].mean()),
               'mean_CSPR_F1_best_classifier': float(best_cspr.mean())}
    with open(pdir / 'cspr_overall.json', 'w') as f:
        json.dump({'overall': overall, 'per_endpoint': cspr_ep.to_dict(),
                   'best_classifier_cspr_f1': best_cspr.to_dict()}, f, indent=2)

    print(f"\n[{protocol}] Best F1 per input:\n{best_f1.round(3)}")
    print(f"[{protocol}] CSPR per endpoint (mean over 5 classifiers):\n{cspr_ep.round(3)}")
    print(f"[{protocol}] Overall: {json.dumps({k: round(v, 3) for k, v in overall.items()})}")
    return summary


protocols = ['nested', 'fixed'] if ARGS.protocol == 'both' else [ARGS.protocol]
for p in protocols:
    t0 = time.time()
    run_protocol(p)
    print(f"[{p}] done in {time.time() - t0:.0f}s")
