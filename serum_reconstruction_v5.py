#!/usr/bin/env python3
"""
=============================================================================
Advanced Serum Reconstruction Pipeline v5
=============================================================================
Reconstructing Serum Metabolomic Profiles from Urine NMR Data

Authors: Juan de Dios Marín-Manzano, Francisco Manuel Arrabal Campos,
         Ignacio Fernández de las Nieves

Changes with respect to v4 (revision, October 2026):
  - PQN reference spectrum and Shannon MI feature weights are now estimated
    inside every training partition (outer and inner folds); in v4 both were
    computed once on all 289 samples before cross-validation (the MI weights
    used the serum targets of the test samples).
  - Weighted ensemble and stacking meta-learner are cross-fitted: for each
    outer test fold, candidate models, ensemble weights and meta-learner
    hyperparameters are chosen using only the remaining outer folds.
  - Training R², overfitting gap and MAE are computed for every model.
  - Each model is reseeded before it runs, so models can be run in parallel
    processes (--models) and combined afterwards (--ensemble).
  - TabPFN runs on GPU when available and its model version is explicit
    (default TabPFN-3.5, tabpfn>=9.2; v4 used TabPFN-2.5 on CPU).

Preprocessing:
  - Urine: PQN normalization → PowerTransform → RobustScaler (→ MI weights) → PCA
  - Serum (targets): normalised to total spectral intensity in AMIX;
    StandardScaler → PCA inside each fold

Usage:
  python serum_reconstruction_v5.py --models BaggingSVR+,BaggingKNN
  python serum_reconstruction_v5.py --models TabPFN --tabpfn-version v2.5
  python serum_reconstruction_v5.py --ensemble
=============================================================================
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os, gc, warnings, time, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Authentication (TabPFN model download): set as environment variables
#   TABPFN_TOKEN  (https://ux.priorlabs.ai)   HF_TOKEN (gated TabPFN weights)

# ============================================================
# 1. CONFIGURATION
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--models', default='all',
                    help="comma-separated model names, or 'all'")
parser.add_argument('--ensemble', action='store_true',
                    help='combine saved model outputs into the ensembles')
parser.add_argument('--smoke', action='store_true',
                    help='very few trials/epochs, to test the pipeline end to end')
parser.add_argument('--tabpfn-version', default='v3.5', choices=['v2.5', 'v3', 'v3.5'])
parser.add_argument('--results-dir', default='Results_v5')
ARGS = parser.parse_args()

DATA_DIR = Path("Data")
RESULTS_DIR = Path(ARGS.results_dir)
MODELS_DIR = RESULTS_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTER_CV = 5
INNER_CV = 3
N_TRIALS = 2 if ARGS.smoke else 30        # Optuna trials per model
N_TRIALS_DEEP = 1 if ARGS.smoke else 20   # For slow models (PyTorch nets, TabNet, TabPFN)
EPOCH_CAP = 3 if ARGS.smoke else None
N_JOBS = int(os.environ.get('N_JOBS', '-1'))
RS = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Config: OUTER_CV={OUTER_CV}, INNER_CV={INNER_CV}, N_TRIALS={N_TRIALS}, "
      f"N_TRIALS_DEEP={N_TRIALS_DEEP}, device={DEVICE}, smoke={ARGS.smoke}")

# ============================================================
# 2. DATA LOADING
# ============================================================
urine_raw = pd.read_excel(DATA_DIR / 'bucket_table_orina_COVID+PRECANCER_noscaling.xlsx')
urine_raw = urine_raw.drop(urine_raw.columns[:2], axis=1)
X_RAW = urine_raw.values.astype(np.float64)

serum_full = pd.read_excel(DATA_DIR / 'bucket_table_suero_COVID+PRECANCER_scaling.xlsx')
serum_full = serum_full.drop(serum_full.columns[:2], axis=1)
clinical_cols = [
    'COVID/Control','Hospital Days','Severity','Age Range','Sex',
    'GOT','GPT','GGT','Urea','Creatinina','Filtrado Glomerulal',
    'Colesterol Total','Colesterol de HDL','Colesterol de LDL','Triglicéridos',
    'LDH (0_Normal, 1_Alta)','Ferritina (0_Normal, 1_Alta)',
    'Prot C reactiva (0_Normal, 1_Alta)','IL6 (0_Normal, 1_Alta)',
    'Leucocitos (0_Normal, 1_Alta)','Neutrofilos (0_Normal, 1_Alta)',
    'Linfocitos (0_Normal, 1_Bajo)','Fibrinogeno (0_Normal, 1_Alta)',
    'Dimero D (0_Normal, 1_Alta)'
]
Y = serum_full.drop(
    columns=[c for c in clinical_cols if c in serum_full.columns]
).values.astype(np.float64)
del serum_full, urine_raw

N_SAMPLES, N_URINE_FEATURES = X_RAW.shape
N_SERUM_FEATURES = Y.shape[1]
print(f"  X (urine, raw): {X_RAW.shape}   Y (serum): {Y.shape}")


# ============================================================
# 3. PARTITION-LEVEL PREPROCESSING: PQN AND MI WEIGHTS
# ============================================================
def pqn_reference(X_raw_train):
    """Median spectrum of the total-area-normalised training spectra."""
    Xn = X_raw_train / (X_raw_train.sum(axis=1, keepdims=True) + 1e-12)
    return np.median(Xn, axis=0)


def pqn_apply(X_raw_any, ref):
    """Probabilistic Quotient Normalisation against a fixed reference."""
    Xn = X_raw_any / (X_raw_any.sum(axis=1, keepdims=True) + 1e-12)
    factors = np.median(Xn / (ref + 1e-12), axis=1, keepdims=True)
    return X_raw_any / (factors + 1e-12)


def compute_mi_weights(X_data, Y_data, n_pca_y=4, n_neighbors=5):
    """
    Mutual information between each urine feature and the first n_pca_y
    principal components of serum, weighted by variance explained and
    rescaled to [0.1, 1.0]. Must only be given training data.
    """
    X_s = StandardScaler().fit_transform(X_data)
    pca = PCA(n_components=n_pca_y)
    Y_pca = pca.fit_transform(StandardScaler().fit_transform(Y_data))

    mi_total = np.zeros(X_s.shape[1])
    explained = pca.explained_variance_ratio_
    for pc in range(n_pca_y):
        mi = mutual_info_regression(X_s, Y_pca[:, pc],
                                    n_neighbors=n_neighbors, random_state=RS)
        mi_total += mi * explained[pc]

    mi_norm = (mi_total - mi_total.min()) / (mi_total.max() - mi_total.min() + 1e-12)
    return 0.1 + 0.9 * mi_norm


def build_partition(raw_tr, raw_te, y_tr):
    ref = pqn_reference(raw_tr)
    Xtr_ = pqn_apply(raw_tr, ref)
    Xte_ = pqn_apply(raw_te, ref)
    return Xtr_, Xte_, compute_mi_weights(Xtr_, y_tr)


SPLITS = list(KFold(n_splits=OUTER_CV, shuffle=True, random_state=RS).split(X_RAW))
OUTER, INNER = [], []


def build_fold_cache():
    """PQN + MI for every outer training set and every inner training set."""
    t0 = time.time()
    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr_, Xte_, mi_ = build_partition(X_RAW[tri], X_RAW[tei], Y[tri])
        OUTER.append(dict(Xtr=Xtr_, Xte=Xte_, Ytr=Y[tri], Yte=Y[tei], mi=mi_))
        inner = []
        for itr, ival in KFold(n_splits=INNER_CV, shuffle=True,
                               random_state=RS + fi).split(tri):
            Xi_tr, Xi_val, mi_i = build_partition(X_RAW[tri][itr], X_RAW[tri][ival],
                                                  Y[tri][itr])
            inner.append(dict(Xtr=Xi_tr, Xval=Xi_val, Ytr=Y[tri][itr],
                              Yval=Y[tri][ival], mi=mi_i))
        INNER.append(inner)
    print(f"  Fold cache (PQN + MI per partition) built in {time.time() - t0:.0f}s")


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================
def aug_gaussian(X, Y, n=2, sigma=0.02):
    Xa, Ya = [X], [Y]
    for _ in range(n):
        noise_x = np.random.normal(0, sigma, X.shape) * np.std(X, axis=0, keepdims=True)
        noise_y = np.random.normal(0, sigma * 0.5, Y.shape) * np.std(Y, axis=0, keepdims=True)
        Xa.append(X + noise_x)
        Ya.append(Y + noise_y)
    return np.vstack(Xa), np.vstack(Ya)

def aug_mixup(X, Y, n=2, alpha=0.3):
    Xa, Ya = [X], [Y]
    m = len(X)
    for _ in range(n):
        i1 = np.random.permutation(m)
        i2 = np.random.permutation(m)
        lam = np.random.beta(alpha, alpha, (m, 1))
        Xa.append(lam * X[i1] + (1 - lam) * X[i2])
        Ya.append(lam * Y[i1] + (1 - lam) * Y[i2])
    return np.vstack(Xa), np.vstack(Ya)

def aug_smogn(X, Y, n=2, k=5):
    from sklearn.neighbors import NearestNeighbors
    Xa, Ya = [X], [Y]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn_.kneighbors(X)
    m = len(X)
    for _ in range(n):
        nX, nY = [], []
        for i in range(m):
            j = idx[i, np.random.randint(1, k + 1)]
            lam = np.random.uniform()
            nX.append(X[i] + lam * (X[j] - X[i]))
            nY.append(Y[i] + lam * (Y[j] - Y[i]))
        Xa.append(np.array(nX))
        Ya.append(np.array(nY))
    return np.vstack(Xa), np.vstack(Ya)

def apply_aug(X, Y, aug_type):
    if aug_type == 'none': return X, Y
    elif aug_type == 'noise': return aug_gaussian(X, Y, 2)
    elif aug_type == 'mixup': return aug_mixup(X, Y, 2)
    elif aug_type == 'smogn': return aug_smogn(X, Y, 2)
    elif aug_type == 'noise+mixup':
        Xa, Ya = aug_gaussian(X, Y, 1)
        return aug_mixup(Xa, Ya, 1)
    return X, Y

# ============================================================
# 5. PREPROCESSING PIPELINE (per fold)
# ============================================================
def preprocess_fold(X_train, X_test, Y_train, n_pca_x, n_pca_y,
                    aug_type='none', use_mi_weights=False, mi_w=None):
    # --- X pipeline ---
    pt_x = PowerTransformer(method='yeo-johnson')
    sc_x = RobustScaler()
    X_tr = sc_x.fit_transform(pt_x.fit_transform(X_train))
    X_te = sc_x.transform(pt_x.transform(X_test))

    if use_mi_weights and mi_w is not None:
        X_tr = X_tr * mi_w[np.newaxis, :]
        X_te = X_te * mi_w[np.newaxis, :]

    pca_x = PCA(n_components=n_pca_x)
    X_tr_pca = pca_x.fit_transform(X_tr)
    X_te_pca = pca_x.transform(X_te)

    # --- Y pipeline ---
    sc_y = StandardScaler()
    pca_y = PCA(n_components=n_pca_y)
    Y_tr_scaled = sc_y.fit_transform(Y_train)
    Y_tr_pca = pca_y.fit_transform(Y_tr_scaled)

    # --- Augmentation in PCA space ---
    X_tr_aug, Y_tr_pca_aug = X_tr_pca, Y_tr_pca
    if aug_type != 'none':
        X_tr_aug, Y_tr_pca_aug = apply_aug(X_tr_pca, Y_tr_pca, aug_type)

    return (X_tr_aug, X_tr_pca, X_te_pca, Y_tr_pca_aug, Y_tr_scaled,
            sc_y, pca_y, pt_x, sc_x, pca_x)

def inverse_transform_y(Y_pca_pred, sc_y, pca_y):
    """Inverse PCA + StandardScaler on Y predictions, with clipping."""
    Y_reconstructed = pca_y.inverse_transform(Y_pca_pred)
    Y_out = sc_y.inverse_transform(Y_reconstructed)
    lower = sc_y.mean_ - 3 * sc_y.scale_
    upper = sc_y.mean_ + 3 * sc_y.scale_
    return np.clip(Y_out, lower, upper)

# ============================================================
# 6. NEURAL ARCHITECTURES
# ============================================================
class SpectralAttention(nn.Module):
    """Learnable attention over input features (optionally MI-initialised)."""
    def __init__(self, n_features, mi_weights=None):
        super().__init__()
        self.attn_logits = nn.Parameter(torch.zeros(n_features))
        if mi_weights is not None:
            with torch.no_grad():
                mi_t = torch.FloatTensor(mi_weights)
                self.attn_logits.copy_(torch.log(mi_t + 1e-6))
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = torch.softmax(self.attn_logits, dim=0)
        x_attended = x * attn.unsqueeze(0)
        gate_val = self.gate(x)
        return gate_val * x_attended + (1 - gate_val) * x


class SpectralAttentionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128, 64),
                 dropout=0.3, activation='relu', mi_weights=None):
        super().__init__()
        self.attention = SpectralAttention(input_dim, mi_weights)
        act_fn = nn.ReLU() if activation == 'relu' else nn.GELU()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), act_fn, nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.attention(x))


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_blocks=4,
                 dropout=0.3, use_attention=True, mi_weights=None):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpectralAttention(input_dim, mi_weights)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, x):
        if self.use_attention:
            x = self.attention(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class Spectral1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters=64, kernel_size=7, dropout=0.3):
        super().__init__()
        ks2 = max(3, kernel_size // 2)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_filters), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(n_filters, n_filters * 2, ks2, padding=ks2 // 2),
            nn.BatchNorm1d(n_filters * 2), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(n_filters * 2, n_filters * 4, 3, padding=1),
            nn.BatchNorm1d(n_filters * 4), nn.GELU(),
            nn.AdaptiveAvgPool1d(8), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * 4 * 8, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.head(self.encoder(x.unsqueeze(1)))


def train_torch_model(model, X_train, Y_train, X_val=None, X_extra=None,
                      lr=1e-3, wd=1e-4, epochs=200, batch_size=32, patience=20):
    """Generic PyTorch training loop with early stopping on training loss.
    Returns predictions for X_val and, if given, X_extra (both clipped)."""
    if EPOCH_CAP:
        epochs = min(epochs, EPOCH_CAP)
    device = torch.device(DEVICE)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    Xt = torch.FloatTensor(X_train).to(device)
    Yt = torch.FloatTensor(Y_train).to(device)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    no_improve = 0
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.eval()
    preds = []
    with torch.no_grad():
        train_pred = model(Xt).cpu().numpy()
        pred_min = train_pred.min(axis=0) - 2 * train_pred.std(axis=0)
        pred_max = train_pred.max(axis=0) + 2 * train_pred.std(axis=0)
        for Xq in (X_val, X_extra):
            if Xq is None:
                preds.append(None)
            else:
                p = model(torch.FloatTensor(Xq).to(device)).cpu().numpy()
                preds.append(np.clip(p, pred_min, pred_max))

    del model, optimizer, Xt, Yt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return preds[0] if X_extra is None else (preds[0], preds[1])

# ============================================================
# 7. NESTED CV RUNNERS
# ============================================================
def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)): v = int(v)
        elif isinstance(v, (np.floating,)): v = float(v)
        elif isinstance(v, (np.bool_,)): v = bool(v)
        out[k] = v
    return out


def save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                      fold_params, t0):
    result = {
        'r2_test_mean': float(np.mean(fold_r2t)), 'r2_test_std': float(np.std(fold_r2t)),
        'r2_train_mean': float(np.mean(fold_r2tr)),
        'gap': float(np.mean(fold_r2tr) - np.mean(fold_r2t)),
        'mae': float(np.mean(fold_mae)), 'time': time.time() - t0,
        'fold_r2_test': [float(v) for v in fold_r2t],
        'fold_r2_train': [float(v) for v in fold_r2tr],
        'fold_mae': [float(v) for v in fold_mae],
        'fold_params': [_jsonable(p) for p in fold_params],
    }
    safe = name.replace('+', 'plus')
    with open(MODELS_DIR / f'{safe}.json', 'w') as f:
        json.dump(result, f, indent=2)
    np.save(MODELS_DIR / f'{safe}_oos.npy', oos)
    np.save(MODELS_DIR / f'{safe}_train.npy', train_preds.astype(np.float32))
    print(f"  >> {name}: R²={result['r2_test_mean']:.4f}±{result['r2_test_std']:.4f}, "
          f"train={result['r2_train_mean']:.4f}, gap={result['gap']:.4f}, "
          f"MAE={result['mae']:.5f} [{result['time']:.0f}s]")
    return result


def _start(name):
    np.random.seed(RS)
    torch.manual_seed(RS)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    return time.time()


def _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds):
    tri, tei = SPLITS[fi]
    o = OUTER[fi]
    oos[tei] = Yp_te
    train_preds[fi][tri] = Yp_tr
    return (r2_score(o['Yte'], Yp_te, multioutput='uniform_average'),
            r2_score(o['Ytr'], Yp_tr, multioutput='uniform_average'),
            mean_absolute_error(o['Yte'], Yp_te))


def _new_buffers():
    return (np.zeros_like(Y), np.full((OUTER_CV, N_SAMPLES, N_SERUM_FEATURES), np.nan))


def run_sklearn_model(name, make_fn, space_fn, n_trials=None):
    """sklearn regressor (one per serum PC) with Optuna inside nested CV."""
    n_trials = n_trials or N_TRIALS
    t0 = _start(name)
    fold_r2t, fold_r2tr, fold_mae, fold_params = [], [], [], []
    oos, train_preds = _new_buffers()

    for fi in range(OUTER_CV):
        o = OUTER[fi]

        def objective(trial):
            p = space_fn(trial)
            npx = p.pop('npx'); npy = p.pop('npy')
            at = p.pop('aug'); use_mi = p.pop('use_mi', False)
            r2s = []
            for ip in INNER[fi]:
                try:
                    res = preprocess_fold(ip['Xtr'], ip['Xval'], ip['Ytr'],
                                          npx, npy, at, use_mi, ip['mi'])
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res
                    m = MultiOutputRegressor(make_fn(p))
                    m.fit(X_aug, Y_pca)
                    Yp = inverse_transform_y(m.predict(X_val_pca), sc_y, pca_y)
                    r2s.append(r2_score(ip['Yval'], Yp, multioutput='uniform_average'))
                except Exception:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        fold_params.append(dict(bp))
        npx = bp.pop('npx'); npy = bp.pop('npy')
        at = bp.pop('aug'); use_mi = bp.pop('use_mi', False)

        res = preprocess_fold(o['Xtr'], o['Xte'], o['Ytr'], npx, npy, at, use_mi, o['mi'])
        X_aug, X_tr_pca, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res
        m = MultiOutputRegressor(make_fn(bp))
        m.fit(X_aug, Y_pca)
        Yp_te = inverse_transform_y(m.predict(X_te_pca), sc_y, pca_y)
        Yp_tr = inverse_transform_y(m.predict(X_tr_pca), sc_y, pca_y)

        r2t, r2tr, mae_v = _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}, "
              f"aug={at}, mi={use_mi}, pca_x={npx}, pca_y={npy}")
        del study; gc.collect()

    return save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                             fold_params, t0)


def run_torch_model(name, build_model_fn, space_fn, n_trials=None):
    """PyTorch network with Optuna inside nested CV."""
    n_trials = n_trials or N_TRIALS_DEEP
    t0 = _start(name)
    fold_r2t, fold_r2tr, fold_mae, fold_params = [], [], [], []
    oos, train_preds = _new_buffers()

    for fi in range(OUTER_CV):
        o = OUTER[fi]

        def objective(trial):
            p = space_fn(trial)
            npx = p.pop('npx'); npy = p.pop('npy')
            at = p.pop('aug'); use_mi = p.pop('use_mi', False)
            lr = p.pop('lr'); wd = p.pop('wd')
            epochs = p.pop('epochs', 200)
            r2s = []
            for ip in INNER[fi]:
                try:
                    res = preprocess_fold(ip['Xtr'], ip['Xval'], ip['Ytr'],
                                          npx, npy, at, use_mi, ip['mi'])
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res
                    model = build_model_fn(p, X_aug.shape[1], Y_pca.shape[1])
                    pred = train_torch_model(model, X_aug, Y_pca, X_val_pca,
                                             lr=lr, wd=wd, epochs=epochs)
                    Yp = inverse_transform_y(pred, sc_y, pca_y)
                    r2s.append(r2_score(ip['Yval'], Yp, multioutput='uniform_average'))
                except Exception:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        fold_params.append(dict(bp))
        space_defaults = space_fn(optuna.trial.FixedTrial(study.best_params))
        npx = bp.pop('npx'); npy = bp.pop('npy')
        at = bp.pop('aug')
        use_mi = bp.pop('use_mi', space_defaults.get('use_mi', False))
        lr = bp.pop('lr'); wd = bp.pop('wd')
        epochs = space_defaults.get('epochs', 200)

        res = preprocess_fold(o['Xtr'], o['Xte'], o['Ytr'], npx, npy, at, use_mi, o['mi'])
        X_aug, X_tr_pca, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res
        model = build_model_fn(bp, X_aug.shape[1], Y_pca.shape[1])
        pred_te, pred_tr = train_torch_model(model, X_aug, Y_pca, X_te_pca, X_tr_pca,
                                             lr=lr, wd=wd, epochs=epochs)
        Yp_te = inverse_transform_y(pred_te, sc_y, pca_y)
        Yp_tr = inverse_transform_y(pred_tr, sc_y, pca_y)

        r2t, r2tr, mae_v = _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}, mi={use_mi}")
        del study; gc.collect()

    return save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                             fold_params, t0)


def run_pls():
    """PLS jointly optimises the latent space (no separate PCA)."""
    name = 'PLS'
    t0 = _start(name)
    fold_r2t, fold_r2tr, fold_mae, fold_params = [], [], [], []
    oos, train_preds = _new_buffers()

    def fit_predict(X_tr_raw, X_q_list, Y_tr, n_comp, use_mi, aug_type, mi_w):
        pt = PowerTransformer('yeo-johnson')
        sc_x = RobustScaler()
        X_t = sc_x.fit_transform(pt.fit_transform(X_tr_raw))
        X_qs = [sc_x.transform(pt.transform(Xq)) for Xq in X_q_list]
        if use_mi:
            X_t = X_t * mi_w[np.newaxis, :]
            X_qs = [Xq * mi_w[np.newaxis, :] for Xq in X_qs]
        sc_y = StandardScaler()
        Y_t = sc_y.fit_transform(Y_tr)
        X_a, Y_a = X_t, Y_t
        if aug_type != 'none':
            X_a, Y_a = apply_aug(X_t, Y_t, aug_type)
        n_c = min(n_comp, X_a.shape[1], X_a.shape[0], Y_a.shape[1])
        pls = PLSRegression(n_components=n_c, max_iter=1000)
        pls.fit(X_a, Y_a)
        lower = sc_y.mean_ - 4 * sc_y.scale_
        upper = sc_y.mean_ + 4 * sc_y.scale_
        return [np.clip(sc_y.inverse_transform(pls.predict(Xq)), lower, upper) for Xq in X_qs], n_c

    for fi in range(OUTER_CV):
        o = OUTER[fi]

        def obj(trial):
            n_comp = trial.suggest_int('n_comp', 2, 25)
            use_mi = trial.suggest_categorical('use_mi', [True, False])
            aug_type = trial.suggest_categorical('aug', ['none', 'noise', 'mixup', 'smogn'])
            r2s = []
            for ip in INNER[fi]:
                try:
                    (Yp,), _ = fit_predict(ip['Xtr'], [ip['Xval']], ip['Ytr'],
                                           n_comp, use_mi, aug_type, ip['mi'])
                    r2s.append(r2_score(ip['Yval'], Yp, multioutput='uniform_average'))
                except Exception:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
        bp = study.best_params
        fold_params.append(dict(bp))

        (Yp_te, Yp_tr), n_c = fit_predict(o['Xtr'], [o['Xte'], o['Xtr']], o['Ytr'],
                                          bp['n_comp'], bp['use_mi'], bp['aug'], o['mi'])
        r2t, r2tr, mae_v = _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}, "
              f"n_comp={n_c}, aug={bp['aug']}, mi={bp['use_mi']}")
        del study; gc.collect()

    return save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                             fold_params, t0)


def run_tabnet():
    """TabNet on the scaled spectra (no MI weights, as in v4)."""
    from pytorch_tabnet.tab_model import TabNetRegressor as TabNetReg
    name = 'TabNet'
    t0 = _start(name)
    fold_r2t, fold_r2tr, fold_mae, fold_params = [], [], [], []
    oos, train_preds = _new_buffers()
    ep_inner = EPOCH_CAP or 100
    ep_final = EPOCH_CAP or 150

    def fit_predict(X_tr_raw, X_q_list, Y_tr, p, max_epochs, patience):
        pt = PowerTransformer('yeo-johnson'); sc = RobustScaler()
        X_t = sc.fit_transform(pt.fit_transform(X_tr_raw))
        X_qs = [sc.transform(pt.transform(Xq)) for Xq in X_q_list]
        sc_y = StandardScaler(); pca_y = PCA(n_components=p['npy'])
        Y_t_pca = pca_y.fit_transform(sc_y.fit_transform(Y_tr))
        X_a, Y_a = X_t, Y_t_pca
        if p['aug'] != 'none':
            X_a, Y_a = apply_aug(X_t, Y_t_pca, p['aug'])
        tn = TabNetReg(n_d=p['nd'], n_a=p['nd'], n_steps=p['ns'], gamma=p['gm'],
                       lambda_sparse=p['ls'], optimizer_params=dict(lr=p['lr_tn']),
                       verbose=0, seed=42, device_name=DEVICE)
        tn.fit(X_a, Y_a, max_epochs=max_epochs, patience=patience,
               batch_size=64, drop_last=False)
        train_pred = tn.predict(X_a)
        pmin = train_pred.min(axis=0) - 2 * train_pred.std(axis=0)
        pmax = train_pred.max(axis=0) + 2 * train_pred.std(axis=0)
        return [sc_y.inverse_transform(pca_y.inverse_transform(np.clip(tn.predict(Xq), pmin, pmax)))
                for Xq in X_qs]

    for fi in range(OUTER_CV):
        o = OUTER[fi]

        def obj(trial):
            p = dict(npy=trial.suggest_int('npy', 3, 8),
                     aug=trial.suggest_categorical('aug', ['none', 'noise', 'mixup']),
                     nd=trial.suggest_categorical('nd', [8, 16, 32, 64]),
                     ns=trial.suggest_int('ns', 3, 7),
                     gm=trial.suggest_float('gm', 1.0, 2.0),
                     ls=trial.suggest_float('ls', 1e-6, 1e-2, log=True),
                     lr_tn=trial.suggest_float('lr_tn', 1e-3, 5e-2, log=True))
            r2s = []
            for ip in INNER[fi]:
                try:
                    (Yp,) = fit_predict(ip['Xtr'], [ip['Xval']], ip['Ytr'], p, ep_inner, 15)
                    r2s.append(r2_score(ip['Yval'], Yp, multioutput='uniform_average'))
                except Exception:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS_DEEP, show_progress_bar=False)
        bp = study.best_params
        fold_params.append(dict(bp))

        Yp_te, Yp_tr = fit_predict(o['Xtr'], [o['Xte'], o['Xtr']], o['Ytr'], bp, ep_final, 20)
        r2t, r2tr, mae_v = _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        print(f"    Fold {fi+1}: R²={r2t:.4f}, inner={study.best_value:.4f}")
        del study; gc.collect()

    return save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                             fold_params, t0)


def run_tabpfn():
    """TabPFN regressor, one model per serum PC, inside nested CV."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    version = ModelVersion(ARGS.tabpfn_version)
    name = 'TabPFN'
    t0 = _start(name)
    print(f"  TabPFN model version: {version.value}, device: {DEVICE}")
    fold_r2t, fold_r2tr, fold_mae, fold_params = [], [], [], []
    oos, train_preds = _new_buffers()

    def fit_predict_pcs(X_aug, Y_pca, X_q_list, output_type):
        # output_type: point estimate of TabPFN's predictive distribution. The mean is
        # pulled by a few extreme serum samples in the low-variance PCs with v3/v3.5,
        # so mean/median/mode is tuned in the inner CV.
        preds = [np.zeros((len(Xq), Y_pca.shape[1])) for Xq in X_q_list]
        for c in range(Y_pca.shape[1]):
            reg = TabPFNRegressor.create_default_for_version(version, device=DEVICE)
            reg.fit(X_aug, Y_pca[:, c])
            for k, Xq in enumerate(X_q_list):
                preds[k][:, c] = reg.predict(Xq, output_type=output_type)
        return preds

    for fi in range(OUTER_CV):
        o = OUTER[fi]

        def obj(trial):
            npx = trial.suggest_int('npx', 5, 80)
            npy = trial.suggest_int('npy', 3, 8)
            at = trial.suggest_categorical('aug', ['none', 'noise', 'mixup'])
            use_mi = trial.suggest_categorical('use_mi', [True, False])
            ot = trial.suggest_categorical('output_type', ['mean', 'median', 'mode'])
            r2s = []
            for ip in INNER[fi]:
                try:
                    res = preprocess_fold(ip['Xtr'], ip['Xval'], ip['Ytr'],
                                          npx, npy, at, use_mi, ip['mi'])
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res
                    (pred_pca,) = fit_predict_pcs(X_aug, Y_pca, [X_val_pca], ot)
                    Yp = inverse_transform_y(pred_pca, sc_y, pca_y)
                    r2s.append(r2_score(ip['Yval'], Yp, multioutput='uniform_average'))
                except Exception as e:
                    print(f"      inner error: {e}")
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS_DEEP, show_progress_bar=False)
        bp = study.best_params
        fold_params.append(dict(bp))

        res = preprocess_fold(o['Xtr'], o['Xte'], o['Ytr'], bp['npx'], bp['npy'],
                              bp['aug'], bp['use_mi'], o['mi'])
        X_aug, X_tr_pca, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res
        pred_te, pred_tr = fit_predict_pcs(X_aug, Y_pca, [X_te_pca, X_tr_pca], bp['output_type'])
        Yp_te = inverse_transform_y(pred_te, sc_y, pca_y)
        Yp_tr = inverse_transform_y(pred_tr, sc_y, pca_y)

        r2t, r2tr, mae_v = _fold_metrics(fi, Yp_te, Yp_tr, oos, train_preds)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        print(f"    Fold {fi+1}: R²={r2t:.4f}, inner={study.best_value:.4f}, mi={bp['use_mi']}, "
              f"output={bp['output_type']}")
        del study; gc.collect()

    return save_model_output(name, fold_r2t, fold_r2tr, fold_mae, oos, train_preds,
                             fold_params, t0)

# ============================================================
# 8. MODEL DEFINITIONS (search spaces identical to v4)
# ============================================================
def svr_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn','noise+mixup']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'C': t.suggest_float('C', 0.1, 500, log=True),
            'eps': t.suggest_float('eps', 0.01, 0.5, log=True),
            'gamma': t.suggest_categorical('gamma', ['scale','auto']),
            'nest': t.suggest_int('nest', 10, 50)}
def make_svr(p):
    return BaggingRegressor(estimator=SVR(kernel='rbf', C=p['C'], epsilon=p['eps'],
        gamma=p['gamma']), n_estimators=p['nest'], random_state=42)

def xgb_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'nest': t.suggest_int('nest', 50, 400), 'md': t.suggest_int('md', 2, 7),
            'lr_xgb': t.suggest_float('lr_xgb', 0.005, 0.2, log=True),
            'ss': t.suggest_float('ss', 0.5, 1.0), 'cs': t.suggest_float('cs', 0.5, 1.0),
            'ra': t.suggest_float('ra', 1e-5, 10, log=True),
            'rl': t.suggest_float('rl', 1e-5, 10, log=True),
            'mcw': t.suggest_int('mcw', 1, 10)}
def make_xgb(p):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=p['nest'], max_depth=p['md'],
        learning_rate=p['lr_xgb'], subsample=p['ss'], colsample_bytree=p['cs'],
        reg_alpha=p['ra'], reg_lambda=p['rl'], min_child_weight=p['mcw'],
        random_state=42, verbosity=0, n_jobs=N_JOBS)

def lgb_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'nest': t.suggest_int('nest', 50, 400), 'md': t.suggest_int('md', 2, 7),
            'lr_lgb': t.suggest_float('lr_lgb', 0.005, 0.2, log=True),
            'ss': t.suggest_float('ss', 0.5, 1.0), 'cs': t.suggest_float('cs', 0.5, 1.0),
            'ra': t.suggest_float('ra', 1e-5, 10, log=True),
            'rl': t.suggest_float('rl', 1e-5, 10, log=True),
            'mcs': t.suggest_int('mcs', 3, 30)}
def make_lgb(p):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=p['nest'], max_depth=p['md'],
        learning_rate=p['lr_lgb'], subsample=p['ss'], colsample_bytree=p['cs'],
        reg_alpha=p['ra'], reg_lambda=p['rl'], min_child_samples=p['mcs'],
        random_state=42, verbosity=-1, n_jobs=N_JOBS)

def ridge_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'alpha': t.suggest_float('alpha', 1e-3, 1000, log=True)}
def make_ridge(p): return Ridge(alpha=p['alpha'])

def enet_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'alpha': t.suggest_float('alpha', 1e-5, 10, log=True),
            'l1r': t.suggest_float('l1r', 0.01, 0.99)}
def make_enet(p): return ElasticNet(alpha=p['alpha'], l1_ratio=p['l1r'],
    max_iter=5000, random_state=42)

def et_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'nest': t.suggest_int('nest', 50, 300), 'md': t.suggest_int('md', 3, 15),
            'msl': t.suggest_int('msl', 2, 20), 'mf': t.suggest_float('mf', 0.3, 1.0)}
def make_et(p): return ExtraTreesRegressor(n_estimators=p['nest'],
    max_depth=p['md'], min_samples_leaf=p['msl'], max_features=p['mf'],
    random_state=42, n_jobs=N_JOBS)

def knn_space(t):
    return {'npx': t.suggest_int('npx', 3, 20), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'k': t.suggest_int('k', 3, 20), 'w': t.suggest_categorical('w', ['uniform','distance']),
            'nest': t.suggest_int('nest', 10, 40), 'ms': t.suggest_float('ms', 0.5, 1.0)}
def make_knn(p): return BaggingRegressor(estimator=KNeighborsRegressor(
    n_neighbors=p['k'], weights=p['w']), n_estimators=p['nest'],
    max_samples=p['ms'], random_state=42)

def mlp_wide_space(t):
    return {'npx': t.suggest_int('npx', 5, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'h1': t.suggest_int('h1', 128, 1024), 'h2': t.suggest_int('h2', 64, 512),
            'act': t.suggest_categorical('act', ['relu','tanh']),
            'a': t.suggest_float('a', 1e-5, 1, log=True),
            'lr_mlp': t.suggest_float('lr_mlp', 1e-4, 1e-2, log=True)}
def make_mlp_wide(p): return MLPRegressor(
    hidden_layer_sizes=(p['h1'], p['h2']),
    activation=p['act'], alpha=p['a'], learning_rate_init=p['lr_mlp'],
    max_iter=1500, early_stopping=True, validation_fraction=0.15, random_state=42)

def mlp_deep_space(t):
    return {'npx': t.suggest_int('npx', 5, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'h1': t.suggest_int('h1', 128, 512), 'h2': t.suggest_int('h2', 64, 256),
            'h3': t.suggest_int('h3', 32, 128), 'h4': t.suggest_int('h4', 16, 64),
            'act': t.suggest_categorical('act', ['relu','tanh']),
            'a': t.suggest_float('a', 1e-5, 1, log=True),
            'lr_mlp': t.suggest_float('lr_mlp', 1e-4, 1e-2, log=True)}
def make_mlp_deep(p): return MLPRegressor(
    hidden_layer_sizes=(p['h1'], p['h2'], p['h3'], p['h4']),
    activation=p['act'], alpha=p['a'], learning_rate_init=p['lr_mlp'],
    max_iter=1500, early_stopping=True, validation_fraction=0.15, random_state=42)

def san_mlp_space(t):
    return {'npx': t.suggest_int('npx', 10, 50), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': True,  # MI weighting of the urine bins is always applied
            'h1': t.suggest_int('h1', 128, 512), 'h2': t.suggest_int('h2', 64, 256),
            'h3': t.suggest_int('h3', 32, 128),
            'act': t.suggest_categorical('act', ['relu','gelu']),
            'do': t.suggest_float('do', 0.1, 0.5),
            'lr': t.suggest_float('lr', 1e-4, 5e-3, log=True),
            'wd': t.suggest_float('wd', 1e-6, 1e-2, log=True),
            'epochs': 200}
def build_san_mlp(p, in_dim, out_dim):
    # The attention acts on PCA scores, so it is not MI-initialised (as in v4,
    # where the MI vector never matched the post-PCA input dimension).
    return SpectralAttentionMLP(in_dim, out_dim,
        hidden_dims=(p['h1'], p['h2'], p['h3']),
        dropout=p['do'], activation=p['act'], mi_weights=None)

def resmlp_space(t):
    return {'npx': t.suggest_int('npx', 10, 50), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'hdim': t.suggest_categorical('hdim', [128, 256, 512]),
            'nblocks': t.suggest_int('nblocks', 2, 6),
            'do': t.suggest_float('do', 0.1, 0.5),
            'use_attn': t.suggest_categorical('use_attn', [True, False]),
            'lr': t.suggest_float('lr', 1e-4, 5e-3, log=True),
            'wd': t.suggest_float('wd', 1e-6, 1e-2, log=True),
            'epochs': 250}
def build_resmlp(p, in_dim, out_dim):
    return DeepResidualMLP(in_dim, out_dim, hidden_dim=p['hdim'],
        n_blocks=p['nblocks'], dropout=p['do'],
        use_attention=p['use_attn'], mi_weights=None)

def cnn_space(t):
    return {'npx': t.suggest_int('npx', 15, 60), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'nf': t.suggest_categorical('nf', [32, 64, 128]),
            'ks': t.suggest_categorical('ks', [5, 7, 9]),
            'do': t.suggest_float('do', 0.1, 0.5),
            'lr': t.suggest_float('lr', 1e-4, 5e-3, log=True),
            'wd': t.suggest_float('wd', 1e-6, 1e-2, log=True),
            'epochs': 200}
def build_cnn(p, in_dim, out_dim):
    return Spectral1DCNN(in_dim, out_dim, n_filters=p['nf'],
        kernel_size=p['ks'], dropout=p['do'])

REGISTRY = {
    'BaggingSVR+': lambda: run_sklearn_model('BaggingSVR+', make_svr, svr_space),
    'PLS':         run_pls,
    'XGBoost':     lambda: run_sklearn_model('XGBoost', make_xgb, xgb_space),
    'LightGBM':    lambda: run_sklearn_model('LightGBM', make_lgb, lgb_space),
    'Ridge':       lambda: run_sklearn_model('Ridge', make_ridge, ridge_space),
    'ElasticNet':  lambda: run_sklearn_model('ElasticNet', make_enet, enet_space),
    'ExtraTrees':  lambda: run_sklearn_model('ExtraTrees', make_et, et_space),
    'BaggingKNN':  lambda: run_sklearn_model('BaggingKNN', make_knn, knn_space),
    'MLP_Wide':    lambda: run_sklearn_model('MLP_Wide', make_mlp_wide, mlp_wide_space),
    'MLP_Deep':    lambda: run_sklearn_model('MLP_Deep', make_mlp_deep, mlp_deep_space),
    'SAN_MLP':     lambda: run_torch_model('SAN_MLP', build_san_mlp, san_mlp_space),
    'ResidualMLP': lambda: run_torch_model('ResidualMLP', build_resmlp, resmlp_space),
    '1D_CNN':      lambda: run_torch_model('1D_CNN', build_cnn, cnn_space),
    'TabNet':      run_tabnet,
    'TabPFN':      run_tabpfn,
}

# ============================================================
# 9. CROSS-FITTED ENSEMBLES
# ============================================================
def load_model_outputs():
    res, oos, trn = {}, {}, {}
    for name in REGISTRY:
        safe = name.replace('+', 'plus')
        f = MODELS_DIR / f'{safe}.json'
        if not f.exists():
            print(f"  WARNING: no output for {name}")
            continue
        res[name] = json.load(open(f))
        oos[name] = np.load(MODELS_DIR / f'{safe}_oos.npy')
        trn[name] = np.load(MODELS_DIR / f'{safe}_train.npy').astype(np.float64)
    return res, oos, trn


def candidates_for(k, res, n_top=6, min_r2=0.05):
    """Top stable models ranked on the outer folds other than k."""
    others = [j for j in range(OUTER_CV) if j != k]
    score = {n: np.mean([r['fold_r2_test'][j] for j in others]) for n, r in res.items()}
    ranked = sorted((n for n in score if score[n] > min_r2), key=lambda n: -score[n])
    return ranked[:n_top], score


def optimise_weights(names, oos, folds, n_trials=200):
    def wobj(trial):
        ws = {m: trial.suggest_float(f'w_{m}', 0, 1) for m in names}
        t = sum(ws.values())
        if t < 1e-8: return -1
        Yp = sum((ws[m] / t) * oos[m] for m in names)
        return np.mean([r2_score(Y[SPLITS[j][1]], Yp[SPLITS[j][1]],
                                 multioutput='uniform_average') for j in folds])
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    st.optimize(wobj, n_trials=n_trials, show_progress_bar=False)
    tw = sum(st.best_params.values())
    return {m: st.best_params[f'w_{m}'] / tw for m in names}, st.best_value


def stacking_predict(names, oos, train_folds, test_fold, n_trials=50):
    """Ridge meta-learner on OOF predictions; hyperparameters chosen by
    leave-one-fold-out over the training folds only."""
    meta = np.hstack([oos[m] for m in names])
    rows = lambda folds: np.concatenate([SPLITS[j][1] for j in folds])

    def fit_pred(tr_rows, te_rows, alpha, npc, npy):
        sc = StandardScaler(); pc = PCA(n_components=npc)
        Xt = pc.fit_transform(sc.fit_transform(meta[tr_rows]))
        Xq = pc.transform(sc.transform(meta[te_rows]))
        scy = StandardScaler(); pcy = PCA(n_components=npy)
        Yt = pcy.fit_transform(scy.fit_transform(Y[tr_rows]))
        m = MultiOutputRegressor(Ridge(alpha=alpha)).fit(Xt, Yt)
        return scy.inverse_transform(pcy.inverse_transform(m.predict(Xq)))

    def sobj(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100, log=True)
        npc = trial.suggest_int('npc', 3, min(30, meta.shape[1]))
        npy = trial.suggest_int('npy', 3, 8)
        r2s = []
        for j in train_folds:
            tr = rows([f for f in train_folds if f != j]); te = SPLITS[j][1]
            r2s.append(r2_score(Y[te], fit_pred(tr, te, alpha, npc, npy),
                                multioutput='uniform_average'))
        return np.mean(r2s)

    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    st.optimize(sobj, n_trials=n_trials, show_progress_bar=False)
    bp = st.best_params
    te = SPLITS[test_fold][1]
    return fit_pred(rows(train_folds), te, bp['alpha'], bp['npc'], bp['npy']), bp


def run_ensemble():
    print(f"\n{'='*70}\n[ENSEMBLE] Cross-fitted weighted ensemble and stacking\n{'='*70}")
    res, oos, trn = load_model_outputs()
    all_results = {n: {k: r[k] for k in ('r2_test_mean', 'r2_test_std', 'r2_train_mean',
                                         'gap', 'mae', 'time')} for n, r in res.items()}

    Y_ens = np.zeros_like(Y)
    ens_r2, ens_r2tr, ens_mae, stk_r2, folds_info = [], [], [], [], []
    for k in range(OUTER_CV):
        others = [j for j in range(OUTER_CV) if j != k]
        names, _ = candidates_for(k, res)
        w, inner_val = optimise_weights(names, oos, others)
        tri, tei = SPLITS[k]
        Yp_te = sum(w[m] * oos[m][tei] for m in names)
        Yp_tr = sum(w[m] * trn[m][k][tri] for m in names)
        Y_ens[tei] = Yp_te
        ens_r2.append(r2_score(Y[tei], Yp_te, multioutput='uniform_average'))
        ens_r2tr.append(r2_score(Y[tri], Yp_tr, multioutput='uniform_average'))
        ens_mae.append(mean_absolute_error(Y[tei], Yp_te))

        Ys, sbp = stacking_predict(names, oos, others, k)
        stk_r2.append(r2_score(Y[tei], Ys, multioutput='uniform_average'))
        folds_info.append({'fold': k + 1, 'candidates': names,
                           'weights': {m: round(float(v), 4) for m, v in w.items()},
                           'weight_fit_r2_other_folds': float(inner_val),
                           'ensemble_r2_test': float(ens_r2[-1]),
                           'stacking_params': sbp, 'stacking_r2_test': float(stk_r2[-1])})
        print(f"  Fold {k+1}: ensemble R²={ens_r2[-1]:.4f}  stacking R²={stk_r2[-1]:.4f}  "
              f"weights={ {m: round(v, 3) for m, v in w.items()} }")

    mean_w = {}
    for fo in folds_info:
        for m, v in fo['weights'].items():
            mean_w[m] = mean_w.get(m, 0) + v / OUTER_CV
    all_results['WeightedEnsemble'] = {
        'r2_test_mean': float(np.mean(ens_r2)), 'r2_test_std': float(np.std(ens_r2)),
        'r2_train_mean': float(np.mean(ens_r2tr)),
        'gap': float(np.mean(ens_r2tr) - np.mean(ens_r2)),
        'mae': float(np.mean(ens_mae)), 'time': 0}
    all_results['StackingMeta'] = {
        'r2_test_mean': float(np.mean(stk_r2)), 'r2_test_std': float(np.std(stk_r2)),
        'r2_train_mean': np.nan, 'gap': np.nan, 'mae': np.nan, 'time': 0}

    # Reference only: v4-style weights fitted on all folds (not a nested estimate)
    names_all, _ = candidates_for(-1, res)
    _, insample = optimise_weights(names_all, oos, list(range(OUTER_CV)))
    print(f"\n  Weighted ensemble (cross-fitted): R²={np.mean(ens_r2):.4f}±{np.std(ens_r2):.4f}")
    print(f"  Stacking meta-learner (cross-fitted): R²={np.mean(stk_r2):.4f}")
    print(f"  [reference] v4-style in-sample weights: R²={insample:.4f}")
    print(f"  Mean weights across folds: { {m: round(v, 3) for m, v in sorted(mean_w.items(), key=lambda x: -x[1])} }")

    with open(RESULTS_DIR / 'ensemble_v5.json', 'w') as f:
        json.dump({'folds': folds_info, 'mean_weights': mean_w,
                   'insample_reference_r2': float(insample)}, f, indent=2)
    pd.DataFrame(Y_ens).to_excel(RESULTS_DIR / 'predicted_serum_ensemble_full.xlsx', index=False)

    sorted_all = sorted(all_results.items(), key=lambda x: x[1]['r2_test_mean'], reverse=True)
    rows_ = []
    for n, r in sorted_all:
        mi_sel = ''
        if n in res:
            sel = [p.get('use_mi') for p in res[n]['fold_params'] if 'use_mi' in p]
            mi_sel = f"{sum(bool(s) for s in sel)}/{len(sel)}" if sel else \
                ('5/5' if n == 'SAN_MLP' else 'n/a')
        rows_.append({'Model': n, 'R2_test': r['r2_test_mean'], 'R2_std': r['r2_test_std'],
                      'R2_train': r['r2_train_mean'], 'Gap': r['gap'], 'MAE': r['mae'],
                      'MI_selected_folds': mi_sel, 'Time_s': r['time']})
    pd.DataFrame(rows_).to_excel(RESULTS_DIR / 'results_comparison_v5.xlsx', index=False)
    with open(RESULTS_DIR / 'results_v5.json', 'w') as f:
        json.dump(dict(sorted_all), f, indent=2, default=str)
    print(pd.DataFrame(rows_).to_string(index=False))


# ============================================================
# 10. MAIN
# ============================================================
if ARGS.ensemble:
    run_ensemble()
else:
    selected = list(REGISTRY) if ARGS.models == 'all' else ARGS.models.split(',')
    unknown = [m for m in selected if m not in REGISTRY]
    assert not unknown, f"unknown models: {unknown}"
    build_fold_cache()
    for name in selected:
        REGISTRY[name]()
        gc.collect()
    print("\nDONE:", ", ".join(selected))
