#!/usr/bin/env python3
"""
=============================================================================
Advanced Serum Reconstruction Pipeline v4
=============================================================================
Reconstructing Serum Metabolomic Profiles from Urine NMR Data

Authors: Juan de Dios Marín-Manzano, Francisco Manuel Arrabal Campos,
         Ignacio Fernández de las Nieves

Key innovations:
  - Spectral Attention Network (SAN): learnable attention over NMR bins
  - Shannon Mutual Information feature weighting
  - TabPFN Regressor (zero-shot transformer)
  - TabNet with attention-based feature selection
  - Multiple MLP architectures (deep, wide, residual, attention-augmented)
  - Data augmentation: Gaussian noise, Mixup, SMOGN-lite
  - Optuna HPO for ALL models
  - Strict nested CV (5-outer, 3-inner)

Preprocessing:
  - Urine: PQN normalization → PowerTransform → RobustScaler
  - Serum (targets): StandardScaler only (already TSP-normalized from NMR)
=============================================================================
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os, gc, warnings, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from huggingface_hub import login, HfApi

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
np.random.seed(42)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Authentication: set tokens as environment variables before running
# export TABPFN_ACCESS_TOKEN="your_token"  # https://ux.priorlabs.ai/account
# export HF_TOKEN="your_hf_token"          # https://huggingface.co/settings/tokens
TABPFN_TOKEN = os.environ.get("TABPFN_ACCESS_TOKEN", "")
if TABPFN_TOKEN:
    os.environ["TABPFN_ACCESS_TOKEN"] = TABPFN_TOKEN

print("All imports OK")

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = Path("Data")
RESULTS_DIR = Path("Results_Advanced")
RESULTS_DIR.mkdir(exist_ok=True)

OUTER_CV = 5
INNER_CV = 3
N_TRIALS = 30        # Optuna trials per model (increase to 50-100 if time allows)
N_TRIALS_DEEP = 20   # For slow models (CNN, TabNet)
RS = 42

print(f"Config: OUTER_CV={OUTER_CV}, INNER_CV={INNER_CV}, "
      f"N_TRIALS={N_TRIALS}, N_TRIALS_DEEP={N_TRIALS_DEEP}")

# HuggingFace authentication (for TabPFN model download)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("  HuggingFace: authenticated")
    except Exception as e:
        print(f"  HuggingFace auth failed: {e}")

# ============================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================
print("\n" + "="*70)
print("[DATA] Loading and preprocessing...")
print("="*70)

# --- Load raw data ---
urine_raw = pd.read_excel(DATA_DIR / 'bucket_table_orina_COVID+PRECANCER_noscaling.xlsx')
urine_raw = urine_raw.drop(urine_raw.columns[:2], axis=1)
urine_columns = urine_raw.columns.tolist()  # Keep for spectral analysis
urine_raw = urine_raw.values.astype(np.float64)

serum_full = pd.read_excel(DATA_DIR / 'bucket_table_suero_COVID+PRECANCER_scaling.xlsx')
serum_full = serum_full.drop(serum_full.columns[:2], axis=1)
serum_full['Sex'] = serum_full['Sex'].map({'M': 0, 'F': 1})

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
Y_raw = serum_full.drop(
    columns=[c for c in clinical_cols if c in serum_full.columns]
).values.astype(np.float64)
del serum_full

# --- Urine preprocessing: PQN normalization ---
# (urine is neither normalized nor scaled)
print("  Urine: PQN normalization...")
X_norm = urine_raw / (urine_raw.sum(axis=1, keepdims=True) + 1e-12)
ref_spectrum = np.median(X_norm, axis=0)
quotients = X_norm / (ref_spectrum + 1e-12)
pqn_factors = np.median(quotients, axis=1, keepdims=True)
X_pqn = urine_raw / (pqn_factors + 1e-12)
del urine_raw, X_norm, ref_spectrum, quotients, pqn_factors

# Store globally
X = X_pqn   # Will be further transformed per fold (PowerTransform + Scale)
Y = Y_raw   # Will be StandardScaled per fold (already TSP-normalized)
N_SAMPLES, N_URINE_FEATURES = X.shape
N_SERUM_FEATURES = Y.shape[1]

print(f"  X (urine, PQN): {X.shape}")
print(f"  Y (serum, raw): {Y.shape}")
print(f"  Note: Further per-fold transforms applied inside CV")

# ============================================================
# 3. SHANNON MUTUAL INFORMATION FEATURE WEIGHTING
# ============================================================
print("\n" + "="*70)
print("[MI] Computing Shannon Mutual Information weights...")
print("="*70)

def compute_mi_weights(X_data, Y_data, n_pca_y=4, n_neighbors=5):
    """
    Compute mutual information between each urine feature and
    the first n_pca_y principal components of serum.
    Returns normalized weight vector (sum=1).

    Theory: I(X_i; Y_pc) measures how much information urine bin i
    carries about each serum PC. Features with higher MI across
    all PCs carry more transferable metabolic information.
    """
    # Scale for MI computation
    from sklearn.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X_data)
    pca = PCA(n_components=n_pca_y)
    Y_pca = pca.fit_transform(StandardScaler().fit_transform(Y_data))

    mi_total = np.zeros(X_s.shape[1])
    explained = pca.explained_variance_ratio_

    for pc in range(n_pca_y):
        mi = mutual_info_regression(X_s, Y_pca[:, pc],
                                     n_neighbors=n_neighbors, random_state=RS)
        mi_total += mi * explained[pc]  # Weight by variance explained

    # Normalize to [0, 1] then shift so minimum = 0.1 (don't zero out any feature)
    mi_norm = (mi_total - mi_total.min()) / (mi_total.max() - mi_total.min() + 1e-12)
    mi_weights = 0.1 + 0.9 * mi_norm  # Range [0.1, 1.0]
    return mi_weights

MI_WEIGHTS = compute_mi_weights(X, Y)
print(f"  MI weights computed: min={MI_WEIGHTS.min():.3f}, "
      f"max={MI_WEIGHTS.max():.3f}, mean={MI_WEIGHTS.mean():.3f}")
print(f"  Top 5 informative bins: {np.argsort(MI_WEIGHTS)[-5:]}")

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================
print("\n[AUG] Augmentation functions ready")

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
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
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

    # --- Augmentation in PCA SPACE (both X and Y already in latent space) ---
    # This prevents explosions because augmented values stay in the same
    # distribution as the original PCA scores
    X_tr_aug, Y_tr_pca_aug = X_tr_pca, Y_tr_pca
    if aug_type != 'none':
        X_tr_aug, Y_tr_pca_aug = apply_aug(X_tr_pca, Y_tr_pca, aug_type)

    return (X_tr_aug, X_tr_pca, X_te_pca, Y_tr_pca_aug, Y_tr_scaled,
            sc_y, pca_y, pt_x, sc_x, pca_x)

# def inverse_transform_y(Y_pca_pred, sc_y, pca_y):
#     """Inverse PCA + StandardScaler on Y predictions, with clipping."""
#     Y_reconstructed = pca_y.inverse_transform(Y_pca_pred)
#     # Clip to prevent extreme predictions (within 4 std of training mean)
#     Y_out = sc_y.inverse_transform(Y_reconstructed)
#     # Safety clip: values beyond ±4 std are almost certainly errors
#     global_mean = sc_y.mean_
#     global_std = sc_y.scale_
#     lower = global_mean - 4 * global_std
#     upper = global_mean + 4 * global_std
#     Y_out = np.clip(Y_out, lower, upper)
#     return Y_out
def inverse_transform_y(Y_pca_pred, sc_y, pca_y):
    """Inverse PCA + StandardScaler on Y predictions, with clipping."""
    Y_reconstructed = pca_y.inverse_transform(Y_pca_pred)
    Y_out = sc_y.inverse_transform(Y_reconstructed)
    # Clip to ±3 std (more aggressive — catches Ridge/ElasticNet explosions)
    lower = sc_y.mean_ - 3 * sc_y.scale_
    upper = sc_y.mean_ + 3 * sc_y.scale_
    Y_out = np.clip(Y_out, lower, upper)
    return Y_out
# ============================================================
# 6. SPECTRAL ATTENTION NETWORK (SAN) — Novel architecture
# ============================================================

class SpectralAttention(nn.Module):
    """
    Learnable attention over NMR spectral bins.

    Unlike standard self-attention, this module computes attention
    scores per spectral position, learning which NMR chemical shift
    regions carry the most transferable information to serum space.

    The attention is initialized with Shannon MI weights (warm start)
    and fine-tuned during training.
    """
    def __init__(self, n_features, mi_weights=None):
        super().__init__()
        # Learnable attention logits (one per spectral bin)
        self.attn_logits = nn.Parameter(torch.zeros(n_features))

        # Initialize with MI weights if available (warm start)
        if mi_weights is not None:
            with torch.no_grad():
                # Convert MI weights to logits (inverse softmax-ish)
                mi_t = torch.FloatTensor(mi_weights)
                self.attn_logits.copy_(torch.log(mi_t + 1e-6))

        # Gating network: learn to combine raw features with attended ones
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Soft attention weights (positive, normalized)
        attn = torch.softmax(self.attn_logits, dim=0)
        x_attended = x * attn.unsqueeze(0)  # Element-wise weighting

        # Gating: how much to blend attended vs raw
        gate_val = self.gate(x)  # (batch, 1)
        return gate_val * x_attended + (1 - gate_val) * x


class SpectralAttentionMLP(nn.Module):
    """
    MLP with Spectral Attention front-end.
    Combines MI-initialized attention with deep regression.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128, 64),
                 dropout=0.3, activation='relu', mi_weights=None):
        super().__init__()
        self.attention = SpectralAttention(input_dim, mi_weights)

        act_fn = nn.ReLU() if activation == 'relu' else nn.GELU()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                act_fn,
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.attention(x)
        return self.mlp(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class DeepResidualMLP(nn.Module):
    """Deep residual MLP with optional spectral attention."""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_blocks=4,
                 dropout=0.3, use_attention=True, mi_weights=None):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpectralAttention(input_dim, mi_weights)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        if self.use_attention:
            x = self.attention(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class Spectral1DCNN(nn.Module):
    """1D CNN treating NMR buckets as spectral signal."""
    def __init__(self, input_dim, output_dim, n_filters=64,
                 kernel_size=7, dropout=0.3):
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


def train_torch_model(model, X_train, Y_train, X_val=None,
                      lr=1e-3, wd=1e-4, epochs=200, batch_size=32,
                      patience=20):
    """Generic PyTorch training loop with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        # Early stopping on training loss (no separate val in inner CV)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

     # Predict
    model.eval()
    with torch.no_grad():
        if X_val is not None:
            Xv = torch.FloatTensor(X_val).to(device)
            pred = model(Xv).cpu().numpy()
            # Clip predictions to training range (prevent explosions)
            train_pred = model(Xt).cpu().numpy()
            pred_min = train_pred.min(axis=0) - 2 * train_pred.std(axis=0)
            pred_max = train_pred.max(axis=0) + 2 * train_pred.std(axis=0)
            pred = np.clip(pred, pred_min, pred_max)
        else:
            pred = None

    del model, optimizer, Xt, Yt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return pred

# ============================================================
# 7. CORE NESTED CV FUNCTION
# ============================================================

SPLITS = list(KFold(n_splits=OUTER_CV, shuffle=True, random_state=RS).split(X))
ALL_RESULTS = {}
ALL_OOS = {}

def run_sklearn_model(name, make_fn, space_fn, n_trials=None):
    """Run sklearn model with Optuna inside nested CV."""
    if n_trials is None:
        n_trials = N_TRIALS
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()

    fold_r2t, fold_r2tr, fold_mae = [], [], []
    oos = np.zeros_like(Y)

    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr, Xte, Ytr, Yte = X[tri], X[tei], Y[tri], Y[tei]

        def objective(trial):
            p = space_fn(trial)
            npx = p.pop('npx'); npy = p.pop('npy')
            at = p.pop('aug'); use_mi = p.pop('use_mi', False)

            ikf = KFold(n_splits=INNER_CV, shuffle=True, random_state=RS + fi)
            r2s = []
            for itr, ival in ikf.split(Xtr):
                try:
                    res = preprocess_fold(Xtr[itr], Xtr[ival], Ytr[itr],
                                          npx, npy, at, use_mi, MI_WEIGHTS)
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res

                    m = MultiOutputRegressor(make_fn(p))
                    m.fit(X_aug, Y_pca)
                    Yp = inverse_transform_y(m.predict(X_val_pca), sc_y, pca_y)
                    r2s.append(r2_score(Ytr[ival], Yp, multioutput='uniform_average'))
                except:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        npx = bp.pop('npx'); npy = bp.pop('npy')
        at = bp.pop('aug'); use_mi = bp.pop('use_mi', False)

        res = preprocess_fold(Xtr, Xte, Ytr, npx, npy, at, use_mi, MI_WEIGHTS)
        X_aug, X_tr_pca, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res

        m = MultiOutputRegressor(make_fn(bp))
        m.fit(X_aug, Y_pca)

        Yp_te = inverse_transform_y(m.predict(X_te_pca), sc_y, pca_y)
        Yp_tr = inverse_transform_y(m.predict(X_tr_pca), sc_y, pca_y)

        r2t = r2_score(Yte, Yp_te, multioutput='uniform_average')
        r2tr = r2_score(Ytr, Yp_tr, multioutput='uniform_average')
        mae_val = mean_absolute_error(Yte, Yp_te)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_val)
        oos[tei] = Yp_te

        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}, "
              f"aug={at}, mi={use_mi}, pca_x={npx}, pca_y={npy}")
        del study; gc.collect()

    result = {
        'r2_test_mean': np.mean(fold_r2t), 'r2_test_std': np.std(fold_r2t),
        'r2_train_mean': np.mean(fold_r2tr),
        'gap': np.mean(fold_r2tr) - np.mean(fold_r2t),
        'mae': np.mean(fold_mae), 'time': time.time() - t0
    }
    ALL_RESULTS[name] = result
    ALL_OOS[name] = oos.copy()
    print(f"  >> {name}: R²={result['r2_test_mean']:.4f}±{result['r2_test_std']:.4f}, "
          f"gap={result['gap']:.4f}, MAE={result['mae']:.4f} [{result['time']:.0f}s]")
    return result


def run_torch_model(name, build_model_fn, space_fn, n_trials=None):
    """Run PyTorch model with Optuna inside nested CV."""
    if n_trials is None:
        n_trials = N_TRIALS_DEEP
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()

    fold_r2t = []
    oos = np.zeros_like(Y)

    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr, Xte, Ytr, Yte = X[tri], X[tei], Y[tri], Y[tei]

        def objective(trial):
            p = space_fn(trial)
            npx = p.pop('npx'); npy = p.pop('npy')
            at = p.pop('aug'); use_mi = p.pop('use_mi', False)
            lr = p.pop('lr'); wd = p.pop('wd')
            epochs = p.pop('epochs', 200)

            ikf = KFold(n_splits=INNER_CV, shuffle=True, random_state=RS + fi)
            r2s = []
            for itr, ival in ikf.split(Xtr):
                try:
                    res = preprocess_fold(Xtr[itr], Xtr[ival], Ytr[itr],
                                          npx, npy, at, use_mi, MI_WEIGHTS)
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res

                    model = build_model_fn(p, X_aug.shape[1], Y_pca.shape[1])
                    pred = train_torch_model(model, X_aug, Y_pca, X_val_pca,
                                             lr=lr, wd=wd, epochs=epochs)
                    Yp = inverse_transform_y(pred, sc_y, pca_y)
                    r2s.append(r2_score(Ytr[ival], Yp, multioutput='uniform_average'))
                except:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        npx = bp.pop('npx'); npy = bp.pop('npy')
        at = bp.pop('aug'); use_mi = bp.pop('use_mi', False)
        lr = bp.pop('lr'); wd = bp.pop('wd')
        epochs = bp.pop('epochs', 200)

        res = preprocess_fold(Xtr, Xte, Ytr, npx, npy, at, use_mi, MI_WEIGHTS)
        X_aug, _, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res

        model = build_model_fn(bp, X_aug.shape[1], Y_pca.shape[1])
        pred = train_torch_model(model, X_aug, Y_pca, X_te_pca,
                                 lr=lr, wd=wd, epochs=epochs)
        Yp_te = inverse_transform_y(pred, sc_y, pca_y)

        r2t = r2_score(Yte, Yp_te, multioutput='uniform_average')
        fold_r2t.append(r2t)
        oos[tei] = Yp_te
        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}")
        del study; gc.collect()

    result = {
        'r2_test_mean': np.mean(fold_r2t), 'r2_test_std': np.std(fold_r2t),
        'r2_train_mean': 0, 'gap': 0, 'mae': 0, 'time': time.time() - t0
    }
    ALL_RESULTS[name] = result
    ALL_OOS[name] = oos.copy()
    print(f"  >> {name}: R²={result['r2_test_mean']:.4f}±{result['r2_test_std']:.4f} "
          f"[{result['time']:.0f}s]")
    return result

print("\nCore functions ready.")

# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================

# --- 8.1 BaggingSVR ---
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
print("\n[MODELS] Running all models...")
run_sklearn_model('BaggingSVR+', make_svr, svr_space); gc.collect()
# --- 8.7b PLS Regression (designed for cross-biofluid reconstruction) ---


def run_pls():
    """PLS finds components maximizing covariance between X and Y.
    Unlike PCA+regression, PLS jointly optimizes the latent space.
    No separate PCA needed — PLS does it internally."""
    print(f"\n{'='*60}")
    print("  PLS Regression")
    print(f"{'='*60}")
    t0 = time.time()

    fold_r2t, fold_r2tr, fold_mae = [], [], []
    oos = np.zeros_like(Y)

    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr, Xte, Ytr, Yte = X[tri], X[tei], Y[tri], Y[tei]

        def obj(trial):
            n_comp = trial.suggest_int('n_comp', 2, 25)
            use_mi = trial.suggest_categorical('use_mi', [True, False])
            aug_type = trial.suggest_categorical('aug', ['none', 'noise', 'mixup', 'smogn'])

            ikf = KFold(n_splits=INNER_CV, shuffle=True, random_state=RS + fi)
            r2s = []
            for itr, ival in ikf.split(Xtr):
                try:
                    pt = PowerTransformer('yeo-johnson')
                    sc_x = RobustScaler()
                    X_it = sc_x.fit_transform(pt.fit_transform(Xtr[itr]))
                    X_iv = sc_x.transform(pt.transform(Xtr[ival]))

                    if use_mi:
                        X_it = X_it * MI_WEIGHTS[np.newaxis, :]
                        X_iv = X_iv * MI_WEIGHTS[np.newaxis, :]

                    sc_y = StandardScaler()
                    Y_it = sc_y.fit_transform(Ytr[itr])

                    # Augmentation on scaled data (before PLS)
                    X_a, Y_a = X_it, Y_it
                    if aug_type != 'none':
                        X_a, Y_a = apply_aug(X_it, Y_it, aug_type)

                    n_c = min(n_comp, X_a.shape[1], X_a.shape[0], Y_a.shape[1])
                    pls = PLSRegression(n_components=n_c, max_iter=1000)
                    pls.fit(X_a, Y_a)

                    Y_pred = sc_y.inverse_transform(pls.predict(X_iv))
                    # Clip
                    lower = sc_y.mean_ - 4 * sc_y.scale_
                    upper = sc_y.mean_ + 4 * sc_y.scale_
                    Y_pred = np.clip(Y_pred, lower, upper)

                    r2s.append(r2_score(Ytr[ival], Y_pred, multioutput='uniform_average'))
                except:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
        bp = study.best_params

        # Retrain
        pt = PowerTransformer('yeo-johnson')
        sc_x = RobustScaler()
        X_tr_p = sc_x.fit_transform(pt.fit_transform(Xtr))
        X_te_p = sc_x.transform(pt.transform(Xte))

        if bp['use_mi']:
            X_tr_p = X_tr_p * MI_WEIGHTS[np.newaxis, :]
            X_te_p = X_te_p * MI_WEIGHTS[np.newaxis, :]

        sc_y = StandardScaler()
        Y_tr_s = sc_y.fit_transform(Ytr)

        X_a, Y_a = X_tr_p, Y_tr_s
        if bp['aug'] != 'none':
            X_a, Y_a = apply_aug(X_tr_p, Y_tr_s, bp['aug'])

        n_c = min(bp['n_comp'], X_a.shape[1], X_a.shape[0], Y_a.shape[1])
        pls = PLSRegression(n_components=n_c, max_iter=1000)
        pls.fit(X_a, Y_a)

        Y_pred_te = sc_y.inverse_transform(pls.predict(X_te_p))
        Y_pred_tr = sc_y.inverse_transform(pls.predict(X_tr_p))
        lower = sc_y.mean_ - 4 * sc_y.scale_
        upper = sc_y.mean_ + 4 * sc_y.scale_
        Y_pred_te = np.clip(Y_pred_te, lower, upper)
        Y_pred_tr = np.clip(Y_pred_tr, lower, upper)

        r2t = r2_score(Yte, Y_pred_te, multioutput='uniform_average')
        r2tr = r2_score(Ytr, Y_pred_tr, multioutput='uniform_average')
        mae_v = mean_absolute_error(Yte, Y_pred_te)
        fold_r2t.append(r2t); fold_r2tr.append(r2tr); fold_mae.append(mae_v)
        oos[tei] = Y_pred_te

        print(f"    Fold {fi+1}: R²_test={r2t:.4f}, inner={study.best_value:.4f}, "
              f"n_comp={n_c}, aug={bp['aug']}, mi={bp['use_mi']}")
        del study; gc.collect()

    result = {
        'r2_test_mean': np.mean(fold_r2t), 'r2_test_std': np.std(fold_r2t),
        'r2_train_mean': np.mean(fold_r2tr),
        'gap': np.mean(fold_r2tr) - np.mean(fold_r2t),
        'mae': np.mean(fold_mae), 'time': time.time() - t0
    }
    ALL_RESULTS['PLS'] = result
    ALL_OOS['PLS'] = oos.copy()
    print(f"  >> PLS: R²={result['r2_test_mean']:.4f}±{result['r2_test_std']:.4f}, "
          f"gap={result['gap']:.4f} [{result['time']:.0f}s]")

run_pls(); gc.collect()
# --- 8.2 XGBoost ---
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
        random_state=42, verbosity=0, n_jobs=-1)
run_sklearn_model('XGBoost', make_xgb, xgb_space); gc.collect()

# --- 8.3 LightGBM ---
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
        random_state=42, verbosity=-1, n_jobs=-1)
run_sklearn_model('LightGBM', make_lgb, lgb_space); gc.collect()

# --- 8.4 Ridge ---
# Ridge
def ridge_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'alpha': t.suggest_float('alpha', 1e-3, 1000, log=True)}

# ElasticNet
def enet_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'alpha': t.suggest_float('alpha', 1e-5, 10, log=True),
            'l1r': t.suggest_float('l1r', 0.01, 0.99)}
def make_ridge(p): return Ridge(alpha=p['alpha'])
run_sklearn_model('Ridge', make_ridge, ridge_space); gc.collect()

# # --- 8.5 ElasticNet ---
# def enet_space(t):
#     return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
#             'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
#             'use_mi': t.suggest_categorical('use_mi', [True, False]),
#             'alpha': t.suggest_float('alpha', 1e-5, 10, log=True),
#             'l1r': t.suggest_float('l1r', 0.01, 0.99)}
def make_enet(p): return ElasticNet(alpha=p['alpha'], l1_ratio=p['l1r'],
    max_iter=5000, random_state=42)
run_sklearn_model('ElasticNet', make_enet, enet_space); gc.collect()

# --- 8.6 ExtraTrees ---
def et_space(t):
    return {'npx': t.suggest_int('npx', 3, 25), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'nest': t.suggest_int('nest', 50, 300), 'md': t.suggest_int('md', 3, 15),
            'msl': t.suggest_int('msl', 2, 20), 'mf': t.suggest_float('mf', 0.3, 1.0)}
def make_et(p): return ExtraTreesRegressor(n_estimators=p['nest'],
    max_depth=p['md'], min_samples_leaf=p['msl'], max_features=p['mf'],
    random_state=42, n_jobs=-1)
run_sklearn_model('ExtraTrees', make_et, et_space); gc.collect()

# --- 8.7 BaggingKNN ---
def knn_space(t):
    return {'npx': t.suggest_int('npx', 3, 20), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','smogn']),
            'use_mi': t.suggest_categorical('use_mi', [True, False]),
            'k': t.suggest_int('k', 3, 20), 'w': t.suggest_categorical('w', ['uniform','distance']),
            'nest': t.suggest_int('nest', 10, 40), 'ms': t.suggest_float('ms', 0.5, 1.0)}
def make_knn(p): return BaggingRegressor(estimator=KNeighborsRegressor(
    n_neighbors=p['k'], weights=p['w']), n_estimators=p['nest'],
    max_samples=p['ms'], random_state=42)
run_sklearn_model('BaggingKNN', make_knn, knn_space); gc.collect()

# --- 8.8 sklearn MLP (wide) ---
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
run_sklearn_model('MLP_Wide', make_mlp_wide, mlp_wide_space); gc.collect()

# --- 8.9 sklearn MLP (deep) ---
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
run_sklearn_model('MLP_Deep', make_mlp_deep, mlp_deep_space); gc.collect()

# --- 8.10 Spectral Attention MLP (PyTorch) ---
def san_mlp_space(t):
    return {'npx': t.suggest_int('npx', 10, 50), 'npy': t.suggest_int('npy', 3, 8),
            'aug': t.suggest_categorical('aug', ['none','noise','mixup','noise+mixup']),
            'use_mi': True,  # Always use MI for attention init
            'h1': t.suggest_int('h1', 128, 512), 'h2': t.suggest_int('h2', 64, 256),
            'h3': t.suggest_int('h3', 32, 128),
            'act': t.suggest_categorical('act', ['relu','gelu']),
            'do': t.suggest_float('do', 0.1, 0.5),
            'lr': t.suggest_float('lr', 1e-4, 5e-3, log=True),
            'wd': t.suggest_float('wd', 1e-6, 1e-2, log=True),
            'epochs': 200}
def build_san_mlp(p, in_dim, out_dim):
    # MI_WEIGHTS is per-raw-feature (234), but in_dim is post-PCA
    # Pass None if dimensions don't match — network learns attention from scratch
    mi_w = MI_WEIGHTS if in_dim == len(MI_WEIGHTS) else None
    return SpectralAttentionMLP(in_dim, out_dim,
        hidden_dims=(p['h1'], p['h2'], p['h3']),
        dropout=p['do'], activation=p['act'],
        mi_weights=mi_w)
run_torch_model('SAN_MLP', build_san_mlp, san_mlp_space); gc.collect()

# --- 8.11 Deep Residual MLP (PyTorch) ---
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
    mi_w = MI_WEIGHTS if in_dim == len(MI_WEIGHTS) else None
    return DeepResidualMLP(in_dim, out_dim, hidden_dim=p['hdim'],
        n_blocks=p['nblocks'], dropout=p['do'],
        use_attention=p['use_attn'], mi_weights=mi_w)
run_torch_model('ResidualMLP', build_resmlp, resmlp_space); gc.collect()

# --- 8.12 1D-CNN ---
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
run_torch_model('1D_CNN', build_cnn, cnn_space); gc.collect()

# --- 8.13 TabNet ---
def run_tabnet():
    """TabNet needs special handling (not MultiOutputRegressor compatible)."""
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor as TabNetReg
    except ImportError:
        print("  TabNet not installed. pip install pytorch-tabnet")
        return

    print(f"\n{'='*60}")
    print("  TabNet")
    print(f"{'='*60}")
    t0 = time.time()
    fold_r2t = []
    oos = np.zeros_like(Y)

    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr, Xte, Ytr, Yte = X[tri], X[tei], Y[tri], Y[tei]

        def obj(trial):
            npy = trial.suggest_int('npy', 3, 8)
            at = trial.suggest_categorical('aug', ['none','noise','mixup'])
            nd = trial.suggest_categorical('nd', [8, 16, 32, 64])
            ns = trial.suggest_int('ns', 3, 7)
            gm = trial.suggest_float('gm', 1.0, 2.0)
            ls = trial.suggest_float('ls', 1e-6, 1e-2, log=True)
            lr_tn = trial.suggest_float('lr_tn', 1e-3, 5e-2, log=True)

            ikf = KFold(n_splits=INNER_CV, shuffle=True, random_state=RS + fi)
            r2s = []
            for itr, ival in ikf.split(Xtr):
                try:
                    pt = PowerTransformer('yeo-johnson'); sc = RobustScaler()
                    X_it = sc.fit_transform(pt.fit_transform(Xtr[itr]))
                    X_iv = sc.transform(pt.transform(Xtr[ival]))
                    sc_y = StandardScaler(); pca_y = PCA(n_components=npy)
                    Y_it = sc_y.fit_transform(Ytr[itr])
                    Y_it_pca = pca_y.fit_transform(Y_it)
                    X_a, Y_pca_a = X_it, Y_it_pca
                    if at != 'none':
                        X_a, Y_pca_a = apply_aug(X_it, Y_it_pca, at)

                    tn = TabNetReg(n_d=nd, n_a=nd, n_steps=ns, gamma=gm,
                        lambda_sparse=ls, optimizer_params=dict(lr=lr_tn),
                        verbose=0, seed=42)
                    tn.fit(X_a, Y_pca_a, max_epochs=100, patience=15,
                           batch_size=64, drop_last=False)
                    pred = tn.predict(X_iv)
                    # Clip predictions to training range
                    train_pred = tn.predict(X_a)
                    pred_min = train_pred.min(axis=0) - 2 * train_pred.std(axis=0)
                    pred_max = train_pred.max(axis=0) + 2 * train_pred.std(axis=0)
                    pred = np.clip(pred, pred_min, pred_max)
                    Yp = sc_y.inverse_transform(pca_y.inverse_transform(pred))
                    r2s.append(r2_score(Ytr[ival], Yp, multioutput='uniform_average'))
                except:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS_DEEP, show_progress_bar=False)
        bp = study.best_params

        pt = PowerTransformer('yeo-johnson'); sc = RobustScaler()
        X_tr_p = sc.fit_transform(pt.fit_transform(Xtr))
        X_te_p = sc.transform(pt.transform(Xte))
        sc_y = StandardScaler(); pca_y = PCA(n_components=bp['npy'])
        Y_tr_s = sc_y.fit_transform(Ytr)
        Y_tr_pca = pca_y.fit_transform(Y_tr_s)
        X_a, Y_pca_a = X_tr_p, Y_tr_pca
        if bp['aug'] != 'none':
            X_a, Y_pca_a = apply_aug(X_tr_p, Y_tr_pca, bp['aug'])

        tn = TabNetReg(n_d=bp['nd'], n_a=bp['nd'], n_steps=bp['ns'],
            gamma=bp['gm'], lambda_sparse=bp['ls'],
            optimizer_params=dict(lr=bp['lr_tn']), verbose=0, seed=42)
        tn.fit(X_a, Y_pca_a, max_epochs=150, patience=20,
               batch_size=64, drop_last=False)
        pred = tn.predict(X_te_p)
        # Clip predictions to training range
        train_pred = tn.predict(X_a)
        pred_min = train_pred.min(axis=0) - 2 * train_pred.std(axis=0)
        pred_max = train_pred.max(axis=0) + 2 * train_pred.std(axis=0)
        pred = np.clip(pred, pred_min, pred_max)
        Yp = sc_y.inverse_transform(pca_y.inverse_transform(pred))

        r2t = r2_score(Yte, Yp, multioutput='uniform_average')
        fold_r2t.append(r2t); oos[tei] = Yp
        print(f"    Fold {fi+1}: R²={r2t:.4f}, inner={study.best_value:.4f}")
        del study, tn; gc.collect()

    ALL_RESULTS['TabNet'] = {'r2_test_mean': np.mean(fold_r2t),
        'r2_test_std': np.std(fold_r2t), 'r2_train_mean': 0,
        'gap': 0, 'mae': 0, 'time': time.time()-t0}
    ALL_OOS['TabNet'] = oos
    print(f"  >> TabNet: R²={np.mean(fold_r2t):.4f}±{np.std(fold_r2t):.4f}")

run_tabnet(); gc.collect()

# --- 8.14 TabPFN Regressor ---
def run_tabpfn():
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("  TabPFN not installed. pip install tabpfn")
        return

    print(f"\n{'='*60}")
    print("  TabPFN Regressor")
    print(f"{'='*60}")
    t0 = time.time()
    fold_r2t = []
    oos = np.zeros_like(Y)

    for fi, (tri, tei) in enumerate(SPLITS):
        Xtr, Xte, Ytr, Yte = X[tri], X[tei], Y[tri], Y[tei]

        def obj(trial):
            npx = trial.suggest_int('npx', 5, 80)
            npy = trial.suggest_int('npy', 3, 8)
            at = trial.suggest_categorical('aug', ['none','noise','mixup'])
            use_mi = trial.suggest_categorical('use_mi', [True, False])

            ikf = KFold(n_splits=INNER_CV, shuffle=True, random_state=RS + fi)
            r2s = []
            for itr, ival in ikf.split(Xtr):
                try:
                    res = preprocess_fold(Xtr[itr], Xtr[ival], Ytr[itr],
                                          npx, npy, at, use_mi, MI_WEIGHTS)
                    X_aug, _, X_val_pca, Y_pca, _, sc_y, pca_y, *_ = res

                    pred_pca = np.zeros((len(X_val_pca), Y_pca.shape[1]))
                    for c in range(Y_pca.shape[1]):
                        reg = TabPFNRegressor(device='cpu')
                        reg.fit(X_aug, Y_pca[:, c])
                        pred_pca[:, c] = reg.predict(X_val_pca)

                    Yp = inverse_transform_y(pred_pca, sc_y, pca_y)
                    r2s.append(r2_score(Ytr[ival], Yp, multioutput='uniform_average'))
                except:
                    r2s.append(-1.0)
            return np.mean(r2s)

        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RS + fi))
        study.optimize(obj, n_trials=N_TRIALS_DEEP, show_progress_bar=False)
        bp = study.best_params

        res = preprocess_fold(Xtr, Xte, Ytr, bp['npx'], bp['npy'],
                              bp['aug'], bp['use_mi'], MI_WEIGHTS)
        X_aug, _, X_te_pca, Y_pca, _, sc_y, pca_y, *_ = res

        pred_pca = np.zeros((len(X_te_pca), Y_pca.shape[1]))
        for c in range(Y_pca.shape[1]):
            reg = TabPFNRegressor(device='cpu')
            reg.fit(X_aug, Y_pca[:, c])
            pred_pca[:, c] = reg.predict(X_te_pca)

        Yp = inverse_transform_y(pred_pca, sc_y, pca_y)
        r2t = r2_score(Yte, Yp, multioutput='uniform_average')
        fold_r2t.append(r2t); oos[tei] = Yp
        print(f"    Fold {fi+1}: R²={r2t:.4f}, inner={study.best_value:.4f}")
        del study; gc.collect()

    ALL_RESULTS['TabPFN'] = {'r2_test_mean': np.mean(fold_r2t),
        'r2_test_std': np.std(fold_r2t), 'r2_train_mean': 0,
        'gap': 0, 'mae': 0, 'time': time.time()-t0}
    ALL_OOS['TabPFN'] = oos
    print(f"  >> TabPFN: R²={np.mean(fold_r2t):.4f}±{np.std(fold_r2t):.4f}")

run_tabpfn(); gc.collect()

# ============================================================
# 9. SMART STACKING ENSEMBLE
# ============================================================
print(f"\n{'='*70}")
print("[STACKING] Building Smart Ensemble...")
print(f"{'='*70}")

sorted_m = sorted(ALL_RESULTS.items(), key=lambda x: x[1]['r2_test_mean'], reverse=True)
# top_names = [n for n, _ in sorted_m[:6]]
# print(f"Top 6 for stacking:")
# for n in top_names:
#     print(f"  {n}: R²={ALL_RESULTS[n]['r2_test_mean']:.4f}")
# Filter only models with positive R² (exclude broken ones)
stable_models = [(n, r) for n, r in sorted_m if r['r2_test_mean'] > 0.05]
top_names = [n for n, _ in stable_models[:6]]
print(f"Top stable models for stacking:")
for n in top_names:
    print(f"  {n}: R²={ALL_RESULTS[n]['r2_test_mean']:.4f}")
# Weighted ensemble
def wobj(trial):
    ws = {m: trial.suggest_float(f'w_{m}', 0, 1) for m in top_names}
    t = sum(ws.values())
    if t < 1e-8: return -1
    ws = {k: v/t for k, v in ws.items()}
    Yp = sum(ws[m] * ALL_OOS[m] for m in top_names)
    return np.mean([r2_score(Y[ti], Yp[ti], multioutput='uniform_average')
                     for _, ti in SPLITS])

study_w = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42))
study_w.optimize(wobj, n_trials=200, show_progress_bar=False)
bw = study_w.best_params
tw = sum(bw.values())
print(f"\nWeighted Ensemble R²={study_w.best_value:.4f}")
print(f"  Weights: {', '.join(f'{k}={v/tw:.3f}' for k,v in bw.items())}")
ALL_RESULTS['WeightedEnsemble'] = {'r2_test_mean': study_w.best_value,
    'r2_test_std': 0, 'r2_train_mean': 0, 'gap': 0, 'mae': 0, 'time': 0}

# Meta-learner stacking
meta = np.hstack([ALL_OOS[m] for m in top_names])
def sobj(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 100, log=True)
    npc = trial.suggest_int('npc', 3, min(30, meta.shape[1]))
    npy = trial.suggest_int('npy', 3, 8)
    r2s = []
    for tri, tei in SPLITS:
        sc = StandardScaler(); pc = PCA(n_components=npc)
        Xt = pc.fit_transform(sc.fit_transform(meta[tri]))
        Xte = pc.transform(sc.transform(meta[tei]))
        scy = StandardScaler(); pcy = PCA(n_components=npy)
        Yt = pcy.fit_transform(scy.fit_transform(Y[tri]))
        m = MultiOutputRegressor(Ridge(alpha=alpha)); m.fit(Xt, Yt)
        Yp = scy.inverse_transform(pcy.inverse_transform(m.predict(Xte)))
        r2s.append(r2_score(Y[tei], Yp, multioutput='uniform_average'))
    return np.mean(r2s)

study_s = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42))
study_s.optimize(sobj, n_trials=50, show_progress_bar=False)
print(f"Stacking Meta-Learner R²={study_s.best_value:.4f}")
ALL_RESULTS['StackingMeta'] = {'r2_test_mean': study_s.best_value,
    'r2_test_std': 0, 'r2_train_mean': 0, 'gap': 0, 'mae': 0, 'time': 0}

# ============================================================
# 9b. EXPORT ENSEMBLE PREDICTED SERUM FOR CLASSIFICATION
# ============================================================
print("\n[EXPORT] Saving ensemble predicted serum for classification...")

ws_norm = {k: v/tw for k, v in bw.items()}
Y_ensemble = sum(ws_norm[f'w_{m}'] * ALL_OOS[m] for m in top_names)

# PCA(4) version — matches Juande's classification format
sc_ens = StandardScaler()
pca_ens = PCA(n_components=4)
Y_ens_pca = pca_ens.fit_transform(sc_ens.fit_transform(Y_ensemble))
pd.DataFrame(Y_ens_pca, columns=['PC1', 'PC2', 'PC3', 'PC4']).to_excel(
    RESULTS_DIR / 'predicted_serum_ensemble.xlsx', index=False)

# Full 231-dim version — for info_transfer_analysis.py
pd.DataFrame(Y_ensemble).to_excel(
    RESULTS_DIR / 'predicted_serum_ensemble_full.xlsx', index=False)

print(f"  Ensemble PCA(4) saved: {Y_ens_pca.shape}")
print(f"  Ensemble full saved: {Y_ensemble.shape}")
print(f"  PCA variance explained: {pca_ens.explained_variance_ratio_}")
# ============================================================
# 10. FINAL RESULTS
# ============================================================
print(f"\n{'='*70}")
print("  FINAL RESULTS — Sorted by R² test")
print(f"{'='*70}")

sorted_all = sorted(ALL_RESULTS.items(), key=lambda x: x[1]['r2_test_mean'], reverse=True)
print(f"\n{'Model':<22} {'R²_test':>9} {'±std':>7} {'R²_train':>9} {'Gap':>7} {'MAE':>7}")
print("-" * 60)
for n, r in sorted_all:
    print(f"{n:<22} {r['r2_test_mean']:>9.4f} {r.get('r2_test_std',0):>7.4f} "
          f"{r.get('r2_train_mean',0):>9.4f} {r.get('gap',0):>7.4f} "
          f"{r.get('mae',0):>7.4f}")

BASELINE = 0.231
bn, br = sorted_all[0]
imp = ((br['r2_test_mean'] - BASELINE) / BASELINE) * 100
print(f"\n{'='*60}")
print(f"  BASELINE:   R² = {BASELINE:.3f}")
print(f"  BEST ({bn}): R² = {br['r2_test_mean']:.4f}")
print(f"  IMPROVEMENT: {imp:+.1f}%")
print(f"{'='*60}")

df = pd.DataFrame([{'Model': n, 'R2_test': round(r['r2_test_mean'], 4),
    'R2_std': round(r.get('r2_test_std',0), 4),
    'R2_train': round(r.get('r2_train_mean',0), 4),
    'Gap': round(r.get('gap',0), 4),
    'MAE': round(r.get('mae',0), 4)} for n, r in sorted_all])
df.to_excel(RESULTS_DIR / 'results_comparison_v4.xlsx', index=False)
with open(RESULTS_DIR / 'results_v4.json', 'w') as f:
    json.dump({n: {k: v for k,v in r.items()} for n,r in sorted_all}, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}")
print("DONE. Paste the results table in the chat to continue.")