#!/bin/bash
# =============================================================================
# Setup script — Advanced Serum Reconstruction Pipeline v4
# =============================================================================
#
# USO:
#   chmod +x setup.sh
#   ./setup.sh          # CPU only
#   ./setup.sh gpu      # Con soporte CUDA (NVIDIA GPU)
#
# =============================================================================

set -e

ENV_NAME="serum_recon"
GPU_MODE=${1:-cpu}

echo "============================================="
echo "  Serum Reconstruction Pipeline v4 — Setup"
echo "  Mode: $GPU_MODE"
echo "============================================="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda no encontrado. Instala Miniconda/Anaconda primero."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing env if present
conda env remove -n $ENV_NAME -y 2>/dev/null || true

echo ""
echo "[1/4] Creando entorno conda '$ENV_NAME' con Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y

echo ""
echo "[2/4] Instalando paquetes conda..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Core packages from conda-forge
conda install -c conda-forge numpy pandas scipy openpyxl scikit-learn \
    xgboost lightgbm optuna matplotlib seaborn jupyter ipykernel notebook -y

# PyTorch
echo ""
echo "[3/4] Instalando PyTorch ($GPU_MODE)..."
if [ "$GPU_MODE" = "gpu" ]; then
    # CUDA 12.1 — ajustar según tu versión de CUDA
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Pip-only packages
echo ""
echo "[4/4] Instalando paquetes pip (TabNet, TabPFN)..."
pip install pytorch-tabnet>=4.1 tabpfn>=2.0

# Register Jupyter kernel
python -m ipykernel install --user --name $ENV_NAME --display-name "Serum Reconstruction v4"

echo ""
echo "============================================="
echo "  INSTALACIÓN COMPLETA"
echo "============================================="
echo ""
echo "Para activar el entorno:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Para abrir el notebook:"
echo "  conda activate $ENV_NAME"
echo "  jupyter notebook Advanced_Serum_Reconstruction_v4.ipynb"
echo ""
echo "Verificación rápida:"
python -c "
import torch, optuna, xgboost, lightgbm, sklearn
print(f'  PyTorch:      {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  Optuna:       {optuna.__version__}')
print(f'  XGBoost:      {xgboost.__version__}')
print(f'  LightGBM:     {lightgbm.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    print(f'  TabNet:       OK')
except: print(f'  TabNet:       FALLO')
try:
    from tabpfn import TabPFNRegressor
    print(f'  TabPFN:       OK')
except: print(f'  TabPFN:       FALLO')
"
echo ""
echo "Si seleccionas kernel en Jupyter, elige: 'Serum Reconstruction v4'"