"""
Global configuration for the SERS multi-component analysis project.
Strategy 3: MBA treated as a regular component (NOT internal standard).
"""
from pathlib import Path

# ====================== Paths ======================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "混合数据" / "美Km混合光谱"
OUTPUT_DIR = ROOT / "data"
PROCESSED_DIR = OUTPUT_DIR / "processed"
SPLITS_DIR = OUTPUT_DIR / "splits"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = ROOT / "figures"
FIG_EDA = FIGURES_DIR / "eda"
FIG_PREPROCESS = FIGURES_DIR / "preprocessing"
FIG_MODELS = FIGURES_DIR / "models"
REPORTS_DIR = ROOT / "reports"

# ====================== Preprocessing ======================
WN_MIN = 400.0
WN_MAX = 1800.0
WN_STEP = 1.0
SG_WINDOW = 11
SG_POLY = 3
ALS_LAM = 1e6
ALS_P = 0.01
ALS_NITER = 10
COSMIC_THRESHOLD = 7.0
COSMIC_WINDOW = 5

# ====================== Splitting ======================
N_FOLDS = 5
SEED = 42

# ====================== 1D-CNN / 1D-ResNet Hyperparameters ======================
DL_LR = 1e-3
DL_EPOCHS = 200      # more headroom with larger model; early stopping guards overfitting
DL_PATIENCE = 15
DL_BATCH = 256       # large batch for 3090-24GB; augmented data compensates SGD noise
DL_WEIGHT_DECAY = 1e-4
DL_VAL_FRAC = 0.15  # internal validation for early stopping

# ====================== Substance Naming ======================
# Folder naming convention: 美X=Thiram Xppm, KX=MG Xppm, mX=MBA Xppm
# e.g., 美4K5m6 = Thiram 4ppm + MG 5ppm + MBA 6ppm
SUBSTANCE_CN = {'美': 'Thiram', 'K': 'MG', 'm': 'MBA'}
CONC_LEVELS = [0, 4, 5, 6]  # possible ppm levels (0 = absent)

# ====================== Task Definitions (Strategy 3) ======================
# All three substances treated equally as target analytes
TASKS = [
    {'id': 'T1_thiram_conc',  'name': 'Thiram Concentration',  'col': 'c_thiram', 'classes': [0, 4, 5, 6]},
    {'id': 'T2_mg_conc',      'name': 'MG Concentration',      'col': 'c_mg',     'classes': [0, 4, 5, 6]},
    {'id': 'T3_mba_conc',     'name': 'MBA Concentration',     'col': 'c_mba',    'classes': [0, 4, 5, 6]},
    {'id': 'T4_thiram_pres',  'name': 'Thiram Presence',       'col': 'has_thiram','classes': [0, 1]},
    {'id': 'T5_mg_pres',      'name': 'MG Presence',           'col': 'has_mg',   'classes': [0, 1]},
    {'id': 'T6_mba_pres',     'name': 'MBA Presence',          'col': 'has_mba',  'classes': [0, 1]},
    {'id': 'T7_mixture_order','name': 'Mixture Complexity',    'col': 'mixture_order', 'classes': [1, 2, 3]},
]

# ====================== KAN Hyperparameters ======================
KAN_GRID_SIZE = 8
KAN_SPLINE_ORDER = 3

# ====================== Data Augmentation ======================
AUG_N = 3          # augmentation multiplier (3x → ~3000 training samples)
AUG_ENABLED = True  # toggle for ablation

# Preprocessing pipeline names
PREPROCESS_NAMES = {
    'raw': 'Interpolated raw spectra',
    'p1': 'Cosmic removal + SG smooth + ALS baseline + SNV',
    'p2': 'SG smooth + ALS baseline + 1st derivative + SNV',
    'p3': 'ALS baseline + Vector normalization',
    'p4': 'Cosmic removal + SG smooth + ALS baseline (no normalization)',
}
