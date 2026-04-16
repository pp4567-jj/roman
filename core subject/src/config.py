"""
Global configuration for the SERS multi-component analysis project.
Strategy 3: MBA treated as a regular component (NOT internal standard).
Round 4: 3-class concentration + Focal Loss + class_weight + 2nd-derivative features.
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

# ====================== DL Hyperparameters ======================
DL_LR = 3e-4
DL_EPOCHS = 200
DL_PATIENCE = 20
DL_BATCH = 128
DL_WEIGHT_DECAY = 1e-4
DL_VAL_FRAC = 0.15
DL_SCHEDULER = 'cosine'

# ====================== Substance Naming ======================
SUBSTANCE_CN = {'美': 'Thiram', 'K': 'MG', 'm': 'MBA'}
CONC_LEVELS = [0, 4, 5, 6]

# ====================== Task Definitions ======================
TASKS = [
    {'id': 'T1_thiram_conc',  'name': 'Thiram Concentration',  'col': 'c_thiram', 'classes': [0, 4, 5, 6]},
    {'id': 'T2_mg_conc',      'name': 'MG Concentration',      'col': 'c_mg',     'classes': [0, 4, 5, 6]},
    {'id': 'T3_mba_conc',     'name': 'MBA Concentration',     'col': 'c_mba',    'classes': [0, 4, 5, 6]},
    {'id': 'T4_thiram_pres',  'name': 'Thiram Presence',       'col': 'has_thiram','classes': [0, 1]},
    {'id': 'T5_mg_pres',      'name': 'MG Presence',           'col': 'has_mg',   'classes': [0, 1]},
    {'id': 'T6_mba_pres',     'name': 'MBA Presence',          'col': 'has_mba',  'classes': [0, 1]},
    {'id': 'T7_mixture_order','name': 'Mixture Complexity',    'col': 'mixture_order', 'classes': [1, 2, 3]},
]

TASKS_3CLASS = [
    {'id': 'T1b_thiram_3c', 'name': 'Thiram Semi-Quant', 'col': 'c_thiram_3c', 'classes': [0, 1, 2]},
    {'id': 'T2b_mg_3c', 'name': 'MG Semi-Quant', 'col': 'c_mg_3c', 'classes': [0, 1, 2]},
    {'id': 'T3b_mba_3c', 'name': 'MBA Semi-Quant', 'col': 'c_mba_3c', 'classes': [0, 1, 2]},
]

TASKS_MT_FULL = [
    {'id': 'T1b_thiram_3c', 'name': 'Thiram Semi-Quant', 'col': 'c_thiram_3c', 'classes': [0, 1, 2]},
    {'id': 'T2b_mg_3c', 'name': 'MG Semi-Quant', 'col': 'c_mg_3c', 'classes': [0, 1, 2]},
    {'id': 'T3b_mba_3c', 'name': 'MBA Semi-Quant', 'col': 'c_mba_3c', 'classes': [0, 1, 2]},
    {'id': 'T4_thiram_pres', 'name': 'Thiram Presence', 'col': 'has_thiram', 'classes': [0, 1]},
    {'id': 'T5_mg_pres', 'name': 'MG Presence', 'col': 'has_mg', 'classes': [0, 1]},
    {'id': 'T6_mba_pres', 'name': 'MBA Presence', 'col': 'has_mba', 'classes': [0, 1]},
    {'id': 'T7_mixture_order', 'name': 'Mixture Complexity', 'col': 'mixture_order', 'classes': [1, 2, 3]},
]

# ====================== KAN Hyperparameters ======================
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3

# ====================== Data Augmentation ======================
AUG_N = 4
AUG_ENABLED = True
MIXUP_ENABLED = True
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.1

# ====================== Focal Loss ======================
FOCAL_LOSS_GAMMA = 2.0
USE_CLASS_WEIGHT = True

# ====================== Preprocessing Names ======================
PREPROCESS_NAMES = {
    'raw': 'Interpolated raw spectra',
    'p1': 'Cosmic removal + SG smooth + ALS baseline + SNV',
    'p2': 'SG smooth + ALS baseline + 1st derivative + SNV',
    'p3': 'ALS baseline + Vector normalization',
    'p4': 'Cosmic removal + SG smooth + ALS baseline (no normalization)',
}
