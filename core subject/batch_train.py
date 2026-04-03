"""
Batch training script: runs all models × all preprocessing variants.
ML models first (fast), then DL models, then MT models.
Results are accumulated via save_results() merge logic.
"""
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from src.config import SPLITS_DIR, PROCESSED_DIR, TASKS
from src.models import MODEL_REGISTRY
from src.train_eval import run_cv_evaluation, aggregate_results, save_results

# ---------- Load data once ----------
print("Loading data...")
meta = pd.read_csv(SPLITS_DIR / 'cv_split_v5.csv')
for col in ['has_thiram', 'has_mg', 'has_mba']:
    if col in meta.columns:
        meta[col] = meta[col].astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        ).fillna(0).astype(int)

wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
X_dict = {}
for tag in ['raw', 'p1', 'p2', 'p3', 'p4']:
    X_dict[tag] = np.load(PROCESSED_DIR / f'X_{tag}.npy')
print(f"Data loaded. Spectra shape: {X_dict['p1'].shape}")

# ---------- Training plan ----------
VARIANTS = ['raw', 'p1', 'p2', 'p3', 'p4']
ML_MODELS = ['RF', 'SVM', 'PLS-DA']
DL_MODELS = ['1D-CNN', '1D-ResNet']
MT_MODELS = ['MT-CNN', 'MT-ResNet', 'MT-KAN-CNN']

total_start = time.time()

# Phase 1: ML models (fast — skip per-variant if results already exist)
for tag in VARIANTS:
    summary_path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(summary_path):
        df_existing = pd.read_csv(summary_path)
        existing_ml = [m for m in ML_MODELS if m in df_existing['Model'].values]
        if len(existing_ml) == len(ML_MODELS):
            print(f"  ML on {tag}: all {len(ML_MODELS)} models exist, skipping")
            continue
    t0 = time.time()
    print(f"\n  ML on {tag}...")
    X = X_dict[tag]
    res_df = run_cv_evaluation(meta, X, model_names=ML_MODELS, preprocess_tag=tag)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  ML on {tag} done in {time.time()-t0:.0f}s")

# Phase 2: DL single-task models (slower)
for tag in VARIANTS:
    for mname in DL_MODELS:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"{mname} on {tag}")
        print(f"{'='*60}")
        X = X_dict[tag]
        res_df = run_cv_evaluation(meta, X, model_names=[mname], preprocess_tag=tag)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        elapsed = time.time() - t0
        print(f"  {mname} on {tag} done in {elapsed:.0f}s")

# Phase 3: MT models (one fit per fold, faster than ST-DL)
for tag in VARIANTS:
    for mname in MT_MODELS:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"{mname} on {tag}")
        print(f"{'='*60}")
        X = X_dict[tag]
        res_df = run_cv_evaluation(meta, X, model_names=[mname], preprocess_tag=tag)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        elapsed = time.time() - t0
        print(f"  {mname} on {tag} done in {elapsed:.0f}s")

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"ALL TRAINING COMPLETE in {total_elapsed/60:.1f} minutes")
print(f"{'='*60}")

# Final summary
for tag in VARIANTS:
    path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        models = sorted(df['Model'].unique())
        print(f"  {tag}: {len(df)} rows, {len(models)} models: {models}")
