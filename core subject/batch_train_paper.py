"""
batch_train_paper.py — 论文所需实验的完整训练脚本

完整模型矩阵（15个单任务模型）:
  ML 全谱:   RF, SVM, PLS-DA
  ML 特征:   RF-feat, SVM-feat, PLS-DA-feat
  DL 全谱:   1D-CNN, 1D-ResNet, Spectrum-KAN, KAN-CNN
  DL 特征:   1D-CNN-feat, 1D-ResNet-feat, Feature-KAN, KAN-CNN-feat
  集成:      Ensemble (RF-feat + Feature-KAN)

Phase 1: ML 全谱 × 10 tasks × 5 variants
Phase 2: DL 全谱 × 10 tasks × 5 variants
Phase 3: ML 手工特征 × 10 tasks × 5 variants
Phase 4: DL 手工特征 × 10 tasks × 5 variants
Phase 5: Ensemble × 10 tasks × 5 variants

结果会安全合并到现有 CSV（不覆盖已有行）。
"""
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from src.config import SPLITS_DIR, PROCESSED_DIR, TASKS, TASKS_3CLASS
from src.models import MODEL_REGISTRY
from src.train_eval import run_cv_evaluation, aggregate_results, save_results

# ---------- Load data ----------
print("Loading data...", flush=True)
split_path = SPLITS_DIR / 'cv_split_v5.csv'
meta = pd.read_csv(split_path)
for col in ['has_thiram', 'has_mg', 'has_mba']:
    if col in meta.columns:
        meta[col] = meta[col].astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        ).fillna(0).astype(int)

wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
X_dict = {}
for tag in ['raw', 'p1', 'p2', 'p3', 'p4']:
    path = PROCESSED_DIR / f'X_{tag}.npy'
    if path.exists():
        X_dict[tag] = np.load(path)
        print(f"  Loaded {tag}: {X_dict[tag].shape}", flush=True)
print(f"Data loaded. {len(X_dict)} variants.\n", flush=True)

ALL_VARIANTS = list(X_dict.keys())
total_start = time.time()

# ============================================================
# PHASE 1: ML 全谱基线 (RF, SVM, PLS-DA)
# ============================================================
ML_MODELS = ['RF', 'SVM', 'PLS-DA']
print(f"{'='*60}\nPHASE 1: ML baselines (10 tasks × 5 variants)\n{'='*60}", flush=True)
for tag in ALL_VARIANTS:
    t0 = time.time()
    print(f"\n  ML on {tag} (7 tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=ML_MODELS, preprocess_tag=tag, wn=wn)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  ML on {tag} (3-class tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=ML_MODELS, preprocess_tag=tag, wn=wn, tasks_override=TASKS_3CLASS)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  ML on {tag} done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 2: DL 全谱 (1D-CNN, 1D-ResNet, Spectrum-KAN, KAN-CNN)
# 每个模型单独跑，方便断点续跑
# ============================================================
DL_MODELS = ['1D-CNN', '1D-ResNet', 'Spectrum-KAN', 'KAN-CNN']
print(f"\n{'='*60}\nPHASE 2: DL full-spectrum (10 tasks × 5 variants)\n{'='*60}", flush=True)
for tag in ALL_VARIANTS:
    for mname in DL_MODELS:
        t0 = time.time()
        print(f"\n{'='*60}\n{mname} on {tag} (7 tasks)\n{'='*60}", flush=True)
        res_df = run_cv_evaluation(meta, X_dict[tag], model_names=[mname], preprocess_tag=tag, wn=wn)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        print(f"  {mname} on {tag} (3-class tasks)...", flush=True)
        res_df = run_cv_evaluation(meta, X_dict[tag], model_names=[mname], preprocess_tag=tag, wn=wn, tasks_override=TASKS_3CLASS)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        print(f"  {mname} on {tag} done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 3: ML 手工特征 (RF-feat, SVM-feat, PLS-DA-feat)
# ============================================================
ML_FEAT_MODELS = ['RF-feat', 'SVM-feat', 'PLS-DA-feat']
print(f"\n{'='*60}\nPHASE 3: ML-feat (10 tasks × 5 variants)\n{'='*60}", flush=True)
for tag in ALL_VARIANTS:
    t0 = time.time()
    print(f"\n  ML-feat on {tag} (7 tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=ML_FEAT_MODELS, preprocess_tag=tag, wn=wn)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  ML-feat on {tag} (3-class tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=ML_FEAT_MODELS, preprocess_tag=tag, wn=wn, tasks_override=TASKS_3CLASS)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  ML-feat on {tag} done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 4: DL 手工特征 (1D-CNN-feat, 1D-ResNet-feat, Feature-KAN, KAN-CNN-feat)
# ============================================================
DL_FEAT_MODELS = ['1D-CNN-feat', '1D-ResNet-feat', 'Feature-KAN', 'KAN-CNN-feat']
print(f"\n{'='*60}\nPHASE 4: DL-feat (10 tasks × 5 variants)\n{'='*60}", flush=True)
for tag in ALL_VARIANTS:
    for mname in DL_FEAT_MODELS:
        t0 = time.time()
        print(f"\n{'='*60}\n{mname} on {tag} (7 tasks)\n{'='*60}", flush=True)
        res_df = run_cv_evaluation(meta, X_dict[tag], model_names=[mname], preprocess_tag=tag, wn=wn)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        print(f"  {mname} on {tag} (3-class tasks)...", flush=True)
        res_df = run_cv_evaluation(meta, X_dict[tag], model_names=[mname], preprocess_tag=tag, wn=wn, tasks_override=TASKS_3CLASS)
        agg_df = aggregate_results(res_df)
        save_results(res_df, agg_df, tag=tag)
        print(f"  {mname} on {tag} done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 5: Ensemble (RF-feat + Feature-KAN)
# ============================================================
print(f"\n{'='*60}\nPHASE 5: Ensemble (10 tasks × 5 variants)\n{'='*60}", flush=True)
for tag in ALL_VARIANTS:
    t0 = time.time()
    print(f"\n  Ensemble on {tag} (7 tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=['Ensemble'], preprocess_tag=tag, wn=wn)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  Ensemble on {tag} (3-class tasks)...", flush=True)
    res_df = run_cv_evaluation(meta, X_dict[tag], model_names=['Ensemble'], preprocess_tag=tag, wn=wn, tasks_override=TASKS_3CLASS)
    agg_df = aggregate_results(res_df)
    save_results(res_df, agg_df, tag=tag)
    print(f"  Ensemble on {tag} done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# Summary
# ============================================================
total_elapsed = time.time() - total_start
print(f"\n{'='*60}", flush=True)
print(f"ALL TRAINING COMPLETE in {total_elapsed/60:.1f} minutes", flush=True)
print(f"{'='*60}\n", flush=True)

for tag in ALL_VARIANTS:
    path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        models = sorted(df['Model'].unique())
        tasks = sorted(df['Task'].unique())
        print(f"  {tag}: {len(df)} rows, {len(models)} models, {len(tasks)} tasks")
        print(f"    Models: {models}")
