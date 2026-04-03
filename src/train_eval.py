"""
Training and evaluation: K-fold cross-validation loop, metrics aggregation.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report,
)

from src.config import TASKS, N_FOLDS, MODELS_DIR
from src.models import MODEL_REGISTRY


def run_cv_evaluation(meta: pd.DataFrame, X: np.ndarray,
                      model_names: list[str] = None,
                      task_ids: list[str] = None,
                      preprocess_tag: str = 'p1') -> pd.DataFrame:
    """Run cross-validated evaluation for specified models on specified tasks.

    Single-task models (RF, SVM, PLS-DA, 1D-CNN, 1D-ResNet):
        for task → for model → for fold → fit(X, y) / predict(X)

    Multi-task models (MT-CNN, MT-ResNet, MT-KAN-CNN):
        for model → for fold → fit(X, y_dict) / predict(X) → score all tasks
        One fit covers all tasks simultaneously.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())
    if task_ids is None:
        task_ids = [t['id'] for t in TASKS]

    tasks = [t for t in TASKS if t['id'] in task_ids]
    results = []
    all_preds = {}  # (task_id, model_name) -> (y_true_all, y_pred_all)

    # Separate single-task vs multi-task model names
    st_names = [m for m in model_names if not m.startswith('MT-')]
    mt_names = [m for m in model_names if m.startswith('MT-')]

    # --- Prepare y arrays for all tasks (shared by ST and MT paths) ---
    y_arrays = {}
    for tk in tasks:
        col_data = meta[tk['col']]
        if col_data.dtype == object:
            col_data = col_data.map({'True': 1, 'False': 0}).fillna(col_data)
        y_arrays[tk['id']] = col_data.values.astype(int)

    # ==================== Single-task models ====================
    if st_names:
        for tk in tasks:
            y = y_arrays[tk['id']]
            print(f"\n  Task: {tk['name']} ({tk['col']}), classes={tk['classes']}")

            for mname in st_names:
                if mname not in MODEL_REGISTRY:
                    print(f"    WARNING: model '{mname}' not in registry, skipping.")
                    continue

                yt_all, yp_all = [], []
                fold_f1s, fold_bas = [], []

                for fold in range(N_FOLDS):
                    tri = meta[meta['fold_id'] != fold].index.values
                    vai = meta[meta['fold_id'] == fold].index.values

                    clf = MODEL_REGISTRY[mname]()
                    if hasattr(clf, 'device'):
                        groups_train = meta.loc[tri, 'folder_name'].values
                        clf.fit(X[tri], y[tri], groups=groups_train)
                    else:
                        clf.fit(X[tri], y[tri])
                    pred = clf.predict(X[vai])

                    f1 = f1_score(y[vai], pred, average='macro', zero_division=0)
                    ba = balanced_accuracy_score(y[vai], pred)
                    fold_f1s.append(f1)
                    fold_bas.append(ba)
                    yt_all.extend(y[vai])
                    yp_all.extend(pred)

                    results.append({
                        'Task': tk['id'], 'TaskName': tk['name'],
                        'Model': mname, 'Preprocess': preprocess_tag,
                        'Fold': fold, 'MacroF1': f1, 'BalancedAcc': ba,
                    })

                mean_f1 = np.mean(fold_f1s)
                print(f"    {mname}: F1={mean_f1:.3f}±{np.std(fold_f1s):.3f}, "
                      f"BA={np.mean(fold_bas):.3f}±{np.std(fold_bas):.3f}")
                all_preds[(tk['id'], mname)] = (np.array(yt_all), np.array(yp_all))

    # ==================== Multi-task models ====================
    for mname in mt_names:
        if mname not in MODEL_REGISTRY:
            print(f"\n  WARNING: model '{mname}' not in registry, skipping.")
            continue

        print(f"\n  Multi-Task Model: {mname}")
        task_yt  = {tk['id']: [] for tk in tasks}
        task_yp  = {tk['id']: [] for tk in tasks}
        task_f1s = {tk['id']: [] for tk in tasks}
        task_bas = {tk['id']: [] for tk in tasks}

        for fold in range(N_FOLDS):
            tri = meta[meta['fold_id'] != fold].index.values
            vai = meta[meta['fold_id'] == fold].index.values

            y_dict_train = {tid: y_arrays[tid][tri] for tid in y_arrays}
            y_dict_val   = {tid: y_arrays[tid][vai] for tid in y_arrays}

            clf = MODEL_REGISTRY[mname]()
            groups_train = meta.loc[tri, 'folder_name'].values
            clf.fit(X[tri], y_dict_train, groups=groups_train)
            pred_dict = clf.predict(X[vai])

            for tk in tasks:
                tid = tk['id']
                f1 = f1_score(y_dict_val[tid], pred_dict[tid],
                              average='macro', zero_division=0)
                ba = balanced_accuracy_score(y_dict_val[tid], pred_dict[tid])
                task_f1s[tid].append(f1)
                task_bas[tid].append(ba)
                task_yt[tid].extend(y_dict_val[tid])
                task_yp[tid].extend(pred_dict[tid])
                results.append({
                    'Task': tid, 'TaskName': tk['name'],
                    'Model': mname, 'Preprocess': preprocess_tag,
                    'Fold': fold, 'MacroF1': f1, 'BalancedAcc': ba,
                })

        for tk in tasks:
            tid = tk['id']
            mf = np.mean(task_f1s[tid])
            sf = np.std(task_f1s[tid])
            mb = np.mean(task_bas[tid])
            sb = np.std(task_bas[tid])
            print(f"    {tk['name']}: F1={mf:.3f}±{sf:.3f}, BA={mb:.3f}±{sb:.3f}")
            all_preds[(tid, mname)] = (np.array(task_yt[tid]), np.array(task_yp[tid]))

    res_df = pd.DataFrame(results)
    res_df.attrs['predictions'] = all_preds
    return res_df


def aggregate_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level results to mean ± std per (Task, Model, Preprocess)."""
    agg = res_df.groupby(['Task', 'TaskName', 'Model', 'Preprocess']).agg(
        F1_mean=('MacroF1', 'mean'),
        F1_std=('MacroF1', 'std'),
        BA_mean=('BalancedAcc', 'mean'),
        BA_std=('BalancedAcc', 'std'),
    ).reset_index()
    return agg


def get_best_models(agg_df: pd.DataFrame) -> pd.DataFrame:
    """For each task, find the model with highest mean F1."""
    idx = agg_df.groupby('Task')['F1_mean'].idxmax()
    return agg_df.loc[idx].reset_index(drop=True)


def save_results(res_df: pd.DataFrame, agg_df: pd.DataFrame, tag: str = ''):
    """Save detailed and aggregated results to CSV.
    Merges with existing files so partial runs (e.g., ML then DL) accumulate."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f'_{tag}' if tag else ''

    # Merge detail CSV
    detail_path = MODELS_DIR / f'cv_results_detail{suffix}.csv'
    if detail_path.exists():
        old = pd.read_csv(detail_path)
        new_models = res_df['Model'].unique()
        old = old[~old['Model'].isin(new_models)]
        merged = pd.concat([old, res_df], ignore_index=True)
    else:
        merged = res_df
    merged.to_csv(detail_path, index=False, encoding='utf-8-sig')

    # Merge summary CSV
    summary_path = MODELS_DIR / f'cv_results_summary{suffix}.csv'
    if summary_path.exists():
        old_agg = pd.read_csv(summary_path)
        new_models = agg_df['Model'].unique()
        old_agg = old_agg[~old_agg['Model'].isin(new_models)]
        merged_agg = pd.concat([old_agg, agg_df], ignore_index=True)
    else:
        merged_agg = agg_df
    merged_agg.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # Merge predictions CSV
    predictions = res_df.attrs.get('predictions', {})
    if predictions:
        pred_records = []
        for (task_id, model_name), (yt, yp) in predictions.items():
            for t, p in zip(yt, yp):
                pred_records.append({
                    'task_id': task_id, 'model': model_name,
                    'y_true': int(t), 'y_pred': int(p),
                })
        new_pred = pd.DataFrame(pred_records)
        pred_path = MODELS_DIR / f'cv_predictions{suffix}.csv'
        if pred_path.exists():
            old_pred = pd.read_csv(pred_path)
            new_model_names = new_pred['model'].unique()
            old_pred = old_pred[~old_pred['model'].isin(new_model_names)]
            new_pred = pd.concat([old_pred, new_pred], ignore_index=True)
        new_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')

    print(f"  Results saved to {MODELS_DIR}")


def load_predictions(tag: str = '') -> dict:
    """Load predictions from CSV for report generation."""
    suffix = f'_{tag}' if tag else ''
    pred_path = MODELS_DIR / f'cv_predictions{suffix}.csv'
    if not pred_path.exists():
        return {}
    df = pd.read_csv(pred_path)
    predictions = {}
    for (task_id, model), grp in df.groupby(['task_id', 'model']):
        predictions[(task_id, model)] = (grp['y_true'].values, grp['y_pred'].values)
    return predictions
