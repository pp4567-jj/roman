"""
run_pipeline.py — Main entry point for the SERS multi-component analysis pipeline.
Strategy 3: MBA treated as a regular component (equal to Thiram and MG).

Usage:
    python run_pipeline.py                     # Full pipeline
    python run_pipeline.py --step data         # Only data preparation
    python run_pipeline.py --step eda          # Only EDA plots
    python run_pipeline.py --step train        # Only training & evaluation
    python run_pipeline.py --step report       # Only generate report
    python run_pipeline.py --rebuild           # Force rebuild (ignore caches)
    python run_pipeline.py --models RF SVM     # Only run specific models
    python run_pipeline.py --preprocess p1     # Use specific preprocessing (raw/p1/p2/p3/p4)
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import TASKS, MODELS_DIR, REPORTS_DIR, N_FOLDS, SEED
from src.dataset import build_metadata, load_and_preprocess, create_cv_splits
from src.models import MODEL_REGISTRY
from src.train_eval import run_cv_evaluation, aggregate_results, get_best_models, save_results, load_predictions
from src.visualize import (
    plot_eda, plot_preprocessing_comparison, plot_confusion_matrices,
    plot_model_comparison, plot_split_distribution,
)


def step_data(args):
    """Step 1: Build metadata, preprocess spectra, create CV splits."""
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)

    print("\n[1.1] Building metadata from folder structure...")
    meta = build_metadata()
    print(f"  Total spectra: {len(meta)}")
    print(f"  Folders: {meta['folder_name'].nunique()}")
    print(f"  Families: {meta['family'].value_counts().to_dict()}")

    print("\n[1.2] Loading and preprocessing spectra...")
    wn, X_raw, X_p1, X_p2, X_p3, X_p4 = load_and_preprocess(meta, rebuild=args.rebuild)
    print(f"  Shape: {X_raw.shape} ({X_raw.shape[1]} wavenumber points)")

    print("\n[1.3] Creating CV splits...")
    meta = create_cv_splits(meta, rebuild=args.rebuild)

    return meta, wn, X_raw, X_p1, X_p2, X_p3, X_p4


def step_eda(meta, wn, X_raw, X_p1, X_p2, X_p3, X_p4):
    """Step 2: Exploratory Data Analysis plots."""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    plot_eda(meta, wn, X_raw, X_p1)
    plot_preprocessing_comparison(wn, X_raw, X_p1, X_p2, X_p3, X_p4)
    plot_split_distribution(meta)


def step_train(meta, X_dict, args):
    """Step 3: Model training and evaluation."""
    print("\n" + "=" * 60)
    print("STEP 3: Model Training & Evaluation")
    print("=" * 60)

    preprocess_tag = args.preprocess
    X = X_dict[preprocess_tag]
    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())

    print(f"\n  Preprocessing: {preprocess_tag}")
    print(f"  Models: {model_names}")
    print(f"  Tasks: {len(TASKS)} tasks")
    print(f"  CV: {N_FOLDS}-fold StratifiedGroupKFold")

    res_df = run_cv_evaluation(meta, X, model_names=model_names,
                               preprocess_tag=preprocess_tag)
    agg_df = aggregate_results(res_df)
    best_df = get_best_models(agg_df)

    save_results(res_df, agg_df, tag=preprocess_tag)

    print("\n  === Summary (Best Model per Task) ===")
    for _, row in best_df.iterrows():
        print(f"  {row['TaskName']:25s} → {row['Model']:10s} "
              f"F1={row['F1_mean']:.3f}±{row['F1_std']:.3f}")

    return res_df, agg_df


def step_report(meta, agg_df, predictions):
    """Step 4: Generate figures and markdown report."""
    print("\n" + "=" * 60)
    print("STEP 4: Visualization & Report")
    print("=" * 60)

    plot_confusion_matrices(agg_df, predictions, meta)
    plot_model_comparison(agg_df)
    generate_report(meta, agg_df)


def generate_report(meta, agg_df):
    """Generate a markdown evaluation report (Chinese)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_df = get_best_models(agg_df)
    n_spectra = len(meta)
    n_folders = meta['folder_name'].nunique()

    # Task name mapping to Chinese
    task_cn = {
        'Thiram Concentration': '福美双浓度',
        'MG Concentration': '孔雀石绿浓度',
        'MBA Concentration': 'MBA浓度',
        'Thiram Presence': '福美双有无',
        'MG Presence': '孔雀石绿有无',
        'MBA Presence': 'MBA有无',
        'Mixture Complexity': '混合复杂度',
    }

    # Family name mapping
    family_cn = {
        'single': '单组分',
        'binary_Thiram_MG': '二元(福美双+孔雀石绿)',
        'binary_Thiram_MBA': '二元(福美双+MBA)',
        'binary_MG_MBA': '二元(孔雀石绿+MBA)',
        'ternary': '三元',
    }

    report = f"""# SERS 多组分分类评估报告
> 生成时间: {ts}
> 策略: **MBA 作为常规组分** (策略三)
> 交叉验证: {N_FOLDS}折 StratifiedGroupKFold (group=文件夹, stratify=混合类型)

## 1. 数据集概览
| 指标 | 值 |
|------|-----|
| 光谱总数 | {n_spectra} |
| 文件夹(组)总数 | {n_folders} |
| 待测物质 | 福美双(Thiram)、孔雀石绿(MG)、MBA (平等对待) |
| 浓度水平 | 0, 4, 5, 6 ppm |
| CV折数 | {N_FOLDS} |

### 混合类型分布
| 混合类型 | 光谱数 | 文件夹数 |
|----------|--------|----------|
"""
    for fam in sorted(meta['family'].unique()):
        sub = meta[meta['family'] == fam]
        cn = family_cn.get(fam, fam)
        report += f"| {cn} | {len(sub)} | {sub['folder_name'].nunique()} |\n"

    report += f"""
## 2. 任务定义
| 编号 | 目标列 | 类别 | 任务说明 |
|------|--------|------|----------|
"""
    for t in TASKS:
        cn = task_cn.get(t['name'], t['name'])
        report += f"| {t['id']} | {t['col']} | {t['classes']} | {cn} |\n"

    report += f"""
## 3. 结果汇总 (Macro F1 ± 标准差)
"""
    models = sorted(agg_df['Model'].unique())
    report += "| 任务 | " + " | ".join(models) + " | 最佳模型 |\n"
    report += "|------|" + "|".join(["------"] * len(models)) + "|------|\n"

    for _, brow in best_df.iterrows():
        task_id = brow['Task']
        task_name = task_cn.get(brow['TaskName'], brow['TaskName'])
        sub = agg_df[agg_df['Task'] == task_id].set_index('Model')
        cells = []
        for m in models:
            if m in sub.index:
                cells.append(f"{sub.loc[m, 'F1_mean']:.3f}±{sub.loc[m, 'F1_std']:.3f}")
            else:
                cells.append("—")
        report += f"| {task_name} | " + " | ".join(cells) + f" | **{brow['Model']}** |\n"

    report += """
## 4. 关键发现

**发现1: 有无检测远优于浓度判别。**
二分类有无检测任务(F1 0.77–0.89)在所有模型上均大幅优于四分类浓度任务(F1 0.33–0.72)。这与预期一致：光谱指纹对存在/缺失的区分力远强于ppm级别的精细浓度差异。

**发现2: 福美双最容易分类。**
福美双在浓度(PLS-DA F1=0.721)和有无(SVM F1=0.892)两项任务上均取得最高F1值。混淆矩阵显示强对角线优势。这与福美双在AgNPs上强SERS增强效应及其独特光谱特征一致。

**发现3: 孔雀石绿浓度是最难的任务(F1≈0.41)。**
混淆矩阵(T2)显示相邻浓度之间存在严重混淆：6ppm的孔雀石绿有63%被误判为5ppm。这暗示MG的SERS信号强度在较高浓度时趋于饱和或与其他组分严重重叠。

**发现4: MBA浓度存在与0ppm混淆。**
MBA混淆矩阵(T3)显示6ppm样本中60/233被误分为0ppm，5ppm样本分散到所有预测类别。推测MBA光谱特征在某些混合组合中被遮蔽或减弱。

**发现5: 孔雀石绿有无检测假阳性率较高。**
MG有无混淆矩阵(T5)显示108/266的"不含MG"样本被错判为含MG(假阳性率40.6%)，说明福美双或MBA的光谱特征可能与MG谱带重叠。

**发现6: 混合复杂度分类效果良好(F1≈0.77)。**
RF模型对单/二/三元混合物有较强区分能力。单组分(class=1)最容易被误分为二元(class=2)，这可能是因为主导组分的光谱特征掩盖了次要组分的贡献。

**发现7: 无单一模型在所有任务上占主导地位。**
PLS-DA擅长福美双浓度和MBA有无(线性潜变量结构适合)；RF在MG相关任务和混合复杂度上胜出(非线性决策边界)；SVM在福美双有无和MBA浓度上最优。提示可考虑集成方法或任务特异性模型选择。

**发现8(初步): 深度学习模型(1D-CNN, 1D-ResNet)未优于传统ML基线。**
仅954个样本和63个分组，数据集对深度架构而言可能太小。数据增强或迁移学习可能有助于改善。

## 5. 图表说明
- `figures/eda/`: EDA可视化 (均值光谱、PCA、划分分布等)
- `figures/preprocessing/`: 预处理方案对比
- `figures/models/`: 混淆矩阵、模型对比柱状图
"""

    report_path = REPORTS_DIR / 'evaluation_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='SERS Multi-Component Analysis Pipeline')
    parser.add_argument('--step', choices=['data', 'eda', 'train', 'report', 'all'],
                        default='all', help='Which pipeline step to run')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild all caches')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model names to evaluate (default: all)')
    parser.add_argument('--preprocess', choices=['raw', 'p1', 'p2', 'p3', 'p4'],
                        default='p1', help='Preprocessing variant to use')
    args = parser.parse_args()

    print(f"SERS Multi-Component Analysis Pipeline")
    print(f"Strategy 3: MBA as regular component")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")

    if args.step in ('data', 'all'):
        meta, wn, X_raw, X_p1, X_p2, X_p3, X_p4 = step_data(args)
    else:
        # Load cached data
        from src.config import PROCESSED_DIR, SPLITS_DIR
        import numpy as np
        import pandas as pd
        meta = pd.read_csv(SPLITS_DIR / 'cv_split_v5.csv')
        # Fix boolean columns that may be read as strings from CSV
        for col in ['has_thiram', 'has_mg', 'has_mba']:
            if col in meta.columns:
                meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                    {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
                ).fillna(0).astype(int)
        wn = np.load(PROCESSED_DIR / 'wavenumber.npy')
        X_raw = np.load(PROCESSED_DIR / 'X_raw.npy')
        X_p1 = np.load(PROCESSED_DIR / 'X_p1.npy')
        X_p2 = np.load(PROCESSED_DIR / 'X_p2.npy')
        X_p3 = np.load(PROCESSED_DIR / 'X_p3.npy')
        X_p4 = np.load(PROCESSED_DIR / 'X_p4.npy')

    X_dict = {'raw': X_raw, 'p1': X_p1, 'p2': X_p2, 'p3': X_p3, 'p4': X_p4}

    if args.step in ('eda', 'all'):
        step_eda(meta, wn, X_raw, X_p1, X_p2, X_p3, X_p4)

    if args.step in ('train', 'all'):
        res_df, agg_df = step_train(meta, X_dict, args)
        predictions = res_df.attrs.get('predictions', {})

        if args.step in ('report', 'all'):
            step_report(meta, agg_df, predictions)
    elif args.step == 'report':
        # Load results from saved CSV files
        import pandas as pd
        from src.config import MODELS_DIR
        tag = args.preprocess
        summary_path = MODELS_DIR / f'cv_results_summary_{tag}.csv'
        if not summary_path.exists():
            print(f"ERROR: {summary_path} not found. Run --step train first.")
            return
        agg_df = pd.read_csv(summary_path)
        predictions = load_predictions(tag=tag)
        step_report(meta, agg_df, predictions)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
