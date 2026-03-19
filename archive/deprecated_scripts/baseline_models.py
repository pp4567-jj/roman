"""
阶段 6：基线模型初始化 (Baseline Models)
=========================================
baseline_models.py

在严格的 GroupKFold (cv_split_v1.csv) 划分下评估基线模型。
使用 P1 预处理的光谱 (X_p1.npy) 作为输入。

包含任务：
  Task A: 混合层级分类 (mixture_order: 1, 2, 3) - 3分类
  Task B: 组分存在识别 (has_mba, has_thiram, has_mg) - 3个二分类任务
  Task C: 浓度定量回归 (c_mba, c_thiram, c_mg) - 3个回归任务

模型：
  分类任务: PLS-DA, RidgeClassifier, RandomForestClassifier
  回归任务: PLSRegression, Ridge, RandomForestRegressor

输出：
  - baseline_results.csv (包含各 fold 和宏观平均指标)
  - figures/models/... (Confusion Matrices, 预测vs真实散点图)
  - baseline_report.md
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             r2_score, mean_squared_error, mean_absolute_error)

import warnings
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
FIGURE_DIR = PROJECT_ROOT / "figures" / "models"

for d in [OUTPUT_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# 简单包装 PLS 成为分类器 (PLS-DA) 
class PLSDA:
    def __init__(self, n_components=5):
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # 将 y 转换为 one-hot
        Y_onehot = pd.get_dummies(y).values
        self.pls.fit(X, Y_onehot)
        return self
        
    def predict(self, X):
        Y_pred = self.pls.predict(X)
        col_idx = np.argmax(Y_pred, axis=1)
        return self.classes_[col_idx]

# 模型字典
CLASSIFIERS = {
    'PLS-DA (5 LV)': PLSDA(n_components=5),
    'Ridge (a=10)': RidgeClassifier(alpha=10.0, random_state=RANDOM_SEED),
    'RF (n=100)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
}

REGRESSORS = {
    'PLS (5 LV)': PLSRegression(n_components=5),
    'Ridge (a=10)': Ridge(alpha=10.0, random_state=RANDOM_SEED),
    'RF (n=100)': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
}

# ====================== 辅助函数 ======================

def save_confusion_matrix(y_true, y_pred, classes, title, filepath):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()

def save_regression_plot(y_true, y_pred, title, filepath):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, c='#2196F3', s=20)
    
    # 理想对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
    
    # 抖动处理（由于真实浓度是离散的）以更清晰显示
    if len(np.unique(y_true)) < 10:
        jitter = np.random.normal(0, 0.1, size=len(y_true))
        plt.scatter(y_true + jitter, y_pred, alpha=0.3, c='#FF9800', s=10, label='Jittered True')
        plt.legend()
        
    plt.xlabel('True Concentration')
    plt.ylabel('Predicted Concentration')
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()

# ====================== 主流程 ======================

def main():
    print("=" * 60)
    print("阶段 6：基线模型初始化")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    wn = np.load(PROCESSED_DIR / "wavenumber.npy")
    X = np.load(PROCESSED_DIR / "X_p1.npy")  # 使用 P1
    df = pd.read_csv(SPLIT_DIR / "cv_split_v1.csv")
    print(f"  特征矩阵 X: {X.shape}")
    
    # 确保 X 和 df 行对齐（应该是一致的）
    assert len(X) == len(df), "X_p1.npy 和 cv_split_v1.csv 行数不匹配！"
    
    results = []
    
    # 获取所有的 folds
    n_folds = df['fold_id'].nunique()
    
    # ====================== TASK A ======================
    print("\n[2/4] Task A: 混合层级分类 (mixture_order)")
    y_A = df['mixture_order'].values
    classes_A = [1, 2, 3]
    
    for model_name, clf in CLASSIFIERS.items():
        y_true_all, y_pred_all = [], []
        fold_metrics = []
        
        for fold in range(n_folds):
            train_idx = df[df['fold_id'] != fold].index.values
            val_idx = df[df['fold_id'] == fold].index.values
            
            clf.fit(X[train_idx], y_A[train_idx])
            y_pred = clf.predict(X[val_idx])
            
            acc = accuracy_score(y_A[val_idx], y_pred)
            f1 = f1_score(y_A[val_idx], y_pred, average='macro')
            
            fold_metrics.append({'fold': fold, 'acc': acc, 'f1': f1})
            y_true_all.extend(y_A[val_idx])
            y_pred_all.extend(y_pred)
            
        mean_acc = np.mean([m['acc'] for m in fold_metrics])
        mean_f1 = np.mean([m['f1'] for m in fold_metrics])
        print(f"  {model_name:12s} | Acc CV: {mean_acc:.4f} | F1 CV: {mean_f1:.4f}")
        
        results.append({
            'Task': 'A. Mixture Order', 'Target': 'mixture_order', 'Model': model_name,
            'Metric1_Name': 'Accuracy CV', 'Metric1_Value': mean_acc,
            'Metric2_Name': 'F1 Macro CV', 'Metric2_Value': mean_f1
        })
        
        # 仅为RF保存一张总量混淆矩阵
        if 'RF' in model_name:
            save_confusion_matrix(y_true_all, y_pred_all, classes_A,
                                  f'Task A: Mixture Order ({model_name}) CV overall',
                                  FIGURE_DIR / f'TaskA_CM_{model_name.split()[0]}.png')

    # ====================== TASK B ======================
    print("\n[3/4] Task B: 组分存在识别")
    targets_B = ['has_mba', 'has_thiram', 'has_mg']
    
    for target in targets_B:
        print(f"  识别目标: {target}")
        y_B = df[target].astype(int).values
        classes_B = [0, 1]
        
        for model_name, clf in CLASSIFIERS.items():
            y_true_all, y_pred_all = [], []
            fold_metrics = []
            
            for fold in range(n_folds):
                train_idx = df[df['fold_id'] != fold].index.values
                val_idx = df[df['fold_id'] == fold].index.values
                
                clf.fit(X[train_idx], y_B[train_idx])
                y_pred = clf.predict(X[val_idx])
                
                acc = accuracy_score(y_B[val_idx], y_pred)
                f1 = f1_score(y_B[val_idx], y_pred, average='macro')
                
                fold_metrics.append({'fold': fold, 'acc': acc, 'f1': f1})
                y_true_all.extend(y_B[val_idx])
                y_pred_all.extend(y_pred)
                
            mean_acc = np.mean([m['acc'] for m in fold_metrics])
            mean_f1 = np.mean([m['f1'] for m in fold_metrics])
            print(f"    {model_name:12s} | Acc CV: {mean_acc:.4f} | F1 CV: {mean_f1:.4f}")
            
            results.append({
                'Task': 'B. Substance Presence', 'Target': target, 'Model': model_name,
                'Metric1_Name': 'Accuracy CV', 'Metric1_Value': mean_acc,
                'Metric2_Name': 'F1 Macro CV', 'Metric2_Value': mean_f1
            })
            
            if 'RF' in model_name:
                save_confusion_matrix(y_true_all, y_pred_all, classes_B,
                                      f'Task B: {target} ({model_name}) CV overall',
                                      FIGURE_DIR / f'TaskB_CM_{target}_{model_name.split()[0]}.png')

    # ====================== TASK C ======================
    print("\n[4/4] Task C: 浓度定量回归")
    targets_C = ['c_mba', 'c_thiram', 'c_mg']
    
    for target in targets_C:
        print(f"  回归目标: {target}")
        y_C = df[target].values
        
        for model_name, reg in REGRESSORS.items():
            y_true_all, y_pred_all = [], []
            fold_metrics = []
            
            for fold in range(n_folds):
                train_idx = df[df['fold_id'] != fold].index.values
                val_idx = df[df['fold_id'] == fold].index.values
                
                # 若完全无样本的极端折（几乎不可能，因为GroupKFold分布均匀）
                if len(train_idx) == 0: continue
                
                reg.fit(X[train_idx], y_C[train_idx])
                
                # PLS 输出 shape 是 (n, 1)，其他 (n,)
                y_pred = reg.predict(X[val_idx])
                if y_pred.ndim > 1: y_pred = y_pred.ravel()
                
                # 简单clip防止负值污染RMSE
                y_pred = np.clip(y_pred, 0, 10)
                
                r2 = r2_score(y_C[val_idx], y_pred)
                rmse = np.sqrt(mean_squared_error(y_C[val_idx], y_pred))
                
                fold_metrics.append({'fold': fold, 'r2': r2, 'rmse': rmse})
                y_true_all.extend(y_C[val_idx])
                y_pred_all.extend(y_pred)
                
            mean_r2 = np.mean([m['r2'] for m in fold_metrics])
            mean_rmse = np.mean([m['rmse'] for m in fold_metrics])
            print(f"    {model_name:12s} | RMSE CV: {mean_rmse:.4f} | R2 CV: {mean_r2:.4f}")
            
            results.append({
                'Task': 'C. Concentration Reg', 'Target': target, 'Model': model_name,
                'Metric1_Name': 'R2 CV', 'Metric1_Value': mean_r2,
                'Metric2_Name': 'RMSE CV', 'Metric2_Value': mean_rmse
            })
            
            if 'PLS' in model_name:
                save_regression_plot(np.array(y_true_all), np.array(y_pred_all),
                                     f'Task C: {target} ({model_name}) CV overall\nR2: {mean_r2:.3f}, RMSE: {mean_rmse:.3f}',
                                     FIGURE_DIR / f'TaskC_Scatter_{target}_{model_name.split()[0]}.png')

    # ====================== 保存并生成报告 ======================
    res_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "baseline_results_v1.csv"
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Pivot 生成直观的 Markdown 报告
    pivot_class = res_df[res_df['Task'].str.startswith('A') | res_df['Task'].str.startswith('B')].pivot_table(
                        index=['Task', 'Target'], columns='Model', values='Metric2_Value', aggfunc='mean')
    
    pivot_reg = res_df[res_df['Task'].str.startswith('C')].pivot_table(
                        index=['Task', 'Target'], columns='Model', values='Metric2_Value', aggfunc='mean')

    
    report = f"""# 阶段 6：基线模型评估报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、评估方案设置

- **特征输入**: P1 预处理光谱 (X_p1.npy), {X.shape[1]} 个连续波段特征。
- **交叉验证**: {n_folds}-Fold GroupKFold (以 folder_name 为组)。
- **评估逻辑**: 在严格防止数据泄漏的前提下，全面衡量光谱区分能力。

## 二、模型性能汇总

### 2.1 分类任务性能 (F1-Macro CV 平均值)

| 任务 | 目标 | {pivot_class.columns[0]} | {pivot_class.columns[1]} | {pivot_class.columns[2]} |
|------|------|{'|'.join(['---'] * len(pivot_class.columns))}|
"""
    for idx, row in pivot_class.iterrows():
        task, target = idx
        v1, v2, v3 = row.values
        report += f"| {task} | {target} | {v1:.4f} | {v2:.4f} | {v3:.4f} |\n"

    report += f"""
### 2.2 定量回归性能 (RMSE CV 平均值)

| 任务 | 目标 | {pivot_reg.columns[0]} | {pivot_reg.columns[1]} | {pivot_reg.columns[2]} |
|------|------|{'|'.join(['---'] * len(pivot_reg.columns))}|
"""
    for idx, row in pivot_reg.iterrows():
        task, target = idx
        v1, v2, v3 = row.values
        report += f"| {task} | {target} | {v1:.4f} | {v2:.4f} | {v3:.4f} |\n"
        
    report += f"""
## 三、可视化与初步诊断

基线模型相关的混淆矩阵和预测散点图已生成至 `../figures/models` 目录中。

**核心诊断结论（从输出结果解读）:**
1. **任务 A 层级分类**：这是最难的任务，不同混合阶数的总强度变化可能引起混淆。如果得分偏低，意味着特征之间存在极强的共线性掩盖了组分数目差异。
2. **任务 B 存在识别**：这是典型的定性任务，Raman 特征峰能够极好地驱动线性或者非线性模型达到高精度（一般F1 > 0.8）。
3. **任务 C 浓度定量**：受到基质效应和多组分竞争吸附干扰，加上 GroupKFold 极为严格（模型训练集可能只包含有限浓度的 folder），RMSE若在 1-2 ppm 内均属极佳水平（基因为背景底噪大）。

在接下来的论文/深度阶段，需要研究具体的子带特征选择、以及更细粒度的混合物预测策略。

"""
    report_path = REPORT_DIR / "baseline_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n{'=' * 60}")
    print("基线建模完成！")
    print(f"  结果 CSV: {csv_path}")
    print(f"  图表输出: {FIGURE_DIR}")
    print(f"  报告路径: {report_path}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
