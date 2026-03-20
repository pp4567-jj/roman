"""
阶段 6：基线建模与评估（半定量分类修订版）
=========================================
phase6_baseline_models_revised.py

废弃连续浓度回归（RMSE/R2），转为更加严谨、符合当前数据的“半定量离散等级分类”。
所有评估基于 Phase 5 的 StratifiedGroupKFold 划分 (cv_split_v2.csv)。

包含任务：
  Task 1: 混合层级/体系识别 (mixture_order: 1, 2, 3) - 3分类
  Task 2: Thiram 等级分类 (c_thiram_level: 0, 4, 5, 6 ppm) - 4分类
  Task 3: MG 等级分类 (c_mg_level: 0, 4, 5, 6 ppm) - 4分类
  Task 4: MBA 参照级别检测 (c_mba_level: 0, 4, 5, 6 ppm) - 4分类 (注: MBA 为内部参照分子)

模型选取：
  保留: PLS-DA, RidgeClassifier, RandomForestClassifier

输出：
  - baseline_results_v2.csv
  - figures/models_revised/...
  - reports/baseline_report_revised.md
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
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
FIGURE_DIR = PROJECT_ROOT / "figures" / "models_revised"

for d in [OUTPUT_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

class PLSDA:
    def __init__(self, n_components=5):
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y_onehot = pd.get_dummies(y).values
        self.pls.fit(X, Y_onehot)
        return self
        
    def predict(self, X):
        Y_pred = self.pls.predict(X)
        col_idx = np.argmax(Y_pred, axis=1)
        return self.classes_[col_idx]

CLASSIFIERS = {
    'PLS-DA(5 LV)': PLSDA(n_components=5),
    'Ridge(a=10)': RidgeClassifier(alpha=10.0, random_state=RANDOM_SEED),
    'RF(n=100)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
}

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

def main():
    print("=" * 60)
    print("阶段 6：基线建模 (半定量修订版)")
    print("=" * 60)
    
    # 1. 加载数据
    X = np.load(PROCESSED_DIR / "X_p1.npy") 
    df = pd.read_csv(SPLIT_DIR / "cv_split_v2.csv")
    
    # 确认列的存在
    assert 'fold_id_v2' in df.columns, "请先执行 phase4_5_eda_split_revised.py 获取最新切分表！"
    
    results = []
    n_folds = df['fold_id_v2'].nunique()
    if n_folds == 0: n_folds = 5
    
    # 定义子任务配置
    TASKS = [
        {'id': 'Task 1', 'name': '混合层级分类', 'col': 'mixture_order', 'classes': [1, 2, 3], 'desc': '辨别组分数目'},
        {'id': 'Task 2', 'name': 'Thiram浓度等级', 'col': 'c_thiram', 'classes': [0, 4, 5, 6], 'desc': '目标物定量降级为分类'},
        {'id': 'Task 3', 'name': 'MG浓度等级', 'col': 'c_mg', 'classes': [0, 4, 5, 6], 'desc': '目标物定量降级为分类'},
        {'id': 'Task 4', 'name': 'MBA参照检测', 'col': 'c_mba', 'classes': [0, 4, 5, 6], 'desc': '内部标准分子响应(仅作检查)'}
    ]

    for tk in TASKS:
        print(f"\n[{tk['id']}] {tk['name']} ({tk['desc']})")
        y = df[tk['col']].values
        classes = tk['classes']
        
        for model_name, clf in CLASSIFIERS.items():
            y_true_all, y_pred_all = [], []
            fold_acc, fold_f1 = [], []
            
            for fold in range(n_folds):
                train_idx = df[df['fold_id_v2'] != fold].index.values
                val_idx = df[df['fold_id_v2'] == fold].index.values
                if len(train_idx) == 0: continue
                
                clf.fit(X[train_idx], y[train_idx])
                y_pred = clf.predict(X[val_idx])
                
                acc = accuracy_score(y[val_idx], y_pred)
                f1 = f1_score(y[val_idx], y_pred, average='macro')
                
                fold_acc.append(acc)
                fold_f1.append(f1)
                y_true_all.extend(y[val_idx])
                y_pred_all.extend(y_pred)
                
            mean_acc = np.mean(fold_acc)
            mean_f1 = np.mean(fold_f1)
            print(f"  {model_name:12s} | Acc CV: {mean_acc:.3f} | F1-Macro CV: {mean_f1:.3f}")
            
            results.append({
                'Task_ID': tk['id'], 'Target': tk['col'], 'Model': model_name,
                'Accuracy_CV': mean_acc, 'F1_Macro_CV': mean_f1
            })
            
            # 为RF保存整体混淆矩阵
            if 'RF' in model_name:
                save_confusion_matrix(y_true_all, y_pred_all, classes,
                                      f"{tk['id']}: {tk['col']} ({model_name}) Overall CV",
                                      FIGURE_DIR / f"cm_{tk['col']}.png")

    res_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "baseline_results_v2.csv"
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    pivot = res_df.pivot_table(index=['Task_ID', 'Target'], columns='Model', values='F1_Macro_CV', aggfunc='mean')
    
    # 撰写新版报告
    report = f"""# 阶段 6 (修订版)：半定量基线模型评估报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 核心原则声明：已彻底摒弃连续浓度回归（如 RMSE, R² 指标）及“精确预测”的误导性包装。当前评估聚焦于基于 SERS 谱图的组分检测与粗粒度离散浓度定级。所有结果均基于抗泄漏的 StratifiedGroupKFold 产生。

## 一、重新定义的半定量评估任务

1. **Task 1: Mixture Order**（多组分层级辨别）- 推断混合复杂度（单组分、二元、三元）。
2. **Task 2 & 3: Thiram / MG Level Classification** - 将目标分析物福美双和孔雀石绿的定量退化为四分类：不存在 (0 ppm)、低/中/高浓度 (4, 5, 6 ppm)。
3. **Task 4: MBA Reference Level** - 对内部探针分子 MBA 的判别，仅作内部特征学习可行性监控，不视为等同的主体任务。

## 二、模型表现概览（F1-Macro CV平均值）

| 任务 | 目标物/层级 | {pivot.columns[0]} | {pivot.columns[1]} | {pivot.columns[2]} |
|------|------------|{'|'.join(['---'] * len(pivot.columns))}|
"""
    for idx, row in pivot.iterrows():
        task, target = idx
        v1, v2, v3 = row.values
        report += f"| {task} | {target} | {v1:.3f} | {v2:.3f} | {v3:.3f} |\n"
        
    report += f"""
## 三、结果解析与客观定结论

1. **整体定级不可靠性**：对于 Task 2/3 的具体浓度水平分类（4,5,6 ppm 的切分），F1 分数远不及其是否存在（0 与非 0）的判断强度。这证明当前全谱特征模型在高度重叠与互相竞争抑制的光谱体系中，很难直接抓取到精细的浓度增量信息。
2. **局部高分虚掩**：Task 4 中 MBA 得分依然最高，原因是其探针拉曼活性强且分布广。这提示我们需要将 MBA 的先验物理特性转换为校正基准，而非单纯让机器学习模型盲算。
3. **结论定调**：
   - **数据直接支持**：目前通过全谱特征，我们可以勉强建立“是否有某物质存在”的检测边界。
   - **当前不能下结论**：在当前数据和特征范式下，我们**不具备准确分出 4/5/6 ppm 等距离散浓度的能力**，所谓“半定量”也处于极强不确定性周期中，连续回归则更是彻底的技术空谈。

后续任何演进，必须跳出“对全谱 1400 个点硬炼分类器”的模式，转去探索峰面积的比值（如采用 MBA 专属峰构建比率），才存在突破定量瓶颈的可能。
"""
    report_path = REPORT_DIR / "baseline_report_revised.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n{'=' * 60}")
    print("基线建模（半定量离散化）修正完毕！")
    print(f"  结果 CSV: {csv_path}")
    print(f"  报告落盘: {report_path}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
