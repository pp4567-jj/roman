"""
generate_report.py — 从 data/models/ 的 CSV 汇总生成训练结果可视化报告。
可反复运行，随着训练数据的增加自动更新。
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(ROOT) / 'data' / 'models'
REPORT_DIR = Path(ROOT) / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

# ── 1. Load & merge all summary CSVs ──
frames = []
for tag in ['raw', 'p1', 'p2', 'p3', 'p4']:
    path = DATA_DIR / f'cv_results_summary_{tag}.csv'
    if path.exists():
        df = pd.read_csv(path)
        frames.append(df)
if not frames:
    print("No data found in data/models/. Run training first.")
    sys.exit(1)
all_df = pd.concat(frames, ignore_index=True)

# ── 2. Define display categories ──
TASK_ORDER = [
    'T4_thiram_pres', 'T5_mg_pres', 'T6_mba_pres',
    'T1_thiram_conc', 'T2_mg_conc', 'T3_mba_conc',
    'T7_mixture_order',
    'T1b_thiram_3c', 'T2b_mg_3c', 'T3b_mba_3c',
]
TASK_SHORT = {
    'T4_thiram_pres': 'Thiram\nPresence',
    'T5_mg_pres': 'MG\nPresence',
    'T6_mba_pres': 'MBA\nPresence',
    'T1_thiram_conc': 'Thiram\nConc(4c)',
    'T2_mg_conc': 'MG\nConc(4c)',
    'T3_mba_conc': 'MBA\nConc(4c)',
    'T7_mixture_order': 'Mixture\nOrder',
    'T1b_thiram_3c': 'Thiram\n3-class',
    'T2b_mg_3c': 'MG\n3-class',
    'T3b_mba_3c': 'MBA\n3-class',
}

MODEL_CATEGORIES = {
    'ML全谱': ['RF', 'SVM', 'PLS-DA'],
    'ML特征': ['RF-feat', 'SVM-feat', 'PLS-DA-feat'],
    'DL全谱': ['1D-CNN', '1D-ResNet', 'Spectrum-KAN', 'KAN-CNN'],
    'DL特征': ['1D-CNN-feat', '1D-ResNet-feat', 'Feature-KAN', 'KAN-CNN-feat'],
    '集成': ['Ensemble'],
}

MODEL_ORDER = []
for cat_models in MODEL_CATEGORIES.values():
    MODEL_ORDER.extend(cat_models)

PREPROCESS_ORDER = ['raw', 'p1', 'p2', 'p3', 'p4']
METRICS = ['F1_mean', 'BA_mean', 'Acc_mean']
METRIC_LABELS = {'F1_mean': 'Macro F1', 'BA_mean': 'Balanced Accuracy', 'Acc_mean': 'Accuracy'}

# ── 3. Stats ──
available_models = sorted(all_df['Model'].unique())
available_tasks = sorted(all_df['Task'].unique())
total_expected = 15 * 10 * 5  # 750
total_actual = len(all_df)

# Best preprocessing per model+task (by F1)
best_pp = all_df.loc[all_df.groupby(['Task', 'Model'])['F1_mean'].idxmax()]

# ── 4. Generate markdown report ──
lines = []
lines.append(f"# 模型训练结果报告")
lines.append(f"")
lines.append(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 数据来源: `data/models/cv_results_summary_*.csv`")
lines.append(f"")
lines.append(f"## 训练进度")
lines.append(f"")
lines.append(f"- 已完成实验单元: **{total_actual}** / 750 ({total_actual*100//750}%)")
lines.append(f"- 已有模型: {len(available_models)} — {', '.join(available_models)}")
lines.append(f"- 已有任务: {len(available_tasks)}")
lines.append(f"")

# Progress matrix: model × preprocess
lines.append(f"### 模型×预处理 完成度矩阵")
lines.append(f"")
header = "| 模型 | " + " | ".join(PREPROCESS_ORDER) + " |"
sep = "|---" + "|---" * len(PREPROCESS_ORDER) + "|"
lines.append(header)
lines.append(sep)
for m in MODEL_ORDER:
    row = f"| {m} "
    for pp in PREPROCESS_ORDER:
        subset = all_df[(all_df['Model'] == m) & (all_df['Preprocess'] == pp)]
        n_tasks = len(subset)
        if n_tasks == 0:
            row += "| ⬜ "
        elif n_tasks == 10:
            row += "| ✅ "
        else:
            row += f"| 🔶{n_tasks}/10 "
    row += "|"
    lines.append(row)
lines.append(f"")
lines.append(f"✅ = 10/10 tasks 完成, 🔶 = 部分完成, ⬜ = 未开始")
lines.append(f"")

# ── 5. Best-preprocessing results table (main results) ──
lines.append(f"---")
lines.append(f"")
lines.append(f"## 最佳预处理下的模型性能")
lines.append(f"")
lines.append(f"每个 (模型, 任务) 取 F1 最高的预处理变体。")
lines.append(f"")

# Group tasks by category
task_groups = [
    ("定性检测 (二分类)", ['T4_thiram_pres', 'T5_mg_pres', 'T6_mba_pres']),
    ("浓度分类 (四分类)", ['T1_thiram_conc', 'T2_mg_conc', 'T3_mba_conc']),
    ("混合物复杂度", ['T7_mixture_order']),
    ("半定量 (三分类)", ['T1b_thiram_3c', 'T2b_mg_3c', 'T3b_mba_3c']),
]

for group_name, task_ids in task_groups:
    lines.append(f"### {group_name}")
    lines.append(f"")
    
    task_cols = [t for t in task_ids if t in available_tasks]
    if not task_cols:
        lines.append(f"*暂无数据*")
        lines.append(f"")
        continue
    
    header = "| 模型 | 类别 | " + " | ".join(TASK_SHORT.get(t, t) for t in task_cols) + " |"
    # Replace newlines in header
    header = header.replace('\n', ' ')
    sep = "|---|---" + "|---" * len(task_cols) + "|"
    lines.append(header)
    lines.append(sep)
    
    for cat_name, cat_models in MODEL_CATEGORIES.items():
        for m in cat_models:
            if m not in available_models:
                continue
            row = f"| **{m}** | {cat_name} "
            for t in task_cols:
                cell = best_pp[(best_pp['Task'] == t) & (best_pp['Model'] == m)]
                if len(cell) == 0:
                    row += "| — "
                else:
                    r = cell.iloc[0]
                    f1 = r['F1_mean']
                    ba = r['BA_mean']
                    acc = r['Acc_mean']
                    pp = r['Preprocess']
                    row += f"| {f1:.3f}/{ba:.3f}/{acc:.3f} ({pp}) "
            row += "|"
            lines.append(row)
    lines.append(f"")
    lines.append(f"*格式: F1/BA/Acc (最佳预处理)*")
    lines.append(f"")

# ── 6. Per-task best model ranking ──
lines.append(f"---")
lines.append(f"")
lines.append(f"## 各任务 Top-3 模型")
lines.append(f"")

for t in TASK_ORDER:
    if t not in available_tasks:
        continue
    t_data = best_pp[best_pp['Task'] == t].sort_values('F1_mean', ascending=False).head(3)
    tname = TASK_SHORT.get(t, t).replace('\n', ' ')
    lines.append(f"**{tname}** ({t}):")
    for i, (_, r) in enumerate(t_data.iterrows(), 1):
        lines.append(f"  {i}. {r['Model']} — F1={r['F1_mean']:.3f} BA={r['BA_mean']:.3f} Acc={r['Acc_mean']:.3f} ({r['Preprocess']})")
    lines.append(f"")

# ── 7. Preprocess comparison ──
lines.append(f"---")
lines.append(f"")
lines.append(f"## 预处理方法对比")
lines.append(f"")
lines.append(f"各预处理在所有 (模型, 任务) 上的平均 F1:")
lines.append(f"")

for pp in PREPROCESS_ORDER:
    pp_data = all_df[all_df['Preprocess'] == pp]
    if len(pp_data) == 0:
        continue
    avg_f1 = pp_data['F1_mean'].mean()
    n_exps = len(pp_data)
    lines.append(f"- **{pp}**: 平均F1={avg_f1:.3f} (n={n_exps})")
lines.append(f"")

# ── 8. Category summary ──
lines.append(f"---")
lines.append(f"")
lines.append(f"## 模型类别汇总")
lines.append(f"")
lines.append(f"| 类别 | 模型数 | 平均F1 | 最佳F1 | 最佳模型+任务 |")
lines.append(f"|---|---|---|---|---|")

for cat_name, cat_models in MODEL_CATEGORIES.items():
    cat_data = best_pp[best_pp['Model'].isin(cat_models)]
    if len(cat_data) == 0:
        lines.append(f"| {cat_name} | {len(cat_models)} | — | — | — |")
        continue
    avg_f1 = cat_data['F1_mean'].mean()
    best_row = cat_data.loc[cat_data['F1_mean'].idxmax()]
    lines.append(f"| {cat_name} | {len([m for m in cat_models if m in available_models])}/{len(cat_models)} | {avg_f1:.3f} | {best_row['F1_mean']:.3f} | {best_row['Model']} @ {best_row['Task']} |")
lines.append(f"")

# ── 9. Write report ──
report_path = REPORT_DIR / 'training_results.md'
report_path.write_text('\n'.join(lines), encoding='utf-8')
print(f"Report saved to {report_path}")
print(f"  {total_actual}/750 experiments ({total_actual*100//750}% complete)")
print(f"  {len(available_models)} models, {len(available_tasks)} tasks")
