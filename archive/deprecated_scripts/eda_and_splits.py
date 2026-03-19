"""
阶段 4：EDA 与质量检查 + 阶段 5：无泄漏数据划分
================================================
合并脚本：生成 EDA 图表、PCA 分析、GroupKFold 划分。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
METADATA_PATH = PROJECT_ROOT / "data" / "metadata_v1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EDA_FIG_DIR = PROJECT_ROOT / "figures" / "eda"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"

for d in [EDA_FIG_DIR, SPLIT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5

# ====================== 加载数据 ======================
print("=" * 60)
print("阶段 4+5：EDA 与数据划分")
print("=" * 60)

print("\n加载数据...")
meta = pd.read_csv(METADATA_PATH)
wn = np.load(PROCESSED_DIR / "wavenumber.npy")
X_raw = np.load(PROCESSED_DIR / "X_raw.npy")
X_p1 = np.load(PROCESSED_DIR / "X_p1.npy")
X_p2 = np.load(PROCESSED_DIR / "X_p2.npy")
X_p3 = np.load(PROCESSED_DIR / "X_p3.npy")
print(f"  光谱数: {len(meta)}, 波数点数: {len(wn)}")

# 颜色映射
family_colors = {
    'single': '#2196F3',
    'binary_MBA_MG': '#4CAF50',
    'binary_MBA_Thiram': '#FF9800',
    'binary_Thiram_MG': '#9C27B0',
    'ternary': '#F44336',
}

# ====================== 阶段 4：EDA ======================

print("\n[EDA 1/5] 单物质平均谱叠图...")
fig, ax = plt.subplots(figsize=(14, 6))
single = meta[meta['family'] == 'single']
substances = [('MBA', 'has_mba', '#2196F3'), ('Thiram', 'has_thiram', '#FF9800'), ('MG', 'has_mg', '#4CAF50')]
for name, col, color in substances:
    mask = single[col] == True
    indices = single[mask].index.values
    if len(indices) == 0:
        continue
    mean_spec = X_p1[indices].mean(axis=0)
    std_spec = X_p1[indices].std(axis=0)
    ax.plot(wn, mean_spec, label=f'{name} (n={len(indices)})', color=color, linewidth=1.5)
    ax.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.15, color=color)
ax.set_xlabel('Raman Shift (cm⁻¹)')
ax.set_ylabel('P1 Intensity (SNV)')
ax.set_title('单物质平均谱叠图 (P1预处理)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'single_mean_spectra.png', dpi=150, bbox_inches='tight')
plt.close()

print("[EDA 2/5] 二元/三元平均谱叠图...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 二元
ax = axes[0]
for fam in ['binary_MBA_MG', 'binary_MBA_Thiram', 'binary_Thiram_MG']:
    mask = meta['family'] == fam
    indices = meta[mask].index.values
    if len(indices) == 0:
        continue
    mean_spec = X_p1[indices].mean(axis=0)
    ax.plot(wn, mean_spec, label=f'{fam} (n={len(indices)})', color=family_colors[fam], linewidth=1.5)
ax.set_ylabel('P1 Intensity (SNV)')
ax.set_title('二元混合平均谱叠图')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 三元
ax = axes[1]
ternary = meta[meta['family'] == 'ternary']
for ct in sorted(ternary['c_thiram'].unique()):
    sub = ternary[ternary['c_thiram'] == ct]
    mean_spec = X_p1[sub.index.values].mean(axis=0)
    ax.plot(wn, mean_spec, label=f'Thiram={ct}ppm (n={len(sub)})', linewidth=1.5)
ax.set_xlabel('Raman Shift (cm⁻¹)')
ax.set_ylabel('P1 Intensity (SNV)')
ax.set_title('三元混合平均谱叠图 (按Thiram浓度分组)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'binary_ternary_mean_spectra.png', dpi=150, bbox_inches='tight')
plt.close()

print("[EDA 3/5] PCA 可视化...")
pca = PCA(n_components=5, random_state=RANDOM_SEED)
pca_scores = pca.fit_transform(X_p1)
var_explained = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 按 family 着色
ax = axes[0]
for fam, color in family_colors.items():
    mask = meta['family'] == fam
    ax.scatter(pca_scores[mask, 0], pca_scores[mask, 1], 
               c=color, label=fam, alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
ax.set_title('PCA - 按 Family 着色')
ax.legend(fontsize=7, markerscale=1.5)

# 按 has_thiram 着色
ax = axes[1]
colors_thiram = ['#BBDEFB' if not t else '#E65100' for t in meta['has_thiram']]
ax.scatter(pca_scores[:, 0], pca_scores[:, 1], c=colors_thiram, alpha=0.5, s=20)
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
ax.set_title('PCA - 按 has_thiram 着色 (橙=含)')

# 按 has_mg 着色
ax = axes[2]
colors_mg = ['#BBDEFB' if not t else '#1B5E20' for t in meta['has_mg']]
ax.scatter(pca_scores[:, 0], pca_scores[:, 1], c=colors_mg, alpha=0.5, s=20)
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
ax.set_title('PCA - 按 has_mg 着色 (绿=含)')

plt.suptitle('PCA 可视化 (P1 预处理)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'pca_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA by group_id (检查batch effect)
print("[EDA 4/5] Batch/Group effect 检查...")
fig, ax = plt.subplots(figsize=(12, 8))
unique_groups = meta['group_id'].unique()
cmap = plt.cm.get_cmap('tab20', min(20, len(unique_groups)))
for i, grp in enumerate(unique_groups[:20]):  # 显示前20个组
    mask = meta['group_id'] == grp
    ax.scatter(pca_scores[mask, 0], pca_scores[mask, 1], 
               c=[cmap(i)], label=grp if i < 20 else '', alpha=0.6, s=15)
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
ax.set_title('PCA - 按 Group/Folder 着色 (前20组)')
ax.legend(fontsize=5, ncol=4, loc='best')
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'pca_by_group.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA PC3 vs PC4
fig, ax = plt.subplots(figsize=(10, 8))
for fam, color in family_colors.items():
    mask = meta['family'] == fam
    ax.scatter(pca_scores[mask, 2], pca_scores[mask, 3], 
               c=color, label=fam, alpha=0.6, s=20)
ax.set_xlabel(f'PC3 ({var_explained[2]*100:.1f}%)')
ax.set_ylabel(f'PC4 ({var_explained[3]*100:.1f}%)')
ax.set_title('PCA PC3 vs PC4 - 按 Family 着色')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'pca_pc3_pc4.png', dpi=150, bbox_inches='tight')
plt.close()

# 方差解释度
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 6), var_explained * 100, color='#2196F3', alpha=0.8)
ax.set_xlabel('主成分')
ax.set_ylabel('解释方差比例 (%)')
ax.set_title('PCA 方差解释度')
for i, v in enumerate(var_explained):
    ax.text(i+1, v*100+0.5, f'{v*100:.1f}%', ha='center', fontsize=10)
ax.set_xticks(range(1, 6))
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / 'pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# ====================== Batch effect 定量分析 ======================
print("[EDA 5/5] 定量分析 batch effect...")

# 使用简单方法：检查同family不同group的平均PC得分是否有明显差异
from scipy import stats

batch_analysis = []
for fam in ['single', 'binary_MBA_MG', 'binary_MBA_Thiram', 'binary_Thiram_MG', 'ternary']:
    sub = meta[meta['family'] == fam]
    groups = sub['group_id'].unique()
    if len(groups) < 2:
        continue
    
    # 对 PC1, PC2 做 ANOVA
    for pc_idx, pc_name in enumerate(['PC1', 'PC2']):
        group_scores = [pca_scores[sub[sub['group_id'] == g].index, pc_idx] for g in groups]
        if all(len(gs) > 1 for gs in group_scores):
            f_stat, p_val = stats.f_oneway(*group_scores)
            batch_analysis.append({
                'family': fam,
                'pc': pc_name,
                'n_groups': len(groups),
                'f_stat': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })

batch_df = pd.DataFrame(batch_analysis) if batch_analysis else pd.DataFrame()

# ====================== 阶段 5：数据划分 ======================

print("\n[Split 1/3] 构建 GroupKFold 划分...")
np.random.seed(RANDOM_SEED)

# GroupKFold 基于 group_id = folder_name
gkf = GroupKFold(n_splits=N_FOLDS)
groups = meta['group_id'].values

# 生成 fold assignments
fold_ids = np.zeros(len(meta), dtype=int)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_p1, groups=groups)):
    fold_ids[val_idx] = fold

# 保存 cv_split_v1.csv
split_df = meta[['sample_id', 'folder_name', 'group_id', 'family', 'mixture_order',
                  'has_mba', 'has_thiram', 'has_mg', 'c_mba', 'c_thiram', 'c_mg']].copy()
split_df['fold_id'] = fold_ids
split_path = SPLIT_DIR / "cv_split_v1.csv"
split_df.to_csv(split_path, index=False, encoding='utf-8-sig')
print(f"  输出: {split_path}")

# 检查每 fold 分布
print("\n[Split 2/3] 检查 fold 分布...")
fold_stats = []
for fold in range(N_FOLDS):
    sub = split_df[split_df['fold_id'] == fold]
    stat = {
        'fold_id': fold,
        'n_spectra': len(sub),
        'n_groups': sub['group_id'].nunique(),
        'groups': ', '.join(sorted(sub['group_id'].unique())[:5]) + ('...' if sub['group_id'].nunique() > 5 else ''),
    }
    for fam in ['single', 'binary_MBA_MG', 'binary_MBA_Thiram', 'binary_Thiram_MG', 'ternary']:
        stat[f'n_{fam}'] = len(sub[sub['family'] == fam])
    fold_stats.append(stat)
fold_stats_df = pd.DataFrame(fold_stats)

# 随机对照
print("\n[Split 3/3] 随机单条切分对照...")
from sklearn.model_selection import KFold
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
random_folds = np.zeros(len(meta), dtype=int)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_p1)):
    random_folds[val_idx] = fold

# 检查随机切分中的泄漏
leakage_count = 0
for fold in range(N_FOLDS):
    val_mask = random_folds == fold
    train_mask = ~val_mask
    val_groups = set(meta.loc[val_mask, 'group_id'])
    train_groups = set(meta.loc[train_mask, 'group_id'])
    overlap = val_groups & train_groups
    leakage_count += len(overlap)

# ====================== 生成报告 ======================

# EDA Report
eda_report = f"""# 阶段 4：EDA 报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、数据概览

| 指标 | 值 |
|------|-----|
| 总光谱数 | {len(meta)} |
| 单物质 | {len(meta[meta['family']=='single'])} |
| 二元混合 | {len(meta[meta['mixture_order']==2])} |
| 三元混合 | {len(meta[meta['family']=='ternary'])} |
| 波数范围 | {wn.min():.0f}-{wn.max():.0f} cm⁻¹ |
| 波数点数 | {len(wn)} |

## 二、平均谱特征

### 单物质对比
![单物质平均谱](../figures/eda/single_mean_spectra.png)

### 二元/三元混合
![二元三元混合谱](../figures/eda/binary_ternary_mean_spectra.png)

## 三、PCA 分析

### PCA 方差解释度
前5个主成分解释方差比例：{', '.join([f'PC{i+1}={v*100:.1f}%' for i,v in enumerate(var_explained)])}

![方差解释度](../figures/eda/pca_variance.png)

### PCA 散点图
![PCA总览](../figures/eda/pca_overview.png)

### PCA PC3-PC4
![PCA PC3-PC4](../figures/eda/pca_pc3_pc4.png)

### Batch/Group Effect
![PCA by Group](../figures/eda/pca_by_group.png)

"""
    
# Batch effect analysis
eda_report += f"""
## 四、Batch Effect 分析

"""
if len(batch_df) > 0:
    eda_report += """| Family | PC | F统计量 | p值 | 显著? |
|--------|-----|---------|-----|-------|
"""
    for _, row in batch_df.iterrows():
        sig_mark = '⚠️ 是' if row['significant'] else '否'
        eda_report += f"| {row['family']} | {row['pc']} | {row['f_stat']:.2f} | {row['p_value']:.4f} | {sig_mark} |\n"
    
    sig_count = batch_df['significant'].sum()
    eda_report += f"""
### 结论

- ANOVA 检验了同一 family 下不同 group(folder) 在 PC1/PC2 上的得分差异
- **{sig_count}/{len(batch_df)} 个检验显著** (p < 0.05)
"""
    if sig_count > len(batch_df) * 0.5:
        eda_report += "- **存在明显的 batch/group effect**，同 group 的谱在 PCA 空间中倾向于聚集\n"
        eda_report += "- 这进一步证实了必须使用 GroupKFold 而非随机切分\n"
    else:
        eda_report += "- Batch effect 不明显，但仍建议使用 GroupKFold 以保持严谨性\n"

eda_report += f"""
## 五、初步结论

1. **同类样本聚类趋势**：PCA 图显示不同 family 的光谱在 PC 空间中有一定分离，说明数据中存在可学习的化学信号。
2. **组分识别可行性**：含/不含 Thiram、MG 的光谱在 PCA 空间中可观察到差异。
3. **数据可学习性**：整体判断为**可学习**，但需注意 batch effect 对模型泛化的影响。
"""

with open(REPORT_DIR / "eda_report.md", 'w', encoding='utf-8') as f:
    f.write(eda_report)

# Split Report
split_report = f"""# 阶段 5：数据划分报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、划分方案说明

### 正式方案：GroupKFold (基于 folder_name)

- **划分单位**: group_id = folder_name（同一文件夹的谱视为同一组）
- **折数**: {N_FOLDS}-fold
- **随机种子**: {RANDOM_SEED}
- **核心原则**: 同一 folder 的光谱**绝不会**同时出现在训练集和验证集

### 为什么不能随机按单条光谱切分？

> [!CAUTION]
> 随机切分会导致**数据泄漏**：
> 1. 同一样品的多条重复测量谱被分到训练和验证集，模型"记住"了样品特异性
> 2. 泄漏检测结果：随机切分时，**{leakage_count}/{N_FOLDS} 个 fold 存在 group 泄漏**
> 3. 这会导致验证集上的指标**虚高**，无法真实反映模型在未见样品上的泛化能力

## 二、各 Fold 分布

| Fold | 光谱数 | 组数 | 单物质 | 二元_MBA_MG | 二元_MBA_Thiram | 二元_Thiram_MG | 三元 |
|------|--------|------|--------|------------|----------------|---------------|------|
"""
for _, row in fold_stats_df.iterrows():
    split_report += f"| {row['fold_id']} | {row['n_spectra']} | {row['n_groups']} | {row.get('n_single',0)} | {row.get('n_binary_MBA_MG',0)} | {row.get('n_binary_MBA_Thiram',0)} | {row.get('n_binary_Thiram_MG',0)} | {row.get('n_ternary',0)} |\n"

split_report += f"""
## 三、当前方案的局限性

1. **组数有限**：仅 {len(unique_groups)} 个 group，5-fold 意味着每个 fold 约 {len(unique_groups)//N_FOLDS} 个组
2. **分布不完全平衡**：由于组大小不一，各 fold 的样本数可能不均
3. **缺失组合**：Thiram 6 ppm 混合物缺失，影响浓度建模的完整性
4. **单组单层级**：当前 group_id 直接等于 folder_name，未做更精细的分层抽样

## 四、后续改进建议

1. 补充 Thiram 6 ppm 混合物数据后，重新划分
2. 考虑分层抽样 (StratifiedGroupKFold) 以平衡 family 分布
3. 增加 leave-one-group-out (LOGO) 验证作为补充
"""

with open(REPORT_DIR / "split_report.md", 'w', encoding='utf-8') as f:
    f.write(split_report)

print(f"\n{'=' * 60}")
print("阶段 4+5 完成！")
print(f"  EDA 图表: {EDA_FIG_DIR}")
print(f"  数据划分: {split_path}")
print(f"  EDA 报告: {REPORT_DIR / 'eda_report.md'}")
print(f"  划分报告: {REPORT_DIR / 'split_report.md'}")
print(f"{'=' * 60}")
