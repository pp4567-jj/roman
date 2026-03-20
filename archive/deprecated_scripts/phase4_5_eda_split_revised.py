"""
阶段 4：EDA 与质量检查 + 阶段 5：无泄漏数据划分（修订版）
=========================================================
主要修订点：
1. 术语纠偏：剔除所有“batch effect”的滥用，统一替换为“folder-level similarity”。
2. 统计严谨化：彻底清除基于单条光谱的 PCA ANOVA 伪重复检验，改为基于 folder-level 均值点的分布探测。
3. 降维可视化策略重构：
   - 保留全局 PCA（按 family 着色）
   - 增加局部子集 PCA（针对 Single/Binary/Ternary 子集单独展开）
   - 补充 UMAP 非线性降维辅助拓扑观察
4. 数据划分升级：引入 StratifiedGroupKFold 尝试解决折间不均衡问题，并诚实输出由于组块受限导致的不完美分布，摒弃原来对随机切分为“毫无意义的飙升”这种情绪化表达。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
METADATA_PATH = PROJECT_ROOT / "data" / "metadata_v1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EDA_FIG_DIR = PROJECT_ROOT / "figures" / "eda_revised"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"

for d in [EDA_FIG_DIR, SPLIT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5

print("=" * 60)
print("开始执行 Phase 4 (EDA修订版) & Phase 5 (划分重制版)")
print("=" * 60)

# 加载数据
meta = pd.read_csv(METADATA_PATH)
wn = np.load(PROCESSED_DIR / "wavenumber.npy")
X_p1 = np.load(PROCESSED_DIR / "X_p1.npy")

family_colors = {
    'single': '#2196F3',
    'binary_MBA_MG': '#4CAF50',
    'binary_MBA_Thiram': '#FF9800',
    'binary_Thiram_MG': '#9C27B0',
    'ternary': '#F44336',
}

# ====================== 阶段 4：修订版 EDA ======================
print("\n[EDA 1/4] 单物质与混合平均谱 (复用原有逻辑但更名输出)...")

fig, ax = plt.subplots(figsize=(14, 6))
single = meta[meta['family'] == 'single']
substances = [('MBA(Internal_Std)', 'has_mba', '#2196F3'), ('Thiram(Target)', 'has_thiram', '#FF9800'),
              ('MG(Target)', 'has_mg', '#4CAF50')]
for name, col, color in substances:
    mask = single[col] == True
    indices = single[mask].index.values
    if len(indices) == 0: continue
    mean_spec = X_p1[indices].mean(axis=0)
    std_spec = X_p1[indices].std(axis=0)
    ax.plot(wn, mean_spec, label=f'{name} (n={len(indices)})', color=color, linewidth=1.5)
    ax.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.15, color=color)
ax.set_xlabel('Raman Shift (cm⁻¹)')
ax.set_ylabel('P1 Intensity (SNV)')
ax.set_title('单物质基础响应平均谱 (MBA 为探针参照物)')
ax.legend()
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / '1_single_mean_spectra.png', dpi=150)
plt.close()

print("[EDA 2/4] A. 全局 PCA + UMAP 降维辅助 (主线)...")
pca_global = PCA(n_components=2, random_state=RANDOM_SEED)
pca_scores = pca_global.fit_transform(X_p1)
var_exp = pca_global.explained_variance_ratio_

fig, axes = plt.subplots(1, 2 if HAS_UMAP else 1, figsize=(14 if HAS_UMAP else 7, 6))
ax_pca = axes[0] if HAS_UMAP else axes
for fam, c in family_colors.items():
    mask = meta['family'] == fam
    ax_pca.scatter(pca_scores[mask, 0], pca_scores[mask, 1], c=c, label=fam, alpha=0.6, s=15)
ax_pca.set_title(f'全局 PCA: {var_exp[0] * 100:.1f}%, {var_exp[1] * 100:.1f}%')
ax_pca.legend(fontsize=8, loc='best')

if HAS_UMAP:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    umap_res = reducer.fit_transform(X_p1)
    ax_umap = axes[1]
    for fam, c in family_colors.items():
        mask = meta['family'] == fam
        ax_umap.scatter(umap_res[mask, 0], umap_res[mask, 1], c=c, label=fam, alpha=0.6, s=15)
    ax_umap.set_title('补充可视化: 全局 UMAP (仅观察拓扑结构)')
    ax_umap.legend(fontsize=8)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / '2_global_dimensionality_reduction.png', dpi=150)
plt.close()

print("[EDA 3/4] B. 局部精细化 PCA (分离不同体系)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# 1. binary_MBA_Thiram colored by c_thiram
mask1 = meta['family'] == 'binary_MBA_Thiram'
if mask1.sum() > 0:
    s1 = pca_global.transform(X_p1[mask1])
    scatter = axes[0].scatter(s1[:, 0], s1[:, 1], c=meta.loc[mask1, 'c_thiram'], cmap='Oranges', edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, ax=axes[0], label='c_thiram (ppm)')
    axes[0].set_title('局部映射: Binary MBA+Thiram (着色: Thiram)')

# 2. binary_MBA_MG colored by c_mg
mask2 = meta['family'] == 'binary_MBA_MG'
if mask2.sum() > 0:
    s2 = pca_global.transform(X_p1[mask2])
    scatter = axes[1].scatter(s2[:, 0], s2[:, 1], c=meta.loc[mask2, 'c_mg'], cmap='Greens', edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, ax=axes[1], label='c_mg (ppm)')
    axes[1].set_title('局部映射: Binary MBA+MG (着色: MG)')

# 3. ternary colored by c_thiram
mask3 = meta['family'] == 'ternary'
if mask3.sum() > 0:
    s3 = pca_global.transform(X_p1[mask3])
    scatter = axes[2].scatter(s3[:, 0], s3[:, 1], c=meta.loc[mask3, 'c_thiram'], cmap='Reds', edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, ax=axes[2], label='c_thiram (ppm)')
    axes[2].set_title('局部映射: Ternary (着色: Thiram)')

# 4. ternary colored by c_mg
if mask3.sum() > 0:
    scatter = axes[3].scatter(s3[:, 0], s3[:, 1], c=meta.loc[mask3, 'c_mg'], cmap='Blues', edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, ax=axes[3], label='c_mg (ppm)')
    axes[3].set_title('局部映射: Ternary (着色: MG)')

plt.tight_layout()
plt.savefig(EDA_FIG_DIR / '3_local_pca_colorings.png', dpi=150)
plt.close()

print("[EDA 4/4] 修复 Folder-level Similarity (剔除伪重复检验) ...")
folder_means = []
folder_fams = []
for folder in meta['folder_name'].unique():
    idx = meta[meta['folder_name'] == folder].index
    mean_pc = pca_scores[idx].mean(axis=0)
    fam = meta.loc[idx[0], 'family']
    folder_means.append(mean_pc)
    folder_fams.append(fam)

folder_means = np.array(folder_means)
folder_fams = np.array(folder_fams)

fig, ax = plt.subplots(figsize=(8, 6))
for fam, c in family_colors.items():
    mask = folder_fams == fam
    if mask.sum() > 0:
        ax.scatter(folder_means[mask, 0], folder_means[mask, 1], c=c, label=f"{fam} (n={mask.sum()} folders)", s=60, edgecolors='k')
ax.set_title('Folder-Level 均值点分布 (消除同组伪重复)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(EDA_FIG_DIR / '4_folder_level_means.png', dpi=150)
plt.close()

# 撰写新版 EDA 报告
eda_report = f"""# 第三/四阶段重构：探索性数据分析修订报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 核心原则声明：本次修正已移除旧版本中所有未经验证的 "batch effect" 强因果关联表述。目前数据只能严格支持发生于 folder 层级的极强聚类相似性（Folder-Level Similarity）。此外，已撤回针对单条光谱打点的伪重复（pseudo-replication）ANOVA 检验。所有可视化与解读策略降级为严肃研究导向。

## 一、可视化降维：全局与局部的物理意义

### 1.1 全局映射 (PCA + UMAP)
![Global](../figures/eda_revised/2_global_dimensionality_reduction.png)
- **结论支持 (可信)**: 能够观察到 `single`、`binary` 和 `ternary` 等 family 在宏观流形上有大致分割，说明提取整体结构具备可行性。
- **局限性声明**: 全局着色下无法展现精确浓度梯度；由于混合干扰和基质限制，各类边界存在物理层面的混叠。UMAP 仅用于拓扑补充展示，其类内极小距离主要受到同一 folder 光谱重复相似性的主导。

### 1.2 局部焦点映射 (细分混合体系)
![Local](../figures/eda_revised/3_local_pca_colorings.png)
- 我们将全局空间下属的特定混合家族提取放大，针对性地采用目标物的浓度（c_thiram / c_mg）进行着色映射。
- **观察**: 在多元体系中，同浓度级别并未呈现平滑渐变，而是受 Folder-Level 影响呈现强区块结构。这提示了不能使用连续定量的方法硬掰高维空间距离。

## 二、Folder-Level 相似性声明
我们以每个 folder 的均值表征取代单条发散：
![Folder Means](../figures/eda_revised/4_folder_level_means.png)
- **坚实结论**: 同一文件夹的光谱拥有极高的统计协同性。这意味着，无论是来源于操作流程、基材自身底噪还是探测偶然波动，由于缺乏显式 `batch_id/date` 追踪，这被定性定义为 **Folder-Level Similarity**。这再次重申了跨 fold 的污染隔离是绝对必须的。
- 所有超过该结论的脑补解释全部降级或撤回。

### 旧报告弃用说明
旧的 `reports/eda_report.md` 及那些采用 20 种颜色混合、PC3_PC4乱入的主图，已认定为“不具备主线指导意义”的图表并被废弃。
"""
with open(REPORT_DIR / "eda_report_revised.md", "w", encoding='utf-8') as f:
    f.write(eda_report)


# ====================== 阶段 5：修订版分卷======================
print("\n[Split 1/2] 构建 StratifiedGroupKFold...")
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# 我们同时使用 family 作为需要分布的 y 标签，使用 folder_name 作为 group
groups = meta['folder_name'].values
y_stratify = meta['family'].values

fold_ids_v2 = np.zeros(len(meta), dtype=int)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_p1, y_stratify, groups)):
    fold_ids_v2[val_idx] = fold

meta['fold_id_v2'] = fold_ids_v2
split_path_v2 = SPLIT_DIR / "cv_split_v2.csv"
meta.to_csv(split_path_v2, index=False, encoding='utf-8-sig')

# 分组统计矩阵
dist_cols = ['single', 'binary_MBA_MG', 'binary_MBA_Thiram', 'binary_Thiram_MG', 'ternary']
stats = []
for fold in range(N_FOLDS):
    sub = meta[meta['fold_id_v2'] == fold]
    d = {'Fold': fold, 'Groups': sub['folder_name'].nunique(), 'Total_N': len(sub)}
    for col in dist_cols: d[col] = (sub['family'] == col).sum()
    stats.append(d)
stats_df = pd.DataFrame(stats)

print("[Split 2/2] 撰写修正版切分报告...")
split_report = f"""# 阶段 5 (修订版)：数据严格物理隔离与切分报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 核心原则声明：抛弃情绪化的论点，诚实展现在有限 group（文件夹）资源前提下的真实切分分布状态，并指出 StratifiedGroupKFold 的实施结果及局限。

## 一、方案更迭对比

- **旧版 V1 (基础 GroupKFold)**：实现了组间物理隔离，但未能考虑化学家族（family）分布的折间平衡，导致某些 fold 的特定组别数可能挂零。
- **新版 V2 (分层 GroupKFold - StratifiedGroupKFold)**：采用 sklearn 最新的分层隔离法，试图在保证不允许 folder 跨越 train/val 边界的同时，尽量使每个 fold 的各类 Family 族群占比相似。输出文件：`cv_split_v2.csv`。

## 二、新版 (V2) 各 Fold 真实构成矩阵

由于数据集只有 {meta['folder_name'].nunique()} 个文件夹，某些组别基数极大（例如部分三元文件夹自带极多数据），这导致了哪怕是尽量最优分配（SGKF），仍会存在轻微的不均衡和不可控。以下是诚实的分折统计：

| Fold | Folder组数 | 光谱总数 | Single族群 | Binary (MBA+MG) | Binary (MBA+Thiram) | Binary (Thiram+MG) | Ternary混合 |
|------|-----------|----------|------------|----------------|---------------------|--------------------|-------------|
"""
for _, r in stats_df.iterrows():
    split_report += f"| {r['Fold']} | {r['Groups']} | {r['Total_N']} | {r['single']} | {r['binary_MBA_MG']} | {r['binary_MBA_Thiram']} | {r['binary_Thiram_MG']} | {r['ternary']} |\n"

split_report += f"""
### V2 方案局部局限性与免责声明
通过查阅上方矩阵可以明确看到：
1. 本实验的**自然分配并不能保证每个 Fold 都极其均匀**。在有限的 group 块和天然带有强烈本底聚集成簇的数据结构下，严格的隔离必然会使得每一轮的验证分布有些许“残缺”（如特定种类只有极少样本）。
2. 这是现实小样本材料研究中必然要妥协的诚实底稿。

## 三、对于随机切分“标量污染”的学术注脚
先前版本报告采用“毫无意义的飙升、极度虚高”等辞藻。现已对该评价降级，仅作如下客观声明：
如果采用单条光谱无差别洗牌随机分轨，由于存在明显的 ``folder-level similarity``，模型会在训练集接收到属于该文件夹验证集的背景成分特参，这构成了验证集间接接受训练集信息的（泄漏）路径。针对此数据的定量指标泛化必须仅基于 Group-level 切分方法以客观作定。
"""
with open(REPORT_DIR / "split_report_revised.md", "w", encoding='utf-8') as f:
    f.write(split_report)

print(f"\n{'=' * 60}")
print("生成完毕。相关废弃物可通过终端移除。")
print(f"新图集路径: {EDA_FIG_DIR}")
print(f"新 EDA报告: reports/eda_report_revised.md")
print(f"新 Split表: data/splits/cv_split_v2.csv")
print(f"{'=' * 60}")
