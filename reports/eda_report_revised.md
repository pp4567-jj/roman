# 阶段 4：EDA 报告 (修订版)
> 生成时间: 2026-03-19 13:28:39

## 一、核心事实
1. 每个 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，是 **technical replicates**，不是独立样本。
2. 不同 folder 之间通常不是同一天测的，不能默认同条件。
3. 因此存在 **folder-level dependence / folder-level clustering**。
4. **不能**将此直接写成"已证实的 batch effect"——缺乏显式元数据（日期、操作员、基底批号）来确认具体来源。

## 二、现役图表清单
| 编号 | 文件名 | 用途 |
|------|--------|------|
| 1 | 1_single_mean_spectra.png | 单物质平均谱（MBA标注为Probe） |
| 2 | 2_global_pca_by_family.png | 全局PCA按family着色 |
| 3 | 3_pca_variance_bar.png | PCA方差解释度柱状图 |
| 4 | 4_folder_level_mean_pca.png | Folder均值点PCA（消除伪重复） |
| 5 | 5_local_pca_by_family.png | 各family子集的局部PCA |
| 6 | 6_umap_supplement.png | UMAP补充图（不替代PCA） |

## 三、关键观察
### 数据直接支持
- 不同 family 在 PC1-PC2 空间有结构性分离，说明化学信号可学习。
- 同一 folder 的光谱在 PCA 空间高度聚集（见图4），印证了 folder-level dependence。
- 随机切分会拆散技术重复，造成泄漏风险。

### 当前不能下结论
- 聚集现象的具体物理来源（操作员差异？基底批次？环境温度？）——缺乏元数据，无法确认。

## 四、统计检验声明
旧版基于单条光谱的 ANOVA 检验已被撤回（伪重复问题）。如需组间差异分析，须先聚合至 folder 级别均值后再进行，且仅作 exploratory analysis。
