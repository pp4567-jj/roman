# 阶段 4：EDA 报告

> 生成时间: 2026-03-14 19:21:59

## 一、数据概览

| 指标 | 值 |
|------|-----|
| 总光谱数 | 807 |
| 单物质 | 187 |
| 二元混合 | 355 |
| 三元混合 | 265 |
| 波数范围 | 400-1800 cm⁻¹ |
| 波数点数 | 1401 |

## 二、平均谱特征

### 单物质对比
![单物质平均谱](../figures/eda/single_mean_spectra.png)

### 二元/三元混合
![二元三元混合谱](../figures/eda/binary_ternary_mean_spectra.png)

## 三、PCA 分析

### PCA 方差解释度
前5个主成分解释方差比例：PC1=46.4%, PC2=32.7%, PC3=7.0%, PC4=2.5%, PC5=1.6%

![方差解释度](../figures/eda/pca_variance.png)

### PCA 散点图
![PCA总览](../figures/eda/pca_overview.png)

### PCA PC3-PC4
![PCA PC3-PC4](../figures/eda/pca_pc3_pc4.png)

### Batch/Group Effect
![PCA by Group](../figures/eda/pca_by_group.png)


## 四、Batch Effect 分析

| Family | PC | F统计量 | p值 | 显著? |
|--------|-----|---------|-----|-------|
| single | PC1 | 354.79 | 0.0000 | ⚠️ 是 |
| single | PC2 | 530.25 | 0.0000 | ⚠️ 是 |
| binary_MBA_MG | PC1 | 19217.69 | 0.0000 | ⚠️ 是 |
| binary_MBA_MG | PC2 | 3505.63 | 0.0000 | ⚠️ 是 |
| binary_MBA_Thiram | PC1 | 2776.11 | 0.0000 | ⚠️ 是 |
| binary_MBA_Thiram | PC2 | 2152.00 | 0.0000 | ⚠️ 是 |
| binary_Thiram_MG | PC1 | 87.10 | 0.0000 | ⚠️ 是 |
| binary_Thiram_MG | PC2 | 721.90 | 0.0000 | ⚠️ 是 |
| ternary | PC1 | 5938.61 | 0.0000 | ⚠️ 是 |
| ternary | PC2 | 4233.84 | 0.0000 | ⚠️ 是 |

### 结论

- ANOVA 检验了同一 family 下不同 group(folder) 在 PC1/PC2 上的得分差异
- **10/10 个检验显著** (p < 0.05)
- **存在明显的 batch/group effect**，同 group 的谱在 PCA 空间中倾向于聚集
- 这进一步证实了必须使用 GroupKFold 而非随机切分

## 五、初步结论

1. **同类样本聚类趋势**：PCA 图显示不同 family 的光谱在 PC 空间中有一定分离，说明数据中存在可学习的化学信号。
2. **组分识别可行性**：含/不含 Thiram、MG 的光谱在 PCA 空间中可观察到差异。
3. **数据可学习性**：整体判断为**可学习**，但需注意 batch effect 对模型泛化的影响。
