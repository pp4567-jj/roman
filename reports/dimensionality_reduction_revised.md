# 降维可视化策略 (修订版)
> 生成时间: 2026-03-20 16:50:09

## 图表规范
- **A. 全局 PCA (按 family 着色)**: PC1=46.4%, PC2=32.7%。
- **B. PCA Variance Bar**: 前5主成分解释方差。
- **C. Folder-Level Mean PCA**: 每个 folder 缩减为1个质心，消除伪重复视觉膨胀。
- **D. 局部 PCA**: 按 family 子集分别展示。
- **E. UMAP (补充)**: 仅作非线性拓扑辅助，不替代 PCA。

## 使用原则
1. PCA 为主线降维，UMAP 仅补充。
2. 任何"分离度"观察须区分化学信号驱动 vs folder-level dependence 驱动。
