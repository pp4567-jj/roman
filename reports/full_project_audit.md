# 全项目总审计报告
> 生成时间: 2026-03-20 16:50:09

## Split 版本变迁
- V1 (5-fold GroupKFold): 已归档 archive/legacy_splits/
- V2 (5-fold SGKF stratify=family): 已归档 archive/legacy_splits_v2/ (c_thiram 6ppm 3 fold 挂零)
- **V3 (3-fold SGKF stratify=c_thiram): 当前现役** (尽最大努力覆盖 c_thiram 各等级)

## 审计结论
- MBA 统一标注为 Probe/Internal Standard
- 连续回归已废弃，所有任务为分类
- 混淆矩阵按每任务最佳模型绘制（不再硬绑 RF）
- batch effect 统一改为 folder-level dependence/clustering
- c_thiram level 标注为受限结果
