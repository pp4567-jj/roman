# 全项目总审计报告
> 生成时间: 2026-03-19 13:28:39

## 1. 哪些内容仍然把 MBA 当成普通第三目标物？
- **已修正**: 所有修订版报告和图表中，MBA 统一标注为 Probe/Internal Standard。
- **已归档**: 旧版 `baseline_models.py` (将 MBA 与 Thiram/MG 并列预测)。
- **已归档**: 旧版 `figures/models/TaskB_CM_has_mba_RF.png`。
- **当前现役**: Task6 (c_mba) 在结果表中明确标注为 "Internal Monitor"，不作主要成果。

## 2. 哪些内容仍然把连续回归当主线？
- **已归档**: `figures/models/TaskC_Scatter_*.png` (连续回归散点图，3张)。
- **已归档**: 旧版 `baseline_results_v1.csv` (含 RMSE/R² 指标)。
- **已归档**: 旧版 `baseline_models.py` (含 PLSRegression 连续回归)。
- **当前现役**: 所有任务均为分类任务，指标为 Macro-F1 和 Balanced Accuracy。

## 3. 哪些内容把 folder/group effect 错写成 batch effect？
- **已归档**: 旧版 `eda_report.md`, `split_report.md`, `eda_and_splits.py` (均含 "batch effect")。
- **已归档**: `round2_audit_report.md` (含 "batch effect" 表述)。
- **当前现役**: 所有现役报告统一使用 "folder-level dependence/clustering/technical similarity"。

## 4. 哪些内容的表、图、文字不一致？
- **已修正**: 旧版 models_revised 中 Task1-4 和 cm_* 两套命名并存，已删除 Task* 系列，统一保留 cm_* 系列。
- **已修正**: baseline_report_revised.md 现在直接引用 baseline_results_final.csv 中的数据生成表格。

## 5. 哪些内容不适合当前主线？
- **已归档**: 所有连续回归图、旧版汇报文件、旧版审计报告。
- **已归档**: `report_for_supervisor.md` (面向导师的包装文件)。

## 6. 重复/命名混乱问题
- **已清理**: `figures/models/` 整个目录已归档至 `archive/legacy_figures/models_old/`。
- **已清理**: `figures/models_revised/` 中的重复命名图 (Task1-4) 已删除。
- **已清理**: `data/splits/cv_split_v1.csv` 已归档至 `archive/legacy_splits/`。
- **已清理**: 根目录 `analyze_dataset.py` 已归档。
- **已清理**: `scripts/` 中的旧版脚本 (eda_and_splits.py, audit_and_evaluate.py, _check_inventory.py) 已归档。

## 7. 保留/归档/删除决策
见 `reports/file_cleanup_manifest.md`。
