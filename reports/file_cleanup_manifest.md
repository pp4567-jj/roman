# 文件清理清单 (File Cleanup Manifest)
> 生成时间: 2026-03-19 13:28:39

## 现役文件 (Active)
| 路径 | 用途 |
|------|------|
| README.md | 项目唯一入口说明 |
| requirements.txt | Python依赖 |
| scripts/parse_metadata.py | 阶段2: 元数据解析 |
| scripts/preprocess_spectra.py | 阶段3: 预处理 |
| scripts/phase1_inventory.py | 阶段1: 数据清点 |
| scripts/run_realign.py | 总控重构脚本(历史) |
| scripts/full_rebuild.py | 本次总重构脚本 (唯一临时入口) |
| data/metadata_v1.csv | 主元数据 |
| data/splits/cv_split_v2.csv | 现役划分方案 |
| data/models/baseline_results_final.csv | 现役模型结果 |
| data/processed/*.npy | 预处理后的光谱数据 |
| figures/eda_revised/*.png | 现役EDA图表(6张) |
| figures/models_revised/cm_*.png | 现役混淆矩阵(6张) |
| figures/preprocessing/*.png | 预处理对比图 |
| reports/*.md | 所有现役报告 |

## 已归档文件 (Archived)
| 原路径 | 归档位置 | 原因 |
|--------|----------|------|
| figures/models/*.png | archive/legacy_figures/models_old/ | 含连续回归散点图和旧版CM |
| data/splits/cv_split_v1.csv | archive/legacy_splits/ | 旧版GroupKFold，已被V2替代 |
| scripts/eda_and_splits.py | archive/deprecated_scripts/ | 含batch effect术语 |
| scripts/audit_and_evaluate.py | archive/deprecated_scripts/ | 旧版审计脚本 |
| scripts/_check_inventory.py | archive/deprecated_scripts/ | 临时检查脚本 |
| scripts/baseline_models.py | archive/deprecated_scripts/ | 含连续回归和MBA并列 |
| scripts/phase4_5_eda_split_revised.py | archive/deprecated_scripts/ | 已由full_rebuild.py收口，阻断旧流程复现污染 |
| scripts/phase6_baseline_models_revised.py | archive/deprecated_scripts/ | 已由full_rebuild.py收口，阻断旧图表命名污染 |
| scripts/phase7_mba_reference_strategy.py | archive/deprecated_scripts/ | 已由full_rebuild.py收口 |
| analyze_dataset.py | archive/deprecated_scripts/ | 根目录遗留脚本 |
| reports/report_for_supervisor.md | archive/deprecated_reports/ | 导师汇报文件 |
| reports/current_research_state.md | archive/deprecated_reports/ | 含夸大数据判定，发生了一致性漂移，不作为当前真源 |
| reports/round2_audit_report.md | archive/deprecated_reports/ | 旧版审计(含batch effect) |
| reports/eda_report.md | archive/deprecated_reports/ | 旧版EDA(含batch effect) |
| reports/split_report.md | archive/deprecated_reports/ | 旧版划分报告 |
| reports/baseline_report.md | archive/deprecated_reports/ | 旧版建模报告(含回归) |
| data/models/baseline_results_v1.csv | archive/deprecated_reports/ | 旧版结果(含RMSE) |
| data/models/baseline_results_v2.csv | archive/deprecated_reports/ | 中间版本结果 |

## 已删除文件 (Deleted)
| 路径 | 原因 |
|------|------|
| figures/models_revised/Task1_mixture_order_RF.png | 与cm_mixture_order.png重复 |
| figures/models_revised/Task2_c_thiram_RF.png | 与cm_c_thiram.png重复 |
| figures/models_revised/Task3_c_mg_RF.png | 与cm_c_mg.png重复 |
| figures/models_revised/Task4_c_mba_RF.png | 与cm_c_mba.png重复 |
| figures/models/ (整个目录) | 已归档后删除 |
