# SERS Multi-Component Analysis Framework

## 1. 项目目标
基于 AgNPs-SERS 的多组分体系，建立严谨、可复现的半定量分析基线框架。

### 角色定义
- **MBA (4-Mercaptobenzoic acid)**: probe / internal standard / reference molecule。不是主要目标物。
- **Thiram (福美双)**: 主要目标分析物。
- **MG (孔雀石绿)**: 主要目标分析物。

### 主线任务
1. Component presence recognition (Thiram/MG 是否存在)
2. Concentration-level classification (0/4/5/6 ppm 离散等级)
3. Semi-quantitative analysis

### 当前不做
- 连续浓度定量回归 (RMSE/R² 已废弃)
- 将 MBA 作为主要待测污染物报告

## 2. 数据与实验背景
- 约 50 组 (~807 条光谱)，覆盖 single/binary/ternary 混合体系
- **同一 folder 内**: 同一天、同一台机器、同一条件的重复测量 → **technical replicates**，不是独立样本
- **不同 folder 间**: 通常不是同一天，不能默认同条件
- 因此: group-aware split 是强制要求，随机切分会造成泄漏
- 术语: 使用 folder-level dependence/clustering，不使用 "batch effect"

## 3. 当前正式任务
| Task | 目标 | 类别 | 说明 |
|------|------|------|------|
| Task1 | mixture_order | 1/2/3 | 混合复杂度 |
| Task2 | has_thiram | 0/1 | Thiram存在性 |
| Task3 | has_mg | 0/1 | MG存在性 |
| Task4 | c_thiram | 0/4/5/6 | Thiram浓度等级 |
| Task5 | c_mg | 0/4/5/6 | MG浓度等级 |
| Task6 | c_mba | 0/4/5/6 | MBA等级(内部监控) |

## 4. 当前模型
保留三个基线模型横向比较: RF, Ridge, PLS-DA。混淆矩阵图仅展示 RF。

## 5. 目录结构
```
├── README.md                          # 本文件(唯一入口说明)
├── requirements.txt                   # Python依赖
├── scripts/                           # 现役脚本
│   ├── phase1_inventory.py            # 阶段1: 数据清点
│   ├── parse_metadata.py              # 阶段2: 元数据解析
│   ├── preprocess_spectra.py          # 阶段3: 预处理
│   ├── phase4_5_eda_split_revised.py  # 阶段4+5(修订版)
│   ├── phase6_baseline_models_revised.py # 阶段6(修订版)
│   ├── phase7_mba_reference_strategy.py  # 阶段7: MBA探索
│   ├── run_realign.py                 # 历史重构脚本
│   └── full_rebuild.py                # 总重构脚本
├── data/
│   ├── metadata_v1.csv                # 主元数据
│   ├── processed/                     # 预处理后的 .npy 文件
│   ├── splits/cv_split_v2.csv         # 现役划分(SGKF)
│   └── models/baseline_results_final.csv # 现役模型结果
├── figures/
│   ├── eda_revised/                   # 现役EDA图(6张)
│   ├── models_revised/                # 现役混淆矩阵(6张)
│   └── preprocessing/                 # 预处理对比图
├── reports/                           # 所有现役报告
│   ├── full_project_audit.md          # 总审计报告
│   ├── file_cleanup_manifest.md       # 文件清理清单
│   ├── eda_report_revised.md
│   ├── dimensionality_reduction_revised.md
│   ├── split_report_revised.md
│   ├── baseline_report_revised.md
│   ├── modeling_scope_revised.md
│   ├── mba_reference_strategy.md
│   ├── current_research_state.md
│   └── change_log_revised.md
└── archive/                           # 归档区(不再现役)
    ├── deprecated_reports/            # 旧版报告
    ├── deprecated_scripts/            # 旧版脚本
    ├── legacy_figures/                # 旧版图表
    └── legacy_splits/                 # 旧版划分
```

## 6. 现役图表清单
### EDA (figures/eda_revised/)
| 文件 | 用途 |
|------|------|
| 1_single_mean_spectra.png | 单物质平均谱(MBA标注为Probe) |
| 2_global_pca_by_family.png | 全局PCA按family着色 |
| 3_pca_variance_bar.png | PCA方差解释度 |
| 4_folder_level_mean_pca.png | Folder均值PCA(消除伪重复) |
| 5_local_pca_by_family.png | 各family子集局部PCA |
| 6_umap_supplement.png | UMAP补充(不替代PCA) |

### Models (figures/models_revised/)
| 文件 | 用途 |
|------|------|
| cm_mixture_order.png | Task1混淆矩阵 |
| cm_has_thiram.png | Task2混淆矩阵 |
| cm_has_mg.png | Task3混淆矩阵 |
| cm_c_thiram.png | Task4混淆矩阵 |
| cm_c_mg.png | Task5混淆矩阵 |
| cm_c_mba.png | Task6混淆矩阵(内部监控) |

## 7. 已知限制
1. Thiram 6 ppm 混合物数据结构性缺失
2. Folder 内为技术重复，限制了有效独立样本量
3. Level classification (4/5/6 ppm) 表现偏弱
4. 当前不支持连续定量结论
5. Folder-level clustering 的物理来源未确认

## 8. 更新规则
每次模型、split、图表或任务变化，优先更新本 README。不通过生成额外说明文档替代 README。