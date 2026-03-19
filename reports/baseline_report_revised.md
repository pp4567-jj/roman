# 阶段 6：基线分类建模评估 (修订版)
> 生成时间: 2026-03-19 13:28:39

## 一、任务定义
| Task | 目标 | 类型 | 类别 |
|------|------|------|------|
| Task1 | mixture_order | 体系复杂度 | 1, 2, 3 |
| Task2 | has_thiram | Thiram存在性 | 0, 1 |
| Task3 | has_mg | MG存在性 | 0, 1 |
| Task4 | c_thiram | Thiram浓度等级 | 0, 4, 5, 6 ppm |
| Task5 | c_mg | MG浓度等级 | 0, 4, 5, 6 ppm |
| Task6 | c_mba | MBA等级(内部监控) | 0, 4, 5, 6 ppm |

**说明**: Task2/3 是 presence 任务; Task4/5 是 level classification 任务。Task6 仅作内部监控，MBA 是 probe/internal standard，不是主要目标物。

## 二、模型说明
当前保留三个基线模型进行横向比较:
- **RF**: RandomForestClassifier(n=100, max_depth=10)
- **Ridge**: RidgeClassifier(alpha=10)
- **PLS-DA**: PLS-DA(5 LV)

混淆矩阵图仅展示 RF 的结果（位于 `figures/models_revised/cm_*.png`）。

## 三、Macro-F1 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA |
|------|------|------|----|-------|--------|
| Task1 | Mixture Order | order | 0.562 | 0.657 | 0.594 |
| Task2 | Thiram Presence | presence | 0.737 | 0.826 | 0.723 |
| Task3 | MG Presence | presence | 0.732 | 0.646 | 0.574 |
| Task4 | Thiram Level | level | 0.413 | 0.513 | 0.443 |
| Task5 | MG Level | level | 0.382 | 0.384 | 0.354 |
| Task6 | MBA Level (Internal) | internal | 0.623 | 0.534 | 0.517 |

## 四、Balanced Accuracy 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA |
|------|------|------|----|-------|--------|
| Task1 | Mixture Order | order | 0.592 | 0.685 | 0.607 |
| Task2 | Thiram Presence | presence | 0.796 | 0.856 | 0.739 |
| Task3 | MG Presence | presence | 0.778 | 0.664 | 0.619 |
| Task4 | Thiram Level | level | 0.590 | 0.680 | 0.570 |
| Task5 | MG Level | level | 0.449 | 0.404 | 0.417 |
| Task6 | MBA Level (Internal) | internal | 0.692 | 0.592 | 0.638 |

## 五、结论
### 数据直接支持
- Presence 任务（Task2/3）表现显著优于 Level 任务（Task4/5），说明当前全谱特征对"是否存在"的判别力远强于"具体浓度等级"。
- Task6 (MBA) 得分最高，因为 MBA 作为探针分子信号极强且分布广泛，这不代表检测难度高，仅作内部参考。

### 合理假设但证据不足
- Level 分类困难可能源于: (a) 4/5/6 ppm 浓度梯度过小; (b) 多组分竞争吸附导致特征退化; (c) Thiram 6ppm 混合物数据缺失。但无法仅从模型分数确定具体原因。

### 当前不能下结论
- 不能声称当前体系支持连续定量。
- 不能将 Level 分类的低分直接归因于模型能力不足——可能是数据和划分局限。
