# 阶段 6：基线分类建模评估 (V3 Split)
> 生成时间: 2026-03-20 16:50:09
> Split: 3-fold StratifiedGroupKFold (stratify=c_thiram, group=folder_name)

## 一、任务定义
| Task | 目标 | 类型 | 类别 |
|------|------|------|------|
| Task1 | mixture_order | 体系复杂度 | 1, 2, 3 |
| Task2 | has_thiram | Thiram存在性 | 0, 1 |
| Task3 | has_mg | MG存在性 | 0, 1 |
| Task4 | c_thiram | Thiram浓度等级 (受限结果) | 0, 4, 5, 6 ppm |
| Task5 | c_mg | MG浓度等级 | 0, 4, 5, 6 ppm |
| Task6 | c_mba | MBA等级 (内部监控) | 0, 4, 5, 6 ppm |

**说明**: Task2/3 = presence 主结果; Task4 = 受限结果(c_thiram 6ppm仅3 folder); Task5 = 辅助结果; Task6 = 内部监控(MBA=probe)。

## 二、模型说明
三个传统基线模型 + 一个深度学习模型横向比较:
- **RF**: RandomForestClassifier(n=100, max_depth=10)
- **Ridge**: RidgeClassifier(alpha=10)
- **PLS-DA**: PLS-DA(5 LV)
- **1D-CNN**: 4层Conv1D+BN+ReLU+MaxPool → GAP → Dropout(0.5) → Dense (PyTorch, Adam, EarlyStopping patience=20)

混淆矩阵图展示**每个任务得分最高的模型**。

## 三、Macro-F1 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA | 1D-CNN | Best |
|------|------|------|----|-------|--------|--------|------|
| Task1 | Mixture Order | order | 0.784 | 0.750 | 0.683 | 0.579 | RF |
| Task2 | Thiram Presence | presence | 0.879 | 0.893 | 0.810 | 0.798 | Ridge |
| Task3 | MG Presence | presence | 0.639 | 0.670 | 0.583 | 0.519 | Ridge |
| Task4 | Thiram Level | level | 0.589 | 0.570 | 0.478 | 0.496 | RF |
| Task5 | MG Level | level | 0.403 | 0.391 | 0.398 | 0.395 | RF |
| Task6 | MBA Level (Internal) | internal | 0.637 | 0.586 | 0.513 | 0.478 | RF |

## 四、Balanced Accuracy 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA | 1D-CNN |
|------|------|------|----|-------|--------|--------|
| Task1 | Mixture Order | order | 0.790 | 0.750 | 0.665 | 0.598 |
| Task2 | Thiram Presence | presence | 0.895 | 0.888 | 0.814 | 0.797 |
| Task3 | MG Presence | presence | 0.640 | 0.714 | 0.632 | 0.538 |
| Task4 | Thiram Level | level | 0.615 | 0.647 | 0.557 | 0.559 |
| Task5 | MG Level | level | 0.417 | 0.402 | 0.464 | 0.415 |
| Task6 | MBA Level (Internal) | internal | 0.651 | 0.593 | 0.536 | 0.507 |

## 五、结论
### 数据直接支持
- Presence 任务 (Task2/3) 表现显著优于 Level 任务 (Task4/5)，全谱特征对"是否存在"的判别力远强于"具体浓度等级"。
- 传统基线与 1D-CNN 深度学习模型在同一防泄漏 split 下横向比较，最佳模型因任务而异。
- 1D-CNN 利用卷积核自动提取局部峰形特征，但在小样本(~834条)下需关注过拟合风险(已通过 EarlyStopping+Dropout+GAP 缓解)。

### 合理假设但证据不足
- Level 分类困难可能源于: (a) 4→5→6 ppm 梯度过小; (b) 多组分竞争吸附; (c) c_thiram 6ppm 仅 3 个 folder。

### 当前不能下结论
- 不能声称体系支持连续定量。
- 不能将 c_thiram level 低分直接归因于模型能力不足——c_thiram 6ppm 的 folder 数量极少，评估底座本身受限。

### 受限结果声明
- **Task4 (c_thiram level)**: c_thiram=6 仅存在于 3 个 folder，即便 3-fold 仍可能有 fold 验证集挂零。此结果须谨慎解释，不可作为强主结论。
