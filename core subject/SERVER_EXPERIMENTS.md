# 服务器实验执行计划

> 最后更新: 2026-04-16 | 服务器: RTX 3060 (i-1.gpushare.com:40148)

## 当前状态
- ✅ Phase 1 (ML全谱) — 已完成
- 🔄 Phase 2 (DL全谱) — 进行中 (1D-CNN raw)
- ⏳ Phase 3 (ML特征) — 等待
- ⏳ Phase 4 (DL特征) — 等待
- ⏳ Phase 5 (Ensemble) — 等待

---

## 完整模型矩阵（15个单任务模型）

|  | 全谱 (1401-dim) | 手工特征 (78-dim) |
|---|---|---|
| RF | RF | RF-feat |
| SVM | SVM | SVM-feat |
| PLS-DA | PLS-DA | PLS-DA-feat |
| 1D-CNN | 1D-CNN | 1D-CNN-feat |
| 1D-ResNet | 1D-ResNet | 1D-ResNet-feat |
| KAN | Spectrum-KAN | Feature-KAN |
| KAN-CNN | KAN-CNN | KAN-CNN-feat |
| + Ensemble (RF-feat + Feature-KAN) |

每个模型 × 10 tasks × 5 preprocessing variants (raw, p1, p2, p3, p4)
= 15 × 10 × 5 = **750 个实验单元**

---

## Part A: 模型性能数据

### A1. 完整基线训练（15模型 × 10任务 × 5预处理） ✅ 进行中
```bash
cd '/hy-tmp/core subject'
nohup python3 -u batch_train_paper.py > logs/run_20260416_v2.log 2>&1 &
```

**5个Phase**:
- [x] Phase 1: ML全谱 (RF, SVM, PLS-DA)
- [ ] Phase 2: DL全谱 (1D-CNN, 1D-ResNet, Spectrum-KAN, KAN-CNN)
- [ ] Phase 3: ML特征 (RF-feat, SVM-feat, PLS-DA-feat)
- [ ] Phase 4: DL特征 (1D-CNN-feat, 1D-ResNet-feat, Feature-KAN, KAN-CNN-feat)
- [ ] Phase 5: Ensemble

**预估时间 (GPU)**:
- Phase 1 (ML全谱): ~10分钟
- Phase 2 (DL全谱): 4模型 × 10tasks × 5variants × ~2min ≈ 7小时
- Phase 3 (ML特征): ~10分钟
- Phase 4 (DL特征): 4模型 × 10tasks × 5variants × ~1min ≈ 3.5小时
- Phase 5 (Ensemble): ~30分钟
- **总计 ≈ 11小时**

### A2. GroupKFold vs RandomSplit 泄漏对比
- 对 p2 预处理，用 RandomSplit 重跑 RF/SVM/PLS-DA/1D-CNN 在 T4-T7
- 量化数据泄漏影响（历史数据显示 ~+0.28 F1 膨胀）
- **状态**: ⏳ 等待 A1 完成后执行

---

## Part B: 消融实验

### B1. 特征消融（Feature Ablation）
- 在最佳预处理上，对 RF-feat / Feature-KAN，依次去掉每组特征
- 4组: Peak intensities(24d), Peak ratios(10d), Regional stats(20d), 2nd-deriv(24d)
- **状态**: ⏳ 需新建 `run_ablation_features.py`

### B2. 架构消融（KAN vs MLP）
- 新建 Feature-MLP（等参数量 MLP 替换 KAN 层）
- 对比 Feature-KAN vs Feature-MLP
- **状态**: ⏳ 需新建 `run_ablation_architecture.py`

### B3. 预处理消融
- 从 A1 的 CSV 中聚合，不需额外实验
- **状态**: ⏳ A1 完成后从数据中提取

---

## Part C: 可解释性实验

### C1. SHAP 分析
- TreeExplainer (RF) → 波数重要性 → 化学峰位对应
- 特征模型 → 78维特征重要性排名
- **状态**: ⏳ 需新建 `run_shap.py`

### C2. 统计显著性检验
- Wilcoxon signed-rank test
- 对比对: RF vs RF-feat, RF-feat vs Feature-KAN, GroupKFold vs RandomSplit
- **状态**: ⏳ 需新建 `run_stats_test.py`

---

## Part D: 稳定性分析

### D1. RSD 分析
- 从 5-fold 结果计算每个模型的 F1 RSD(%)
- **状态**: ⏳ A1 完成后从数据计算，无需额外实验

---

## Part E: 化学实验（实验室，不涉及服务器）

- [ ] E1. AgNPs 表征 (UV-Vis + SEM/TEM)
- [ ] E2. 真实样品验证 (Spiking Experiment)
- [ ] E3. 重复性验证 (同一样品多点位 RSD)

---

## 执行优先级

```
优先级 1: A1 → 完整 15 模型基线数据 (当前正在执行)
优先级 2 (A1完成后并行): B1 + B2 + C1 + C2 + D1
优先级 3 (实验室并行): E1 + E2 + E3
优先级 4 (补充): A2 泄漏对比
```

---

## 需要新建的代码文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `batch_train_paper.py` | ✅ 已完成 | 15模型 × 10tasks × 5variants |
| `models.py` | ✅ 已完成 | 23模型注册(含MT) |
| `run_ablation_features.py` | ⏳ 待建 | 特征消融 |
| `run_ablation_architecture.py` | ⏳ 待建 | KAN vs MLP 消融 |
| `run_shap.py` | ⏳ 待建 | SHAP 可解释性 |
| `run_stats_test.py` | ⏳ 待建 | Wilcoxon 检验 |
| `run_leakage_test.py` | ⏳ 待建 | 泄漏对比 |

---

## 数据文件说明

本地 `data/models/` 中的 CSV 文件:
- `cv_results_summary_*.csv` — 每个模型×任务的 5-fold 平均指标
- `cv_results_detail_*.csv` — 每个 fold 的逐条指标
- `cv_predictions_*.csv` — 每个样本的真实值和预测值

这些文件的 merge 逻辑（`save_results()`）基于 (Task, Model, Preprocess) 键合并，新数据覆盖旧数据，不会产生重复行。
