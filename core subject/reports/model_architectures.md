# 模型架构与超参数手册

> 自动生成于 2026-04-16 | SERS 多组分分类项目

---

## 1. 全局训练超参数

| 参数 | 值 | 说明 |
|---|---|---|
| **交叉验证** | 5-fold StratifiedGroupKFold | group=文件夹, stratify=家族 |
| **DL 学习率** | 3×10⁻⁴ | Adam optimizer |
| **DL 最大轮次** | 200 epochs | 含早停 |
| **早停耐心** | 20 epochs | 基于验证集loss |
| **Batch Size** | 128 | |
| **权重衰减** | 1×10⁻⁴ | L2正则化 |
| **验证集比例** | 15% | GroupShuffleSplit划分 |
| **学习率调度** | CosineAnnealingLR | eta_min=1×10⁻⁶ |
| **Focal Loss γ** | 2.0 | 处理类别不平衡 |
| **类别权重** | 是 (inverse-frequency) | 逆频率加权 |
| **标签平滑** | 0.1 | Label Smoothing |
| **数据增强** | 是 (AUG_N=4) | 噪声注入+缩放 |
| **Mixup** | 是 (α=0.3) | 光谱混合增强 |
| **随机种子** | 42 | 可复现性 |

### KAN 专用参数

| 参数 | 值 |
|---|---|
| Grid Size | 5 |
| Spline Order | 3 (cubic B-spline) |

---

## 2. 数据概况

| 项目 | 值 |
|---|---|
| 总光谱数 | 954 |
| 文件夹组数 | 63 |
| 波数点数 | 1401 (400–1800 cm⁻¹) |
| 手工特征维度 | 78 |
| 预处理变体数 | 5 (raw, p1, p2, p3, p4) |
| 分类任务数 | 10 (7 主任务 + 3 半定量) |

### 预处理方案

| 标记 | 流程 |
|---|---|
| raw | 仅插值到等间距波数 |
| p1 | 宇宙射线去除 → SG平滑(w=11,p=3) → ALS基线校正(λ=10⁶) → SNV |
| p2 | SG平滑 → ALS基线校正 → 一阶导数 → SNV |
| p3 | ALS基线校正 → 向量归一化 |
| p4 | 宇宙射线去除 → SG平滑 → ALS基线校正 (无归一化) |

---

## 3. 模型架构详解

### 3.1 ML 全谱模型 (输入: 1401维)

#### RF (Random Forest)
```
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,         # 不限深度
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1
)
```
- **参数量**: 非参数模型, 取决于树的规模
- **训练轮次**: N/A (单次拟合)

#### SVM (Support Vector Machine)
```
SVC(
    kernel='rbf',
    C=10,
    gamma='scale',          # 1/(n_features × var(X))
    class_weight='balanced'
)
```
- **参数量**: 取决于支持向量数量
- **训练轮次**: N/A (单次拟合)

#### PLS-DA (Partial Least Squares Discriminant Analysis)
```
PLSRegression(n_components=10) + argmax 判别
```
- **参数量**: ~28,000 (10个隐变量)
- **训练轮次**: N/A (解析解)

---

### 3.2 ML 特征模型 (输入: 78维手工特征)

#### RF-feat / SVM-feat / PLS-DA-feat
- 与全谱版本**完全相同的超参数**
- 外层为 `_MLFeatureWrapper`: 先提取78维特征, 再送入对应ML分类器
- 特征包括: 峰强度、峰面积、峰比值、统计特征、光谱矩等

---

### 3.3 DL 全谱模型 (输入: 1×1401)

#### 1D-CNN

```
Input (1, 1401)
  ├── Conv1d(1→16, k=7, pad=3) → BN → ReLU → SE(16) → MaxPool(2)    # → (16, 700)
  ├── Conv1d(16→32, k=5, pad=2) → BN → ReLU → SE(32) → MaxPool(2)   # → (32, 350)
  ├── Conv1d(32→64, k=5, pad=2) → BN → ReLU → SE(64) → MaxPool(2)   # → (64, 175)
  ├── Conv1d(64→64, k=3, pad=1) → BN → ReLU → SE(64)                # → (64, 175)
  ├── GAP → (64,)
  ├── Dropout(0.3)
  └── Linear(64 → num_classes)
```
- **SE模块** (Squeeze-and-Excitation): `GAP → FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → channel-wise multiply`
- **参数量**: ~26K (不含分类头)
- **最大训练轮次**: 200 epochs (早停 patience=20)

#### 1D-ResNet

```
Input (1, 1401)
  ├── Stem: Conv1d(1→16, k=7, pad=3) → BN → ReLU → MaxPool(2)       # → (16, 700)
  ├── Stage1: ResBlock(16)×2 [每个: Conv→BN→ReLU→Conv→BN + SE + skip] # → (16, 700)
  ├── Down1: Conv1d(16→32, k=1, stride=2) → BN                       # → (32, 350)
  ├── Stage2: ResBlock(32)×2                                          # → (32, 350)
  ├── Down2: Conv1d(32→64, k=1, stride=2) → BN                       # → (64, 175)
  ├── Stage3: ResBlock(64)×2                                          # → (64, 175)
  ├── GAP → (64,)
  ├── Dropout(0.3)
  └── Linear(64 → num_classes)
```
- **ResBlock**: `x + SE(Conv→BN→ReLU→Conv→BN)(x)`, kernel_size=3
- **参数量**: ~95K (不含分类头)
- **最大训练轮次**: 200 epochs (早停 patience=20)

#### Spectrum-KAN (纯KAN全谱)

```
Input (1, 1401) → squeeze → (1401,)
  ├── BatchNorm1d(1401)
  ├── KANLinear(1401 → 128) → Dropout(0.3)
  ├── KANLinear(128 → 64) → Dropout(0.2)
  ├── KANLinear(64 → 32) → Dropout(0.2)
  └── KANLinear(32 → num_classes)
```
- **KANLinear**: `spline_out + SiLU_base_out`, B-spline基函数(grid=5, order=3)
  - 每层参数: `out × in × (grid+order)` (spline权重) + `out × in` (base权重)
  - 第1层: 128×1401×8 + 128×1401 ≈ **1.61M**
- **总参数量**: ~1.63M
- **最大训练轮次**: 200 epochs (早停 patience=20)

#### KAN-CNN (CNN骨干 + KAN分类头)

```
Input (1, 1401)
  ├── CNN Backbone (与1D-CNN相同的4层卷积+SE+池化)
  │     Conv1d(1→16) → Conv1d(16→32) → Conv1d(32→64) → Conv1d(64→64)
  ├── GAP → (64,)
  ├── Dropout(0.3)
  └── KAN Head:
        KANLinear(64 → 32) → Dropout(0.2) → KANLinear(32 → num_classes)
```
- **参数量**: CNN骨干~26K + KAN头~18K ≈ **44K**
- **最大训练轮次**: 200 epochs (早停 patience=20)

---

### 3.4 DL 特征模型 (输入: 78维)

#### 1D-CNN-feat / 1D-ResNet-feat / KAN-CNN-feat
- 外层为 `_DLFeatureWrapper`: 先提取78维特征, 再用 `_DLWrapper` 包裹对应网络
- 网络结构**与全谱版相同**, 但 input_dim=78 (远小于1401)
- 参数量相应减少 (特别是第一层卷积的感受野覆盖整个输入)

#### Feature-KAN (纯KAN特征)

```
Input (78,)
  ├── BatchNorm1d(78)
  ├── KANLinear(78 → 64)  → Dropout(0.2)
  ├── KANLinear(64 → 32)  → Dropout(0.2)
  └── KANLinear(32 → num_classes)
```
- **参数量**: ~50K
- **自定义增强**: 高斯噪声(σ=0.02×std) + 缩放(0.95~1.05)
- **最大训练轮次**: 200 epochs (早停 patience=20)

---

### 3.5 Ensemble (集成模型)

```
Ensemble = Soft Voting(RF-feat, Feature-KAN)
  ├── RF-feat: predict_proba → P₁
  ├── Feature-KAN: softmax(logits) → P₂
  └── Final: argmax(0.5 × P₁ + 0.5 × P₂)
```
- 等权重软投票
- 特征模型组合, 输入78维手工特征

---

## 4. 训练矩阵总览

共 **15 个单任务模型 × 5 预处理 × 10 任务 = 750 个实验单元**

| Phase | 类别 | 模型 | 输入维度 | 框架 | 训练方式 |
|---|---|---|---|---|---|
| 1 | ML 全谱 | RF | 1401 | sklearn | 单次拟合 |
| 1 | ML 全谱 | SVM | 1401 | sklearn | 单次拟合 |
| 1 | ML 全谱 | PLS-DA | 1401 | sklearn | 单次拟合 |
| 2 | DL 全谱 | 1D-CNN | 1×1401 | PyTorch | 200ep+早停 |
| 2 | DL 全谱 | 1D-ResNet | 1×1401 | PyTorch | 200ep+早停 |
| 2 | DL 全谱 | Spectrum-KAN | 1401 | PyTorch | 200ep+早停 |
| 2 | DL 全谱 | KAN-CNN | 1×1401 | PyTorch | 200ep+早停 |
| 3 | ML 特征 | RF-feat | 78 | sklearn | 单次拟合 |
| 3 | ML 特征 | SVM-feat | 78 | sklearn | 单次拟合 |
| 3 | ML 特征 | PLS-DA-feat | 78 | sklearn | 单次拟合 |
| 4 | DL 特征 | 1D-CNN-feat | 1×78 | PyTorch | 200ep+早停 |
| 4 | DL 特征 | 1D-ResNet-feat | 1×78 | PyTorch | 200ep+早停 |
| 4 | DL 特征 | Feature-KAN | 78 | PyTorch | 200ep+早停 |
| 4 | DL 特征 | KAN-CNN-feat | 1×78 | PyTorch | 200ep+早停 |
| 5 | 集成 | Ensemble | 78 | hybrid | RF+KAN |

---

## 5. 分类任务定义

| Task ID | 名称 | 目标列 | 类别数 | 类别值 |
|---|---|---|---|---|
| T1 | Thiram Concentration | c_thiram | 4 | 0/4/5/6 ppm |
| T2 | MG Concentration | c_mg | 4 | 0/4/5/6 ppm |
| T3 | MBA Concentration | c_mba | 4 | 0/4/5/6 ppm |
| T4 | Thiram Presence | has_thiram | 2 | 0/1 |
| T5 | MG Presence | has_mg | 2 | 0/1 |
| T6 | MBA Presence | has_mba | 2 | 0/1 |
| T7 | Mixture Complexity | mixture_order | 3 | 1/2/3 组分 |
| T1b | Thiram Semi-Quant | c_thiram_3c | 3 | 0(无)/1(低)/2(高) |
| T2b | MG Semi-Quant | c_mg_3c | 3 | 0(无)/1(低)/2(高) |
| T3b | MBA Semi-Quant | c_mba_3c | 3 | 0(无)/1(低)/2(高) |

---

## 6. 核心组件说明

### SE模块 (Squeeze-and-Excitation, 1D版)
```
x → GAP(x) → FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → x × weights
```
- reduction=4, 最小中间维度=4
- 用于所有CNN和ResNet的每个卷积块

### KANLinear (Kolmogorov-Arnold Network 线性层)
```
output = spline_out + base_out
  spline_out = einsum(B-spline_bases(tanh(x)), spline_weight)  # 非线性拟合
  base_out   = SiLU(x) @ base_weight.T                         # 线性残差
```
- B-spline基函数: grid_size=5, order=3 → n_bases=8
- 输入经tanh归一化到[-1,1]
- 自适应非线性: 每条输入-输出连接都有独立的可学习激活函数

### Focal Loss
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- γ=2.0, 配合 inverse-frequency class weights
- 训练集使用 Focal Loss, 验证集使用标准 CrossEntropy

### 数据增强
- **噪声注入**: 高斯噪声(σ=0.01×std), 重复AUG_N=4次 → 训练集扩大5倍
- **Mixup**: α=0.3, 生成额外 N_orig 个混合样本 → 训练集再扩大~1倍
- **总增强倍率**: 约6倍 (954 → ~5700 per fold)

---

## 7. 评估指标

| 指标 | 说明 |
|---|---|
| **Macro F1** | 主评估指标, 各类别F1的算术平均 |
| **Balanced Accuracy** | 各类别召回率的算术平均 |
| **Accuracy** | 整体准确率 |

所有指标均为5-fold CV的均值±标准差。
