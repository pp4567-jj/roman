# SERS 多组分分析：通过拉曼光谱预测混合物质及其浓度

## 项目目标
利用 SERS（表面增强拉曼光谱）技术，基于 AgNPs（Lee-Meisel 法制备银纳米颗粒）基底，
对 **Thiram（福美双）、Malachite Green（孔雀石绿）、MBA（4-巯基苯甲酸）** 三种物质的
单组分/二元/三元混合体系进行 **定性识别（是否存在）** 和 **浓度等级分类（0/4/5/6 ppm）**。

## 核心策略
**Strategy 3: MBA 作为普通组分**
- MBA、Thiram、MG 三种物质地位平等，均作为目标分析物
- 不使用 MBA 作为内标/探针进行归一化

## 数据概况
- **光谱来源**: BWRam 785nm 拉曼光谱仪，AgNPs 基底（Lee-Meisel 法，无聚集剂，干燥法）
- **混合体系**: 63 个文件夹，覆盖所有单组分/二元/三元浓度组合
- **浓度等级**: 4, 5, 6 ppm（编码自 0.4, 0.5, 0.6 实际浓度）
- **命名规则**: 美X=Thiram Xppm, KX=MG Xppm, mX=MBA Xppm

## 项目结构
```
├── run_pipeline.py          # 主入口脚本
├── src/
│   ├── config.py            # 全局配置（路径、超参数、任务定义）
│   ├── dataset.py           # 数据加载、解析、预处理、分割
│   ├── models.py            # 模型定义（RF, SVM, PLS-DA, 1D-CNN, 1D-ResNet）
│   ├── train_eval.py        # 交叉验证训练与评估
│   └── visualize.py         # EDA 与结果可视化
├── 混合数据/美Km混合光谱/   # 原始光谱数据（63 个文件夹）
├── data/
│   ├── processed/           # 预处理后的 .npy 缓存
│   ├── splits/              # CV 划分文件
│   └── models/              # 模型评估结果
├── figures/                 # 所有图表输出
├── reports/                 # 自动生成的报告
├── scripts/                 # 独立工具脚本（inventory, preprocessing）
└── archive/                 # 历史废弃文件
```

## 使用方法
```bash
# 完整流水线（数据准备 → EDA → 训练 → 报告）
python run_pipeline.py

# 仅数据准备
python run_pipeline.py --step data

# 仅训练（指定模型和预处理）
python run_pipeline.py --step train --models RF SVM 1D-CNN --preprocess p1

# 强制重建所有缓存
python run_pipeline.py --rebuild
```

## 分类任务
| 任务 | 目标 | 类别 |
|------|------|------|
| T1 | Thiram 浓度分类 | 0/4/5/6 ppm |
| T2 | MG 浓度分类 | 0/4/5/6 ppm |
| T3 | MBA 浓度分类 | 0/4/5/6 ppm |
| T4 | Thiram 存在性 | 0/1 |
| T5 | MG 存在性 | 0/1 |
| T6 | MBA 存在性 | 0/1 |
| T7 | 混合复杂度 | 1/2/3 |

## 模型
- **RF**: Random Forest (200 trees)
- **SVM**: RBF kernel SVM (C=10)
- **PLS-DA**: Partial Least Squares Discriminant Analysis (10 LV)
- **1D-CNN**: 4-block Conv1D + BN + ReLU + MaxPool → GAP → Dense
- **1D-ResNet**: Residual 1D-CNN with skip connections

## 交叉验证
- 5-fold StratifiedGroupKFold
- Group = folder_name（防止同一物理样本的技术重复进入不同折）
- Stratify = family（确保各混合类型在每折中均衡分布）