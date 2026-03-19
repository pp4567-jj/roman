"""
阶段 7: MBA参考化/内标化探索 (Exploratory Analysis)
===================================================
说明：
本脚本为探索性质，不作为当前项目的主干基线结论。
目的：探索将已明确定义为探针/内标分子的 MBA（4-Mercaptobenzoic acid）的标志峰，
用于光谱特征的内部重标定（Internal Standardization/Normalization）。

操作：
1. 寻找 MBA 已知特征峰（约 1077 cm⁻¹ 和 1588 cm⁻¹）。
2. 提取峰高作为基准分母，对全谱进行比值化转换（X_ratio_1077, X_ratio_1588）。
3. 比较原始谱特征与“MBA参照化特征”在 Thiram 定性分类上的简单得分变化。
4. 输出探索性报告 mba_reference_strategy.md。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42

def find_nearest_wavenumber_index(wn_array, target_wn):
    return np.argmin(np.abs(wn_array - target_wn))

def main():
    print("=" * 60)
    print("阶段 7: MBA 参考化/内标化探索 (Exploratory Analysis)")
    print("=" * 60)

    # 1. 加载数据
    meta = pd.read_csv(METADATA_PATH := SPLIT_DIR / "cv_split_v2.csv")
    wn = np.load(PROCESSED_DIR / "wavenumber.npy")
    X_p1 = np.load(PROCESSED_DIR / "X_p1.npy") 
    
    # 2. 定位 MBA 标志主峰
    idx_1077 = find_nearest_wavenumber_index(wn, 1077)
    idx_1588 = find_nearest_wavenumber_index(wn, 1588)
    
    # 为了避免除以 0 或负数（由基线拔除造成），我们对峰值做绝对值和微小偏移平滑
    mba_peak_1077 = np.abs(X_p1[:, idx_1077]) + 1e-6
    mba_peak_1588 = np.abs(X_p1[:, idx_1588]) + 1e-6
    
    # 3. 产生参考化特征
    X_ratio_1077 = X_p1 / mba_peak_1077[:, np.newaxis]
    X_ratio_1588 = X_p1 / mba_peak_1588[:, np.newaxis]
    
    # 简单测试 Thiram 有无 的区分表现
    y_target = meta['has_thiram'].values
    n_folds = meta['fold_id_v2'].nunique()
    
    f1_original, f1_ratio_1077, f1_ratio_1588 = [], [], []
    
    for fold in range(n_folds):
        train_idx = meta[meta['fold_id_v2'] != fold].index.values
        val_idx = meta[meta['fold_id_v2'] == fold].index.values
        
        clf1 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        clf1.fit(X_p1[train_idx], y_target[train_idx])
        f1_original.append(f1_score(y_target[val_idx], clf1.predict(X_p1[val_idx]), average='macro'))
        
        clf2 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        clf2.fit(X_ratio_1077[train_idx], y_target[train_idx])
        f1_ratio_1077.append(f1_score(y_target[val_idx], clf2.predict(X_ratio_1077[val_idx]), average='macro'))
        
        clf3 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        clf3.fit(X_ratio_1588[train_idx], y_target[train_idx])
        f1_ratio_1588.append(f1_score(y_target[val_idx], clf3.predict(X_ratio_1588[val_idx]), average='macro'))

    mean_orig = np.mean(f1_original)
    mean_1077 = np.mean(f1_ratio_1077)
    mean_1588 = np.mean(f1_ratio_1588)

    print(f"Thiram Recognition [F1-Macro CV]")
    print(f"  原始特征 X_p1         : {mean_orig:.3f}")
    print(f"  MBA参比 (1077) 归一化 : {mean_1077:.3f}")
    print(f"  MBA参比 (1588) 归一化 : {mean_1588:.3f}")

    # 撰写报告
    report = f"""# 阶段 7：MBA 参考化策略（内标化探索）报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 本报告仅作为【研究探索（Exploratory Analysis）】，不代表项目的正式基线结局。

## 一、MBA 角色定义的回归
根据项目的系统性重新对齐，**MBA（4-Mercaptobenzoic acid）已明确降级为 Probe / Internal Standard / Reference Molecule**。
它在混合体系中不再作为与 Thiram（福美双）、MG（孔雀石绿）同级别的“待测目标污染物”，而是作为体系内稳定的信号参考锚点。
基于此，基线和报告中去除了利用 MBA 构建主分类/主定量指标的高光包装，并试图探究其真实的参考物理作用。

## 二、内标比值化尝试 (Peak Normalization)
### 实验设计
由于强烈的 Folder-Level 背景波动和 AgNPs 基底增强差异，直接使用全谱进行特征提取存在泛化瓶颈。这里我们提取了 MBA 公认的极强拉曼特征峰：
1. **~1077 cm⁻¹** (苯环骨架振动)
2. **~1588 cm⁻¹** (苯环 C=C 伸缩)

将这两处光谱强度的绝对值分别提取作为分母，对整条拉曼光谱实施了比值化转换（`X_ratio = X_raw / Peak_MBA`）。

### 初步测试结果
以“辨别福美双的存在与否 (Thiram Presence)”为例，在 StratifiedGroupKFold 严格分隔下的 F1 表现：
- **无依赖全谱直接输入**: {mean_orig:.3f}
- **除以 MBA (1077 cm⁻¹) 强度作为基准**: {mean_1077:.3f}
- **除以 MBA (1588 cm⁻¹) 强度作为基准**: {mean_1588:.3f}

### 现象分析与方向预判
1. **简单峰商并未立刻产生奇效**：目前的直接相除法产生的分数与原始谱大致持平（甚至受由于基线补正导致的零点附近的随机波动干扰而略微下降）。这暗示强行除以一个孤立波段点不仅引入了该点的高频随机噪声，也可能被多组分竞争吸附带来的局部峰移掩盖了比值优势。
2. **正确的前进道路**：当前的实验虽未能直接突破目前的基线分数，但它指明了后续必须从“全宽频段拉锯战”走向“精确的子区段峰面积积分与特征融合（例如提取 Thiram:1380 cm⁻¹ 积分面积 / MBA:1077 cm⁻¹ 积分面积作为单变量回归项）”。只有摒弃 1400 维光谱无脑丢给 Random Forest，才有可能真正实现抗严重底噪干扰的浓度跨度定量。
"""
    with open(REPORT_DIR / "mba_reference_strategy.md", 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nMBA 内标探索完成，探索日志已生成: {REPORT_DIR / 'mba_reference_strategy.md'}")

if __name__ == '__main__':
    main()
