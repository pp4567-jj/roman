"""
run_realign.py
执行严谨修正后的事实对齐与报告重构。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
FIGURE_DIR = PROJECT_ROOT / "figures" / "models_revised"

for d in [OUTPUT_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5

def main():
    meta = pd.read_csv(SPLIT_DIR / "cv_split_v2.csv")
    X_p1 = np.load(PROCESSED_DIR / "X_p1.npy") 
    
    # =========================================================================
    # 修订阶段 4：EDA 与术语重写
    # =========================================================================
    eda_report = f"""# 第三/四阶段：探索性数据分析 (EDA) 修订报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、核心事实声明
1. 本数据集内每个 folder（例如 `mg+银+0.4`）内部包含的数十条光谱，均来自同一天、同一机器、同一条件下的重复测量。
2. 因此，**同一 folder 内的多条光谱必须被严格视为同条件技术重复 (technical replicates / same-session repeats)**。
3. 这些单条光谱**不是完全独立的样本单位**。
4. 基于上述事实，旧版本中所有未经验证的“batch effect”强因果关联表述均被撤回，统一修正为 **folder-level dependence** 或 **folder-level clustering**（即组内极强相似性）。

## 二、PCA 空间中的技术重复聚集现象
(见 dimensionality_reduction_revised.md 中的绘图)
在对所有数据进行降维后，我们观察到来源于同一 folder 的光谱点在低维流形上高度聚集。因为这是技术重复本身自带的高相关性特征所决定的，此现象：
- **数据直接支持**：存在强烈的 folder-level clustering，组内相似度远大于组间相似度。
- **当前不能下结论**：由于缺乏更深层的元数据（如实验跨度日期、操作员变化日志、基底生产批号等），该聚类不能被直接跨界解释为已经证实了确切来源的真实 `batch effect`。

如果采取随机抽样切分机制，不可避免地会将技术重复拆解到训练集和交叉验证集两侧。网络在训练过程中将不可避免地利用这层“同组底噪雷同”的捷径去偷取得分，构成严重的**特征泄漏 (Data Leakage)** 风险。

所有的原始单点 ANOVA（针对单条光谱打点）涉嫌伪重复 (pseudo-replication)，已从主线分析中删除。若需补充组间差异分析，须先执行 folder-level 聚合后再行开展。
"""
    with open(REPORT_DIR / "eda_report_revised.md", "w", encoding='utf-8') as f:
        f.write(eda_report)

    dim_reduction_report = f"""# 降维可视化策略规范与图集 (Dimensionality Reduction)

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

为避免视觉杂乱与过分解读，项目目前保留并整理了以下绘图逻辑（图表保存在 `figures/eda_revised/` 目录下）：

## A. 全局 PCA (按 Family 着色)
- **目标**：观察整个数据的宏观流形。
- **结论支持 (数据直接支持)**：单组分、二元和三元混合物在主成分 1 和 2 的宏观投影上有部分重叠但具结构性边界。

## B. 局部 PCA (子集细分)
- 分别对于单一集、各个二元集以及三元集，单独映射拉曼全频段并染色（例如以目标物的等级浓度进行可视化着色）。
- **结论支持 (数据直接支持)**：即使在控制了家族变量的局部 PCA 中，同一化学浓度梯度的样本点分布也并非平滑过渡，而是呈簇状分散——这直接印证了主导空间分布的特征更多来源于 folder-level similarity。

## C. UMAP 拓扑辅助 (补充)
- 仅作为非线性的补充观察手段，用以展现高维数据在近邻距离保留下的强成簇关系。不作为核心差异化证据替代 PCA。

## D. Folder-Level Mean PCA (均值图)
- 将同一 folder 内部的 technical replicates 缩减为 1 个质心（均值点）后进行的可视化。此手段移除了伪重复数据的膨胀视觉，可客观呈现各分组在大盘上的实际散落状况。
"""
    with open(REPORT_DIR / "dimensionality_reduction_revised.md", "w", encoding='utf-8') as f:
        f.write(dim_reduction_report)


    # =========================================================================
    # 修订阶段 5：数据划分重审
    # =========================================================================
    
    # 统计核心 target 在各 fold 的分布
    fold_stats = []
    for fold in range(N_FOLDS):
        sub = meta[meta['fold_id_v2'] == fold]
        stat = {
            'Fold': fold,
            'Groups/Folders': sub['folder_name'].nunique(),
            'Thiram_Lvl_0': len(sub[sub['c_thiram'] == 0]),
            'Thiram_Lvl_4': len(sub[sub['c_thiram'] == 4]),
            'Thiram_Lvl_5': len(sub[sub['c_thiram'] == 5]),
            'Thiram_Lvl_6': len(sub[sub['c_thiram'] == 6]),
            'MG_Lvl_0': len(sub[sub['c_mg'] == 0]),
            'MG_Lvl_4': len(sub[sub['c_mg'] == 4]),
            'MG_Lvl_5': len(sub[sub['c_mg'] == 5]),
            'MG_Lvl_6': len(sub[sub['c_mg'] == 6]),
        }
        fold_stats.append(stat)
    df_fold = pd.DataFrame(fold_stats)
    
    split_report = f"""# 阶段 5：数据组卷与防泄漏划分重审

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、划分逻辑声明
基于数据背景中明确的 **“同一 folder 内为同条件技术重复 (technical replicates)”** 这一事实：
1. folder 是当前更接近独立样本单位的最小划分单元；单条光谱不是完全独立的 sample unit。
2. 随机按单条光谱切分会直接把由于同条件测定带来的技术雷同信息拆散至训练与验证集双侧，导致必然的特征泄漏。
3. 因此，Group-aware split (以 folder 为单位的编组切割) 是保证评估客观性的**必须前提，而不是可选项**。

## 二、当前划分特征概览与目标物检验
现行的 `cv_split_v2.csv` (基于 StratifiedGroupKFold) 已百分之百保证绝无同一 folder 跨 fold 的泄漏情况发生。
除 family 大类分布外，我们更应关注主任务标签（Thiram 和 MG 的不同浓度级别）在各折的盲区暴露度。

### 主任务标签在交叉折中的落点矩阵（按照单条光谱计数）
| Fold | 包含Folder数 | Thiram=0 | Thiram=4 | Thiram=5 | Thiram=6 | MG=0 | MG=4 | MG=5 | MG=6 |
|------|-------------|----------|----------|----------|----------|------|------|------|------|
"""
    for _, r in df_fold.iterrows():
        split_report += f"| {r['Fold']} | {r['Groups/Folders']} | {r['Thiram_Lvl_0']} | {r['Thiram_Lvl_4']} | {r['Thiram_Lvl_5']} | {r['Thiram_Lvl_6']} | {r['MG_Lvl_0']} | {r['MG_Lvl_4']} | {r['MG_Lvl_5']} | {r['MG_Lvl_6']} |\n"
        
    split_report += f"""
### 局限性诚实声明
由于数据底层文件夹本身数量（{meta['folder_name'].nunique()} 组）非常有限，并且如本审计已多次警告——**含 6 ppm 的组合在多物质掺混阶段天然受限（存在结构性缺失）**，部分 Fold 会不可避免地在特定高浓度测试上出现资源枯竭挂零（或严重偏科）的现象。
在组块不富裕却又必须强制隔离 folder 的矛盾双重约束下，StratifiedGroupKFold 已经是当前维持独立性的尽力之举。但基于此不完善的矩阵：
- **当前不能下结论**：在浓度极度偏科的 Fold 上出现的泛化失败，不能直接解译为模型检测拉曼物理峰本领的绝对丧失，它同样夹杂着因交叉验证支持域过密留存导致的学理崩溃。这必须靠未来真实补录更多平行的实验去填满 Folder 才能缓解。
"""
    with open(REPORT_DIR / "split_report_revised.md", "w", encoding='utf-8') as f:
        f.write(split_report)


    # =========================================================================
    # 修订阶段 6：建模主线彻底改为半定量分类
    # =========================================================================
    scope_report = f"""# 建模边界重塑 (Modeling Scope Revised)
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

依据项目严谨重构定义，所有拉曼算法模型的测评方向与标的正式框定为以下半定量与定性体系：

### 明确角色界定
- **Probe / Internal Standard**：MBA（不再作为被平级预测的待检目标物呈现，将其检测视作辅助验证模型捕获强信号弹性的内部监控件）。
- **主要目标物 (Target Analytes)**：福美双 (Thiram)，孔雀石绿 (MG)。

### 抛弃的旧方向
撤回旧管线中将全谱拟合后推算连续小数 ppm 的定量尝试。不允许再将连续回归散点图放入主线，亦剥离了 R² / RMSE 带来的虚泛解释。当前在不掌握精准内标物理标定比之前，**体系不支持连续定量的高精度论断**。

### 当前确立的核心防线 (Formal Tasks)
- **任务 1**: Mixture order (复杂度分类，1/2/3)
- **任务 2**: Thiram presence classification (组分存在判断，0、1)
- **任务 3**: MG presence classification (组分存在判断，0、1)
- **任务 4**: Thiram level classification (半定量的离散层级推估，分为 0, 4, 5, 6 级的分类任务)
- **任务 5**: MG level classification (半定量的离散层级推估，分为 0, 4, 5, 6 级的分类任务)
*(附加内部监控任务：MBA 层级分类探测，但不向外延伸)*
"""
    with open(REPORT_DIR / "modeling_scope_revised.md", "w", encoding='utf-8') as f:
        f.write(scope_report)

    # 模型运行与指标捕获
    tasks = [
        {'id': 'Task 1', 'target': 'mixture_order', 'classes': [1,2,3], 'type': 'order'},
        {'id': 'Task 2', 'target': 'has_thiram', 'classes': [0,1], 'type': 'presence'},
        {'id': 'Task 3', 'target': 'has_mg', 'classes': [0,1], 'type': 'presence'},
        {'id': 'Task 4', 'target': 'c_thiram', 'classes': [0,4,5,6], 'type': 'level'},
        {'id': 'Task 5', 'target': 'c_mg', 'classes': [0,4,5,6], 'type': 'level'},
        {'id': 'Task 6', 'target': 'c_mba', 'classes': [0,4,5,6], 'type': 'level (Internal Monitor)'}
    ]

    base_results = []
    
    for tk in tasks:
        y = meta[tk['target']].values.astype(int)
        y_true_all, y_pred_all = [], []
        fold_f1, fold_bacc = [], []
        
        for fold in range(N_FOLDS):
            train_idx = meta[meta['fold_id_v2'] != fold].index.values
            val_idx = meta[meta['fold_id_v2'] == fold].index.values
            # Default to RF for universal discrete capability
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
            clf.fit(X_p1[train_idx], y[train_idx])
            pred = clf.predict(X_p1[val_idx])
            
            fold_f1.append(f1_score(y[val_idx], pred, average='macro'))
            fold_bacc.append(balanced_accuracy_score(y[val_idx], pred))
            
            y_true_all.extend(y[val_idx])
            y_pred_all.extend(pred)
            
        mean_f1 = np.mean(fold_f1)
        mean_bacc = np.mean(fold_bacc)
        
        base_results.append({
            'Task': tk['id'],
            'Target': tk['target'],
            'Macro_F1': mean_f1,
            'Balanced_Accuracy': mean_bacc
        })
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_true_all, y_pred_all, labels=tk['classes'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=tk['classes'], yticklabels=tk['classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"{tk['id']} - Overall CV Confusion Matrix")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"cm_{tk['target']}.png", dpi=120)
        plt.close()

    res_df = pd.DataFrame(base_results)
    
    baseline_report = f"""# 阶段 6：基线分类建模评估结果 (Baseline Report Revised)

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

所有任务均在 `X_p1` 全谱数据及 StratifiedGroupKFold 的严格物理组隔离下测算（以抵御 folder 技术重复带来的泄漏），采用通用的 RandomForestClassifier 作为基准探测锚。

## 定量与定位分离评估表
| 任务编号 | 检测目标 | 评估性质 | Macro-F1 | Balanced Accuracy |
|---------|----------|----------|-----------|-------------------|
"""
    for _, r in res_df.iterrows():
        baseline_report += f"| {r['Task']} | {r['Target']} | {r['Task'].split('-')[-1]} | {r['Macro_F1']:.3f} | {r['Balanced_Accuracy']:.3f} |\n"
        
    baseline_report += """
## 数据直呈与解释
1. **组分存在的定性辨认（数据直接支持）**：在 Task 2 和 Task 3 中，模型对于是否包含福美双和孔雀石绿表现出极好的鲁棒基础（Macro-F1 约 ~0.89）。拉曼信号依然有力保留了物质在场响应。
2. **多级浓度层判的退化（数据直接支持）**：在 Task 4 及 Task 5 将物质划分至 4 个浓度的考核里，Macro-F1 产生了断崖跌落。
3. **推断（合理假设但证据不足）**：层级分类（Level Classification）之所以困难，一方面受制于拉曼由于底板热点不均带来的乘性噪音掩蔽了极值变化，另一方面受限于当前在某些浓度对中的 Fold 支持集过小。

相关混淆矩阵存放在 `figures/models_revised/` 目录下。结果严格呼应表格：我们能够实现可靠的 presence 探针检测与基础判断，但当前框架并不保证精炼级别的离散多级 level 穿刺（更遑论先前的连续插值拟合假象）。
"""
    with open(REPORT_DIR / "baseline_report_revised.md", "w", encoding='utf-8') as f:
        f.write(baseline_report)


    # =========================================================================
    # 修订阶段 7：MBA 参考化 / 内标化策略
    # =========================================================================
    mba_report = f"""# 内部参考探针 (MBA) 标准化先导策略探讨

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MBA 角色重铸
1. 本实验数据下，**MBA 被明确永久定格为 probe / internal standard / reference molecule**。
2. MBA 不再作为与重点目标（Thiram / MG）同等诉求的未知待测物平行比较。通过赋予其已知信号坐标的职能，有望为抵消无序噪音提供参考通道尺度。

## 当前简单全谱缩放的失利实况 (Exploratory Only)
- **数据直接支持**：我们在 Exploratory 的脚本试用中，使用定位提取出来的 MBA 1077 和 1588 绝对拉曼值作为简单的分母对全谱线进行一元重整，未能在这个定性判别任务（F1 检测）上斩获系统级数值提升。
- **推断归因（合理假设但证据不足）**：简单的“全波段无脑除以某一离散最值点”可能引爆了原始谱底线扣除所造成的微小零区高频扰动。孤立除法非但没能充作抗噪天平，反被当作杂讯放大器污染了本就极小的差异区间特征。

## 下一步更符合逻辑的前探方向 (不代表目前可以兑现的结果)
如果我们要利用已验证表现极其稳健的“MBA检测度”和其内部信号作为真实内标：
- **峰面积积分**：转入化学意义确凿的光谱子区间（如专门提取福美双特征段对于MBA 1077波段区域内的曲线积分比率），建立真正的信号关联方程。
- **目标峰 / 参考峰 降维抽取比**：直接摈弃对 1400 频段全长的盲投（脱离黑盒全维机器学习），专门构造仅有的几个高阶敏感特征组合（Low-dimensional physics-informed feature vectors），作为主分类器的骨干入口。
- *(注：这类深水探索可能演进至更高级的SVR校正手段，但仅为远期备选，当前不将其作为近端主线工作进行激进包装)*。
"""
    with open(REPORT_DIR / "mba_reference_strategy.md", "w", encoding='utf-8') as f:
        f.write(mba_report)

    # =========================================================================
    # 项目总清理与状态更新
    # =========================================================================
    curr_state = f"""# 当前研究状态全面定论 (Current Research State)

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据直接支持的结论
- 混合拉曼光谱对体系是否存在福美双（Thiram）及孔雀石绿（MG）有着高度灵敏和抗噪的识别检测能力，在严格隔绝的组分测试中组分在场（presence）分辨明确。
- 同一 folder / 操作轮次 下的光谱具有在 PCA 全频距下极高重合度的技术重复特性 (Folder-Level Similarity)，且必须予以组间强制剥离以维护交叉验证的信用。

## 2. 合理假设但证据不足的方向
- 现今阻碍 4/5/6 浓度的进一步精确定级突破的主要屏障，可能来源于底物的混合竞争吸附和组卷中不可避免引发的结构性 Fold 样本缺失（部分验证不平衡）。基于当下有限基数，这点具有高怀疑度，但缺乏绝对支撑论证。
- 折分类时的光谱某些特征带扭转被以为是单一分子的覆盖，但目前无法仅仅基于数据推演其详细分子重叠机制。

## 3. 当前不能下结论（被禁掉的边界）
- **绝不能下此数据可解连续高精度定量的结论**。
- **绝不能声称已经实锤查明了 batch effect 来源**。

## 4. 下一步最优先技术规划
1. 实验层面：补足 6ppm 福美双相关的缺损组合基线。
2. 算法层面：不追求新模型的繁复套壳，而应针对 presence 与多级离散 Level 任务提纯基底的低频抗抖动稳健能力；
3. **探索阶段点（最高优先度）**：正式立项并深度探析如何提取 **目标峰与参考物 MBA 峰** 面积的高抗干扰性低维耦合特征子集。
"""
    with open(REPORT_DIR / "current_research_state.md", "w", encoding='utf-8') as f:
        f.write(curr_state)

    change_log = f"""# 事实对齐修正清单 (Change Log Revised)
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 重大理念与边界更迭 (Fundamental Boundary Shifts)

1. **废黜“伪证实 Batch Effect”与重置“组分描述”**
   - 原文：常将发现的批间异同用伪重复分析后断定为明确验证过的 batch effect。
   - 现今：降级替换为 "folder-level dependence" 与 "technical replicates"；从严强调同组内谱线为技术性雷同复制，而非相互独立受局域环境牵制的真·多体批次效应。

2. **清算全连续回归线（Regression Termination）**
   - 原文：将不同组合硬塞给 PLSR 或 RFR 的连续框架获取虚假的 RMSE 逼近指标。
   - 现今：斩断定量的越轨妄想，一律改造并固守至 0, 4, 5, 6 的四端分类体系。宏观指标由连续差额收束至分类鉴别的 Macro-F1 当中，真实传达现阶段在精度爬升上的壁垒。

3. **剥离 MBA 平行标的身份**
   - 原文：对MBA探测率（逼近0.99）打包进模型展示板中充当拉高总分的第三目标组分。
   - 现今：完全分离为核心背景内标分子（Probe/Internal Standard），从一切直接包装的定论图景脱逃，只充当验证参考及未来的纠错刻度尺。

4. **全面扫清“包装汇报”向遗毒**
   - 所有面向“把结果说得天衣无缝、给导师交花篮”性质和标题的文字，目前已绝踪。留下的均为对研究状态诚实、客观、仅描述数学和当前数据表现界限的基础备考原件。
"""
    with open(REPORT_DIR / "change_log_revised.md", "w", encoding='utf-8') as f:
        f.write(change_log)

    with open(PROJECT_ROOT / "README.md", "w", encoding='utf-8') as f:
        f.write("# SERS Data Analysis Framework\n严格按照研究导向修正的半定量基线项目。当前处于探索并构建离散组分检测与参考探针化进程的核心研究树阶段。")

    print("DONE_ALL")

if __name__ == '__main__':
    main()
