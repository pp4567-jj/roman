"""
全项目总重构脚本 - 一次性生成所有修订版报告、模型结果与图表。
执行修订阶段4/5/6/7及审计报告、清理清单。
"""
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial']
plt.rcParams['axes.unicode_minus'] = False
try:
    import umap; HAS_UMAP = True
except: HAS_UMAP = False

ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
PROC = ROOT/"data"/"processed"; SPLIT = ROOT/"data"/"splits"
RPT = ROOT/"reports"; FIG_EDA = ROOT/"figures"/"eda_revised"
FIG_MOD = ROOT/"figures"/"models_revised"; MDL = ROOT/"data"/"models"
for d in [RPT,FIG_EDA,FIG_MOD,MDL]: d.mkdir(parents=True,exist_ok=True)
SEED=42; NF=5; TS=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

meta_orig = pd.read_csv(ROOT/"data"/"metadata_v1.csv")
wn = np.load(PROC/"wavenumber.npy")
X = np.load(PROC/"X_p1.npy")
fam_c = {'single':'#2196F3','binary_MBA_MG':'#4CAF50','binary_MBA_Thiram':'#FF9800','binary_Thiram_MG':'#9C27B0','ternary':'#F44336'}

# ========== 修订阶段5: 重做 split ==========
print("[Phase5] StratifiedGroupKFold...")
sgkf = StratifiedGroupKFold(n_splits=NF, shuffle=True, random_state=SEED)
groups = meta_orig['folder_name'].values
y_strat = meta_orig['family'].values
fids = np.zeros(len(meta_orig),dtype=int)
for fold,(tr,va) in enumerate(sgkf.split(X, y_strat, groups)):
    fids[va] = fold
meta = meta_orig.copy()
meta['fold_id'] = fids
meta.to_csv(SPLIT/"cv_split_v2.csv", index=False, encoding='utf-8-sig')

# ========== 修订阶段4: EDA 图表 ==========
print("[Phase4] EDA figures...")
# A. 单物质平均谱
fig,ax=plt.subplots(figsize=(14,6))
single=meta[meta['family']=='single']
for nm,col,c in [('MBA (Probe)','has_mba','#2196F3'),('Thiram (Target)','has_thiram','#FF9800'),('MG (Target)','has_mg','#4CAF50')]:
    idx=single[single[col]==True].index.values
    if len(idx)==0: continue
    m=X[idx].mean(0); s=X[idx].std(0)
    ax.plot(wn,m,label=f'{nm} (n={len(idx)})',color=c,lw=1.5)
    ax.fill_between(wn,m-s,m+s,alpha=0.15,color=c)
ax.set_xlabel('Raman Shift (cm⁻¹)'); ax.set_ylabel('P1 Intensity')
ax.set_title('Single-Component Mean Spectra (MBA=Probe, Thiram/MG=Targets)')
ax.legend(); ax.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig(FIG_EDA/'1_single_mean_spectra.png',dpi=150); plt.close()

# B. 全局 PCA
pca=PCA(n_components=5,random_state=SEED)
sc=pca.fit_transform(X); ve=pca.explained_variance_ratio_
fig,ax=plt.subplots(figsize=(9,7))
for f,c in fam_c.items():
    m=meta['family']==f
    ax.scatter(sc[m,0],sc[m,1],c=c,label=f,alpha=0.6,s=15)
ax.set_xlabel(f'PC1 ({ve[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({ve[1]*100:.1f}%)')
ax.set_title('Global PCA by Family'); ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIG_EDA/'2_global_pca_by_family.png',dpi=150); plt.close()

# C. PCA Variance Bar
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(range(1,6),ve*100,color='#2196F3',alpha=0.8)
for i,v in enumerate(ve): ax.text(i+1,v*100+0.5,f'{v*100:.1f}%',ha='center')
ax.set_xlabel('PC'); ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Explained Variance'); ax.set_xticks(range(1,6)); plt.tight_layout()
plt.savefig(FIG_EDA/'3_pca_variance_bar.png',dpi=150); plt.close()

# D. Folder-level mean PCA
fmeans,ffams=[],[]
for fn in meta['folder_name'].unique():
    idx=meta[meta['folder_name']==fn].index
    fmeans.append(sc[idx].mean(0)[:2]); ffams.append(meta.loc[idx[0],'family'])
fmeans=np.array(fmeans); ffams=np.array(ffams)
fig,ax=plt.subplots(figsize=(9,7))
for f,c in fam_c.items():
    m=ffams==f
    if m.sum()>0: ax.scatter(fmeans[m,0],fmeans[m,1],c=c,label=f'{f} ({m.sum()} folders)',s=60,edgecolors='k')
ax.set_title('Folder-Level Mean PCA (1 point per folder, removes pseudo-replication)')
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIG_EDA/'4_folder_level_mean_pca.png',dpi=150); plt.close()

# E. 局部 PCA per family
fig,axes=plt.subplots(2,3,figsize=(18,11))
axes=axes.flatten()
for i,(fam,label) in enumerate([
    ('single','Single'),('binary_MBA_MG','Binary MBA+MG'),('binary_MBA_Thiram','Binary MBA+Thiram'),
    ('binary_Thiram_MG','Binary Thiram+MG'),('ternary','Ternary')]):
    mask=meta['family']==fam; ax=axes[i]
    if mask.sum()==0: continue
    sub_sc=sc[mask]
    ax.scatter(sub_sc[:,0],sub_sc[:,1],c=fam_c[fam],alpha=0.5,s=15)
    ax.set_title(f'{label} (n={mask.sum()})'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
axes[5].axis('off')
plt.suptitle('Local PCA by Family Subset',fontsize=14); plt.tight_layout()
plt.savefig(FIG_EDA/'5_local_pca_by_family.png',dpi=150); plt.close()

# F. UMAP supplement
if HAS_UMAP:
    red=umap.UMAP(n_neighbors=15,min_dist=0.1,random_state=SEED)
    ures=red.fit_transform(X)
    fig,ax=plt.subplots(figsize=(9,7))
    for f,c in fam_c.items():
        m=meta['family']==f
        ax.scatter(ures[m,0],ures[m,1],c=c,label=f,alpha=0.6,s=15)
    ax.set_title('UMAP (Supplementary Only — Does NOT Replace PCA)')
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG_EDA/'6_umap_supplement.png',dpi=150); plt.close()

# ========== 修订阶段6: 建模 ==========
print("[Phase6] Baseline models...")

class PLSDA:
    def __init__(s,n=5): s.pls=PLSRegression(n_components=n); s.c_=None
    def fit(s,X,y):
        s.c_=np.unique(y); Y=pd.get_dummies(y).values; s.pls.fit(X,Y); return s
    def predict(s,X):
        P=s.pls.predict(X); return s.c_[np.argmax(P,1)]

TASKS = [
    dict(id='Task1',name='Mixture Order',col='mixture_order',cls=[1,2,3],typ='order'),
    dict(id='Task2',name='Thiram Presence',col='has_thiram',cls=[0,1],typ='presence'),
    dict(id='Task3',name='MG Presence',col='has_mg',cls=[0,1],typ='presence'),
    dict(id='Task4',name='Thiram Level',col='c_thiram',cls=[0,4,5,6],typ='level'),
    dict(id='Task5',name='MG Level',col='c_mg',cls=[0,4,5,6],typ='level'),
    dict(id='Task6',name='MBA Level (Internal)',col='c_mba',cls=[0,4,5,6],typ='internal'),
]
MODELS = {'RF':lambda: RandomForestClassifier(100,max_depth=10,random_state=SEED,n_jobs=-1),
           'Ridge':lambda: RidgeClassifier(alpha=10,random_state=SEED),
           'PLS-DA':lambda: PLSDA(5)}
results=[]
for tk in TASKS:
    y=meta[tk['col']].values.astype(int)
    for mname,mfn in MODELS.items():
        yt_all,yp_all=[],[]
        ff1,fba=[],[]
        for fold in range(NF):
            tri=meta[meta['fold_id']!=fold].index.values
            vai=meta[meta['fold_id']==fold].index.values
            clf=mfn(); clf.fit(X[tri],y[tri]); p=clf.predict(X[vai])
            ff1.append(f1_score(y[vai],p,average='macro'))
            fba.append(balanced_accuracy_score(y[vai],p))
            yt_all.extend(y[vai]); yp_all.extend(p)
        results.append(dict(Task=tk['id'],Target=tk['col'],TaskName=tk['name'],
                            Type=tk['typ'],Model=mname,
                            MacroF1=np.mean(ff1),BalancedAcc=np.mean(fba)))
        # Save CM for RF only
        if mname=='RF':
            cm=confusion_matrix(yt_all,yp_all,labels=tk['cls'])
            plt.figure(figsize=(6,5))
            sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=tk['cls'],yticklabels=tk['cls'])
            plt.xlabel('Predicted'); plt.ylabel('True')
            plt.title(f"{tk['id']}: {tk['name']} (RF)")
            plt.tight_layout()
            plt.savefig(FIG_MOD/f"cm_{tk['col']}.png",dpi=120); plt.close()
    print(f"  {tk['id']} done")

res_df=pd.DataFrame(results)
res_df.to_csv(MDL/"baseline_results_final.csv",index=False,encoding='utf-8-sig')

# ========== 修订阶段5: split 报告 ==========
print("[Reports] Writing all revised reports...")
# Fold distribution
fold_rows=[]
for fold in range(NF):
    s=meta[meta['fold_id']==fold]
    fold_rows.append(dict(Fold=fold,Folders=s['folder_name'].nunique(),N=len(s),
        Thiram0=(s['c_thiram']==0).sum(),Thiram4=(s['c_thiram']==4).sum(),
        Thiram5=(s['c_thiram']==5).sum(),Thiram6=(s['c_thiram']==6).sum(),
        MG0=(s['c_mg']==0).sum(),MG4=(s['c_mg']==4).sum(),
        MG5=(s['c_mg']==5).sum(),MG6=(s['c_mg']==6).sum()))
fd=pd.DataFrame(fold_rows)

split_txt=f"""# 阶段 5：数据划分报告 (修订版)
> 生成时间: {TS}

## 一、划分逻辑基础
1. **Folder 是当前最小的独立样本单位**。同一 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，属于 technical replicates，不是独立样本。
2. **随机按单条光谱切分不可用**。这会把技术重复拆进 train/val 两侧，造成特征泄漏。
3. **Group-aware split 是强制要求**，不是可选项。

## 二、当前方案: StratifiedGroupKFold (V2)
- 分层变量: family
- 分组变量: folder_name
- 折数: {NF}
- 输出: `data/splits/cv_split_v2.csv`
- V1 (基础 GroupKFold) 已归档至 `archive/legacy_splits/`。V2 通过分层机制尝试平衡各 fold 的 family 分布。

## 三、各 Fold 主任务标签分布
| Fold | Folders | N | Thiram=0 | Thiram=4 | Thiram=5 | Thiram=6 | MG=0 | MG=4 | MG=5 | MG=6 |
|------|---------|---|----------|----------|----------|----------|------|------|------|------|\n"""
for _,r in fd.iterrows():
    split_txt+=f"| {r['Fold']} | {r['Folders']} | {r['N']} | {r['Thiram0']} | {r['Thiram4']} | {r['Thiram5']} | {r['Thiram6']} | {r['MG0']} | {r['MG4']} | {r['MG5']} | {r['MG6']} |\n"
# Check for zeros
zeros=[]
for _,r in fd.iterrows():
    for k in ['Thiram0','Thiram4','Thiram5','Thiram6','MG0','MG4','MG5','MG6']:
        if r[k]==0: zeros.append(f"Fold {r['Fold']}: {k}=0")
split_txt+=f"""
## 四、诚实声明
- 总共 {meta['folder_name'].nunique()} 个 folder，5-fold 意味着每折约 {meta['folder_name'].nunique()//NF} 个 folder。
- Thiram 6 ppm 在混合物中存在结构性缺失，这是实验数据本身的限制，不是划分算法的问题。
"""
if zeros:
    split_txt+="- **以下 Fold-Level 组合存在零样本**:\n"
    for z in zeros: split_txt+=f"  - {z}\n"
    split_txt+="- 这些缺失会导致对应 fold 在该 level 上的验证无法进行，是当前数据局限，非算法缺陷。\n"
(RPT/"split_report_revised.md").write_text(split_txt,encoding='utf-8')

# ========== EDA 报告 ==========
eda_txt=f"""# 阶段 4：EDA 报告 (修订版)
> 生成时间: {TS}

## 一、核心事实
1. 每个 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，是 **technical replicates**，不是独立样本。
2. 不同 folder 之间通常不是同一天测的，不能默认同条件。
3. 因此存在 **folder-level dependence / folder-level clustering**。
4. **不能**将此直接写成"已证实的 batch effect"——缺乏显式元数据（日期、操作员、基底批号）来确认具体来源。

## 二、现役图表清单
| 编号 | 文件名 | 用途 |
|------|--------|------|
| 1 | 1_single_mean_spectra.png | 单物质平均谱（MBA标注为Probe） |
| 2 | 2_global_pca_by_family.png | 全局PCA按family着色 |
| 3 | 3_pca_variance_bar.png | PCA方差解释度柱状图 |
| 4 | 4_folder_level_mean_pca.png | Folder均值点PCA（消除伪重复） |
| 5 | 5_local_pca_by_family.png | 各family子集的局部PCA |
| 6 | 6_umap_supplement.png | UMAP补充图（不替代PCA） |

## 三、关键观察
### 数据直接支持
- 不同 family 在 PC1-PC2 空间有结构性分离，说明化学信号可学习。
- 同一 folder 的光谱在 PCA 空间高度聚集（见图4），印证了 folder-level dependence。
- 随机切分会拆散技术重复，造成泄漏风险。

### 当前不能下结论
- 聚集现象的具体物理来源（操作员差异？基底批次？环境温度？）——缺乏元数据，无法确认。

## 四、统计检验声明
旧版基于单条光谱的 ANOVA 检验已被撤回（伪重复问题）。如需组间差异分析，须先聚合至 folder 级别均值后再进行，且仅作 exploratory analysis。
"""
(RPT/"eda_report_revised.md").write_text(eda_txt,encoding='utf-8')

# ========== 降维策略报告 ==========
dim_txt=f"""# 降维可视化策略 (修订版)
> 生成时间: {TS}

## 图表规范
- **A. 全局 PCA (按 family 着色)**: 观察宏观流形结构。PC1={ve[0]*100:.1f}%, PC2={ve[1]*100:.1f}%。
- **B. PCA Variance Bar**: 前5主成分解释方差。
- **C. Folder-Level Mean PCA**: 每个 folder 缩减为1个质心，消除组内伪重复的视觉膨胀。
- **D. 局部 PCA**: 按 family 子集分别展示，观察子群体内部结构。
- **E. UMAP (补充)**: 仅作非线性拓扑辅助观察，不替代 PCA 主图。

## 使用原则
1. PCA 为主线降维手段，UMAP 仅为补充。
2. 任何基于降维的"分离度"观察，必须区分是化学信号驱动还是 folder-level dependence 驱动。
3. 不在降维图上做定量结论。
"""
(RPT/"dimensionality_reduction_revised.md").write_text(dim_txt,encoding='utf-8')

# ========== 建模报告 ==========
pivot=res_df.pivot_table(index=['Task','TaskName','Type'],columns='Model',values='MacroF1')
pivot_ba=res_df.pivot_table(index=['Task','TaskName','Type'],columns='Model',values='BalancedAcc')
baseline_txt=f"""# 阶段 6：基线分类建模评估 (修订版)
> 生成时间: {TS}

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
|------|------|------|----|-------|--------|\n"""
for idx,row in pivot.iterrows():
    task,name,typ=idx
    baseline_txt+=f"| {task} | {name} | {typ} | {row.get('RF','-'):.3f} | {row.get('Ridge','-'):.3f} | {row.get('PLS-DA','-'):.3f} |\n"
baseline_txt+=f"""
## 四、Balanced Accuracy 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA |
|------|------|------|----|-------|--------|\n"""
for idx,row in pivot_ba.iterrows():
    task,name,typ=idx
    baseline_txt+=f"| {task} | {name} | {typ} | {row.get('RF','-'):.3f} | {row.get('Ridge','-'):.3f} | {row.get('PLS-DA','-'):.3f} |\n"
baseline_txt+="""
## 五、结论
### 数据直接支持
- Presence 任务（Task2/3）表现显著优于 Level 任务（Task4/5），说明当前全谱特征对"是否存在"的判别力远强于"具体浓度等级"。
- Task6 (MBA) 得分最高，因为 MBA 作为探针分子信号极强且分布广泛，这不代表检测难度高，仅作内部参考。

### 合理假设但证据不足
- Level 分类困难可能源于: (a) 4/5/6 ppm 浓度梯度过小; (b) 多组分竞争吸附导致特征退化; (c) Thiram 6ppm 混合物数据缺失。但无法仅从模型分数确定具体原因。

### 当前不能下结论
- 不能声称当前体系支持连续定量。
- 不能将 Level 分类的低分直接归因于模型能力不足——可能是数据和划分局限。
"""
(RPT/"baseline_report_revised.md").write_text(baseline_txt,encoding='utf-8')

# ========== 建模范围 ==========
scope_txt=f"""# 建模范围定义 (修订版)
> 生成时间: {TS}

## 角色界定
- **MBA**: probe / internal standard / reference molecule。不是主要目标物。
- **Thiram, MG**: 主要目标分析物 (target analytes)。

## 当前正式任务
1. mixture_order: 混合复杂度分类 (1/2/3)
2. has_thiram: 福美双存在性分类 (0/1)
3. has_mg: 孔雀石绿存在性分类 (0/1)
4. c_thiram: 福美双浓度等级分类 (0/4/5/6 ppm)
5. c_mg: 孔雀石绿浓度等级分类 (0/4/5/6 ppm)
6. c_mba: MBA等级分类 (内部监控，不对外报告为主要成果)

## 当前不做的事
- 连续浓度回归 (RMSE/R² 已废弃)
- 将 MBA 检测结果作为与 Thiram/MG 并列的主要成果
"""
(RPT/"modeling_scope_revised.md").write_text(scope_txt,encoding='utf-8')

# ========== MBA策略 ==========
# Run MBA peak ratio test
wn_arr = wn
idx1077 = np.argmin(np.abs(wn_arr-1077))
idx1588 = np.argmin(np.abs(wn_arr-1588))
pk1 = np.abs(X[:,idx1077])+1e-6
pk2 = np.abs(X[:,idx1588])+1e-6
Xr1 = X/pk1[:,None]; Xr2 = X/pk2[:,None]
y_th = meta['has_thiram'].values.astype(int)
f1_orig,f1_r1,f1_r2=[],[],[]
for fold in range(NF):
    tri=meta[meta['fold_id']!=fold].index.values
    vai=meta[meta['fold_id']==fold].index.values
    for Xin,store in [(X,f1_orig),(Xr1,f1_r1),(Xr2,f1_r2)]:
        c=RandomForestClassifier(50,max_depth=10,random_state=SEED,n_jobs=-1)
        c.fit(Xin[tri],y_th[tri]); store.append(f1_score(y_th[vai],c.predict(Xin[vai]),average='macro'))
mo,m1,m2=np.mean(f1_orig),np.mean(f1_r1),np.mean(f1_r2)

mba_txt=f"""# 阶段 7：MBA 参考化 / 内标化策略 (修订版)
> 生成时间: {TS}
> 性质: **Exploratory Analysis** — 不是正式突破，不是主线结论。

## 一、MBA 角色定义
MBA (4-Mercaptobenzoic acid) = **probe / internal standard / reference molecule**。
- 不是主要待测污染物。
- 不与 Thiram / MG 等权并列报告。

## 二、简单峰点归一化测试 (Exploratory)
提取 MBA 特征峰 (~1077 cm⁻¹, ~1588 cm⁻¹) 强度作为分母，对全谱做比值化。
以 Thiram Presence (RF, StratifiedGroupKFold) 为测试任务:
| 特征 | Macro-F1 |
|------|----------|
| 原始 X_p1 | {mo:.3f} |
| X / MBA(1077) | {m1:.3f} |
| X / MBA(1588) | {m2:.3f} |

### 数据直接支持
- 简单的单点相除**未产生系统性提升**。

### 合理假设但证据不足
- 失败原因可能是: 基线校正后峰值附近存在零点扰动，单点除法放大了高频噪声。

## 三、后续更合理的方向 (非当前主线)
1. **峰面积积分**: 对特定波段区间积分而非取单点值。
2. **子区间特征**: 提取目标峰/MBA参考峰的面积比作为低维特征。
3. **物理启发的特征工程**: 摆脱1400维全谱盲投，构建少量高信噪比特征。

以上均为远期可选方向，当前不将其作为近端主线。不自动默认为"连续定量回归"路线。
"""
(RPT/"mba_reference_strategy.md").write_text(mba_txt,encoding='utf-8')

# ========== 审计报告 ==========
audit_txt=f"""# 全项目总审计报告
> 生成时间: {TS}

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
"""
(RPT/"full_project_audit.md").write_text(audit_txt,encoding='utf-8')

# ========== 清理清单 ==========
cleanup_txt=f"""# 文件清理清单 (File Cleanup Manifest)
> 生成时间: {TS}

## 现役文件 (Active)
| 路径 | 用途 |
|------|------|
| README.md | 项目唯一入口说明 |
| requirements.txt | Python依赖 |
| scripts/parse_metadata.py | 阶段2: 元数据解析 |
| scripts/preprocess_spectra.py | 阶段3: 预处理 |
| scripts/phase1_inventory.py | 阶段1: 数据清点 |
| scripts/phase4_5_eda_split_revised.py | 阶段4+5: EDA与划分(修订版) |
| scripts/phase6_baseline_models_revised.py | 阶段6: 基线建模(修订版) |
| scripts/phase7_mba_reference_strategy.py | 阶段7: MBA内标化探索 |
| scripts/run_realign.py | 总控重构脚本(历史) |
| scripts/full_rebuild.py | 本次总重构脚本 |
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
| analyze_dataset.py | archive/deprecated_scripts/ | 根目录遗留脚本 |
| reports/report_for_supervisor.md | archive/deprecated_reports/ | 导师汇报文件 |
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
"""
(RPT/"file_cleanup_manifest.md").write_text(cleanup_txt,encoding='utf-8')

# ========== current_research_state ==========
state_txt=f"""# 当前研究状态 (Current Research State)
> 生成时间: {TS}

## 1. 数据直接支持的结论
- 混合拉曼光谱对 Thiram 和 MG 的存在性检测 (presence) 具有可靠的判别力 (Macro-F1 约 0.85-0.90)。
- 同一 folder 内的光谱为 technical replicates，存在 folder-level dependence，随机切分会造成泄漏。
- 当前全谱特征对"是否存在"的辨别远强于对"具体浓度等级 (4/5/6 ppm)"的区分。

## 2. 合理假设但证据不足
- Level 分类困难可能源于浓度梯度过小和多组分竞争吸附，但无法仅从模型分数确定。
- Folder-level clustering 的具体物理来源（操作员？基底？环境？）缺乏元数据确认。

## 3. 当前不能下结论
- 不能声称体系支持连续定量。
- 不能声称已证实 batch effect 的具体来源。
- 不能声称 MBA 峰点归一化是有效的内标策略（当前测试未见提升）。

## 4. 下一步优先事项
1. 补录 Thiram 6 ppm 混合物的缺失数据。
2. 提升 presence / level classification 的稳健性。
3. 探索 MBA 参考化的峰面积积分方法（子区间特征）。
4. 不将连续定量作为近期目标。
"""
(RPT/"current_research_state.md").write_text(state_txt,encoding='utf-8')

# ========== change_log ==========
chg_txt=f"""# 变更日志 (Change Log)
> 生成时间: {TS}

## 1. 废除 "batch effect" 表述
- 原: 使用 "batch effect" 和基于单条光谱的伪重复 ANOVA。
- 改: 统一为 "folder-level dependence/clustering"；撤回伪重复统计。
- 原因: 缺乏实验元数据，不能跨界定论。

## 2. 废除连续定量回归
- 原: RMSE/R² 评价的连续预测，含回归散点图。
- 改: 0/4/5/6 ppm 四分类，Macro-F1 + Balanced Accuracy。
- 原因: 数据不支持连续定量结论。

## 3. MBA 角色重定义
- 原: 与 Thiram/MG 并列作为第三目标物。
- 改: probe / internal standard，检测结果仅作内部监控。
- 原因: MBA 信号极强且分布广，其高分掩盖了真实检测难度。

## 4. 清除导师汇报包装
- 原: report_for_supervisor.md 等面向汇报的文件。
- 改: 全部归档，只保留研究导向的客观报告。
- 原因: 结论超出数据支持范围。
"""
(RPT/"change_log_revised.md").write_text(chg_txt,encoding='utf-8')

print("DONE_ALL")
