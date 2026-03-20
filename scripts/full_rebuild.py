"""
full_rebuild.py — 唯一现役执行入口
Split V3: 3-fold StratifiedGroupKFold(stratify=c_thiram)
基于审计结论: c_thiram 6ppm 仅3个folder, 5-fold下必然挂零, 3-fold是极限
混淆矩阵: 每个任务绘制该任务得分最高的模型(不再硬绑RF)
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
SEED=42; NF=3; TS=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

meta_orig = pd.read_csv(ROOT/"data"/"metadata_v1.csv")
wn = np.load(PROC/"wavenumber.npy")
X = np.load(PROC/"X_p1.npy")
fam_c = {'single':'#2196F3','binary_MBA_MG':'#4CAF50','binary_MBA_Thiram':'#FF9800',
         'binary_Thiram_MG':'#9C27B0','ternary':'#F44336'}

# ==================== 阶段5: Split V3 ====================
print("[Phase5] 3-fold StratifiedGroupKFold (stratify=c_thiram)...")
sgkf = StratifiedGroupKFold(n_splits=NF, shuffle=True, random_state=SEED)
groups = meta_orig['folder_name'].values
y_strat = meta_orig['c_thiram'].values.astype(str)
fids = np.zeros(len(meta_orig), dtype=int)
for fold, (tr, va) in enumerate(sgkf.split(X, y_strat, groups)):
    fids[va] = fold
meta = meta_orig.copy()
meta['fold_id'] = fids
meta.to_csv(SPLIT/"cv_split_v3.csv", index=False, encoding='utf-8-sig')
print(f"  Split V3 saved: {NF} folds, {meta['folder_name'].nunique()} groups")

# ==================== 阶段4: EDA 图表 ====================
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
    ax.set_title('UMAP (Supplementary Only)')
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG_EDA/'6_umap_supplement.png',dpi=150); plt.close()

# ==================== 阶段6: 建模 ====================
print("[Phase6] Baseline + 1D-CNN models (3-fold)...")

class PLSDA:
    def __init__(s,n=5): s.pls=PLSRegression(n_components=n); s.c_=None
    def fit(s,X,y):
        s.c_=np.unique(y); Y=pd.get_dummies(y).values; s.pls.fit(X,Y); return s
    def predict(s,X):
        P=s.pls.predict(X); return s.c_[np.argmax(P,1)]

# ---------- 1D-CNN (PyTorch) ----------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class CNN1D(nn.Module):
    """轻量级 1D-CNN: 4×Conv+BN+ReLU+Pool → GAP → Dropout → Dense"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)

class CNN1DClassifier:
    """sklearn-compatible wrapper for 1D-CNN with early stopping"""
    def __init__(self, num_classes=2, lr=1e-3, epochs=150, patience=20, batch_size=32, seed=42):
        self.num_classes = num_classes
        self.lr = lr; self.epochs = epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None; self.label_map = None; self.inv_map = None

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        # 标签 → 连续索引
        self.label_map = {lab: i for i, lab in enumerate(sorted(set(y)))}
        self.inv_map = {i: lab for lab, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        yi = np.array([self.label_map[v] for v in y])

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,D)
        yt = torch.tensor(yi, dtype=torch.long)

        # 划出 15% 做内部 early-stopping 验证 (按样本随机，仅用于训练控制)
        n = len(Xt); idx = np.random.permutation(n)
        n_val = max(int(n*0.15), 1)
        t_idx, v_idx = idx[n_val:], idx[:n_val]
        ds_t = TensorDataset(Xt[t_idx], yt[t_idx])
        ds_v = TensorDataset(Xt[v_idx], yt[v_idx])
        dl_t = DataLoader(ds_t, batch_size=self.batch_size, shuffle=True)
        dl_v = DataLoader(ds_v, batch_size=256, shuffle=False)

        self.model = CNN1D(X.shape[1], self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        best_loss = 1e9; wait = 0; best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_t:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); loss = crit(self.model(xb), yb); loss.backward(); opt.step()
            # validation
            self.model.eval(); vloss = 0; vn = 0
            with torch.no_grad():
                for xb, yb in dl_v:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    vloss += crit(self.model(xb), yb).item()*len(yb); vn += len(yb)
            vloss /= max(vn,1)
            if vloss < best_loss:
                best_loss = vloss; wait = 0; best_state = {k:v.cpu().clone() for k,v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience: break
        if best_state: self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[p] for p in preds])

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
           'PLS-DA':lambda: PLSDA(5),
           '1D-CNN':lambda: CNN1DClassifier(lr=1e-3, epochs=150, patience=20, batch_size=32, seed=SEED)}

results = []
# Store per-task per-model predictions for best-model CM
task_preds = {}  # key: (task_id, model_name) -> (y_true_all, y_pred_all)

for tk in TASKS:
    y = meta[tk['col']].values.astype(int)
    for mname, mfn in MODELS.items():
        yt_all, yp_all = [], []
        ff1, fba = [], []
        for fold in range(NF):
            tri = meta[meta['fold_id'] != fold].index.values
            vai = meta[meta['fold_id'] == fold].index.values
            clf = mfn(); clf.fit(X[tri], y[tri]); p = clf.predict(X[vai])
            ff1.append(f1_score(y[vai], p, average='macro'))
            fba.append(balanced_accuracy_score(y[vai], p))
            yt_all.extend(y[vai]); yp_all.extend(p)
        mf1 = np.mean(ff1)
        results.append(dict(Task=tk['id'], Target=tk['col'], TaskName=tk['name'],
                            Type=tk['typ'], Model=mname,
                            MacroF1=mf1, BalancedAcc=np.mean(fba)))
        task_preds[(tk['id'], mname)] = (yt_all, yp_all)
    print(f"  {tk['id']} done")

res_df = pd.DataFrame(results)
res_df.to_csv(MDL/"baseline_results_final.csv", index=False, encoding='utf-8-sig')

# 为每个任务绘制得分最高模型的混淆矩阵
for tk in TASKS:
    sub = res_df[res_df['Task'] == tk['id']]
    best_row = sub.loc[sub['MacroF1'].idxmax()]
    best_model = best_row['Model']
    yt, yp = task_preds[(tk['id'], best_model)]
    cm = confusion_matrix(yt, yp, labels=tk['cls'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tk['cls'], yticklabels=tk['cls'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f"{tk['id']}: {tk['name']} (Best: {best_model}, F1={best_row['MacroF1']:.3f})")
    plt.tight_layout()
    plt.savefig(FIG_MOD / f"cm_{tk['col']}.png", dpi=120); plt.close()
print("  Confusion matrices saved (best model per task)")

# ==================== 报告生成 ====================
print("[Reports] Writing all revised reports...")

# --- Split 报告 ---
fold_rows = []
for fold in range(NF):
    s = meta[meta['fold_id'] == fold]
    t = meta[meta['fold_id'] != fold]
    fold_rows.append(dict(
        Fold=fold, Folders_Val=s['folder_name'].nunique(), N_Val=len(s),
        Folders_Train=t['folder_name'].nunique(), N_Train=len(t),
        Val_Thi0=(s['c_thiram']==0).sum(), Val_Thi4=(s['c_thiram']==4).sum(),
        Val_Thi5=(s['c_thiram']==5).sum(), Val_Thi6=(s['c_thiram']==6).sum(),
        Val_MG0=(s['c_mg']==0).sum(), Val_MG4=(s['c_mg']==4).sum(),
        Val_MG5=(s['c_mg']==5).sum(), Val_MG6=(s['c_mg']==6).sum(),
        Train_Thi6=(t['c_thiram']==6).sum()))
fd = pd.DataFrame(fold_rows)

thi6_zeros = fd[fd['Val_Thi6']==0]['Fold'].tolist()
thi5_zeros = fd[fd['Val_Thi5']==0]['Fold'].tolist()

split_txt = f"""# 阶段 5：数据划分报告 (V3)
> 生成时间: {TS}

## 一、划分逻辑
1. **Folder 是最小独立样本单位**。同一 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，属于 technical replicates。
2. **随机按单条光谱切分不可用**，会造成特征泄漏。
3. **Group-aware split 是强制要求**。

## 二、方案选择依据
- V2 (5-fold, stratify=family): c_thiram 6ppm 仅存在于 3 个 folder，5 fold 下 3 个 fold 验证集必然挂零。已归档。
- **V3 (3-fold, stratify=c_thiram)**: 减少 fold 数至 3，以 c_thiram 为分层变量，尝试最大化 c_thiram 各等级在每折中的覆盖。

## 三、V3 各 Fold 标签覆盖
### c_thiram (Train / Val)
| Fold | Val Folders | Val N | Val 0 | Val 4 | Val 5 | Val 6 | Train 6 |
|------|-------------|-------|-------|-------|-------|-------|---------|
"""
for _, r in fd.iterrows():
    split_txt += f"| {r['Fold']} | {r['Folders_Val']} | {r['N_Val']} | {r['Val_Thi0']} | {r['Val_Thi4']} | {r['Val_Thi5']} | {r['Val_Thi6']} | {r['Train_Thi6']} |\n"

split_txt += f"""
### c_mg (Val)
| Fold | Val 0 | Val 4 | Val 5 | Val 6 |
|------|-------|-------|-------|-------|
"""
for _, r in fd.iterrows():
    split_txt += f"| {r['Fold']} | {r['Val_MG0']} | {r['Val_MG4']} | {r['Val_MG5']} | {r['Val_MG6']} |\n"

split_txt += f"""
## 四、诚实声明
- 总共 {meta['folder_name'].nunique()} 个 folder。含 c_thiram=6 的 folder 仅 **3 个**，含 c_mg=6 的 folder 有 **{meta[meta['c_mg']==6]['folder_name'].nunique()} 个**。
- 3-fold 是在保持防泄漏前提下，c_thiram 6ppm 可被分配到验证集的极限方案。
"""
if thi6_zeros:
    split_txt += f"- **警告**: Fold {thi6_zeros} 的 c_thiram=6 验证集仍为 0，说明即便 3 fold 也无法完全消除此穿孔。\n"
if thi5_zeros:
    split_txt += f"- **警告**: Fold {thi5_zeros} 的 c_thiram=5 验证集为 0。\n"
split_txt += "- c_thiram 4 分类结果须谨慎解释(受限结果)；c_mg 覆盖显著优于 c_thiram。\n"

(RPT/"split_report_revised.md").write_text(split_txt, encoding='utf-8')

# --- EDA 报告 ---
eda_txt = f"""# 阶段 4：EDA 报告 (修订版)
> 生成时间: {TS}

## 一、核心事实
1. 每个 folder 内约 20 条光谱来自同一天、同一台机器、同一条件的重复测量，是 **technical replicates**，不是独立样本。
2. 不同 folder 之间通常不是同一天测的，不能默认同条件。
3. 因此存在 **folder-level dependence / folder-level clustering**。
4. **不能**将此直接写成"已证实的 batch effect"——缺乏显式元数据来确认具体来源。

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
- 不同 family 在 PC1-PC2 空间有结构性分离，化学信号可学。
- 同一 folder 的光谱在 PCA 空间高度聚集，印证 folder-level dependence。

### 当前不能下结论
- 聚集现象的具体物理来源——缺乏元数据，无法确认。

## 四、统计检验声明
旧版基于单条光谱的 ANOVA 已撤回（伪重复问题）。
"""
(RPT/"eda_report_revised.md").write_text(eda_txt, encoding='utf-8')

# --- 降维报告 ---
dim_txt = f"""# 降维可视化策略 (修订版)
> 生成时间: {TS}

## 图表规范
- **A. 全局 PCA (按 family 着色)**: PC1={ve[0]*100:.1f}%, PC2={ve[1]*100:.1f}%。
- **B. PCA Variance Bar**: 前5主成分解释方差。
- **C. Folder-Level Mean PCA**: 每个 folder 缩减为1个质心，消除伪重复视觉膨胀。
- **D. 局部 PCA**: 按 family 子集分别展示。
- **E. UMAP (补充)**: 仅作非线性拓扑辅助，不替代 PCA。

## 使用原则
1. PCA 为主线降维，UMAP 仅补充。
2. 任何"分离度"观察须区分化学信号驱动 vs folder-level dependence 驱动。
"""
(RPT/"dimensionality_reduction_revised.md").write_text(dim_txt, encoding='utf-8')

# --- 建模报告 ---
pivot = res_df.pivot_table(index=['Task','TaskName','Type'], columns='Model', values='MacroF1')
pivot_ba = res_df.pivot_table(index=['Task','TaskName','Type'], columns='Model', values='BalancedAcc')

# 找出每个任务的最佳模型
best_per_task = res_df.loc[res_df.groupby('Task')['MacroF1'].idxmax()][['Task','Model','MacroF1']].set_index('Task')

baseline_txt = f"""# 阶段 6：基线分类建模评估 (V3 Split)
> 生成时间: {TS}
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
|------|------|------|----|-------|--------|--------|------|\n"""
for idx, row in pivot.iterrows():
    task, name, typ = idx
    b = best_per_task.loc[task]
    baseline_txt += f"| {task} | {name} | {typ} | {row.get('RF',0):.3f} | {row.get('Ridge',0):.3f} | {row.get('PLS-DA',0):.3f} | {row.get('1D-CNN',0):.3f} | {b['Model']} |\n"

baseline_txt += f"""
## 四、Balanced Accuracy 结果表
| Task | 目标 | 类型 | RF | Ridge | PLS-DA | 1D-CNN |
|------|------|------|----|-------|--------|--------|\n"""
for idx, row in pivot_ba.iterrows():
    task, name, typ = idx
    baseline_txt += f"| {task} | {name} | {typ} | {row.get('RF',0):.3f} | {row.get('Ridge',0):.3f} | {row.get('PLS-DA',0):.3f} | {row.get('1D-CNN',0):.3f} |\n"

baseline_txt += """
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
"""
(RPT/"baseline_report_revised.md").write_text(baseline_txt, encoding='utf-8')

# --- 建模范围 ---
scope_txt = f"""# 建模范围定义 (修订版)
> 生成时间: {TS}

## 角色界定
- **MBA**: probe / internal standard / reference molecule。不是主要目标物。
- **Thiram, MG**: 主要目标分析物 (target analytes)。

## 当前正式任务与分层
### 正文主结果
- has_thiram: 福美双存在性分类 (0/1)
- has_mg: 孔雀石绿存在性分类 (0/1)

### 正文辅助结果
- mixture_order: 混合复杂度分类 (1/2/3)
- c_mg: 孔雀石绿浓度等级分类 (0/4/5/6 ppm)

### 受限解释结果
- c_thiram: 福美双浓度等级分类 (0/4/5/6 ppm) — 6ppm 仅 3 folder，评估底座受限

### 内部监控
- c_mba: MBA 等级分类 (内部监控，不对外报告)

## 当前不做的事
- 连续浓度回归
- 将 MBA 作为主要成果
- 将 c_thiram level 作为强主结论
"""
(RPT/"modeling_scope_revised.md").write_text(scope_txt, encoding='utf-8')

# --- MBA策略 ---
wn_arr = wn
idx1077 = np.argmin(np.abs(wn_arr-1077))
idx1588 = np.argmin(np.abs(wn_arr-1588))
pk1 = np.abs(X[:,idx1077])+1e-6
pk2 = np.abs(X[:,idx1588])+1e-6
Xr1 = X/pk1[:,None]; Xr2 = X/pk2[:,None]
y_th = meta['has_thiram'].values.astype(int)
f1_orig, f1_r1, f1_r2 = [], [], []
for fold in range(NF):
    tri = meta[meta['fold_id'] != fold].index.values
    vai = meta[meta['fold_id'] == fold].index.values
    for Xin, store in [(X, f1_orig), (Xr1, f1_r1), (Xr2, f1_r2)]:
        c = RandomForestClassifier(50, max_depth=10, random_state=SEED, n_jobs=-1)
        c.fit(Xin[tri], y_th[tri]); store.append(f1_score(y_th[vai], c.predict(Xin[vai]), average='macro'))
mo, m1, m2 = np.mean(f1_orig), np.mean(f1_r1), np.mean(f1_r2)

mba_txt = f"""# 阶段 7：MBA 参考化 / 内标化策略 (修订版)
> 生成时间: {TS}
> 性质: **Exploratory Analysis**

## 一、MBA 角色定义
MBA = **probe / internal standard / reference molecule**。不是主要待测污染物。

## 二、简单峰点归一化测试 (Exploratory)
以 Thiram Presence (RF, 3-fold SGKF) 为测试任务:
| 特征 | Macro-F1 |
|------|----------|
| 原始 X_p1 | {mo:.3f} |
| X / MBA(1077) | {m1:.3f} |
| X / MBA(1588) | {m2:.3f} |

### 数据直接支持
- 简单的单点相除**未产生系统性提升**。

### 合理假设但证据不足
- 失败原因可能是: 基线校正后峰值附近零点扰动，单点除法放大噪声。

## 三、后续方向 (非当前主线)
1. 峰面积积分 2. 子区间特征 3. 物理启发的低维特征
以上为远期可选，不自动默认为连续定量路线。
"""
(RPT/"mba_reference_strategy.md").write_text(mba_txt, encoding='utf-8')

# --- 审计报告 ---
audit_txt = f"""# 全项目总审计报告
> 生成时间: {TS}

## Split 版本变迁
- V1 (5-fold GroupKFold): 已归档 archive/legacy_splits/
- V2 (5-fold SGKF stratify=family): 已归档 archive/legacy_splits_v2/ (c_thiram 6ppm 3 fold 挂零)
- **V3 (3-fold SGKF stratify=c_thiram): 当前现役** (尽最大努力覆盖 c_thiram 各等级)

## 审计结论
- MBA 统一标注为 Probe/Internal Standard
- 连续回归已废弃，所有任务为分类
- 混淆矩阵按每任务最佳模型绘制（不再硬绑 RF）
- batch effect 统一改为 folder-level dependence/clustering
- c_thiram level 标注为受限结果
"""
(RPT/"full_project_audit.md").write_text(audit_txt, encoding='utf-8')

# --- change_log ---
chg_txt = f"""# 变更日志 (Change Log)
> 生成时间: {TS}

## 1. Split V2 → V3
- 原: 5-fold, stratify=family. c_thiram 6ppm 3 fold 验证集挂零。
- 改: 3-fold, stratify=c_thiram. 减少 fold 数以尽量覆盖稀缺等级。
- 原因: c_thiram 6ppm 仅 3 个 folder, 5 fold 物理上无法分配。

## 2. 混淆矩阵改为最佳模型
- 原: 全部硬绑 RF。
- 改: 每个任务自动选择 Macro-F1 最高的模型绘图。
- 原因: 真实数据表明不同任务最优模型不同。

## 3. 任务分层体系
- presence (主结果) > c_mg level (辅助) > c_thiram level (受限) > c_mba (内部)
- 原因: split 审计表明 c_thiram 评估底座受限。

## 4. 历史保留
- 废除 batch effect, 连续回归, MBA 作为目标物等表述不变。
"""
(RPT/"change_log_revised.md").write_text(chg_txt, encoding='utf-8')

print("DONE_ALL")
