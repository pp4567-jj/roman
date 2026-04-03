# 服务器部署手册 — SERS 多组分拉曼光谱分类项目

> **给 AI 助手的说明**：这是一个完整的项目迁移手册。请严格按照步骤执行，不要修改任何源代码，直接跑即可。

---

## 一、项目背景

这是一个 **SERS（表面增强拉曼光谱）多组分分析** 项目，目标是 SCI 三区论文。通过拉曼光谱同时识别混合溶液中 **三种物质（Thiram 福美双、MG 孔雀石绿、MBA 巯基苯甲酸）的存在和浓度**。

### 数据概况
- **954 条光谱**，来自 63 个文件夹（每个文件夹对应一种浓度组合）
- 光谱范围：400–1800 cm⁻¹，1401 个波数点
- 浓度等级：0（不存在）、4ppm、5ppm、6ppm
- 文件夹命名规则：`美X` = Thiram Xppm，`KX` = MG Xppm，`mX` = MBA Xppm
  - 例：`美4K5m6` = Thiram 4ppm + MG 5ppm + MBA 6ppm

### 7 个分类任务
| 任务 ID | 描述 | 类别数 |
|---------|------|--------|
| T1_thiram_conc | Thiram 浓度分类 | 4 (0/4/5/6) |
| T2_mg_conc | MG 浓度分类 | 4 |
| T3_mba_conc | MBA 浓度分类 | 4 |
| T4_thiram_pres | Thiram 有无检测 | 2 (0/1) |
| T5_mg_pres | MG 有无检测 | 2 |
| T6_mba_pres | MBA 有无检测 | 2 |
| T7_mixture_order | 混合物复杂度 | 3 (单/双/三组分) |

### 8 个模型
| 类型 | 模型名 | 说明 |
|------|--------|------|
| ML 基线 | RF | Random Forest |
| ML 基线 | SVM | Support Vector Machine |
| ML 基线 | PLS-DA | Partial Least Squares |
| DL 单任务 | 1D-CNN | 4层1D-CNN (64/128/256/256 通道) |
| DL 单任务 | 1D-ResNet | 残差网络 (64/128/256 通道) |
| DL 多任务 | MT-CNN | 共享CNN + 任务专属线性头 |
| DL 多任务 | MT-ResNet | 共享ResNet + 任务专属线性头 |
| DL 多任务 | **MT-KAN-CNN** | 共享CNN + **KAN(B-spline)头**（创新点） |

### 5 个预处理变体
每个模型在 5 种预处理数据上分别训练：raw, p1, p2, p3, p4

### 评估方式
- 5-fold **StratifiedGroupKFold**（group=文件夹名，保证同一文件夹不跨fold）
- 指标：Macro-F1、Balanced Accuracy

---

## 二、项目文件结构

```
项目根目录/
├── src/
│   ├── config.py       # 全局配置（路径、超参数、任务定义）
│   ├── dataset.py      # 数据加载、预处理、CV划分、数据增强
│   ├── models.py       # 全部8个模型定义（ML+DL+MT+KAN）
│   ├── train_eval.py   # 交叉验证训练循环
│   └── visualize.py    # 可视化
├── batch_train.py      # ★ 一键批量训练脚本（你需要运行的）
├── run_pipeline.py     # CLI 入口（可选）
├── requirements.txt    # 依赖
├── data/
│   ├── processed/      # 预处理后的 .npy 文件（已有，不需重新生成）
│   │   ├── wavenumber.npy
│   │   ├── X_raw.npy, X_p1.npy, X_p2.npy, X_p3.npy, X_p4.npy
│   ├── splits/
│   │   └── cv_split_v5.csv   # 交叉验证划分（已有）
│   └── models/         # 训练结果输出目录
│       ├── cv_results_summary_*.csv  # ML结果已有
│       └── cv_results_detail_*.csv
└── 混合数据/            # 原始光谱 CSV（已处理为 .npy，不需要直接读）
```

---

## 三、服务器环境信息

- **GPU**: NVIDIA RTX 3090 24GB
- **CPU**: AMD EPYC 7601 (8 cores)
- **RAM**: 30GB
- **CUDA**: 12.2

---

## 四、步骤 1 — 上传项目

将整个项目文件夹上传到服务器，建议放在 `/root/sers_project/` 或 home 目录下。

**必须上传的目录/文件**：
```
src/                   (全部 .py 文件)
data/processed/        (6 个 .npy 文件，约 60MB)
data/splits/           (cv_split_v5.csv)
data/models/           (已有的 ML 结果 CSV，约 15 个文件)
batch_train.py
requirements.txt
```

**不需要上传的**：
- `混合数据/` 文件夹（几百MB原始CSV，不需要）
- `figures/`、`reports/`、`附/`
- `.venv/`

---

## 五、步骤 2 — 环境配置

```bash
cd /path/to/sers_project

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 期望输出：CUDA: True, GPU: NVIDIA GeForce RTX 3090
```

---

## 六、步骤 3 — 验证数据和代码

```bash
# 检查数据文件完整性
python -c "
import numpy as np, pandas as pd
X = np.load('data/processed/X_p1.npy')
meta = pd.read_csv('data/splits/cv_split_v5.csv')
print(f'Spectra: {X.shape}, Samples: {len(meta)}')
print(f'Columns: {list(meta.columns)}')
print(f'Folds: {sorted(meta.fold.unique())}')
"
# 期望输出：Spectra: (954, 1401), Samples: 954, Folds: [0, 1, 2, 3, 4]

# 检查模型导入
python -c "from src.models import MODEL_REGISTRY; print(list(MODEL_REGISTRY.keys()))"
# 期望输出：['RF', 'SVM', 'PLS-DA', '1D-CNN', '1D-ResNet', 'MT-CNN', 'MT-ResNet', 'MT-KAN-CNN']

# 检查已有 ML 结果
python -c "
import os, pandas as pd
for tag in ['raw','p1','p2','p3','p4']:
    path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        models = sorted(df['Model'].unique())
        print(f'{tag}: {models}')
"
# 期望：每个 tag 都有 ['PLS-DA', 'RF', 'SVM']
```

---

## 七、步骤 4 — 运行训练 ★

```bash
# 一键训练（ML会自动跳过，只跑DL和MT）
python batch_train.py 2>&1 | tee training_log.txt
```

### 训练流程说明

`batch_train.py` 会自动执行：

1. **Phase 1 — ML 模型**（RF/SVM/PLS-DA）：检测到 CSV 已存在，**自动跳过**
2. **Phase 2 — DL 单任务模型**（1D-CNN × 5变体 × 7任务 × 5折 + 1D-ResNet 同上）
   - 每个模型独立训练每个任务，使用数据增强（3x）
   - 早停机制：patience=15，最多 200 epoch
   - batch_size=256 充分利用 3090
3. **Phase 3 — MT 多任务模型**（MT-CNN / MT-ResNet / MT-KAN-CNN × 5变体 × 5折）
   - 一次训练同时覆盖 7 个任务（不确定性加权损失）
   - 比 DL 单任务快很多

### 预计时间
- DL 阶段：~45-50 分钟
- MT 阶段：~10-15 分钟
- **总计：约 60 分钟**

### 训练期间监控

另开一个终端监控 GPU：
```bash
watch -n 2 nvidia-smi
```
期望看到：GPU 利用率 > 80%，显存占用 8-16GB

---

## 八、步骤 5 — 验证结果

训练完成后运行：
```bash
python -c "
import os, pandas as pd
print('='*70)
print('TRAINING RESULTS VERIFICATION')
print('='*70)
for tag in ['raw','p1','p2','p3','p4']:
    path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        models = sorted(df['Model'].unique())
        print(f'\n{tag}: {len(models)} models — {models}')
        # Show best F1 per task
        for task in df['Task'].unique():
            sub = df[df['Task']==task].sort_values('F1_mean', ascending=False)
            best = sub.iloc[0]
            print(f'  {task}: best={best.Model} F1={best.F1_mean:.4f}±{best.F1_std:.4f}')
    else:
        print(f'{tag}: NO RESULTS FILE')
print('\n' + '='*70)
# Check expected model count
expected = ['RF','SVM','PLS-DA','1D-CNN','1D-ResNet','MT-CNN','MT-ResNet','MT-KAN-CNN']
for tag in ['raw','p1','p2','p3','p4']:
    path = f'data/models/cv_results_summary_{tag}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = [m for m in expected if m not in df['Model'].values]
        if missing:
            print(f'WARNING: {tag} missing models: {missing}')
        else:
            print(f'{tag}: ✓ all 8 models present')
"
```

### 期望结果
- 每个 tag（raw/p1/p2/p3/p4）都有 **8 个模型**的结果
- 每个 summary CSV 有 7 tasks × 8 models = **56 行**
- F1 分数参考范围：
  - 存在检测（T4-T6）：> 0.85
  - 浓度分类（T1-T3）：> 0.50
  - 混合复杂度（T7）：> 0.50

---

## 九、步骤 6 — 下载结果

训练完成并验证后，需要下载的文件：

```
data/models/
├── cv_results_summary_raw.csv    # 汇总指标
├── cv_results_summary_p1.csv
├── cv_results_summary_p2.csv
├── cv_results_summary_p3.csv
├── cv_results_summary_p4.csv
├── cv_results_detail_raw.csv     # 逐折明细
├── cv_results_detail_p1.csv
├── cv_results_detail_p2.csv
├── cv_results_detail_p3.csv
├── cv_results_detail_p4.csv
├── cv_predictions_raw.csv        # 逐样本预测（可选）
├── cv_predictions_p1.csv
├── cv_predictions_p2.csv
├── cv_predictions_p3.csv
└── cv_predictions_p4.csv
```

共 15 个 CSV 文件，体积很小（几 MB）。

也下载 `training_log.txt`（训练日志）。

---

## 十、常见问题

### Q: CUDA out of memory
降低 batch_size：编辑 `src/config.py`，把 `DL_BATCH = 256` 改为 `128`

### Q: `ModuleNotFoundError: No module named 'src'`
确保在项目根目录运行 `python batch_train.py`，不要 cd 到 src 里

### Q: 训练中断了怎么办？
直接重新运行 `python batch_train.py`。ML 结果会自动跳过，但 DL/MT 会从头训练（因为模型权重不保存，只保存评估结果 CSV）。如果 DL 部分已完成但 MT 中断，可以只跑 MT：
```python
# 在 batch_train.py 里注释掉 Phase 2 的 DL 循环，只保留 Phase 3
```

### Q: 想只测试某一个模型
```bash
python -c "
from src.config import SPLITS_DIR, PROCESSED_DIR
import numpy as np, pandas as pd
from src.models import MODEL_REGISTRY
from src.train_eval import run_cv_evaluation, aggregate_results, save_results

meta = pd.read_csv(SPLITS_DIR / 'cv_split_v5.csv')
for col in ['has_thiram','has_mg','has_mba']:
    meta[col] = meta[col].astype(str).str.strip().str.lower().map({'true':1,'false':0,'1':1,'0':0,'1.0':1,'0.0':0}).fillna(0).astype(int)
X = np.load(PROCESSED_DIR / 'X_p1.npy')
res = run_cv_evaluation(meta, X, model_names=['MT-KAN-CNN'], preprocess_tag='p1')
print(res)
"
```

---

## 关键超参数汇总（已针对 3090 优化）

| 参数 | 值 | 说明 |
|------|------|------|
| DL_BATCH | 256 | 大batch利用3090算力 |
| DL_EPOCHS | 200 | 更多收敛空间 |
| DL_PATIENCE | 15 | 早停 |
| DL_LR | 1e-3 | Adam 学习率 |
| AUG_N | 3 | 3倍数据增强 |
| KAN_GRID_SIZE | 8 | B-spline 网格密度 |
| N_FOLDS | 5 | 5折交叉验证 |
| CNN 通道 | 64/128/256/256 | 翻倍后的通道宽度 |
| ResNet 通道 | 64/128/256 | 翻倍后的通道宽度 |
