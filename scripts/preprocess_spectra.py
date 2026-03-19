"""
阶段 3：光谱标准化与预处理流水线
=================================
preprocess_spectra.py

将所有 BWRam 格式光谱统一为相同波数轴，并生成三套预处理版本：
  P1: cosmic spike removal → SG smoothing → baseline correction → SNV
  P2: SG smoothing → baseline correction → 1st derivative → SNV
  P3: baseline correction → vector normalization

输出:
  data/processed/wavenumber.npy
  data/processed/X_raw.npy
  data/processed/X_p1.npy
  data/processed/X_p2.npy
  data/processed/X_p3.npy
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
PROJECT_ROOT = Path(r"d:\通过拉曼光谱预测物及其浓度")
RAW_DATA_DIR = PROJECT_ROOT / "混合数据" / "mba+福+孔"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "figures" / "preprocessing"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 统一波数轴参数
# 基于数据清点，最大的公共区间约 400-1800 cm-1
# 但为了保留更多信息，我们用 200-1800 cm-1，步长约 2 cm-1
WN_MIN = 400.0
WN_MAX = 1800.0
WN_STEP = 1.0  # ~1 cm-1 resolution
RANDOM_SEED = 42

# SG 平滑参数
SG_WINDOW = 11
SG_POLYORDER = 3

# 基线校正参数 (ALS)
ALS_LAM = 1e6
ALS_P = 0.01
ALS_NITER = 10


# ====================== 读取函数 ======================

def read_bwram_spectrum(filepath):
    """读取 BWRam CSV 文件，返回 (wavenumber, intensity)。"""
    header_line = None
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('Pixel,Wavelength') or line.strip().startswith('Pixel, Wavelength'):
                header_line = i
                break
    if header_line is None:
        return None, None
    
    df = pd.read_csv(filepath, skiprows=header_line, header=0,
                     skipinitialspace=True, on_bad_lines='skip')
    cols = df.columns.tolist()
    
    rs_col = None
    int_col = None
    for j, c in enumerate(cols):
        cl = c.strip().lower()
        if 'raman shift' in cl:
            rs_col = c
        if 'dark subtracted' in cl:
            int_col = c
    
    if rs_col is None or int_col is None:
        if len(cols) >= 8:
            rs_col, int_col = cols[3], cols[7]
        else:
            return None, None
    
    wn = pd.to_numeric(df[rs_col], errors='coerce').values
    intensity = pd.to_numeric(df[int_col], errors='coerce').values
    
    valid = ~np.isnan(wn) & ~np.isnan(intensity)
    return wn[valid], intensity[valid]


# ====================== 预处理函数 ======================

def baseline_als(y, lam=ALS_LAM, p=ALS_P, niter=ALS_NITER):
    """Asymmetric Least Squares Smoothing baseline estimation."""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def snv(X):
    """Standard Normal Variate normalization (row-wise)."""
    means = np.mean(X, axis=1, keepdims=True)
    stds = np.std(X, axis=1, keepdims=True)
    stds[stds == 0] = 1  # avoid division by zero
    return (X - means) / stds


def vector_normalize(X):
    """Vector normalization (L2 norm per row)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def remove_cosmic_spikes(y, threshold=7.0, window=5):
    """
    Remove cosmic ray spikes using modified z-score detection.
    """
    y_out = y.copy()
    dy = np.diff(y_out)
    median_dy = np.median(dy)
    mad_dy = np.median(np.abs(dy - median_dy))
    if mad_dy == 0:
        return y_out
    modified_z = 0.6745 * (dy - median_dy) / mad_dy
    
    spike_indices = np.where(np.abs(modified_z) > threshold)[0]
    for idx in spike_indices:
        w_start = max(0, idx - window)
        w_end = min(len(y_out) - 1, idx + window)
        neighbors = list(range(w_start, idx)) + list(range(idx + 1, w_end + 1))
        # Remove other spike indices from neighbors
        neighbors = [n for n in neighbors if n not in spike_indices]
        if len(neighbors) > 0:
            y_out[idx] = np.mean(y_out[neighbors])
    return y_out


def sg_smooth(y, window_length=SG_WINDOW, polyorder=SG_POLYORDER):
    """Savitzky-Golay smoothing."""
    if len(y) < window_length:
        return y
    return savgol_filter(y, window_length, polyorder)


def first_derivative(y, window_length=SG_WINDOW, polyorder=SG_POLYORDER):
    """Compute first derivative using SG filter."""
    if len(y) < window_length:
        return np.gradient(y)
    return savgol_filter(y, window_length, polyorder, deriv=1)


# ====================== 主流程 ======================

def main():
    np.random.seed(RANDOM_SEED)
    
    print("=" * 60)
    print("阶段 3：光谱标准化与预处理流水线")
    print("=" * 60)
    
    # 1. 读取 metadata
    print("\n[1/6] 读取 metadata_v1.csv...")
    meta = pd.read_csv(METADATA_PATH)
    n_spectra = len(meta)
    print(f"  共 {n_spectra} 条光谱")
    
    # 2. 构建统一波数轴
    print(f"\n[2/6] 构建统一波数轴: {WN_MIN} - {WN_MAX} cm⁻¹, 步长 {WN_STEP} cm⁻¹")
    wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
    n_points = len(wn_common)
    print(f"  统一波数点数: {n_points}")
    
    # 3. 读取并插值所有光谱到统一波数轴
    print(f"\n[3/6] 读取并插值所有光谱...")
    X_raw = np.zeros((n_spectra, n_points))
    read_errors = []
    interpolation_notes = []
    
    for i, row in meta.iterrows():
        fpath = row['file_path']
        wn, intensity = read_bwram_spectrum(fpath)
        
        if wn is None or len(wn) == 0:
            read_errors.append((i, row['sample_id'], fpath, "无法读取"))
            continue
        
        # 检查波数覆盖范围
        if wn.min() > WN_MIN or wn.max() < WN_MAX:
            # 有些光谱可能不完全覆盖目标范围
            actual_min = max(wn.min(), WN_MIN)
            actual_max = min(wn.max(), WN_MAX)
            if actual_max - actual_min < 500:
                read_errors.append((i, row['sample_id'], fpath, 
                                    f"波数范围太窄: {wn.min():.1f}-{wn.max():.1f}"))
                continue
            interpolation_notes.append(
                f"{row['sample_id']}: 部分波数范围缺失 ({wn.min():.1f}-{wn.max():.1f})")
        
        # 确保波数是升序排列
        if wn[0] > wn[-1]:
            wn = wn[::-1]
            intensity = intensity[::-1]
        
        # 插值到统一波数轴
        try:
            f_interp = interp1d(wn, intensity, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
            X_raw[i, :] = f_interp(wn_common)
        except Exception as e:
            read_errors.append((i, row['sample_id'], fpath, str(e)[:80]))
    
    print(f"  读取成功: {n_spectra - len(read_errors)}")
    if read_errors:
        print(f"  读取失败: {len(read_errors)}")
        for idx, sid, fp, note in read_errors[:5]:
            print(f"    {sid}: {note}")
    
    # 4. 三套预处理
    print(f"\n[4/6] 执行预处理流水线...")
    
    # P1: cosmic spike → SG smooth → baseline → SNV
    print("  P1: spike removal → SG smooth → baseline → SNV")
    X_p1 = np.zeros_like(X_raw)
    for i in range(n_spectra):
        y = X_raw[i, :].copy()
        y = remove_cosmic_spikes(y)
        y = sg_smooth(y)
        bl = baseline_als(y)
        y = y - bl
        X_p1[i, :] = y
    X_p1 = snv(X_p1)
    
    # P2: SG smooth → baseline → 1st derivative → SNV
    print("  P2: SG smooth → baseline → 1st derivative → SNV")
    X_p2 = np.zeros_like(X_raw)
    for i in range(n_spectra):
        y = X_raw[i, :].copy()
        y = sg_smooth(y)
        bl = baseline_als(y)
        y = y - bl
        y = first_derivative(y)
        X_p2[i, :] = y
    X_p2 = snv(X_p2)
    
    # P3: baseline → vector normalization
    print("  P3: baseline → vector normalization")
    X_p3 = np.zeros_like(X_raw)
    for i in range(n_spectra):
        y = X_raw[i, :].copy()
        bl = baseline_als(y)
        y = y - bl
        X_p3[i, :] = y
    X_p3 = vector_normalize(X_p3)
    
    # 5. 保存
    print(f"\n[5/6] 保存预处理结果...")
    np.save(OUTPUT_DIR / "wavenumber.npy", wn_common)
    np.save(OUTPUT_DIR / "X_raw.npy", X_raw)
    np.save(OUTPUT_DIR / "X_p1.npy", X_p1)
    np.save(OUTPUT_DIR / "X_p2.npy", X_p2)
    np.save(OUTPUT_DIR / "X_p3.npy", X_p3)
    print(f"  wavenumber.npy: shape={wn_common.shape}")
    print(f"  X_raw.npy: shape={X_raw.shape}")
    print(f"  X_p1.npy: shape={X_p1.shape}")
    print(f"  X_p2.npy: shape={X_p2.shape}")
    print(f"  X_p3.npy: shape={X_p3.shape}")
    
    # 6. 生成示例图和报告
    print(f"\n[6/6] 生成预处理报告和示例图...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 示例图：选3条不同类型的谱
        single_idx = meta[meta['family'] == 'single'].index[0]
        binary_idx = meta[meta['family'].str.startswith('binary')].index[0]
        ternary_idx = meta[meta['family'] == 'ternary'].index[0]
        
        example_indices = [single_idx, binary_idx, ternary_idx]
        example_labels = [
            f"单物质 ({meta.loc[single_idx, 'folder_name']})",
            f"二元 ({meta.loc[binary_idx, 'folder_name']})",
            f"三元 ({meta.loc[ternary_idx, 'folder_name']})"
        ]
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        for j, (idx, label) in enumerate(zip(example_indices, example_labels)):
            axes[0].plot(wn_common, X_raw[idx, :], label=label, alpha=0.8)
            axes[1].plot(wn_common, X_p1[idx, :], label=label, alpha=0.8)
            axes[2].plot(wn_common, X_p2[idx, :], label=label, alpha=0.8)
            axes[3].plot(wn_common, X_p3[idx, :], label=label, alpha=0.8)
        
        titles = ['原始谱 (Raw)', 'P1: Spike→SG→Baseline→SNV', 
                  'P2: SG→Baseline→1st Deriv→SNV', 'P3: Baseline→Vector Norm']
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=13)
            ax.set_ylabel('Intensity (a.u.)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Raman Shift (cm⁻¹)')
        plt.suptitle('预处理流水线对比 — 示例光谱', fontsize=15, y=0.98)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / 'preprocessing_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 原始 vs P1 单谱对比图
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        idx0 = single_idx
        wn0, int0 = read_bwram_spectrum(meta.loc[idx0, 'file_path'])
        if wn0[0] > wn0[-1]:
            wn0 = wn0[::-1]
            int0 = int0[::-1]
        axes[0].plot(wn0, int0, 'b-', alpha=0.7, label='原始谱（完整波数范围）')
        axes[0].set_title('原始 BWRam 光谱', fontsize=13)
        axes[0].set_ylabel('Dark Subtracted Intensity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(wn_common, X_p1[idx0, :], 'r-', alpha=0.7, label='P1 预处理后')
        axes[1].set_title('P1 预处理后', fontsize=13)
        axes[1].set_xlabel('Raman Shift (cm⁻¹)')
        axes[1].set_ylabel('SNV Intensity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'示例: {meta.loc[idx0, "folder_name"]} / {meta.loc[idx0, "file_name"]}', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / 'raw_vs_p1_example.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  图片已保存到 figures/preprocessing/")
        figures_ok = True
    except Exception as e:
        print(f"  警告：图片生成失败: {e}")
        figures_ok = False
    
    # 生成报告
    report = f"""# 阶段 3：预处理报告

> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、波数轴统一

| 参数 | 值 |
|------|-----|
| 目标波数范围 | {WN_MIN} - {WN_MAX} cm⁻¹ |
| 波数步长 | {WN_STEP} cm⁻¹ |
| 统一后点数 | {n_points} |
| 输入光谱数 | {n_spectra} |
| 插值方法 | 线性插值 (scipy.interpolate.interp1d) |

## 二、预处理流水线说明

### P1: Spike Removal → SG Smoothing → Baseline Correction → SNV
- **Cosmic spike removal**: Modified Z-score 检测，阈值={7.0}，窗口={5}
- **SG smoothing**: window={SG_WINDOW}, polyorder={SG_POLYORDER}
- **Baseline correction**: Asymmetric Least Squares (ALS), λ={ALS_LAM:.0e}, p={ALS_P}, niter={ALS_NITER}
- **SNV**: Standard Normal Variate (行级标准化)

### P2: SG Smoothing → Baseline Correction → 1st Derivative → SNV
- **SG smoothing**: window={SG_WINDOW}, polyorder={SG_POLYORDER}
- **Baseline correction**: ALS 同 P1
- **1st derivative**: SG 一阶导数，window={SG_WINDOW}, polyorder={SG_POLYORDER}
- **SNV**: Standard Normal Variate

### P3: Baseline Correction → Vector Normalization
- **Baseline correction**: ALS 同 P1
- **Vector normalization**: L2 范数归一化

## 三、输出文件

| 文件 | 形状 | 说明 |
|------|------|------|
| wavenumber.npy | ({n_points},) | 统一波数轴 |
| X_raw.npy | ({n_spectra}, {n_points}) | 插值后原始谱 |
| X_p1.npy | ({n_spectra}, {n_points}) | P1 预处理 |
| X_p2.npy | ({n_spectra}, {n_points}) | P2 预处理 |
| X_p3.npy | ({n_spectra}, {n_points}) | P3 预处理 |
"""
    
    if read_errors:
        report += f"""
## 四、读取/插值问题

> [!WARNING]
> {len(read_errors)} 条光谱在读取或插值过程中出现问题，对应行填充为全零。

| sample_id | 文件 | 问题 |
|-----------|------|------|
"""
        for idx, sid, fp, note in read_errors:
            report += f"| {sid} | {Path(fp).name} | {note} |\n"
    
    if interpolation_notes:
        report += f"""
## 五、波数范围不完整的光谱

| 说明 |
|------|
"""
        for note in interpolation_notes[:20]:
            report += f"| {note} |\n"
        if len(interpolation_notes) > 20:
            report += f"| ... 共 {len(interpolation_notes)} 条 |\n"
    
    if figures_ok:
        report += f"""
## 六、预处理前后对比图

### 三类谱对比
![预处理对比](../figures/preprocessing/preprocessing_comparison.png)

### 原始 vs P1
![原始vs预处理](../figures/preprocessing/raw_vs_p1_example.png)
"""
    
    report += f"""
## 七、数据质量备注

1. 所有预处理脚本使用固定随机种子 (seed={RANDOM_SEED})，保证可复现性。
2. 波数范围 {WN_MIN}-{WN_MAX} cm⁻¹ 是根据所有光谱的公共覆盖区域确定的。
3. ALS 基线校正的参数选取是保守估计，后续可根据 EDA 结果微调。
4. P1 适合直接作为大多数模型的输入（去噪+去基线+标准化）。
5. P2 包含一阶导数，能突出光谱特征的变化率，对微小差异更敏感。
6. P3 最简单，仅做基线校正和归一化，保留更多原始光谱形态。
"""
    
    report_path = REPORT_DIR / "preprocess_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'=' * 60}")
    print(f"预处理完成！")
    print(f"  光谱数: {n_spectra}")
    print(f"  波数点数: {n_points}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  报告: {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
