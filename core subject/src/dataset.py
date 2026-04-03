"""
Dataset module: folder scanning, metadata parsing, spectrum reading,
preprocessing, and cross-validation splitting.

Folder naming convention (美Km混合光谱):
  美X = Thiram at X ppm    (X ∈ {4, 5, 6})
  KX  = MG at X ppm
  mX  = MBA at X ppm
  Combinations concatenated: 美4K5m6 = Thiram 4 + MG 5 + MBA 6

Strategy 3: MBA is a regular component, NOT internal standard.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    DATA_DIR, PROCESSED_DIR, SPLITS_DIR, OUTPUT_DIR,
    WN_MIN, WN_MAX, WN_STEP,
    SG_WINDOW, SG_POLY, ALS_LAM, ALS_P, ALS_NITER,
    COSMIC_THRESHOLD, COSMIC_WINDOW,
    N_FOLDS, SEED,
)


# ====================== Folder Name Parsing ======================

def parse_folder_name(folder_name: str) -> dict:
    """Parse folder name into substance composition and concentrations.

    Returns dict with keys:
        c_thiram, c_mg, c_mba (int: 0/4/5/6 ppm),
        has_thiram, has_mg, has_mba (bool),
        mixture_order (int: 1/2/3),
        family (str: single/binary_XX_YY/ternary)
    """
    fn = folder_name.strip()
    c_thiram, c_mg, c_mba = 0, 0, 0

    # Ternary: 美XKYmZ
    m = re.match(r'^美(\d)K(\d)m(\d)$', fn)
    if m:
        c_thiram, c_mg, c_mba = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: 美XKY (Thiram + MG)
    m = re.match(r'^美(\d)K(\d)$', fn)
    if m:
        c_thiram, c_mg = int(m.group(1)), int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: 美XmY (Thiram + MBA)
    m = re.match(r'^美(\d)m(\d)$', fn)
    if m:
        c_thiram = int(m.group(1))
        c_mba = int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Binary: KXmY (MG + MBA)
    m = re.match(r'^K(\d)m(\d)$', fn)
    if m:
        c_mg, c_mba = int(m.group(1)), int(m.group(2))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: 美X (Thiram only)
    m = re.match(r'^美(\d)$', fn)
    if m:
        c_thiram = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: KX (MG only)
    m = re.match(r'^K(\d)$', fn)
    if m:
        c_mg = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    # Single: mX (MBA only)
    m = re.match(r'^m(\d)$', fn)
    if m:
        c_mba = int(m.group(1))
        return _build_result(c_thiram, c_mg, c_mba)

    raise ValueError(f"Cannot parse folder name: {folder_name}")


def _build_result(c_thiram: int, c_mg: int, c_mba: int) -> dict:
    has_t = c_thiram > 0
    has_m = c_mg > 0
    has_b = c_mba > 0
    order = sum([has_t, has_m, has_b])

    if order == 1:
        family = 'single'
    elif order == 2:
        parts = []
        if has_t: parts.append('Thiram')
        if has_m: parts.append('MG')
        if has_b: parts.append('MBA')
        family = f'binary_{"_".join(parts)}'
    else:
        family = 'ternary'

    return {
        'c_thiram': c_thiram, 'c_mg': c_mg, 'c_mba': c_mba,
        'has_thiram': has_t, 'has_mg': has_m, 'has_mba': has_b,
        'mixture_order': order, 'family': family,
    }


# ====================== Spectrum I/O ======================

def read_bwram_spectrum(filepath: str | Path):
    """Read a BWRam CSV file, return (wavenumber, intensity) arrays or (None, None)."""
    filepath = Path(filepath)
    header_line = None
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith('Pixel,Wavelength') or stripped.startswith('Pixel, Wavelength'):
                header_line = i
                break
    if header_line is None:
        return None, None

    df = pd.read_csv(filepath, skiprows=header_line, header=0,
                     skipinitialspace=True, on_bad_lines='skip')
    cols = df.columns.tolist()

    rs_col = int_col = None
    for c in cols:
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


# ====================== Preprocessing Functions ======================

def baseline_als(y, lam=ALS_LAM, p=ALS_P, niter=ALS_NITER):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = csc_matrix(W + D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def remove_cosmic_spikes(y, threshold=COSMIC_THRESHOLD, window=COSMIC_WINDOW):
    y_out = y.copy()
    dy = np.diff(y_out)
    med = np.median(dy)
    mad = np.median(np.abs(dy - med))
    if mad == 0:
        return y_out
    mz = 0.6745 * (dy - med) / mad
    spikes = np.where(np.abs(mz) > threshold)[0]
    for idx in spikes:
        lo = max(0, idx - window)
        hi = min(len(y_out) - 1, idx + window)
        nbrs = [n for n in range(lo, hi + 1) if n != idx and n not in spikes]
        if nbrs:
            y_out[idx] = np.mean(y_out[nbrs])
    return y_out


def snv(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (X - mu) / sd


def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def augment_spectra(X, y, n_aug=5, seed=SEED):
    """Online-style spectral augmentation for training set only.

    Augmentations (applied randomly per sample):
      1. Gaussian noise: std = 0.01 * max(|x|)
      2. Intensity scaling: uniform [0.9, 1.1]
      3. Wavenumber shift: roll ±2 points
      4. Baseline tilt: add small linear slope

    Returns augmented (X_aug, y_aug) appended to originals.
    """
    rng = np.random.RandomState(seed)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_aug):
        X_new = X.copy()
        n = len(X_new)

        # 1. Gaussian noise
        noise_mask = rng.rand(n) < 0.8
        for i in np.where(noise_mask)[0]:
            scale = 0.01 * np.max(np.abs(X_new[i]))
            X_new[i] += rng.normal(0, scale, X_new[i].shape)

        # 2. Intensity scaling
        scale_mask = rng.rand(n) < 0.5
        scales = rng.uniform(0.9, 1.1, n)
        X_new[scale_mask] *= scales[scale_mask, None]

        # 3. Wavenumber shift (roll)
        shift_mask = rng.rand(n) < 0.5
        for i in np.where(shift_mask)[0]:
            shift = rng.randint(-2, 3)
            X_new[i] = np.roll(X_new[i], shift)

        # 4. Small baseline tilt
        tilt_mask = rng.rand(n) < 0.3
        for i in np.where(tilt_mask)[0]:
            slope = rng.uniform(-0.001, 0.001)
            X_new[i] += slope * np.arange(X_new[i].shape[0])

        X_aug_list.append(X_new)
        y_aug_list.append(y.copy())

    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)


def augment_spectra_mt(X, y_dict, n_aug=5, seed=SEED):
    """Multi-task version: augments X and replicates all task labels in sync."""
    rng = np.random.RandomState(seed)
    # Augment X using the single-task function with a dummy y
    dummy_y = np.zeros(len(X), dtype=int)
    X_aug, _ = augment_spectra(X, dummy_y, n_aug=n_aug, seed=seed)

    # Replicate each task's labels (n_aug+1) times
    y_dict_aug = {}
    for tid, y in y_dict.items():
        y_dict_aug[tid] = np.tile(y, n_aug + 1)

    return X_aug, y_dict_aug


# ====================== Build Metadata ======================

def build_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Scan data_dir, parse folder names, list all valid spectra."""
    records = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        csv_files = sorted(folder.glob('SP_*.csv'))
        for fp in csv_files:
            # Skip background files (SP_0.csv) and summary files (3333.csv)
            if fp.stem == 'SP_0' or fp.stem == '3333':
                continue
            records.append({
                'file_path': str(fp),
                'file_name': fp.name,
                'folder_name': folder.name,
                **parsed,
            })
    df = pd.DataFrame(records)
    df['sample_id'] = [f'S{i+1:04d}' for i in range(len(df))]
    df['group_id'] = df['folder_name']
    return df


# ====================== Preprocessing Pipeline ======================

def load_and_preprocess(meta: pd.DataFrame, rebuild=False):
    """Load spectra, interpolate to common axis, apply 4 preprocessing pipelines.

    Returns: wavenumber (1D), X_raw, X_p1, X_p2, X_p3, X_p4 (2D arrays).
    Caches results as .npy files.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cache_files = {
        'wn': PROCESSED_DIR / 'wavenumber.npy',
        'raw': PROCESSED_DIR / 'X_raw.npy',
        'p1': PROCESSED_DIR / 'X_p1.npy',
        'p2': PROCESSED_DIR / 'X_p2.npy',
        'p3': PROCESSED_DIR / 'X_p3.npy',
        'p4': PROCESSED_DIR / 'X_p4.npy',
    }

    if not rebuild and all(f.exists() for f in cache_files.values()):
        print("[Dataset] Loading cached preprocessed data...")
        return (
            np.load(cache_files['wn']),
            np.load(cache_files['raw']),
            np.load(cache_files['p1']),
            np.load(cache_files['p2']),
            np.load(cache_files['p3']),
            np.load(cache_files['p4']),
        )

    print("[Dataset] Reading and preprocessing spectra from scratch...")
    wn_common = np.arange(WN_MIN, WN_MAX + WN_STEP, WN_STEP)
    n_pts = len(wn_common)
    n = len(meta)
    X_raw = np.zeros((n, n_pts))
    errors = []

    for i, (_, row) in enumerate(meta.iterrows()):
        wn, intensity = read_bwram_spectrum(row['file_path'])
        if wn is None or len(wn) < 50:
            errors.append((i, row['file_path']))
            continue
        # Sort by wavenumber
        order = np.argsort(wn)
        wn, intensity = wn[order], intensity[order]
        # Clip to valid range
        mask = (wn >= WN_MIN - 20) & (wn <= WN_MAX + 20)
        wn, intensity = wn[mask], intensity[mask]
        if len(wn) < 50:
            errors.append((i, row['file_path']))
            continue
        # Interpolate to common axis
        f_interp = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)
        X_raw[i] = f_interp(wn_common)

    if errors:
        print(f"  WARNING: {len(errors)} files could not be read/interpolated.")

    # Pipeline P1: cosmic → smooth → baseline → SNV
    print("  P1: cosmic removal → SG smoothing → ALS baseline → SNV")
    X_p1 = np.zeros_like(X_raw)
    for i in range(n):
        y = remove_cosmic_spikes(X_raw[i])
        y = savgol_filter(y, SG_WINDOW, SG_POLY)
        bl = baseline_als(y)
        X_p1[i] = y - bl
    X_p1 = snv(X_p1)

    # Pipeline P2: smooth → baseline → 1st derivative → SNV
    print("  P2: SG smoothing → ALS baseline → 1st derivative → SNV")
    X_p2 = np.zeros_like(X_raw)
    for i in range(n):
        y = savgol_filter(X_raw[i], SG_WINDOW, SG_POLY)
        bl = baseline_als(y)
        y = y - bl
        X_p2[i] = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=1)
    X_p2 = snv(X_p2)

    # Pipeline P3: baseline → vector normalization
    print("  P3: ALS baseline → vector normalization")
    X_p3 = np.zeros_like(X_raw)
    for i in range(n):
        bl = baseline_als(X_raw[i])
        X_p3[i] = X_raw[i] - bl
    X_p3 = vector_normalize(X_p3)

    # Pipeline P4: cosmic → smooth → baseline (NO normalization)
    print("  P4: cosmic removal → SG smoothing → ALS baseline (no normalization)")
    X_p4 = np.zeros_like(X_raw)
    for i in range(n):
        y = remove_cosmic_spikes(X_raw[i])
        y = savgol_filter(y, SG_WINDOW, SG_POLY)
        bl = baseline_als(y)
        X_p4[i] = y - bl

    # Save caches
    np.save(cache_files['wn'], wn_common)
    np.save(cache_files['raw'], X_raw)
    np.save(cache_files['p1'], X_p1)
    np.save(cache_files['p2'], X_p2)
    np.save(cache_files['p3'], X_p3)
    np.save(cache_files['p4'], X_p4)
    print(f"  Saved: {n} spectra × {n_pts} points, 4 preprocessing versions")
    return wn_common, X_raw, X_p1, X_p2, X_p3, X_p4


# ====================== Splitting ======================

def create_cv_splits(meta: pd.DataFrame, rebuild=False) -> pd.DataFrame:
    """5-fold StratifiedGroupKFold, group=folder_name, stratify=family.

    Returns meta with 'fold_id' column appended.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLITS_DIR / 'cv_split_v5.csv'

    if not rebuild and split_path.exists():
        print("[Split] Loading cached split...")
        return pd.read_csv(split_path)

    print(f"[Split] Creating {N_FOLDS}-fold StratifiedGroupKFold...")
    groups = meta['folder_name'].values
    # Stratify by 7-class presence pattern (finer than 5-class family)
    y_strat = ('T' + meta['has_thiram'].astype(int).astype(str)
               + '_M' + meta['has_mg'].astype(int).astype(str)
               + '_B' + meta['has_mba'].astype(int).astype(str)).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_ids = np.zeros(len(meta), dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(meta, y_strat, groups)):
        fold_ids[val_idx] = fold

    meta_split = meta.copy()
    meta_split['fold_id'] = fold_ids
    meta_split.to_csv(split_path, index=False, encoding='utf-8-sig')

    # Print split summary
    for fold in range(N_FOLDS):
        val = meta_split[meta_split['fold_id'] == fold]
        print(f"  Fold {fold}: {val['folder_name'].nunique()} folders, {len(val)} spectra")

    return meta_split
