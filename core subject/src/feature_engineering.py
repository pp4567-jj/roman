"""
Domain-knowledge-driven feature extraction for SERS spectra.

Feature types:
    1. Peak intensities at characteristic positions
    2. Peak area (trapezoidal integration)
    3. Peak ratios (inter/intra-substance)
    4. Statistical features per spectral region
    5. Second-derivative curvature features (Round 4 new)
"""
import numpy as np
from scipy.signal import savgol_filter


def _trapz(y, x=None, axis=-1):
    """Compatibility wrapper for NumPy 1.x and 2.x."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


THIRAM_PEAKS = [
    (560,  15, 'T_560'),
    (1145, 15, 'T_1145'),
    (1380, 15, 'T_1380'),
    (1510, 15, 'T_1510'),
]

MG_PEAKS = [
    (800,  15, 'MG_800'),
    (1175, 15, 'MG_1175'),
    (1395, 15, 'MG_1395'),
    (1615, 15, 'MG_1615'),
]

MBA_PEAKS = [
    (1080, 15, 'MBA_1080'),
    (1180, 15, 'MBA_1180'),
    (1490, 15, 'MBA_1490'),
    (1590, 15, 'MBA_1590'),
]

ALL_PEAKS = THIRAM_PEAKS + MG_PEAKS + MBA_PEAKS

REGIONS = [
    (400,  600,  'region_low'),
    (600,  900,  'region_mid1'),
    (900,  1200, 'region_mid2'),
    (1200, 1500, 'region_mid3'),
    (1500, 1800, 'region_high'),
]


def _wn_to_idx(wn_array, target_wn):
    return int(np.argmin(np.abs(wn_array - target_wn)))


def extract_peak_features(X, wn, window=5):
    features = []
    names = []

    for center, half_w, label in ALL_PEAKS:
        cidx = _wn_to_idx(wn, center)

        lo = max(0, cidx - window)
        hi = min(X.shape[1], cidx + window + 1)
        peak_max = np.max(X[:, lo:hi], axis=1)
        features.append(peak_max)
        names.append(f'{label}_intensity')

        alo = _wn_to_idx(wn, center - half_w)
        ahi = _wn_to_idx(wn, center + half_w) + 1
        peak_area = _trapz(X[:, alo:ahi], wn[alo:ahi], axis=1)
        features.append(peak_area)
        names.append(f'{label}_area')

    return np.column_stack(features), names


def extract_peak_ratios(X, wn, window=5):
    features = []
    names = []

    def _peak_val(center):
        cidx = _wn_to_idx(wn, center)
        lo = max(0, cidx - window)
        hi = min(X.shape[1], cidx + window + 1)
        return np.max(X[:, lo:hi], axis=1)

    t1380 = _peak_val(1380)
    t560 = _peak_val(560)
    t1145 = _peak_val(1145)
    mg1615 = _peak_val(1615)
    mg1175 = _peak_val(1175)
    mg1395 = _peak_val(1395)
    mba1080 = _peak_val(1080)
    mba1590 = _peak_val(1590)
    mba1180 = _peak_val(1180)

    eps = 1e-8
    ratios = [
        (t1380, t560, 'ratio_T1380_T560'),
        (t1380, t1145, 'ratio_T1380_T1145'),
        (mg1615, mg1175, 'ratio_MG1615_MG1175'),
        (mg1615, mg1395, 'ratio_MG1615_MG1395'),
        (mba1590, mba1080, 'ratio_MBA1590_MBA1080'),
        (t1380, mg1615, 'ratio_T1380_MG1615'),
        (t1380, mba1080, 'ratio_T1380_MBA1080'),
        (mg1615, mba1590, 'ratio_MG1615_MBA1590'),
        (t1380 + mg1615, mba1080, 'ratio_TplusMG_MBA'),
        (mg1175, mba1180 + eps, 'ratio_MG1175_MBA1180'),
    ]

    for num, den, name in ratios:
        features.append(num / (den + eps))
        names.append(name)

    return np.column_stack(features), names


def extract_region_stats(X, wn):
    features = []
    names = []

    for lo_wn, hi_wn, label in REGIONS:
        lo_idx = _wn_to_idx(wn, lo_wn)
        hi_idx = _wn_to_idx(wn, hi_wn) + 1
        region = X[:, lo_idx:hi_idx]
        features.append(np.mean(region, axis=1))
        names.append(f'{label}_mean')
        features.append(np.std(region, axis=1))
        names.append(f'{label}_std')
        features.append(np.max(region, axis=1))
        names.append(f'{label}_max')
        features.append(np.max(region, axis=1) - np.min(region, axis=1))
        names.append(f'{label}_range')

    return np.column_stack(features), names


def extract_second_derivative_features(X, wn, sg_window=11, sg_poly=3):
    """Extract curvature-sensitive second-derivative features around key peaks."""
    d2X = np.zeros_like(X)
    for idx in range(len(X)):
        d2X[idx] = savgol_filter(X[idx], sg_window, sg_poly, deriv=2)

    features = []
    names = []
    for center, half_w, label in ALL_PEAKS:
        cidx = _wn_to_idx(wn, center)
        lo = max(0, cidx - 5)
        hi = min(X.shape[1], cidx + 6)
        features.append(np.min(d2X[:, lo:hi], axis=1))
        names.append(f'{label}_d2min')

        alo = _wn_to_idx(wn, center - half_w)
        ahi = _wn_to_idx(wn, center + half_w) + 1
        features.append(np.sum(np.abs(d2X[:, alo:ahi]), axis=1))
        names.append(f'{label}_d2energy')

    return np.column_stack(features), names


def extract_all_features(X, wn):
    f1, n1 = extract_peak_features(X, wn)
    f2, n2 = extract_peak_ratios(X, wn)
    f3, n3 = extract_region_stats(X, wn)
    f4, n4 = extract_second_derivative_features(X, wn)

    features = np.column_stack([f1, f2, f3, f4])
    names = n1 + n2 + n3 + n4
    return features, names
