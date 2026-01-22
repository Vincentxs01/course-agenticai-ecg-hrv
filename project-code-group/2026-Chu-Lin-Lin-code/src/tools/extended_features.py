# SPDX-License-Identifier: Apache-2.0
"""Extended HRV feature extraction: 20 ECG-derived features for comprehensive analysis."""

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
from typing import Optional


def extract_extended_features(
    rr_intervals: np.ndarray,
    r_peaks: Optional[np.ndarray] = None,
    filtered_signal: Optional[np.ndarray] = None,
    fs: int = 700
) -> dict:
    """
    Extract 20 HRV features from RR intervals and optionally from ECG signal.

    Features extracted:
    Time-domain (10):
        1. mean_rr - Mean RR interval (ms)
        2. sdnn - Standard deviation of RR intervals (ms)
        3. rmssd - Root mean square of successive differences (ms)
        4. pnn50 - Percentage of successive RR differences > 50ms
        5. mean_hr - Mean heart rate (bpm)
        6. std_hr - Standard deviation of heart rate (bpm)
        7. cv_rr - Coefficient of variation (SDNN/meanRR)
        8. range_rr - Range of RR intervals (max - min) (ms)
        9. median_rr - Median RR interval (ms)
        10. iqr_rr - Interquartile range of RR intervals (ms)

    Frequency-domain (6):
        11. vlf_power - Very low frequency power (ms^2)
        12. lf_power - Low frequency power (ms^2)
        13. hf_power - High frequency power (ms^2)
        14. lf_hf_ratio - LF/HF ratio
        15. lf_nu - LF normalized units (%)
        16. hf_nu - HF normalized units (%)

    Non-linear (4):
        17. sd1 - Poincare plot SD1 (ms)
        18. sd2 - Poincare plot SD2 (ms)
        19. sd_ratio - SD1/SD2 ratio
        20. sample_entropy - Sample entropy of RR intervals

    Args:
        rr_intervals: RR intervals in milliseconds
        r_peaks: Optional R-peak indices (for signal-based features)
        filtered_signal: Optional filtered ECG signal
        fs: Sampling frequency in Hz

    Returns:
        dict: Dictionary with 20 named features
    """
    features = {}

    # Handle empty or insufficient data
    if len(rr_intervals) < 10:
        return _get_nan_features()

    rr = np.array(rr_intervals, dtype=np.float64)

    # =========================================================================
    # TIME-DOMAIN FEATURES (10)
    # =========================================================================

    # 1. Mean RR interval
    features['mean_rr'] = np.mean(rr)

    # 2. SDNN - Standard deviation of NN intervals
    features['sdnn'] = np.std(rr, ddof=1)

    # 3. RMSSD - Root mean square of successive differences
    diff_rr = np.diff(rr)
    features['rmssd'] = np.sqrt(np.mean(diff_rr ** 2))

    # 4. pNN50 - Percentage of successive differences > 50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    features['pnn50'] = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0

    # 5. Mean heart rate
    hr = 60000 / rr  # Convert RR (ms) to HR (bpm)
    features['mean_hr'] = np.mean(hr)

    # 6. STD heart rate
    features['std_hr'] = np.std(hr, ddof=1)

    # 7. Coefficient of variation (CV = SDNN/meanRR)
    features['cv_rr'] = features['sdnn'] / features['mean_rr'] if features['mean_rr'] > 0 else np.nan

    # 8. Range of RR intervals
    features['range_rr'] = np.max(rr) - np.min(rr)

    # 9. Median RR interval
    features['median_rr'] = np.median(rr)

    # 10. Interquartile range of RR intervals
    features['iqr_rr'] = np.percentile(rr, 75) - np.percentile(rr, 25)

    # =========================================================================
    # FREQUENCY-DOMAIN FEATURES (6)
    # =========================================================================

    freq_features = _extract_frequency_features(rr)
    features.update(freq_features)

    # =========================================================================
    # NON-LINEAR FEATURES (4)
    # =========================================================================

    nonlinear_features = _extract_nonlinear_features(rr)
    features.update(nonlinear_features)

    return features


def _extract_frequency_features(
    rr: np.ndarray,
    fs_resample: float = 4.0,
    vlf_band: tuple = (0.003, 0.04),
    lf_band: tuple = (0.04, 0.15),
    hf_band: tuple = (0.15, 0.4)
) -> dict:
    """Extract frequency-domain features using Welch's method."""

    if len(rr) < 20:
        return {
            'vlf_power': np.nan,
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'lf_nu': np.nan,
            'hf_nu': np.nan,
        }

    # Create time vector (cumulative sum of RR intervals)
    time_rr = np.cumsum(rr) / 1000  # Convert to seconds
    time_rr = time_rr - time_rr[0]  # Start from 0

    # Resample to uniform sampling rate
    duration = time_rr[-1]
    if duration < 10:  # Need at least 10 seconds
        return {
            'vlf_power': np.nan,
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'lf_nu': np.nan,
            'hf_nu': np.nan,
        }

    time_uniform = np.arange(0, duration, 1 / fs_resample)

    if len(time_uniform) < 20:
        return {
            'vlf_power': np.nan,
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'lf_nu': np.nan,
            'hf_nu': np.nan,
        }

    # Interpolate RR intervals
    try:
        interp_func = interp1d(time_rr, rr, kind='cubic', fill_value='extrapolate')
        rr_resampled = interp_func(time_uniform)
    except Exception:
        return {
            'vlf_power': np.nan,
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'lf_nu': np.nan,
            'hf_nu': np.nan,
        }

    # Remove mean (detrend)
    rr_detrended = rr_resampled - np.mean(rr_resampled)

    # Compute PSD using Welch's method
    nperseg = min(256, len(rr_detrended))
    freqs, psd = welch(rr_detrended, fs=fs_resample, nperseg=nperseg)

    # Calculate band powers using trapezoid integration
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

    vlf_power = _trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
    lf_power = _trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
    hf_power = _trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0

    # 14. LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    # 15-16. Normalized units (excluding VLF)
    lf_hf_sum = lf_power + hf_power
    lf_nu = (lf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
    hf_nu = (hf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan

    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'lf_nu': lf_nu,
        'hf_nu': hf_nu,
    }


def _extract_nonlinear_features(rr: np.ndarray) -> dict:
    """Extract non-linear features (Poincare + entropy)."""

    if len(rr) < 10:
        return {
            'sd1': np.nan,
            'sd2': np.nan,
            'sd_ratio': np.nan,
            'sample_entropy': np.nan,
        }

    # Poincare plot analysis
    rr_n = rr[:-1]  # RR(n)
    rr_n1 = rr[1:]  # RR(n+1)

    # 17. SD1: perpendicular to line of identity (short-term variability)
    sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)

    # 18. SD2: along line of identity (long-term variability)
    sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)

    # 19. SD ratio
    sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan

    # 20. Sample entropy (approximation using binned distribution)
    sample_entropy = _compute_sample_entropy(rr, m=2, r_factor=0.2)

    return {
        'sd1': sd1,
        'sd2': sd2,
        'sd_ratio': sd_ratio,
        'sample_entropy': sample_entropy,
    }


def _compute_sample_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Compute sample entropy of RR intervals.

    Args:
        rr: RR intervals
        m: Embedding dimension
        r_factor: Tolerance factor (r = r_factor * std(rr))

    Returns:
        float: Sample entropy value
    """
    N = len(rr)
    if N < m + 2:
        return np.nan

    r = r_factor * np.std(rr)
    if r == 0:
        return np.nan

    def _count_matches(template_len):
        count = 0
        for i in range(N - template_len):
            for j in range(i + 1, N - template_len):
                if np.max(np.abs(rr[i:i + template_len] - rr[j:j + template_len])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan


def _get_nan_features() -> dict:
    """Return dictionary with all features set to NaN."""
    return {
        'mean_rr': np.nan,
        'sdnn': np.nan,
        'rmssd': np.nan,
        'pnn50': np.nan,
        'mean_hr': np.nan,
        'std_hr': np.nan,
        'cv_rr': np.nan,
        'range_rr': np.nan,
        'median_rr': np.nan,
        'iqr_rr': np.nan,
        'vlf_power': np.nan,
        'lf_power': np.nan,
        'hf_power': np.nan,
        'lf_hf_ratio': np.nan,
        'lf_nu': np.nan,
        'hf_nu': np.nan,
        'sd1': np.nan,
        'sd2': np.nan,
        'sd_ratio': np.nan,
        'sample_entropy': np.nan,
    }


# List of all 20 feature names in order
FEATURE_NAMES = [
    'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr',
    'std_hr', 'cv_rr', 'range_rr', 'median_rr', 'iqr_rr',
    'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_nu',
    'hf_nu', 'sd1', 'sd2', 'sd_ratio', 'sample_entropy'
]

FEATURE_DESCRIPTIONS = {
    'mean_rr': 'Mean RR interval (ms)',
    'sdnn': 'Std dev of RR intervals (ms)',
    'rmssd': 'Root mean square successive diff (ms)',
    'pnn50': 'Pct successive diff > 50ms (%)',
    'mean_hr': 'Mean heart rate (bpm)',
    'std_hr': 'Std dev heart rate (bpm)',
    'cv_rr': 'Coefficient of variation',
    'range_rr': 'Range of RR intervals (ms)',
    'median_rr': 'Median RR interval (ms)',
    'iqr_rr': 'Interquartile range RR (ms)',
    'vlf_power': 'Very low freq power (ms²)',
    'lf_power': 'Low freq power (ms²)',
    'hf_power': 'High freq power (ms²)',
    'lf_hf_ratio': 'LF/HF ratio',
    'lf_nu': 'LF normalized units (%)',
    'hf_nu': 'HF normalized units (%)',
    'sd1': 'Poincare SD1 (ms)',
    'sd2': 'Poincare SD2 (ms)',
    'sd_ratio': 'SD1/SD2 ratio',
    'sample_entropy': 'Sample entropy',
}

FEATURE_CATEGORIES = {
    'Time-domain': ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr',
                    'std_hr', 'cv_rr', 'range_rr', 'median_rr', 'iqr_rr'],
    'Frequency-domain': ['vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_nu', 'hf_nu'],
    'Non-linear': ['sd1', 'sd2', 'sd_ratio', 'sample_entropy'],
}
