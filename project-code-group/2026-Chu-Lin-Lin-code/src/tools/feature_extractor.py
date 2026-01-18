# SPDX-License-Identifier: Apache-2.0
"""HRV feature extraction: time and frequency domain metrics."""

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
from typing import Optional


def extract_time_domain_features(rr_intervals: np.ndarray) -> dict:
    """
    Extract time-domain HRV features.

    Args:
        rr_intervals: RR intervals in milliseconds

    Returns:
        dict: Time-domain features
    """
    if len(rr_intervals) < 2:
        return {
            "mean_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "mean_hr": np.nan,
            "std_hr": np.nan,
        }

    rr = np.array(rr_intervals)

    # Basic statistics
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)

    # Successive differences
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # pNN50: percentage of successive differences > 50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0

    # Heart rate statistics
    hr = 60000 / rr  # Convert RR (ms) to HR (bpm)
    mean_hr = np.mean(hr)
    std_hr = np.std(hr, ddof=1)

    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "nn50": nn50,
        "mean_hr": mean_hr,
        "std_hr": std_hr,
    }


def extract_frequency_domain_features(
    rr_intervals: np.ndarray,
    fs_resample: float = 4.0,
    vlf_band: tuple = (0.003, 0.04),
    lf_band: tuple = (0.04, 0.15),
    hf_band: tuple = (0.15, 0.4)
) -> dict:
    """
    Extract frequency-domain HRV features using Welch's method.

    Args:
        rr_intervals: RR intervals in milliseconds
        fs_resample: Resampling frequency in Hz (default: 4.0)
        vlf_band: VLF frequency band in Hz
        lf_band: LF frequency band in Hz
        hf_band: HF frequency band in Hz

    Returns:
        dict: Frequency-domain features
    """
    if len(rr_intervals) < 10:
        return {
            "vlf_power": np.nan,
            "lf_power": np.nan,
            "hf_power": np.nan,
            "lf_hf_ratio": np.nan,
            "total_power": np.nan,
            "lf_nu": np.nan,
            "hf_nu": np.nan,
        }

    rr = np.array(rr_intervals)

    # Create time vector (cumulative sum of RR intervals)
    time_rr = np.cumsum(rr) / 1000  # Convert to seconds
    time_rr = time_rr - time_rr[0]  # Start from 0

    # Resample to uniform sampling rate
    duration = time_rr[-1]
    time_uniform = np.arange(0, duration, 1 / fs_resample)

    if len(time_uniform) < 10:
        return {
            "vlf_power": np.nan,
            "lf_power": np.nan,
            "hf_power": np.nan,
            "lf_hf_ratio": np.nan,
            "total_power": np.nan,
            "lf_nu": np.nan,
            "hf_nu": np.nan,
        }

    # Interpolate RR intervals
    interp_func = interp1d(time_rr, rr, kind='cubic', fill_value='extrapolate')
    rr_resampled = interp_func(time_uniform)

    # Remove mean (detrend)
    rr_detrended = rr_resampled - np.mean(rr_resampled)

    # Compute PSD using Welch's method
    nperseg = min(256, len(rr_detrended))
    freqs, psd = welch(rr_detrended, fs=fs_resample, nperseg=nperseg)

    # Calculate band powers
    vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

    # Use np.trapezoid (NumPy 2.0+) or fallback to np.trapz for older versions
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    vlf_power = _trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
    lf_power = _trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
    hf_power = _trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0

    total_power = vlf_power + lf_power + hf_power

    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    # Normalized units (excluding VLF)
    lf_hf_sum = lf_power + hf_power
    lf_nu = (lf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
    hf_nu = (hf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan

    return {
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "total_power": total_power,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
    }


def extract_nonlinear_features(rr_intervals: np.ndarray) -> dict:
    """
    Extract non-linear HRV features (Poincare plot).

    Args:
        rr_intervals: RR intervals in milliseconds

    Returns:
        dict: Non-linear features (SD1, SD2)
    """
    if len(rr_intervals) < 3:
        return {
            "sd1": np.nan,
            "sd2": np.nan,
            "sd_ratio": np.nan,
        }

    rr = np.array(rr_intervals)

    # Poincare plot analysis
    rr_n = rr[:-1]  # RR(n)
    rr_n1 = rr[1:]  # RR(n+1)

    # SD1: perpendicular to line of identity
    sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)

    # SD2: along line of identity
    sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)

    # Ratio
    sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd_ratio": sd_ratio,
    }


def extract_hrv_features(
    rr_intervals: np.ndarray,
    include_nonlinear: bool = True
) -> dict:
    """
    Extract all HRV features from RR intervals.

    Args:
        rr_intervals: RR intervals in milliseconds
        include_nonlinear: Whether to include non-linear features

    Returns:
        dict: All HRV features
    """
    # Time domain
    time_features = extract_time_domain_features(rr_intervals)

    # Frequency domain
    freq_features = extract_frequency_domain_features(rr_intervals)

    # Combine
    features = {**time_features, **freq_features}

    # Non-linear (optional)
    if include_nonlinear:
        nonlinear_features = extract_nonlinear_features(rr_intervals)
        features.update(nonlinear_features)

    # Add metadata
    features["n_intervals"] = len(rr_intervals)
    features["recording_duration_sec"] = np.sum(rr_intervals) / 1000

    return features


def get_feature_vector(features: dict) -> np.ndarray:
    """
    Convert feature dictionary to numpy array for classification.

    Args:
        features: Dictionary of HRV features

    Returns:
        np.ndarray: Feature vector [sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio]
    """
    feature_names = ["sdnn", "rmssd", "pnn50", "lf_power", "hf_power", "lf_hf_ratio"]
    vector = np.array([features.get(name, np.nan) for name in feature_names])
    return vector
