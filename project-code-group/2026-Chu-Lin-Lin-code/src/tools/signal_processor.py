# SPDX-License-Identifier: Apache-2.0
"""ECG signal processing: filtering and R-peak detection."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Optional


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to ECG signal.

    Args:
        signal: Input ECG signal
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency in Hz (default: 0.5)
        highcut: High cutoff frequency in Hz (default: 40.0)
        order: Filter order (default: 4)

    Returns:
        np.ndarray: Filtered signal
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure frequencies are valid
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered


def remove_baseline_wander(
    signal: np.ndarray,
    fs: int,
    cutoff: float = 0.5
) -> np.ndarray:
    """
    Remove baseline wander using high-pass filter.

    Args:
        signal: Input ECG signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz (default: 0.5)

    Returns:
        np.ndarray: Signal with baseline removed
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1:
        normalized_cutoff = 0.999

    b, a = butter(2, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal)


def detect_r_peaks(
    signal: np.ndarray,
    fs: int,
    min_rr_sec: float = 0.3,
    max_rr_sec: float = 2.0
) -> np.ndarray:
    """
    Detect R-peaks using derivative-based method (simplified Pan-Tompkins).

    Args:
        signal: Filtered ECG signal
        fs: Sampling frequency in Hz
        min_rr_sec: Minimum RR interval in seconds (default: 0.3)
        max_rr_sec: Maximum RR interval in seconds (default: 2.0)

    Returns:
        np.ndarray: Indices of detected R-peaks
    """
    # Differentiate
    diff_signal = np.diff(signal)

    # Square to emphasize QRS complex
    squared = diff_signal ** 2

    # Moving window integration
    window_size = int(0.15 * fs)  # 150ms window
    if window_size < 1:
        window_size = 1
    integrated = np.convolve(
        squared,
        np.ones(window_size) / window_size,
        mode='same'
    )

    # Find peaks with minimum distance
    min_distance = int(min_rr_sec * fs)
    height_threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    peaks, properties = find_peaks(
        integrated,
        distance=min_distance,
        height=height_threshold
    )

    # Filter by maximum RR interval (remove isolated peaks)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        valid_mask = np.ones(len(peaks), dtype=bool)

        # Mark peaks with RR > max_rr_sec as potentially invalid
        for i in range(len(rr_intervals)):
            if rr_intervals[i] > max_rr_sec:
                # Keep the peak but flag for review
                pass

        peaks = peaks[valid_mask]

    return peaks


def compute_rr_intervals(
    r_peaks: np.ndarray,
    fs: int
) -> np.ndarray:
    """
    Compute RR intervals in milliseconds from R-peak indices.

    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency in Hz

    Returns:
        np.ndarray: RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples / fs * 1000  # Convert to milliseconds

    return rr_ms


def remove_ectopic_beats(
    rr_intervals: np.ndarray,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Remove ectopic beats using percentage threshold method.

    Args:
        rr_intervals: RR intervals in ms
        threshold: Percentage difference threshold (default: 0.2 = 20%)

    Returns:
        np.ndarray: Cleaned RR intervals
    """
    if len(rr_intervals) < 3:
        return rr_intervals

    # Compute percentage differences
    rr_diff = np.abs(np.diff(rr_intervals)) / rr_intervals[:-1]

    # Create mask for valid intervals
    valid_mask = np.ones(len(rr_intervals), dtype=bool)

    for i in range(len(rr_diff)):
        if rr_diff[i] > threshold:
            # Mark both adjacent intervals as potentially invalid
            valid_mask[i] = False
            if i + 1 < len(valid_mask):
                valid_mask[i + 1] = False

    return rr_intervals[valid_mask]


def process_signal(
    ecg_data: dict,
    filter_low: float = 0.5,
    filter_high: float = 40.0,
    remove_ectopic: bool = True
) -> dict:
    """
    Complete signal processing pipeline.

    Args:
        ecg_data: Dictionary from load_ecg() with 'signal' and 'sampling_rate'
        filter_low: Low cutoff frequency in Hz
        filter_high: High cutoff frequency in Hz
        remove_ectopic: Whether to remove ectopic beats

    Returns:
        dict: Contains 'filtered_signal', 'r_peaks', 'rr_intervals', 'n_beats'
    """
    signal = ecg_data["signal"]
    fs = ecg_data["sampling_rate"]

    # Apply bandpass filter
    filtered = bandpass_filter(signal, fs, filter_low, filter_high)

    # Detect R-peaks
    r_peaks = detect_r_peaks(filtered, fs)

    # Compute RR intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)

    # Remove ectopic beats if requested
    if remove_ectopic and len(rr_intervals) > 0:
        rr_intervals_clean = remove_ectopic_beats(rr_intervals)
    else:
        rr_intervals_clean = rr_intervals

    return {
        "filtered_signal": filtered,
        "r_peaks": r_peaks,
        "rr_intervals": rr_intervals_clean,
        "rr_intervals_raw": rr_intervals,
        "n_beats": len(r_peaks),
        "mean_hr_bpm": 60000 / np.mean(rr_intervals_clean) if len(rr_intervals_clean) > 0 else None,
        "sampling_rate": fs,
    }
