# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests to ensure full parameter coverage."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.ecg_loader import (
    load_ecg,
)
from src.tools.signal_processor import (
    bandpass_filter,
    remove_baseline_wander,
    detect_r_peaks,
    compute_rr_intervals,
    remove_ectopic_beats,
    process_signal,
)







# =============================================================================
# SIGNAL PROCESSOR TESTS - Additional Coverage
# =============================================================================

class TestSignalProcessorParameters:
    """Tests for signal processor parameter variations."""

    def test_bandpass_filter_custom_cutoffs(self):
        """Test bandpass filter with custom cutoff frequencies."""
        fs = 500
        t = np.arange(0, 10, 1/fs)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

        # Default cutoffs (0.5-40 Hz)
        filtered_default = bandpass_filter(signal, fs)
        assert len(filtered_default) == len(signal)

        # Custom cutoffs (1-30 Hz)
        filtered_custom = bandpass_filter(signal, fs, lowcut=1.0, highcut=30.0)
        assert len(filtered_custom) == len(signal)

        # Different order
        filtered_order2 = bandpass_filter(signal, fs, order=2)
        filtered_order6 = bandpass_filter(signal, fs, order=6)
        assert len(filtered_order2) == len(signal)
        assert len(filtered_order6) == len(signal)

    def test_remove_baseline_wander(self):
        """Test baseline wander removal."""
        fs = 500
        t = np.arange(0, 10, 1/fs)

        # Signal with DC offset and low-frequency drift
        signal = 1.0 + 0.5 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 10 * t)

        # Remove baseline wander
        cleaned = remove_baseline_wander(signal, fs, cutoff=0.5)

        assert len(cleaned) == len(signal)
        # DC component should be removed
        assert abs(np.mean(cleaned)) < abs(np.mean(signal))

    def test_remove_baseline_wander_different_cutoffs(self):
        """Test baseline removal with different cutoff frequencies."""
        fs = 500
        t = np.arange(0, 10, 1/fs)
        signal = 1.0 + np.sin(2 * np.pi * 10 * t)

        # Different cutoffs
        cleaned_05 = remove_baseline_wander(signal, fs, cutoff=0.5)
        cleaned_1 = remove_baseline_wander(signal, fs, cutoff=1.0)

        assert len(cleaned_05) == len(signal)
        assert len(cleaned_1) == len(signal)

    def test_detect_r_peaks_with_parameters(self):
        """Test R-peak detection with different parameters."""
        fs = 500
        duration = 10
        t = np.arange(0, duration, 1/fs)

        # Create synthetic ECG-like signal
        signal = np.zeros_like(t)
        for i in range(10):
            peak_idx = int(i * fs)  # 1 peak per second
            if peak_idx < len(signal):
                signal[peak_idx] = 1.0

        # Default parameters
        peaks_default = detect_r_peaks(signal, fs)

        # Custom min_rr_sec
        peaks_minrr = detect_r_peaks(signal, fs, min_rr_sec=0.5)

        # Custom max_rr_sec
        peaks_maxrr = detect_r_peaks(signal, fs, max_rr_sec=1.5)

        assert len(peaks_default) >= 5
        assert len(peaks_minrr) >= 5
        assert len(peaks_maxrr) >= 5

    def test_compute_rr_intervals_edge_cases(self):
        """Test RR interval computation edge cases."""
        # Empty peaks
        result_empty = compute_rr_intervals(np.array([]), fs=500)
        assert len(result_empty) == 0

        # Single peak
        result_single = compute_rr_intervals(np.array([100]), fs=500)
        assert len(result_single) == 0

        # Normal case
        peaks = np.array([0, 500, 1000, 1500])
        result = compute_rr_intervals(peaks, fs=500)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [1000, 1000, 1000])

    def test_remove_ectopic_beats(self):
        """Test ectopic beat removal."""
        # Normal RR intervals with one ectopic
        rr = np.array([1000, 1000, 500, 1500, 1000, 1000])  # Ectopic at index 2-3

        cleaned = remove_ectopic_beats(rr, threshold=0.2)

        # Ectopic intervals should be removed
        assert len(cleaned) < len(rr)

    def test_remove_ectopic_beats_different_thresholds(self):
        """Test ectopic removal with different thresholds."""
        rr = np.array([1000, 1050, 1000, 950, 1000])  # Mild variation

        # Strict threshold (10%)
        cleaned_strict = remove_ectopic_beats(rr, threshold=0.1)

        # Lenient threshold (30%)
        cleaned_lenient = remove_ectopic_beats(rr, threshold=0.3)

        # Lenient threshold should keep more intervals
        assert len(cleaned_lenient) >= len(cleaned_strict)

    def test_remove_ectopic_beats_short_array(self):
        """Test ectopic removal with short arrays."""
        # Less than 3 intervals
        rr_short = np.array([1000, 1000])
        result = remove_ectopic_beats(rr_short)
        np.testing.assert_array_equal(result, rr_short)

    def test_process_signal_full_parameters(self):
        """Test process_signal with all parameter combinations."""
        fs = 500
        duration = 60
        t = np.arange(0, duration, 1/fs)

        # Create synthetic ECG
        signal = np.sin(2 * np.pi * 1 * t)
        for i in range(0, len(signal), fs):
            if i < len(signal):
                signal[i] += 2.0

        ecg_data = {"signal": signal, "sampling_rate": fs}

        # Test with different filter parameters
        result1 = process_signal(ecg_data, filter_low=0.5, filter_high=40.0)
        result2 = process_signal(ecg_data, filter_low=1.0, filter_high=30.0)

        assert result1["n_beats"] > 0
        assert result2["n_beats"] > 0

        # Test with remove_ectopic=False
        result3 = process_signal(ecg_data, remove_ectopic=False)
        assert "rr_intervals_raw" in result3





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
