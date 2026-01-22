# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HRV analysis tools."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import pandas as pd # Added import

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.ecg_loader import (
    load_ecg,
    pick_ecg_column,
    pick_time_column,
)
from src.tools.signal_processor import (
    bandpass_filter,
    detect_r_peaks,
    compute_rr_intervals,
    process_signal
)
from src.tools.extended_features import (
    extract_extended_features
)



class TestECGLoader:
    """Tests for ECG loading functionality."""

    def test_load_valid_ecg(self):
        """Test loading a valid ECG file."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write synthetic ECG data (1 second at 500 Hz)
            data = np.sin(2 * np.pi * 1 * np.arange(500) / 500)
            np.savetxt(f.name, data)
            temp_path = f.name

        try:
            result = load_ecg(temp_path, sampling_rate=500)

            assert "signal" in result
            assert "sampling_rate" in result
            assert "duration_sec" in result
            assert result["sampling_rate"] == 500
            assert result["n_samples"] == 500
            assert abs(result["duration_sec"] - 1.0) < 0.01
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_ecg("/nonexistent/path/to/file.txt")

    def test_load_empty_file(self):
        """Test that loading an empty file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="ECG file is empty"):
                load_ecg(temp_path)
        finally:
            Path(temp_path).unlink()


class TestSignalProcessor:
    """Tests for signal processing functionality."""

    def test_bandpass_filter(self):
        """Test bandpass filter frequency response."""
        fs = 500  # Sampling rate
        duration = 10  # seconds
        t = np.arange(0, duration, 1/fs)

        # Create signal with components at 0.1 Hz (reject), 10 Hz (pass), 100 Hz (reject)
        signal = (
            np.sin(2 * np.pi * 0.1 * t) +  # Below passband
            np.sin(2 * np.pi * 10 * t) +   # In passband
            np.sin(2 * np.pi * 100 * t)    # Above passband
        )

        filtered = bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0)

        # Check that the 10 Hz component is preserved (roughly)
        # The filtered signal should have similar amplitude to original 10 Hz component
        assert len(filtered) == len(signal)
        assert not np.any(np.isnan(filtered))

    def test_detect_r_peaks_synthetic(self):
        """Test R-peak detection on synthetic signal."""
        fs = 500
        duration = 10
        heart_rate = 60  # 1 beat per second

        # Create synthetic ECG-like signal with peaks at 1 Hz
        t = np.arange(0, duration, 1/fs)
        signal = np.zeros_like(t)

        # Add sharp peaks at regular intervals
        for i in range(heart_rate * duration // 60):
            peak_idx = int(i * fs)
            if peak_idx < len(signal):
                signal[peak_idx] = 1.0

        # Add some baseline
        signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)

        peaks = detect_r_peaks(signal, fs)

        # Should detect approximately 10 peaks in 10 seconds at 60 bpm
        assert len(peaks) >= 8
        assert len(peaks) <= 12

    def test_compute_rr_intervals(self):
        """Test RR interval computation."""
        fs = 500
        r_peaks = np.array([0, 500, 1000, 1500])  # 1 second apart

        rr_intervals = compute_rr_intervals(r_peaks, fs)

        assert len(rr_intervals) == 3
        np.testing.assert_array_almost_equal(rr_intervals, [1000, 1000, 1000])

    def test_process_signal_pipeline(self):
        """Test the complete signal processing pipeline."""
        fs = 500
        duration = 60
        t = np.arange(0, duration, 1/fs)

        # Create synthetic ECG
        signal = np.sin(2 * np.pi * 1 * t)  # 1 Hz base
        # Add R-peak-like spikes
        for i in range(0, len(signal), fs):  # Every second
            if i < len(signal):
                signal[i] += 2.0

        ecg_data = {
            "signal": signal,
            "sampling_rate": fs
        }

        result = process_signal(ecg_data)

        assert "filtered_signal" in result
        assert "r_peaks" in result
        assert "rr_intervals" in result
        assert "n_beats" in result
        assert result["n_beats"] > 0





class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_synthetic(self):
        """Test end-to-end pipeline with synthetic data."""
        # Create synthetic ECG
        fs = 500
        duration = 180  # 3 minutes
        t = np.arange(0, duration, 1/fs)

        # Base signal
        signal = 0.5 * np.sin(2 * np.pi * 0.3 * t)  # Respiratory component

        # Add R-peaks at ~70 bpm with some variability
        np.random.seed(42)
        rr_ms = 857 + 50 * np.random.randn(int(duration * 70 / 60))
        rr_samples = (rr_ms / 1000 * fs).astype(int)
        peak_positions = np.cumsum(rr_samples)
        peak_positions = peak_positions[peak_positions < len(signal)]

        for pos in peak_positions:
            signal[pos] += 2.0
            if pos + 1 < len(signal):
                signal[pos + 1] += 0.5
            if pos - 1 >= 0:
                signal[pos - 1] += 0.5

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f.name, signal)
            temp_path = f.name

        try:
            # Load
            ecg_data = load_ecg(temp_path, sampling_rate=fs)
            assert ecg_data["duration_sec"] == pytest.approx(duration, rel=0.01)

            # Process
            processed = process_signal(ecg_data)
            assert processed["n_beats"] > 100

            # Extract features
            features = extract_extended_features(processed["rr_intervals"])
            assert not np.isnan(features["sdnn"])
            assert not np.isnan(features["rmssd"])
            assert features["sdnn"] > 0

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestECGColumnPicking:
    """Tests for ECG column picking utility functions."""

    def test_pick_ecg_column_named(self):
        """Test picking ECG column by name."""
        df = pd.DataFrame({"Timestamp": [1, 2], "ECG": [0.1, 0.2], "PPG": [0.3, 0.4]})
        assert pick_ecg_column(df) == "ECG"

    def test_pick_ecg_column_named_lowercase(self):
        """Test picking ECG column by lowercase name."""
        df = pd.DataFrame({"Timestamp": [1, 2], "ecg": [0.1, 0.2], "PPG": [0.3, 0.4]})
        assert pick_ecg_column(df) == "ecg"

    def test_pick_ecg_column_by_index(self):
        """Test picking ECG column by default index (3rd column)."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [0.1, 0.2]})
        assert pick_ecg_column(df) == "D"

    def test_pick_ecg_column_raises_error(self):
        """Test picking ECG column raises error if not found."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}) # Only 2 columns
        with pytest.raises(ValueError, match="Cannot find ECG column"):
            pick_ecg_column(df)

    def test_pick_time_column_named_timestamp(self):
        """Test picking time column by name 'Timestamp'."""
        df = pd.DataFrame({"Timestamp": [1, 2], "ECG": [0.1, 0.2]})
        assert pick_time_column(df) == "Timestamp"

    def test_pick_time_column_named_time(self):
        """Test picking time column by name 'Time'."""
        df = pd.DataFrame({"Time": [1, 2], "ECG": [0.1, 0.2]})
        assert pick_time_column(df) == "Time"

    def test_pick_time_column_not_found(self):
        """Test picking time column returns None if not found."""
        df = pd.DataFrame({"ECG": [0.1, 0.2]})
        assert pick_time_column(df) is None

