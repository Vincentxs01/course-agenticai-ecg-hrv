# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the HRV Analysis Orchestrator's dataset processing."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import HRVAnalysisOrchestrator
from src.tools.ecg_loader import read_ecg_csv_column # New import


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("src.orchestrator.process_signal") as mock_process_signal, patch(
        "src.orchestrator.extract_extended_features"
    ) as mock_extract_extended_features, patch(
        "src.orchestrator.generate_report"
    ) as mock_generate_report, patch(
        "pandas.read_csv"
    ) as mock_read_csv:
        # Configure mocks to return predictable data
        mock_process_signal.return_value = {
            "rr_intervals": np.array([1000, 950, 1050]),
            "r_peaks": np.array([10, 20, 30]),
            "filtered_signal": np.array([1, 2, 3]),
            "n_beats": 3,
            "mean_hr_bpm": 60.0,
        }
        mock_extract_extended_features.return_value = {
            "sdnn": 50.0,
            "rmssd": 40.0,
            "lf_hf_ratio": 1.5,
        }
        mock_read_csv.return_value = pd.DataFrame(
            {"ECG": np.random.rand(5000)} # Sample ECG data
        )

        yield


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmp_data_dir, tempfile.TemporaryDirectory() as tmp_output_dir:
        yield Path(tmp_data_dir), Path(tmp_output_dir)


@pytest.fixture
def sample_config(temp_dirs):
    data_dir, output_dir = temp_dirs

    # Create dummy data files
    (data_dir / "person1" / "Rest").mkdir(parents=True, exist_ok=True)
    (data_dir / "person1" / "Active").mkdir(parents=True, exist_ok=True)
    (data_dir / "person2" / "Rest").mkdir(parents=True, exist_ok=True)
    (data_dir / "person2" / "Active").mkdir(parents=True, exist_ok=True)

    # Dummy CSV content
    dummy_csv_content = "0,1,2,0.5\n1,2,3,0.6\n" # Minimal for read_csv

    # Create 4 CSVs per person * state for easy testing of config glob
    (data_dir / "person1" / "Rest" / "file1.csv").write_text(dummy_csv_content)
    (data_dir / "person1" / "Rest" / "file2.csv").write_text(dummy_csv_content)
    (data_dir / "person1" / "Active" / "file3.csv").write_text(dummy_csv_content)
    (data_dir / "person1" / "Active" / "file4.csv").write_text(dummy_csv_content)

    (data_dir / "person2" / "Rest" / "file5.csv").write_text(dummy_csv_content)
    (data_dir / "person2" / "Rest" / "file6.csv").write_text(dummy_csv_content)
    (data_dir / "person2" / "Active" / "file7.csv").write_text(dummy_csv_content)
    (data_dir / "person2" / "Active" / "file8.csv").write_text(dummy_csv_content)

    config = {
        "dataset": {
            "name": "test_dataset",
            "data_dir": str(data_dir),
            "persons": [
                {"id": "person1", "conditions": {"Rest": {"glob": "person1/Rest/*.csv"}, "Active": {"glob": "person1/Active/*.csv"}}},
                {"id": "person2", "conditions": {"Rest": {"glob": "person2/Rest/*.csv"}, "Active": {"glob": "person2/Active/*.csv"}}},
            ],
        },
        "signal": {
            "sampling_rate": 50,
            "bandpass_low": 0.5,
            "bandpass_high": 20.0,
        },
        "features": {
            "window_size_sec": 60,
            "overlap": 0.5,
        },
        "r_peak": {
            "min_rr_sec": 0.3,
            "max_rr_sec": 2.0,
        },
        "baseline": {
            "k_rest": 2.5,
            "k_active": 2.0,
        },
        "output": {
            "dir": str(output_dir),
        },
    }
    return config


class TestHRVAnalysisOrchestrator:
    """Tests for the HRVAnalysisOrchestrator class."""

    def test_init(self):
        """Test default initialization."""
        orchestrator = HRVAnalysisOrchestrator()
        assert orchestrator.state == {}
        assert orchestrator.execution_log == []

    def test_scan_dataset_from_config(self, temp_dirs):
        data_dir, _ = temp_dirs

        (data_dir / "personA" / "Rest").mkdir(parents=True, exist_ok=True)
        (data_dir / "personA" / "Active").mkdir(parents=True, exist_ok=True)
        (data_dir / "personA" / "Rest" / "file1.csv").touch()
        (data_dir / "personA" / "Active" / "file2.csv").touch()

        persons_cfg = [
            {"id": "personA", "conditions": {"Rest": {"glob": "personA/Rest/*.csv"}, "Active": {"glob": "personA/Active/*.csv"}}},
        ]

        orchestrator = HRVAnalysisOrchestrator()
        records = orchestrator._scan_dataset_from_config(data_dir, persons_cfg)

        assert len(records) == 2
        assert any(r["person"] == "personA" and r["state"] == "Rest" for r in records)
        assert any(r["person"] == "personA" and r["state"] == "Active" for r in records)

    
    def test_window_slices(self):
        orchestrator = HRVAnalysisOrchestrator()
        signal_length = 100
        win = 20
        stride = 10
        windows = list(orchestrator._window_slices(signal_length, win, stride))
        assert len(windows) == 9 # (100 - 20) / 10 + 1
        assert windows[0] == (0, 20)
        assert windows[-1] == (80, 100)

    @patch("src.orchestrator.process_signal")
    @patch("src.orchestrator.extract_extended_features")
    def test_window_metrics(self, mock_extract_extended_features, mock_process_signal):
        orchestrator = HRVAnalysisOrchestrator()
        ecg_seg = np.array([1, 2, 3, 4, 5])
        fs = 50
        filter_low = 0.5
        filter_high = 20.0

        mock_process_signal.return_value = {"rr_intervals": np.array([1000, 950]), "n_beats": 2}
        mock_extract_extended_features.return_value = {"sdnn": 50.0, "rmssd": 40.0, "lf_hf_ratio": 1.5}

        metrics = orchestrator._window_metrics(ecg_seg, fs, filter_low, filter_high)

        assert metrics["sdnn"] == 50.0
        assert metrics["rmssd"] == 40.0
        assert metrics["lf_hf_ratio"] == 1.5
        mock_process_signal.assert_called_once()
        mock_extract_extended_features.assert_called_once()

    def test_fit_baseline(self):
        orchestrator = HRVAnalysisOrchestrator()
        win_rows = [
            {"rr": np.array([1000]), "sdnn": 50.0, "rmssd": 40.0},
            {"rr": np.array([1050]), "sdnn": 55.0, "rmssd": 45.0},
        ]
        baseline = orchestrator._fit_baseline(win_rows)
        assert baseline["rr_mean"]["mean"] == pytest.approx(1.025)
        assert baseline["sdnn"]["mean"] == pytest.approx(52.5)
        assert baseline["rmssd"]["mean"] == pytest.approx(42.5)

    def test_in_range(self):
        orchestrator = HRVAnalysisOrchestrator()
        # Test within range
        assert orchestrator._in_range(5.0, 5.0, 1.0, 1.0) is True
        assert orchestrator._in_range(5.5, 5.0, 1.0, 1.0) is True # mu + k*sd
        assert orchestrator._in_range(4.5, 5.0, 1.0, 1.0) is True # mu - k*sd
        # Test outside range
        assert orchestrator._in_range(6.1, 5.0, 1.0, 1.0) is False
        assert orchestrator._in_range(3.9, 5.0, 1.0, 1.0) is False
        # Test with zero std (should fallback to 10% tolerance)
        assert orchestrator._in_range(5.0, 5.0, 0.0, 1.0) is True
        assert orchestrator._in_range(5.05, 5.0, 0.0, 1.0) is True # 1% diff
        assert orchestrator._in_range(5.6, 5.0, 0.0, 1.0) is False # >10% diff

    def test_window_pass(self):
        orchestrator = HRVAnalysisOrchestrator()
        base = {
            "rr_mean": {"mean": 1.0, "std": 0.05}, # In seconds
            "sdnn": {"mean": 50.0, "std": 5.0},
            "rmssd": {"mean": 40.0, "std": 4.0},
        }
        rr_min, rr_max = 0.5, 1.5 # seconds
        k = 1.0

        # Pass case
        m_pass = {"rr": np.array([1000.0]), "sdnn": 50.0, "rmssd": 40.0}
        assert orchestrator._window_pass(m_pass, base, k, rr_min, rr_max) is True

        # Fail physiological RR
        m_fail_rr = {"rr": np.array([0.2]), "sdnn": 50.0, "rmssd": 40.0}
        assert orchestrator._window_pass(m_fail_rr, base, k, rr_min, rr_max) is False

        # Fail baseline constraint
        m_fail_sdnn = {"rr": np.array([1000.0]), "sdnn": 60.0, "rmssd": 40.0}
        assert orchestrator._window_pass(m_fail_sdnn, base, k, rr_min, rr_max) is False

    @patch("pandas.DataFrame.to_csv")
    @patch("pathlib.Path.write_text")
    @patch("src.orchestrator.HRVAnalysisOrchestrator._window_pass")
    @patch("src.orchestrator.HRVAnalysisOrchestrator._fit_baseline")
    @patch("src.orchestrator.HRVAnalysisOrchestrator._window_metrics")
    @patch("src.tools.ecg_loader.read_ecg_csv_column") # Updated patch target
    @patch("src.orchestrator.HRVAnalysisOrchestrator._scan_dataset_from_config")
    def test_run_dataset(self, mock_scan_dataset_from_config, mock_read_ecg_csv_column, mock_window_metrics, mock_fit_baseline, mock_window_pass, mock_write_text, mock_to_csv, sample_config, temp_dirs): # Updated mock argument name
        orchestrator = HRVAnalysisOrchestrator()
        data_dir, output_dir = temp_dirs

        # Mock records
        mock_scan_records = [
            {"person": "person1", "state": "Rest", "path": data_dir / "person1" / "Rest" / "file1.csv"},
            {"person": "person1", "state": "Active", "path": data_dir / "person1" / "Active" / "file3.csv"},
        ]
        mock_scan_dataset_from_config.return_value = mock_scan_records

        # Mock read_ecg_csv_column to return a signal with enough length for windowing
        mock_read_ecg_csv_column.return_value = np.zeros(50 * 60 * 2) # 2 minutes of signal

        # Mock window metrics
        mock_window_metrics.return_value = {"rr": np.array([1000]), "sdnn": 50.0, "rmssd": 40.0}

        # Mock baseline fitting
        mock_fit_baseline.return_value = {
            "rr_mean": {"mean": 1.0, "std": 0.05}, # In seconds
            "sdnn": {"mean": 50.0, "std": 5.0},
            "rmssd": {"mean": 40.0, "std": 4.0},
        }

        # Mock window pass/fail
        mock_window_pass.return_value = True

        result = orchestrator.run_dataset(sample_config)

        assert result["status"] == "success"
        assert Path(result["outdir"]) == output_dir
        assert result["n_files"] == 2

        # Verify baselines.json is written
        expected_baselines_json_data = {
            "person1": {"Rest": mock_fit_baseline.return_value, "Active": mock_fit_baseline.return_value},
            "person2": {"Rest": mock_fit_baseline.return_value, "Active": mock_fit_baseline.return_value}
        }
        
        found_baselines_call = False
        for i, (call_args, call_kwargs) in enumerate(mock_write_text.call_args_list):
            print(f"Call {i} to mock_write_text: args={call_args}, kwargs={call_kwargs}")
            if call_args and len(call_args) > 0:
                written_content = call_args[0]
                try:
                    actual_json_content = json.loads(written_content)
                    print(f"  Parsed JSON Content: {actual_json_content}")
                    print(f"  Expected JSON Content: {expected_baselines_json_data}")
                    if actual_json_content == expected_baselines_json_data:
                        found_baselines_call = True
                        break
                except json.JSONDecodeError as e:
                    print(f"  Not JSON content (error: {e}): {written_content[:100]}...") # Print first 100 chars
                    continue # Not a JSON call

        assert found_baselines_call, "Expected baselines.json content not found or did not match in mock_write_text calls"

        # Verify pass_rates.csv is written
        mock_to_csv.assert_called_once()

        # Verify per-file JSON is written
        assert mock_write_text.call_count == 1 + len(mock_scan_records) # baselines.json (1) + per-file.json (2) = 3 calls
    



if __name__ == "__main__":
    pytest.main([__file__, "-v"])