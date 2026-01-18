# SPDX-License-Identifier: Apache-2.0
"""Unit tests for helpers utility functions."""

import logging
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import (
    setup_logging,
    load_config,
    validate_ecg_data,
    normalize_signal,
    get_env_variable,
    ensure_directory,
)


class TestSetupLogging:
    """Tests for logging setup function."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "hrv_agent"

    def test_setup_logging_default_level(self):
        """Test default logging level is INFO."""
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test custom logging level."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test logging with file output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = Path(f.name) # Use Path object for convenience
        
        try:
            logger = setup_logging(log_file=str(log_path)) # Pass string path to setup_logging
            logger.info("Test message")

            # Verify file handler was added
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1
            # Ensure the file handle is closed before unlink
            for handler in file_handlers:
                handler.close()
                logger.removeHandler(handler)
        finally:
            log_path.unlink(missing_ok=True)


class TestLoadConfig:
    """Tests for configuration loading function."""

    def test_load_config_valid_yaml(self):
        """Test loading a valid YAML config file."""
        config_content = """
data:
  sampling_rate: 700
  subjects:
    - S2
    - S3
processing:
  filter_low: 0.5
  filter_high: 40.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert "data" in config
            assert "processing" in config
            assert config["data"]["sampling_rate"] == 700
            assert config["processing"]["filter_low"] == 0.5
        finally:
            Path(config_path).unlink()

    def test_load_config_nonexistent_file(self):
        """Test that loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")

    def test_load_config_empty_file(self):
        """Test loading an empty YAML file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config == {}
        finally:
            Path(config_path).unlink()

    def test_load_config_path_as_string(self):
        """Test loading config with path as string."""
        config_content = "key: value"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(str(config_path))
            assert config["key"] == "value"
        finally:
            Path(config_path).unlink()


class TestValidateEcgData:
    """Tests for ECG data validation function."""

    def test_validate_valid_data(self):
        """Test validation of valid ECG data."""
        np.random.seed(42)
        # 60 seconds at 500 Hz
        data = np.random.randn(30000) * 0.5

        result = validate_ecg_data(data, sampling_rate=500, min_duration=60.0)

        assert result["valid"] is True
        assert len(result["issues"]) == 0
        assert result["duration_sec"] == 60.0
        assert result["n_samples"] == 30000

    def test_validate_empty_data(self):
        """Test validation of empty data."""
        data = np.array([])

        result = validate_ecg_data(data)

        assert result["valid"] is False
        assert "Data is empty" in result["issues"]

    def test_validate_short_duration(self):
        """Test validation of data with insufficient duration."""
        # Only 10 seconds at 500 Hz
        data = np.random.randn(5000)

        result = validate_ecg_data(data, sampling_rate=500, min_duration=60.0)

        assert result["valid"] is False
        assert any("below minimum" in issue for issue in result["issues"])

    def test_validate_long_duration(self):
        """Test validation of data with excessive duration."""
        # 2 hours at 500 Hz
        data = np.random.randn(3600000)

        result = validate_ecg_data(data, sampling_rate=500, max_duration=3600.0)

        assert result["valid"] is False
        assert any("exceeds maximum" in issue for issue in result["issues"])

    def test_validate_data_with_nan(self):
        """Test validation of data containing NaN values."""
        data = np.random.randn(30000)
        data[100:110] = np.nan

        result = validate_ecg_data(data, sampling_rate=500)

        assert result["valid"] is False
        assert any("NaN values" in issue for issue in result["issues"])

    def test_validate_data_with_inf(self):
        """Test validation of data containing infinite values."""
        data = np.random.randn(30000)
        data[100] = np.inf
        data[200] = -np.inf

        result = validate_ecg_data(data, sampling_rate=500)

        assert result["valid"] is False
        assert any("infinite values" in issue for issue in result["issues"])

    def test_validate_flat_signal(self):
        """Test validation of flat (constant) signal."""
        data = np.ones(30000)

        result = validate_ecg_data(data, sampling_rate=500)

        assert result["valid"] is False
        assert any("flat" in issue for issue in result["issues"])

    def test_validate_extreme_values(self):
        """Test validation of signal with extreme values."""
        data = np.random.randn(30000) * 100  # Very large values

        result = validate_ecg_data(data, sampling_rate=500)

        assert result["valid"] is False
        assert any("extreme values" in issue for issue in result["issues"])

    def test_validate_saturated_signal(self):
        """Test validation of saturated signal."""
        # Only a few unique values
        data = np.array([1, 2, 3, 1, 2, 3] * 5000)

        result = validate_ecg_data(data, sampling_rate=500)

        assert result["valid"] is False
        assert any("saturated" in issue for issue in result["issues"])

    def test_validate_returns_statistics(self):
        """Test that validation returns signal statistics."""
        np.random.seed(42)
        data = np.random.randn(30000) * 0.5

        result = validate_ecg_data(data, sampling_rate=500)

        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert isinstance(result["mean"], float)


class TestNormalizeSignal:
    """Tests for signal normalization function."""

    def test_normalize_standard_signal(self):
        """Test normalization produces zero mean and unit variance."""
        np.random.seed(42)
        data = np.random.randn(1000) * 10 + 50  # Mean ~50, std ~10

        normalized = normalize_signal(data)

        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized) - 1.0) < 1e-10

    def test_normalize_flat_signal(self):
        """Test normalization of flat signal."""
        data = np.ones(1000) * 5

        normalized = normalize_signal(data)

        # Should subtract mean but not divide by zero std
        assert np.abs(np.mean(normalized)) < 1e-10

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves array shape."""
        data = np.random.randn(500)

        normalized = normalize_signal(data)

        assert normalized.shape == data.shape

    def test_normalize_already_normalized(self):
        """Test normalizing already normalized data."""
        np.random.seed(42)
        data = np.random.randn(1000)  # Already mean=0, std~1

        normalized = normalize_signal(data)

        np.testing.assert_array_almost_equal(
            normalized,
            (data - np.mean(data)) / np.std(data),
            decimal=10
        )


class TestGetEnvVariable:
    """Tests for environment variable getter."""

    def test_get_existing_env_variable(self):
        """Test getting an existing environment variable."""
        os.environ["TEST_HRV_VAR"] = "test_value"

        try:
            result = get_env_variable("TEST_HRV_VAR")
            assert result == "test_value"
        finally:
            del os.environ["TEST_HRV_VAR"]

    def test_get_nonexistent_env_variable(self):
        """Test getting a nonexistent environment variable."""
        result = get_env_variable("NONEXISTENT_HRV_VAR_12345")
        assert result is None

    def test_get_env_variable_with_default(self):
        """Test getting nonexistent variable with default value."""
        result = get_env_variable("NONEXISTENT_VAR", default="default_value")
        assert result == "default_value"

    def test_get_env_variable_overrides_default(self):
        """Test that existing variable overrides default."""
        os.environ["TEST_VAR_OVERRIDE"] = "actual_value"

        try:
            result = get_env_variable("TEST_VAR_OVERRIDE", default="default")
            assert result == "actual_value"
        finally:
            del os.environ["TEST_VAR_OVERRIDE"]


class TestEnsureDirectory:
    """Tests for directory creation function."""

    def test_ensure_new_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"

            result = ensure_directory(new_dir)

            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_existing_directory(self):
        """Test with already existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_directory(tmpdir)

            assert result == Path(tmpdir)
            assert Path(tmpdir).exists()

    def test_ensure_nested_directory(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "level1" / "level2" / "level3"

            result = ensure_directory(nested)

            assert result == nested
            assert nested.exists()

    def test_ensure_directory_with_string_path(self):
        """Test with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = str(Path(tmpdir) / "string_path_dir")

            result = ensure_directory(new_dir)

            assert isinstance(result, Path)
            assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
