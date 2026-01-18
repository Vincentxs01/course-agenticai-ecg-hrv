# SPDX-License-Identifier: Apache-2.0
"""Unit tests for extended HRV features module."""

import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.extended_features import (
    extract_extended_features,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_CATEGORIES,
)


class TestFeatureConstants:
    """Tests for feature constants."""

    def test_feature_names_count(self):
        """Test that there are exactly 20 feature names."""
        assert len(FEATURE_NAMES) == 20

    def test_feature_names_unique(self):
        """Test that all feature names are unique."""
        assert len(set(FEATURE_NAMES)) == 20

    def test_feature_descriptions_complete(self):
        """Test that all features have descriptions."""
        for name in FEATURE_NAMES:
            assert name in FEATURE_DESCRIPTIONS
            assert len(FEATURE_DESCRIPTIONS[name]) > 10

    def test_feature_categories_complete(self):
        """Test that all features are assigned to a category."""
        # FEATURE_CATEGORIES is {category_name: [list of features]}
        all_categorized_features = []
        for category, features in FEATURE_CATEGORIES.items():
            all_categorized_features.extend(features)

        for name in FEATURE_NAMES:
            assert name in all_categorized_features, f"Feature {name} not in any category"

    def test_feature_categories_valid(self):
        """Test that categories are valid."""
        valid_categories = {"Time-domain", "Frequency-domain", "Non-linear"}
        for category_name in FEATURE_CATEGORIES.keys():
            assert category_name in valid_categories, f"Invalid category: {category_name}"


class TestTimeDomainFeatures:
    """Tests for time-domain features."""

    def test_regular_rr_intervals(self):
        """Test features with perfectly regular RR intervals."""
        rr_intervals = np.array([1000.0] * 100)  # 60 bpm, no variation

        features = extract_extended_features(rr_intervals)

        assert features["mean_rr"] == 1000.0
        assert features["sdnn"] == 0.0
        assert features["rmssd"] == 0.0
        assert features["pnn50"] == 0.0
        assert features["mean_hr"] == 60.0
        assert features["std_hr"] == 0.0
        assert features["cv_rr"] == 0.0

    def test_variable_rr_intervals(self):
        """Test features with variable RR intervals."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert features["sdnn"] > 0
        assert features["rmssd"] > 0
        assert features["cv_rr"] > 0
        assert features["range_rr"] > 0
        assert features["iqr_rr"] > 0

    def test_mean_rr_calculation(self):
        """Test mean RR interval calculation."""
        # Need at least 10 intervals for the function to work
        rr_intervals = np.array([900, 950, 1000, 1050, 1100, 900, 950, 1000, 1050, 1100])

        features = extract_extended_features(rr_intervals)

        assert features["mean_rr"] == 1000.0

    def test_median_rr_calculation(self):
        """Test median RR interval calculation."""
        # Need at least 10 intervals - use values where median is clearly 1000
        rr_intervals = np.array([800, 850, 900, 950, 1000, 1000, 1050, 1100, 1200, 1500])

        features = extract_extended_features(rr_intervals)

        assert features["median_rr"] == 1000.0

    def test_mean_hr_calculation(self):
        """Test mean heart rate calculation."""
        rr_intervals = np.array([1000] * 10)  # 1000 ms = 60 bpm

        features = extract_extended_features(rr_intervals)

        assert features["mean_hr"] == pytest.approx(60.0, rel=0.01)

    def test_pnn50_with_large_differences(self):
        """Test pNN50 with large successive differences."""
        # Alternating pattern ensures all differences > 50ms (need 10+ intervals)
        rr_intervals = np.array([900, 1000, 900, 1000, 900, 1000, 900, 1000, 900, 1000])

        features = extract_extended_features(rr_intervals)

        assert features["pnn50"] == 100.0  # All differences are 100ms > 50ms


class TestFrequencyDomainFeatures:
    """Tests for frequency-domain features."""

    def test_frequency_features_present(self):
        """Test that all frequency features are extracted."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(200)

        features = extract_extended_features(rr_intervals)

        assert "vlf_power" in features
        assert "lf_power" in features
        assert "hf_power" in features
        assert "lf_hf_ratio" in features
        assert "lf_nu" in features
        assert "hf_nu" in features

    def test_frequency_powers_non_negative(self):
        """Test that frequency powers are non-negative."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(200)

        features = extract_extended_features(rr_intervals)

        assert features["vlf_power"] >= 0
        assert features["lf_power"] >= 0
        assert features["hf_power"] >= 0

    def test_normalized_units_sum(self):
        """Test that LF_nu + HF_nu approximately equals 100."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(200)

        features = extract_extended_features(rr_intervals)

        # LF_nu and HF_nu should sum to approximately 100
        total = features["lf_nu"] + features["hf_nu"]
        assert total == pytest.approx(100.0, rel=0.1)

    def test_lf_hf_ratio_positive(self):
        """Test that LF/HF ratio is positive."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(200)

        features = extract_extended_features(rr_intervals)

        assert features["lf_hf_ratio"] > 0


class TestNonLinearFeatures:
    """Tests for non-linear features."""

    def test_poincare_features_present(self):
        """Test that Poincaré features are extracted."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert "sd1" in features
        assert "sd2" in features
        assert "sd_ratio" in features

    def test_sd1_sd2_relationship(self):
        """Test SD1/SD2 relationship."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        # SD ratio should equal SD1/SD2
        expected_ratio = features["sd1"] / features["sd2"] if features["sd2"] > 0 else np.nan
        if not np.isnan(expected_ratio):
            assert features["sd_ratio"] == pytest.approx(expected_ratio, rel=0.01)

    def test_sample_entropy_present(self):
        """Test that sample entropy is extracted."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert "sample_entropy" in features
        assert not np.isnan(features["sample_entropy"])

    def test_sample_entropy_regular_signal(self):
        """Test sample entropy for regular signal is low."""
        # Perfectly regular signal
        rr_intervals = np.array([1000.0] * 100)

        features = extract_extended_features(rr_intervals)

        # Sample entropy of constant signal should be very low or 0
        # (though numerical issues may give small non-zero value)
        assert features["sample_entropy"] < 0.5 or np.isnan(features["sample_entropy"])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_rr_intervals(self):
        """Test with very short RR interval array (<10 returns NaN)."""
        rr_intervals = np.array([1000, 1010])

        features = extract_extended_features(rr_intervals)

        # Function returns NaN for < 10 intervals
        assert np.isnan(features["mean_rr"])
        assert np.isnan(features["rmssd"])

    def test_minimum_intervals_for_frequency(self):
        """Test minimum intervals needed for frequency analysis."""
        # Less than 10 intervals - all features should be NaN
        rr_intervals = np.array([1000, 1010, 1020])

        features = extract_extended_features(rr_intervals)

        # With < 10 intervals, all features are NaN
        assert np.isnan(features["mean_rr"])

    def test_high_variability(self):
        """Test with high variability RR intervals."""
        np.random.seed(42)
        rr_intervals = 1000 + 200 * np.random.randn(100)
        rr_intervals = np.clip(rr_intervals, 400, 2000)  # Physiological limits

        features = extract_extended_features(rr_intervals)

        assert features["sdnn"] > 100  # High variability
        assert not np.isnan(features["lf_hf_ratio"])

    def test_all_features_returned(self):
        """Test that all 20 features are returned."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(200)

        features = extract_extended_features(rr_intervals)

        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"


class TestFeatureValues:
    """Tests for feature value sanity checks."""

    def test_sdnn_less_than_range(self):
        """Test that SDNN is always less than range."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert features["sdnn"] <= features["range_rr"]

    def test_rmssd_positive(self):
        """Test that RMSSD is always positive or zero."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert features["rmssd"] >= 0

    def test_pnn50_percentage(self):
        """Test that pNN50 is between 0 and 100."""
        np.random.seed(42)
        rr_intervals = 1000 + 50 * np.random.randn(100)

        features = extract_extended_features(rr_intervals)

        assert 0 <= features["pnn50"] <= 100

    def test_heart_rate_physiological(self):
        """Test that mean HR is in physiological range."""
        rr_intervals = np.array([1000] * 10)  # 60 bpm

        features = extract_extended_features(rr_intervals)

        assert 20 <= features["mean_hr"] <= 250  # Physiological range


class TestWithRealLikeData:
    """Tests with realistic data patterns."""

    def test_stressed_pattern(self):
        """Test with stress-like pattern (low HRV)."""
        np.random.seed(42)
        # Stress: faster HR, lower variability
        rr_intervals = 750 + 20 * np.random.randn(200)  # ~80 bpm, low variability

        features = extract_extended_features(rr_intervals)

        assert features["mean_hr"] > 70
        assert features["sdnn"] < 50
        assert features["rmssd"] < 30

    def test_baseline_pattern(self):
        """Test with baseline-like pattern (normal HRV)."""
        np.random.seed(42)
        # Baseline: normal HR, higher variability
        rr_intervals = 1000 + 60 * np.random.randn(200)  # ~60 bpm, normal variability

        features = extract_extended_features(rr_intervals)

        assert 55 <= features["mean_hr"] <= 70
        assert features["sdnn"] > 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
