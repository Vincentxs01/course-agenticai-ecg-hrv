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
    load_ecg_batch,
    WESAD_SAMPLING_RATE,
    WESAD_SUBJECTS,
    WESAD_LABELS,
)
from src.tools.signal_processor import (
    bandpass_filter,
    remove_baseline_wander,
    detect_r_peaks,
    compute_rr_intervals,
    remove_ectopic_beats,
    process_signal,
)
from src.tools.feature_extractor import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    extract_nonlinear_features,
    extract_hrv_features,
    get_feature_vector,
)
from src.tools.classifier import (
    list_classifiers,
    get_classifier_info,
    create_classifier,
    train_classifier,
    save_classifier,
    load_classifier,
    predict_stress,
    recommend_classifier,
    get_feature_importance,
    CLASSIFIER_REGISTRY,
    EXTENDED_FEATURE_NAMES,
)


# =============================================================================
# ECG LOADER TESTS - Additional Coverage
# =============================================================================

class TestECGLoaderParameters:
    """Tests for ECG loader parameter variations."""

    def test_load_ecg_different_sampling_rates(self):
        """Test loading with different sampling rates."""
        # Create synthetic data
        signal = np.sin(2 * np.pi * 1 * np.arange(1000) / 500)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f.name, signal)
            temp_path = f.name

        try:
            # Test with 500 Hz
            result_500 = load_ecg(temp_path, sampling_rate=500)
            assert result_500["sampling_rate"] == 500
            assert result_500["duration_sec"] == pytest.approx(2.0, rel=0.01)

            # Test with 700 Hz (WESAD default)
            result_700 = load_ecg(temp_path, sampling_rate=700)
            assert result_700["sampling_rate"] == 700
            assert result_700["duration_sec"] == pytest.approx(1000/700, rel=0.01)

            # Test with 1000 Hz
            result_1000 = load_ecg(temp_path, sampling_rate=1000)
            assert result_1000["sampling_rate"] == 1000
            assert result_1000["duration_sec"] == pytest.approx(1.0, rel=0.01)
        finally:
            Path(temp_path).unlink()

    def test_load_ecg_expected_duration_validation(self):
        """Test expected_duration parameter validation."""
        # Create 2 seconds of data at 500 Hz
        signal = np.sin(2 * np.pi * 1 * np.arange(1000) / 500)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f.name, signal)
            temp_path = f.name

        try:
            # Should pass with correct expected duration
            result = load_ecg(temp_path, sampling_rate=500, expected_duration=2.0)
            assert result["duration_sec"] == pytest.approx(2.0, rel=0.01)

            # Should fail with incorrect expected duration
            with pytest.raises(ValueError, match="Duration mismatch"):
                load_ecg(temp_path, sampling_rate=500, expected_duration=10.0)
        finally:
            Path(temp_path).unlink()

    def test_load_ecg_batch_success_and_failure(self):
        """Test batch loading with mixed success/failure."""
        # Create one valid file
        signal = np.sin(2 * np.pi * 1 * np.arange(500) / 500)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f.name, signal)
            valid_path = f.name

        try:
            results = load_ecg_batch(
                [valid_path, "/nonexistent/file.txt"],
                sampling_rate=500
            )

            assert len(results) == 2
            assert results[0]["status"] == "success"
            assert results[1]["status"] == "error"
        finally:
            Path(valid_path).unlink()

    def test_wesad_constants(self):
        """Test WESAD constants are defined correctly."""
        assert WESAD_SAMPLING_RATE == 700
        assert len(WESAD_SUBJECTS) == 15
        assert "S2" in WESAD_SUBJECTS
        assert "S12" not in WESAD_SUBJECTS  # S12 is excluded
        assert len(WESAD_LABELS) == 5
        assert WESAD_LABELS[1] == "baseline"
        assert WESAD_LABELS[2] == "stress"


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


# =============================================================================
# FEATURE EXTRACTOR TESTS - Additional Coverage
# =============================================================================

class TestFeatureExtractorFunctions:
    """Tests for feature extractor functions."""

    @pytest.fixture
    def sample_rr_intervals(self):
        """Generate sample RR intervals."""
        np.random.seed(42)
        return 1000 + 50 * np.random.randn(200)

    def test_extract_time_domain_features_short_input(self):
        """Test time domain with short input."""
        rr_short = np.array([1000])
        features = extract_time_domain_features(rr_short)

        assert np.isnan(features["sdnn"])
        assert np.isnan(features["rmssd"])

    def test_extract_time_domain_features_normal(self, sample_rr_intervals):
        """Test time domain with normal input."""
        features = extract_time_domain_features(sample_rr_intervals)

        assert not np.isnan(features["mean_rr"])
        assert not np.isnan(features["sdnn"])
        assert not np.isnan(features["rmssd"])
        assert not np.isnan(features["pnn50"])
        assert not np.isnan(features["mean_hr"])
        assert not np.isnan(features["std_hr"])
        assert 0 <= features["pnn50"] <= 100

    def test_extract_frequency_domain_custom_bands(self, sample_rr_intervals):
        """Test frequency domain with custom bands."""
        # Custom frequency bands
        features = extract_frequency_domain_features(
            sample_rr_intervals,
            fs_resample=4.0,
            vlf_band=(0.003, 0.04),
            lf_band=(0.04, 0.15),
            hf_band=(0.15, 0.4)
        )

        assert "lf_power" in features
        assert "hf_power" in features
        assert "lf_hf_ratio" in features

    def test_extract_frequency_domain_different_resample(self, sample_rr_intervals):
        """Test frequency domain with different resample rates."""
        features_4hz = extract_frequency_domain_features(
            sample_rr_intervals, fs_resample=4.0
        )
        features_2hz = extract_frequency_domain_features(
            sample_rr_intervals, fs_resample=2.0
        )

        # Both should produce valid features
        assert features_4hz["lf_power"] >= 0
        assert features_2hz["lf_power"] >= 0

    def test_extract_nonlinear_features(self, sample_rr_intervals):
        """Test nonlinear feature extraction."""
        features = extract_nonlinear_features(sample_rr_intervals)

        assert not np.isnan(features["sd1"])
        assert not np.isnan(features["sd2"])
        assert not np.isnan(features["sd_ratio"])
        assert features["sd1"] > 0
        assert features["sd2"] > 0

    def test_extract_nonlinear_features_short(self):
        """Test nonlinear with short input."""
        rr_short = np.array([1000, 1000])
        features = extract_nonlinear_features(rr_short)

        assert np.isnan(features["sd1"])
        assert np.isnan(features["sd2"])

    def test_extract_hrv_features_include_nonlinear(self, sample_rr_intervals):
        """Test extract_hrv_features with include_nonlinear parameter."""
        # With nonlinear features
        features_with = extract_hrv_features(sample_rr_intervals, include_nonlinear=True)
        assert "sd1" in features_with
        assert "sd2" in features_with

        # Without nonlinear features
        features_without = extract_hrv_features(sample_rr_intervals, include_nonlinear=False)
        assert "sd1" not in features_without
        assert "sd2" not in features_without

    def test_get_feature_vector(self):
        """Test feature vector extraction."""
        features = {
            "sdnn": 50.0,
            "rmssd": 30.0,
            "pnn50": 10.0,
            "lf_power": 1000.0,
            "hf_power": 500.0,
            "lf_hf_ratio": 2.0,
            "extra_feature": 100.0
        }

        vector = get_feature_vector(features)

        assert len(vector) == 6
        assert vector[0] == 50.0  # sdnn
        assert vector[1] == 30.0  # rmssd
        assert vector[5] == 2.0   # lf_hf_ratio

    def test_get_feature_vector_missing_feature(self):
        """Test feature vector with missing features."""
        features = {"sdnn": 50.0, "rmssd": 30.0}  # Missing other features

        vector = get_feature_vector(features)

        assert len(vector) == 6
        assert np.isnan(vector[2])  # pnn50 missing


# =============================================================================
# CLASSIFIER TESTS - Additional Coverage for All 20 Classifiers
# =============================================================================

class TestAllClassifiersComprehensive:
    """Comprehensive tests for all 20 classifiers."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, y_train, X_test, y_test

    @pytest.mark.parametrize("clf_name", list_classifiers())
    def test_train_all_classifiers(self, classification_data, clf_name):
        """Test training all 20 classifiers."""
        X_train, y_train, X_test, y_test = classification_data

        clf, scaler, feature_names = train_classifier(
            X_train, y_train,
            classifier_name=clf_name,
            feature_names=EXTENDED_FEATURE_NAMES
        )

        assert clf is not None
        assert scaler is not None

        # Test prediction
        X_test_scaled = scaler.transform(X_test)
        predictions = clf.predict(X_test_scaled)

        assert len(predictions) == len(y_test)
        assert all(p in [0, 1] for p in predictions)

    @pytest.mark.parametrize("clf_name", list_classifiers())
    def test_save_load_all_classifiers(self, classification_data, clf_name):
        """Test save/load for all 20 classifiers."""
        X_train, y_train, _, _ = classification_data

        clf, scaler, feature_names = train_classifier(
            X_train, y_train,
            classifier_name=clf_name
        )

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            save_classifier(clf, scaler, temp_path, clf_name, feature_names)
            clf_loaded, scaler_loaded, names_loaded = load_classifier(temp_path)

            assert clf_loaded is not None
            assert scaler_loaded is not None
        finally:
            Path(temp_path).unlink()


class TestRecommendClassifierPriorities:
    """Tests for all priority options in recommend_classifier."""

    @pytest.mark.parametrize("priority", ["accuracy", "speed", "interpretability"])
    def test_all_priority_options(self, priority):
        """Test all priority options."""
        result = recommend_classifier(n_samples=500, n_features=20, priority=priority)
        assert result in list_classifiers()

    @pytest.mark.parametrize("n_samples", [50, 100, 500, 1000, 5000, 10000])
    def test_different_sample_sizes_accuracy(self, n_samples):
        """Test recommendations for different sample sizes with accuracy priority."""
        result = recommend_classifier(n_samples=n_samples, n_features=20, priority="accuracy")
        assert result in list_classifiers()

    @pytest.mark.parametrize("n_samples", [50, 100, 1000, 10000])
    def test_different_sample_sizes_speed(self, n_samples):
        """Test recommendations for different sample sizes with speed priority."""
        result = recommend_classifier(n_samples=n_samples, n_features=20, priority="speed")
        assert result in list_classifiers()


class TestClassifierCategories:
    """Tests for classifier categories."""

    def test_ensemble_classifiers(self):
        """Test all ensemble classifiers."""
        ensemble_clfs = ["random_forest", "gradient_boosting", "adaboost",
                        "extra_trees", "bagging", "hist_gradient_boosting"]
        for clf_name in ensemble_clfs:
            info = get_classifier_info(clf_name)
            assert info["category"] == "Ensemble"

    def test_linear_classifiers(self):
        """Test all linear classifiers."""
        linear_clfs = ["logistic_regression", "ridge", "sgd",
                      "perceptron", "passive_aggressive"]
        for clf_name in linear_clfs:
            info = get_classifier_info(clf_name)
            assert info["category"] == "Linear"

    def test_svm_classifiers(self):
        """Test all SVM classifiers."""
        svm_clfs = ["svm_linear", "svm_rbf"]
        for clf_name in svm_clfs:
            info = get_classifier_info(clf_name)
            assert info["category"] == "SVM"

    def test_distance_based_classifiers(self):
        """Test all distance-based classifiers."""
        dist_clfs = ["knn", "nearest_centroid"]
        for clf_name in dist_clfs:
            info = get_classifier_info(clf_name)
            assert info["category"] == "Distance-based"

    def test_other_classifiers(self):
        """Test other classifier categories."""
        assert get_classifier_info("decision_tree")["category"] == "Tree"
        assert get_classifier_info("gaussian_nb")["category"] == "Probabilistic"
        assert get_classifier_info("lda")["category"] == "Discriminant"
        assert get_classifier_info("qda")["category"] == "Discriminant"
        assert get_classifier_info("mlp")["category"] == "Neural Network"


class TestFeatureImportanceAllClassifiers:
    """Tests for feature importance from different classifier types."""

    @pytest.fixture
    def trained_classifiers(self):
        """Train multiple classifier types."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)
        feature_names = ["f1", "f2", "f3", "f4", "f5", "f6"]

        classifiers = {}
        for clf_name in ["random_forest", "logistic_regression", "gradient_boosting", "knn"]:
            clf, scaler, _ = train_classifier(X, y, classifier_name=clf_name)
            classifiers[clf_name] = (clf, scaler)

        return classifiers, feature_names

    def test_feature_importance_tree_based(self, trained_classifiers):
        """Test feature importance from tree-based classifiers."""
        classifiers, feature_names = trained_classifiers

        clf, _ = classifiers["random_forest"]
        importance = get_feature_importance(clf, feature_names)

        assert len(importance) == 6
        assert sum(importance.values()) == pytest.approx(1.0, rel=0.01)

    def test_feature_importance_linear(self, trained_classifiers):
        """Test feature importance from linear classifiers."""
        classifiers, feature_names = trained_classifiers

        clf, _ = classifiers["logistic_regression"]
        importance = get_feature_importance(clf, feature_names)

        assert len(importance) == 6

    def test_feature_importance_knn(self, trained_classifiers):
        """Test feature importance from KNN (no native importance)."""
        classifiers, feature_names = trained_classifiers

        clf, _ = classifiers["knn"]
        importance = get_feature_importance(clf, feature_names)

        # KNN should return equal importance
        assert len(importance) == 6


# =============================================================================
# PREDICT STRESS TESTS
# =============================================================================

class TestPredictStress:
    """Tests for stress prediction function."""

    @pytest.fixture
    def prediction_setup(self):
        """Set up classifier for prediction tests."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)
        feature_names = ["f1", "f2", "f3", "f4", "f5", "f6"]

        clf, scaler, _ = train_classifier(
            X, y,
            classifier_name="logistic_regression",
            feature_names=feature_names
        )

        return clf, scaler, feature_names

    def test_predict_stress_complete_features(self, prediction_setup):
        """Test prediction with complete features."""
        clf, scaler, feature_names = prediction_setup

        features = {
            "f1": -2.0, "f2": 0.1, "f3": -0.5,
            "f4": 0.2, "f5": -0.3, "f6": 0.1
        }

        result = predict_stress(clf, scaler, features, feature_names)

        assert "prediction" in result
        assert "confidence" in result
        assert "stress_probability" in result
        assert result["prediction"] in ["Stressed", "Baseline"]

    def test_predict_stress_missing_features(self, prediction_setup):
        """Test prediction with missing features."""
        clf, scaler, feature_names = prediction_setup

        features = {"f1": -2.0, "f2": 0.1}  # Missing f3-f6

        result = predict_stress(clf, scaler, features, feature_names)

        assert "error" in result
        assert "Missing features" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
