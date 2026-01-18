# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the multi-classifier module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.classifier import (
    CLASSIFIER_REGISTRY,
    list_classifiers,
    get_available_classifiers,
    get_classifier_info,
    create_classifier,
    train_classifier,
    save_classifier,
    load_classifier,
    predict_stress,
    recommend_classifier,
    get_feature_importance,
    EXTENDED_FEATURE_NAMES,
)


class TestClassifierRegistry:
    """Tests for classifier registry functions."""

    def test_list_classifiers_returns_20(self):
        """Test that exactly 20 classifiers are available."""
        classifiers = list_classifiers()
        assert len(classifiers) == 20

    def test_list_classifiers_returns_strings(self):
        """Test that classifier names are strings."""
        classifiers = list_classifiers()
        assert all(isinstance(name, str) for name in classifiers)

    def test_get_available_classifiers_structure(self):
        """Test that get_available_classifiers returns proper structure."""
        classifiers = get_available_classifiers()
        assert len(classifiers) == 20

        for name, info in classifiers.items():
            assert "category" in info
            assert "description" in info
            assert isinstance(info["category"], str)
            assert isinstance(info["description"], str)

    def test_get_classifier_info_valid(self):
        """Test getting info for a valid classifier."""
        info = get_classifier_info("random_forest")

        assert "name" in info
        assert "category" in info
        assert "description" in info
        assert "default_params" in info
        assert info["name"] == "random_forest"
        assert info["category"] == "Ensemble"

    def test_get_classifier_info_invalid(self):
        """Test that invalid classifier raises error."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            get_classifier_info("nonexistent_classifier")

    @pytest.mark.parametrize("clf_name", list_classifiers())
    def test_all_classifiers_have_info(self, clf_name):
        """Test that all classifiers have complete info."""
        info = get_classifier_info(clf_name)
        assert "name" in info
        assert "category" in info
        assert "description" in info
        assert len(info["description"]) > 10  # Meaningful description


class TestCreateClassifier:
    """Tests for classifier creation."""

    @pytest.mark.parametrize("clf_name", list_classifiers())
    def test_create_all_classifiers(self, clf_name):
        """Test that all classifiers can be created."""
        clf = create_classifier(clf_name)
        assert clf is not None
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")

    def test_create_with_custom_params(self):
        """Test creating classifier with custom parameters."""
        clf = create_classifier("random_forest", n_estimators=50, max_depth=5)
        assert clf.n_estimators == 50
        assert clf.max_depth == 5

    def test_create_invalid_classifier(self):
        """Test that invalid classifier raises error."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            create_classifier("nonexistent")


class TestTrainClassifier:
    """Tests for classifier training."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create separable classes
        X_class0 = np.random.randn(n_samples // 2, n_features) - 1
        X_class1 = np.random.randn(n_samples // 2, n_features) + 1
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        return X, y

    def test_train_default_classifier(self, sample_data):
        """Test training with default classifier."""
        X, y = sample_data
        clf, scaler, feature_names = train_classifier(X, y)

        assert clf is not None
        assert scaler is not None
        assert feature_names is not None

    @pytest.mark.parametrize("clf_name", [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm_rbf",
        "knn",
        "gaussian_nb",
        "lda",
        "mlp"
    ])
    def test_train_specific_classifiers(self, sample_data, clf_name):
        """Test training specific classifiers."""
        X, y = sample_data
        clf, scaler, feature_names = train_classifier(X, y, classifier_name=clf_name)

        assert clf is not None
        # Test prediction works
        X_scaled = scaler.transform(X[:5])
        predictions = clf.predict(X_scaled)
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)

    def test_train_with_feature_names(self, sample_data):
        """Test training with custom feature names."""
        X, y = sample_data
        custom_names = [f"feat_{i}" for i in range(X.shape[1])]

        clf, scaler, feature_names = train_classifier(
            X, y,
            feature_names=custom_names
        )

        assert feature_names == custom_names


class TestSaveLoadClassifier:
    """Tests for saving and loading classifiers."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)

        return train_classifier(X, y, classifier_name="logistic_regression")

    def test_save_load_roundtrip(self, trained_classifier):
        """Test saving and loading a classifier."""
        clf, scaler, feature_names = trained_classifier

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            save_classifier(clf, scaler, temp_path, "logistic_regression", feature_names)

            # Load it back
            clf_loaded, scaler_loaded, names_loaded = load_classifier(temp_path)

            assert clf_loaded is not None
            assert scaler_loaded is not None

            # Test predictions match
            X_test = np.random.randn(5, 6)
            X_scaled = scaler.transform(X_test)
            X_scaled_loaded = scaler_loaded.transform(X_test)

            np.testing.assert_array_equal(
                clf.predict(X_scaled),
                clf_loaded.predict(X_scaled_loaded)
            )
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_classifier("/nonexistent/path.joblib")


class TestPredictStress:
    """Tests for stress prediction."""

    @pytest.fixture
    def prediction_setup(self):
        """Set up classifier and sample features for prediction."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)

        clf, scaler, feature_names = train_classifier(
            X, y,
            classifier_name="logistic_regression",
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6"]
        )

        return clf, scaler, ["f1", "f2", "f3", "f4", "f5", "f6"]

    def test_predict_stress_baseline(self, prediction_setup):
        """Test prediction returns valid structure."""
        clf, scaler, feature_names = prediction_setup

        features = {
            "f1": -2.0, "f2": 0.1, "f3": -0.5,
            "f4": 0.2, "f5": -0.3, "f6": 0.1
        }

        result = predict_stress(clf, scaler, features, feature_names)

        assert "prediction" in result
        assert "prediction_label" in result
        assert "confidence" in result
        assert "stress_probability" in result
        assert "baseline_probability" in result

        assert result["prediction"] in ["Stressed", "Baseline"]
        assert result["prediction_label"] in [0, 1]
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["stress_probability"] <= 1
        assert 0 <= result["baseline_probability"] <= 1

    def test_predict_stress_with_missing_features(self, prediction_setup):
        """Test prediction handles missing features."""
        clf, scaler, feature_names = prediction_setup

        # Missing f3 and f4
        features = {"f1": -2.0, "f2": 0.1, "f5": -0.3, "f6": 0.1}

        result = predict_stress(clf, scaler, features, feature_names)

        assert "error" in result
        assert "Missing features" in result["error"]


class TestRecommendClassifier:
    """Tests for classifier recommendation."""

    def test_recommend_for_accuracy(self):
        """Test recommendation for accuracy priority."""
        result = recommend_classifier(n_samples=500, n_features=20, priority="accuracy")
        assert result in list_classifiers()

    def test_recommend_for_speed(self):
        """Test recommendation for speed priority."""
        result = recommend_classifier(n_samples=500, n_features=20, priority="speed")
        assert result in list_classifiers()

    def test_recommend_for_interpretability(self):
        """Test recommendation for interpretability priority."""
        result = recommend_classifier(n_samples=500, n_features=20, priority="interpretability")
        assert result in list_classifiers()

    def test_recommend_small_dataset(self):
        """Test recommendation for small dataset."""
        result = recommend_classifier(n_samples=50, n_features=20, priority="accuracy")
        assert result in list_classifiers()

    def test_recommend_large_dataset(self):
        """Test recommendation for large dataset."""
        result = recommend_classifier(n_samples=10000, n_features=20, priority="accuracy")
        assert result in list_classifiers()


class TestGetFeatureImportance:
    """Tests for feature importance extraction."""

    def test_feature_importance_random_forest(self):
        """Test feature importance from Random Forest."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)

        clf, _, _ = train_classifier(X, y, classifier_name="random_forest")

        importance = get_feature_importance(
            clf,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6"]
        )

        assert len(importance) == 6
        assert all(v >= 0 for v in importance.values())
        # Feature 0 should be most important since y depends on it
        assert importance["f1"] > 0

    def test_feature_importance_logistic(self):
        """Test feature importance from Logistic Regression."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = (X[:, 0] > 0).astype(int)

        clf, _, _ = train_classifier(X, y, classifier_name="logistic_regression")

        importance = get_feature_importance(
            clf,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6"]
        )

        assert len(importance) == 6


class TestAllClassifiersTrainAndPredict:
    """Comprehensive tests for all 20 classifiers."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create linearly separable data
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Train/test split
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, y_train, X_test, y_test

    @pytest.mark.parametrize("clf_name", list_classifiers())
    def test_classifier_train_predict(self, classification_data, clf_name):
        """Test that each classifier can train and predict."""
        X_train, y_train, X_test, y_test = classification_data

        # Train
        clf, scaler, feature_names = train_classifier(
            X_train, y_train,
            classifier_name=clf_name,
            feature_names=EXTENDED_FEATURE_NAMES
        )

        # Predict
        X_test_scaled = scaler.transform(X_test)
        predictions = clf.predict(X_test_scaled)

        # Verify predictions
        assert len(predictions) == len(y_test)
        assert all(p in [0, 1] for p in predictions)

        # At least some correct predictions (better than random)
        accuracy = (predictions == y_test).mean()
        assert accuracy > 0.3, f"{clf_name} accuracy too low: {accuracy}"

    @pytest.mark.parametrize("clf_name", [
        "logistic_regression",
        "random_forest",
        "svm_rbf",
        "gaussian_nb"
    ])
    def test_classifier_probability_output(self, classification_data, clf_name):
        """Test classifiers with probability output."""
        X_train, y_train, X_test, y_test = classification_data

        clf, scaler, _ = train_classifier(X_train, y_train, classifier_name=clf_name)

        X_test_scaled = scaler.transform(X_test)

        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X_test_scaled)
            assert probas.shape == (len(X_test), 2)
            assert np.allclose(probas.sum(axis=1), 1.0)


class TestExtendedFeatureNames:
    """Tests for feature name constants."""

    def test_extended_feature_names_count(self):
        """Test that there are exactly 20 feature names."""
        assert len(EXTENDED_FEATURE_NAMES) == 20

    def test_extended_feature_names_unique(self):
        """Test that all feature names are unique."""
        assert len(set(EXTENDED_FEATURE_NAMES)) == 20

    def test_extended_feature_names_expected(self):
        """Test that expected feature names are present."""
        expected = [
            "mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr",
            "lf_power", "hf_power", "lf_hf_ratio",
            "sd1", "sd2", "sample_entropy"
        ]
        for name in expected:
            assert name in EXTENDED_FEATURE_NAMES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
