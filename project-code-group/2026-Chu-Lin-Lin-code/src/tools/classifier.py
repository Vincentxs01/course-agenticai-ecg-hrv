# SPDX-License-Identifier: Apache-2.0
"""
Multi-classifier module for stress detection from HRV features.

This module provides 20 different classifiers that the orchestrator can select
from based on the task requirements. Each classifier has different strengths:
- Ensemble methods: High accuracy, good generalization
- Linear models: Fast, interpretable
- SVMs: Good with high-dimensional data
- Neural networks: Can learn complex patterns
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
import joblib
from sklearn.preprocessing import StandardScaler

# Classifier imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier


# =============================================================================
# CLASSIFIER REGISTRY
# =============================================================================

CLASSIFIER_REGISTRY = {
    # Ensemble methods (6)
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "category": "Ensemble",
        "description": "Ensemble of decision trees with bagging - good balance of accuracy and speed",
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "category": "Ensemble",
        "description": "Sequential boosting with gradient descent - high accuracy",
    },
    "adaboost": {
        "class": AdaBoostClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "category": "Ensemble",
        "description": "Adaptive boosting with weighted samples",
    },
    "extra_trees": {
        "class": ExtraTreesClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "category": "Ensemble",
        "description": "Extremely randomized trees - faster than random forest",
    },
    "bagging": {
        "class": BaggingClassifier,
        "params": {"n_estimators": 50, "random_state": 42, "n_jobs": 1},
        "category": "Ensemble",
        "description": "Bootstrap aggregating classifier",
    },
    "hist_gradient_boosting": {
        "class": HistGradientBoostingClassifier,
        "params": {"max_iter": 100, "max_depth": 10, "random_state": 42},
        "category": "Ensemble",
        "description": "Histogram-based gradient boosting - fast for large datasets",
    },
    # Linear models (5)
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42},
        "category": "Linear",
        "description": "Linear model with logistic function - interpretable probabilities",
    },
    "ridge": {
        "class": RidgeClassifier,
        "params": {"random_state": 42},
        "category": "Linear",
        "description": "Linear classifier with L2 regularization",
    },
    "sgd": {
        "class": SGDClassifier,
        "params": {"max_iter": 1000, "random_state": 42},
        "category": "Linear",
        "description": "Stochastic gradient descent - good for large datasets",
    },
    "perceptron": {
        "class": Perceptron,
        "params": {"max_iter": 1000, "random_state": 42},
        "category": "Linear",
        "description": "Single-layer neural network - simple and fast",
    },
    "passive_aggressive": {
        "class": PassiveAggressiveClassifier,
        "params": {"max_iter": 1000, "random_state": 42},
        "category": "Linear",
        "description": "Online learning algorithm - good for streaming data",
    },
    # Support Vector Machines (2)
    "svm_linear": {
        "class": LinearSVC,
        "params": {"max_iter": 10000, "random_state": 42},
        "category": "SVM",
        "description": "Support vector machine with linear kernel - fast",
    },
    "svm_rbf": {
        "class": SVC,
        "params": {"kernel": "rbf", "probability": True, "random_state": 42},
        "category": "SVM",
        "description": "SVM with RBF kernel - captures non-linear patterns",
    },
    # Distance-based (2)
    "knn": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5},
        "category": "Distance-based",
        "description": "K-nearest neighbors - simple, no training required",
    },
    "nearest_centroid": {
        "class": NearestCentroid,
        "params": {},
        "category": "Distance-based",
        "description": "Classification by nearest class centroid - very fast",
    },
    # Tree-based (1)
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "params": {"max_depth": 10, "random_state": 42},
        "category": "Tree",
        "description": "Single decision tree - interpretable but prone to overfitting",
    },
    # Probabilistic (1)
    "gaussian_nb": {
        "class": GaussianNB,
        "params": {},
        "category": "Probabilistic",
        "description": "Naive Bayes with Gaussian likelihood - fast, works well with small data",
    },
    # Discriminant Analysis (2)
    "lda": {
        "class": LinearDiscriminantAnalysis,
        "params": {},
        "category": "Discriminant",
        "description": "Linear discriminant analysis - dimensionality reduction + classification",
    },
    "qda": {
        "class": QuadraticDiscriminantAnalysis,
        "params": {"reg_param": 0.1},
        "category": "Discriminant",
        "description": "Quadratic discriminant analysis - captures non-linear boundaries",
    },
    # Neural Network (1)
    "mlp": {
        "class": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 1000,
            "random_state": 42,
        },
        "category": "Neural Network",
        "description": "Multi-layer perceptron - learns complex patterns",
    },
}


# Default feature names (6 basic HRV features for backward compatibility)
DEFAULT_FEATURE_NAMES = [
    "sdnn",
    "rmssd",
    "pnn50",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
]

# Extended feature names (all 20 HRV features)
EXTENDED_FEATURE_NAMES = [
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "std_hr",
    "cv_rr",
    "range_rr",
    "median_rr",
    "iqr_rr",
    "vlf_power",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "lf_nu",
    "hf_nu",
    "sd1",
    "sd2",
    "sd_ratio",
    "sample_entropy",
]


# =============================================================================
# CLASSIFIER FUNCTIONS
# =============================================================================


def get_available_classifiers() -> dict:
    """
    Get dictionary of all available classifiers with their metadata.

    Returns:
        dict: {classifier_name: {'category', 'description', 'class', 'params'}}
    """
    return {
        name: {
            "category": info["category"],
            "description": info["description"],
        }
        for name, info in CLASSIFIER_REGISTRY.items()
    }


def list_classifiers() -> list[str]:
    """
    Get list of all available classifier names.

    Returns:
        list: List of classifier names (e.g., ['random_forest', 'logistic_regression', ...])
    """
    return list(CLASSIFIER_REGISTRY.keys())


def get_classifier_info(classifier_name: str) -> dict:
    """
    Get detailed information about a specific classifier.

    Args:
        classifier_name: Name of the classifier

    Returns:
        dict: Classifier metadata including category, description, and default params

    Raises:
        ValueError: If classifier_name is not recognized
    """
    if classifier_name not in CLASSIFIER_REGISTRY:
        available = list_classifiers()
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. Available: {available}"
        )

    info = CLASSIFIER_REGISTRY[classifier_name]
    return {
        "name": classifier_name,
        "category": info["category"],
        "description": info["description"],
        "default_params": info["params"],
    }


def create_classifier(classifier_name: str, **kwargs):
    """
    Create a classifier instance by name.

    Args:
        classifier_name: Name of the classifier (e.g., 'random_forest', 'logistic_regression')
        **kwargs: Override default parameters

    Returns:
        sklearn classifier instance

    Raises:
        ValueError: If classifier_name is not recognized
    """
    if classifier_name not in CLASSIFIER_REGISTRY:
        available = list_classifiers()
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. Available: {available}"
        )

    info = CLASSIFIER_REGISTRY[classifier_name]
    params = {**info["params"], **kwargs}  # Merge defaults with overrides
    return info["class"](**params)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_name: str = "random_forest",
    feature_names: Optional[list[str]] = None,
    **classifier_kwargs,
) -> tuple:
    """
    Train a classifier for stress detection.

    Args:
        X_train: Training features [n_samples, n_features]
        y_train: Training labels (0=baseline, 1=stressed)
        classifier_name: Name of classifier to use (default: 'random_forest')
        feature_names: List of feature names (default: EXTENDED_FEATURE_NAMES)
        **classifier_kwargs: Additional parameters to pass to the classifier

    Returns:
        tuple: (trained classifier, fitted scaler, feature_names)
    """
    if feature_names is None:
        if X_train.shape[1] == len(EXTENDED_FEATURE_NAMES):
            feature_names = EXTENDED_FEATURE_NAMES
        elif X_train.shape[1] == len(DEFAULT_FEATURE_NAMES):
            feature_names = DEFAULT_FEATURE_NAMES
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Create and train classifier
    clf = create_classifier(classifier_name, **classifier_kwargs)
    clf.fit(X_scaled, y_train)

    return clf, scaler, feature_names


def save_classifier(
    clf,
    scaler: StandardScaler,
    model_path: Union[str, Path],
    classifier_name: str = "unknown",
    feature_names: Optional[list[str]] = None,
) -> None:
    """
    Save trained classifier and scaler to disk.

    Args:
        clf: Trained classifier
        scaler: Fitted scaler
        model_path: Path to save the model
        classifier_name: Name of the classifier type
        feature_names: List of feature names used
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    joblib.dump(
        {
            "classifier": clf,
            "scaler": scaler,
            "classifier_name": classifier_name,
            "feature_names": feature_names,
        },
        model_path,
    )


def load_classifier(model_path: Union[str, Path]) -> tuple:
    """
    Load trained classifier and scaler from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        tuple: (classifier, scaler, feature_names)

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = joblib.load(model_path)

    return (
        model_data["classifier"],
        model_data["scaler"],
        model_data.get("feature_names", DEFAULT_FEATURE_NAMES),
    )


def predict_stress(
    clf,
    scaler: StandardScaler,
    features: dict,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """
    Predict stress state from HRV features.

    Args:
        clf: Trained classifier
        scaler: Fitted scaler
        features: Dictionary of HRV features
        feature_names: List of feature names to use

    Returns:
        dict: Prediction results with probabilities and confidence
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    # Extract feature vector
    X = np.array([[features.get(name, np.nan) for name in feature_names]])

    # Check for missing values
    if np.any(np.isnan(X)):
        missing = [name for name, val in zip(feature_names, X[0]) if np.isnan(val)]
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "stress_probability": np.nan,
            "baseline_probability": np.nan,
            "error": f"Missing features: {missing}",
        }

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    prediction = clf.predict(X_scaled)[0]

    # Get probabilities if available
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X_scaled)[0]
        stress_prob = float(probabilities[1])
        baseline_prob = float(probabilities[0])
    elif hasattr(clf, "decision_function"):
        # Convert decision function to pseudo-probability
        decision = clf.decision_function(X_scaled)[0]
        stress_prob = 1 / (1 + np.exp(-decision))  # Sigmoid
        baseline_prob = 1 - stress_prob
    else:
        stress_prob = float(prediction)
        baseline_prob = 1 - stress_prob

    # Calculate confidence
    if hasattr(clf, "estimators_"):
        # For ensemble methods: use tree agreement
        tree_predictions = np.array(
            [tree.predict(X_scaled)[0] for tree in clf.estimators_]
        )
        confidence = float(np.mean(tree_predictions == prediction))
    else:
        # For other classifiers: use probability difference
        confidence = float(abs(stress_prob - 0.5) * 2)

    # Get feature importance if available
    feature_importances = None
    if hasattr(clf, "feature_importances_"):
        feature_importances = dict(zip(feature_names, clf.feature_importances_))
    elif hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_.flatten())
        if len(coef) == len(feature_names):
            feature_importances = dict(zip(feature_names, coef / coef.sum()))

    result = {
        "prediction": "Stressed" if prediction == 1 else "Baseline",
        "prediction_label": int(prediction),
        "confidence": confidence,
        "stress_probability": stress_prob,
        "baseline_probability": baseline_prob,
    }

    if feature_importances:
        result["feature_importances"] = feature_importances

    return result


def get_feature_importance(clf, feature_names: Optional[list[str]] = None) -> dict:
    """
    Get feature importance from trained classifier.

    Args:
        clf: Trained classifier
        feature_names: List of feature names

    Returns:
        dict: Feature name to importance mapping (sorted by importance)
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_.flatten())
        importances = importances / importances.sum()  # Normalize
    else:
        return {name: 1.0 / len(feature_names) for name in feature_names}

    if len(importances) != len(feature_names):
        return {}

    return dict(
        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    )


def recommend_classifier(
    n_samples: int, n_features: int, priority: str = "accuracy"
) -> str:
    """
    Recommend a classifier based on dataset characteristics and priority.

    Args:
        n_samples: Number of training samples
        n_features: Number of features
        priority: 'accuracy', 'speed', or 'interpretability'

    Returns:
        str: Recommended classifier name
    """
    if priority == "speed":
        if n_samples < 1000:
            return "gaussian_nb"
        elif n_samples < 10000:
            return "logistic_regression"
        else:
            return "sgd"

    elif priority == "interpretability":
        if n_samples < 500:
            return "decision_tree"
        else:
            return "logistic_regression"

    else:  # priority == 'accuracy'
        if n_samples < 100:
            return "gaussian_nb"
        elif n_samples < 500:
            return "random_forest"
        elif n_samples < 5000:
            return "gradient_boosting"
        else:
            return "hist_gradient_boosting"
