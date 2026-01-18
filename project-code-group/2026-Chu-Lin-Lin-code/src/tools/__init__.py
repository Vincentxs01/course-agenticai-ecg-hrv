# SPDX-License-Identifier: Apache-2.0
"""Tool implementations for HRV Analysis Agent (WESAD dataset)."""

from .ecg_loader import (
    load_ecg,
    load_wesad_subject,
    extract_condition_segments,
    WESAD_SAMPLING_RATE,
    WESAD_SUBJECTS,
    WESAD_LABELS,
)
from .signal_processor import process_signal, bandpass_filter, detect_r_peaks
from .feature_extractor import extract_hrv_features
from .extended_features import (
    extract_extended_features,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_CATEGORIES,
)
from .classifier import (
    # Core functions
    load_classifier,
    save_classifier,
    train_classifier,
    predict_stress,
    # Classifier selection
    get_available_classifiers,
    list_classifiers,
    get_classifier_info,
    create_classifier,
    recommend_classifier,
    get_feature_importance,
    # Constants
    CLASSIFIER_REGISTRY,
    DEFAULT_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
)
from .report_generator import generate_report, generate_interpretation

__all__ = [
    # WESAD data loading
    "load_ecg",
    "load_wesad_subject",
    "extract_condition_segments",
    "WESAD_SAMPLING_RATE",
    "WESAD_SUBJECTS",
    "WESAD_LABELS",
    # Signal processing
    "process_signal",
    "bandpass_filter",
    "detect_r_peaks",
    # Feature extraction (basic)
    "extract_hrv_features",
    # Feature extraction (extended - 20 features)
    "extract_extended_features",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "FEATURE_CATEGORIES",
    # Classification (20 classifiers)
    "load_classifier",
    "save_classifier",
    "train_classifier",
    "predict_stress",
    "get_available_classifiers",
    "list_classifiers",
    "get_classifier_info",
    "create_classifier",
    "recommend_classifier",
    "get_feature_importance",
    "CLASSIFIER_REGISTRY",
    "DEFAULT_FEATURE_NAMES",
    "EXTENDED_FEATURE_NAMES",
    # Report generation
    "generate_report",
    "generate_interpretation",
]
