# SPDX-License-Identifier: Apache-2.0
"""Tool implementations for HRV Analysis Agent (WESAD dataset)."""

from .ecg_loader import (
    load_ecg,
    read_ecg_csv_column,
    pick_ecg_column,
    pick_time_column,
)
from .signal_processor import process_signal, bandpass_filter, detect_r_peaks

from .extended_features import (
    extract_extended_features,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_CATEGORIES,
)

from .report_generator import generate_report, generate_interpretation

__all__ = [
    # Data loading
    "load_ecg",
    "read_ecg_csv_column",
    "pick_ecg_column",
    "pick_time_column",
    # Signal processing
    "process_signal",
    "bandpass_filter",
    "detect_r_peaks",

    # Feature extraction (extended - 20 features)
    "extract_extended_features",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "FEATURE_CATEGORIES",

    # Report generation
    "generate_report",
    "generate_interpretation",
]
