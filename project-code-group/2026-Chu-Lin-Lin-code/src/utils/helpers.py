# SPDX-License-Identifier: Apache-2.0
"""Utility functions for HRV Analysis Agent."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml


# repo root (adjust if project structure changes)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger("hrv_agent")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    import yaml
    from pathlib import Path

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # IMPORTANT: force UTF-8 on Windows (handles BOM too)
    with open(p, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    return config or {}



def validate_ecg_data(
    data: np.ndarray,
    sampling_rate: int = 500,
    min_duration: float = 60.0,
    max_duration: float = 3600.0
) -> dict:
    """
    Validate ECG data quality.

    Args:
        data: ECG signal array
        sampling_rate: Sampling rate in Hz
        min_duration: Minimum acceptable duration in seconds
        max_duration: Maximum acceptable duration in seconds

    Returns:
        dict: Validation results with 'valid' boolean and 'issues' list
    """
    issues = []

    # Check for empty data
    if len(data) == 0:
        issues.append("Data is empty")
        return {"valid": False, "issues": issues}

    # Check duration
    duration = len(data) / sampling_rate
    if duration < min_duration:
        issues.append(f"Duration ({duration:.1f}s) is below minimum ({min_duration}s)")
    if duration > max_duration:
        issues.append(f"Duration ({duration:.1f}s) exceeds maximum ({max_duration}s)")

    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(data))
    if nan_count > 0:
        issues.append(f"Data contains {nan_count} NaN values")

    inf_count = np.sum(np.isinf(data))
    if inf_count > 0:
        issues.append(f"Data contains {inf_count} infinite values")

    # Check for flat signal (no variation)
    if np.std(data) < 1e-10:
        issues.append("Signal appears to be flat (no variation)")

    # Check for extreme values (potential clipping)
    max_val = np.max(np.abs(data))
    if max_val > 10:  # Assuming normalized or mV scale
        issues.append(f"Signal has extreme values (max: {max_val:.2f})")

    # Check for saturation
    unique_vals = len(np.unique(data))
    if unique_vals < 10:
        issues.append(f"Signal may be saturated (only {unique_vals} unique values)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "duration_sec": duration,
        "n_samples": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize signal to zero mean and unit variance.

    Args:
        data: Input signal

    Returns:
        np.ndarray: Normalized signal
    """
    mean = np.mean(data)
    std = np.std(data)

    if std < 1e-10:
        return data - mean

    return (data - mean) / std


def get_env_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        str or None: Environment variable value
    """
    return os.environ.get(name, default)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path: The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_dir(cfg: dict) -> Path:
    """
    Resolves the data directory path from the configuration,
    checking common base paths.
    """
    candidates = [
        cfg.get("dataset", {}).get("data_dir"),
        cfg.get("data", {}).get("data_dir"),
        cfg.get("custom", {}).get("data_dir"),
        cfg.get("wesad", {}).get("data_dir"),
    ]

    bases = [
        REPO_ROOT,                  # .../src/utils
        REPO_ROOT.parent,           # .../src
        REPO_ROOT.parent.parent,    # .../project-code-group/2026-Chu-Lin-Lin-code
        REPO_ROOT.parent.parent.parent, # .../project-code-group
        REPO_ROOT.parent.parent.parent.parent, # .../course-agenticai-ecg-hrv
    ]

    for c in candidates:
        if not c:
            continue

        # Try relative to common bases
        for b in bases:
            p = (b / c).resolve()
            if p.exists():
                return p

        # Also try as absolute
        p2 = Path(c).expanduser().resolve()
        if p2.exists():
            return p2

    # Common fallbacks relative to course root
    for b in bases:
        for p in [(b / "data-group" / "data"), (b / "data-group")]:
            p = p.resolve()
            if p.exists():
                return p

    raise FileNotFoundError("Cannot find data directory (checked config + common fallbacks).")


def resolve_sampling_rate(cfg: dict) -> float:
    """
    Resolves the sampling rate from the configuration.
    """
    return float(cfg.get("signal", {}).get("sampling_rate", 50))


def infer_persons_states(data_dir: Path):
    """
    Infers person IDs and states (Rest, Active) by scanning the data directory structure.
    Assumes a structure like data_dir/{person}/{state}/*.csv
    """
    persons = [p.name for p in data_dir.iterdir() if p.is_dir()]
    states = ["Rest", "Active"] # Assuming these are the fixed states for this project
    return persons, states


def scan_csv_files(data_dir: Path, file_glob: str = "*.csv"):
    """
    Scans a data directory for CSV files organized by person and state.
    Returns a list of dictionaries, each representing a record with 'person', 'state', 'path'.
    """
    persons, states = infer_persons_states(data_dir)
    records = []
    for pid in persons:
        for st in states:
            folder = data_dir / pid / st
            if not folder.exists():
                continue
            for f in sorted(folder.glob(file_glob)):
                records.append({"person": pid, "state": st, "path": f})
    return records

