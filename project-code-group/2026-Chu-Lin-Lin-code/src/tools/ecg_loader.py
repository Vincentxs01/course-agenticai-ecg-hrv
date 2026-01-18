# SPDX-License-Identifier: Apache-2.0
"""ECG data loading and validation tool for WESAD dataset."""

import pickle
import numpy as np
from pathlib import Path
from typing import Union, Optional

# WESAD constants
WESAD_SAMPLING_RATE = 700  # Hz
WESAD_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
                  'S11', 'S13', 'S14', 'S15', 'S16', 'S17']  # S12 excluded
WESAD_LABELS = {0: 'undefined', 1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}


def load_wesad_subject(
    data_dir: Union[str, Path],
    subject_id: str
) -> dict:
    """
    Load WESAD subject data from pickle file.

    Args:
        data_dir: Path to WESAD data directory
        subject_id: Subject identifier (e.g., 'S2', 'S3', etc.)

    Returns:
        dict: Contains 'signal', 'labels', 'sampling_rate', 'subject_id'

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
        ValueError: If subject_id is invalid
    """
    data_dir = Path(data_dir)
    subject_path = data_dir / subject_id / f"{subject_id}.pkl"

    if subject_id not in WESAD_SUBJECTS:
        raise ValueError(f"Invalid subject ID: {subject_id}. Valid: {WESAD_SUBJECTS}")

    if not subject_path.exists():
        raise FileNotFoundError(f"WESAD file not found: {subject_path}")

    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract chest ECG (700 Hz)
    ecg = data['signal']['chest']['ECG'].flatten()
    labels = data['label']

    n_samples = len(ecg)
    duration_sec = n_samples / WESAD_SAMPLING_RATE

    return {
        "signal": ecg,
        "labels": labels,
        "sampling_rate": WESAD_SAMPLING_RATE,
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "subject_id": subject_id,
        "file_path": str(subject_path.absolute()),
    }


def extract_condition_segments(
    ecg_data: dict,
    condition: int = 2,
    min_duration_sec: float = 60.0
) -> list[dict]:
    """
    Extract continuous segments for a specific condition from WESAD data.

    Args:
        ecg_data: Dictionary from load_wesad_subject()
        condition: Label to extract (1=baseline, 2=stress, 3=amusement, 4=meditation)
        min_duration_sec: Minimum segment duration to keep

    Returns:
        list: List of segment dictionaries with 'signal', 'sampling_rate', etc.
    """
    signal = ecg_data["signal"]
    labels = ecg_data["labels"]
    fs = ecg_data["sampling_rate"]

    # Find continuous segments of the condition
    condition_mask = (labels == condition)
    segments = []

    # Find segment boundaries
    in_segment = False
    start_idx = 0

    for i, is_condition in enumerate(condition_mask):
        if is_condition and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_condition and in_segment:
            end_idx = i
            duration = (end_idx - start_idx) / fs
            if duration >= min_duration_sec:
                segments.append({
                    "signal": signal[start_idx:end_idx],
                    "sampling_rate": fs,
                    "duration_sec": duration,
                    "n_samples": end_idx - start_idx,
                    "condition": condition,
                    "condition_name": WESAD_LABELS.get(condition, 'unknown'),
                    "subject_id": ecg_data.get("subject_id"),
                    "start_sample": start_idx,
                    "end_sample": end_idx,
                })
            in_segment = False

    # Handle segment at end of recording
    if in_segment:
        end_idx = len(labels)
        duration = (end_idx - start_idx) / fs
        if duration >= min_duration_sec:
            segments.append({
                "signal": signal[start_idx:end_idx],
                "sampling_rate": fs,
                "duration_sec": duration,
                "n_samples": end_idx - start_idx,
                "condition": condition,
                "condition_name": WESAD_LABELS.get(condition, 'unknown'),
                "subject_id": ecg_data.get("subject_id"),
                "start_sample": start_idx,
                "end_sample": end_idx,
            })

    return segments


def load_ecg(
    file_path: Union[str, Path],
    sampling_rate: int = 700,
    expected_duration: float = None
) -> dict:
    """
    Load ECG data from a text file or WESAD pickle.

    Args:
        file_path: Path to ECG data file (.txt) or WESAD pickle (.pkl)
        sampling_rate: Sampling rate in Hz (default: 700 for WESAD)
        expected_duration: Expected duration in seconds (optional)

    Returns:
        dict: Contains 'signal', 'sampling_rate', 'duration_sec', 'n_samples'

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the data format is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Handle WESAD pickle files
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        if 'signal' in data and 'chest' in data['signal']:
            # WESAD format
            ecg = data['signal']['chest']['ECG'].flatten()
            labels = data.get('label', None)
            sampling_rate = WESAD_SAMPLING_RATE
        else:
            raise ValueError("Unrecognized pickle file format")

        n_samples = len(ecg)
        duration_sec = n_samples / sampling_rate

        result = {
            "signal": ecg,
            "sampling_rate": sampling_rate,
            "duration_sec": duration_sec,
            "n_samples": n_samples,
            "file_path": str(path.absolute()),
        }
        if labels is not None:
            result["labels"] = labels
        return result

    # Handle text files (original format)
    try:
        data = np.loadtxt(path)
    except Exception as e:
        raise ValueError(f"Failed to load ECG data: {e}")

    # Validate shape
    if data.ndim == 0 or data.size == 0:
        raise ValueError("ECG file is empty")

    if data.ndim > 1:
        if data.shape[1] == 1:
            data = data.flatten()
        else:
            raise ValueError(
                f"Expected single-column ECG data, got shape {data.shape}"
            )

    # Validate values
    if np.any(np.isnan(data)):
        nan_count = np.sum(np.isnan(data))
        raise ValueError(f"ECG data contains {nan_count} NaN values")

    if np.any(np.isinf(data)):
        raise ValueError("ECG data contains infinite values")

    # Calculate duration
    n_samples = len(data)
    duration_sec = n_samples / sampling_rate

    # Validate expected duration if provided
    if expected_duration is not None:
        if abs(duration_sec - expected_duration) > 1.0:
            raise ValueError(
                f"Duration mismatch: expected {expected_duration}s, "
                f"got {duration_sec:.1f}s"
            )

    return {
        "signal": data,
        "sampling_rate": sampling_rate,
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "file_path": str(path.absolute()),
    }


def load_ecg_batch(
    file_paths: list[Union[str, Path]],
    sampling_rate: int = 500
) -> list[dict]:
    """
    Load multiple ECG files.

    Args:
        file_paths: List of paths to ECG files
        sampling_rate: Sampling rate in Hz

    Returns:
        list: List of ECG data dictionaries
    """
    results = []
    for path in file_paths:
        try:
            data = load_ecg(path, sampling_rate)
            data["status"] = "success"
            results.append(data)
        except Exception as e:
            results.append({
                "file_path": str(path),
                "status": "error",
                "error": str(e)
            })
    return results
