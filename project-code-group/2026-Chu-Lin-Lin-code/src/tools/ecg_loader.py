# SPDX-License-Identifier: Apache-2.0
"""ECG data loading and validation tool for WESAD dataset."""

import numpy as np
from pathlib import Path
from typing import Union, Optional
import pandas as pd











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




def read_ecg_csv_column(csv_path: Path, ecg_col_index: int = 3, header: bool = True) -> np.ndarray:
    """
    Reads an ECG signal from a CSV file, selecting a specific column.

    Assumes a schema where ECG data is in a specified column.
    Handles NaN values by replacing them with the median.

    Args:
        csv_path: Path to the CSV file.
        ecg_col_index: 0-based index of the ECG column. Default is 3 (D column).
        header: Whether the CSV has a header row.

    Returns:
        np.ndarray: The ECG signal as a NumPy array.

    Raises:
        ValueError: If the specified ECG column index is out of bounds.
    """
    df = pd.read_csv(csv_path, header=0 if header else None)
    
    # Check if a named column 'ECG' exists, otherwise use index
    if 'ECG' in df.columns:
        x = df['ECG']
    elif 'ecg' in df.columns:
        x = df['ecg']
    elif ecg_col_index < df.shape[1]:
        x = df.iloc[:, ecg_col_index]
    else:
        raise ValueError(f"{csv_path}: expected ECG column index {ecg_col_index} or 'ECG' column, "
                         f"but got only {df.shape[1]} columns and no 'ECG' column.")
    
    x = x.astype(float).to_numpy()
    x = np.nan_to_num(x, nan=np.nanmedian(x)) # Replace NaNs with median
    return x


def pick_ecg_column(df: pd.DataFrame) -> str:
    """
    Identifies the most likely ECG column in a DataFrame.
    Prefers columns named 'ECG' (case-insensitive), then falls back to the 4th column (index 3).

    Args:
        df: Pandas DataFrame containing ECG data.

    Returns:
        str: The name of the identified ECG column.

    Raises:
        ValueError: If no suitable ECG column can be found.
    """
    for name in ["ECG", "ecg", "Ecg"]:
        if name in df.columns:
            return name
    # Otherwise fall back to 4th column (A,B,C,D -> ECG is D)
    if df.shape[1] >= 4:
        return df.columns[3]
    raise ValueError(f"Cannot find ECG column. Columns={list(df.columns)}")


def pick_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifies the most likely time column in a DataFrame.
    Prefers columns named 'Timestamp' or 'Time' (case-insensitive).

    Args:
        df: Pandas DataFrame containing data.

    Returns:
        Optional[str]: The name of the identified time column, or None if not found.
    """
    for name in ["Timestamp", "timestamp", "Time", "time"]:
        if name in df.columns:
            return name
    return None
