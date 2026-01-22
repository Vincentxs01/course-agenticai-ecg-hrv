#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Batch ECG evaluation for a small custom dataset (2 persons × Rest/Active × CSV).

What it does:
1) Scans all CSV files under:
   data_dir/{person}/{Rest|Active}/*.csv
2) Builds a personalized baseline per person using Rest data (HR + HRV).
3) For each CSV file, computes a window-based pass rate:
   - RR intervals must be within physiological range
   - HR/HRV must be within that person's baseline range
   - Rest vs Active use different thresholds (Rest stricter, Active looser)
4) Writes:
   - reports/baselines/{person_id}_baseline.json
   - reports/pass_rates.csv (16 rows in your case)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import load_config from src.utils.helpers
from src.utils.helpers import load_config 
# Import read_ecg_csv_column from src.tools.ecg_loader
from src.tools.ecg_loader import read_ecg_csv_column

# Import bandpass_filter and detect_r_peaks from src.tools.signal_processor
from src.tools.signal_processor import bandpass_filter, detect_r_peaks

# -----------------------------
# Config loading (removed local definition)
# -----------------------------


# -----------------------------
# Signal processing + features (moved bandpass_filter and detect_r_peaks to src/tools)
# -----------------------------


def rr_intervals_seconds(r_peaks: np.ndarray, fs: int) -> np.ndarray:
    if r_peaks is None or len(r_peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(r_peaks) / float(fs)


def hrv_metrics(rr_sec: np.ndarray) -> dict:
    if rr_sec.size < 2:
        return {"mean_hr": np.nan, "sdnn": np.nan, "rmssd": np.nan}

    rr_ms = rr_sec * 1000.0
    mean_rr = np.mean(rr_sec)
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan

    sdnn = np.std(rr_ms, ddof=1) if rr_ms.size >= 2 else np.nan
    diff_ms = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_ms ** 2)) if diff_ms.size >= 1 else np.nan

    return {"mean_hr": float(mean_hr), "sdnn": float(sdnn), "rmssd": float(rmssd)}


# -----------------------------
# Dataset scanning / CSV loading (removed local read_ecg_column)
# -----------------------------


def scan_files(data_dir: Path, persons_cfg, states: list[str], file_glob: str) -> list[dict]:
    records = []

    # persons_cfg can be list[str] or list[dict]
    if isinstance(persons_cfg, list) and len(persons_cfg) > 0 and isinstance(persons_cfg[0], dict):
        # dict form: each person may define condition globs
        for p in persons_cfg:
            pid = p["id"]
            conditions = p.get("conditions", {})
            for st in states:
                # if condition has its own glob, use it; otherwise fallback
                cond = conditions.get(st, {})
                glob_pat = cond.get("glob", f"{pid}/{st}/{file_glob}")
                # glob_pat can be relative to data_dir
                for fp in sorted((data_dir).glob(glob_pat)):
                    records.append({"person_id": pid, "state": st, "path": fp})
    else:
        # list[str] form
        for pid in persons_cfg:
            for st in states:
                folder = data_dir / pid / st
                for fp in sorted(folder.glob(file_glob)):
                    records.append({"person_id": pid, "state": st, "path": fp})

    return records



# -----------------------------
# Windowing + pass/fail
# -----------------------------
def sliding_windows(x: np.ndarray, fs: int, win_sec: int, overlap: float):
    win = int(win_sec * fs)
    step = max(1, int(win * (1.0 - overlap)))
    if win <= 0 or len(x) < win:
        return
    for start in range(0, len(x) - win + 1, step):
        yield x[start : start + win]

def collect_windows_metrics_for_files(files, fs, sig_cfg, r_cfg, win_sec, overlap, ecg_col, header):
    all_metrics = []
    for rf in files:
        x = read_ecg_csv_column(rf["path"], ecg_col_index=ecg_col, header=header) # Updated call

        # filter
        x_f = bandpass_filter(
            x, fs,
            float(sig_cfg["bandpass_low"]),
            float(sig_cfg["bandpass_high"]),
            int(sig_cfg.get("filter_order", 4))
        )

        for w in sliding_windows(x_f, fs, win_sec, overlap):
            peaks = detect_r_peaks(w, fs, min_rr_sec=float(r_cfg["min_rr_sec"]))
            rr = rr_intervals_seconds(peaks, fs)

            if rr.size < 2:
                continue

            # hard physiological RR constraints for baseline collection
            if np.any(rr < float(r_cfg["min_rr_sec"])) or np.any(rr > float(r_cfg["max_rr_sec"])):
                continue

            all_metrics.append(hrv_metrics(rr))
    return all_metrics

def build_baseline(rest_windows_metrics: list[dict]) -> dict:
    # Robust-ish baseline: mean/std on available windows (small dataset)
    df = pd.DataFrame(rest_windows_metrics)
    baseline = {}
    for k in ["mean_hr", "sdnn", "rmssd"]:
        vals = df[k].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        baseline[k] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": float(np.std(vals, ddof=1)) if vals.size >= 2 else float("nan"),
        }
    baseline["n_windows"] = int(len(df))
    return baseline


def in_range(v: float, lo: float, hi: float) -> bool:
    if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return (v >= lo) and (v <= hi)


def evaluate_file(
    ecg_raw: np.ndarray,
    fs: int,
    sig_cfg: dict,
    r_cfg: dict,
    win_sec: int,
    overlap: float,
    baseline: dict,
    state: str,
    k_rest: float,
    k_active: float,
) -> dict:
    # Filter
    ecg = bandpass_filter(
        ecg_raw,
        fs=fs,
        low=float(sig_cfg["bandpass_low"]),
        high=float(sig_cfg["bandpass_high"]),
        order=int(sig_cfg.get("filter_order", 4)),
    )

    min_rr = float(r_cfg["min_rr_sec"])
    max_rr = float(r_cfg["max_rr_sec"])

    k = k_rest if state.lower() == "rest" else k_active

    # build allowed ranges from baseline
    ranges = {}
    for m in ["mean_hr", "sdnn", "rmssd"]:
        mu = baseline[m]["mean"]
        sd = baseline[m]["std"]
        if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
            # fallback: must be finite; otherwise fail any check
            ranges[m] = (float("nan"), float("nan"))
        else:
            ranges[m] = (mu - k * sd, mu + k * sd)

    total = 0
    passed = 0

    for w in sliding_windows(ecg, fs, win_sec, overlap):
        total += 1
        peaks = detect_r_peaks(w, fs, min_rr_sec=min_rr)
        rr = rr_intervals_seconds(peaks, fs)

        # 1) Hard physiological RR constraints
        if rr.size < 2:
            continue
        if np.any(rr < min_rr) or np.any(rr > max_rr):
            continue

        met = hrv_metrics(rr)

        # 2) Personalized baseline constraints
        ok = True
        for m in ["mean_hr", "sdnn", "rmssd"]:
            lo, hi = ranges[m]
            if not in_range(met[m], lo, hi):
                ok = False
                break

        if ok:
            passed += 1

    pass_rate = (passed / total) if total > 0 else 0.0
    return {
        "n_windows": total,
        "n_pass": passed,
        "pass_rate": float(pass_rate),
    }


# -----------------------------
# Main
# -----------------------------

