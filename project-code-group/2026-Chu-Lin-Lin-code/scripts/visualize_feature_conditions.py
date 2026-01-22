#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualize windowed HRV features for custom dataset CSVs.

Default (no args):
    python visualize_feature_conditions.py
Reads config from ../config/config.yaml
Outputs figures to reports/figures/hrv_features/
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# repo root (local for output path resolution)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logging, load_config
from src.utils.helpers import (
    REPO_ROOT as UTILS_REPO_ROOT,
    resolve_data_dir,
    resolve_sampling_rate,
    infer_persons_states, # New import
    scan_csv_files,       # New import
)
from src.tools.signal_processor import process_signal
from src.tools.extended_features import extract_extended_features
from src.tools.ecg_loader import pick_ecg_column # New import


# Removed local _infer_persons_states and scan_csv_files


def window_slices(n: int, win: int, stride: int):
    start = 0
    while start + win <= n:
        yield start, start + win
        start += stride


def parse_args():
    p = argparse.ArgumentParser(description="Visualize HRV features (windowed) for custom dataset")
    p.add_argument("--config", "-c", default=None, help="Config path (default: ../config/config.yaml)")
    p.add_argument("--outdir", default="reports/figures/hrv_features", help="Output dir (default: reports/figures/hrv_features)")
    p.add_argument("--file-glob", default="*.csv", help="File pattern (default: *.csv)")
    p.add_argument("--win-sec", type=float, default=60.0, help="Window size seconds (default: 60)")
    p.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio (default: 0.5)")
    p.add_argument("--features", default="mean_hr,rmssd,sdnn", help="Comma-separated features to plot (default: mean_hr,rmssd,sdnn)")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    cfg_path = Path(args.config).resolve() if args.config else (UTILS_REPO_ROOT / "config" / "config.yaml").resolve() # Use UTILS_REPO_ROOT
    cfg = load_config(str(cfg_path))
    data_dir = resolve_data_dir(cfg) # Updated call
    fs = resolve_sampling_rate(cfg) # Updated call

    outdir = (REPO_ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    feats = [x.strip() for x in args.features.split(",") if x.strip()]

    logger.info(f"Config: {cfg_path}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Sampling rate: {fs} Hz")
    logger.info(f"Output dir: {outdir}")
    logger.info(f"Features: {feats}")

    records = scan_csv_files(data_dir, args.file_glob)
    if not records:
        raise RuntimeError(f"No CSV files found under {data_dir} with pattern {args.file_glob}")

    win = int(args.win_sec * fs)
    stride = max(1, int(win * (1.0 - args.overlap)))

    for r in records:
        f = r["path"]
        df = pd.read_csv(f)

        ecg_col = pick_ecg_column(df) # Updated call
        ecg = df[ecg_col].astype(float).to_numpy()
        n = len(ecg)

        times = []
        feat_series = {k: [] for k in feats}

        for s, e in window_slices(n, win, stride):
            seg = ecg[s:e]
            center_t = (s + e) / 2.0 / fs

            # process -> rr
            try:
                processed = process_signal({"signal": seg, "sampling_rate": fs})
                rr = processed.get("rr_intervals", np.array([]))
            except Exception:
                rr = np.array([])

            if rr is None or len(rr) < 5:
                # too few beats; mark NaN
                for k in feats:
                    feat_series[k].append(np.nan)
                times.append(center_t)
                continue

            # extract features
            try:
                F = extract_extended_features(rr, fs=fs)
            except Exception:
                F = {}

            for k in feats:
                feat_series[k].append(float(F.get(k, np.nan)))
            times.append(center_t)

        times = np.array(times)

        plt.figure(figsize=(11.3, 7.87))
        for k in feats:
            plt.plot(times, np.array(feat_series[k]), marker="o", linewidth=1.2, markersize=3, label=k)

        plt.xlabel("Time (sec)")
        plt.ylabel("Feature value")
        plt.title(f"{r['person']} | {r['state']} | {f.name} (win={args.win_sec}s, overlap={args.overlap})")
        plt.legend()
        plt.tight_layout()

        save_name = f"{r['person']}__{r['state']}__{f.stem}__features.png"
        out_path = outdir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()

        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
