#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualize ECG waveforms for custom dataset:
- 2 persons × (Rest/Active) × CSV files

Default (no args):
    python visualize_ecg_conditions.py
Reads config from ../config/config.yaml
Outputs figures to reports/figures/ecg_waveforms/
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logging, load_config


def _default_config_path() -> Path:
    return (REPO_ROOT / "config" / "config.yaml").resolve()


def _resolve_data_dir(cfg: dict) -> Path:
    candidates = [
        cfg.get("dataset", {}).get("data_dir"),
        cfg.get("data", {}).get("data_dir"),
        cfg.get("custom", {}).get("data_dir"),
        cfg.get("wesad", {}).get("data_dir"),
    ]

    bases = [
        REPO_ROOT,                  # .../project-code-group/2026-Chu-Lin-Lin-code
        REPO_ROOT.parent,           # .../project-code-group
        REPO_ROOT.parent.parent,    # .../course-agenticai-ecg-hrv
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

    # Common fallbacks
    for b in bases:
        for p in [(b / "data-group" / "data"), (b / "data-group")]:
            p = p.resolve()
            if p.exists():
                return p

    raise FileNotFoundError("Cannot find data directory (checked config + common fallbacks).")



def _resolve_sampling_rate(cfg: dict) -> float:
    return float(cfg.get("signal", {}).get("sampling_rate", 50))


def _infer_persons_states(data_dir: Path):
    persons = [p.name for p in data_dir.iterdir() if p.is_dir()]
    states = ["Rest", "Active"]
    return persons, states


def _pick_ecg_column(df: pd.DataFrame) -> str:
    # Prefer explicit ECG column name
    for name in ["ECG", "ecg", "Ecg"]:
        if name in df.columns:
            return name
    # Otherwise fall back to 4th column (A,B,C,D -> ECG is D)
    if df.shape[1] >= 4:
        return df.columns[3]
    raise ValueError(f"Cannot find ECG column. Columns={list(df.columns)}")


def _pick_time_column(df: pd.DataFrame):
    # Prefer Timestamp
    for name in ["Timestamp", "timestamp", "Time", "time"]:
        if name in df.columns:
            return name
    return None


def scan_csv_files(data_dir: Path, file_glob: str = "*.csv"):
    persons, states = _infer_persons_states(data_dir)
    records = []
    for pid in persons:
        for st in states:
            folder = data_dir / pid / st
            if not folder.exists():
                continue
            for f in sorted(folder.glob(file_glob)):
                records.append({"person": pid, "state": st, "path": f})
    return records


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ECG waveforms for custom dataset")
    p.add_argument("--config", "-c", default=None, help="Config path (default: ../config/config.yaml)")
    p.add_argument("--outdir", default="reports/figures/ecg_waveforms", help="Output dir (default: reports/figures/ecg_waveforms)")
    p.add_argument("--file-glob", default="*.csv", help="File pattern (default: *.csv)")
    p.add_argument("--max-seconds", type=float, default=None, help="Plot only first N seconds (optional)")
    p.add_argument("--downsample", type=int, default=5, help="Plot every N samples (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    cfg_path = Path(args.config).resolve() if args.config else _default_config_path()
    cfg = load_config(str(cfg_path))
    data_dir = _resolve_data_dir(cfg)
    fs = _resolve_sampling_rate(cfg)

    outdir = (REPO_ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {cfg_path}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Sampling rate: {fs} Hz")
    logger.info(f"Output dir: {outdir}")

    records = scan_csv_files(data_dir, args.file_glob)
    if not records:
        raise RuntimeError(f"No CSV files found under {data_dir} with pattern {args.file_glob}")

    for r in records:
        f = r["path"]
        df = pd.read_csv(f)

        ecg_col = _pick_ecg_column(df)
        ecg = df[ecg_col].astype(float).to_numpy()

        # time axis
        time_col = _pick_time_column(df)
        if time_col is not None:
            # try parse numeric seconds; if datetime, fallback to sample index
            try:
                t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
                if np.isfinite(t).sum() < len(t) * 0.8:
                    raise ValueError("timestamp not numeric enough")
                # normalize to start at 0
                t = t - t[0]
            except Exception:
                t = np.arange(len(ecg)) / fs
        else:
            t = np.arange(len(ecg)) / fs

        # truncate
        if args.max_seconds is not None:
            max_n = int(args.max_seconds * fs)
            ecg = ecg[:max_n]
            t = t[:max_n]

        # downsample for plotting
        ds = max(1, args.downsample)
        idx = np.arange(0, len(ecg), ds)

        plt.figure(figsize=(11.3, 7.87))  # A4 landscape-ish
        plt.plot(t[idx], ecg[idx], linewidth=0.8)
        plt.xlabel("Time (sec)")
        plt.ylabel(f"ECG ({ecg_col})")
        plt.title(f"{r['person']} | {r['state']} | {f.name}")
        plt.tight_layout()

        save_name = f"{r['person']}__{r['state']}__{f.stem}.png"
        out_path = outdir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()

        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
