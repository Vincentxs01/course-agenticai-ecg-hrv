# SPDX-License-Identifier: Apache-2.0
"""
HRV-based Stress Classifier using ECG Input Data.

This module implements an orchestrator that coordinates an HRV (Heart Rate Variability)
analysis pipeline for stress detection from ECG signals.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import numpy as np

from .tools import (
    process_signal,
    extract_extended_features,
    generate_report
)
from .tools.ecg_loader import read_ecg_csv_column # New import

class HRVAnalysisOrchestrator:
    """
    HRV Analysis Orchestrator.

    This orchestrator manages the complete pipeline from raw ECG data to
    baseline construction and pass/fail evaluation for stress detection.
    It is designed for batch processing of datasets.
    """

    def __init__(self):
        """
        Initialize the HRV stress classifier orchestrator.
        """
        self.state = {}
        self.execution_log = []

    def _scan_dataset_from_config(self, data_dir: Path, persons_cfg: list[dict]):
        records = []

        for p in persons_cfg:
            pid = p["id"]
            conditions = p.get("conditions", {})

            for state, cfg in conditions.items():
                pattern = cfg.get("glob")
                if not pattern:
                    continue

                for f in sorted((data_dir).glob(pattern)):
                    records.append({
                        "person": pid,
                        "state": state,
                        "path": f
                    })

        return records

    def _window_slices(self, n: int, win: int, stride: int):
        s = 0
        while s + win <= n:
            yield s, s + win
            s += stride

    def _window_metrics(self, ecg_seg: np.ndarray, fs: int, filter_low: float, filter_high: float):
        ecg_data = {"signal": ecg_seg, "sampling_rate": fs}
        processed = process_signal(ecg_data, filter_low=filter_low, filter_high=filter_high)

        rr = processed.get("rr_intervals", None)
        if rr is None:
            return None
        if len(rr) < 2:   # baseline 可以放寬
            return None

        feats = extract_extended_features(
            rr,
            r_peaks=processed.get("r_peaks"),
            filtered_signal=processed.get("filtered_signal"),
            fs=fs
        )

        # 你之後檢查用到的最小集合（可以再加）
        out = {
            "rr": np.array(rr, dtype=float),
            "mean_hr_bpm": float(processed.get("mean_hr_bpm", np.nan)),
            "sdnn": float(feats.get("sdnn", np.nan)),
            "rmssd": float(feats.get("rmssd", np.nan)),
            "lf_hf_ratio": float(feats.get("lf_hf_ratio", np.nan)),
        }
        return out

    def _fit_baseline(self, rows: list[dict]) -> dict:
            """
            rows: list of per-window metrics dicts (from _window_metrics)
            """
            rr_means = []
            sdnn = []
            rmssd = []

            for r in rows:
                rr = np.array(r["rr"], dtype=float)

                # ms -> sec if needed
                if np.nanmean(rr) > 10:   # e.g., 600~1200 means ms
                    rr = rr / 1000.0

                rr_means.append(float(np.nanmean(rr)))
                sdnn.append(float(r["sdnn"]))
                rmssd.append(float(r["rmssd"]))

            def stat(x):
                x = np.array(x, dtype=float)
                x = x[np.isfinite(x)]
                return {
                    "mean": float(np.mean(x)) if len(x) else np.nan,
                    "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                }

            return {
                "rr_mean": stat(rr_means),
                "sdnn": stat(sdnn),
                "rmssd": stat(rmssd),
            }


    def _in_range(self, x: float, mu: float, sd: float, k: float) -> bool:
        if not np.isfinite(x) or not np.isfinite(mu):
            return False
        if not np.isfinite(sd) or sd == 0:
            # sd 不足時退化成「只比對 mu」：允許 ±10%（你可改）
            return (abs(x - mu) / (abs(mu) + 1e-9)) <= 0.10
        return (mu - k*sd) <= x <= (mu + k*sd)


    def _window_pass(self, m: dict, base: dict, k: float, rr_min: float, rr_max: float) -> bool:
        """
        rr_min/rr_max: physiological constraints from config (seconds)
        """
        
        rr = np.array(m["rr"], dtype=float)
        if np.nanmean(rr) > 10:
            rr = rr / 1000.0
        rr_mean = float(np.nanmean(rr))

        # 先做生理範圍（硬限制）
        if not (rr_min <= rr_mean <= rr_max):
            return False

        ok_rr = self._in_range(rr_mean, base["rr_mean"]["mean"], base["rr_mean"]["std"], k)
        ok_sdnn = self._in_range(m["sdnn"], base["sdnn"]["mean"], base["sdnn"]["std"], k)
        ok_rmssd = self._in_range(m["rmssd"], base["rmssd"]["mean"], base["rmssd"]["std"], k)

        # 你可以調整規則：這裡用「三個都要過」
        return bool(ok_rr and ok_sdnn and ok_rmssd)

    def run_dataset(self, config: dict) -> dict:
        # ---- config ----
        dataset_cfg = config.get("dataset")

        if dataset_cfg is None:
            raise KeyError("config.yaml missing top-level 'dataset:' section")

        data_dir = Path(dataset_cfg["data_dir"]).expanduser()
        if not data_dir.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            data_dir = (repo_root / data_dir).resolve()

        persons_cfg = dataset_cfg.get("persons", [])
        if not persons_cfg:
            raise ValueError("dataset.persons is empty")

        # persons list
        persons = [p["id"] for p in persons_cfg]

        # states list (from dataset.conditions)
        states = set()
        for p in persons_cfg:
            states.update(p.get("conditions", {}).keys())
        states = sorted(states)

        fs = int(config["signal"]["sampling_rate"])
        filter_low = float(config["signal"].get("bandpass_low", 0.5))
        filter_high = float(config["signal"].get("bandpass_high", 20.0))

        win_sec = float(config["features"].get("window_size_sec", 60))
        overlap = float(config["features"].get("overlap", 0.5))
        win = int(win_sec * fs)
        stride = max(1, int(win * (1.0 - overlap)))

        rr_min = float(config["r_peak"].get("min_rr_sec", 0.3))
        rr_max = float(config["r_peak"].get("max_rr_sec", 2.0))

        k_rest = float(config.get("baseline", {}).get("k_rest", 2.5))
        k_active = float(config.get("baseline", {}).get("k_active", 2.0))

        outdir = Path(config.get("output", {}).get("dir", "reports"))
        if not outdir.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            outdir = (repo_root / outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        # ---- scan files ----
        records = self._scan_dataset_from_config(data_dir, persons_cfg)

        if not records:
            raise RuntimeError(f"No CSV files found under {data_dir} using dataset config")


        # ---- 1) Build baselines per person/state ----
        baselines = {}  # baselines[person][state] = baseline dict
        for pid in persons:
            baselines[pid] = {}
            for st in states:
                # collect windows from all files of that person/state
                win_rows = []
                for rec in records:
                    if rec["person"] != pid or rec["state"] != st:
                        continue
                    sig = read_ecg_csv_column(rec["path"]) # Updated call
                    for s, e in self._window_slices(len(sig), win, stride):
                        m = self._window_metrics(sig[s:e], fs, filter_low, filter_high)
                        if m is not None:
                            win_rows.append(m)

                baselines[pid][st] = self._fit_baseline(win_rows)

        # ---- save baselines ----
        (outdir / "baselines.json").write_text(
            json.dumps(baselines, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # ---- 2) Evaluate each file against its own (person,state) baseline ----
        rows = []
        per_file_dir = outdir / "per_file"
        per_file_dir.mkdir(exist_ok=True)

        for rec in records:
            pid, st, fpath = rec["person"], rec["state"], rec["path"]
            base = baselines[pid][st]
            k = k_rest if st.lower() == "rest".lower() else k_active

            sig = read_ecg_csv_column(fpath) # Updated call

            n_win = 0
            n_pass = 0
            win_details = []

            for s, e in self._window_slices(len(sig), win, stride):
                m = self._window_metrics(sig[s:e], fs, filter_low, filter_high)
                if m is None:
                    continue
                n_win += 1
                ok = self._window_pass(m, base, k=k, rr_min=rr_min, rr_max=rr_max)
                n_pass += int(ok)

                win_details.append({
                    "start_sec": s / fs,
                    "end_sec": e / fs,
                    "pass": bool(ok),
                    "rr_mean": float(np.mean(m["rr"])),
                    "sdnn": m["sdnn"],
                    "rmssd": m["rmssd"],
                })

            pass_rate = (n_pass / n_win) if n_win > 0 else 0.0

            rows.append({
                "person": pid,
                "state": st,
                "file": str(fpath),
                "pass_rate": pass_rate,
                "n_windows": n_win,
                "n_pass": n_pass
            })

            # per-file detail json (optional but useful)
            detail = {
                "person": pid,
                "state": st,
                "file": str(fpath),
                "k_used": k,
                "window_size_sec": win_sec,
                "overlap": overlap,
                "n_windows": n_win,
                "n_pass": n_pass,
                "pass_rate": pass_rate,
                "windows": win_details,
            }
            (per_file_dir / f"{pid}__{st}__{Path(fpath).stem}.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # ---- save pass_rates.csv ----
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["person", "state", "pass_rate"])
        df.to_csv(outdir / "pass_rates.csv", index=False, encoding="utf-8-sig")

        return {"status": "success", "outdir": str(outdir), "n_files": len(rows)}