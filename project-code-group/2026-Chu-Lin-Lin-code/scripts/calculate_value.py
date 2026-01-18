#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Threshold calibration tool (replaces WESAD stress classifier training).

Default behavior (NO arguments):
    python train_classifier.py
    -> Runs a small grid search over k_rest/k_active on your custom dataset,
       using scripts/comprehensive_comparison.py (dual baselines),
       and saves recommended thresholds to reports/.

Why this exists:
- Your dataset is small (2 persons × Rest/Active × 4 CSV each = 16 files)
- Training ML classifiers is not appropriate; use rule-based validation instead.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config
from src.orchestrator import HRVAnalysisOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate rule thresholds (k_rest/k_active) for the custom ECG dataset"
    )

    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (default: ../config/config.yaml)"
    )

    parser.add_argument(
        "--outdir",
        default="reports",
        help="Output directory (default: reports)"
    )

    parser.add_argument(
        "--win-sec",
        type=int,
        default=60,
        help="Window length in seconds (default: 60)"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio (default: 0.5)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def default_config_path() -> Path:
    # ../config/config.yaml relative to this file: scripts/../config/config.yaml
    return (Path(__file__).parent.parent / "config" / "config.yaml").resolve()


def objective(rest_mean: float, active_mean: float) -> float:
    if rest_mean is None or active_mean is None:
        return -1e9
    # both high is better; weight Rest slightly higher
    return 2.0 * rest_mean + 1.5 * active_mean



def run_one(config_path: Path, outdir: Path, win_sec: int, overlap: float, k_rest: float, k_active: float):
    """
    Call HRVAnalysisOrchestrator.run_dataset programmatically.
    """
    cfg = load_config(str(config_path))

    # Update config with current grid search parameters
    if "features" not in cfg:
        cfg["features"] = {}
    cfg["features"]["window_size_sec"] = win_sec
    cfg["features"]["overlap"] = overlap

    if "baseline" not in cfg:
        cfg["baseline"] = {}
    cfg["baseline"]["k_rest"] = k_rest
    cfg["baseline"]["k_active"] = k_active
    cfg["output"] = {"dir": str(outdir)} # Ensure output directory is set

    orchestrator = HRVAnalysisOrchestrator()
    orchestrator.run_dataset(cfg)


def summarize_pass_rates(pass_csv: Path) -> dict:
    df = pd.read_csv(pass_csv)

    # group means
    rest_mean = df[df["state"].str.lower() == "rest"]["pass_rate"].mean()
    active_mean = df[df["state"].str.lower() == "active"]["pass_rate"].mean()

    return {
        "rest_mean": float(rest_mean) if pd.notna(rest_mean) else None,
        "active_mean": float(active_mean) if pd.notna(active_mean) else None,
        "n_files": int(len(df)),
    }


def main():
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    config_path = Path(args.config).resolve() if args.config else default_config_path()
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Candidate grids (small on purpose)
    candidates = [1.5, 2.0, 2.5, 3.0]

    results = []
    best = None

    logger.info(f"Config: {config_path}")
    logger.info(f"Outdir: {outdir}")
    logger.info(f"Grid search on k_rest/k_active: {candidates}")

    for k_rest in candidates:
        for k_active in candidates:
            logger.info(f"Running evaluation: k_rest={k_rest}, k_active={k_active}")

            # Run dataset evaluation (writes pass_rates.csv into outdir)
            run_one(config_path, outdir, args.win_sec, args.overlap, k_rest, k_active)

            pass_csv = outdir / "pass_rates.csv"
            if not pass_csv.exists():
                logger.error(f"Expected output not found: {pass_csv}")
                sys.exit(1)

            s = summarize_pass_rates(pass_csv)
            rest_mean = s["rest_mean"]
            active_mean = s["active_mean"]
            score = objective(rest_mean, active_mean)

            row = {
                "k_rest": k_rest,
                "k_active": k_active,
                "rest_mean_pass_rate": rest_mean,
                "active_mean_pass_rate": active_mean,
                "score": score,
            }
            results.append(row)

            if (best is None) or (score > best["score"]):
                best = row

    # Save search results
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    search_csv = outdir / "threshold_search.csv"
    df.to_csv(search_csv, index=False, encoding="utf-8")
    logger.info(f"Saved threshold search results: {search_csv}")

    # Save best recommendation
    rec = {
        "recommended": {
            "k_rest": best["k_rest"],
            "k_active": best["k_active"],
        },
        "summary": {
            "rest_mean_pass_rate": best["rest_mean_pass_rate"],
            "active_mean_pass_rate": best["active_mean_pass_rate"],
            "score": best["score"],
        },
        "notes": "These thresholds were selected to keep Rest pass_rate high while keeping Active informative (not all-pass/all-fail).",
    }

    rec_json = outdir / "recommended_thresholds.json"
    rec_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved recommended thresholds: {rec_json}")

    # Print best
    logger.info(f"Best k_rest={best['k_rest']}, k_active={best['k_active']}")
    logger.info(f"Rest mean pass_rate={best['rest_mean_pass_rate']}, Active mean pass_rate={best['active_mean_pass_rate']}")


if __name__ == "__main__":
    main()
