#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Command-line interface for HRV Analysis Agent.

Default behavior (NO arguments):
    python run_analysis.py
    -> Run dataset evaluation using ../config/config.yaml
       (dual baseline, Rest/Active, pass_rate for all CSVs)

Single-file mode (optional, legacy):
    python run_analysis.py --input ecg.txt --output report.pdf
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import HRVAnalysisOrchestrator
from src.utils import setup_logging, load_config


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HRV Analysis Agent - Dataset Runner")

    parser.add_argument(
        "--config", "-c",
        default=str(Path(__file__).parent.parent / "config" / "config.yaml"),
        help="Path to configuration file (.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()



# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    # Load configuration (make sure load_config uses utf-8-sig)
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info("Initializing HRV Analysis Agent...")
    orchestrator = HRVAnalysisOrchestrator()

    try:
        result = orchestrator.run_dataset(config)
        logger.info(f"Dataset analysis complete: {result}")
        print(f"\n[OK] Output dir: {result.get('outdir')}")
        print(f"[OK] Files processed: {result.get('n_files')}")
        print(f"[OK] Wrote: pass_rates.csv, baselines.json, per_file/*.json")
    except Exception as e:
        logger.error(f"Error during dataset analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()
