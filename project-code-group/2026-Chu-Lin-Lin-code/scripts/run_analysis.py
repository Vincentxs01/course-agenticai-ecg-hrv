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

# New imports for report generation
import pandas as pd
from src.tools.report_generator import generate_report
from src.tools.ecg_loader import read_ecg_csv_column
from src.tools.signal_processor import process_signal
from src.tools.extended_features import extract_extended_features
from src.utils.helpers import resolve_data_dir, resolve_sampling_rate

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

        # --- Generate Overall Analysis Report ---
        output_dir = Path(result["outdir"])
        pass_rates_path = output_dir / "pass_rates.csv"
        
        if pass_rates_path.exists():
            df_pass_rates = pd.read_csv(pass_rates_path)
            
            # Calculate overall pass rate (e.g., mean of all file pass rates)
            overall_pass_rate = df_pass_rates["pass_rate"].mean()
            
            # Create a simple summary
            if overall_pass_rate >= 0.95:
                evaluation_summary = "The system found high consistency with personalized baselines across all analyzed files."
            elif overall_pass_rate >= 0.70:
                evaluation_summary = "The system found moderate consistency with personalized baselines; some deviations were noted."
            else:
                evaluation_summary = "The system indicated low consistency with personalized baselines; significant deviations were observed."
            
            # --- Get representative data for the report ---
            # This is a simplification; in a real scenario, you might pick a representative file
            # or process a summary of all files.
            
            # Find the correct data_dir based on config.
            data_dir_resolved = resolve_data_dir(config)
            
            # Find one CSV file path from the records scanned by the orchestrator
            all_records = orchestrator._scan_dataset_from_config(data_dir_resolved, config["dataset"]["persons"])
            
            if all_records:
                representative_file_path = all_records[0]["path"]
                fs = int(config["signal"]["sampling_rate"]) # Get sampling rate from config
                
                # Load raw ECG signal
                raw_ecg_signal = read_ecg_csv_column(representative_file_path)
                
                ecg_data_for_report = {
                    "signal": raw_ecg_signal,
                    "sampling_rate": fs,
                    "duration_sec": len(raw_ecg_signal) / fs,
                    "n_samples": len(raw_ecg_signal),
                    "file_path": str(representative_file_path)
                }
                
                # Process signal
                processed_for_report = process_signal(ecg_data_for_report, 
                                                        filter_low=float(config["signal"].get("bandpass_low", 0.5)),
                                                        filter_high=float(config["signal"].get("bandpass_high", 20.0)))
                
                # Extract features
                features_for_report = extract_extended_features(processed_for_report["rr_intervals"])
                
                report_output_path = output_dir / "overall_analysis_report.md"
                
                logger.info(f"Generating overall analysis report to: {report_output_path}")
                generated_report_path = generate_report(
                    ecg_data=ecg_data_for_report,
                    processed=processed_for_report,
                    features=features_for_report,
                    pass_rate=overall_pass_rate,
                    evaluation_summary=evaluation_summary,
                    output_path=report_output_path,
                    include_ai_interpretation=False # Assume no ANTHROPIC_API_KEY
                )
                print(f"\n[OK] Overall analysis report generated at: {generated_report_path}")
            else:
                logger.warning("No records found from orchestrator scan to generate representative report data.")
        else:
            logger.warning(f"pass_rates.csv not found at {pass_rates_path}, skipping overall report generation.")
                
    except Exception as e:
        logger.error(f"Error during dataset analysis or report generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()