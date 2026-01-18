#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Summarize pass rates produced by comprehensive_comparison.py

Outputs:
- 16 file-level pass rates
- Aggregated pass rate by person and by state
- Quick "who is failing" view
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="reports/pass_rates.csv", help="Path to pass_rates.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    df = pd.read_csv(p)

    print("=" * 70)
    print("FILE-LEVEL PASS RATES (should be 16 rows)")
    print("=" * 70)
    print(df[["person_id", "state", "file", "pass_rate", "n_windows", "n_pass"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("AGGREGATION BY PERSON")
    print("=" * 70)
    g_person = df.groupby("person_id")["pass_rate"].agg(["count", "mean", "min", "max"]).reset_index()
    print(g_person.to_string(index=False))

    print("\n" + "=" * 70)
    print("AGGREGATION BY PERSON × STATE")
    print("=" * 70)
    g_ps = df.groupby(["person_id", "state"])["pass_rate"].agg(["count", "mean", "min", "max"]).reset_index()
    print(g_ps.to_string(index=False))

    print("\n" + "=" * 70)
    print("WORST 5 FILES")
    print("=" * 70)
    worst = df.sort_values("pass_rate").head(5)
    print(worst[["person_id", "state", "file", "pass_rate", "n_windows", "n_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
