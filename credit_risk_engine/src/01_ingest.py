"""
Step 1 of the pipeline: Validate that required Home Credit CSV files
are present in the local data directory and print a summary.

Usage:
    python -m credit_risk_engine.src.01_ingest
    python -m credit_risk_engine.src.01_ingest --data-dir credit_risk_engine/data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ["application_train"]
OPTIONAL_FILES = [
    "bureau",
    "previous_application",
    "credit_card_balance",
    "installments_payments",
    "POS_CASH_balance",
]

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")


def main(data_dir: str = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    # Check required files
    for name in REQUIRED_FILES:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            sys.exit(
                f"Required file missing: {csv_path}\n"
                "Download from https://www.kaggle.com/c/home-credit-default-risk"
            )

    # Summarise all available files
    all_files = REQUIRED_FILES + OPTIONAL_FILES
    found = []
    for name in all_files:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"  SKIP  {name}.csv (not found)")
            continue
        df = pd.read_csv(csv_path, nrows=5)
        row_count = sum(1 for _ in open(csv_path)) - 1  # exclude header
        print(f"  OK    {name}.csv: ~{row_count:,} rows, {len(df.columns)} columns")
        found.append(name)

    print(f"\nData directory ready: {data_dir}")
    print(f"  Found {len(found)}/{len(all_files)} files: {found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Home Credit CSV files")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Path to directory containing CSV files",
    )
    args = parser.parse_args()
    main(args.data_dir)
