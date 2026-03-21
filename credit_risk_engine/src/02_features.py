"""
Step 2 of the pipeline: Read CSV files from local data directory,
engineer features, and save the feature matrix as a CSV.

Usage:
    python -m credit_risk_engine.src.02_features
    python -m credit_risk_engine.src.02_features --data-dir credit_risk_engine/data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from credit_risk_engine.src.feature_utils import build_feature_matrix

DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent / "data"
)
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent / "data"
)


def load_csv(data_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a CSV file; return None if it doesn't exist."""
    csv_path = data_dir / f"{filename}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main(data_dir: str = DEFAULT_DATA_DIR, output_dir: str | None = None) -> None:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load required table
    app = load_csv(data_dir, "application_train")
    if app is None:
        sys.exit(
            f"Required file missing: {data_dir}/application_train.csv\n"
            "Download from https://www.kaggle.com/c/home-credit-default-risk"
        )

    print(f"  Loaded application_train: {app.shape}")

    # Load optional enrichment tables
    bureau = load_csv(data_dir, "bureau")
    if bureau is not None:
        print(f"  Loaded bureau: {bureau.shape}")

    prev = load_csv(data_dir, "previous_application")
    if prev is not None:
        print(f"  Loaded previous_application: {prev.shape}")

    # Build feature matrix
    features = build_feature_matrix(app, bureau=bureau, prev=prev)
    print(f"  Feature matrix: {features.shape}")

    # Handle missing values — fill with median for numeric columns
    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].fillna(
        features[numeric_cols].median()
    )

    # Save as CSV
    output_path = output_dir / "features.csv"
    features.to_csv(output_path, index=True)
    print(f"  Written feature matrix to {output_path}")

    # Summary stats
    if "target" in features.columns:
        target_counts = features["target"].value_counts()
        print(f"\n  Target distribution:")
        print(f"    0 (no default): {target_counts.get(0, 0):,}")
        print(f"    1 (default):    {target_counts.get(1, 0):,}")
        ratio = target_counts.get(1, 0) / len(features) * 100
        print(f"    Default rate:   {ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix from local CSVs")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Path to CSV data directory"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory for output (defaults to data-dir)"
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
