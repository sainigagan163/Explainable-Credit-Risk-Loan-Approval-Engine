"""
Step 2 of the pipeline: Read raw tables from SQLite, engineer features,
and write the feature matrix back to the database.

Usage:
    python -m credit_risk_engine.src.02_features
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from credit_risk_engine.src.feature_utils import build_feature_matrix

DEFAULT_DB = str(
    Path(__file__).resolve().parent.parent / "database" / "credit_risk.db"
)


def load_table(engine, table_name: str) -> pd.DataFrame | None:
    """Load a table from SQLite; return None if it doesn't exist."""
    with engine.connect() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        ]
    if table_name not in tables:
        return None
    return pd.read_sql_table(table_name, engine)


def main(db_path: str = DEFAULT_DB) -> None:
    engine = create_engine(f"sqlite:///{db_path}")

    # Load required table
    app = load_table(engine, "application_train")
    if app is None:
        sys.exit("Table 'application_train' not found. Run 01_ingest.py first.")

    print(f"  Loaded application_train: {app.shape}")

    # Load optional enrichment tables
    bureau = load_table(engine, "bureau")
    if bureau is not None:
        print(f"  Loaded bureau: {bureau.shape}")

    prev = load_table(engine, "previous_application")
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

    # Write to database
    features.to_sql("features", engine, if_exists="replace", index=True)
    print(f"  Written 'features' table to {db_path}")

    # Summary stats
    if "target" in features.columns:
        target_counts = features["target"].value_counts()
        print(f"\n  Target distribution:")
        print(f"    0 (no default): {target_counts.get(0, 0):,}")
        print(f"    1 (default):    {target_counts.get(1, 0):,}")
        ratio = target_counts.get(1, 0) / len(features) * 100
        print(f"    Default rate:   {ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix from SQLite")
    parser.add_argument("--db-path", default=DEFAULT_DB, help="SQLite database path")
    args = parser.parse_args()
    main(args.db_path)
