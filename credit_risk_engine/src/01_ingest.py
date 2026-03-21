"""
Step 1 of the pipeline: Ingest Home Credit CSV files into a local SQLite database.

Usage:
    python -m credit_risk_engine.src.01_ingest --data-dir credit_risk_engine/data
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Tables we expect from the Home Credit dataset
REQUIRED_TABLES = ["application_train"]
OPTIONAL_TABLES = [
    "bureau",
    "previous_application",
    "credit_card_balance",
    "installments_payments",
    "POS_CASH_balance",
]


def csv_path_for(table_name: str, data_dir: Path) -> Path:
    return data_dir / f"{table_name}.csv"


def ingest_table(
    engine, table_name: str, csv_file: Path, chunksize: int = 50_000
) -> int:
    """Read a CSV in chunks and write it into SQLite. Returns row count."""
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        chunk.to_sql(
            table_name,
            engine,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )
        total_rows += len(chunk)
    return total_rows


def main(data_dir: str, db_path: str | None = None) -> None:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    if db_path is None:
        db_path = str(
            Path(__file__).resolve().parent.parent / "database" / "credit_risk.db"
        )

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    # Check required tables
    for table in REQUIRED_TABLES:
        csv_file = csv_path_for(table, data_dir)
        if not csv_file.exists():
            sys.exit(
                f"Required file missing: {csv_file}\n"
                "Download from https://www.kaggle.com/c/home-credit-default-risk"
            )

    # Ingest all available tables
    all_tables = REQUIRED_TABLES + OPTIONAL_TABLES
    for table in all_tables:
        csv_file = csv_path_for(table, data_dir)
        if not csv_file.exists():
            print(f"  SKIP  {table} (file not found)")
            continue
        row_count = ingest_table(engine, table, csv_file)
        print(f"  OK    {table}: {row_count:,} rows")

    # Verify
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    print(f"\nDatabase ready at {db_path} with tables: {[t[0] for t in tables]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Home Credit CSVs into SQLite")
    parser.add_argument(
        "--data-dir",
        default="credit_risk_engine/data",
        help="Path to directory containing CSV files",
    )
    parser.add_argument("--db-path", default=None, help="SQLite database output path")
    args = parser.parse_args()
    main(args.data_dir, args.db_path)
