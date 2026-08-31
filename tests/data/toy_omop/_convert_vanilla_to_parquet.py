"""Script to convert vanilla OMOP CSV files to Parquet format.

This script is run once to create the Parquet test data.
Run from the tests directory:
    python data/toy_omop/_convert_vanilla_to_parquet.py
"""

from pathlib import Path

import pandas as pd


def convert_vanilla_to_parquet():
    """Convert all CSV files in vanilla directory to Parquet format in vanilla_parquet directory."""
    vanilla_path = Path(__file__).parent / "vanilla"
    parquet_path = Path(__file__).parent / "vanilla_parquet"

    # Create parquet directory if it doesn't exist
    parquet_path.mkdir(exist_ok=True)

    # Convert each CSV file to Parquet
    for csv_file in vanilla_path.glob("*.csv"):
        # Read CSV
        df = pd.read_csv(csv_file)

        # Write as Parquet
        parquet_file = parquet_path / f"{csv_file.stem}.parquet"
        df.to_parquet(parquet_file, index=False)

        print(f"Converted {csv_file.name} -> {parquet_file.name}")

    print(f"\nConversion complete! Parquet files saved to: {parquet_path}")


if __name__ == "__main__":
    convert_vanilla_to_parquet()
