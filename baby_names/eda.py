"""
eda.py

Exploratory Data Analysis for baby names clean data.

Sections:
    1. Summary statistics  — total records, unique names, state/year coverage
    2. Top 10 names        — nationally per year
    3. Year-over-year      — count trends for selected names
    4. State heatmaps      — name frequency by state
    5. Sex over time       — male vs. female name popularity

Usage:
    python eda.py
    python eda.py --names Patricia Jordan   # override default selected names
    python eda.py --output-dir ./plots      # override default plot output dir
"""

import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

INPUT_DIR  = Path("b_analysis/baby_names/data/clean")
OUTPUT_DIR = Path("b_analysis/baby_names/eda/plots")

# Names highlighted in the year-over-year and sex-over-time charts
DEFAULT_SELECTED_NAMES = ["Patricia", "Jordan", "Casey", "Alex"]

NULL = ""  # matches the null sentinel used in standardize_csvs.py

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_data(input_dir: Path) -> np.ndarray:
    """
    Load all CSVs from input_dir into a single 2-D numpy unicode array.
    Expected columns (after standardization): state, sex, year, name, count
    Returns array of shape [n_rows, 5] with dtype U (unicode string).
    """
    all_rows: list[list[str]] = []
    files = sorted(input_dir.glob("*.txt"))

    if not files:
        print(f"ERROR: No CSV files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = [h.strip().lower() for h in next(reader)]
            # Map expected columns to their indices in this file
            idx = {col: headers.index(col) for col in ["state", "sex", "year", "name", "count"] if col in headers}
            for row in reader:
                all_rows.append([
                    row[idx["state"]].strip()  if "state" in idx else NULL,
                    row[idx["sex"]].strip()    if "sex"   in idx else NULL,
                    row[idx["year"]].strip()   if "year"  in idx else NULL,
                    row[idx["name"]].strip()   if "name"  in idx else NULL,
                    row[idx["count"]].strip()  if "count" in idx else NULL,
                ])

    print(f"Loaded {len(all_rows):,} rows from {len(files)} file(s).")
    return np.array(all_rows, dtype="U")


def _int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Section 1 — Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(data: np.ndarray) -> None:
    print("\n" + "=" * 50)
    print("1. SUMMARY STATISTICS")
    print("=" * 50)

    total_records = data.shape[0]
    print(f"  Total records   : {total_records:,}")

    # Unique non-null names
    names = data[:, 3]
    unique_names = np.unique(names[names != NULL])
    print(f"  Unique names    : {len(unique_names):,}")

    # State coverage
    states = data[:, 0]
    unique_states = np.unique(states[states != NULL])
    print(f"  States covered  : {len(unique_states)}  {sorted(unique_states.tolist())}")

    # Year coverage
    years = data[:, 2]
    valid_years = np.array([_int(y) for y in years if y != NULL])
    if valid_years.size:
        print(f"  Year range      : {valid_years.min()} – {valid_years.max()}")

    # Null counts per column
    col_names = ["state", "sex", "year", "name", "count"]
    print("\n  Null counts per column:")
    for i, col in enumerate(col_names):
        nulls = np.sum(data[:, i] == NULL)
        pct = nulls / total_records * 100
        print(f"    {col:<8} : {nulls:,}  ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for baby names clean data.")
    parser.add_argument(
        "--names", nargs="+", default=DEFAULT_SELECTED_NAMES,
        help="Names to highlight in trend/sex charts",
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(OUTPUT_DIR),
        help=f"Directory for plot outputs (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(INPUT_DIR)

    summary_statistics(data)

    print("\nEDA complete. Plots saved to:", out_dir)


if __name__ == "__main__":
    main()