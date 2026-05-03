"""
standardize_csvs.py

Standardizes CSV files to the expected schema:
    state, sex, year, name, count

- Missing columns are added and populated with empty string ("") representing null
- Extra/unexpected columns are preserved at the end
- Uses numpy for fast in-memory column manipulation on large files

Usage:
    python standardize_csvs.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

EXPECTED_HEADERS = ["state", "sex", "year", "name", "count"]
NULL_VALUE = ""  # sentinel written to CSV for missing/null cells

INPUT_DIR  = Path("b_analysis/baby_names/data/raw")
OUTPUT_DIR = Path("b_analysis/baby_names/data/clean")

# Vectorized string stripping via numpy char module
_vstrip = np.vectorize(str.strip)


def _clean_state(col: np.ndarray) -> np.ndarray:
    """Strip whitespace, uppercase. Keep empty strings as null."""
    cleaned = np.char.upper(np.char.strip(col))
    return cleaned


def _clean_sex(col: np.ndarray) -> np.ndarray:
    """Strip whitespace, uppercase. Valid values: M, F, or empty."""
    return np.char.upper(np.char.strip(col))


def _clean_year(col: np.ndarray) -> np.ndarray:
    """Strip whitespace, convert float strings (1943.0) to int strings (1943).
    Leaves empty/null values as empty string."""
    result = np.char.strip(col)
    def _to_int_str(val: str) -> str:
        if val == NULL_VALUE:
            return NULL_VALUE
        try:
            return str(int(float(val)))
        except ValueError:
            return NULL_VALUE
    return np.vectorize(_to_int_str)(result)


def _clean_name(col: np.ndarray) -> np.ndarray:
    """Strip leading/trailing whitespace."""
    return np.char.strip(col)


def _clean_count(col: np.ndarray) -> np.ndarray:
    """Strip whitespace, cast float strings to int strings (601.0 -> 601).
    Leaves empty/null values as empty string."""
    result = np.char.strip(col)
    def _to_int_str(val: str) -> str:
        if val == NULL_VALUE:
            return NULL_VALUE
        try:
            return str(int(float(val)))
        except ValueError:
            return NULL_VALUE
    return np.vectorize(_to_int_str)(result)


# Map each expected column to its cleaning function
CLEANERS = {
    "state": _clean_state,
    "sex":   _clean_sex,
    "year":  _clean_year,
    "name":  _clean_name,
    "count": _clean_count,
}


def read_csv_numpy(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Read a CSV into a 2D numpy unicode array of shape [n_rows, n_cols].
    Uses the csv module for parsing so quoted fields and embedded
    commas are handled correctly without pandas.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        return headers, np.empty((0, len(headers)), dtype="U")

    # Pad short rows to match header length (handles malformed CSVs)
    n_cols = len(headers)
    padded = [
        r + [NULL_VALUE] * (n_cols - len(r)) if len(r) < n_cols else r
        for r in rows
    ]

    return headers, np.array(padded, dtype="U")


def standardize(
    headers: list[str], data: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """
    Reorder/add columns so EXPECTED_HEADERS come first, then clean each column:
      - state : strip whitespace, uppercase
      - sex   : strip whitespace, uppercase (M/F)
      - year  : strip, cast float string -> int string (1943.0 -> 1943)
      - name  : strip whitespace
      - count : strip, cast float string -> int string
    Missing columns are filled with NULL_VALUE. Extra columns are preserved as-is.
    Returns (new_headers, new_data).
    """
    n_rows = data.shape[0]
    col_index = {h: i for i, h in enumerate(headers)}

    output_cols: list[np.ndarray] = []
    for col in EXPECTED_HEADERS:
        if col in col_index:
            raw = data[:, col_index[col]]
        else:
            raw = np.full(n_rows, NULL_VALUE, dtype="U")
        # Apply the column-specific cleaner
        output_cols.append(CLEANERS[col](raw))

    # Append any extra columns that weren't in the expected set (no cleaning)
    extra_headers = [h for h in headers if h not in EXPECTED_HEADERS]
    for col in extra_headers:
        output_cols.append(data[:, col_index[col]])

    new_headers = EXPECTED_HEADERS + extra_headers

    if n_rows == 0:
        new_data = np.empty((0, len(new_headers)), dtype="U")
    else:
        new_data = (
            np.column_stack(output_cols) if output_cols
            else np.empty((n_rows, 0), dtype="U")
        )

    return new_headers, new_data


def write_csv_numpy(path: Path, headers: list[str], data: np.ndarray) -> None:
    """Write headers + numpy array back to CSV using the csv module."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        # tolist() converts numpy strings to native Python strings
        writer.writerows(data.tolist())


def process_file(input_path: Path, output_dir: Path) -> None:
    try:
        headers, data = read_csv_numpy(input_path)
    except Exception as e:
        print(f"  ERROR reading {input_path.name}: {e}", file=sys.stderr)
        return

    missing = [c for c in EXPECTED_HEADERS if c not in headers]
    extra   = [c for c in headers if c not in EXPECTED_HEADERS]

    new_headers, new_data = standardize(headers, data)

    output_path = output_dir / input_path.name
    write_csv_numpy(output_path, new_headers, new_data)

    status_parts = []
    if missing:
        status_parts.append(f"added null cols: {missing}")
    if extra:
        status_parts.append(f"kept extra cols: {extra}")
    if not status_parts:
        status_parts.append("already valid")

    print(
        f"  {input_path.name}  ->  {output_path}  "
        f"[{'; '.join(status_parts)}]  ({data.shape[0]:,} rows)"
    )


def main():
    if not INPUT_DIR.is_dir():
        print(f"ERROR: '{INPUT_DIR}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(INPUT_DIR.glob("*.txt"))
    if not paths:
        print(f"No CSV files found in '{INPUT_DIR}'.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Standardizing {len(paths)} file(s)...")
    for path in paths:
        process_file(path, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()