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
# Section 1 — Top 10 names nationally per year
# ---------------------------------------------------------------------------

def top_names_per_year(data: np.ndarray, output_dir: Path, top_n: int = 10) -> None:
    print("\n" + "=" * 50)
    print("2. TOP 10 NAMES NATIONALLY PER YEAR")
    print("=" * 50)

    # Aggregate: (year, name) -> total count
    agg: dict[tuple[str, str], int] = defaultdict(int)
    for row in data:
        year, name, count = row[2], row[3], row[4]
        if year == NULL or name == NULL or count == NULL:
            continue
        agg[(year, name)] += _int(count)

    # Group by year
    by_year: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (year, name), total in agg.items():
        by_year[year].append((name, total))

    years_sorted = sorted(by_year.keys(), key=lambda y: _int(y))

    # Print top 10 for each year
    for year in years_sorted:
        top = sorted(by_year[year], key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n  {year}: " + ", ".join(f"{n} ({c:,})" for n, c in top))

    # Plot: top 10 names across all years combined
    all_totals: dict[str, int] = defaultdict(int)
    for (_, name), total in agg.items():
        all_totals[name] += total
    top_overall = sorted(all_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names_plot, counts_plot = zip(*top_overall)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names_plot[::-1], counts_plot[::-1], color="steelblue")
    ax.set_xlabel("Total Count (all years)")
    ax.set_title(f"Top {top_n} Names Nationally (All Years Combined)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = output_dir / "top_names_overall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Section 2 — Year-over-year count trends for selected names
# ---------------------------------------------------------------------------

def yoy_trends(data: np.ndarray, output_dir: Path, selected_names: list[str]) -> None:
    print("\n" + "=" * 50)
    print("3. YEAR-OVER-YEAR TRENDS FOR SELECTED NAMES")
    print("=" * 50)

    # Aggregate: (name, year) -> total count
    agg: dict[tuple[str, str], int] = defaultdict(int)
    for row in data:
        year, name, count = row[2], row[3], row[4]
        if year == NULL or name == NULL or count == NULL:
            continue
        agg[(name, year)] += _int(count)

    fig, ax = plt.subplots(figsize=(12, 5))
    found_any = False

    for name in selected_names:
        entries = {_int(year): total for (n, year), total in agg.items() if n == name}
        if not entries:
            print(f"  '{name}': no data found, skipping.")
            continue
        years_s = sorted(entries.keys())
        counts_s = [entries[y] for y in years_s]
        ax.plot(years_s, counts_s, marker="o", markersize=3, label=name)
        found_any = True
        print(f"  '{name}': {len(years_s)} years of data, "
              f"peak {max(counts_s):,} in {years_s[counts_s.index(max(counts_s))]}")

    if found_any:
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Count")
        ax.set_title("Year-over-Year Name Trends")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend()
        plt.tight_layout()
        out = output_dir / "yoy_trends.png"
        fig.savefig(out, dpi=150)
        print(f"  Plot saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 3 — State-level heatmap of name frequency
# ---------------------------------------------------------------------------

def state_heatmap(data: np.ndarray, output_dir: Path) -> None:
    print("\n" + "=" * 50)
    print("4. STATE-LEVEL NAME FREQUENCY HEATMAP")
    print("=" * 50)

    # Aggregate total count per state
    state_totals: dict[str, int] = defaultdict(int)
    for row in data:
        state, count = row[0], row[4]
        if state == NULL or count == NULL:
            continue
        state_totals[state] += _int(count)

    if not state_totals:
        print("  No state data available.")
        return

    states_sorted = sorted(state_totals.keys())
    totals = [state_totals[s] for s in states_sorted]

    # Bar chart (heatmap-style colour mapping) as substitute for geo heatmap
    norm = plt.Normalize(min(totals), max(totals))
    colours = plt.cm.YlOrRd(norm(totals))

    fig, ax = plt.subplots(figsize=(max(8, len(states_sorted) * 0.4), 5))
    bars = ax.bar(states_sorted, totals, color=colours)
    ax.set_xlabel("State")
    ax.set_ylabel("Total Count")
    ax.set_title("Total Name Frequency by State")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Total Count")

    plt.tight_layout()
    out = output_dir / "state_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")

    # Print top 5 states
    top5 = sorted(state_totals.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 states by total count:")
    for state, total in top5:
        print(f"    {state}: {total:,}")


# ---------------------------------------------------------------------------
# Section 4 — Male vs. female name popularity over time
# ---------------------------------------------------------------------------

def sex_over_time(data: np.ndarray, output_dir: Path, selected_names: list[str]) -> None:
    print("\n" + "=" * 50)
    print("5. MALE VS. FEMALE POPULARITY OVER TIME")
    print("=" * 50)

    # Overall M vs F totals per year
    sex_year: dict[tuple[str, str], int] = defaultdict(int)
    for row in data:
        sex, year, count = row[1], row[2], row[4]
        if sex not in ("M", "F") or year == NULL or count == NULL:
            continue
        sex_year[(sex, year)] += _int(count)

    years_all = sorted({_int(y) for (_, y) in sex_year.keys()})
    m_counts = [sex_year.get(("M", str(y)), 0) for y in years_all]
    f_counts = [sex_year.get(("F", str(y)), 0) for y in years_all]

    fig, axes = plt.subplots(1 + len(selected_names), 1,
                              figsize=(12, 4 + 3 * len(selected_names)),
                              sharex=False)
    if len(selected_names) == 0:
        axes = [axes]

    # Panel 0: overall M vs F
    ax0 = axes[0]
    ax0.plot(years_all, m_counts, label="Male",   color="steelblue")
    ax0.plot(years_all, f_counts, label="Female", color="salmon")
    ax0.set_title("Overall Male vs. Female Counts per Year")
    ax0.set_ylabel("Total Count")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax0.legend()

    # One panel per selected name showing M vs F split
    name_sex_year: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in data:
        sex, year, name, count = row[1], row[2], row[3], row[4]
        if sex not in ("M", "F") or year == NULL or name == NULL or count == NULL:
            continue
        name_sex_year[(name, sex, year)] += _int(count)

    for i, name in enumerate(selected_names, start=1):
        ax = axes[i]
        years_n = sorted({_int(y) for (n, _, y) in name_sex_year if n == name})
        m = [name_sex_year.get((name, "M", str(y)), 0) for y in years_n]
        f = [name_sex_year.get((name, "F", str(y)), 0) for y in years_n]
        ax.plot(years_n, m, label="Male",   color="steelblue", marker="o", markersize=2)
        ax.plot(years_n, f, label="Female", color="salmon",    marker="o", markersize=2)
        ax.set_title(f"'{name}' — Male vs. Female over Time")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend()
        m_total = sum(m)
        f_total = sum(f)
        print(f"  '{name}': M total={m_total:,}  F total={f_total:,}  "
              f"({'more male' if m_total >= f_total else 'more female'})")

    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    out = output_dir / "sex_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")


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

    top_names_per_year(data, out_dir)
    yoy_trends(data, out_dir, args.names)
    state_heatmap(data, out_dir)
    sex_over_time(data, out_dir, args.names)

    print("\nEDA complete. Plots saved to:", out_dir)


if __name__ == "__main__":
    main()