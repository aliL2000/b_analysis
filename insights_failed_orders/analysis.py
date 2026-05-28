import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Load data ---
orders = pd.read_csv("b_analysis/insights_failed_orders/data/raw/data_orders.csv")

# --- Categorise each failed order ---
def categorise(row):
    if row["order_status_key"] == 9:
        return "Rejected by system"
    elif row["order_status_key"] == 4:
        if row["is_driver_assigned_key"] == 1:
            return "Cancelled after driver assigned"
        else:
            return "Cancelled before driver assigned"
    return "Other"

orders["failure_reason"] = orders.apply(categorise, axis=1)

# Keep only failed orders (status 4 or 9)
failed = orders[orders["order_status_key"].isin([4, 9])].copy()

# --- Count by category ---
counts = (
    failed["failure_reason"]
    .value_counts()
    .reindex([
        "Cancelled before driver assigned",
        "Cancelled after driver assigned",
        "Rejected by system",
    ])
    .fillna(0)
    .astype(int)
)

total = counts.sum()
pcts = (counts / total * 100).round(1)

# --- Summary printout ---
print("\n=== Failure reason distribution ===")
for reason, count, pct in zip(counts.index, counts.values, pcts.values):
    print(f"  {reason:<40} {count:>6,}  ({pct:.1f}%)")
print(f"  {'TOTAL':<40} {total:>6,}")
print(f"\nHighest category: {counts.idxmax()} ({counts.max():,} orders)")