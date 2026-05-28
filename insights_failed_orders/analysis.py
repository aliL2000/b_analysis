import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
failed = orders[orders["order_status_key"].isin([4, 9])].copy()

# Step 1: Calculate Ratio of Failed Orders and Reasoning
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

# Step 2: Calculate Ratio of Failed Orders and Reasoning

failed["hour"] = pd.to_datetime(failed["order_datetime"], format="%H:%M:%S").dt.hour

# --- Pivot: count by hour x category ---
pivot = (
    failed.groupby(["hour", "failure_reason"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[
        "Cancelled before driver assigned",
        "Cancelled after driver assigned",
        "Rejected by system",
    ], fill_value=0)
)

# Ensure all 24 hours present
pivot = pivot.reindex(range(24), fill_value=0)

# --- Proportion (stacked 100%) ---
pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100

# ================================================================
# FIGURE 1 — Stacked absolute counts by hour
# ================================================================
colors = ["#4A90D9", "#E8A838", "#D94A4A"]
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor("#F7F7F5")

ax1 = axes[0]
ax1.set_facecolor("#F7F7F5")

bottom = np.zeros(24)
for col, color in zip(pivot.columns, colors):
    ax1.bar(pivot.index, pivot[col], bottom=bottom, color=color,
            label=col, width=0.75, edgecolor="white", linewidth=0.6)
    bottom += pivot[col].values

ax1.set_title("Failed orders by hour — absolute counts", fontsize=13, fontweight="600", pad=10)
ax1.set_xlabel("Hour of day", fontsize=10)
ax1.set_ylabel("Number of orders", fontsize=10)
ax1.set_xticks(range(24))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

# Annotate total on each bar
totals = pivot.sum(axis=1)
for h, total in totals.items():
    if total > 0:
        ax1.text(h, total + totals.max() * 0.01, str(int(total)),
                 ha="center", va="bottom", fontsize=7, color="#555")

# ================================================================
# FIGURE 2 — Stacked 100% proportion by hour
# ================================================================
ax2 = axes[1]
ax2.set_facecolor("#F7F7F5")

bottom_pct = np.zeros(24)
for col, color in zip(pivot_pct.columns, colors):
    vals = pivot_pct[col].fillna(0).values
    ax2.bar(pivot_pct.index, vals, bottom=bottom_pct, color=color,
            label=col, width=0.75, edgecolor="white", linewidth=0.6)
    bottom_pct += vals

ax2.set_title("Failed orders by hour — category proportion (%)", fontsize=13, fontweight="600", pad=10)
ax2.set_xlabel("Hour of day", fontsize=10)
ax2.set_ylabel("Proportion (%)", fontsize=10)
ax2.set_xticks(range(24))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.tight_layout(pad=3)
plt.savefig("b_analysis/insights_failed_orders/data/figures/failed_orders_by_hour.png", dpi=150, bbox_inches="tight")
plt.show()
