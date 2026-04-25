# Baby Names Data Exploration (1950–1980)


## Project Overview

This project performs exploratory data analysis (EDA) on a dataset of baby names registered across all 50 US states between **1950 and 1980**. The goal is to uncover naming trends, regional preferences, and cultural shifts reflected in baby name popularity over this 30-year period — spanning the post-war baby boom through to the cultural changes of the late 1970s.

---

## Dataset

| Field   | Description                                      |
|---------|--------------------------------------------------|
| `state` | US state abbreviation (e.g. `CA`, `TX`, `NY`)   |
| `sex`   | Recorded sex at birth (`M` / `F`)                |
| `year`  | Year of birth (1950–1980)                        |
| `name`  | Given first name                                 |
| `count` | Number of babies registered with that name       |

**Coverage:** All 50 US states  
**Time range:** 1950–1980 (31 years)

> **Note:** Files are provided per state/year and may have missing columns, which are filled with appropriate null values during ingestion (`''` for string fields, `'0'` for `count`).

---

## Objectives

1. **Trend Analysis** — Identify names that rose or fell in popularity over the three decades.
2. **Regional Variation** — Compare naming preferences across states and regions (e.g. Northeast vs. South vs. Midwest).
3. **Gender Distribution** — Explore how name popularity differs between male and female registrations.
4. **Decade Snapshots** — Characterise the most popular names per decade (1950s, 1960s, 1970s).
5. **Unique Names** — Track the emergence of rare or unique names over time.
6. **Baby Boom Signal** — Detect the post-WWII baby boom (late 1940s–1960s) in birth registration counts.

---

## Project Structure

```
baby-names-exploration/
│
├── data/
│   ├── raw/                  # Original per-state CSV files
│   └── processed/            # Cleaned, concatenated dataset
│
├── notebooks/
│   ├── 01_ingestion.ipynb    # Data loading, null-filling, concatenation
│   ├── 02_cleaning.ipynb     # Deduplication, type casting, validation
│   ├── 03_eda.ipynb          # Core exploratory analysis
│   └── 04_visualisation.ipynb# Charts and figures
│
├── src/
│   ├── load.py               # File ingestion pipeline (numpy-based)
│   └── utils.py              # Helper functions
│
├── outputs/
│   └── figures/              # Saved plots and charts
│
├── requirements.txt
└── README.md
```

---

## Pipeline Summary

### 1. Ingestion (`01_ingestion.ipynb`)

- Iterate over all state CSV files using `np.loadtxt`
- Detect and fill missing columns with typed nulls
- Strip duplicate headers from multi-file concatenation
- Reorder columns to canonical schema: `[state, sex, year, name, count]`
- Concatenate all chunks into a single `final_data` array

### 2. Cleaning (`02_cleaning.ipynb`)

- Cast `year` and `count` to numeric types
- Normalise `name` casing (title case)
- Validate `state` values against known 50-state list
- Filter to year range **1950–1980**
- Remove rows with null `name` or invalid `count`

### 3. EDA (`03_eda.ipynb`)

- Summary statistics: total records, unique names, state/year coverage
- Top 10 names nationally per year
- Year-over-year count trends for selected names
- State-level heatmaps of name frequency
- Male vs. female name popularity over time

### 4. Visualisation (`04_visualisation.ipynb`)

- Line charts: popularity of top names over time
- Choropleth maps: regional name dominance by state
- Bar charts: top names per decade
- Bump charts: rank changes of top-20 names (1950 vs. 1980)

---

## Key Questions to Answer

- What were the **top 5 names** for boys and girls in each decade?
- Which names experienced the **sharpest decline** after 1960?
- Are there names that are popular in one region but virtually absent in others?
- How did the **total number of registered births** change year-over-year?
- Which names crossed gender lines — popular for both `M` and `F`?

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
plotly
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Expected Outputs

| Output | Description |
|--------|-------------|
| `processed/final_data.csv` | Clean, merged dataset ready for analysis |
| `figures/top_names_by_decade.png` | Bar chart of top names per decade |
| `figures/national_trend_lines.png` | Line chart of top-10 name trends 1950–1980 |
| `figures/state_heatmap.html` | Interactive choropleth by state |
| `figures/rank_bump_chart.png` | Name rank changes over the full period |

---

## Notes & Assumptions

- Records with `count < 5` may be suppressed by the SSA for privacy; these appear as missing rows rather than zeros.
- State abbreviations follow standard **USPS two-letter codes**.
- Names are treated as **case-insensitive** (e.g. `MARY` and `Mary` are merged).
- The dataset reflects **registered births only** and may not capture all births in the period.