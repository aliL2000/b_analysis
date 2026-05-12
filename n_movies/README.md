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
│   ├── raw/                           # Original per-state CSV text files
│   └── clean/                         # Cleaned, concatenated dataset
│
├── notebooks/
│   ├── standardize.py                 # Data loading, null-filling, concatenation
│   ├── eda.py                         # Core exploratory analysis
│   └── visualisation.py               # Charts and figures
│ 
├── src/
│   └──  baby_name_data_generator.py   # File ingestion pipeline (numpy-based)
│
├── eda/
│   └── plots/                         # Saved plots and charts
│
├── requirements.txt
└── README.md
```

---

## Pipeline Summary

### 1. Standardization (`standardize.py`)

- Iterate over all state CSV files using `np.loadtxt`
- Detect and fill missing columns with typed nulls
- Reorder columns to canonical schema: `[state, sex, year, name, count]`
- Normalise `name` casing (title case)
- Validate `state` values against known 50-state list

### 2. EDA (`eda.py`)

- Summary statistics: total records, unique names, state/year coverage
- Top 10 names nationally per year
- Year-over-year count trends for selected names
- State-level heatmaps of name frequency
- Male vs. female name popularity over time

### 3. Visualisation (`visualisation.py`)

- Line charts: popularity of top names over time
- Choropleth maps: regional name dominance by state
- Bar charts: top names per decade
- Bump charts: rank changes of top-20 names (1950 vs. 1980)

---

## Key Questions to Answer

- Which names experienced the **sharpest decline** after 1960?
- How did the **total number of registered births** change year-over-year?
- Which names crossed gender lines — popular for both `M` and `F`?

---

## Dependencies

```
contourpy
cycler
fonttools
kiwisolver
matplotlib
numpy
packaging
pandas
pillow
py4j
pyparsing
pyspark
python-dateutil
six
tzdata
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Expected Outputs

| Output | Description |
|--------|-------------|
| `plots/yoy_trends.png` |  Line chart of top names over time |
| `plots/top_names_overall.png` | Bar chart of top names overall |
| `plots/sex_over_time.png` | Line chart of gender prevalance over time |
| `plots/state_heatmap.png` | Bar chart of state name counts overall |

---

## Notes & Assumptions

- State abbreviations follow standard **USPS two-letter codes**.
- The dataset reflects **registered births only** and may not capture all births in the period.