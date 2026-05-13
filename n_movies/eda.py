"""
eda.py  —  Netflix Dataset Exploratory Data Analysis
Reads netflix_data.csv and prints answers to 7 analytical questions.

Dependencies: pandas, numpy
"""

import textwrap
import numpy as np
import pandas as pd

CSV_PATH = "b_analysis/n_movies/data/raw/netflix_data.csv"
DIVIDER  = "=" * 70

# ── Utilities ─────────────────────────────────────────────────────────────────

def header(n, title):
    print(f"\n{DIVIDER}")
    print(f"  Q{n}: {title}")
    print(DIVIDER)

def subhead(text):
    print(f"\n── {text} ──")

def bar(label, value, total, width=30):
    filled = int(round(value / total * width)) if total else 0
    pct    = value / total * 100 if total else 0
    print(f"  {label:<30s} {'█' * filled:<{width}s}  {value:>7,}  ({pct:5.1f}%)")

def top_n_bars(series, n=10, total=None):
    top = series.value_counts().head(n)
    tot = total or top.sum()
    for label, val in top.items():
        bar(str(label)[:30], val, tot)

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"\nLoading '{CSV_PATH}' …")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

# Basic cleaning
df["date_added"]   = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["month_added"]  = df["date_added"].dt.month
df["year_added"]   = df["date_added"].dt.year

# ─────────────────────────────────────────────────────────────────────────────
# Q1: What type of content is available in different countries?
# ─────────────────────────────────────────────────────────────────────────────
header(1, "Content types available in different countries")

country_type = (
    df.dropna(subset=["country"])
    .groupby(["country", "type"])
    .size()
    .unstack(fill_value=0)
)
country_type["Total"]        = country_type.sum(axis=1)
country_type["Movie %"]      = (country_type.get("Movie", 0)   / country_type["Total"] * 100).round(1)
country_type["TV Show %"]    = (country_type.get("TV Show", 0) / country_type["Total"] * 100).round(1)
country_type                 = country_type.sort_values("Total", ascending=False)

subhead("Top 15 countries by total titles")
print(f"\n  {'Country':<22} {'Movies':>8} {'TV Shows':>9} {'Total':>7} {'Movie %':>8} {'TVShow %':>9}")
print("  " + "-" * 65)
for country, row in country_type.head(15).iterrows():
    movies   = int(row.get("Movie",   0))
    tvshows  = int(row.get("TV Show", 0))
    total    = int(row["Total"])
    m_pct    = row["Movie %"]
    tv_pct   = row["TV Show %"]
    print(f"  {country:<22} {movies:>8,} {tvshows:>9,} {total:>7,} {m_pct:>7.1f} {tv_pct:>8.1f}")

subhead("Most movie-dominant countries (min 500 titles)")
movie_dom = country_type[country_type["Total"] >= 500].sort_values("Movie %", ascending=False).head(5)
for c, r in movie_dom.iterrows():
    print(f"  {c:<22}  {r['Movie %']:>5.1f}% movies")

subhead("Most TV-show-dominant countries (min 500 titles)")
tv_dom = country_type[country_type["Total"] >= 500].sort_values("TV Show %", ascending=False).head(5)
for c, r in tv_dom.iterrows():
    print(f"  {c:<22}  {r['TV Show %']:>5.1f}% TV shows")

# ─────────────────────────────────────────────────────────────────────────────
# Q2: How has the number of movies released per year changed?
# ─────────────────────────────────────────────────────────────────────────────
header(2, "Movies released per year (last 30 years)")

cutoff = df["release_year"].max() - 30
movies_by_year = (
    df[(df["type"] == "Movie") & (df["release_year"] >= cutoff)]
    .groupby("release_year")
    .size()
    .sort_index()
)

peak_year  = int(movies_by_year.idxmax())
peak_count = int(movies_by_year.max())
max_val    = peak_count

subhead(f"Annual movie count  (peak: {peak_year} with {peak_count:,})")
for year, cnt in movies_by_year.items():
    bar(str(int(year)), cnt, max_val)

# Year-over-year growth
yoy = movies_by_year.pct_change().dropna()
avg_growth = yoy.mean() * 100
subhead("Year-over-year growth summary")
print(f"  Average annual growth rate : {avg_growth:+.1f}%")
print(f"  Biggest single-year jump   : {yoy.idxmax():.0f}  ({yoy.max()*100:+.1f}%)")
print(f"  Biggest single-year drop   : {yoy.idxmin():.0f}  ({yoy.min()*100:+.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Q3: Comparison of TV Shows vs Movies
# ─────────────────────────────────────────────────────────────────────────────
header(3, "TV Shows vs Movies — head-to-head comparison")

counts = df["type"].value_counts()
total  = counts.sum()

subhead("Overall split")
for t, c in counts.items():
    bar(t, c, total)

# Rating distribution
subhead("Rating distribution by type")
rating_split = (
    df.dropna(subset=["rating"])
    .groupby(["type", "rating"])
    .size()
    .unstack(fill_value=0)
)
for content_type in rating_split.index:
    row      = rating_split.loc[content_type]
    row_tot  = row.sum()
    top3     = row.sort_values(ascending=False).head(3)
    ratings_str = "  |  ".join([f"{r}: {v/row_tot*100:.1f}%" for r, v in top3.items()])
    print(f"  {content_type:<10}  top ratings →  {ratings_str}")

# Duration insight
subhead("Movie duration distribution (minutes)")
movie_dur = (
    df[df["type"] == "Movie"]["duration"]
    .dropna()
    .str.replace(" min", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
    .dropna()
)
bins   = [0, 60, 90, 120, 150, 999]
labels = ["<60 min", "60–90 min", "90–120 min", "120–150 min", "150+ min"]
bucketed = pd.cut(movie_dur, bins=bins, labels=labels)
dur_counts = bucketed.value_counts().reindex(labels)
for label, cnt in dur_counts.items():
    bar(label, cnt, dur_counts.sum())

subhead("TV show season distribution")
show_seasons = (
    df[df["type"] == "TV Show"]["duration"]
    .dropna()
    .str.extract(r"(\d+)")[0]
    .pipe(pd.to_numeric, errors="coerce")
    .dropna()
)
s_bins   = [0, 1, 2, 3, 5, 999]
s_labels = ["1 Season", "2 Seasons", "3 Seasons", "4–5 Seasons", "6+ Seasons"]
s_bucketed = pd.cut(show_seasons, bins=s_bins, labels=s_labels)
s_counts   = s_bucketed.value_counts().reindex(s_labels)
for label, cnt in s_counts.items():
    bar(label, cnt, s_counts.sum())

# ─────────────────────────────────────────────────────────────────────────────
# Q4: Best time to launch a TV show
# ─────────────────────────────────────────────────────────────────────────────
header(4, "Best time to launch a TV show")

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

shows_added = df[(df["type"] == "TV Show")].dropna(subset=["month_added"])

subhead("TV shows added per calendar month")
monthly = shows_added["month_added"].value_counts().sort_index()
max_m   = monthly.max()
for m, cnt in monthly.items():
    bar(MONTH_NAMES[int(m)], cnt, max_m)

best_month  = int(monthly.idxmax())
worst_month = int(monthly.idxmin())
print(f"\n  ★ Most additions : {MONTH_NAMES[best_month]}  ({monthly[best_month]:,} shows)")
print(f"  ✗ Fewest additions: {MONTH_NAMES[worst_month]}  ({monthly[worst_month]:,} shows)")

subhead("TV shows added per day of week (Mon=0)")
shows_added2 = df[(df["type"] == "TV Show")].dropna(subset=["date_added"])
shows_added2 = shows_added2.copy()
shows_added2["dow"] = shows_added2["date_added"].dt.dayofweek
DOW = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
dow_counts = shows_added2["dow"].value_counts().sort_index()
max_d = dow_counts.max()
for d, cnt in dow_counts.items():
    bar(DOW[d], cnt, max_d)

best_dow = int(dow_counts.idxmax())
print(f"\n  ★ Best launch day: {DOW[best_dow]}  ({dow_counts[best_dow]:,} shows added)")

# ─────────────────────────────────────────────────────────────────────────────
# Q5: Analysis of actors/directors by content type
# ─────────────────────────────────────────────────────────────────────────────
header(5, "Actors & directors by content type")

# Directors
subhead("Top 10 directors — Movies")
movie_dirs = df[df["type"] == "Movie"].dropna(subset=["director"])
top_n_bars(movie_dirs["director"], n=10)

subhead("Top 10 directors — TV Shows")
show_dirs = df[df["type"] == "TV Show"].dropna(subset=["director"])
top_n_bars(show_dirs["director"], n=10)

# Actors — explode the comma-separated cast column
subhead("Top 10 actors — Movies")
movie_cast = (
    df[df["type"] == "Movie"]
    .dropna(subset=["cast"])["cast"]
    .str.split(", ")
    .explode()
    .str.strip()
)
top_n_bars(movie_cast, n=10)

subhead("Top 10 actors — TV Shows")
show_cast = (
    df[df["type"] == "TV Show"]
    .dropna(subset=["cast"])["cast"]
    .str.split(", ")
    .explode()
    .str.strip()
)
top_n_bars(show_cast, n=10)

# ─────────────────────────────────────────────────────────────────────────────
# Q6: Does Netflix have more focus on TV Shows than Movies in recent years?
# ─────────────────────────────────────────────────────────────────────────────
header(6, "Netflix focus shift: Movies vs TV Shows over time")

recent = df.dropna(subset=["year_added"])
recent = recent[recent["year_added"] >= 2015]

yearly_type = (
    recent.groupby(["year_added", "type"])
    .size()
    .unstack(fill_value=0)
)
yearly_type["Total"]    = yearly_type.sum(axis=1)
yearly_type["TV %"]     = (yearly_type.get("TV Show", 0) / yearly_type["Total"] * 100).round(1)
yearly_type["Movie %"]  = (yearly_type.get("Movie",   0) / yearly_type["Total"] * 100).round(1)

subhead("Year-by-year type breakdown (titles added to platform)")
print(f"\n  {'Year':<6} {'Movies':>8} {'TV Shows':>9} {'Total':>7} {'Movie%':>8} {'TV%':>6}")
print("  " + "-" * 50)
for yr, row in yearly_type.iterrows():
    movies  = int(row.get("Movie",   0))
    tvs     = int(row.get("TV Show", 0))
    total   = int(row["Total"])
    m_pct   = row["Movie %"]
    tv_pct  = row["TV %"]
    trend   = "↑ TV" if tv_pct > 40 else "↑ Movie"
    print(f"  {int(yr):<6} {movies:>8,} {tvs:>9,} {total:>7,} {m_pct:>7.1f}% {tv_pct:>5.1f}%  {trend}")

first_tv_pct = yearly_type["TV %"].iloc[0]
last_tv_pct  = yearly_type["TV %"].iloc[-1]
delta        = last_tv_pct - first_tv_pct
subhead("Verdict")
if delta > 5:
    verdict = f"TV Show share rose by {delta:.1f} pp → Netflix IS increasingly focused on TV Shows."
elif delta < -5:
    verdict = f"TV Show share fell by {abs(delta):.1f} pp → Netflix is leaning MORE toward Movies."
else:
    verdict = f"TV Show share shifted by only {delta:+.1f} pp → Mix has stayed relatively balanced."
print(f"  {verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# Q7: Content available in different countries (genre-level deep dive)
# ─────────────────────────────────────────────────────────────────────────────
header(7, "Content available in different countries — genre deep-dive")

# Explode genres per row
genre_country = (
    df.dropna(subset=["country", "listed_in"])
    [["country", "listed_in", "type"]]
    .copy()
)
genre_country["genre"] = genre_country["listed_in"].str.split(", ")
genre_country = genre_country.explode("genre")

subhead("Top genre per country (top 12 countries by volume)")
top_countries = df["country"].value_counts().head(12).index.tolist()
gc_pivot = (
    genre_country[genre_country["country"].isin(top_countries)]
    .groupby(["country", "genre"])
    .size()
    .reset_index(name="count")
)
gc_top = (
    gc_pivot.sort_values("count", ascending=False)
    .groupby("country")
    .first()
    .reset_index()
    [["country", "genre", "count"]]
    .sort_values("count", ascending=False)
)
print(f"\n  {'Country':<22} {'Top Genre':<28} {'Count':>7}")
print("  " + "-" * 60)
for _, row in gc_top.iterrows():
    print(f"  {row['country']:<22} {row['genre']:<28} {int(row['count']):>7,}")

subhead("Genre diversity index (unique genres) — top 15 countries")
diversity = (
    genre_country[genre_country["country"].isin(
        df["country"].value_counts().head(15).index
    )]
    .groupby("country")["genre"]
    .nunique()
    .sort_values(ascending=False)
)
for country, n_genres in diversity.items():
    bar(country, n_genres, diversity.max(), width=20)

subhead("Exclusive content focus — unique genres only in one country")
genre_per_country = (
    genre_country.groupby(["genre", "country"])
    .size()
    .reset_index(name="n")
)
genre_country_count = genre_per_country.groupby("genre")["country"].nunique()
exclusive_genres    = genre_country_count[genre_country_count == 1].index
exclusive_df        = genre_per_country[genre_per_country["genre"].isin(exclusive_genres)]
if exclusive_df.empty:
    print("  No genres found exclusive to a single country in this dataset.")
else:
    for _, row in exclusive_df.sort_values("n", ascending=False).head(10).iterrows():
        print(f"  {row['genre']:<30} → only in {row['country']}  ({int(row['n']):,} titles)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  EDA complete.")
print(DIVIDER + "\n")