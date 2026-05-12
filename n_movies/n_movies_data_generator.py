"""
generate_netflix_data.py
Generates a large dummy Netflix dataset (1,000,000 rows) with realistic
missing-data patterns, saved to netflix_data.csv.

Dependencies: numpy, pandas  (pip install numpy pandas)
"""

import numpy as np
import pandas as pd

# ── Seed ──────────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
NUM_ROWS = 1_000_000
OUTPUT_FILE = "b_analysis/n_movies/data/raw/netflix_data.csv"
CHUNK_SIZE = 100_000          # write in chunks to keep memory reasonable

# ── Reference pools ───────────────────────────────────────────────────────────
TYPES = ["Movie", "TV Show"]

TITLES_MOVIES = [
    "The Last Horizon", "Dark Waters", "Crimson Peak", "Shadow Protocol",
    "Eternal Flames", "The Lost City", "Beyond the Stars", "Iron Will",
    "Silent Storm", "The Forgotten Path", "Midnight Sun", "Edge of Tomorrow",
    "Broken Wings", "The Final Chapter", "Neon Lights", "Cold Pursuit",
    "The Great Escape", "Burning Sands", "White Noise", "Fractured",
    "Glass Ceiling", "Hollow Ground", "Parallel Lives", "The Red Line",
    "Vanishing Point", "Echoes of War", "The Soft Machine", "Open Water",
    "Landfall", "Static",
]

TITLES_SHOWS = [
    "Stranger Things", "The Crown", "Narcos", "Ozark", "Mindhunter",
    "Dark", "Money Heist", "Squid Game", "Bridgerton", "Lupin",
    "The Witcher", "Emily in Paris", "Sex Education", "Cobra Kai",
    "You", "Outer Banks", "Virgin River", "Never Have I Ever",
    "Fate: The Winx Saga", "Cursed", "Haunted", "Clickbait",
    "The Umbrella Academy", "Locke & Key", "Sweet Magnolias",
    "Ginny & Georgia", "Shadow and Bone", "Manifest", "1899", "Wednesday",
]

DIRECTORS = [
    "James Cameron", "Christopher Nolan", "Steven Spielberg", "Ava DuVernay",
    "Bong Joon-ho", "Greta Gerwig", "Martin Scorsese", "Denis Villeneuve",
    "Jordan Peele", "Kathryn Bigelow", "Alfonso Cuarón", "Wes Anderson",
    "Chloe Zhao", "David Fincher", "Sofia Coppola", "Ryan Coogler",
    "Taika Waititi", "Lulu Wang", "Barry Jenkins", "Dee Rees",
]

ACTORS = [
    "Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Scarlett Johansson",
    "Denzel Washington", "Cate Blanchett", "Brad Pitt", "Viola Davis",
    "Ryan Gosling", "Jennifer Lawrence", "Idris Elba", "Zendaya",
    "Pedro Pascal", "Ana de Armas", "Mahershala Ali", "Florence Pugh",
    "Timothée Chalamet", "Awkwafina", "Daniel Kaluuya", "Lupita Nyong'o",
    "John Boyega", "Saoirse Ronan", "Adam Driver", "Margot Robbie",
]

COUNTRIES = [
    "United States", "United Kingdom", "India", "Canada", "France",
    "Germany", "South Korea", "Japan", "Spain", "Brazil",
    "Australia", "Mexico", "Italy", "Nigeria", "Sweden",
    "Argentina", "Turkey", "Poland", "Denmark", "Thailand",
]

RATINGS = ["G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-G", "TV-PG", "TV-14", "TV-MA"]

GENRES = [
    "Dramas", "Comedies", "Thrillers", "Action & Adventure", "Documentaries",
    "Horror Movies", "Romantic Movies", "Sci-Fi & Fantasy", "Crime TV Shows",
    "International Movies", "Children & Family Movies", "Stand-Up Comedy",
    "Anime Series", "Reality TV", "Music & Musicals",
]

DESCRIPTIONS = [
    "A gripping tale of survival against all odds in a post-apocalyptic world.",
    "Two unlikely strangers discover a shared secret that changes everything.",
    "A seasoned detective races against time to solve a mysterious disappearance.",
    "Set in the near future, humanity faces its greatest challenge yet.",
    "An inspiring journey of self-discovery, love, and redemption.",
    "When the truth emerges, nothing will ever be the same again.",
    "A family torn apart by circumstance must find a way back to each other.",
    "In a world of deception, only one person knows the real story.",
    "A thrilling adventure across three continents and two decades.",
    "Secrets, lies, and betrayal collide in this edge-of-your-seat drama.",
    "An unlikely hero must choose between love and duty.",
    "The past comes back to haunt a small coastal town.",
    "Beneath the surface of a perfect life lies a dangerous truth.",
    "A coming-of-age story set against the backdrop of revolution.",
    "Three strangers are bound together by a single, fateful night.",
]

# ── Missing-data rates per column ─────────────────────────────────────────────
MISSING_RATES = {
    "director"   : 0.30,   # 30%
    "cast"       : 0.08,   # 8%
    "country"    : 0.15,   # 15%
    "date_added" : 0.12,   # 12%
    "rating"     : 0.05,   # 5%
    "duration"   : 0.04,   # 4%
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_cast(n):
    sizes = RNG.integers(2, 6, size=n)
    actors = np.array(ACTORS)
    out = []
    for s in sizes:
        idxs = RNG.choice(len(actors), size=s, replace=False)
        out.append(", ".join(actors[idxs]))
    return np.array(out, dtype=object)

def _build_genre(n):
    sizes = RNG.integers(1, 4, size=n)
    genres = np.array(GENRES)
    out = []
    for s in sizes:
        idxs = RNG.choice(len(genres), size=s, replace=False)
        out.append(", ".join(genres[idxs]))
    return np.array(out, dtype=object)

def _apply_missing(series, rate):
    mask = RNG.random(len(series)) < rate
    series = series.copy().astype(object)
    series[mask] = np.nan
    return series

# ── Chunk generator ───────────────────────────────────────────────────────────
def generate_chunk(start_id, n):
    types    = RNG.choice(TYPES, size=n)
    is_movie = types == "Movie"

    movie_titles = np.array(TITLES_MOVIES)
    show_titles  = np.array(TITLES_SHOWS)
    titles = np.where(
        is_movie,
        movie_titles[RNG.integers(0, len(movie_titles), size=n)],
        show_titles [RNG.integers(0, len(show_titles),  size=n)],
    )

    directors    = np.array(DIRECTORS)[RNG.integers(0, len(DIRECTORS), size=n)]
    cast         = _build_cast(n)
    countries    = np.array(COUNTRIES)[RNG.integers(0, len(COUNTRIES), size=n)]

    base       = pd.Timestamp("2008-01-01")
    days_range = (pd.Timestamp("2023-12-31") - base).days
    date_added = (
        pd.to_datetime(base + pd.to_timedelta(RNG.integers(0, days_range, size=n), unit="D"))
        .strftime("%B %d, %Y")
    )

    release_year = RNG.integers(1990, 2024, size=n)
    ratings      = np.array(RATINGS)[RNG.integers(0, len(RATINGS), size=n)]

    season_n = RNG.integers(1, 9, size=n)
    duration = np.where(
        is_movie,
        [f"{d} min" for d in RNG.integers(60, 181, size=n)],
        [f"{s} Season{'s' if s > 1 else ''}" for s in season_n],
    )

    listed_in    = _build_genre(n)
    descriptions = np.array(DESCRIPTIONS)[RNG.integers(0, len(DESCRIPTIONS), size=n)]

    df = pd.DataFrame({
        "show_id"      : [f"s{start_id + i}" for i in range(n)],
        "type"         : types,
        "title"        : titles,
        "director"     : directors,
        "cast"         : cast,
        "country"      : countries,
        "date_added"   : date_added,
        "release_year" : release_year,
        "rating"       : ratings,
        "duration"     : duration,
        "listed_in"    : listed_in,
        "description"  : descriptions,
    })

    for col, rate in MISSING_RATES.items():
        df[col] = _apply_missing(df[col], rate)

    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating {NUM_ROWS:,} rows → {OUTPUT_FILE}")
    rates_display = {k: f"{v*100:.0f}%" for k, v in MISSING_RATES.items()}
    print(f"Missing-data rates: {rates_display}\n")

    header_written = False
    for chunk_start in range(0, NUM_ROWS, CHUNK_SIZE):
        chunk_n  = min(CHUNK_SIZE, NUM_ROWS - chunk_start)
        df_chunk = generate_chunk(chunk_start + 1, chunk_n)
        df_chunk.to_csv(
            OUTPUT_FILE,
            mode  ="w" if not header_written else "a",
            header=not header_written,
            index =False,
        )
        header_written = True
        pct = min((chunk_start + chunk_n) / NUM_ROWS * 100, 100)
        print(f"  {chunk_start + chunk_n:>10,} / {NUM_ROWS:,}  ({pct:.0f}%)")

    print(f"\n✅  Done!  Saved to '{OUTPUT_FILE}'")
    print("\nMissing-value verification (sample read of first 100k rows):")
    sample     = pd.read_csv(OUTPUT_FILE, nrows=100_000)
    missing    = sample.isnull().sum()
    missing_pct = (missing / len(sample) * 100).round(2)
    summary    = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    print(summary[summary["missing_count"] > 0].to_string())

if __name__ == "__main__":
    main()