import pandas as pd
import numpy as np
import os
import random

# -------------------------
# CONFIG
# -------------------------
states = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
    'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
    'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'
]

years = list(range(1910, 2023))
sexes = ['M', 'F']

male_names = ["James", "John", "Robert", "Michael"]
female_names = ["Mary", "Patricia", "Jennifer", "Linda"]
ambiguous_names = ["Taylor", "Jordan", "Alex", "Casey"]

all_names = male_names + female_names + ambiguous_names

output_dir = "messy_baby_names"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def randomize_column_names():
    variants = [
        ["state", "sex", "year", "name", "count"]
    ]
    return random.choice(variants)


def corrupt_value(value, field):
    r = random.random()

    # Inject nulls
    if r < 0.05:
        return None

    # Type corruption
    if r < 0.10:
        if field == "count":
            return str(value)  # numeric as string
        if field == "year":
            return float(value)  # year as float
        if field == "name":
            return value.lower()  # lowercase inconsistency

    # Formatting issues
    if r < 0.15:
        if isinstance(value, str):
            return f" {value} "  # extra spaces

    # Invalid values
    if r < 0.02:
        if field == "sex":
            return random.choice(["X", "", "Male", "Female"])

    return value


def generate_trend(base, year):
    peak = random.choice(range(1950, 2010))
    trend = base * np.exp(-((year - peak) ** 2) / (2 * 400))
    return max(0, int(trend * np.random.normal(1, 0.3)))


# -------------------------
# GENERATION
# -------------------------
for state in states:
    records = []

    for sex in sexes:
        for year in years:
            for name in all_names:

                # Skip strict gender mapping sometimes (introduce ambiguity)
                if random.random() < 0.1:
                    pass
                else:
                    if name in male_names and sex == 'F':
                        continue
                    if name in female_names and sex == 'M':
                        continue

                base = random.randint(10, 1000)
                count = generate_trend(base, year)

                if count < 5:
                    continue

                record = {
                    "state": state,
                    "sex": sex,
                    "year": year,
                    "name": name,
                    "count": count
                }

                # Corrupt values randomly
                for k in record:
                    record[k] = corrupt_value(record[k], k)

                records.append(record)

                # Duplicate rows occasionally
                if random.random() < 0.03:
                    records.append(record.copy())

    df = pd.DataFrame(records)

    # Randomize column names
    df.columns = randomize_column_names()

    # Shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Occasionally drop a column entirely
    if random.random() < 0.1:
        df = df.drop(columns=[random.choice(df.columns)])

    df.to_csv(f"{output_dir}/{state}.txt", index=False)

print("Messy dataset generated.")