import numpy as np
import glob
import os

# 1. Get a list of all CSV files in a directory
path = 'b_analysis/baby_names/messy_baby_names'
file_pattern = os.path.join(path, '*.txt')
print(file_pattern)
files_list = glob.glob(file_pattern)

# 2. Loop through and load data
data_chunks = []

EXPECTED_HEADERS = ['state', 'sex', 'year', 'name', 'count']

for i, f in enumerate(files_list):
    data = np.loadtxt(f, delimiter=',', dtype=str)
    
    if i == 0:
        header = data[0]
    
    file_headers = data[0].tolist()
    data = data[1:]  # strip header row

    # Add any missing columns with appropriate nulls
    for col in EXPECTED_HEADERS:
        if col not in file_headers:
            null_value = '0' if col in ('count',) else ''
            new_col = np.full((data.shape[0], 1), null_value)
            data = np.hstack((data, new_col))
            file_headers.append(col)

    # Reorder columns to match EXPECTED_HEADERS
    col_order = [file_headers.index(col) for col in EXPECTED_HEADERS]
    data = data[:, col_order]

    data_chunks.append(data)

final_data = np.concatenate([header.reshape(1, -1)] + data_chunks, axis=0)

print("Complete")