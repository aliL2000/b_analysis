import numpy as np
import glob
import os

# 1. Get a list of all CSV files in a directory
path = 'messy_baby_names'
file_pattern = os.path.join(path, '*.txt')
print(file_pattern)
files_list = glob.glob(file_pattern)

# 2. Loop through and load data
data_chunks = []
for f in files_list:
    
    # Use loadtxt for text/CSV, specify delimiter if necessary
    data = np.loadtxt(f, delimiter=',',dtype=str)
    while data.shape[1] < 5:
        new_col = np.zeros((data.shape[0], 1))
        data = np.hstack((data, new_col))
    
    data_chunks.append(data)

print(data_chunks[0],data_chunks[25])
# 3. Concatenate all data into one main array
# 'axis=0' appends rows (adds new rows to the bottom)
final_data = np.concatenate(data_chunks, axis=0)

# Perform analysis
print(final_data)
print("Complete")