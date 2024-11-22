import pandas as pd

raw_df = pd.read_csv('ratings.csv')

# Convert the timestamp column to datetime
raw_df['datetime'] = pd.to_datetime(raw_df['timestamp'], unit='s')

# Filter the DataFrame for ratings up to the end of 2008
base_df = raw_df[raw_df['datetime'].dt.year <= 2008]
base_df = base_df[['userId', 'movieId', 'rating', 'timestamp']]
print("Saving base with", len(base_df), "entries")
base_df.to_csv('splits/base.csv', index=False)

# Filter and save DataFrames for each year from 2009 to 2018
for year in range(2009, 2019):
    chunk_df = raw_df[raw_df['datetime'].dt.year == year]
    chunk_df = chunk_df[['userId', 'movieId', 'rating', 'timestamp']]
    print("Saving chunk for year", year, "with", len(chunk_df), "entries")
    chunk_df.to_csv(f'splits/chunk{year - 2008}.csv', index=False)