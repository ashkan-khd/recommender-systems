import pandas as pd

raw_df = pd.read_csv('ratings.csv')

# print(raw_df.head(20))

raw_df['datetime'] = pd.to_datetime(raw_df['timestamp'], unit='s')

print(raw_df.head(10))

min_datetime_row = raw_df.loc[raw_df['datetime'].idxmin()]

# Get the row with the maximum datetime value
max_datetime_row = raw_df.loc[raw_df['datetime'].idxmax()]

# Create a new DataFrame with the min and max datetime rows
min_max_df = pd.DataFrame([min_datetime_row, max_datetime_row])

print(min_max_df)

# Extract the year from the datetime column
raw_df['year'] = raw_df['datetime'].dt.year

# Group by the year and count the number of ratings for each year
ratings_per_year = raw_df.groupby('year').size()

print(ratings_per_year)