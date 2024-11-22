import numpy as np
import pandas as pd

df = pd.read_csv('movies.csv')

unique_genres = set()
for genres in df['genres']:
    if genres:
        unique_genres.update(genres.split('|'))

# Convert to list and sort
unique_genres = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}

print(unique_genres)
print(len(unique_genres))


def get_genre_array(movieId):
    # Initialize a binary array of zeros with length 19
    genre_array = np.zeros(len(unique_genres), dtype=int)

    # Get the genres for the given movieId
    genres = df[df['movieId'] == movieId]['genres'].values[0]

    if genres == '(no genres listed)':
        return genre_array

    # Set the corresponding indices to 1
    for genre in genres.split('|'):
        if genre in genre_to_index:
            genre_array[genre_to_index[genre]] = 1

    return genre_array


def similarity(movieId1, movieId2):
    genre_array1 = get_genre_array(movieId1)
    genre_array2 = get_genre_array(movieId2)
    up = np.dot(genre_array1, genre_array2)
    down = np.linalg.norm(genre_array1) * np.linalg.norm(genre_array2)
    if down == 0:
        return 0
    return up / down


def diversity(movieId1, movieId2):
    return 1 - similarity(movieId1, movieId2)


# print(get_genre_array(147250))
# print(get_genre_array(147142))
print(similarity(146684, 147142))
