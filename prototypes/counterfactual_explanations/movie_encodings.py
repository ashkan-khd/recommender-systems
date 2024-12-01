import random

import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load the dataframes
movie_details = pd.read_csv("data/ml-latest-small/movies.csv")
user_tags_for_movies = pd.read_csv("data/ml-latest-small/tags.csv")

# Select a random sample of 100 movie ids
movie_ids = movie_details['movieId'].unique().tolist()
random_movie_ids = random.sample(movie_ids, 100)


class Encoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, descriptive: str):
        embedding = self.model.encode(descriptive, convert_to_numpy=True)
        return embedding


def embed(movie_id):
    # Get the genres for the current movie id
    genres = movie_details[movie_details['movieId'] == movie_id]['genres'].values[0]
    description_list = set(genre.lower() for genre in genres.split('|'))

    # Get the tags for the current movie id
    tags = user_tags_for_movies[user_tags_for_movies['movieId'] == movie_id]['tag'].tolist()
    tags_set = set(tag.lower() for tag in tags)
    description_list = description_list.union(tags_set)

    # Join the description list into a single string
    description_string = " ".join(description_list)
    return description_string


def cluster_movies(movie_ids, movie_encodings, min_cluster_size=5):
    """
    Performs HDBSCAN clustering on movie embeddings and returns clusters.

    Args:
        movie_ids (list): List of movie IDs.
        movie_encodings (list): List of numeric embeddings (np.ndarray) for each movie.
        min_cluster_size (int): Minimum size of a cluster.

    Returns:
        list: A list of clusters, where each cluster is a list of movie IDs.
        int: The number of clusters (excluding noise).
    """
    # Convert movie_encodings to a numpy array
    encodings_array = np.array(movie_encodings)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(encodings_array)

    # Organize movie IDs into clusters
    clusters = {}
    for movie_id, label in zip(movie_ids, labels):
        if label != -1:  # Ignore noise points (label -1)
            clusters.setdefault(label, []).append(movie_id)

    # Convert clusters dictionary to a list of clusters
    cluster_list = list(clusters.values())

    # Return the clusters and the number of clusters
    return cluster_list, len(cluster_list)


# Initialize the movie_descriptions list
movie_descriptions = []
movie_encodings = []

# Iterate over each random movie id
for movie_id in random_movie_ids:
    # Append the description string to the movie_descriptions list
    movie_descriptions.append(embed(movie_id))

encoder = Encoder()
for i in tqdm(range(len(random_movie_ids)), desc="Embedding movies"):
    # Append the description string to the movie_descriptions list
    movie_encodings.append(encoder(movie_descriptions[i]))

clusters, num_clusters = cluster_movies(random_movie_ids, movie_encodings, min_cluster_size=10)
print(f"Number of clusters: {num_clusters}")
for i, cluster in enumerate(clusters, start=1):
    print(f"Cluster {i}: {cluster}")
