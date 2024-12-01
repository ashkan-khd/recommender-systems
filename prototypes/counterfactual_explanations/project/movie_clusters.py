from typing import List, Optional

import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from black_box.group_recommendation import MovieIDType
from prototypes.counterfactual_explanations.project.constants import MOVIE_DETAILS_DATASET, USER_TAGS_FOR_MOVIES_DATASET


class Encoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, descriptive: str):
        embedding = self.model.encode(descriptive, convert_to_numpy=True)
        return embedding


class MovieDesciber:
    def __init__(self):
        self.movie_details = pd.read_csv(MOVIE_DETAILS_DATASET)
        self.user_tags_for_movies = pd.read_csv(USER_TAGS_FOR_MOVIES_DATASET)

    def __call__(self, movie_id: MovieIDType):
        genres = self.movie_details[self.movie_details['movieId'] == movie_id]['genres'].values[0]
        description_list = set(genre.lower() for genre in genres.split('|'))

        tags = self.user_tags_for_movies[self.user_tags_for_movies['movieId'] == movie_id]['tag'].tolist()
        tags_set = set(tag.lower() for tag in tags)
        description_list = description_list.union(tags_set)

        description_string = " ".join(description_list)
        return description_string


class MoviesClustering:
    def __init__(self, movies: List[MovieIDType], min_cluster_size=None):
        self.movies = movies
        self.min_cluster_size = min_cluster_size
        self.movie_encoder = Encoder()
        self.movie_describer = MovieDesciber()

    def get_movie_encodings(self):
        movie_encodings = []
        for movie_id in self.movies:
            movie_description = self.movie_describer(movie_id)
            movie_encodings.append(self.movie_encoder(movie_description))
        return movie_encodings

    def get_hdbscan_kwargs(self):
        if self.min_cluster_size is not None:
            return {"min_cluster_size": self.min_cluster_size}
        return {}

    def cluster_movies(self, important_index: Optional[int] = None):
        movie_encodings = self.get_movie_encodings()
        encodings_array = np.array(movie_encodings)

        clusterer = hdbscan.HDBSCAN(**self.get_hdbscan_kwargs())
        labels = clusterer.fit_predict(encodings_array)

        # If the important movie is noise, adjust clustering
        if important_index is not None and labels[important_index] == -1:
            important_encoding = movie_encodings[important_index]
            # Find the nearest non-noise cluster
            cluster_indices = [i for i, label in enumerate(labels) if label != -1]
            nearest_cluster_index = min(
                cluster_indices,
                key=lambda i: np.linalg.norm(movie_encodings[i] - important_encoding)
            )
            labels[important_index] = labels[nearest_cluster_index]

        # Organize movie IDs into clusters
        clusters = {}
        for movie_id, label in zip(self.movies, labels):
            if label != -1:  # Ignore noise points (label -1)
                clusters.setdefault(label, []).append(movie_id)

        # Convert clusters dictionary to a list of clusters
        cluster_list = list(clusters.values())

        # Return the clusters and the number of clusters
        return cluster_list, len(cluster_list)

