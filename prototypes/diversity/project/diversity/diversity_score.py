from typing import *

import numpy as np
import pandas as pd

from prototypes.diversity.project.constants import *


class MovieEmbedding:

    def __init__(self) -> None:
        self.unique_genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir',
            'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        self.genre_to_index = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        self.movies_details_df = pd.read_csv(BASE_MOVIES_DETAILS_CSV_FILE_PATH)

    def get_genre_array(self, movie_id: MovieIDType) -> np.ndarray:
        genre_array = np.zeros(len(self.unique_genres), dtype=int)
        genres = self.movies_details_df[self.movies_details_df['movieId'] == movie_id]['genres'].values[0]

        if genres == '(no genres listed)':
            return genre_array

        for genre in genres.split('|'):
            if genre in self.genre_to_index:
                genre_array[self.genre_to_index[genre]] = 1

        return genre_array

    def __call__(self, movie_id: MovieIDType) -> np.ndarray:
        return self.get_genre_array(movie_id)


class DiversityScoreCalculator:
    __instance: "DiversityScoreCalculator" = None
    __initialized: bool = False

    def __init__(self) -> None:
        if self.__initialized:
            return
        self.__movies_similarity: Dict[Tuple[MovieIDType, MovieIDType], float] = dict()
        self.__embedding = MovieEmbedding()
        self.__initialized = True

    def __new__(cls) -> "DiversityScoreCalculator":
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def calculate_similarity(self, movie_id1: MovieIDType, movie_id2: MovieIDType) -> float:
        if (movie_id1, movie_id2) in self.__movies_similarity:
            return self.__movies_similarity[(movie_id1, movie_id2)]
        if (movie_id2, movie_id1) in self.__movies_similarity:
            return self.__movies_similarity[(movie_id2, movie_id1)]

        movie1_embedding = self.__embedding(movie_id1)
        movie2_embedding = self.__embedding(movie_id2)
        up = np.dot(movie1_embedding, movie2_embedding)
        down = np.linalg.norm(movie1_embedding) * np.linalg.norm(movie2_embedding)

        if down == 0:
            sim = 0
        else:
            sim = up / down
        self.__movies_similarity[(movie_id1, movie_id2)] = sim
        return sim

    def calculate_diversity(self, movie_id1: MovieIDType, movie_id2: MovieIDType) -> float:
        return 1 - self.calculate_similarity(movie_id1, movie_id2)

    def calculate_diversity_between_movie_and_list(self, movie_id: MovieIDType, movie_list: List[MovieIDType]) -> float:
        return min([self.calculate_diversity(movie_id, movie) for movie in movie_list])

    def calculate_list_diversity(self, movie_list: List[MovieIDType]) -> float:
        sum_ = 0
        for i in range(len(movie_list)):
            for j in range(i + 1, len(movie_list)):
                sum_ += self.calculate_diversity(movie_list[i], movie_list[j])
        return sum_

    def calculate_diversity_between_two_lists(self, list1: List[MovieIDType], list2: List[MovieIDType]) -> float:
        seen_pais: Set[Tuple[MovieIDType, MovieIDType]] = set()
        sum_ = 0
        for movie1 in list1:
            for movie2 in list2:
                if (movie1, movie2) in seen_pais:
                    continue
                seen_pais.add((movie1, movie2))
                sum_ += self.calculate_diversity(movie1, movie2)
        return sum_ / len(seen_pais)

