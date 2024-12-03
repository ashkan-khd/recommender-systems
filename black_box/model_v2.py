from functools import cached_property
from typing import List, Set, Optional, Dict

import pandas as pd
from pandas import DataFrame
from spacy.tokens.doc import defaultdict

from .contants import *

from part2_utils.predict_user_rating import transform_csv_dataframe_to_user_movies_matrix
from .group_recommendation_v2 import MovieDesciber, get_group_recommendation, get_group_recommendation_v2


class LazyDict:
    def __init__(self, default_factory):
        self._store = {}
        self._default_factory = default_factory

    def __getitem__(self, key):
        if key not in self._store:
            # Generate value using the default factory
            self._store[key] = self._default_factory(key)

        return self._store[key]

    def __contains__(self, key):
        return key in self._store

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default


class GroupRecommendationModel:
    __MOVIE_SEARCH_AREA_SIZE = 100
    __K = 10

    def __get_movies_rated_by_group(self, user_movie_matrix: DataFrame) -> Set[MovieIDType]:
        return set(
            user_movie_matrix[user_movie_matrix["userId"].isin(self.group)].dropna(
                axis=1, how='all'
            ).drop('userId', axis=1).columns
        )

    def __get_movies_not_rated_by_group(self, user_movie_matrix) -> Set[MovieIDType]:
        all_movie_ids = set(user_movie_matrix.drop('userId', axis=1).columns)
        return all_movie_ids - self.__get_movies_rated_by_group(user_movie_matrix)

    @cached_property
    def __movie_search_area(self):
        not_rated_movies = list(self.__get_movies_not_rated_by_group(self.user_movie_matrix))
        global_popularity = []

        user_movie_matrix_without_user_column = self.user_movie_matrix.drop(columns=['userId'])
        movie_rating_counts = user_movie_matrix_without_user_column.notna().sum()

        for i in range(len(not_rated_movies)):
            global_popularity.append(movie_rating_counts[not_rated_movies[i]])

        return [
            m for m, p in sorted(
                zip(not_rated_movies, global_popularity), key=lambda x: x[1], reverse=True
            )[:self.__MOVIE_SEARCH_AREA_SIZE]
        ]

    def __get_user_movie_dict(self) -> Dict[UserIDType, Dict[MovieIDType, float]]:
        user_movie_dict = defaultdict(dict)
        for user in self.group:
            user_ratings_df = self.user_movie_matrix[
                self.user_movie_matrix['userId'] == user
            ].drop('userId', axis=1).dropna(axis=1, how='all')
            user_movie_dict[user] = dict(user_ratings_df.iloc[0].items())

        return user_movie_dict

    def __init__(self, group: List[UserIDType]):
        self.group = group
        self.movie_to_tags = LazyDict(MovieDesciber())
        raw_df = pd.read_csv(BASE_RATINGS_CSV_FILE_PATH)
        self.user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(raw_df)
        self.user_movie_dict: Dict[UserIDType, Dict[MovieIDType, float]] = self.__get_user_movie_dict()
        x = 1

    def __try_exclude_some_preferences(
            self, user_movie_matrix: DataFrame, preferences_to_exclude: Optional[List[MovieIDType]]
    ) -> DataFrame:
        if not preferences_to_exclude:
            return user_movie_matrix

        user_movie_matrix = user_movie_matrix.drop(preferences_to_exclude, axis=1)

        return user_movie_matrix

    def recommend_v1(self, preferences_to_exclude: List[MovieIDType] = None) -> List[MovieIDType]:
        limited_user_movie_matrix = self.user_movie_matrix[
            self.user_movie_matrix["userId"].isin(self.group)
        ]

        limited_user_movie_matrix_excluded = self.__try_exclude_some_preferences(limited_user_movie_matrix,
                                                                                 preferences_to_exclude)

        rated_movies = self.__get_movies_rated_by_group(limited_user_movie_matrix_excluded)
        limited_user_movie_matrix_excluded_only_rated_movies = limited_user_movie_matrix_excluded[
            ["userId"] + list(rated_movies)
        ]
        return get_group_recommendation(
            self.group,
            self.movie_to_tags,
            limited_user_movie_matrix_excluded_only_rated_movies,
            self.__movie_search_area,
            k=self.__K,
        )

    def recommend_v2(self, preferences_to_exclude: List[MovieIDType] = None):
        if preferences_to_exclude is not None:
            preferences_to_exclude = set(preferences_to_exclude)
            limited_user_movie_dict = defaultdict(dict)
            for user in self.group:
                limited_user_movie_dict[user] = {
                    movie: rating
                    for movie, rating in self.user_movie_dict[user].items()
                    if movie not in preferences_to_exclude
                }
        else:
            limited_user_movie_dict = self.user_movie_dict
        return get_group_recommendation_v2(
            self.group,
            self.movie_to_tags,
            limited_user_movie_dict,
            self.__movie_search_area,
            k=self.__K,
        )


    def recommend(self, preferences_to_exclude: List[MovieIDType] = None):
        return self.recommend_v2(preferences_to_exclude)

