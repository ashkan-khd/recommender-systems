from functools import cached_property
from typing import List, Set, Optional

import pandas as pd
from pandas import DataFrame

from .contants import *

from part2_utils.predict_user_rating import transform_csv_dataframe_to_user_movies_matrix
from .group_recommendation import get_group_recommendation, PreferenceScoreRepository


class GroupRecommendationModel:
    __K = 10
    __ALPHA = 0.25
    __MOVIE_SEARCH_AREA_SIZE = 50

    @classmethod
    def __drop_rows_insufficient_data(cls, df: DataFrame) -> DataFrame:
        df_working = df.copy()

        columns_to_check = df.columns[1:]
        non_nan_counts = df_working[columns_to_check].notna().sum(axis=1)

        df_cleaned = df_working[non_nan_counts >= 2]

        return df_cleaned

    def __filter_based_on_union_neighbors(
            self,
            user_movie_matrix: DataFrame,
    ) -> DataFrame:
        neighbors_union: Set[UserIDType] = set()
        for user in self.group:
            user_movies = list(user_movie_matrix[
                                   user_movie_matrix['userId'] == user
                                   ].drop("userId", axis=1).dropna(axis=1, how='all').columns)
            user_movie_matrix_new = self.__drop_rows_insufficient_data(user_movie_matrix[["userId"] + user_movies])
            neighbors_union.update(set(user_movie_matrix_new["userId"]))

        return user_movie_matrix[user_movie_matrix["userId"].isin(neighbors_union)]

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

    def __init__(self, group: List[UserIDType]):
        self.group = group
        raw_df = pd.read_csv(BASE_RATINGS_CSV_FILE_PATH)
        self.user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(raw_df)
        self.calculator_repository = PreferenceScoreRepository(self.user_movie_matrix)

    def __get_group_recommendation(self):
        return get_group_recommendation(
            self.group,
            self.__movie_search_area,
            self.calculator_repository.get_calculator(),
            k=self.__K, alpha=self.__ALPHA,
        )

    def __try_exclude_some_preferences(self, user_movie_matrix: DataFrame,
                                       preferences_to_exclude: Optional[List[MovieIDType]]):
        if not preferences_to_exclude:
            return user_movie_matrix

        for movie in preferences_to_exclude:
            for user in self.group:
                if movie in user_movie_matrix[user_movie_matrix["userId"] == user].dropna(
                        axis=1, how='all'
                ).drop('userId', axis=1).columns:
                    self.calculator_repository.reset_users_preference_scores(user)

        user_movie_matrix = user_movie_matrix.drop(preferences_to_exclude, axis=1)

        return user_movie_matrix

    def recommend(self, preferences_to_exclude: List[MovieIDType] = None) -> List[MovieIDType]:
        user_movie_matrix = self.__try_exclude_some_preferences(self.user_movie_matrix, preferences_to_exclude)

        rated_movies = self.__get_movies_rated_by_group(user_movie_matrix)
        movie_search_area = self.__movie_search_area
        user_movie_matrix = user_movie_matrix[["userId"] + list(rated_movies) + movie_search_area]
        user_movie_matrix = self.__drop_rows_insufficient_data(user_movie_matrix)

        user_movie_matrix = self.__filter_based_on_union_neighbors(user_movie_matrix)

        self.calculator_repository.reset_user_movie_matrix(user_movie_matrix)

        return self.__get_group_recommendation()
