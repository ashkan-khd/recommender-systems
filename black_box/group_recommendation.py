import typing
from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import sleep
from functools import lru_cache
from collections import defaultdict
import heapq

from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from .contants import *

from part2_utils.predict_user_rating import predict_user_rating, transform_csv_dataframe_to_user_movies_matrix


class CalculatePreferenceScoreRepositoryInterface(ABC):
    @abstractmethod
    def get_preference_score(self, user_id: UserIDType, movie_id: MovieIDType) -> Optional[float]:
        pass

    @abstractmethod
    def set_preference_score(
            self, user_id: UserIDType, movie_id: MovieIDType, preference_score: PreferenceScoreType
    ) -> None:
        pass


class CalculatePreferenceScore:
    def __init__(
            self,
            user_movie_matrix: DataFrame,
            preference_score_repository: "CalculatePreferenceScoreRepositoryInterface",
    ):
        self.user_movie_matrix = user_movie_matrix
        self.repository = preference_score_repository

    def __call__(self, user_id: UserIDType, movie_id: MovieIDType) -> float:
        if (preference_score := self.repository.get_preference_score(user_id, movie_id)) is None:
            preference_score = predict_user_rating(self.user_movie_matrix, user_id, movie_id)
            self.repository.set_preference_score(user_id, movie_id, preference_score)
        return preference_score


class PreferenceScoreRepository:
    __user_movie_preference_dict: Dict[UserIDType, Dict[MovieIDType, PreferenceScoreType]]
    __calculator: "CalculatePreferenceScore"

    def __init__(self, user_movie_matrix):
        self.__calculator = self.__create_preference_score_calculator(user_movie_matrix, reset_user_movie_matrix=True)

    def get_calculator(self) -> "CalculatePreferenceScore":
        return self.__calculator

    def __get_preference_score(self, user_id: UserIDType, movie_id: MovieIDType) -> Optional[PreferenceScoreType]:
        return self.__user_movie_preference_dict.get((user_id, movie_id), None)

    def __set_preference_score(
            self, user_id: UserIDType, movie_id: MovieIDType, preference_score: PreferenceScoreType
    ) -> None:
        self.__user_movie_preference_dict[user_id][movie_id] = preference_score

    def __create_preference_score_calculator(
            self, user_movie_matrix: DataFrame, reset_user_movie_matrix: bool,
    ) -> "CalculatePreferenceScore":
        if reset_user_movie_matrix:
            self.__user_movie_preference_dict = defaultdict(lambda: dict())
        calculator = type(
            "CalculatePreferenceScoreRepository",
            (CalculatePreferenceScoreRepositoryInterface,),
            dict(
                get_preference_score=self.__get_preference_score,
                set_preference_score=self.__set_preference_score,
            ),
        )()
        return CalculatePreferenceScore(user_movie_matrix, calculator)

    def reset_user_movie_matrix(self, user_movie_matrix: DataFrame, reset_user_movie_matrix: bool = False):
        self.__calculator = self.__create_preference_score_calculator(user_movie_matrix, reset_user_movie_matrix)

    def reset_users_preference_scores(self, user_id: UserIDType):
        self.__user_movie_preference_dict[user_id] = dict()


@dataclass
class UserPreferenceList:
    user_id: UserIDType
    preference_list: List[MovieIDType]
    list_satisfaction: float

    @property
    def list_average_satisfaction(self):
        return self.list_satisfaction / len(self)

    def __len__(self):
        return len(self.preference_list)


class GroupPreferenceLists:

    def __init__(
            self,
            group: List[UserIDType],
            user_movie_matrix: DataFrame,
            top_n: int,
            n: int,
    ):
        self.group = group
        self.top_n = top_n
        self.n = n
        self.user_movie_matrix: DataFrame = user_movie_matrix
        self.user_preference_lists: Dict[UserIDType, UserPreferenceList] = Optional[None]
        self.__set_user_preferences_list()

    def __set_user_preferences_list(self):
        self.user_preference_lists = dict()
        for user_id in self.group:
            self.user_preference_lists[user_id] = self.__get_user_preference_list(user_id)

    def __get_user_preference_list(self, user_id: UserIDType) -> UserPreferenceList:
        user_1_ratings_df = self.user_movie_matrix.drop('userId', axis=1)[
            self.user_movie_matrix['userId'] == user_id
            ].dropna(axis=1, how='all')
        movies = []
        satisfaction = 0
        try:
            n_preferences = user_1_ratings_df.iloc[0].nlargest(self.n).items()
        except IndexError as e:
            raise ValueError(f"User {user_id} has no preferences") from e
        for i, (movie, rating) in enumerate(n_preferences):
            movies.append(movie)
            if i < self.top_n:
                satisfaction += rating
        return UserPreferenceList(user_id, movies, satisfaction)

    def get_user_list_satisfaction(self, user_id: UserIDType):
        return self.user_preference_lists[user_id].list_satisfaction

    def get_user_list_average_satisfaction(self, user_id: UserIDType):
        return self.user_preference_lists[user_id].list_average_satisfaction

    def get_user_preference_list(self, user_id: UserIDType) -> List[MovieIDType]:
        return self.user_preference_lists[user_id].preference_list


def avg_score(group: List[UserIDType], movie: MovieIDType, preference_score_calculator: "CalculatePreferenceScore"):
    return np.mean([preference_score_calculator(user, movie) for user in group])


def least_score(group: List[UserIDType], movie: MovieIDType, preference_score_calculator: "CalculatePreferenceScore"):
    return min([preference_score_calculator(user, movie) for user in group])


def hybrid_score(
        group: List[UserIDType],
        movie: MovieIDType,
        preference_score_calculator: "CalculatePreferenceScore",
        alpha: float
):
    return (
            (1 - alpha) * avg_score(group, movie, preference_score_calculator)
            +
            alpha * least_score(group, movie, preference_score_calculator)
    )


def get_movies_not_rated_by_group(group: List[UserIDType], user_movie_matrix: DataFrame):
    all_movie_ids = set(user_movie_matrix.drop('userId', axis=1).columns)
    rated_movie_ids = set(
        user_movie_matrix[user_movie_matrix["userId"].isin(group)].dropna(
            axis=1, how='all'
        ).drop('userId', axis=1).columns
    )
    return list(all_movie_ids - rated_movie_ids), list(rated_movie_ids)


def split_all_movies(
        group: List[UserIDType],
        user_movie_matrix: DataFrame,
) -> List[MovieIDType]:
    not_rated_movies, rated_movies = get_movies_not_rated_by_group(group, user_movie_matrix)
    global_popularity = []

    user_movie_matrix_without_user_column = user_movie_matrix.drop(columns=['userId'])
    movie_rating_counts = user_movie_matrix_without_user_column.notna().sum()

    for i in range(len(not_rated_movies)):
        global_popularity.append(movie_rating_counts[not_rated_movies[i]])

    return [m for m, p in
            sorted(zip(not_rated_movies, global_popularity), key=lambda x: x[1], reverse=True)[:100]], rated_movies


def drop_rows_insufficient_data(df):
    df_working = df.copy()

    columns_to_check = df.columns[1:]
    non_nan_counts = df_working[columns_to_check].notna().sum(axis=1)

    df_cleaned = df_working[non_nan_counts >= 2]

    return df_cleaned


def filter_based_on_union_neighbors(
        group: List[UserIDType],
        user_movie_matrix: DataFrame,
) -> DataFrame:
    neighbors_union: Set[UserIDType] = set()
    for user in group:
        user_movies = list(user_movie_matrix[
                               user_movie_matrix['userId'] == user
                               ].drop("userId", axis=1).dropna(axis=1, how='all').columns)
        user_movie_matrix_new = drop_rows_insufficient_data(user_movie_matrix[["userId"] + user_movies])
        neighbors_union.update(set(user_movie_matrix_new["userId"]))

    return user_movie_matrix[user_movie_matrix["userId"].isin(neighbors_union)]


def get_group_recommendation(
        group: List[UserIDType],
        movie_search_area: List[MovieIDType],
        preference_score_calculator: "CalculatePreferenceScore",
        k: int,
        alpha: float,
) -> List[MovieIDType]:

    aggregator = lambda movie_id: hybrid_score(group, movie_id, preference_score_calculator, alpha)

    movies_score: Dict[MovieIDType, float] = {}
    for movie in tqdm(movie_search_area, desc=f"Calculating scores for movie_search_area"):
        movies_score[movie] = aggregator(movie)

    top_k_movies: List[MovieIDType] = heapq.nlargest(k, movies_score, key=movies_score.get)

    return top_k_movies
