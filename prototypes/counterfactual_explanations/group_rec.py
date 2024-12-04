import json
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

from part2_utils.predict_user_rating import predict_user_rating, transform_csv_dataframe_to_user_movies_matrix

UserIDType = int
MovieIDType = int

RatingType = float
PreferenceScoreType = float

BASE_RATINGS_CSV_FILE_PATH = "data/ml-latest-small/ratings.csv"

BASE_MOVIES_DETAILS_CSV_FILE_PATH = "data/ml-latest-small/movies.csv"


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
            preference_score = predict_user_rating(self.user_movie_matrix, user_id, movie_id, use_average_method=False)
            self.repository.set_preference_score(user_id, movie_id, preference_score)
        return preference_score


class PreferenceScoreRepository:
    __user_movie_preference_dict: Dict[Tuple[UserIDType, MovieIDType], PreferenceScoreType]
    __calculator: "CalculatePreferenceScore"

    def __init__(self, user_movie_matrix):
        self.__calculator = self.__create_preference_score_calculator(user_movie_matrix)

    def get_calculator(self) -> "CalculatePreferenceScore":
        return self.__calculator

    def __get_preference_score(self, user_id: UserIDType, movie_id: MovieIDType) -> Optional[PreferenceScoreType]:
        return self.__user_movie_preference_dict.get((user_id, movie_id), None)

    def __set_preference_score(
            self, user_id: UserIDType, movie_id: MovieIDType, preference_score: PreferenceScoreType
    ) -> None:
        self.__user_movie_preference_dict[(user_id, movie_id)] = preference_score

    def __create_preference_score_calculator(
            self, user_movie_matrix: DataFrame
    ) -> "CalculatePreferenceScore":
        self.__user_movie_preference_dict = dict()
        calculator = type(
            "CalculatePreferenceScoreRepository",
            (CalculatePreferenceScoreRepositoryInterface,),
            dict(
                get_preference_score=self.__get_preference_score,
                set_preference_score=self.__set_preference_score,
            ),
        )()
        return CalculatePreferenceScore(user_movie_matrix, calculator)


sample_group = [269, 120, 278, 163, 161, 383, 238, 392, 188, 35]
raw_df = pd.read_csv(BASE_RATINGS_CSV_FILE_PATH)

user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(raw_df)
print(user_movie_matrix.shape)

group_movie_matrices = []
for user_id in sample_group:
    user_specific_movie_matrix = user_movie_matrix.drop('userId', axis=1)[
        user_movie_matrix['userId'] == user_id
        ].dropna(axis=1, how='all').reset_index(drop=True)
    group_movie_matrices.append(user_specific_movie_matrix)

union_movies = set()
for user_specific_movie_matrix in group_movie_matrices:
    union_movies = union_movies.union(set(user_specific_movie_matrix.columns))

calc = PreferenceScoreRepository(user_movie_matrix).get_calculator()
excluded_user_movie_matrix = user_movie_matrix.drop(list(union_movies), axis=1).reset_index(drop=True)
excluded_movies = excluded_user_movie_matrix.drop('userId', axis=1).columns
user_movie_matrix_filled = user_movie_matrix.copy()

common_movies = None

for user in sample_group:
    user_movies = set()
    user_nan_count = 0
    for movie in tqdm(excluded_movies):
        user_movie_new_rating = calc(user, movie)
        if user_movie_new_rating is not None and not np.isnan(user_movie_new_rating):
            user_movies.add(movie)
        else:
            user_nan_count += 1

        user_movie_matrix_filled.loc[user_movie_matrix_filled['userId'] == user, movie] = user_movie_new_rating

    print("{0} had {1} nan ratings".format(user, user_nan_count))
    if common_movies is None:
        common_movies = user_movies
    else:
        common_movies = common_movies.intersection(user_movies)


with open("common_movies.json", "w") as f:
    json.dump(list(common_movies), f)

user_movie_matrix_filled.to_csv("user_movie_matrix_filled.csv", index=False)
