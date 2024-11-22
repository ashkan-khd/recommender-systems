from typing import Dict, Tuple, Optional

from pandas import DataFrame

from .calculator import CalculatePreferenceScoreRepositoryInterface, CalculatePreferenceScore
from prototypes.diversity.project.custom_types import *


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
            (CalculatePreferenceScoreRepositoryInterface, ),
            dict(
                get_preference_score=self.__get_preference_score,
                set_preference_score=self.__set_preference_score,
            ),
        )()
        return CalculatePreferenceScore(user_movie_matrix, calculator)
