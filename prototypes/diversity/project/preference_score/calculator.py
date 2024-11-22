from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame

from part2_utils.predict_user_rating import predict_user_rating
from prototypes.diversity.project.custom_types import *


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
