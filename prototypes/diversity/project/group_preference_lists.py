from dataclasses import dataclass
from typing import *

from pandas import DataFrame

from prototypes.diversity.project.custom_types import *


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
        n_preferences = user_1_ratings_df.iloc[0].nlargest(self.n).items()
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
