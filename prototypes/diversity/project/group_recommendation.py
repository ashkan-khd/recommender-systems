from functools import lru_cache
from typing import List

from prototypes.diversity.project.preference_score.calculator import CalculatePreferenceScore
from prototypes.diversity.project.custom_types import *

from prototypes.diversity.project.group_preference_lists import GroupPreferenceLists


class GroupRecommendationInterface:
    pass


class GroupRecommendationAnalyzer:

    def __init__(
            self,
            group: List[UserIDType],
            group_recommendation: List[MovieIDType],
            group_preference_lists: "GroupPreferenceLists",
            preference_score_calculator: "CalculatePreferenceScore"
    ):
        self.group = group
        self.group_recommendation = group_recommendation
        self.group_preference_lists = group_preference_lists
        self.preference_score_calculator = preference_score_calculator

    def __get_user_group_list_satisfaction(self, user_id: UserIDType):
        return sum(
            self.preference_score_calculator(user_id, movie_id)
            for movie_id in self.group_recommendation
        )

    def __get_user_average_group_list_satisfaction(self, user_id: UserIDType):
        return self.__get_user_group_list_satisfaction(user_id) / len(self.group_recommendation)

    @lru_cache(maxsize=None)
    def get_user_group_recommendation_satisfaction(self, user_id: UserIDType):
        return self.__get_user_group_list_satisfaction(
            user_id
        ) / self.group_preference_lists.get_user_list_satisfaction(user_id)

    def get_group_satisfaction(self):
        return sum(
            self.get_user_group_recommendation_satisfaction(user_id)
            for user_id in self.group
        ) / len(self.group)

    def get_group_disagreement(self):
        scores = [self.get_user_group_recommendation_satisfaction(user_id) for user_id in self.group]
        return max(scores) - min(scores)


class GroupRecommendationsOverallAnalyzer:
    def __init__(
            self, group
    ):
        self.group = group
        self.group_recommendation_analyzers = []

    def add_group_recommendation(self, group_recommendation_analyzer: GroupRecommendationAnalyzer):
        self.group_recommendation_analyzers.append(group_recommendation_analyzer)

    def get_user_group_recommendation_satisfaction(self, user_id):
        return sum(
            group_recommendation_analyzer.get_user_group_recommendation_satisfaction(user_id)
            for group_recommendation_analyzer in self.group_recommendation_analyzers
        ) / len(self.group_recommendation_analyzers)

    def get_group_satisfaction(self):
        return sum(
            self.get_user_group_recommendation_satisfaction(user_id)
            for user_id in self.group
        ) / len(self.group)
