from typing import List, Callable

from prototypes.counterfactual_explanations.project.constants import UserIDType, MovieIDType
from prototypes.counterfactual_explanations.project.data import Group, Explanation


class ExplanationAnalyzer:
    def __init__(
            self,
            group: Group,
            explanation: Explanation,
            users_rated_movie: Callable[[MovieIDType], List[UserIDType]],
            movies_rated_for_user: Callable[[UserIDType], List[MovieIDType]],
    ):
        self.group = group
        self.explanation = explanation
        self.users_rated_movie = users_rated_movie
        self.movies_rated_for_user = movies_rated_for_user

    def get_popularity(self) -> float:
        return sum(
            [
                self.group.get_item_intensity(movie, self.users_rated_movie)
                for movie in self.explanation
            ]
        ) / len(self.explanation)

    def get_normalized_popularity(self) -> float:
        return self.get_popularity() / len(self.group)

    def get_fairness(self) -> float:
        users_intensity_list = [
            self.explanation.get_user_intensity(user, self.movies_rated_for_user)
            for user in self.group
        ]
        return max(users_intensity_list) - min(users_intensity_list)

    def get_normalized_fairness(self) -> float:
        return self.get_fairness() / len(self.explanation)

    def get_conciceness(self) -> float:
        return 1 / len(self.explanation)

    def calculate_explanation_score(self) -> float:
        return (self.get_normalized_popularity() + self.get_normalized_fairness() + self.get_conciceness()) / 3
