from typing import *

import numpy as np

from prototypes.diversity.project.diversity import DiversityScoreCalculator
from prototypes.diversity.project.preference_score.calculator import CalculatePreferenceScore
from prototypes.diversity.project.constants import *


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


def hybrid_score_with_diversity(
        group: List[UserIDType],
        movie: MovieIDType,
        preference_score_calculator: "CalculatePreferenceScore",
        previous_recommendation: List[MovieIDType],
        diversity_score_calculator: "DiversityScoreCalculator",
        alpha: float,
        beta: float
):

    return (
            (1 - beta) * hybrid_score(group, movie, preference_score_calculator, alpha)
            +
            beta * diversity_score_calculator.calculate_diversity_between_movie_and_list(movie, previous_recommendation)
    )
