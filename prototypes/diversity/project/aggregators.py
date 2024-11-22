from typing import *

import numpy as np

from prototypes.diversity.project.preference_score.calculator import CalculatePreferenceScore
from prototypes.diversity.project.custom_types import *


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
