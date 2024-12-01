from typing import List, Callable
from itertools import combinations

import pandas as pd
from pandas import DataFrame

from black_box.model import GroupRecommendationModel
from part2_utils.predict_user_rating import transform_csv_dataframe_to_user_movies_matrix
from prototypes.counterfactual_explanations.project.constants import UserIDType, MovieIDType, RATINGS_DATASET
from prototypes.counterfactual_explanations.project.data import Explanation, Group
from prototypes.counterfactual_explanations.project.explanation_analyzer import ExplanationAnalyzer
from prototypes.counterfactual_explanations.project.preferences_filter import (
    CompositeFilterWithCondition,
    MoviePreferencesFilterInterface, MovieGroupIntensityFilter, MoveGlobalPopularityFilter, MovieQueryClusterFilter,
)


def get_preferences_filter(
        group: Group,
        query: MovieIDType,
        user_movie_matrix: DataFrame,
        users_rated_movie: Callable[[MovieIDType],
        List[UserIDType]],
) -> MoviePreferencesFilterInterface:
    return CompositeFilterWithCondition(
        filters=[
            MovieGroupIntensityFilter(
                group=group,
                users_rated_movie=users_rated_movie,
            ),
            MoveGlobalPopularityFilter(
                user_movie_matrix=user_movie_matrix,
            ),
            MovieQueryClusterFilter(
                query=query,
            )
        ],
        condition=lambda filtered_preferences: len(filtered_preferences) > 5
    )


def get_preferences_for_group(
        group: Group, user_movie_matrix: DataFrame
) -> List[MovieIDType]:
    user_movie_matrix_limited = user_movie_matrix[user_movie_matrix['userId'].isin(group.users)]
    return list(user_movie_matrix_limited.dropna(axis=1, how='all').drop(columns=['userId']).columns)


def get_users_rated_movie(
        users: List[UserIDType],
        movies: List[MovieIDType],
        user_movie_matrix: DataFrame,
) -> Callable[[MovieIDType], List[UserIDType]]:
    user_movie_matrix_limited = user_movie_matrix[user_movie_matrix['userId'].isin(users)]
    user_movie_matrix_limited = user_movie_matrix_limited[["userId"] + movies]

    def users_rated_movie(movie):
        return list(user_movie_matrix_limited[user_movie_matrix_limited[movie].notna()]["userId"])

    return users_rated_movie


def get_movies_rated_for_user(
        users: List[UserIDType],
        movies: List[MovieIDType],
        user_movie_matrix: DataFrame,
) -> Callable[[UserIDType], List[MovieIDType]]:
    user_movie_matrix_limited = user_movie_matrix[user_movie_matrix['userId'].isin(users)]
    user_movie_matrix_limited = user_movie_matrix_limited[["userId"] + movies]

    def movies_rated_for_user(user):
        return list(
            user_movie_matrix_limited[user_movie_matrix_limited["userId"] == user].drop(
                columns=["userId"]
            ).dropna(axis=1, how='any').columns
        )

    return movies_rated_for_user


def find_optimum_explanation(
        group_users: List[UserIDType], user_movie_matrix: DataFrame, query: MovieIDType,
) -> Explanation:
    group = Group(group_users)

    preferences: List[MovieIDType] = get_preferences_for_group(group, user_movie_matrix)

    filter_: MoviePreferencesFilterInterface = get_preferences_filter(
        group, query, user_movie_matrix, get_users_rated_movie(
            group.users, preferences, user_movie_matrix
        ), preferences
    )
    filtered_preferences = filter_(preferences)
    print(len(filtered_preferences))
    return

    model = GroupRecommendationModel(group)
    explanations = []
    for r in range(1, len(filtered_preferences) + 1):
        for combination in combinations(filtered_preferences, r):
            to_be_removed = list(combination)
            recommendation = model.recommend(to_be_removed)
            if query not in recommendation:
                explanation = Explanation(list(combination))
                explanations.append(explanations)

    explanation_scores = []
    for explanation in explanations:
        explanation_scores.append(ExplanationAnalyzer(
            group=group,
            explanation=explanation,
            users_rated_movie=get_users_rated_movie(
                group.users, explanation.movies, user_movie_matrix
            ),
            movies_rated_for_user=get_movies_rated_for_user(
                group.users, explanation.movies, user_movie_matrix
            ),
        ).calculate_explanation_score())

    # find explanation with highest score
    best_explanation = explanations[explanation_scores.index(max(explanation_scores))]
    return best_explanation


user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(pd.read_csv(RATINGS_DATASET))

find_optimum_explanation(
    [599, 68, 274, 448, 480, 608, 590, 307, 91, 606],
    user_movie_matrix,
    query=74,
)
