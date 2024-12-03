import random
import time
from time import sleep
from typing import List, Callable
from itertools import combinations

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from black_box.model_v2 import GroupRecommendationModel
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
    # return lambda preferences: preferences[:3 * len(preferences) // 4]

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
        model: GroupRecommendationModel,
) -> Explanation:
    group = Group(group_users)

    preferences: List[MovieIDType] = get_preferences_for_group(group, user_movie_matrix)
    print("Number of preferences before filter: ", len(preferences))

    filter_: MoviePreferencesFilterInterface = get_preferences_filter(
        group, query, user_movie_matrix, get_users_rated_movie(
            group.users, preferences, user_movie_matrix
        )
    )
    filtered_preferences = filter_(preferences)
    print("Number of filtered preferences: ", len(filtered_preferences))
    # print(filtered_preferences)
    sleep(1)

    explanations = []
    for r in range(1, len(filtered_preferences) + 1):
        for combination in tqdm(
                list(combinations(filtered_preferences, r)),
                desc="checking combinations when r={}/{}".format(r, len(filtered_preferences))
        ):
            to_be_removed = list(combination)
            recommendation = model.recommend(to_be_removed)
            if query not in recommendation:
                explanation = Explanation(list(combination))
                explanations.append(explanation)
    # recommendation = model.recommend(filtered_preferences)
    # if query not in recommendation:
    #     explanation = Explanation(filtered_preferences)
    #     explanations.append(explanation)

    print("Number of found explanations: ", len(explanations))
    if not explanations:
        print("Unfortunately, no explanation was found.")
        return None

    sleep(1)
    explanation_scores = []
    for explanation in tqdm(explanations, desc="calculating explanation scores"):
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

group_users = [280, 528, 251, 43, 237, 149, 372, 114, 360, 366]
model = GroupRecommendationModel(group_users)
# t1 = time.time()
recommended = model.recommend(preferences_to_exclude=None)
# t2 = time.time()
# print("took: ", t2 - t1)
user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(pd.read_csv(RATINGS_DATASET))

result = find_optimum_explanation(
    group_users,
    user_movie_matrix,
    query=786,
    model=model,
)
print(result)
# preferences: List[MovieIDType] = get_preferences_for_group(Group(group_users), user_movie_matrix)
# query = 319
# filter_: MoviePreferencesFilterInterface = get_preferences_filter(
#     None, None, None, None
# )
# filtered_preferences = filter_(preferences)
# new_recommended = model.recommend(filtered_preferences)
# print("recommended: ", recommended)
# print("new_recommended: ", new_recommended)

#
# for i, movie in enumerate(recommended):
#     print("i: {}, recommended movie: {}".format(i + 1, movie))
#
#     result = find_optimum_explanation(
#         group_users,
#         user_movie_matrix,
#         query=movie,
#         model=model,
#     )
#     if result:
#         print(result)


