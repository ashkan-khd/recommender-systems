import typing
from typing import *

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series


def pearson_similarity(user1_ratings: Series, user2_ratings: Series, **kwargs) -> float:
    common_ratings = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index)
    if len(common_ratings) == 0:
        return np.nan
    user1_common = user1_ratings[common_ratings]
    user2_common = user2_ratings[common_ratings]
    mean_user1 = user1_common.mean()
    mean_user2 = user2_common.mean()
    numerator = ((user1_common - mean_user1) * (user2_common - mean_user2)).sum()
    denominator = np.sqrt(((user1_common - mean_user1) ** 2).sum()) * np.sqrt(((user2_common - mean_user2) ** 2).sum())
    with np.errstate(invalid='ignore'):
        return numerator / denominator


def transform_csv_dataframe_to_user_movies_matrix(csv_df: DataFrame) -> DataFrame:
    user_movie_matrix = csv_df.pivot(index='userId', columns='movieId', values='rating')
    user_movie_matrix.reset_index(inplace=True)
    user_movie_matrix.columns.name = None
    return user_movie_matrix


def prepare_user_movie_matrix_for_cf_matrix(
        user_movie_matrix: DataFrame,
        movie_id: int,
        user_id: int,
) -> DataFrame:
    user_ratings = user_movie_matrix[user_movie_matrix['userId'] == user_id].dropna(axis=1, how='all')

    rated_movies = user_ratings.columns[1:]
    if movie_id in rated_movies:
        raise ValueError(f"The user {user_id} has already rated the movie {movie_id}")

    relevant_users = user_movie_matrix[
        user_movie_matrix[rated_movies].notna().any(axis=1) & user_movie_matrix[movie_id].notna()]

    rated_movies = rated_movies.append(pd.Index([movie_id]))

    relevant_users_rows = relevant_users[['userId'] + list(rated_movies)]
    user_row = user_movie_matrix[user_movie_matrix["userId"] == user_id][['userId'] + list(rated_movies)]
    return pd.concat([user_row, relevant_users_rows]).reset_index(drop=True)


def add_similarity_column_to_cf_matrix(
        cf_matrix: DataFrame, movie_id: int, user_id: int,
        similarity_measure: typing.Callable[[Series, Series, Any], np.float64],
) -> DataFrame:
    target_user_ratings = cf_matrix[cf_matrix['userId'] == user_id].drop('userId', axis=1).drop(movie_id, axis=1).iloc[
        0]
    average_non_nan = cf_matrix[cf_matrix['userId'] != user_id].drop(['userId', movie_id], axis=1).notna().sum(
        axis=1).mean()
    similarities = []
    for other_user_id in cf_matrix['userId']:
        if other_user_id == user_id:
            similarities.append(1.0)
        else:
            other_user_ratings = \
                cf_matrix[cf_matrix['userId'] == other_user_id].drop('userId', axis=1).drop(movie_id, axis=1).iloc[0]
            similarity = similarity_measure(target_user_ratings, other_user_ratings, average_non_nan=average_non_nan)
            similarities.append(similarity)
    df_copy = cf_matrix.copy()
    df_copy['similarity'] = similarities
    return df_copy


def limit_neighborhood(
        cf_matrix_with_similarity: DataFrame,
        movie_id: int,
        user_id: int,
        threshold: Optional[float] = None,
) -> DataFrame:
    neighbors = cf_matrix_with_similarity.copy()

    if threshold is not None:
        neighbors = neighbors[neighbors['similarity'] >= threshold]

    if len(neighbors) < 3:
        raise ValueError("Number of neighbors is less than 1")

    return neighbors


def predict_score(cf_matrix_with_similarity: DataFrame, movie_id: int, user_id: int) -> float:
    neighbors_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'] != user_id
        ][['userId', movie_id, 'similarity']].dropna()

    target_user_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'] == user_id
        ].drop(['userId', 'similarity'], axis=1).iloc[0].dropna()
    target_user_avg_rating = target_user_ratings.mean()

    neighbors_avg_ratings = cf_matrix_with_similarity[
        cf_matrix_with_similarity['userId'].isin(neighbors_ratings['userId'])
    ].drop(['userId', 'similarity'], axis=1).mean(axis=1)

    weighted_sum = sum(
        (neighbors_ratings.iloc[i][movie_id] - neighbors_avg_ratings.iloc[i]) * neighbors_ratings.iloc[i]['similarity']
        for i in range(len(neighbors_ratings))
    )

    sum_of_similarities = neighbors_ratings['similarity'].sum()

    predicted_score = target_user_avg_rating + (weighted_sum / sum_of_similarities)

    return predicted_score


def predict_user_rating(
        transformed_df,
        user_id: int,
        movie_id: int
) -> float:
    try:
        limited_df = prepare_user_movie_matrix_for_cf_matrix(
            transformed_df, movie_id, user_id
        )
    except ValueError:
        return transformed_df[transformed_df['userId'] == user_id][movie_id].iloc[0]
    limited_df_with_similarity = add_similarity_column_to_cf_matrix(
        limited_df, movie_id, user_id, pearson_similarity
    )
    try:
        neighbors_df = limit_neighborhood(
            limited_df_with_similarity,
            movie_id,
            user_id,
            threshold=None,
        )
    except ValueError:
        return limited_df_with_similarity.drop(['userId', 'similarity'], axis=1).mean(axis=1).iloc[0]
    predicted_score = predict_score(
        neighbors_df,
        movie_id,
        user_id,
    )
    return max(0.0, min(5.0, predicted_score))
