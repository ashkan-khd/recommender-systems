import heapq
from functools import cached_property
from typing import List, Set, Dict

import pandas as pd
from pandas import DataFrame

from .contants import *


class MovieDesciber:
    def __init__(self):
        self.movie_details = pd.read_csv(MOVIE_DETAILS_DATASET)
        self.user_tags_for_movies = pd.read_csv(USER_TAGS_FOR_MOVIES_DATASET)

    def __call__(self, movie_id: MovieIDType) -> Set[str]:
        genres = self.movie_details[self.movie_details['movieId'] == movie_id]['genres'].values[0]
        description_list = set(genre.lower() for genre in genres.split('|'))

        tags = self.user_tags_for_movies[self.user_tags_for_movies['movieId'] == movie_id]['tag'].tolist()
        tags_set = set(tag.lower() for tag in tags)
        description_list = description_list.union(tags_set)

        return description_list


def score_movie(movie, group_top_movies, movie_to_tags):
    movie_tags = movie_to_tags[movie]
    score = 0
    for group_movie in group_top_movies:
        group_movie_tags = movie_to_tags[group_movie]
        score += len(movie_tags.intersection(group_movie_tags))
    return score


def get_group_recommendation(
        group: List[UserIDType],
        movie_to_tags: Dict[MovieIDType, Set[str]],
        user_movie_matrix: DataFrame,
        movie_search_area: List[MovieIDType],
        k: int,
) -> List[MovieIDType]:
    group_top_movies = []
    for user in group:
        user_ratings_df = user_movie_matrix.drop('userId', axis=1)[
            user_movie_matrix['userId'] == user
            ].dropna(axis=1, how='all')
        two_k_preferences = user_ratings_df.iloc[0].nlargest(2 * k).items()
        group_top_movies.extend([movie for movie, rating in two_k_preferences])

    movies_score: Dict[MovieIDType, int] = {}
    for movie in movie_search_area:
        movies_score[movie] = score_movie(movie, group_top_movies, movie_to_tags)

    top_k_movies: List[MovieIDType] = heapq.nlargest(k, movies_score, key=movies_score.get)
    return top_k_movies


def get_group_recommendation_v2(
        group: List[UserIDType],
        movie_to_tags: Dict[MovieIDType, Set[str]],
        user_movie_dict: Dict[UserIDType, Dict[MovieIDType, float]],
        movie_search_area: List[MovieIDType],
        k: int,
) -> List[MovieIDType]:
    group_top_movies = []
    for user in group:
        top_2k_preferences = heapq.nlargest(2 * k, user_movie_dict[user].items(), key=lambda x: x[1])
        group_top_movies.extend([movie for movie, rating in top_2k_preferences])

    # group_top_movies = [movie for user in group for movie, rating in user_movie_dict[user].items()]

    movies_score: Dict[MovieIDType, int] = {}
    for movie in movie_search_area:
        movies_score[movie] = score_movie(movie, group_top_movies, movie_to_tags)

    top_k_movies: List[MovieIDType] = heapq.nlargest(k, movies_score, key=movies_score.get)
    return top_k_movies
