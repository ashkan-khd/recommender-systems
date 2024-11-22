import heapq
from typing import *

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from prototypes.diversity.project.aggregators import hybrid_score
from prototypes.diversity.project.group_preference_lists import GroupPreferenceLists
from prototypes.diversity.project.group_recommendation import GroupRecommendationAnalyzer, \
    GroupRecommendationsOverallAnalyzer
from prototypes.diversity.project.preference_score import PreferenceScoreRepository
from prototypes.diversity.project.custom_types import *


class Printer:
    __instance: "Printer" = None
    __initialized: bool = False

    def __init__(self):
        if self.__initialized:
            return
        self.__print_list = []
        self.__initialized = True

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Printer, cls).__new__(cls)
        return cls.__instance

    def __call__(self, *args, **kwargs):
        self.__print_list.append((args, kwargs))

    def print(self):
        for args, kwargs in self.__print_list:
            print(*args, **kwargs)
        self.__print_list.clear()


def transform_csv_dataframe_to_user_movies_matrix(csv_df: DataFrame) -> DataFrame:
    user_movie_matrix = csv_df.pivot(index='userId', columns='movieId', values='rating')
    user_movie_matrix.reset_index(inplace=True)
    user_movie_matrix.columns.name = None
    return user_movie_matrix


# Load movies and base dataset
# movies_details = pd.read_csv('movies.csv')
base = pd.read_csv('../splits/base.csv')

# Load yearly chunks
chunks = [pd.read_csv(f'../splits/chunk{i}.csv') for i in range(1, 11)]

# Define a single group
sample_group = [35, 120, 163, 238, 269]


def update_one_round(
        round_: int,
        group: List[UserIDType],
        user_movie_matrix: DataFrame,
        k: int,
        alpha: float,
):
    group_preference_lists = GroupPreferenceLists(group, user_movie_matrix, k, 4 * k)
    preference_score_calculator = PreferenceScoreRepository(user_movie_matrix).get_calculator()

    movies: List[MovieIDType] = []
    for user_id in group:
        movies.extend(group_preference_lists.get_user_preference_list(user_id))

    movies_score: Dict[MovieIDType, float] = {}
    for movie in tqdm(movies, desc=f"Calculating scores for round {round_}"):
        movies_score[movie] = hybrid_score(group, movie, preference_score_calculator, alpha)

    top_k_movies: List[MovieIDType] = heapq.nlargest(k, movies_score, key=movies_score.get)
    return GroupRecommendationAnalyzer(group, top_k_movies, group_preference_lists, preference_score_calculator)


def generate_sequential_recommendations(
        group: List[UserIDType],
        base_df,
        chunk_dfs: List[DataFrame],
        k: int,
        alpha: float = .0,
):
    group_recommendation_overall_analyzer = GroupRecommendationsOverallAnalyzer(group)
    base_user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(base_df)
    group_recommendation_analyzer = update_one_round(0, group, base_user_movie_matrix, k, alpha)
    group_recommendation_overall_analyzer.add_group_recommendation(group_recommendation_analyzer)
    report_group_recommendation(0, group_recommendation_overall_analyzer, alpha)

    alpha = group_recommendation_analyzer.get_group_disagreement()

    current_df = base_df
    round_ = 0
    for chunk in chunk_dfs:
        current_df = pd.concat([current_df, chunk], ignore_index=True)
        current_user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(current_df)
        group_recommendation_analyzer = update_one_round(
            (round_ := round_ + 1), group, current_user_movie_matrix, k, alpha
        )
        group_recommendation_overall_analyzer.add_group_recommendation(group_recommendation_analyzer)
        report_group_recommendation(round_, group_recommendation_overall_analyzer, alpha)

        alpha = group_recommendation_analyzer.get_group_disagreement()

    Printer().print()


def report_group_recommendation(
        iteration: int,
        group_recommendation_overall_analyzer: GroupRecommendationsOverallAnalyzer,
        alpha: float,
):
    group_recommendation_analyzer = group_recommendation_overall_analyzer.get_last_group_recommendation()
    printer = Printer()
    printer(f"++--- Round {iteration} ---++")
    printer(f"Group satisfaction: {group_recommendation_analyzer.get_group_satisfaction()}")
    printer(f"Group disagreement: {group_recommendation_analyzer.get_group_disagreement()}")
    printer(f"Alpha: {alpha}")
    printer("User satisfaction:")
    for user_id in group_recommendation_analyzer.group:
        printer(
            f"\tUser {user_id} satisfaction: "
            f"{group_recommendation_analyzer.get_user_group_recommendation_satisfaction(user_id)}"
        )

    printer("Group overall satisfaction:", group_recommendation_overall_analyzer.get_group_satisfaction())
    for user_id in group_recommendation_analyzer.group:
        printer(
            f"\tUser {user_id} overall satisfaction: "
            f"{group_recommendation_overall_analyzer.get_user_group_recommendation_satisfaction(user_id)}"
        )
    printer("++-----------------++")


generate_sequential_recommendations(sample_group, base, chunks, 10)
