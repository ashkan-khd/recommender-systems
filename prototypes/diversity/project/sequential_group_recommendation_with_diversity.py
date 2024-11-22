import heapq
from typing import *

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from prototypes.diversity.project.aggregators import hybrid_score, hybrid_score_with_diversity
from prototypes.diversity.project.diversity import DiversityScoreCalculator
from prototypes.diversity.project.diversity import Diversifier
from prototypes.diversity.project.group_preference_lists import GroupPreferenceLists
from prototypes.diversity.project.group_recommendation import GroupRecommendationAnalyzer, \
    GroupRecommendationsOverallAnalyzer
from prototypes.diversity.project.plotter import Plotter
from prototypes.diversity.project.preference_score import PreferenceScoreRepository
from prototypes.diversity.project.constants import *


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
base = pd.read_csv(BASE_RATINGS_CSV_FILE_PATH)

# Load yearly chunks
chunks = [pd.read_csv(BASE_RATINGS_CHUNK_FILE_PATH_TEMPLATE(round_=i)) for i in range(1, 11)]

# Define a single group
# sample_group = [35, 120, 163, 238, 269]
sample_group = [269, 120, 278, 163, 161, 383, 238, 392, 188, 35]


def update_one_round(
        round_: int,
        group: List[UserIDType],
        user_movie_matrix: DataFrame,
        group_recommendation_overall_analyzer: GroupRecommendationsOverallAnalyzer,
        k: int,
        alpha: float,
        beta: float,
        landa: float,
) -> None:
    group_preference_lists = GroupPreferenceLists(group, user_movie_matrix, k, 3*k)
    preference_score_calculator = PreferenceScoreRepository(user_movie_matrix).get_calculator()
    diversity_score_calculator = DiversityScoreCalculator()
    movies: List[MovieIDType] = []
    for user_id in group:
        movies.extend(group_preference_lists.get_user_preference_list(user_id))

    movies_score: Dict[MovieIDType, float] = {}
    if len(group_recommendation_overall_analyzer) == 0:
        aggregator = lambda movie_id: hybrid_score(group, movie_id, preference_score_calculator, alpha)
    else:
        aggregator = lambda movie_id: hybrid_score_with_diversity(
            group, movie_id,
            preference_score_calculator,
            group_recommendation_overall_analyzer.get_last_group_recommendation().group_recommendation,
            diversity_score_calculator,
            alpha, beta
        )
    for movie in tqdm(movies, desc=f"Calculating scores for round {round_}"):
        movies_score[movie] = aggregator(movie)

    top_2k_movies: List[MovieIDType] = heapq.nlargest(2 * k, movies_score, key=movies_score.get)

    top_k_movies = Diversifier(
        k, landa,
        sim=lambda movie_list: sum([movies_score[m] for m in movie_list]),
        sim_individual=lambda movie: movies_score[movie],
        div=lambda movie_list: diversity_score_calculator.calculate_list_diversity(movie_list)
    ).diversify(top_2k_movies)

    group_recommendation_overall_analyzer.add_group_recommendation(
        GroupRecommendationAnalyzer(group, top_k_movies, group_preference_lists, preference_score_calculator)
    )


def generate_sequential_recommendations(
        group: List[UserIDType],
        base_df,
        chunk_dfs: List[DataFrame],
        k: int,
        alpha: float, beta: float, landa: float,
):
    plotter = Plotter(list(range(2008, 2019)))
    group_recommendation_overall_analyzer = GroupRecommendationsOverallAnalyzer(group)
    base_user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(base_df)
    update_one_round(
        0, group, base_user_movie_matrix, group_recommendation_overall_analyzer, k, alpha, beta, landa
    )
    report_group_recommendation(0, group_recommendation_overall_analyzer, alpha)
    add_group_recommendation_to_plotter(group_recommendation_overall_analyzer, plotter)

    alpha = group_recommendation_overall_analyzer.get_last_group_recommendation().get_group_disagreement()

    current_df = base_df
    round_ = 0
    for chunk in chunk_dfs:
        current_df = pd.concat([current_df, chunk], ignore_index=True)
        current_user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(current_df)
        update_one_round(
            (round_ := round_ + 1), group, current_user_movie_matrix,
            group_recommendation_overall_analyzer, k, alpha, beta, landa,
        )
        report_group_recommendation(round_, group_recommendation_overall_analyzer, alpha)
        add_group_recommendation_to_plotter(group_recommendation_overall_analyzer, plotter)

        alpha = group_recommendation_overall_analyzer.get_last_group_recommendation().get_group_disagreement()

    Printer().print()
    plotter.plot()


def add_group_recommendation_to_plotter(
        group_recommendation_overall_analyzer: GroupRecommendationsOverallAnalyzer,
        plotter: Plotter,
):
    group_recommendation_analyzer = group_recommendation_overall_analyzer.get_last_group_recommendation()
    plotter.add_group_satisfaction(group_recommendation_analyzer.get_group_satisfaction())
    plotter.add_group_disagreement(group_recommendation_analyzer.get_group_disagreement())
    plotter.add_group_diversity(group_recommendation_analyzer.get_group_diversity())
    for user_id in group_recommendation_analyzer.group:
        plotter.add_user_satisfaction(
            user_id,
            group_recommendation_analyzer.get_user_group_recommendation_satisfaction(user_id)
        )

    plotter.add_group_overall_satisfaction(group_recommendation_overall_analyzer.get_group_satisfaction())
    for user_id in group_recommendation_overall_analyzer.group:
        plotter.add_user_overall_satisfaction(
            user_id,
            group_recommendation_overall_analyzer.get_user_group_recommendation_satisfaction(user_id)
        )

    if len(group_recommendation_overall_analyzer) > 1:
        plotter.add_diversity_from_last_recommendation(
            group_recommendation_overall_analyzer.get_diversity_between_last_two_group_recommendations()
        )


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
    printer("** Overall Section **")
    printer("Group overall satisfaction:", group_recommendation_overall_analyzer.get_group_satisfaction())
    for user_id in group_recommendation_overall_analyzer.group:
        printer(
            f"\tUser {user_id} overall satisfaction: "
            f"{group_recommendation_overall_analyzer.get_user_group_recommendation_satisfaction(user_id)}"
        )
    printer("++-----------------++")


generate_sequential_recommendations(sample_group, base, chunks, 10, alpha=.0, beta=0.3, landa=0.5)
