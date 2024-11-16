import random
from collections import defaultdict
from functools import lru_cache
from time import sleep

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

from part2_utils.predict_user_rating import predict_user_rating

RATING_DATASET = "ratings.csv"

raw_df = pd.read_csv(RATING_DATASET)


def get_movie_genres(csv_file, movie_ids):
    """
    Retrieves a dictionary mapping movie IDs to their respective sets of genres.

    Parameters:
        csv_file (str): Path to the CSV file containing movie data.
        movie_ids (list): List of movie IDs to retrieve.

    Returns:
        dict: A dictionary where keys are movie IDs from the input list and
              values are sets of genres for those movies.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Filter the DataFrame to include only rows with the specified movie IDs
    filtered_df = df[df['movieId'].isin(movie_ids)]

    # Create a dictionary mapping movie IDs to their genres (as sets)
    movie_genres = {
        row['movieId']: set(row['genres'].split('|'))
        for _, row in filtered_df.iterrows()
    }

    return movie_genres


def calculate_genre_diversity(group, movies, movie_genres, user_ratings):
    """
    Calculate genre diversity score for a set of movies.
    Higher score means more diverse genre representation.

    Args:
        group: List of user IDs
        movies: List of movie IDs to evaluate
        movie_genres: Dict mapping movie IDs to their genres
        user_ratings: Dict of user ratings

    Returns:
        Float between 0 and 1 indicating genre diversity
    """
    if not movies:
        return 0

    # Collect all genres in the recommended movies
    movie_genre_set = set()
    for movie in movies:
        if movie in movie_genres:
            movie_genre_set.update(movie_genres[movie])

    # Count genre occurrences
    genre_counts = defaultdict(int)
    for movie in movies:
        if movie in movie_genres:
            for genre in movie_genres[movie]:
                genre_counts[genre] += 1

    if not genre_counts:
        return 0

    # Calculate entropy-based diversity score
    total_genres = sum(genre_counts.values())
    proportions = [count / total_genres for count in genre_counts.values()]
    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
    max_entropy = np.log(len(movie_genre_set)) if movie_genre_set else 1

    # Normalize to [0,1]
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0
    return diversity_score


def transform_csv_dataframe_to_user_movies_matrix(csv_df: DataFrame) -> DataFrame:
    user_movie_matrix = csv_df.pivot(index='userId', columns='movieId', values='rating')
    user_movie_matrix.reset_index(inplace=True)
    user_movie_matrix.columns.name = None
    return user_movie_matrix


user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(raw_df)


@lru_cache(maxsize=None)
def preference_score(user_id, movie_id):
    return predict_user_rating(user_movie_matrix, user_id, movie_id)


def calculate_user_satisfaction(user_id, group_rec, user_ratings, user_satisfactions):
    """
    Calculate user satisfaction based on the group recommendations.
    """
    user_ideal_satisfaction = user_satisfactions[user_id]
    group_satisfaction = sum([preference_score(user_id, movie) for movie in group_rec])
    return group_satisfaction / user_ideal_satisfaction if user_ideal_satisfaction > 0 else 0


def calculate_group_satisfaction(group, group_rec, user_ratings, user_satisfactions):
    """
    Calculate average satisfaction across all users in the group.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings, user_satisfactions) for user
                                in group]
    return np.mean(individual_satisfactions)


def calculate_group_disagreement(group, group_rec, user_ratings, user_satisfactions):
    """
    Calculate disagreement within the group as the difference between max and min satisfaction.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings, user_satisfactions) for user
                                in group]
    return max(individual_satisfactions) - min(individual_satisfactions)


def avgScore(group, movie, user_ratings):
    """Calculate the average preference score for the movie among the group members."""
    return np.mean([preference_score(user, movie) for user in group])


def leastScore(group, movie, user_ratings):
    """Calculate the minimum preference score for the movie among the group members."""
    return min([preference_score(user, movie) for user in group])


def generate_sequential_recommendations_enhanced(
        group,
        user_ratings,
        movies,
        movie_genres,
        iterations=5,
        top_k=10,
        alpha=0,  # weight for least score
        beta=1,  # weight for average score
        gamma=0,  # weight for diversity score
):
    """
    Enhanced sequential recommendation generator with three components:
    1. Average Score (group preference)
    2. Least Score (fairness)
    3. Genre Diversity (variety)

    Args:
        group: List of user IDs
        user_ratings: Dict of user ratings
        movies: List of available movies
        movie_genres: Dict mapping movie IDs to their genres
        iterations: Number of recommendation rounds
        top_k: Number of recommendations per round
        alpha: Weight for least score
        beta: Weight for average score
        gamma: Weight for diversity score
    """
    assert abs((alpha + beta + gamma) - 1.0) < 1e-6, "Weights must sum to 1"

    group_recommendations = []
    movie_scores = defaultdict(float)
    user_satisfactions = defaultdict(float)

    # Calculate ideal satisfaction for each user
    for user in group:
        top_user_prefs = sorted(user_ratings[user].items(), key=lambda x: x[1], reverse=True)[:top_k]
        user_satisfactions[user] = sum([score for _, score in top_user_prefs])

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} progress")
        print(f"Current weights - α: {alpha:.2f}, β: {beta:.2f}, γ: {gamma:.2f}\n")

        # Calculate scores for each movie
        for movie in movies:
            avg_score = avgScore(group, movie, user_ratings)
            least_score = leastScore(group, movie, user_ratings)

            # Calculate temporary recommendations with this movie
            temp_recommendations = group_recommendations[-1][:top_k - 1] + [movie] if group_recommendations else [movie]
            diversity_score = calculate_genre_diversity(group, temp_recommendations, movie_genres, user_ratings)

            # Combine scores with weights
            movie_scores[movie] = (alpha * least_score +
                                   beta * avg_score +
                                   gamma * diversity_score)

        # Select top movies based on combined score
        top_movies = sorted(movie_scores, key=movie_scores.get, reverse=True)[:top_k]
        group_recommendations.append(top_movies)

        # Calculate metrics
        group_sat = calculate_group_satisfaction(group, top_movies, user_ratings, user_satisfactions)
        group_dis = calculate_group_disagreement(group, top_movies, user_ratings, user_satisfactions)
        group_div = calculate_genre_diversity(group, top_movies, movie_genres, user_ratings)

        # Update weights based on metrics with distinct strategies
        # If satisfaction is low, increase beta (satisfaction weight)
        if group_sat < 0.6:  # threshold for "low" satisfaction
            beta = min(0.6, beta + 0.1)  # increase but cap at 0.6

        # If disagreement is high, increase alpha (fairness weight)
        if group_dis > 0.2:  # threshold for "high" disagreement
            alpha = min(0.5, alpha + 0.1)  # increase but cap at 0.5

        # If diversity is low, increase gamma (diversity weight)
        if group_div < 0.9:  # threshold for "low" diversity
            gamma = min(0.4, gamma + 0.1)  # increase but cap at 0.4

        # Normalize weights to sum to 1
        total = alpha + beta + gamma
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total

        print(f"\nIteration {iteration + 1}")
        print(f"Top Movies: {top_movies}")
        print(f"Group Satisfaction: {group_sat:.3f}")
        print(f"Group Disagreement: {group_dis:.3f}")
        print(f"Genre Diversity: {group_div:.3f}")

        sleep(0.001)

    return group_recommendations


sample_group = [597, 217, 66, 177, 274, 391, 483, 561, 414, 509, 160]

user_ratings = raw_df[raw_df["userId"].isin(sample_group)].groupby('userId')[['movieId', 'rating']].apply(
    lambda x: dict(zip(x['movieId'], x['rating']))).to_dict()
group_users_movies = user_movie_matrix[user_movie_matrix['userId'].isin(sample_group)].dropna(axis=1,
                                                                                              how='all').columns.tolist()[
                     1:]
movies = random.sample([m for m in raw_df['movieId'].unique() if m not in group_users_movies], 1000)

movie_genres = get_movie_genres("movies.csv", movies)
# Run the sequential group recommendation process
recommendations = generate_sequential_recommendations_enhanced(sample_group, user_ratings, movies, movie_genres)